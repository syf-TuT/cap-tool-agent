"""LLM client utilities for querying language models.

Extracted from capx/utils/launch_utils.py to separate LLM query logic
from launch/config utilities.
"""

from __future__ import annotations

import concurrent.futures
import copy
import json
import os
import random
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import requests

from capx.llm.context import get_trial_llm_context
from capx.llm.errors import LLMErrorKind, LLMQueryError
from capx.llm.resilience import LLMRetryPolicy

if TYPE_CHECKING:
    from capx.envs.launch import LaunchArgs

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------

GPT_MODELS = [
    "openai/gpt-5.4",
    "openai/o4-mini",
]
VLM_MODELS = [
    "google/gemini-3.1-pro-preview",
    "google/gemini-2.5-flash-lite",
    "anthropic/claude-opus-4-5",
    "anthropic/claude-haiku-4-5",
    "openai/gpt-5.4",
    "openai/o1",
    "openai/o4-mini",
    "deepseek/deepseek-v3.2",
    "deepseek/deepseek-r1-0528",
    "deepseek/deepseek-r1",
    "qwen/qwen3.5-122b-a10b",
    "moonshotai/kimi-k2",
]
CLAUDE_MODELS = ["anthropic/claude-opus-4-5", "anthropic/claude-haiku-4-5"]
OSS_MODELS = [
    "deepseek/deepseek-v3.2",
    "deepseek/deepseek-r1-0528",
    "deepseek/deepseek-r1",
    "qwen/qwen3.5-122b-a10b",
    "moonshotai/kimi-k2",
]
OPENROUTER_MODELS = [
    "openrouter/google/gemini-2.5-pro-preview",
    "openrouter/google/gemini-2.5-flash-preview",
    "openrouter/anthropic/claude-sonnet-4",
    "openrouter/anthropic/claude-opus-4",
    "openrouter/deepseek/deepseek-r1",
    "openrouter/deepseek/deepseek-chat-v3-0324",
    "openrouter/openai/gpt-4.1",
    "openrouter/openai/o4-mini",
    "openrouter/meta-llama/llama-4-maverick",
    "openrouter/qwen/qwen3-235b-a22b",
]
OPENROUTER_SERVER_URL = "http://localhost:8110/chat/completions"

# ---------------------------------------------------------------------------
# Ensemble configuration
# ---------------------------------------------------------------------------

ENSEMBLE_CONFIGS = [
    # Gemini-3-Pro only — best single model per CaP-Bench (Figure 1).
    # 3 temps for diversity; synthesis still uses Gemini-3-Pro.
    # ~45% faster than full multimodel (no Claude/GPT latency bottleneck).
    ("openai/gpt-5.4", [0.1, 0.5, 0.9]),
]

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def is_openrouter_model(model: str) -> bool:
    """Return True if the model should be routed through the OpenRouter proxy."""
    return model.startswith("openrouter/") or model in OPENROUTER_MODELS


@dataclass
class ModelQueryArgs:
    """Arguments for querying a model."""

    model: str
    server_url: str
    api_key: str | None = None
    temperature: float = 0.2
    max_tokens: int = 4096
    reasoning_effort: str = "medium"
    debug: bool = False


def collapse_text_image_inputs(messages: list[dict]) -> list[dict]:
    """
    Collapse a list of messages with sequential text into a single text input, images are still in the same relative position
    """
    new_prompt = []
    current_text_input = ""
    for message in messages:
        if message["type"] == "text":
            current_text_input += message["text"] + "\n"
        else:
            if current_text_input != "":
                new_prompt.append({"type": "text", "text": current_text_input})
                current_text_input = ""
            new_prompt.append(message)
    if current_text_input != "":
        new_prompt.append({"type": "text", "text": current_text_input})
    return new_prompt


def _reasoning_disabled() -> bool:
    return os.getenv("CAPX_DISABLE_REASONING") == "1"


_RETRYABLE_HTTP_STATUSES = {408, 429, 500, 502, 503, 504}


def _http_error_kind(status_code: int) -> LLMErrorKind:
    if status_code == 429:
        return LLMErrorKind.RATE_LIMITED
    if status_code >= 500:
        return LLMErrorKind.HTTP_5XX
    if status_code in (401, 403):
        return LLMErrorKind.AUTH_ERROR
    return LLMErrorKind.REQUEST_REJECTED


def _request_error_kind(error: requests.exceptions.RequestException) -> LLMErrorKind:
    if isinstance(error, requests.exceptions.ConnectTimeout):
        return LLMErrorKind.CONNECT_TIMEOUT
    if isinstance(error, (requests.exceptions.ReadTimeout, requests.exceptions.Timeout)):
        return LLMErrorKind.READ_TIMEOUT
    return LLMErrorKind.CONNECTION_ERROR


def _retry_delay(response: requests.Response | None, policy: LLMRetryPolicy) -> float:
    delay = policy.retry_backoff_seconds + random.uniform(0, policy.retry_jitter_seconds)
    if response is None:
        return delay
    retry_after = response.headers.get("Retry-After")
    if retry_after is not None:
        try:
            numeric_retry_after = max(0.0, float(retry_after))
        except ValueError:
            pass
        else:
            delay = max(delay, min(numeric_retry_after, policy.retry_after_cap_seconds))
    return delay


def _completions_to_responses_convert_prompt(prompt: list[dict]) -> list[dict]:
    """Convert completions api format to responses api format.

    Args:
        prompt: The prompt in completions api format

    Returns:
        The prompt in responses api format

    Switch prompt structure to api responses api format e.g.:
    From
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the image in detail."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ]
    To
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Describe the image in detail."},
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{base64_image}"
                }
            ]
        }
    ]
    """

    for message in prompt:
        for content in message["content"]:
            if type(content) == str:
                continue
            if content.get("type") == "text":
                content["type"] = "input_text"
                content["text"] = content.pop("text")

            elif content.get("type") == "image_url":
                content["type"] = "input_image"
                content["image_url"] = content["image_url"]["url"]
    return prompt


# ---------------------------------------------------------------------------
# Core query functions
# ---------------------------------------------------------------------------


def query_model(args: "LaunchArgs | ModelQueryArgs", prompt: list[dict]) -> str:
    """Query vLLM server for code generation.

    Args:
        args: Configuration with server URL and model settings
        prompt: Full prompt containing environment observation and possibly multi-turn decision prompt
    Returns:
        Model response content
    """

    # Route OpenRouter models to the OpenRouter proxy server
    if is_openrouter_model(args.model):
        server_url = OPENROUTER_SERVER_URL
    else:
        server_url = args.server_url

    if args.model in GPT_MODELS:
        if "codex" in args.model:
            prompt = _completions_to_responses_convert_prompt(prompt)
            payload = {
                "model": args.model,
                "input": prompt,
            }
        else:
            payload = {
                "model": args.model,
                "reasoning_effort": args.reasoning_effort,
                "max_completion_tokens": args.max_tokens,  # Total completion tokens = reasoning + output tokens
                "messages": prompt,
            }
    elif is_openrouter_model(args.model):
        payload = {
            "model": args.model,
            "messages": prompt,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        }
    elif args.model in CLAUDE_MODELS:
        payload = {
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "thinking": {"type": "enabled", "budget_tokens": 4096},
            "messages": prompt,
        }
    elif args.model in OSS_MODELS:
        payload = {
            "model": args.model,
            "messages": prompt,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        }
    else:
        payload = {
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "messages": prompt,
        }
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    elif os.getenv("OPENAI_API_KEY") is not None and args.model in GPT_MODELS:
        headers["Authorization"] = f"Bearer {os.getenv('OPENAI_API_KEY')}"

    reasoning_disabled = _reasoning_disabled()
    if reasoning_disabled:
        payload.pop("reasoning_effort", None)
        payload["thinking"] = {"type": "disabled"}

    if os.getenv("CAPX_FORCE_STREAMING_CHAT_COMPLETIONS") == "1":
        max_attempts = max(1, int(os.getenv("CAPX_STREAMING_CHAT_COMPLETIONS_RETRIES", "1")))
        last_exc: BaseException | None = None
        fallback_to_non_streaming = False
        for attempt in range(1, max_attempts + 1):
            print(
                "Using streaming chat completions for model query "
                f"(attempt {attempt}/{max_attempts})",
                flush=True,
            )
            content = ""
            reasoning = None
            try:
                for chunk in query_model_streaming(args, prompt):
                    if chunk["type"] == "content_delta":
                        content += chunk["content"]
                    elif chunk["type"] == "done":
                        content = chunk["content"]
                        reasoning = chunk.get("reasoning")
                if os.getenv("CAPX_STREAMING_REQUIRE_CONTENT") == "1" and not content.strip():
                    raise TimeoutError("Streaming model query returned empty content")
                return {"content": content, "reasoning": reasoning}
            except (TimeoutError, requests.exceptions.RequestException) as exc:
                last_exc = exc
                empty_streaming_content = (
                    isinstance(exc, TimeoutError)
                    and str(exc) == "Streaming model query returned empty content"
                )
                if attempt >= max_attempts:
                    if empty_streaming_content:
                        print(
                            "Streaming returned empty content after retries; "
                            "falling back to non-streaming chat completion",
                            flush=True,
                        )
                        fallback_to_non_streaming = True
                        break
                    raise
                print(f"Streaming model query failed: {exc}. Retrying...", flush=True)
        if not fallback_to_non_streaming:
            raise RuntimeError("Streaming model query failed") from last_exc

    policy = LLMRetryPolicy.from_env()
    context = get_trial_llm_context()
    call_index = context.next_call_index() if context is not None else 1
    call_started = time.monotonic()

    def raise_query_error(
        kind: LLMErrorKind,
        attempt: int,
        status_code: int | None,
        message: str,
    ) -> None:
        raise LLMQueryError(
            kind=kind,
            call_index=call_index,
            attempt=attempt,
            status_code=status_code,
            elapsed_seconds=max(0.0, time.monotonic() - call_started),
            message=message,
        )

    for attempt in range(1, policy.max_attempts + 1):
        remaining_before = context.remaining_seconds() if context is not None else None
        if remaining_before is not None and remaining_before <= 0:
            raise_query_error(
                LLMErrorKind.TRIAL_BUDGET_EXHAUSTED,
                attempt,
                None,
                "Trial budget exhausted before LLM request",
            )
        request_timeout = policy.request_timeout_seconds
        if remaining_before is not None:
            request_timeout = min(request_timeout, remaining_before)
        if request_timeout <= 0:
            raise_query_error(
                LLMErrorKind.TRIAL_BUDGET_EXHAUSTED,
                attempt,
                None,
                "Trial budget cannot cover a positive LLM request timeout",
            )

        started = time.monotonic()
        remaining_before_ms = (
            None if remaining_before is None else int(round(remaining_before * 1000))
        )
        response: requests.Response | None = None
        status_code: int | None = None
        ttfb_ms: int | None = None
        try:
            response = requests.post(
                server_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=request_timeout,
                stream=True,
            )
            status_code = response.status_code
            body_chunks = []
            for chunk in response.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                if ttfb_ms is None:
                    ttfb_ms = int(round((time.monotonic() - started) * 1000))
                body_chunks.append(chunk)
            body_bytes = b"".join(body_chunks)
        except requests.exceptions.RequestException as error:
            finished = time.monotonic()
            kind = _request_error_kind(error)
            can_retry = attempt < policy.max_attempts
            delay = _retry_delay(response, policy) if can_retry else 0.0
            remaining = context.remaining_seconds() if context is not None else None
            budget_allows_retry = remaining is None or (
                remaining >= policy.minimum_retry_budget_seconds and remaining > delay
            )
            retry_scheduled = can_retry and budget_allows_retry
            if context is not None:
                context.record_attempt(
                    call_index=call_index,
                    attempt=attempt,
                    mode="non_streaming",
                    http_status=status_code,
                    ttfb_ms=ttfb_ms,
                    first_content_ms=None,
                    started_monotonic=started,
                    finished_monotonic=finished,
                    remaining_before_ms=remaining_before_ms,
                    outcome="retryable_transport_error" if can_retry else "transport_error",
                    error_kind=kind.value,
                    retry_scheduled=retry_scheduled,
                )
            if retry_scheduled:
                time.sleep(delay)
                continue
            if can_retry and not budget_allows_retry:
                raise_query_error(
                    LLMErrorKind.TRIAL_BUDGET_EXHAUSTED,
                    attempt,
                    status_code,
                    "Trial budget cannot cover the minimum retry budget and delay",
                )
            raise_query_error(kind, attempt, status_code, f"LLM transport failed: {kind.value}")

        finished = time.monotonic()
        if status_code is not None and status_code >= 400:
            kind = _http_error_kind(status_code)
            can_retry = status_code in _RETRYABLE_HTTP_STATUSES and attempt < policy.max_attempts
            delay = _retry_delay(response, policy) if can_retry else 0.0
            remaining = context.remaining_seconds() if context is not None else None
            budget_allows_retry = remaining is None or (
                remaining >= policy.minimum_retry_budget_seconds and remaining > delay
            )
            retry_scheduled = can_retry and budget_allows_retry
            if context is not None:
                context.record_attempt(
                    call_index=call_index,
                    attempt=attempt,
                    mode="non_streaming",
                    http_status=status_code,
                    ttfb_ms=ttfb_ms,
                    first_content_ms=None,
                    started_monotonic=started,
                    finished_monotonic=finished,
                    remaining_before_ms=remaining_before_ms,
                    outcome="retryable_http_error" if can_retry else "http_error",
                    error_kind=kind.value,
                    retry_scheduled=retry_scheduled,
                )
            if retry_scheduled:
                time.sleep(delay)
                continue
            if can_retry and not budget_allows_retry:
                raise_query_error(
                    LLMErrorKind.TRIAL_BUDGET_EXHAUSTED,
                    attempt,
                    status_code,
                    "Trial budget cannot cover the minimum retry budget and delay",
                )
            raise_query_error(
                kind,
                attempt,
                status_code,
                f"LLM request rejected with HTTP status {status_code}",
            )

        try:
            body = json.loads(body_bytes)
            if args.model in GPT_MODELS and "codex" in args.model:
                content = body["output_text"]
                reasoning = None
            else:
                message = body["choices"][0]["message"]
                content = message["content"]
                reasoning = None if reasoning_disabled else message.get("reasoning")
            if not isinstance(content, str):
                raise TypeError("response content must be a string")
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError, IndexError, TypeError):
            finished = time.monotonic()
            if context is not None:
                context.record_attempt(
                    call_index=call_index,
                    attempt=attempt,
                    mode="non_streaming",
                    http_status=status_code,
                    ttfb_ms=ttfb_ms,
                    first_content_ms=None,
                    started_monotonic=started,
                    finished_monotonic=finished,
                    remaining_before_ms=remaining_before_ms,
                    outcome="invalid_response",
                    error_kind=LLMErrorKind.INVALID_RESPONSE.value,
                    retry_scheduled=False,
                )
            raise_query_error(
                LLMErrorKind.INVALID_RESPONSE,
                attempt,
                status_code,
                "LLM response was not valid JSON with the expected schema",
            )

        finished = time.monotonic()
        if context is not None:
            context.record_attempt(
                call_index=call_index,
                attempt=attempt,
                mode="non_streaming",
                http_status=status_code,
                ttfb_ms=ttfb_ms,
                first_content_ms=None,
                started_monotonic=started,
                finished_monotonic=finished,
                remaining_before_ms=remaining_before_ms,
                outcome="success",
                error_kind=None,
                retry_scheduled=False,
            )
        print(f"Time taken to query model: {finished - call_started:.2f} seconds")
        return {"content": content, "reasoning": reasoning}

    raise RuntimeError("bounded LLM attempt loop exited unexpectedly")


def query_model_streaming(
    args: "LaunchArgs | ModelQueryArgs",
    prompt: list[dict],
) -> Iterable[dict]:
    """Query model with streaming enabled, yielding partial responses.

    Yields dictionaries with:
      - {"type": "content_delta", "content": "partial text"}
      - {"type": "reasoning_delta", "content": "partial reasoning"} (if supported)
      - {"type": "done", "content": "full content", "reasoning": "full reasoning or None"}

    Args:
        args: Configuration with server URL and model settings
        prompt: Full prompt containing environment observation

    Yields:
        Partial response chunks as they arrive
    """
    if args.model in GPT_MODELS:
        payload = {
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "max_completion_tokens": args.max_tokens,
            "messages": prompt,
            "stream": True,
        }
    elif args.model in CLAUDE_MODELS:
        payload = {
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "thinking": {"type": "enabled", "budget_tokens": 4096},
            "messages": prompt,
            "stream": True,
        }
    else:
        payload = {
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "messages": prompt,
            "stream": True,
        }

    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    elif os.getenv("OPENAI_API_KEY") is not None and args.model in GPT_MODELS:
        headers["Authorization"] = f"Bearer {os.getenv('OPENAI_API_KEY')}"

    reasoning_disabled = _reasoning_disabled()
    if reasoning_disabled:
        payload.pop("reasoning_effort", None)
        payload["thinking"] = {"type": "disabled"}

    full_content = ""
    full_reasoning = ""

    start_time = time.time()
    first_content_timeout = float(
        os.getenv("CAPX_STREAMING_FIRST_CONTENT_TIMEOUT_SECONDS", "0") or 0
    )
    request_timeout = float(os.getenv("CAPX_STREAMING_REQUEST_TIMEOUT_SECONDS", "200") or 200)

    stream_chunks = 0

    with requests.post(
        args.server_url,
        headers=headers,
        data=json.dumps(payload),
        timeout=request_timeout,
        stream=True,
    ) as response:
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        is_sse = "text/event-stream" in content_type
        is_json = "application/json" in content_type

        # If it's a regular JSON response (server doesn't support streaming),
        # fall back to non-streaming behavior
        if is_json and not is_sse:
            print("Warning: Server returned JSON instead of SSE stream, falling back to non-streaming")
            body = response.json()
            try:
                full_content = body["choices"][0]["message"]["content"]
                message = body.get("choices", [{}])[0].get("message", {})
                if not reasoning_disabled:
                    full_reasoning = message.get("reasoning") or message.get("reasoning_content")
                if full_reasoning:
                    print(f"Reasoning extracted ({len(full_reasoning)} chars)")
                elif reasoning_disabled:
                    print("Reasoning disabled; ignoring any reasoning returned by model")
                else:
                    print("No reasoning returned by model")
            except (KeyError, IndexError) as exc:
                raise RuntimeError(f"Unexpected response format: {body}") from exc

            yield {"type": "content_delta", "content": full_content}
            yield {
                "type": "done",
                "content": full_content,
                "reasoning": full_reasoning if full_reasoning else None,
            }
            end_time = time.time()
            print(f"Time taken to query model (streaming fallback): {end_time - start_time:.2f} seconds")
            return

        for line in response.iter_lines():
            if not line:
                continue

            line_str = line.decode("utf-8")

            # SSE format: "data: {...}" or "data: [DONE]"
            if line_str.startswith("data: "):
                stream_chunks += 1
                data_str = line_str[6:]  # Remove "data: " prefix

                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                    choices = data.get("choices", [])
                    if not choices:
                        continue

                    delta = choices[0].get("delta", {})

                    # Handle content delta
                    content_delta = delta.get("content", "")
                    if content_delta:
                        full_content += content_delta
                        yield {"type": "content_delta", "content": content_delta}

                    # Handle reasoning delta (some APIs support this)
                    reasoning_delta = delta.get("reasoning") or delta.get("reasoning_content") or ""
                    if reasoning_delta and not reasoning_disabled:
                        full_reasoning += reasoning_delta
                        yield {"type": "reasoning_delta", "content": reasoning_delta}

                    if stream_chunks % 200 == 0:
                        print(
                            "Streaming progress: "
                            f"chunks={stream_chunks}, "
                            f"content_chars={len(full_content)}, "
                            f"reasoning_chars={len(full_reasoning)}",
                            flush=True,
                        )
                    if (
                        first_content_timeout > 0
                        and not full_content
                        and time.time() - start_time > first_content_timeout
                    ):
                        raise TimeoutError(
                            f"No content delta within {first_content_timeout:.1f}s"
                        )

                except json.JSONDecodeError:
                    continue
            else:
                # Try parsing as raw JSON (non-SSE format)
                try:
                    data = json.loads(line_str)
                    choices = data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content_delta = delta.get("content", "")
                        if content_delta:
                            full_content += content_delta
                            yield {"type": "content_delta", "content": content_delta}
                except json.JSONDecodeError:
                    continue

    end_time = time.time()
    print(f"Time taken to query model (streaming): {end_time - start_time:.2f} seconds")
    if full_reasoning:
        print(f"Reasoning extracted ({len(full_reasoning)} chars)")
    elif reasoning_disabled:
        print("Reasoning disabled; ignoring any reasoning returned by model")
    else:
        print("No reasoning returned by model")

    yield {
        "type": "done",
        "content": full_content,
        "reasoning": full_reasoning if full_reasoning else None,
    }


def query_model_ensemble(
    args: "LaunchArgs | ModelQueryArgs",
    prompt: list[dict],
    synthesis_model: str = "openai/gpt-5.4",
    is_multiturn = False
) -> dict[str, Any]:
    """Query 9 models (3 models x 3 temperatures) and synthesize final output."""

    def query_single(model: str, temp: float) -> dict:
        query_args = ModelQueryArgs(
            model=model,
            server_url=args.server_url,
            api_key=args.api_key,
            temperature=temp,
            max_tokens=args.max_tokens,
            reasoning_effort=getattr(args, "reasoning_effort", "medium"),
        )
        try:
            result = query_model(query_args, copy.deepcopy(prompt))
            return {"model": model, "temp": temp, "content": result["content"], "ok": True}
        except Exception as e:
            error_msg = str(e)
            print(f"[Multimodel Ensemble] {model} temp={temp} FAILED: {error_msg}")
            return {"model": model, "temp": temp, "content": error_msg, "ok": False}

    # Build all (model, temp) pairs and query in parallel
    tasks = [(m, t) for m, temps in ENSEMBLE_CONFIGS for t in temps]
    responses = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=9) as executor:
        futures = {executor.submit(query_single, m, t): (m, t) for m, t in tasks}
        for future in concurrent.futures.as_completed(futures):
            resp = future.result()
            responses.append(resp)
            if resp['ok']:
                print(f"[Multimodel Ensemble] {resp['model']} temp={resp['temp']} ok={resp['ok']}")

    successful = [r for r in responses if r["ok"]]
    if not successful:
        # Print all errors for debugging
        print("\n=== All ensemble queries failed. Errors: ===")
        for r in responses:
            print(f"  {r['model']} temp={r['temp']}: {r['content']}")
        raise RuntimeError("All ensemble queries failed")

    # Build synthesis prompt
    original_text = ""
    for msg in prompt:
        if msg["role"] == "user":
            c = msg["content"]
            if isinstance(c, list):
                original_text += "".join(x.get("text", "") for x in c if isinstance(x, dict))
            elif isinstance(c, str):
                original_text += c

    candidates = "\n\n".join(
        f"--- Candidate ({r['model']}, temp={r['temp']}) ---\n{r['content']}"
        for r in successful
    )

    # Detect if this is a multiturn decision (candidates contain REGENERATE/FINISH)
    regenerate_count = sum(1 for r in successful if isinstance(r.get("content"), str) and "REGENERATE" in r["content"])
    finish_count = sum(1 for r in successful if isinstance(r.get("content"), str) and "FINISH" in r["content"])

    if is_multiturn:
        synthesis_system_prompt = f"""You are synthesizing {len(successful)} candidate responses for a multi-turn robot control task.

    DECISION ANALYSIS:
    - {regenerate_count} candidates voted REGENERATE
    - {finish_count} candidates voted FINISH

    SYNTHESIS RULES:
    1. Analyze critically and assume no candidate is fully correct
    2. Prefer explicit checks over assumptions
    3. Combine the best ideas from multiple candidates when appropriate
    4. If candidates disagree fundamentally, choose the more robust approach
    5. Combine best code ideas from REGENERATE candidates

    OUTPUT FORMAT (strict):
    - You may include brief reasoning first
    - Then output "REGENERATE" on its own line followed by exactly ONE fenced code block, OR output "FINISH" on its own line
    """
    else:
        synthesis_system_prompt = f"""You are synthesizing {len(successful)} candidate Python solutions into one optimal program.

    SYNTHESIS RULES:
    1. Analyze critically and assume no candidate is fully correct
    2. Prefer explicit checks over assumptions
    3. Combine the best ideas from multiple candidates when appropriate
    4. If candidates disagree fundamentally, choose the more robust approach

    OUTPUT FORMAT (strict):
    You may include reasoning before the fenced code block.
    Output ONLY ONE fenced code block (```python...```) containing the complete final solution.
    Do NOT include any other code blocks or code snippets outside this single block.
    """

    synthesis_user_prompt = f"""Synthesize the best solution.

    <original_task_description>
    {original_text}
    </original_task_description>

    <candidate_solutions>
    {candidates}
    </candidate_solutions>
    """

    synthesis_prompt = [
        {
            "role": "system",
            "content": synthesis_system_prompt,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": synthesis_user_prompt,
                }
            ],
        },
    ]

    synth_args = ModelQueryArgs(
        model=synthesis_model,
        server_url=args.server_url,
        api_key=args.api_key,
        temperature=0.2,
        max_tokens=args.max_tokens,
    )
    final = query_model(synth_args, synthesis_prompt)

    # Build text content for saving
    candidates_txt = "\n\n".join(
        f"{'='*60}\nModel: {r['model']}\nTemperature: {r['temp']}\nSuccess: {r['ok']}\n{'='*60}\n{r['content']}"
        for r in responses
    )
    synthesis_txt = f"Model: {synthesis_model}\n\n"
    synthesis_txt += f"{'='*60}\nREASONING\n{'='*60}\n{final.get('reasoning') or '(none)'}\n\n"
    synthesis_txt += f"{'='*60}\nOUTPUT\n{'='*60}\n{final['content']}"

    return {
        "content": final["content"],
        "reasoning": final.get("reasoning"),
        "all_responses": responses,
        "ensemble_candidates_txt": candidates_txt,
        "ensemble_synthesis_txt": synthesis_txt,
    }


def query_single_model_ensemble(
    args: "LaunchArgs | ModelQueryArgs",
    prompt: list[dict],
    model: str,
    is_multiturn = False,
) -> dict[str, Any]:
    """Query the same model 9 times (with temperatures 0.1 to 0.9) and synthesize final output.

    Args:
        args: Configuration with server URL and model settings
        prompt: Full prompt containing environment observation and possibly multi-turn decision prompt
        model: The model to use for both candidate generation and synthesis

    Returns:
        Dictionary containing synthesized content, reasoning, all responses, and text artifacts
    """

    def query_single(temp: float) -> dict:
        query_args = ModelQueryArgs(
            model=model,
            server_url=args.server_url,
            api_key=args.api_key,
            temperature=temp,
            max_tokens=args.max_tokens,
            reasoning_effort=getattr(args, "reasoning_effort", "medium"),
        )
        try:
            result = query_model(query_args, copy.deepcopy(prompt))
            return {"model": model, "temp": temp, "content": result["content"], "ok": True}
        except Exception as e:
            error_msg = str(e)
            print(f"[Single Model Ensemble] {model} temp={temp} FAILED: {error_msg}")
            return {"model": model, "temp": temp, "content": error_msg, "ok": False}

    # Query same model with 9 different temperatures (0.1 to 0.9)
    temperatures = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    responses = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=9) as executor:
        futures = {executor.submit(query_single, t): t for t in temperatures}
        for future in concurrent.futures.as_completed(futures):
            resp = future.result()
            responses.append(resp)
            if resp['ok']:
                print(f"[Single Model Ensemble] {resp['model']} temp={resp['temp']} ok={resp['ok']}")

    successful = [r for r in responses if r["ok"]]
    if not successful:
        # Print all errors for debugging
        print("\n=== All single model ensemble queries failed. Errors: ===")
        for r in responses:
            print(f"  {r['model']} temp={r['temp']}: {r['content']}")
        raise RuntimeError("All single model ensemble queries failed")

    # Build synthesis prompt
    original_text = ""
    for msg in prompt:
        if msg["role"] == "user":
            c = msg["content"]
            if isinstance(c, list):
                original_text += "".join(x.get("text", "") for x in c if isinstance(x, dict))
            elif isinstance(c, str):
                original_text += c

    candidates = "\n\n".join(
        f"--- Candidate (temp={r['temp']}) ---\n{r['content']}"
        for r in successful
    )

    # Detect if this is a multiturn decision (candidates contain REGENERATE/FINISH)
    regenerate_count = sum(1 for r in successful if isinstance(r.get("content"), str) and "REGENERATE" in r["content"])
    finish_count = sum(1 for r in successful if isinstance(r.get("content"), str) and "FINISH" in r["content"])

    if is_multiturn:
        synthesis_system_prompt = f"""You are synthesizing {len(successful)} candidate responses for a multi-turn robot control task.

    DECISION ANALYSIS:
    - {regenerate_count} candidates voted REGENERATE
    - {finish_count} candidates voted FINISH

    SYNTHESIS RULES:
    1. Analyze critically and assume no candidate is fully correct
    2. Prefer explicit checks over assumptions
    3. Combine the best ideas from multiple candidates when appropriate
    4. If candidates disagree fundamentally, choose the more robust approach
    5. Combine best code ideas from REGENERATE candidates

    OUTPUT FORMAT (strict):
    - You may include brief reasoning first
    - Then output "REGENERATE" on its own line followed by exactly ONE fenced code block, OR output "FINISH" on its own line
    """
    else: # first generation has no REGEN/FINISH candidates
        synthesis_system_prompt = f"""You are synthesizing {len(successful)} candidate Python solutions into one optimal program.

    SYNTHESIS RULES:
    1. Analyze critically and assume no candidate is fully correct
    2. Prefer explicit checks over assumptions
    3. Combine the best ideas from multiple candidates when appropriate
    4. If candidates disagree fundamentally, choose the more robust approach

    OUTPUT FORMAT (strict):
    You may include reasoning before the fenced code block.
    Output ONLY ONE fenced code block (```python...```) containing the complete final solution.
    Do NOT include any other code blocks or code snippets outside this single block.
    """

    synthesis_user_prompt = f"""Synthesize the best solution.

    <original_task_description>
    {original_text}
    </original_task_description>

    <candidate_solutions>
    {candidates}
    </candidate_solutions>
    """

    synthesis_prompt = [
        {
            "role": "system",
            "content": synthesis_system_prompt,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": synthesis_user_prompt,
                }
            ],
        },
    ]

    # Use the same model for synthesis
    synth_args = ModelQueryArgs(
        model=model,
        server_url=args.server_url,
        api_key=args.api_key,
        temperature=0.2,
        max_tokens=args.max_tokens,
    )
    final = query_model(synth_args, synthesis_prompt)

    # Build text content for saving
    candidates_txt = "\n\n".join(
        f"{'='*60}\nModel: {r['model']}\nTemperature: {r['temp']}\nSuccess: {r['ok']}\n{'='*60}\n{r['content']}"
        for r in responses
    )
    synthesis_txt = f"Model: {model}\n\n"
    synthesis_txt += f"{'='*60}\nREASONING\n{'='*60}\n{final.get('reasoning') or '(none)'}\n\n"
    synthesis_txt += f"{'='*60}\nOUTPUT\n{'='*60}\n{final['content']}"

    return {
        "content": final["content"],
        "reasoning": final.get("reasoning"),
        "all_responses": responses,
        "ensemble_candidates_txt": candidates_txt,
        "ensemble_synthesis_txt": synthesis_txt,
    }


# ---------------------------------------------------------------------------
# Backward-compatible aliases (underscore-prefixed names)
# ---------------------------------------------------------------------------

_query_model = query_model
_query_model_streaming = query_model_streaming
_query_model_ensemble = query_model_ensemble
_query_single_model_ensemble = query_single_model_ensemble
