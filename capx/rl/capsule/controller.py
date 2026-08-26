"""Frozen Controller adapters for building repair traces without simulator replay."""

from __future__ import annotations

import ast
import json
import math
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlparse

from .group import CandidateCollectionError, CollectionInfrastructureError, ProgramCandidate
from .repair import BaseUnitSpan, RepairDraft
from .schema import ProgramReplayResultV1, RepairTraceV1, ReplayOutcome, TaskInstanceV1

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_OUTER_PYTHON_FENCE = re.compile(
    r"\A(?P<open>```(?:python|py)?[^\S\r\n]*(?:\r\n|\n))"
    r"(?P<body>.*)"
    r"(?P<close>```[^\S\r\n]*(?:(?:\r\n|\n))?)\Z",
    re.IGNORECASE | re.DOTALL,
)


class ControllerTransport(Protocol):
    def complete(self, messages: tuple[dict[str, str], ...]) -> str: ...


class ControllerProtocolError(CandidateCollectionError):
    """The Controller response cannot be interpreted as an edit submission."""


@dataclass(frozen=True)
class FrozenControllerConfig:
    endpoint: str
    model: str
    api_key_env: str
    frozen: bool = True
    max_turns: int = 12
    request_timeout_s: float = 300.0
    max_output_tokens: int = 4096
    stream: bool = False
    enable_thinking: bool = False
    temperature: float = 0.7

    def __post_init__(self) -> None:
        parsed = urlparse(self.endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("controller endpoint must be an absolute HTTP(S) URL")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("controller model must not be empty")
        if not isinstance(self.api_key_env, str) or not _ENV_NAME.fullmatch(self.api_key_env):
            raise ValueError("controller api_key_env must be an environment-variable name")
        if self.frozen is not True:
            raise ValueError("Controller must be explicitly frozen")
        if isinstance(self.max_turns, bool) or not isinstance(self.max_turns, int):
            raise TypeError("max_turns must be an integer")
        if self.max_turns < 1 or self.max_turns > 12:
            raise ValueError("max_turns must be between 1 and 12")
        if (
            isinstance(self.request_timeout_s, bool)
            or not isinstance(self.request_timeout_s, (int, float))
            or not math.isfinite(float(self.request_timeout_s))
            or self.request_timeout_s <= 0
        ):
            raise ValueError("request_timeout_s must be a positive finite number")
        if isinstance(self.max_output_tokens, bool) or not isinstance(
            self.max_output_tokens, int
        ):
            raise TypeError("max_output_tokens must be an integer")
        if self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be a positive integer")
        for field_name in ("stream", "enable_thinking"):
            value = getattr(self, field_name)
            if not isinstance(value, bool):
                raise TypeError(f"{field_name} must be a boolean")
            if value is not False:
                raise ValueError(f"{field_name} must be false for the frozen Controller")
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(float(self.temperature))
            or self.temperature < 0
            or self.temperature > 2
        ):
            raise ValueError("temperature must be a finite number between zero and two")


class OpenAICompatibleControllerTransport:
    """Lazy OpenAI-compatible client backed by dedicated Controller credentials."""

    def __init__(
        self,
        config: FrozenControllerConfig,
        *,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        if not isinstance(config, FrozenControllerConfig):
            raise TypeError("config must be FrozenControllerConfig")
        self.config = config
        self._client_factory = client_factory
        self._client: Any | None = None

    def _make_client(self) -> Any:
        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            raise CollectionInfrastructureError(
                f"Controller credential environment variable {self.config.api_key_env!r} is unset"
            )
        factory = self._client_factory
        if factory is None:
            from openai import OpenAI

            factory = OpenAI
        return factory(
            api_key=api_key,
            base_url=self.config.endpoint,
            timeout=float(self.config.request_timeout_s),
        )

    def complete(self, messages: tuple[dict[str, str], ...]) -> str:
        if self._client is None:
            self._client = self._make_client()
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=list(messages),
            temperature=float(self.config.temperature),
            max_tokens=self.config.max_output_tokens,
            stream=self.config.stream,
            extra_body={"enable_thinking": self.config.enable_thinking},
            response_format={"type": "json_object"},
        )
        choices = getattr(response, "choices", None)
        if not isinstance(choices, (list, tuple)) or len(choices) != 1:
            raise ControllerProtocolError("Controller completion must contain exactly one choice")
        content = getattr(getattr(choices[0], "message", None), "content", None)
        if not isinstance(content, str):
            raise ControllerProtocolError("Controller completion content must be text")
        return content

    def close(self) -> None:
        """Release the dedicated HTTP client without creating it on unused trajectories."""

        client, self._client = self._client, None
        close = getattr(client, "close", None)
        if callable(close):
            close()


def _character_offset(lines: list[str], line_starts: list[int], line: int, byte_column: int) -> int:
    line_text = lines[line - 1]
    try:
        character_column = len(line_text.encode("utf-8")[:byte_column].decode("utf-8"))
    except UnicodeDecodeError as error:
        raise ValueError("AST column is not aligned to a UTF-8 character") from error
    return line_starts[line - 1] + character_column


def _python_statement_spans(source: str, *, source_offset: int = 0) -> tuple[BaseUnitSpan, ...]:
    try:
        module = ast.parse(source, mode="exec")
    except (SyntaxError, ValueError):
        return ()
    if not module.body:
        return ()

    lines = source.splitlines(keepends=True)
    line_starts: list[int] = []
    offset = 0
    for line_text in lines:
        line_starts.append(offset)
        offset += len(line_text)
    spans: list[BaseUnitSpan] = []
    for index, node in enumerate(module.body):
        start_node = node
        decorators = getattr(node, "decorator_list", ())
        if decorators:
            start_node = min(
                decorators,
                key=lambda decorator: (decorator.lineno, decorator.col_offset),
            )
        end_line = getattr(node, "end_lineno", None)
        end_column = getattr(node, "end_col_offset", None)
        if end_line is None or end_column is None:
            return ()
        start = _character_offset(lines, line_starts, start_node.lineno, start_node.col_offset)
        end = _character_offset(lines, line_starts, end_line, end_column)
        if start < 0 or end < start or end > len(source):
            return ()
        spans.append(
            BaseUnitSpan(
                f"group_{index}",
                source_offset + start,
                source_offset + end,
                source[start:end],
            )
        )
    if any(left.end_offset > right.start_offset for left, right in zip(spans, spans[1:])):
        return ()
    return tuple(spans)


def python_base_unit_spans(source: str) -> tuple[BaseUnitSpan, ...]:
    """Describe immutable P0 bytes as stable editable units without normalizing source."""

    if not isinstance(source, str):
        raise TypeError("source must be text")
    spans = _python_statement_spans(source)
    if spans:
        return spans

    fenced = _OUTER_PYTHON_FENCE.fullmatch(source)
    if fenced is not None:
        opener = fenced.group("open")
        body = fenced.group("body")
        closer = fenced.group("close")
        body_start = len(opener)
        body_spans = _python_statement_spans(body, source_offset=body_start)
        if body_spans:
            close_start = body_start + len(body)
            return (
                BaseUnitSpan("fence_open", 0, body_start, opener),
                *body_spans,
                BaseUnitSpan("fence_close", close_start, len(source), closer),
            )

    return (BaseUnitSpan("program", 0, len(source), source),)


_SYSTEM_PROMPT = """You are a frozen repair Controller for a robot Python program.
Propose exactly one JSON edit per turn. Do not execute the program and do not ask for simulator
replay. Allowed actions are:
- {"action":"append","generation_id":"...","unit_id":"...","source":"...","rationale":"..."}
- {"action":"replace","target":"base:<unit> or recovery:<generation>:<unit>",
  "source":"...","rationale":"..."}
- {"action":"inspect","message":"..."}
- {"action":"finish","rationale":"..."}
Targets are stable across revisions. Return one JSON object and no Markdown.
Markdown fences in the immutable Actor P0 are protocol errors already observed by replay. Never
treat them as valid Python or silently clean them. When base:fence_open and base:fence_close are
available, remove each fence explicitly with a replace action whose source is the empty string;
both committed edits must occur before finishing or making semantic repairs.
"""


def _state_message(
    task: TaskInstanceV1,
    p0_result: ProgramReplayResultV1,
    draft: RepairDraft,
    feedback: str,
) -> str:
    state = {
        "task_id": task.task_id,
        "environment_seed": task.environment_seed,
        "repair_trajectory_id": draft.repair_trajectory_id,
        "task_prompt": task.prompt,
        "current_revision": draft.current_revision,
        "current_source": draft.current_source,
        "editable_units": draft.editable_units,
        "base_failure": {
            "outcome": p0_result.outcome.value,
            "raw_reward": p0_result.raw_reward,
            "task_completed": p0_result.task_completed,
            "truncated": p0_result.truncated,
            "error_type": p0_result.error_type,
            "error_message": p0_result.error_message,
            "diagnostics": p0_result.to_dict()["diagnostics"],
        },
        "feedback": feedback,
    }
    return json.dumps(state, ensure_ascii=False, sort_keys=True, indent=2)


class ControllerRepairCollector:
    """Collect one complete edit trajectory; never imports or invokes a simulator."""

    def __init__(self, *, transport: ControllerTransport, max_turns: int = 12) -> None:
        if isinstance(max_turns, bool) or not isinstance(max_turns, int):
            raise TypeError("max_turns must be an integer")
        if max_turns < 1 or max_turns > 12:
            raise ValueError("max_turns must be between 1 and 12")
        self.transport = transport
        self.max_turns = max_turns

    def __call__(
        self,
        task: TaskInstanceV1,
        p0: ProgramCandidate,
        p0_result: ProgramReplayResultV1,
        p0_rank: int,
        trajectory_index: int,
        repair_trajectory_id: str,
    ) -> RepairTraceV1:
        if not isinstance(task, TaskInstanceV1):
            raise TypeError("task must be TaskInstanceV1")
        if not isinstance(p0, ProgramCandidate):
            raise TypeError("p0 must be ProgramCandidate")
        if not isinstance(p0_result, ProgramReplayResultV1):
            raise TypeError("p0_result must be ProgramReplayResultV1")
        result_identity = (
            p0_result.task_id,
            p0_result.environment_seed,
            p0_result.program_sample_id,
            p0_result.source,
            p0_result.initial_state_sha256,
        )
        expected_identity = (
            task.task_id,
            task.environment_seed,
            p0.program_sample_id,
            p0.source,
            task.initial_state_sha256,
        )
        if result_identity != expected_identity:
            raise ValueError("p0_result identity does not match task and P0")
        if p0_result.outcome is ReplayOutcome.SUCCESS or p0_result.binary_reward != 0.0:
            raise ValueError("repair collector requires a verified failed P0 result")
        if p0_rank not in {0, 1} or trajectory_index not in {0, 1}:
            raise ValueError("p0_rank and trajectory_index must each be zero or one")
        draft = RepairDraft(
            task_id=task.task_id,
            environment_seed=task.environment_seed,
            program_sample_id=p0.program_sample_id,
            repair_trajectory_id=repair_trajectory_id,
            base_source=p0.source,
            base_units=python_base_unit_spans(p0.source),
            max_turns=self.max_turns,
        )
        system_message = {"role": "system", "content": _SYSTEM_PROMPT}
        state_message = {
            "role": "user",
            "content": _state_message(task, p0_result, draft, "start repair"),
        }
        for _ in range(self.max_turns):
            try:
                response = self.transport.complete((system_message, state_message))
            except ControllerProtocolError:
                raise
            except Exception as error:
                raise CollectionInfrastructureError(
                    f"controller request failed: {type(error).__name__}: {error}"
                ) from error
            if not isinstance(response, str):
                raise ControllerProtocolError("Controller transport response must be text")
            submission = draft.submit_json(response)
            if draft.finished:
                break
            if submission.committed and submission.edit is not None:
                feedback = (
                    f"committed revision {submission.edit.output_revision} at "
                    f"{submission.edit.target}"
                )
            elif submission.audit is not None:
                feedback = (
                    f"{submission.audit.event_type}/{submission.audit.status}: "
                    f"{submission.audit.message}"
                )
            else:
                raise ControllerProtocolError("repair draft returned no edit or audit")
            state_message = {
                "role": "user",
                "content": _state_message(task, p0_result, draft, feedback),
            }
        return draft.to_trace()

    def close(self) -> None:
        close = getattr(self.transport, "close", None)
        if callable(close):
            close()


__all__ = [
    "ControllerProtocolError",
    "ControllerRepairCollector",
    "ControllerTransport",
    "FrozenControllerConfig",
    "OpenAICompatibleControllerTransport",
    "python_base_unit_spans",
]
