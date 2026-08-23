"""Prompt construction and conservative validation for critique-guided regeneration."""

from __future__ import annotations

import ast
import json
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from .repair import RepairInvariantError
from .schema import RepairTraceV1, TaskInstanceV1

TokenCounter = Callable[[str], int]


class RevisionRejectionReason(str, Enum):
    INPUT_OVERFLOW = "input_overflow"
    RESPONSE_OVERFLOW = "response_overflow"
    INCOMPLETE_PROGRAM = "incomplete_program"
    MARKDOWN_FENCE = "markdown_fence"
    SYNTAX_ERROR = "syntax_error"
    TRACE_MISMATCH = "trace_mismatch"
    TRACE_INVALID = "trace_invalid"


class RevisionRejection(ValueError):
    """Typed rejection which callers can persist without parsing error text."""

    def __init__(
        self,
        reason: RevisionRejectionReason,
        message: str,
        *,
        observed_tokens: int | None = None,
        token_limit: int | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.observed_tokens = observed_tokens
        self.token_limit = token_limit


@dataclass(frozen=True)
class RevisionPrompt:
    text: str
    input_token_count: int
    input_token_limit: int
    response_token_limit: int


def _default_token_counter(text: str) -> int:
    """A dependency-free fallback; production collectors should inject the actor tokenizer."""

    return len(text.split())


def _count_tokens(token_counter: TokenCounter, text: str) -> int:
    count = token_counter(text)
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise TypeError("token_counter must return a non-negative integer")
    return count


def build_revision_prompt(
    task: TaskInstanceV1,
    trace: RepairTraceV1,
    *,
    token_counter: TokenCounter = _default_token_counter,
    input_token_limit: int = 8192,
    response_token_limit: int = 2048,
) -> RevisionPrompt:
    """Build a full-regeneration prompt without truncating P0 or committed repair history."""

    if input_token_limit < 1 or response_token_limit < 1:
        raise ValueError("revision token limits must be positive")
    if task.task_id != trace.task_id or task.environment_seed != trace.environment_seed:
        raise RevisionRejection(
            RevisionRejectionReason.TRACE_MISMATCH,
            "repair trace task/environment seed does not match the task instance",
        )
    try:
        trace.reconstruct()
    except RepairInvariantError as error:
        raise RevisionRejection(
            RevisionRejectionReason.TRACE_INVALID,
            f"repair trace failed immutable reconstruction: {error}",
        ) from error

    repair_json = json.dumps(
        trace.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    )
    prompt_text = (
        "You are regenerating a complete independently executable robot program.\n"
        "Use the failed base program and the complete committed repair trajectory as critique.\n"
        "Do not emit edits, patches, explanations, or Markdown fences. Output only one complete "
        "Python program that can run directly after an environment reset.\n\n"
        f"TASK_ID: {task.task_id}\n"
        f"ENVIRONMENT_SEED: {task.environment_seed}\n"
        f"ENVIRONMENT: {task.environment}\n"
        f"API: {task.api}\n"
        f"PRIVILEGE: {task.privilege}\n\n"
        f"TASK:\n{task.prompt}\n\n"
        f"FAILED_BASE_PROGRAM_P0:\n{trace.base_source}\n\n"
        f"COMPLETE_REPAIR_TRACE_RHO:\n{repair_json}\n"
    )
    input_token_count = _count_tokens(token_counter, prompt_text)
    if input_token_count > input_token_limit:
        raise RevisionRejection(
            RevisionRejectionReason.INPUT_OVERFLOW,
            (
                f"revision prompt uses {input_token_count} tokens, exceeding the "
                f"{input_token_limit}-token input limit; prompt was not truncated"
            ),
            observed_tokens=input_token_count,
            token_limit=input_token_limit,
        )
    return RevisionPrompt(
        text=prompt_text,
        input_token_count=input_token_count,
        input_token_limit=input_token_limit,
        response_token_limit=response_token_limit,
    )


def validate_complete_program(
    source: str,
    *,
    token_counter: TokenCounter = _default_token_counter,
    response_token_limit: int = 2048,
    finish_reason: str | None = None,
    truncated: bool = False,
) -> str:
    """Return valid complete source byte-for-byte; reject rather than clean or truncate it."""

    if not isinstance(source, str):
        raise RevisionRejection(
            RevisionRejectionReason.INCOMPLETE_PROGRAM,
            "revision response must be source text",
        )
    if response_token_limit < 1:
        raise ValueError("response_token_limit must be positive")
    length_finish_reasons = {"length", "max_tokens", "max_output_tokens", "token_limit"}
    if truncated or (finish_reason is not None and finish_reason.lower() in length_finish_reasons):
        raise RevisionRejection(
            RevisionRejectionReason.INCOMPLETE_PROGRAM,
            "revision response ended at a token limit and may be truncated",
        )
    token_count = _count_tokens(token_counter, source)
    if token_count > response_token_limit:
        raise RevisionRejection(
            RevisionRejectionReason.RESPONSE_OVERFLOW,
            (
                f"revision response uses {token_count} tokens, exceeding the "
                f"{response_token_limit}-token limit; response was not truncated"
            ),
            observed_tokens=token_count,
            token_limit=response_token_limit,
        )
    if not source.strip():
        raise RevisionRejection(
            RevisionRejectionReason.INCOMPLETE_PROGRAM,
            "revision response is empty",
        )
    if "```" in source:
        raise RevisionRejection(
            RevisionRejectionReason.MARKDOWN_FENCE,
            "revision response contains a Markdown code fence",
        )
    try:
        module = ast.parse(source, mode="exec")
    except (SyntaxError, ValueError) as error:
        raise RevisionRejection(
            RevisionRejectionReason.SYNTAX_ERROR,
            f"revision response is not a complete Python program: {error}",
        ) from error
    if not module.body:
        raise RevisionRejection(
            RevisionRejectionReason.INCOMPLETE_PROGRAM,
            "revision response has no executable Python statements",
        )
    try:
        compile(source, "<capsule-revision>", "exec")
    except (SyntaxError, TypeError, ValueError) as error:
        raise RevisionRejection(
            RevisionRejectionReason.SYNTAX_ERROR,
            f"revision response is not a compilable Python program: {error}",
        ) from error
    return source


__all__ = [
    "RevisionPrompt",
    "RevisionRejection",
    "RevisionRejectionReason",
    "TokenCounter",
    "build_revision_prompt",
    "validate_complete_program",
]
