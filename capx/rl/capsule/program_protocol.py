"""Explicit Program sampling contract, independent of model/runtime dependencies."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from typing import Any


class ProgramProtocolError(ValueError):
    """A saved task or configuration does not match the declared sampling protocol."""


def program_sampling(config: Mapping[str, Any]) -> dict[str, Any]:
    service = config.get("program_service", {})
    if not isinstance(service, Mapping):
        raise ProgramProtocolError("program_service must be a mapping")
    sampling = service.get("sampling")
    fields = ("prompt_sha256", "system_prompt_sha256")
    if sampling is None and not any(field in service for field in fields):
        return {}  # Explicit opt-in preserves other task families and historical configurations.
    required = {"temperature", "top_p", "top_k", "repetition_penalty", "max_tokens"}
    if not isinstance(sampling, Mapping) or set(sampling) != required:
        raise ProgramProtocolError(f"program_service.sampling requires exactly {sorted(required)}")
    for field in fields:
        value = service.get(field)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ProgramProtocolError(f"program_service.{field} must be a lowercase SHA-256")
    system = service.get("system_prompt")
    if not isinstance(system, str) or hashlib.sha256(system.encode()).hexdigest() != service[
        "system_prompt_sha256"
    ]:
        raise ProgramProtocolError("program_service.system_prompt does not match system_prompt_sha256")
    ranges = {
        "temperature": (0, 2, True),
        "top_p": (0, 1, False),
        "repetition_penalty": (0, float("inf"), False),
    }
    for field, (lower, upper, inclusive) in ranges.items():
        value = sampling[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value > upper
            or (value < lower if inclusive else value <= lower)
        ):
            raise ProgramProtocolError(f"program_service.sampling.{field} is outside its valid range")
    top_k = sampling["top_k"]
    if type(top_k) is not int or (top_k != -1 and top_k < 1):
        raise ProgramProtocolError("program_service.sampling.top_k must be -1 or a positive integer")
    max_tokens = sampling["max_tokens"]
    if type(max_tokens) is not int or max_tokens < 1:
        raise ProgramProtocolError("program_service.sampling.max_tokens must be a positive integer")
    if max_tokens < int(config["capsule"]["revision_response_max_tokens"]):
        raise ProgramProtocolError("program_service.sampling.max_tokens cannot truncate revisions")
    return dict(sampling)


def validate_program_prompt(config: Mapping[str, Any], prompt: str) -> None:
    if not program_sampling(config):
        return
    expected = config["program_service"]["prompt_sha256"]
    actual = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if actual != expected:
        raise ProgramProtocolError(
            f"dataset prompt sha256 {actual} does not match Program protocol {expected}; "
            "regenerate the dataset from the matching source prompt and update runtime.dataset_path"
        )


def program_response_token_limit(config: Mapping[str, Any]) -> int:
    return int(
        program_sampling(config).get("max_tokens", config["capsule"]["revision_response_max_tokens"])
    )
