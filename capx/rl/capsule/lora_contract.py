"""Fixed Qwen2.5-Coder-7B LoRA coverage contract used by runtime audits."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

QWEN25_CODER_7B_LAYER_COUNT = 28
QWEN25_ALL_LINEAR_PROJECTIONS = (
    "down_proj",
    "gate_proj",
    "k_proj",
    "o_proj",
    "q_proj",
    "up_proj",
    "v_proj",
)
QWEN25_ALL_LINEAR_TENSOR_COUNT = (
    QWEN25_CODER_7B_LAYER_COUNT * len(QWEN25_ALL_LINEAR_PROJECTIONS) * 2
)
QWEN25_PROJECTION_DIMENSIONS = {
    "q_proj": (3584, 3584),
    "k_proj": (3584, 512),
    "v_proj": (3584, 512),
    "o_proj": (3584, 3584),
    "gate_proj": (3584, 18944),
    "up_proj": (3584, 18944),
    "down_proj": (18944, 3584),
}

_LORA_KEY_RE = re.compile(
    r"(?:^|\.)layers\.(?P<layer>\d+)\."
    r"(?:[^.]+\.)*?(?P<projection>"
    + "|".join(QWEN25_ALL_LINEAR_PROJECTIONS)
    + r")\.(?:[^.]+\.)*?lora_(?P<side>[ab])(?:\.|$)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class QwenAllLinearCoverage:
    layer_count: int
    projection_suffixes: tuple[str, ...]
    tensor_count: int


def qwen_lora_tensor_identity(name: str) -> tuple[int, str, str]:
    """Return ``(layer, projection, side)`` for one Qwen LoRA tensor name."""

    match = _LORA_KEY_RE.search(name)
    if match is None:
        raise ValueError(f"unrecognized Qwen all-linear LoRA tensor name: {name}")
    return (
        int(match.group("layer")),
        match.group("projection").lower(),
        match.group("side").upper(),
    )


def validate_qwen_all_linear_coverage(names: Iterable[str]) -> QwenAllLinearCoverage:
    """Require exactly one A/B tensor for all seven projections in all 28 layers."""

    observed: dict[tuple[int, str, str], str] = {}
    for name in names:
        identity = qwen_lora_tensor_identity(name)
        if identity in observed:
            raise ValueError(
                "duplicate Qwen all-linear LoRA tensor identity "
                f"{identity!r}: {observed[identity]!r}, {name!r}"
            )
        observed[identity] = name
    expected = {
        (layer, projection, side)
        for layer in range(QWEN25_CODER_7B_LAYER_COUNT)
        for projection in QWEN25_ALL_LINEAR_PROJECTIONS
        for side in ("A", "B")
    }
    missing = sorted(expected - observed.keys())
    unexpected = sorted(observed.keys() - expected)
    if missing or unexpected:
        raise ValueError(
            "Qwen2.5-Coder-7B all-linear coverage is incomplete: "
            f"missing={missing[:3]!r} ({len(missing)} total), "
            f"unexpected={unexpected[:3]!r} ({len(unexpected)} total)"
        )
    return QwenAllLinearCoverage(
        layer_count=QWEN25_CODER_7B_LAYER_COUNT,
        projection_suffixes=QWEN25_ALL_LINEAR_PROJECTIONS,
        tensor_count=len(observed),
    )
