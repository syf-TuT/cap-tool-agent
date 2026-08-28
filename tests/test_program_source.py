from __future__ import annotations

import pytest

from capx.utils.launch_utils import _extract_code
from capx.utils.program_source import normalize_program_source


@pytest.mark.parametrize(
    ("raw_source", "expected"),
    [
        ("```python\nmove()\n```", "move()"),
        ("prefix\n```python\nmove()\n```\nsuffix", "move()"),
        ("\n  move()\n", "move()"),
        ("```py\nmove()\n```", "```py\nmove()"),
        ("```\nmove()\n```", "```\nmove()"),
    ],
)
def test_normalize_program_source_preserves_legacy_extraction_behavior(
    raw_source: str, expected: str
) -> None:
    assert normalize_program_source(raw_source) == expected
    assert _extract_code(raw_source) == [expected]


def test_normalize_program_source_uses_the_final_closing_fence() -> None:
    raw_source = "```python\nfirst()\n```\ntext\n```"

    assert normalize_program_source(raw_source) == "first()\n```\ntext"

