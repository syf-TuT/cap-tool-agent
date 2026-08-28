"""Canonicalize model responses before executing them as Python programs."""

from __future__ import annotations


def normalize_program_source(content: str) -> str:
    """Preserve the legacy outer ``python`` Markdown-fence extraction behavior."""

    fence_start = "```python\n"
    fence_end = "```"
    if fence_start in content:
        content = content[content.find(fence_start) + len(fence_start) :]
    if fence_end in content:
        content = content[: content.rfind(fence_end)]
    return content.strip()


__all__ = ["normalize_program_source"]
