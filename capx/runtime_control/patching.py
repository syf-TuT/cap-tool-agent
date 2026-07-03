from __future__ import annotations

from capx.runtime_control.schema import CodeRegion


def replace_region_source(source: str, region: CodeRegion, replacement: str) -> str:
    lines = source.splitlines()
    replacement_lines = replacement.splitlines()
    start = region.start_line - 1
    end = region.end_line
    patched_lines = [*lines[:start], *replacement_lines, *lines[end:]]
    trailing_newline = "\n" if source.endswith("\n") else ""
    return "\n".join(patched_lines) + trailing_newline
