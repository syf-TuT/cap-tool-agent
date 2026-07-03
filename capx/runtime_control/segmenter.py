from __future__ import annotations

import ast

from capx.runtime_control.schema import CodeRegion


def segment_python_code(source: str) -> list[CodeRegion]:
    """Split Python source into deterministic top-level execution regions."""
    module = ast.parse(source)
    lines = source.splitlines()
    regions: list[CodeRegion] = []

    for idx, node in enumerate(module.body, start=1):
        start_line = getattr(node, "lineno", None)
        end_line = getattr(node, "end_lineno", None)
        if start_line is None or end_line is None:
            continue
        region_source = "\n".join(lines[start_line - 1 : end_line])
        regions.append(
            CodeRegion(
                region_id=f"region_{idx}",
                start_line=start_line,
                end_line=end_line,
                source=region_source,
            )
        )

    return regions
