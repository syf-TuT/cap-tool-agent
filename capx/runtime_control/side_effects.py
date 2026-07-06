from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def collect_side_effect_calls(apis: Iterable[Any]) -> set[str]:
    """Gather rollback-relevant effect-primitive names declared by APIs.

    Each API may declare its effect primitives via a ``side_effect_functions()``
    method returning an iterable of function names. APIs predating that method
    are skipped. The union across all APIs is the environment's effect-primitive
    set, used to mark ``has_robot_side_effect`` during segmentation.
    """
    names: set[str] = set()
    for api in apis:
        declare = getattr(api, "side_effect_functions", None)
        if not callable(declare):
            continue
        names.update(declare())
    return names
