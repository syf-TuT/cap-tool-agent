from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from capx.runtime_control.schema import CodeRegion, CodeRegionGroup


class LineageAmbiguityError(ValueError):
    pass


@dataclass(frozen=True)
class SourceRevision:
    revision: int
    source_sha256: str
    edit_kind: Literal["initial", "patch_region", "patch_group", "append_recovery"]
    parent_revision: int | None
    old_line_count: int


@dataclass
class UnitLineage:
    next_region_key: int = 1
    next_group_key: int = 1
    region_key_by_id: dict[str, str] = field(default_factory=dict)
    group_key_by_id: dict[str, str] = field(default_factory=dict)
    executed_region_keys: set[str] = field(default_factory=set)
    executed_group_keys: set[str] = field(default_factory=set)

    @classmethod
    def create(
        cls,
        regions: list[CodeRegion],
        groups: list[CodeRegionGroup],
    ) -> UnitLineage:
        lineage = cls()
        for region in regions:
            lineage.region_key_by_id[region.region_id] = lineage.allocate_region_key()
        for group in groups:
            lineage.group_key_by_id[group.group_id] = lineage.allocate_group_key()
        return lineage

    def allocate_region_key(self) -> str:
        key = f"region_key_{self.next_region_key:06d}"
        self.next_region_key += 1
        return key

    def allocate_group_key(self) -> str:
        key = f"group_key_{self.next_group_key:06d}"
        self.next_group_key += 1
        return key
