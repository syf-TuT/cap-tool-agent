"""Prepare data/config without models; preview with --validate-only or --dry-run."""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from .common import (
    ConfigValidationError,
    add_validation_arguments,
    load_and_validate_server_config,
    validation_requested,
)


@dataclass(frozen=True)
class PreparationResult:
    record_count: int
    dataset_path: Path
    config_path: Path


def _load_source_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ConfigValidationError(f"source dataset must be an existing JSONL file: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ConfigValidationError(
                f"source dataset line {line_number} is not valid JSON: {error}"
            ) from error
        if not isinstance(payload, Mapping):
            raise ConfigValidationError(f"source dataset line {line_number} must be an object")
        record = dict(payload)
        for field_name in ("task_id", "prompt"):
            if not isinstance(record.get(field_name), str) or not record[field_name]:
                raise ConfigValidationError(
                    f"source dataset line {line_number} requires non-empty {field_name}"
                )
        records.append(record)
    if not records:
        raise ConfigValidationError("source dataset has no records")
    return records


def _expand_records(records: list[dict[str, Any]], seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    if not seeds or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ConfigValidationError("seeds must contain one or more integers")
    if any(seed < 0 for seed in seeds):
        raise ConfigValidationError("seeds must contain only non-negative integers")
    if len(seeds) != len(set(seeds)):
        raise ConfigValidationError("seeds must not contain duplicate values")
    expanded: list[dict[str, Any]] = []
    for record in records:
        for seed in seeds:
            item = dict(record)
            # Initial-state identity is server-produced evidence. Never propagate a source
            # dataset claim into the seed-resolution input bundle.
            item.pop("initial_state_sha256", None)
            item["schema_version"] = 1
            item["environment_seed"] = seed
            item["task_instance_id"] = f"{record['task_id']}:seed-{seed}"
            expanded.append(item)
    return expanded


def prepare(
    *,
    config_path: str | Path,
    source_dataset: str | Path,
    output_dir: str | Path,
    seeds: tuple[int, ...],
    validate_only: bool,
) -> PreparationResult:
    config_file = Path(config_path).expanduser().resolve()
    source_file = Path(source_dataset).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    # The dataset and resolved output paths are the products of this command, so they are not
    # required to exist yet. Schema/service invariants and the source JSONL are still validated.
    config = load_and_validate_server_config(config_file, check_runtime_paths=False)
    records = _expand_records(_load_source_records(source_file), seeds)
    dataset_path = destination / "capsule_rl.dataset.jsonl"
    resolved_config_path = destination / "capsule_rl.resolved.yaml"
    if destination.exists():
        raise FileExistsError(f"preparation output already exists: {destination}")
    nearest_parent = destination.parent
    while not nearest_parent.exists() and nearest_parent != nearest_parent.parent:
        nearest_parent = nearest_parent.parent
    if not nearest_parent.is_dir():
        raise ConfigValidationError(f"output directory has no existing parent: {destination}")
    print(
        json.dumps(
            {
                "mode": "VALIDATION ONLY" if validate_only else "WRITE",
                "source": str(source_file),
                "seeds": list(seeds),
                "records": len(records),
                "dataset": str(dataset_path),
                "config": str(resolved_config_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if validate_only:
        return PreparationResult(len(records), dataset_path, resolved_config_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.staging-{uuid4().hex}"
    staging.mkdir()
    staging_dataset = staging / dataset_path.name
    staging_config = staging / resolved_config_path.name
    try:
        with staging_dataset.open("x", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False)
                )
                stream.write("\n")
        config["runtime"]["dataset_path"] = str(dataset_path)
        config["runtime"]["output_dir"] = str(destination / "outputs")
        staging_config.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        staging.replace(destination)
    except BaseException:
        # The unique staging directory is owned by this invocation.  Never remove the
        # destination: another process may have won the exclusive publication race.
        if staging.exists():
            try:
                shutil.rmtree(staging)
            except BaseException:
                pass
        raise
    return PreparationResult(len(records), dataset_path, resolved_config_path)


def _parse_seeds(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare Capsule-RL dataset/config artifacts without launching runtime work."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=_parse_seeds, required=True)
    add_validation_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prepare(
        config_path=args.config,
        source_dataset=args.source_dataset,
        output_dir=args.output_dir,
        seeds=args.seeds,
        validate_only=validation_requested(args),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
