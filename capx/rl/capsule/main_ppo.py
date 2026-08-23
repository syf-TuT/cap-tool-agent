"""Side-effect-safe entrypoint for the project-owned Capsule trainer.

``--validate-only`` and ``--dry-run`` perform YAML, invariant, path, git-SHA, and static VeRL API
checks.  They do not import a trainer factory (and therefore cannot start Ray, a model service,
or an optimizer).  The executable path has no fallback to VeRL's ordinary PPO entrypoint: a
project-owned factory must be named explicitly in the configuration.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from .compat import (
    CapsuleConfigError,
    VeRLCompatibilityError,
    bind_pinned_verl_import,
    check_verl_compatibility,
    validate_capsule_config,
)
from .provenance import file_sha256, runtime_dependency_hashes
from .schema import TaskInstanceV1


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ConfigLoadError(ValueError):
    """The config file could not be safely loaded as a YAML mapping."""


class TrainerFactoryError(RuntimeError):
    """The explicit project trainer factory is absent or has an invalid interface."""


def register_capsule_critique_policy_loss() -> bool:
    """Import the Torch/VeRL loss adapter only after training is explicitly requested."""

    from .policy_loss import register_capsule_critique_policy_loss as register

    return register()


def _expand_string(value: str) -> str:
    return os.path.expanduser(os.path.expandvars(value))


def _expand_config(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _expand_config(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_config(item) for item in value]
    if isinstance(value, str):
        return _expand_string(value)
    return value


def load_and_validate_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load one YAML file, expand environment syntax, and validate local invariants."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigLoadError(f"config file does not exist: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ConfigLoadError(f"cannot parse config {config_path}: {error}") from error
    if not isinstance(raw, Mapping):
        raise ConfigLoadError("config root must be a YAML mapping")
    config = _expand_config(raw)
    validate_capsule_config(config)
    return config, config_path


def load_dotted_object(dotted_path: str) -> Any:
    """Import ``module:object`` (or ``module.object``) only on the executable path."""

    if not isinstance(dotted_path, str) or not dotted_path.strip():
        raise TrainerFactoryError("trainer_factory must be a non-empty dotted path")
    selected = dotted_path.strip()
    if ":" in selected:
        module_name, object_name = selected.rsplit(":", 1)
    else:
        module_name, separator, object_name = selected.rpartition(".")
        if not separator:
            raise TrainerFactoryError(
                "trainer_factory must use 'module:object' or 'module.object' syntax"
            )
    if not module_name or not object_name:
        raise TrainerFactoryError("trainer_factory has an incomplete dotted path")
    try:
        module = importlib.import_module(module_name)
        return getattr(module, object_name)
    except (ImportError, AttributeError) as error:
        raise TrainerFactoryError(f"cannot load trainer_factory {selected!r}: {error}") from error


def _runtime_path(config: Mapping[str, Any], field_name: str) -> Path:
    runtime = config.get("runtime")
    if not isinstance(runtime, Mapping):
        raise TrainerFactoryError("runtime must be a mapping")
    value = runtime.get(field_name)
    if not isinstance(value, str) or not value:
        raise TrainerFactoryError(f"runtime.{field_name} must be a non-empty path")
    project_root_value = runtime.get("project_root")
    repository_root = Path(__file__).resolve().parents[3]
    if isinstance(project_root_value, str) and project_root_value:
        configured_root = Path(project_root_value).expanduser()
        project_root = (
            configured_root.resolve()
            if configured_root.is_absolute()
            else (repository_root / configured_root).resolve()
        )
    else:
        project_root = repository_root
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _validate_existing_runtime_path(
    config: Mapping[str, Any], field_name: str, *, directory: bool
) -> Path:
    path = _runtime_path(config, field_name)
    valid = path.is_dir() if directory else path.is_file()
    if not valid:
        expected_kind = "directory" if directory else "file"
        raise TrainerFactoryError(
            f"runtime.{field_name} must reference an existing {expected_kind}: {path}"
        )
    return path


def _validate_output_dir(config: Mapping[str, Any]) -> Path:
    output_dir = _runtime_path(config, "output_dir")
    if output_dir.exists() and not output_dir.is_dir():
        raise TrainerFactoryError(
            f"runtime.output_dir exists but is not a directory: {output_dir}"
        )
    existing_parent = output_dir
    while not existing_parent.exists() and existing_parent != existing_parent.parent:
        existing_parent = existing_parent.parent
    if not existing_parent.is_dir():
        raise TrainerFactoryError(
            f"runtime.output_dir has no existing parent directory: {output_dir}"
        )
    if not os.access(existing_parent, os.W_OK):
        raise TrainerFactoryError(
            "runtime.output_dir nearest existing parent is not writable: "
            f"{existing_parent}"
        )
    return output_dir


def validate_local_execution_inputs(config: Mapping[str, Any]) -> dict[str, Path]:
    """Validate local filesystem inputs without importing runtime or trainer modules."""

    trainer_factory = config.get("trainer_factory")
    if not isinstance(trainer_factory, str) or not trainer_factory.strip():
        raise TrainerFactoryError("trainer_factory must be a non-empty dotted path")
    return {
        "dataset_path": _validate_existing_runtime_path(
            config, "dataset_path", directory=False
        ),
        "program_model_path": _validate_existing_runtime_path(
            config, "program_model_path", directory=True
        ),
        "verl_resolved_config_path": _validate_existing_runtime_path(
            config, "verl_resolved_config_path", directory=False
        ),
        "bundle_manifest_path": _validate_existing_runtime_path(
            config, "bundle_manifest_path", directory=False
        ),
        "output_dir": _validate_output_dir(config),
    }


def _load_json_mapping(path: Path, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TrainerFactoryError(f"{label} is not valid UTF-8 JSON: {error}") from error
    if not isinstance(payload, Mapping):
        raise TrainerFactoryError(f"{label} root must be a JSON mapping")
    return payload


def _manifest_path(manifest_path: Path, value: object, field_name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise TrainerFactoryError(f"bundle manifest {field_name} must be a non-empty path")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()


def _manifest_sha256(manifest: Mapping[str, Any], field_name: str) -> str:
    value = manifest.get(field_name)
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise TrainerFactoryError(
            f"bundle manifest {field_name} must be lowercase SHA-256"
        )
    return value


def _checked_file_sha256(path: Path, label: str) -> str:
    try:
        return file_sha256(path)
    except OSError as error:
        raise TrainerFactoryError(f"cannot hash {label}: {error}") from error


def _typed_task_identities(value: object, field_name: str) -> list[dict[str, object]]:
    if not isinstance(value, list) or not value:
        raise TrainerFactoryError(f"{field_name} must be a non-empty list")
    normalized: list[dict[str, object]] = []
    observed: set[tuple[str, int]] = set()
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise TrainerFactoryError(f"{field_name}[{index}] must be a mapping")
        task_id = item.get("task_id")
        environment_seed = item.get("environment_seed")
        initial_state_sha256 = item.get("initial_state_sha256")
        if not isinstance(task_id, str) or not task_id:
            raise TrainerFactoryError(f"{field_name}[{index}].task_id must be non-empty")
        if (
            isinstance(environment_seed, bool)
            or not isinstance(environment_seed, int)
            or environment_seed < 0
        ):
            raise TrainerFactoryError(
                f"{field_name}[{index}].environment_seed must be non-negative"
            )
        if (
            not isinstance(initial_state_sha256, str)
            or _SHA256_RE.fullmatch(initial_state_sha256) is None
        ):
            raise TrainerFactoryError(
                f"{field_name}[{index}].initial_state_sha256 must be SHA-256"
            )
        identity = (task_id, environment_seed)
        if identity in observed:
            raise TrainerFactoryError(f"{field_name} contains duplicate identities")
        observed.add(identity)
        normalized.append(
            {
                "task_id": task_id,
                "environment_seed": environment_seed,
                "initial_state_sha256": initial_state_sha256,
            }
        )
    return normalized


def verify_bundle_provenance(
    config: Mapping[str, Any], config_path: str | Path
) -> dict[str, Any]:
    """Re-hash the formal training bundle without importing VeRL, Torch, or a trainer."""

    resolved_config_path = Path(config_path).expanduser().resolve()
    manifest_path = _validate_existing_runtime_path(
        config, "bundle_manifest_path", directory=False
    )
    manifest = _load_json_mapping(manifest_path, "bundle manifest")
    if type(manifest.get("schema_version")) is not int or manifest.get(
        "schema_version"
    ) != 1:
        raise TrainerFactoryError("bundle manifest schema_version must be 1")
    if manifest.get("artifact_type") != "capsule_seed_resolved_dataset":
        raise TrainerFactoryError("bundle manifest artifact_type is not Capsule seed-resolved")

    output_config_path = _manifest_path(
        manifest_path, manifest.get("output_config_path"), "output_config_path"
    )
    if output_config_path != resolved_config_path:
        raise TrainerFactoryError("bundle manifest output config path does not match --config")
    output_config_sha256 = _manifest_sha256(manifest, "output_config_sha256")
    if _checked_file_sha256(resolved_config_path, "output config") != output_config_sha256:
        raise TrainerFactoryError("bundle output config SHA does not match --config bytes")
    if _manifest_path(manifest_path, manifest.get("config_path"), "config_path") != (
        output_config_path
    ) or _manifest_sha256(manifest, "config_sha256") != output_config_sha256:
        raise TrainerFactoryError(
            "bundle legacy config path/hash aliases contradict output config provenance"
        )

    dataset_path = _runtime_path(config, "dataset_path")
    manifest_dataset_path = _manifest_path(
        manifest_path, manifest.get("output_dataset_path"), "output_dataset_path"
    )
    if manifest_dataset_path != dataset_path:
        raise TrainerFactoryError(
            "bundle manifest output dataset path does not match runtime.dataset_path"
        )
    output_dataset_sha256 = _manifest_sha256(manifest, "output_dataset_sha256")
    if _checked_file_sha256(dataset_path, "output dataset") != output_dataset_sha256:
        raise TrainerFactoryError("bundle output dataset SHA does not match dataset bytes")
    if _manifest_path(manifest_path, manifest.get("dataset_path"), "dataset_path") != (
        manifest_dataset_path
    ) or _manifest_sha256(manifest, "dataset_sha256") != output_dataset_sha256:
        raise TrainerFactoryError(
            "bundle legacy dataset path/hash aliases contradict output dataset provenance"
        )

    source_config_path = _manifest_path(
        manifest_path, manifest.get("source_config_path"), "source_config_path"
    )
    source_config_sha256 = _manifest_sha256(manifest, "source_config_sha256")
    if not source_config_path.is_file() or _checked_file_sha256(
        source_config_path, "source config"
    ) != (
        source_config_sha256
    ):
        raise TrainerFactoryError(
            "bundle source config SHA does not match source config bytes"
        )
    source_dataset_path = _manifest_path(
        manifest_path, manifest.get("source_dataset_path"), "source_dataset_path"
    )
    source_dataset_sha256 = _manifest_sha256(manifest, "source_dataset_sha256")
    if not source_dataset_path.is_file() or _checked_file_sha256(
        source_dataset_path, "source dataset"
    ) != (
        source_dataset_sha256
    ):
        raise TrainerFactoryError(
            "bundle source dataset SHA does not match source dataset bytes"
        )

    resolved_verl_path = _runtime_path(config, "verl_resolved_config_path")
    manifest_verl_path = _manifest_path(
        manifest_path,
        manifest.get("verl_resolved_config_path"),
        "verl_resolved_config_path",
    )
    if manifest_verl_path != resolved_verl_path:
        raise TrainerFactoryError(
            "bundle manifest resolved VeRL config path does not match runtime config"
        )
    resolved_verl_sha256 = _manifest_sha256(
        manifest, "verl_resolved_config_sha256"
    )
    if _checked_file_sha256(
        resolved_verl_path, "resolved VeRL config"
    ) != resolved_verl_sha256:
        raise TrainerFactoryError("bundle resolved VeRL config SHA does not match YAML bytes")

    try:
        dependency_hashes = runtime_dependency_hashes(config)
    except (OSError, ValueError) as error:
        raise TrainerFactoryError(f"cannot hash runtime dependencies: {error}") from error
    if dependency_hashes["resolved_environment_sha256"] != _manifest_sha256(
        manifest, "resolved_environment_sha256"
    ):
        raise TrainerFactoryError(
            "bundle resolved environment SHA does not match task/environment bytes"
        )
    if dependency_hashes["verl_resolved_config_sha256"] != resolved_verl_sha256:
        raise TrainerFactoryError("bundle resolved VeRL config dependency hash is inconsistent")

    audit_path = _manifest_path(
        manifest_path, manifest.get("gate7_audit_path"), "gate7_audit_path"
    )
    if audit_path != _runtime_path(config, "gate7_audit_path"):
        raise TrainerFactoryError(
            "bundle manifest Gate7 audit path does not match runtime.gate7_audit_path"
        )
    if not audit_path.is_file():
        raise TrainerFactoryError(f"bundle Gate7 audit does not exist: {audit_path}")
    audit_sha256 = _manifest_sha256(manifest, "gate7_audit_sha256")
    if _checked_file_sha256(audit_path, "Gate7 audit") != audit_sha256:
        raise TrainerFactoryError("bundle Gate7 audit SHA does not match audit bytes")
    audit = _load_json_mapping(audit_path, "Gate7 audit")
    if audit.get("runtime_verified") is not True:
        raise TrainerFactoryError("Gate7 audit must record runtime_verified=true")
    gate7_run_id = manifest.get("gate7_run_id")
    if not isinstance(gate7_run_id, str) or not gate7_run_id:
        raise TrainerFactoryError("bundle manifest gate7_run_id must be non-empty")
    if audit.get("run_id") != gate7_run_id:
        raise TrainerFactoryError("Gate7 audit run_id does not match bundle manifest")
    gate7_identities = _typed_task_identities(
        manifest.get("gate7_typed_task_identities"),
        "bundle manifest gate7_typed_task_identities",
    )
    if _typed_task_identities(
        audit.get("typed_task_identities"), "Gate7 audit typed_task_identities"
    ) != gate7_identities:
        raise TrainerFactoryError("Gate7 typed task identities do not match bundle manifest")
    for audit_field, manifest_field in (
        ("config_sha256", "source_config_sha256"),
        ("dataset_sha256", "source_dataset_sha256"),
        ("resolved_environment_sha256", "resolved_environment_sha256"),
        ("verl_resolved_config_sha256", "verl_resolved_config_sha256"),
    ):
        expected = _manifest_sha256(manifest, manifest_field)
        if audit.get(audit_field) != expected:
            raise TrainerFactoryError(
                f"Gate7 audit {audit_field} does not match bundle manifest"
            )

    try:
        tasks = [
            TaskInstanceV1.from_json(line)
            for line in dataset_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeError, KeyError, TypeError, ValueError) as error:
        raise TrainerFactoryError(
            f"bundle output dataset is not typed TaskInstanceV1 JSONL: {error}"
        ) from error
    output_identities = [
        {
            "task_id": task.task_id,
            "environment_seed": task.environment_seed,
            "initial_state_sha256": task.initial_state_sha256,
        }
        for task in tasks
    ]
    if output_identities != _typed_task_identities(
        manifest.get("output_task_identities"),
        "bundle manifest output_task_identities",
    ):
        raise TrainerFactoryError(
            "bundle output typed task identities do not match dataset rows"
        )
    record_count = manifest.get("record_count")
    if isinstance(record_count, bool) or not isinstance(record_count, int):
        raise TrainerFactoryError("bundle manifest record_count must be an integer")
    if record_count != len(tasks):
        raise TrainerFactoryError("bundle manifest record_count does not match dataset rows")
    output_identity_set = {
        (item["task_id"], item["environment_seed"], item["initial_state_sha256"])
        for item in output_identities
    }
    if not all(
        (item["task_id"], item["environment_seed"], item["initial_state_sha256"])
        in output_identity_set
        for item in gate7_identities
    ):
        raise TrainerFactoryError(
            "bundle output dataset does not contain every Gate7 typed task identity"
        )
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": _checked_file_sha256(manifest_path, "bundle manifest"),
        "gate7_run_id": gate7_run_id,
        "gate7_audit_sha256": audit_sha256,
        "dataset_sha256": output_dataset_sha256,
        "config_sha256": output_config_sha256,
        "resolved_environment_sha256": dependency_hashes[
            "resolved_environment_sha256"
        ],
        "verl_resolved_config_sha256": resolved_verl_sha256,
    }


def _project_git_sha(config: Mapping[str, Any]) -> str:
    runtime = config.get("runtime")
    configured_root = runtime.get("project_root") if isinstance(runtime, Mapping) else None
    repository_root = Path(__file__).resolve().parents[3]
    if isinstance(configured_root, str) and configured_root:
        configured_path = Path(configured_root).expanduser()
        root = (
            configured_path.resolve()
            if configured_path.is_absolute()
            else (repository_root / configured_path).resolve()
        )
    else:
        root = repository_root
    commands = {
        "sha": ["git", "-C", str(root), "rev-parse", "HEAD"],
        "top": ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
        "status": [
            "git",
            "-C",
            str(root),
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
    }
    results: dict[str, str] = {}
    for name, command in commands.items():
        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise TrainerFactoryError(
                f"cannot inspect project Git provenance: {error}"
            ) from error
        results[name] = completed.stdout.strip()
    if Path(results["top"]).resolve() != root:
        raise TrainerFactoryError("runtime project root is not the Git worktree top level")
    sha = results["sha"].lower()
    if len(sha) != 40 or any(character not in "0123456789abcdef" for character in sha):
        raise TrainerFactoryError(f"project Git returned an invalid SHA: {sha!r}")
    if results["status"]:
        raise TrainerFactoryError(
            "project checkout has staged, unstaged, or untracked files"
        )
    return sha


def run_training(
    config: Mapping[str, Any],
    *,
    factory_loader: Callable[[str], Any] = load_dotted_object,
) -> Any:
    """Build only the explicitly configured project trainer and invoke its ``fit`` method."""

    dotted_path = config.get("trainer_factory")
    if not isinstance(dotted_path, str) or not dotted_path.strip():
        raise TrainerFactoryError(
            "trainer_factory is required outside --validate-only; ordinary VeRL fallback "
            "is disabled"
        )
    verl_source_path = _runtime_path(config, "verl_source_path")
    # Registration imports verl.trainer.ppo.core_algos. Bind and verify the pinned checkout first
    # so an installed or stale VeRL can never populate sys.modules ahead of the server factory.
    bind_pinned_verl_import(verl_source_path)
    try:
        registered = register_capsule_critique_policy_loss()
    except Exception as error:
        raise TrainerFactoryError(
            f"Capsule-Critique policy loss registration failed: {type(error).__name__}: {error}"
        ) from error
    if registered is not True:
        raise TrainerFactoryError(
            "Capsule-Critique policy loss could not register because VeRL is unavailable"
        )
    factory = factory_loader(dotted_path)
    if not callable(factory):
        raise TrainerFactoryError(f"trainer_factory {dotted_path!r} is not callable")
    trainer = factory(config)
    fit = getattr(trainer, "fit", None)
    if not callable(fit):
        raise TrainerFactoryError("trainer_factory must return an object with a callable fit()")
    return fit()


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        return repr(value)
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capsule-Critique-GRPO project trainer")
    parser.add_argument("--config", required=True, help="Capsule YAML configuration path")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="perform static checks only; never import a trainer or start runtime services",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="alias for --validate-only; never import a trainer or start runtime services",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    factory_loader: Callable[[str], Any] = load_dotted_object,
) -> int:
    args = _parser().parse_args(argv)
    try:
        config, config_path = load_and_validate_config(args.config)
        runtime = config["runtime"]
        verl_source_path = _runtime_path(config, "verl_source_path")
        if args.validate_only or args.dry_run:
            runtime_paths = validate_local_execution_inputs(config)
            bundle_provenance = verify_bundle_provenance(config, config_path)
            project_git_sha = _project_git_sha(config)
            compatibility = check_verl_compatibility(
                verl_source_path, runtime["verl_pinned_sha"]
            )
            payload = {
                "mode": "validate-only",
                "config_path": str(config_path),
                "project_git_sha": project_git_sha,
                "project_git_clean": True,
                "compatibility": compatibility.to_dict(),
                "bundle_provenance": bundle_provenance,
                "runtime_paths": {
                    "verl_source_path": str(verl_source_path),
                    **{name: str(path) for name, path in runtime_paths.items()},
                },
                "services": {
                    "program_endpoint": config["program_service"]["endpoint"],
                    "controller_endpoint": config["controller_service"]["endpoint"],
                },
                "runtime_started": False,
            }
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
            return 0

        validate_local_execution_inputs(config)
        bundle_provenance_before = verify_bundle_provenance(config, config_path)
        check_verl_compatibility(verl_source_path, runtime["verl_pinned_sha"])
        project_git_sha_before = _project_git_sha(config)
        result = run_training(config, factory_loader=factory_loader)
        bundle_provenance_after = verify_bundle_provenance(config, config_path)
        if bundle_provenance_after != bundle_provenance_before:
            raise TrainerFactoryError(
                "formal training bundle provenance changed while training was executing"
            )
        project_git_sha_after = _project_git_sha(config)
        if project_git_sha_after != project_git_sha_before:
            raise TrainerFactoryError("project Git SHA changed while training was executing")
        print(
            json.dumps(
                {
                    "mode": "train",
                    "project_git_sha": project_git_sha_before,
                    "project_git_clean": True,
                    "bundle_provenance": bundle_provenance_before,
                    "fit_result": _json_safe(result),
                },
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            )
        )
        return 0
    except (
        CapsuleConfigError,
        ConfigLoadError,
        TrainerFactoryError,
        VeRLCompatibilityError,
    ) as error:
        error_code = getattr(error, "code", type(error).__name__)
        print(f"{error_code}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover - exercised through main() in pure tests
    raise SystemExit(main())


__all__ = [
    "ConfigLoadError",
    "TrainerFactoryError",
    "load_and_validate_config",
    "load_dotted_object",
    "main",
    "register_capsule_critique_policy_loss",
    "run_training",
    "validate_local_execution_inputs",
    "verify_bundle_provenance",
]
