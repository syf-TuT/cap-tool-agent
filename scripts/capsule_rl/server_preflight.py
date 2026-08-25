"""Gate 1 preflight; --validate-only/--dry-run performs no probes or writes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from capx.rl.capsule.actor_identity import (
    ActorIdentityError,
    build_actor_identity,
    verify_actor_identity_payload,
)
from capx.rl.capsule.schema import TaskInstanceV1

from .common import (
    CANONICAL_EXECUTION_MODE,
    GateArtifactError,
    add_validation_arguments,
    artifact_file_sha256,
    atomic_write_json,
    gate_failure_artifact_path,
    load_and_validate_server_config,
    load_and_validate_server_config_bytes,
    runtime_dataset_path,
    runtime_dependency_hashes,
    validation_requested,
    verify_preflight_gate_artifact,
    write_gate_failure_artifact,
)


def _endpoint_open(endpoint: str, timeout_s: float = 2.0) -> bool:
    parsed = urlparse(endpoint)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((parsed.hostname or "", port), timeout=timeout_s):
            return True
    except OSError:
        return False


class _RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, *_args: object, **_kwargs: object) -> None:
        return None


def _fetch_actor_identity(
    endpoint: str,
    bearer_token: str,
    *,
    opener: Any | None = None,
    timeout_s: float = 5.0,
    max_response_bytes: int = 65_536,
) -> dict[str, Any]:
    """Fetch one bounded, authenticated identity document without following redirects."""

    if not isinstance(bearer_token, str) or not bearer_token:
        raise GateArtifactError("Program actor identity bearer token is missing")
    identity_url = endpoint.rstrip("/") + "/capx/actor-identity"
    request = Request(
        identity_url,
        method="GET",
        headers={"Authorization": f"Bearer {bearer_token}", "Accept": "application/json"},
    )
    client = opener if opener is not None else build_opener(_RejectRedirects())
    try:
        with client.open(request, timeout=timeout_s) as response:
            if getattr(response, "status", None) != 200:
                raise GateArtifactError(
                    f"Program actor identity returned HTTP {getattr(response, 'status', None)!r}"
                )
            if response.geturl() != identity_url:
                raise GateArtifactError("Program actor identity redirect is forbidden")
            raw_length = response.headers.get("Content-Length")
            if raw_length is not None:
                try:
                    content_length = int(raw_length)
                except (TypeError, ValueError) as error:
                    raise GateArtifactError(
                        "Program actor identity Content-Length is invalid"
                    ) from error
                if content_length < 0 or content_length > max_response_bytes:
                    raise GateArtifactError("Program actor identity response is too large")
            body = response.read(max_response_bytes + 1)
    except GateArtifactError:
        raise
    except HTTPError as error:
        if 300 <= error.code < 400:
            raise GateArtifactError("Program actor identity redirect is forbidden") from error
        raise GateArtifactError(
            f"Program actor identity returned HTTP {error.code}"
        ) from error
    except (OSError, URLError) as error:
        raise GateArtifactError(f"Program actor identity request failed: {error}") from error
    if len(body) > max_response_bytes:
        raise GateArtifactError("Program actor identity response is too large")
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GateArtifactError("Program actor identity response must be UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise GateArtifactError("Program actor identity response must be a JSON object")
    return payload


def _git_sha(path: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    sha = completed.stdout.strip()
    top_level = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    if Path(top_level.stdout.strip()).resolve() != path.resolve():
        raise GateArtifactError(f"Git path is not a worktree top level: {path}")
    status = subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise GateArtifactError(
            f"Git checkout has staged, unstaged, or untracked files: {path}"
        )
    return sha


def _resolved_environment_hash(config: dict[str, Any], project_root: Path) -> str:
    del project_root
    return runtime_dependency_hashes(config)["resolved_environment_sha256"]


def _git_roots(config: Mapping[str, Any]) -> tuple[Path, Path]:
    runtime = config["runtime"]
    repository_root = Path(__file__).resolve().parents[2]
    configured_root = runtime.get("project_root")
    if isinstance(configured_root, str) and configured_root:
        root_path = Path(configured_root).expanduser()
        project_root = (
            root_path.resolve()
            if root_path.is_absolute()
            else (repository_root / root_path).resolve()
        )
    else:
        project_root = repository_root
    verl_path = Path(runtime["verl_source_path"]).expanduser()
    if not verl_path.is_absolute():
        verl_path = project_root / verl_path
    return project_root.resolve(), verl_path.resolve()


def _dataset_rows(dataset_path: Path) -> list[Mapping[str, Any]]:
    suffix = dataset_path.suffix.lower()
    rows: list[Mapping[str, Any]] = []
    if suffix in {".json", ".jsonl"}:
        for line_number, line in enumerate(
            dataset_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise GateArtifactError(
                    f"runtime.dataset_path line {line_number} is invalid JSON: {error}"
                ) from error
            if not isinstance(row, Mapping):
                raise GateArtifactError(
                    f"runtime.dataset_path line {line_number} must be a JSON object"
                )
            rows.append(row)
    elif suffix == ".parquet":
        try:
            import pandas as pd
        except ModuleNotFoundError as error:
            raise GateArtifactError(
                "reading a Parquet task dataset requires pandas"
            ) from error
        rows = [dict(row) for row in pd.read_parquet(dataset_path).to_dict("records")]
    else:
        raise GateArtifactError(
            "runtime.dataset_path must be JSONL, JSON, or Parquet"
        )
    if not rows:
        raise GateArtifactError("runtime.dataset_path contains no task rows")
    return rows


def _dataset_task_identity_summary(
    dataset_path: Path, config: Mapping[str, Any]
) -> list[dict[str, object]]:
    task_config = config["task"]
    identities: list[dict[str, object]] = []
    observed: set[tuple[str, int]] = set()
    for row_index, row in enumerate(_dataset_rows(dataset_path)):
        data = dict(row)
        data.setdefault("schema_version", 1)
        for field_name in ("environment", "api", "privilege"):
            data.setdefault(field_name, str(task_config[field_name]))
        # Prepared smoke-gate datasets intentionally omit the real reset hash.  The
        # placeholder validates the remaining TaskInstanceV1 fields without claiming state.
        data.setdefault("initial_state_sha256", "0" * 64)
        try:
            task = TaskInstanceV1.from_dict(data)
        except (KeyError, TypeError, ValueError) as error:
            raise GateArtifactError(
                f"runtime.dataset_path row {row_index} is not a typed TaskInstanceV1: {error}"
            ) from error
        if task.environment_seed < 0:
            raise GateArtifactError(
                f"runtime.dataset_path row {row_index} environment_seed must be non-negative"
            )
        for field_name in ("environment", "api", "privilege"):
            if getattr(task, field_name) != str(task_config[field_name]):
                raise GateArtifactError(
                    f"runtime.dataset_path row {row_index} {field_name} does not match task config"
                )
        identity = (task.task_id, task.environment_seed)
        if identity in observed:
            raise GateArtifactError(
                f"runtime.dataset_path contains duplicate task identity {identity!r}"
            )
        observed.add(identity)
        identities.append(
            {"task_id": task.task_id, "environment_seed": task.environment_seed}
        )
    return identities


def _collect_preflight_payload(
    config: dict[str, Any],
    *,
    run_id: str,
    config_sha256: str,
    failure_context: dict[str, str | None],
) -> tuple[dict[str, Any], list[str]]:
    """Collect preflight facts without publishing either success or failure evidence."""

    runtime = config["runtime"]
    project_root, verl_path = _git_roots(config)
    program_model_path = Path(runtime["program_model_path"]).expanduser()
    if not program_model_path.is_absolute():
        program_model_path = project_root / program_model_path
    dataset_path = runtime_dataset_path(config)
    dataset_sha256 = artifact_file_sha256(dataset_path)
    failure_context["dataset_sha256"] = dataset_sha256
    dependency_hashes = runtime_dependency_hashes(config)
    dataset_task_identities = _dataset_task_identity_summary(dataset_path, config)
    program_service = config["program_service"]
    controller_service = config["controller_service"]
    pyroki_endpoint = config.get("server_validation", {}).get(
        "pyroki_endpoint", "http://127.0.0.1:8116"
    )
    import torch

    dependency_lock = project_root / "uv.lock"

    project_git_sha = _git_sha(project_root)
    failure_context["git_sha"] = project_git_sha
    verl_actual_sha = _git_sha(verl_path)
    try:
        local_actor_identity = build_actor_identity(config)
        program_token = os.environ.get(str(program_service["api_key_env"]), "")
        service_actor_identity = _fetch_actor_identity(
            str(program_service["endpoint"]), program_token
        )
        verify_actor_identity_payload(service_actor_identity, local_actor_identity)
    except ActorIdentityError as error:
        raise GateArtifactError(f"Program actor identity verification failed: {error}") from error
    checks = {
        "git_sha": project_git_sha,
        "verl_source_path": str(verl_path.resolve()),
        "verl_expected_sha": runtime["verl_pinned_sha"],
        "verl_actual_sha": verl_actual_sha,
        "verl_sha_matches": verl_actual_sha == runtime["verl_pinned_sha"],
        "dependency_lock_present": dependency_lock.is_file(),
        "dependency_lock_sha256": (
            artifact_file_sha256(dependency_lock) if dependency_lock.is_file() else None
        ),
        "cuda_available": bool(torch.cuda.is_available()),
        "egl_configured": os.environ.get("MUJOCO_GL", "").lower() == "egl",
        "program_model_exists": program_model_path.exists(),
        "program_model_file_count": local_actor_identity["program_model_file_count"],
        "program_model_sha256": local_actor_identity["program_model_sha256"],
        "actor_binding_sha256": local_actor_identity["actor_binding_sha256"],
        "program_actor_identity": service_actor_identity,
        "program_actor_identity_verified": True,
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_sha256,
        "dataset_task_count": len(dataset_task_identities),
        "dataset_task_identities": dataset_task_identities,
        "verl_resolved_config_sha256": dependency_hashes[
            "verl_resolved_config_sha256"
        ],
        "program_api_key_present": bool(os.environ.get(program_service["api_key_env"])),
        "controller_api_key_present": bool(os.environ.get(controller_service["api_key_env"])),
        "program_endpoint_ready": True,
        "controller_endpoint_ready": _endpoint_open(controller_service["endpoint"]),
        "pyroki_endpoint_ready": _endpoint_open(pyroki_endpoint),
        "resolved_environment_sha256": dependency_hashes[
            "resolved_environment_sha256"
        ],
    }
    if artifact_file_sha256(dataset_path) != dataset_sha256:
        raise GateArtifactError("runtime.dataset_path changed during preflight")
    if runtime_dependency_hashes(config) != dependency_hashes:
        raise GateArtifactError("runtime dependency bytes changed during preflight")
    required_boolean_checks = [
        key for key, value in checks.items() if isinstance(value, bool) and not value
    ]
    payload = {
        "schema_version": 1,
        "gate": "preflight",
        "passed": True,
        "execution_mode": CANONICAL_EXECUTION_MODE,
        "run_id": run_id,
        "config_sha256": config_sha256,
        "git_sha": project_git_sha,
        "dataset_sha256": dataset_sha256,
        "program_model_sha256": local_actor_identity["program_model_sha256"],
        "actor_binding_sha256": local_actor_identity["actor_binding_sha256"],
        **dependency_hashes,
        "failed_checks": [],
        "checks": checks,
    }
    return payload, required_boolean_checks


def run_preflight(config_path: Path, artifact_path: Path, *, run_id: str) -> dict[str, Any]:
    """Collect, verify, and exclusively publish canonical Gate 1 success evidence.

    A failed preflight never occupies the success path.  Instead it publishes one immutable
    ``<artifact>.failure.json`` containing the stage at which collection or verification failed.
    """

    config_file = config_path.expanduser().resolve()
    artifact_file = artifact_path.expanduser().resolve()
    failure_file = gate_failure_artifact_path(artifact_file)
    if artifact_file.exists() or artifact_file.is_symlink():
        raise FileExistsError(f"artifact already exists: {artifact_file}")
    if failure_file.exists() or failure_file.is_symlink():
        raise FileExistsError(f"failure artifact already exists: {failure_file}")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be non-empty")

    config_sha256: str | None = None
    git_sha: str | None = None
    dataset_sha256: str | None = None
    failure_context: dict[str, str | None] = {
        "git_sha": None,
        "dataset_sha256": None,
    }
    stage = "config_hash"
    try:
        config_bytes = config_file.read_bytes()
        config_sha256 = hashlib.sha256(config_bytes).hexdigest()
        stage = "config_load"
        config = load_and_validate_server_config_bytes(
            config_bytes,
            check_runtime_paths=True,
        )
        stage = "runtime_checks"
        payload, required_boolean_checks = _collect_preflight_payload(
            config,
            run_id=run_id,
            config_sha256=config_sha256,
            failure_context=failure_context,
        )
        git_sha = failure_context["git_sha"]
        dataset_sha256 = failure_context["dataset_sha256"]
        stage = "required_checks"
        if required_boolean_checks:
            raise GateArtifactError(
                "preflight failed: " + ", ".join(required_boolean_checks)
            )
        stage = "artifact_verification"
        verify_preflight_gate_artifact(payload)
        project_root, verl_path = _git_roots(config)
        stage = "post_config"
        if artifact_file_sha256(config_file) != config_sha256:
            raise GateArtifactError("server config bytes changed during preflight")
        stage = "post_project_git"
        if _git_sha(project_root) != payload["git_sha"]:
            raise GateArtifactError("project Git SHA changed during preflight")
        stage = "post_verl_git"
        if _git_sha(verl_path) != payload["checks"]["verl_actual_sha"]:
            raise GateArtifactError("VeRL Git SHA changed during preflight")
        stage = "post_dataset"
        if artifact_file_sha256(runtime_dataset_path(config)) != payload["dataset_sha256"]:
            raise GateArtifactError("runtime dataset bytes changed during preflight")
        stage = "post_runtime_dependencies"
        post_dependency_hashes = runtime_dependency_hashes(config)
        for field_name in (
            "resolved_environment_sha256",
            "verl_resolved_config_sha256",
        ):
            if post_dependency_hashes[field_name] != payload[field_name]:
                label = (
                    "resolved environment"
                    if field_name == "resolved_environment_sha256"
                    else "resolved VeRL config"
                )
                raise GateArtifactError(f"{label} bytes changed during preflight")
        stage = "post_actor_identity"
        try:
            post_local_identity = build_actor_identity(config)
            post_service_identity = _fetch_actor_identity(
                str(config["program_service"]["endpoint"]),
                os.environ.get(str(config["program_service"]["api_key_env"]), ""),
            )
            verify_actor_identity_payload(
                post_local_identity,
                payload["checks"]["program_actor_identity"],
            )
            verify_actor_identity_payload(post_service_identity, post_local_identity)
        except ActorIdentityError as error:
            raise GateArtifactError(
                f"Program actor identity changed during preflight: {error}"
            ) from error
        stage = "artifact_publish"
        if failure_file.exists() or failure_file.is_symlink():
            raise FileExistsError(
                f"failure artifact appeared before success publication: {failure_file}"
            )
        atomic_write_json(artifact_file, payload)
        return payload
    except BaseException as error:
        git_sha = failure_context["git_sha"]
        dataset_sha256 = failure_context["dataset_sha256"]
        if not artifact_file.exists() and not artifact_file.is_symlink():
            try:
                if not failure_file.exists() and not failure_file.is_symlink():
                    write_gate_failure_artifact(
                        artifact_file,
                        gate="preflight",
                        run_id=run_id,
                        config_sha256=config_sha256,
                        git_sha=git_sha,
                        dataset_sha256=dataset_sha256,
                        error=error,
                        stage=stage,
                    )
            except BaseException as recording_error:
                try:
                    setattr(error, "failure_artifact_recording_error", recording_error)
                except BaseException:
                    pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate or execute Capsule-RL server preflight.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    add_validation_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.run_id.strip():
        raise ValueError("run_id must be non-empty")
    validate_only = validation_requested(args)
    config = (
        load_and_validate_server_config(args.config, check_runtime_paths=True)
        if validate_only
        else None
    )
    plan = {
        "mode": "VALIDATION ONLY" if validate_only else "EXECUTE",
        "checks": [
            "project and VeRL Git SHA",
            "CUDA and MUJOCO_GL=egl",
            "Program model path and service",
            "frozen Controller credentials and service",
            "PyRoKi readiness",
            "resolved environment and VeRL config SHA-256",
            "runtime dataset bytes SHA-256 and typed task identities",
        ],
        "artifact": str(args.artifact.resolve()),
        "run_id": args.run_id,
        "controller_model": (
            config["controller_service"]["model"]
            if config is not None
            else "resolved during canonical preflight"
        ),
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if not validate_only:
        run_preflight(args.config.resolve(), args.artifact.resolve(), run_id=args.run_id)
        print(f"preflight: PASS ({args.artifact.resolve()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
