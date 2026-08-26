"""Static safety and compatibility checks for Capsule-Critique-GRPO.

This module intentionally never imports VeRL.  The server entrypoint can therefore validate an
initialized checkout (or report that its submodule is absent) without starting Ray, loading a
model, or mutating the source tree.
"""

from __future__ import annotations

import ast
import importlib
import ipaddress
import math
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .config_validation import validate_capsule_training_config
from .task_profiles import collect_task_profile_errors

# Official VeRL v0.6.1 tag.
PINNED_VERL_SHA = "d62da4950573d7a4b7ef2362337952e7ab59e78d"
CAPSULE_EXTERNAL_LIB = "capx.rl.capsule.verl_external"
CAPSULE_LOSS_MODE = "capsule_critique"
VERL_ROLLOUT_IS_SLOT = "rollout_is_weights"


class VeRLCompatibilityError(RuntimeError):
    """A typed, side-effect-free VeRL checkout validation failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


class CapsuleConfigError(ValueError):
    """Raised when a config could change the approved Capsule training semantics."""


def verify_imported_verl_path(source_path: str | Path) -> None:
    """Require every loaded VeRL module to come from one validated checkout."""

    root = Path(source_path).expanduser().resolve()
    expected_package = (root / "verl").resolve()
    loaded = {
        name: module
        for name, module in sys.modules.items()
        if name == "verl" or name.startswith("verl.")
    }
    if "verl" not in loaded:
        raise VeRLCompatibilityError("import_missing", "the pinned VeRL package was not imported")
    package_paths = getattr(loaded["verl"], "__path__", ())
    resolved_package_paths = {Path(value).resolve() for value in package_paths}
    if resolved_package_paths != {expected_package}:
        raise VeRLCompatibilityError(
            "import_path_mismatch",
            f"loaded VeRL package paths {sorted(map(str, resolved_package_paths))!r} do not "
            f"equal the pinned checkout {expected_package}",
        )
    for name, module in loaded.items():
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue
        resolved = Path(module_file).resolve()
        if not resolved.is_relative_to(expected_package):
            raise VeRLCompatibilityError(
                "import_path_mismatch",
                f"loaded VeRL module {name!r} comes from {resolved}, outside the pinned "
                f"checkout {expected_package}",
            )


def bind_pinned_verl_import(source_path: str | Path) -> Path:
    """Put the validated checkout first on sys.path, import VeRL, and fail on contamination."""

    root = Path(source_path).expanduser().resolve()
    expected_package = root / "verl"
    if not expected_package.is_dir():
        raise VeRLCompatibilityError(
            "uninitialized_source",
            f"pinned VeRL package directory is absent: {expected_package}",
        )
    configured = str(root)
    if not sys.path or Path(sys.path[0] or ".").resolve() != root:
        sys.path.insert(0, configured)
    importlib.invalidate_caches()
    if "verl" not in sys.modules:
        try:
            importlib.import_module("verl")
        except Exception as error:
            raise VeRLCompatibilityError(
                "import_error",
                f"cannot import pinned VeRL from {root}: {type(error).__name__}: {error}",
            ) from error
    verify_imported_verl_path(root)
    return root


@dataclass(frozen=True)
class VeRLCompatibilityReport:
    source_path: str
    expected_sha: str
    actual_sha: str
    compatible: bool
    rollout_is_slot: str
    checked_symbols: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_python(path: Path) -> ast.Module:
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeError) as error:
        raise VeRLCompatibilityError(
            "invalid_source", f"cannot parse required VeRL source {path}: {error}"
        ) from error


def _function_arguments(tree: ast.AST, name: str) -> tuple[str, ...] | None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return tuple(argument.arg for argument in (*node.args.posonlyargs, *node.args.args))
    return None


def _has_assignment(tree: ast.AST, name: str) -> bool:
    for node in ast.walk(tree):
        targets: tuple[ast.expr, ...]
        if isinstance(node, ast.Assign):
            targets = tuple(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = (node.target,)
        else:
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            return True
    return False


def _require_arguments(
    tree: ast.AST,
    function_name: str,
    expected: tuple[str, ...],
    *,
    source: Path,
) -> None:
    actual = _function_arguments(tree, function_name)
    if actual is None:
        raise VeRLCompatibilityError(
            "missing_symbol", f"{source} does not define {function_name}"
        )
    if actual != expected:
        raise VeRLCompatibilityError(
            "signature_mismatch",
            f"{function_name} signature is {actual!r}; expected {expected!r}",
        )


def _read_required_surface(source_path: Path) -> tuple[Path, Path, Path, tuple[Path, ...]]:
    core = source_path / "verl" / "trainer" / "ppo" / "core_algos.py"
    actor = source_path / "verl" / "workers" / "actor" / "dp_actor.py"
    role = source_path / "verl" / "workers" / "roles" / "actor.py"
    ref_candidates = (
        source_path / "verl" / "workers" / "fsdp_workers.py",
        source_path / "verl" / "workers" / "megatron_workers.py",
    )
    required = (core, actor, role)
    missing = tuple(path for path in required if not path.is_file())
    if missing:
        code = "uninitialized_source" if not (source_path / "verl").is_dir() else "missing_file"
        joined = ", ".join(str(path) for path in missing)
        raise VeRLCompatibilityError(code, f"required pinned VeRL source is absent: {joined}")
    if not any(path.is_file() for path in ref_candidates):
        raise VeRLCompatibilityError(
            "missing_file", "neither FSDP nor Megatron reference-worker source is present"
        )
    return core, actor, role, ref_candidates


def _git_head(source_path: Path) -> str:
    git_environment = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
    command = ["git", "-C", str(source_path), "rev-parse", "HEAD"]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
            env=git_environment,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise VeRLCompatibilityError("git_error", f"cannot read VeRL git SHA: {error}") from error
    if completed.returncode != 0:
        diagnostic = completed.stderr.strip() or "git rev-parse failed"
        raise VeRLCompatibilityError("git_error", diagnostic)
    actual_sha = completed.stdout.strip().lower()
    if len(actual_sha) != 40 or any(
        character not in "0123456789abcdef" for character in actual_sha
    ):
        raise VeRLCompatibilityError("git_error", f"invalid git SHA returned: {actual_sha!r}")
    try:
        top_level = subprocess.run(
            ["git", "-C", str(source_path), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
            env=git_environment,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise VeRLCompatibilityError(
            "git_error", f"cannot resolve VeRL worktree root: {error}"
        ) from error
    if top_level.returncode != 0 or Path(top_level.stdout.strip()).resolve() != source_path:
        raise VeRLCompatibilityError(
            "git_error", "VeRL source path must be the Git worktree top level"
        )
    status_command = [
        "git",
        "-C",
        str(source_path),
        "status",
        "--porcelain",
        "--untracked-files=all",
    ]
    try:
        status = subprocess.run(
            status_command,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
            env=git_environment,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise VeRLCompatibilityError(
            "git_error", f"cannot inspect VeRL worktree state: {error}"
        ) from error
    if status.returncode != 0:
        diagnostic = status.stderr.strip() or "git status failed"
        raise VeRLCompatibilityError("git_error", diagnostic)
    if status.stdout.strip():
        raise VeRLCompatibilityError(
            "dirty_source",
            "pinned VeRL checkout has staged, unstaged, or untracked files",
        )
    return actual_sha


def check_verl_compatibility(
    source_path: str | Path,
    expected_sha: str = PINNED_VERL_SHA,
) -> VeRLCompatibilityReport:
    """Check the pinned VeRL git identity and API surface without importing it."""

    root = Path(source_path).expanduser().resolve()
    if not root.is_dir():
        raise VeRLCompatibilityError(
            "uninitialized_source", f"VeRL source directory is not initialized: {root}"
        )
    core_path, actor_path, role_path, ref_candidates = _read_required_surface(root)
    actual_sha = _git_head(root)
    if actual_sha != expected_sha.lower():
        raise VeRLCompatibilityError(
            "sha_mismatch",
            f"VeRL HEAD {actual_sha} does not match pinned SHA {expected_sha}",
        )

    core_tree = _parse_python(core_path)
    if not _has_assignment(core_tree, "POLICY_LOSS_REGISTRY"):
        raise VeRLCompatibilityError(
            "missing_symbol", f"{core_path} does not define POLICY_LOSS_REGISTRY"
        )
    _require_arguments(core_tree, "register_policy_loss", ("name",), source=core_path)
    _require_arguments(
        core_tree,
        "compute_policy_loss_vanilla",
        (
            "old_log_prob",
            "log_prob",
            "advantages",
            "response_mask",
            "loss_agg_mode",
            "config",
            VERL_ROLLOUT_IS_SLOT,
        ),
        source=core_path,
    )

    actor_source = actor_path.read_text(encoding="utf-8")
    actor_tree = _parse_python(actor_path)
    _require_arguments(
        actor_tree,
        "compute_log_prob",
        ("self", "data", "calculate_entropy"),
        source=actor_path,
    )
    _require_arguments(actor_tree, "update_policy", ("self", "data"), source=actor_path)
    if VERL_ROLLOUT_IS_SLOT not in actor_source:
        raise VeRLCompatibilityError(
            "missing_rollout_is_slot",
            f"pinned actor no longer transports {VERL_ROLLOUT_IS_SLOT}",
        )

    role_tree = _parse_python(role_path)
    _require_arguments(role_tree, "compute_log_prob", ("self", "data"), source=role_path)
    _require_arguments(role_tree, "update_actor", ("self", "data"), source=role_path)

    ref_source = next(path for path in ref_candidates if path.is_file())
    _require_arguments(
        _parse_python(ref_source),
        "compute_ref_log_prob",
        ("self", "data"),
        source=ref_source,
    )

    checked_symbols = (
        "POLICY_LOSS_REGISTRY",
        "register_policy_loss",
        "compute_policy_loss_vanilla",
        "compute_log_prob",
        "compute_ref_log_prob",
        "update_actor",
        "update_policy",
    )
    return VeRLCompatibilityReport(
        source_path=str(root),
        expected_sha=expected_sha,
        actual_sha=actual_sha,
        compatible=True,
        rollout_is_slot=VERL_ROLLOUT_IS_SLOT,
        checked_symbols=checked_symbols,
    )


_MISSING = object()
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _get(config: Mapping[str, Any], path: str) -> Any:
    current: Any = config
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _require_exact(
    config: Mapping[str, Any],
    path: str,
    expected: Any,
    reason: str,
    errors: list[str],
) -> None:
    actual = _get(config, path)
    type_mismatch = (
        isinstance(expected, bool)
        and not isinstance(actual, bool)
        or isinstance(expected, int)
        and not isinstance(expected, bool)
        and (isinstance(actual, bool) or not isinstance(actual, int))
    )
    if actual != expected or type_mismatch:
        errors.append(f"{path} must be {expected!r} ({reason}); got {actual!r}")


def _require_nonempty(config: Mapping[str, Any], path: str, errors: list[str]) -> None:
    value = _get(config, path)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{path} must be a non-empty string")


def validate_capsule_config(config: Mapping[str, Any]) -> None:
    """Validate every local invariant that makes the 7+1 objective unambiguous."""

    if not isinstance(config, Mapping):
        raise CapsuleConfigError("Capsule config must be a mapping")
    errors: list[str] = []
    exact_values = (
        ("schema_version", 1, "schema v1 is the only supported artifact contract"),
        ("runtime.verl_pinned_sha", PINNED_VERL_SHA, "the adapter targets one pinned VeRL"),
        (
            "controller_service.frozen",
            True,
            "only the Program actor is updated; Controller is frozen",
        ),
        (
            "program_service.mode",
            "actor_identity",
            "Program HTTP service exposes actor identity only; VeRL owns generation",
        ),
        ("capsule.group_size", 8, "Capsule-Critique uses one eight-member group"),
        ("capsule.base_samples_before_repair", 7, "repair triggers only after seven failures"),
        ("capsule.p0_count", 2, "repair ranks exactly two P0 programs"),
        ("capsule.repair_trajectories_per_p0", 2, "each P0 receives two trajectories"),
        ("capsule.max_controller_turns", 12, "each repair trajectory has a 12-turn cap"),
        ("capsule.revision_input_max_tokens", 8192, "revision prompts are never truncated"),
        ("capsule.revision_response_max_tokens", 2048, "revision responses are never truncated"),
        ("capsule.gamma", 0.1, "guided shaping uses gamma=0.1"),
        ("actor_rollout_ref.rollout.n", 8, "rollout n=8 must match the learning group"),
        (
            "actor_rollout_ref.rollout.calculate_log_probs",
            False,
            "old log-probs are recomputed after guided injection",
        ),
        ("actor_rollout_ref.rollout.mode", "sync", "Capsule uses synchronous rollout workers"),
        (
            "actor_rollout_ref.model.external_lib",
            CAPSULE_EXTERNAL_LIB,
            "the project-owned loss must be registered",
        ),
        (
            "actor_rollout_ref.actor.policy_loss.loss_mode",
            CAPSULE_LOSS_MODE,
            "guided tokens require Capsule-Critique loss",
        ),
        ("actor_rollout_ref.actor.policy_loss.capsule_gamma", 0.1, "guided gamma is fixed"),
        ("actor_rollout_ref.actor.use_kl_loss", True, "reference KL stays in the actor loss"),
        ("actor_rollout_ref.actor.ppo_epochs", 1, "one update_actor call is one PPO epoch"),
        (
            "actor_rollout_ref.actor.ulysses_sequence_parallel_size",
            1,
            "MVP optimizer-step accounting assumes sequence parallelism is disabled",
        ),
        (
            "actor_rollout_ref.actor.ppo_mini_batch_size",
            8,
            "the complete 7+1 group is one optimizer mini-batch",
        ),
        ("algorithm.adv_estimator", "grpo", "group-mean GRPO advantage is required"),
        (
            "algorithm.norm_adv_by_std_in_grpo",
            False,
            "advantages use group mean without std normalization",
        ),
        ("algorithm.rollout_is", False, "standard rollout importance sampling is disabled"),
        (
            "algorithm.rollout_is_threshold",
            None,
            "standard rollout importance sampling threshold is disabled",
        ),
        ("algorithm.use_kl_in_reward", False, "reference KL must not alter binary rewards"),
        ("reward_model.enable", False, "typed clean-replay rewards bypass VeRL reward models"),
        (
            "reward_model.reward_manager",
            "typed_replay",
            "Prime/current reward managers are forbidden",
        ),
        (
            "reward_model.launch_reward_fn_async",
            False,
            "async reward execution is forbidden",
        ),
    )
    for path, expected, reason in exact_values:
        _require_exact(config, path, expected, reason, errors)
    errors.extend(collect_task_profile_errors(config))

    for path in (
        "runtime.verl_source_path",
        "runtime.output_dir",
        "runtime.dataset_path",
        "runtime.program_model_path",
        "runtime.verl_resolved_config_path",
        "task.config_path",
        "program_service.endpoint",
        "program_service.model",
        "program_service.api_key_env",
        "controller_service.endpoint",
        "controller_service.model",
        "controller_service.api_key_env",
    ):
        _require_nonempty(config, path, errors)

    program_endpoint = _get(config, "program_service.endpoint")
    controller_endpoint = _get(config, "controller_service.endpoint")
    if program_endpoint == controller_endpoint and program_endpoint is not _MISSING:
        errors.append("program_service and controller_service must use separate endpoints")
    for path, value in (
        ("program_service.endpoint", program_endpoint),
        ("controller_service.endpoint", controller_endpoint),
    ):
        if isinstance(value, str):
            parsed = urlparse(value)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                errors.append(f"{path} must be an absolute HTTP(S) URL")
            elif parsed.scheme == "http":
                try:
                    hostname = parsed.hostname
                except ValueError:
                    hostname = None
                loopback = False
                if hostname:
                    try:
                        loopback = ipaddress.ip_address(hostname).is_loopback
                    except ValueError:
                        loopback = hostname.lower() == "localhost"
                if not loopback:
                    errors.append(
                        f"{path} must use https unless it targets a loopback address"
                    )
    for path in ("program_service.api_key_env", "controller_service.api_key_env"):
        value = _get(config, path)
        if isinstance(value, str) and not _ENV_NAME.fullmatch(value):
            errors.append(f"{path} must name an environment variable, not contain a credential")
    timeout = _get(config, "controller_service.request_timeout_s")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(float(timeout))
        or timeout <= 0
    ):
        errors.append("controller_service.request_timeout_s must be a positive finite number")
    max_output_tokens = _get(config, "controller_service.max_output_tokens")
    if (
        max_output_tokens is _MISSING
        or isinstance(max_output_tokens, bool)
        or not isinstance(max_output_tokens, int)
        or max_output_tokens < 1
    ):
        errors.append("controller_service.max_output_tokens must be a positive integer")
    for field_name in ("stream", "enable_thinking"):
        value = _get(config, f"controller_service.{field_name}")
        if value is not False:
            errors.append(f"controller_service.{field_name} must be explicitly false")
    temperature = _get(config, "controller_service.temperature")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
        or temperature < 0
        or temperature > 2
    ):
        errors.append("controller_service.temperature must be between zero and two")
    kl_loss_coef = _get(config, "actor_rollout_ref.actor.kl_loss_coef")
    if (
        isinstance(kl_loss_coef, bool)
        or not isinstance(kl_loss_coef, (int, float))
        or not math.isfinite(float(kl_loss_coef))
        or kl_loss_coef <= 0
    ):
        errors.append("actor_rollout_ref.actor.kl_loss_coef must be positive and finite")

    gates = _get(config, "server_validation.gates")
    expected_gates = (
        "preflight",
        "seed",
        "oracle_replay",
        "collector",
        "guided",
        "trainer",
        "result_audit",
    )
    if (
        not isinstance(gates, Sequence)
        or isinstance(gates, (str, bytes))
        or tuple(gates) != expected_gates
    ):
        errors.append(f"server_validation.gates must be ordered as {expected_gates!r}")

    try:
        validate_capsule_training_config(config)
    except (KeyError, TypeError, ValueError) as error:
        message = str(error)
        if not any(message in existing for existing in errors):
            errors.append(message)

    if errors:
        raise CapsuleConfigError("Invalid Capsule-Critique-GRPO config:\n- " + "\n- ".join(errors))


__all__ = [
    "CAPSULE_EXTERNAL_LIB",
    "CAPSULE_LOSS_MODE",
    "PINNED_VERL_SHA",
    "VERL_ROLLOUT_IS_SLOT",
    "CapsuleConfigError",
    "VeRLCompatibilityError",
    "VeRLCompatibilityReport",
    "bind_pinned_verl_import",
    "check_verl_compatibility",
    "validate_capsule_config",
    "verify_imported_verl_path",
]
