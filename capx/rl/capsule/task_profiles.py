"""Pure task-profile contracts for Capsule configuration validation.

The registry is intentionally import-safe: it describes environment and service identity without
loading YAML, Hydra, Robosuite, or simulator modules.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class CapsuleTaskProfile:
    """Static identity and runtime requirements for one supported Capsule task."""

    name: str
    environment: str
    api: str
    privilege: str
    render: bool
    record_video: bool
    env_target: str
    env_config_target: str
    low_level: str
    api_classes: tuple[str, ...]
    required_runtime: tuple[tuple[str, bool], ...]
    required_server_targets: tuple[str, ...]


ROBOSUITE_CUBE_STACK_PRIVILEGED = CapsuleTaskProfile(
    name="robosuite_cube_stack_privileged",
    environment="robosuite_cube_stack",
    api="franka_control_privileged",
    privilege="privileged",
    render=False,
    record_video=False,
    env_target="capx.envs.tasks.franka.franka_pick_place.FrankaPickPlaceCodeEnv",
    env_config_target="capx.envs.tasks.base.CodeExecEnvConfig",
    low_level="franka_robosuite_cubes_low_level",
    api_classes=("FrankaControlPrivilegedApi",),
    required_runtime=(("egl", True), ("pyroki", True)),
    required_server_targets=("capx.serving.launch_pyroki_server.main",),
)

ROBOSUITE_CUBE_LIFT_PRIVILEGED_HIGHLEVEL = CapsuleTaskProfile(
    name="robosuite_cube_lift_privileged_highlevel",
    environment="robosuite_cube_lift",
    api="franka_control_privileged",
    privilege="privileged",
    render=False,
    record_video=False,
    env_target="capx.envs.tasks.franka.franka_lift.FrankaLiftCodeEnv",
    env_config_target="capx.envs.tasks.base.CodeExecEnvConfig",
    low_level="franka_robosuite_cube_lift_low_level",
    api_classes=("FrankaControlPrivilegedApi",),
    required_runtime=(("egl", True), ("pyroki", True)),
    required_server_targets=("capx.serving.launch_pyroki_server.main",),
)

CAPSULE_TASK_PROFILES: Mapping[str, CapsuleTaskProfile] = MappingProxyType(
    {
        profile.name: profile
        for profile in (
            ROBOSUITE_CUBE_STACK_PRIVILEGED,
            ROBOSUITE_CUBE_LIFT_PRIVILEGED_HIGHLEVEL,
        )
    }
)

_LEGACY_CUBE_STACK_IDENTITY = (
    ROBOSUITE_CUBE_STACK_PRIVILEGED.environment,
    ROBOSUITE_CUBE_STACK_PRIVILEGED.api,
    ROBOSUITE_CUBE_STACK_PRIVILEGED.privilege,
)
_MISSING = object()


class CapsuleTaskProfileError(ValueError):
    """Raised when a Capsule task profile cannot be selected unambiguously."""


def _task_mapping(config: Mapping[str, Any]) -> Mapping[str, Any]:
    task = config.get("task", _MISSING)
    if not isinstance(task, Mapping):
        raise CapsuleTaskProfileError("task must be a mapping before task.profile is resolved")
    return task


def resolve_task_profile(config: Mapping[str, Any]) -> CapsuleTaskProfile:
    """Resolve an explicit profile or the sole supported legacy unprofiled identity."""

    if not isinstance(config, Mapping):
        raise CapsuleTaskProfileError("Capsule config must be a mapping")
    task = _task_mapping(config)
    profile_name = task.get("profile", _MISSING)
    if profile_name is _MISSING:
        identity = tuple(task.get(field, _MISSING) for field in ("environment", "api", "privilege"))
        if identity == _LEGACY_CUBE_STACK_IDENTITY:
            return ROBOSUITE_CUBE_STACK_PRIVILEGED
        raise CapsuleTaskProfileError(
            "task.profile must explicitly select a supported profile; only the exact legacy "
            "Cube Stack environment/API/privilege tuple may omit it"
        )
    if not isinstance(profile_name, str) or not profile_name.strip():
        raise CapsuleTaskProfileError("task.profile must be a non-empty string")
    profile = CAPSULE_TASK_PROFILES.get(profile_name)
    if profile is None:
        raise CapsuleTaskProfileError(f"task.profile selects unknown profile {profile_name!r}")
    return profile


def _matches_exact(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    return actual == expected


def _display(value: Any) -> str:
    return "<missing>" if value is _MISSING else repr(value)


def _collect_exact_error(
    config: Mapping[str, Any],
    *,
    path: tuple[str, ...],
    expected: Any,
    profile: CapsuleTaskProfile,
    errors: list[str],
) -> None:
    current: Any = config
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            current = _MISSING
            break
        current = current[part]
    if not _matches_exact(current, expected):
        dotted_path = ".".join(path)
        errors.append(
            f"{dotted_path} must be {expected!r} for task profile {profile.name!r}; "
            f"got {_display(current)}"
        )


def collect_task_profile_errors(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Return every selected-profile mismatch without raising on individual fields."""

    try:
        profile = resolve_task_profile(config)
    except CapsuleTaskProfileError as error:
        return (str(error),)

    errors: list[str] = []
    for path, expected in (
        (("task", "environment"), profile.environment),
        (("task", "api"), profile.api),
        (("task", "privilege"), profile.privilege),
        (("task", "render"), profile.render),
        (("task", "record_video"), profile.record_video),
    ):
        _collect_exact_error(
            config,
            path=path,
            expected=expected,
            profile=profile,
            errors=errors,
        )
    for requirement, expected in profile.required_runtime:
        _collect_exact_error(
            config,
            path=("runtime", "requires", requirement),
            expected=expected,
            profile=profile,
            errors=errors,
        )
    return tuple(errors)


__all__ = [
    "CAPSULE_TASK_PROFILES",
    "ROBOSUITE_CUBE_LIFT_PRIVILEGED_HIGHLEVEL",
    "ROBOSUITE_CUBE_STACK_PRIVILEGED",
    "CapsuleTaskProfile",
    "CapsuleTaskProfileError",
    "collect_task_profile_errors",
    "resolve_task_profile",
]
