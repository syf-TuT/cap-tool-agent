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
_PYROKI_SERVER_FIELDS = MappingProxyType(
    {
        "host": "127.0.0.1",
        "port": 8116,
        "robot": "panda_description",
        "target_link": "panda_hand",
    }
)
_ENV_CONFIG_KEYS = frozenset(
    {
        "_target_",
        "low_level",
        "privileged",
        "enable_render",
        "viser_debug",
        "apis",
    }
)


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
    if isinstance(expected, int):
        return type(actual) is int and actual == expected
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


def collect_environment_profile_errors(
    payload: Mapping[str, Any], profile: CapsuleTaskProfile
) -> tuple[str, ...]:
    """Return environment-YAML mismatches for ``profile`` without file I/O."""

    if not isinstance(payload, Mapping):
        return ("environment config root must be a mapping",)
    if not isinstance(profile, CapsuleTaskProfile):
        raise TypeError("profile must be a CapsuleTaskProfile")

    errors: list[str] = []
    env = payload.get("env", _MISSING)
    env_cfg = env.get("cfg", _MISSING) if isinstance(env, Mapping) else _MISSING
    if not isinstance(env_cfg, Mapping):
        errors.append(
            f"env.cfg must be a mapping with exact keys {sorted(_ENV_CONFIG_KEYS)!r}; "
            f"got {_display(env_cfg)}"
        )
    elif set(env_cfg) != _ENV_CONFIG_KEYS:
        errors.append(
            f"env.cfg must contain exact keys {sorted(_ENV_CONFIG_KEYS)!r}; "
            f"got {sorted(str(key) for key in env_cfg)!r}"
        )
    for path, expected in (
        (("env", "_target_"), profile.env_target),
        (("env", "cfg", "_target_"), profile.env_config_target),
        (("env", "cfg", "low_level"), profile.low_level),
        (("env", "cfg", "privileged"), True),
        (("env", "cfg", "enable_render"), False),
        (("env", "cfg", "viser_debug"), False),
        (("env", "cfg", "apis"), list(profile.api_classes)),
        (("record_video",), False),
        (("num_workers",), 1),
    ):
        _collect_exact_error(
            payload,
            path=path,
            expected=expected,
            profile=profile,
            errors=errors,
        )

    servers = payload.get("api_servers", _MISSING)
    required_count = len(profile.required_server_targets)
    if not isinstance(servers, list):
        errors.append(
            f"api_servers must be a list with exactly {required_count} server(s) for task "
            f"profile {profile.name!r}; got {_display(servers)}"
        )
        return tuple(errors)
    if len(servers) != required_count:
        errors.append(
            f"api_servers must contain exactly {required_count} server(s) for task profile "
            f"{profile.name!r}; got {len(servers)}"
        )

    expected_server_keys = {"_target_", *_PYROKI_SERVER_FIELDS}
    for index, expected_target in enumerate(profile.required_server_targets):
        if index >= len(servers):
            break
        server = servers[index]
        path_prefix = f"api_servers[{index}]"
        if not isinstance(server, Mapping):
            errors.append(f"{path_prefix} must be a mapping; got {server!r}")
            continue
        if set(server) != expected_server_keys:
            errors.append(
                f"{path_prefix} must contain exact keys {sorted(expected_server_keys)!r}; "
                f"got {sorted(str(key) for key in server)!r}"
            )
        for field_name, expected in (
            ("_target_", expected_target),
            *_PYROKI_SERVER_FIELDS.items(),
        ):
            actual = server.get(field_name, _MISSING)
            if not _matches_exact(actual, expected):
                errors.append(
                    f"{path_prefix}.{field_name} must be {expected!r} for task profile "
                    f"{profile.name!r}; got {_display(actual)}"
                )

    return tuple(errors)


__all__ = [
    "CAPSULE_TASK_PROFILES",
    "ROBOSUITE_CUBE_LIFT_PRIVILEGED_HIGHLEVEL",
    "ROBOSUITE_CUBE_STACK_PRIVILEGED",
    "CapsuleTaskProfile",
    "CapsuleTaskProfileError",
    "collect_environment_profile_errors",
    "collect_task_profile_errors",
    "resolve_task_profile",
]
