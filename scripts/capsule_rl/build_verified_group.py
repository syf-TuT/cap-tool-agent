"""Gate 5 verified 7+1 group; preview with --validate-only or --dry-run."""

from __future__ import annotations

from .common import external_gate_main, verify_guided_gate_artifact


def main(argv: list[str] | None = None) -> int:
    return external_gate_main(
        gate_name="verified_guided_group",
        description="Validate or run construction of one verified 7+1 learning group.",
        placeholders={
            "group_size": "8",
            "base_count": "7",
            "guided_count": "1",
            "max_group_attempts": "20",
        },
        required_placeholders=frozenset(
            {
                "config",
                "artifact",
                "run_id",
                "group_size",
                "base_count",
                "guided_count",
                "max_group_attempts",
            }
        ),
        verifier=verify_guided_gate_artifact,
        default_runner_command=(
            "'{python}' -m scripts.capsule_rl.server_adapter "
            "--config '{config}' --artifact '{artifact}' --run-id '{run_id}' "
            "guided --group-size {group_size} --base-count {base_count} "
            "--guided-count {guided_count} --max-group-attempts {max_group_attempts}"
        ),
        lock_runner_command=True,
        argv=argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
