"""Gate 4 frozen Controller smoke; preview with --validate-only or --dry-run."""

from __future__ import annotations

from .common import external_gate_main, verify_collector_gate_artifact


def main(argv: list[str] | None = None) -> int:
    return external_gate_main(
        gate_name="controller_collector",
        description="Validate or run the frozen Controller repair collector smoke.",
        placeholders={"p0_count": "2", "trajectories_per_p0": "2", "max_turns": "12"},
        required_placeholders=frozenset(
            {
                "config",
                "artifact",
                "run_id",
                "p0_count",
                "trajectories_per_p0",
                "max_turns",
            }
        ),
        verifier=verify_collector_gate_artifact,
        default_runner_command=(
            "'{python}' -m scripts.capsule_rl.server_adapter "
            "--config '{config}' --artifact '{artifact}' --run-id '{run_id}' "
            "collector --p0-count {p0_count} --trajectories {trajectories_per_p0} "
            "--max-turns {max_turns}"
        ),
        lock_runner_command=True,
        argv=argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
