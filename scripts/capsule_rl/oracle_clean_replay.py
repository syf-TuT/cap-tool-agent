"""Gate 3 oracle direct replay; preview with --validate-only or --dry-run."""

from __future__ import annotations

from .common import external_gate_main, verify_oracle_gate_artifact


def main(argv: list[str] | None = None) -> int:
    return external_gate_main(
        gate_name="oracle_clean_replay",
        description="Validate or run two same-worker oracle clean replays.",
        placeholders={"seed": "5", "replay_count": "2"},
        required_placeholders=frozenset(
            {"config", "artifact", "run_id", "seed", "replay_count"}
        ),
        verifier=verify_oracle_gate_artifact,
        default_runner_command=(
            "'{python}' -m scripts.capsule_rl.server_adapter "
            "--config '{config}' --artifact '{artifact}' --run-id '{run_id}' "
            "oracle --seed {seed} --replays {replay_count}"
        ),
        lock_runner_command=True,
        argv=argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
