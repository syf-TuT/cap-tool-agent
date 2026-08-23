"""Gate 2 seed 5 -> 6 -> 5; preview safely with --validate-only/--dry-run."""

from __future__ import annotations

from .common import external_gate_main, verify_seed_gate_artifact


def main(argv: list[str] | None = None) -> int:
    return external_gate_main(
        gate_name="seed_determinism",
        description="Validate or run the real deterministic reset gate.",
        placeholders={"seed_sequence": "5,6,5"},
        required_placeholders=frozenset(
            {"config", "artifact", "run_id", "seed_sequence"}
        ),
        verifier=verify_seed_gate_artifact,
        default_runner_command=(
            "'{python}' -m scripts.capsule_rl.server_adapter "
            "--config '{config}' --artifact '{artifact}' --run-id '{run_id}' "
            "seed --seeds {seed_sequence}"
        ),
        lock_runner_command=True,
        argv=argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
