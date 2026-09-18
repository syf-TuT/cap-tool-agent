"""Gate 6 one optimizer step; preview with --validate-only or --dry-run."""

from __future__ import annotations

from .common import external_gate_main, verify_trainer_gate_artifact


def main(argv: list[str] | None = None) -> int:
    return external_gate_main(
        gate_name="one_step_trainer",
        description="Validate or run one Capsule-Critique Program optimizer step.",
        placeholders={"optimizer_steps": "1", "group_rewards": "0,0,0,0,0,0,0,1"},
        required_placeholders=frozenset(
            {
                "config",
                "artifact",
                "run_id",
                "guided_artifact",
                "optimizer_steps",
                "group_rewards",
            }
        ),
        verifier=verify_trainer_gate_artifact,
        default_runner_command=(
            "'{python}' -m scripts.capsule_rl.server_adapter "
            "--config '{config}' --artifact '{artifact}' --run-id '{run_id}' "
            "trainer --optimizer-steps {optimizer_steps} --group-rewards {group_rewards} "
            "--guided-artifact '{guided_artifact}'"
        ),
        direct_artifact_publish=True,
        lock_runner_command=True,
        argv=argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
