"""Audit completed Spill Wipe A/C training and paired evaluation artifacts on the server."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from capx.rl.capsule.checkpoint import checkpoint_tree_sha256
from scripts.capsule_rl.run_cube_stack_ac import audit_training, compare_evaluations


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--candidate-root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    assert read(root / "status.json")["status"] == "completed"
    training = {}
    evaluations = {}
    infrastructure = []
    for arm, trigger in (("A16", "never"), ("C16", "any_failed")):
        path = root / arm
        training[arm] = audit_training(path, trigger, 16)
        result = read(path / "training_result.json")
        assert checkpoint_tree_sha256(result["checkpoint"]) == result["checkpoint_sha256"]
        assert result["optimizer_step_after"] - result["optimizer_step_before"] == result["actor_updates"]
        protocol = read(path / "protocol.json")
        assert protocol["training_seeds"] == list(range(5, 21))
        assert protocol["task"]["environment"] == "robosuite_spill_wipe"
        assert protocol["controller_service"]["model"] == "glm-5"
        evaluations[arm] = root / f"eval_{arm}"
        evaluation = read(evaluations[arm] / "protocol.json")
        assert evaluation["seeds"] == list(range(201, 221))
        assert evaluation["samples_per_seed"] == 4
        assert evaluation["controller_enabled"] is False
        for artifact in (path / "training/groups").glob("*.json"):
            assembly = read(artifact)["assembly"]
            replays = list(assembly["base_results"])
            for attempt in assembly["repair_attempts"]:
                replays.extend(attempt[k] for k in ("pt_result", "revision_result") if attempt.get(k))
            for replay in replays:
                diagnostics = json.dumps(replay.get("diagnostics", {}))
                for marker in ("Connection refused", "Failed to communicate with", "HTTPConnectionPool",
                               "ReadTimeout", "ConnectError", "Server Error:"):
                    if marker in diagnostics:
                        infrastructure.append({"artifact": str(artifact), "marker": marker})
    assert not infrastructure, infrastructure
    comparison = compare_evaluations(evaluations, (16,))
    assert comparison == read(root / "comparison.json")
    candidate_verified = False
    if args.candidate_root:
        for name in ("evaluate_cube_stack", "run_cube_stack_ac"):
            path = args.candidate_root / "scripts/capsule_rl" / f"{name}.py"
            spec = importlib.util.spec_from_file_location(f"candidate_{name}", path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            if name == "evaluate_cube_stack":
                for evaluation in evaluations.values():
                    for ordinal in range(80):
                        record = module.verify_replay(evaluation, "trained_lora", ordinal)
                        protocol = read(evaluation / "protocol.json")
                        assert record["initial_state_sha256"] == protocol["initial_states"][str(record["environment_seed"])]
            else:
                for arm, trigger in (("A16", "never"), ("C16", "any_failed")):
                    assert module.audit_training(root / arm, trigger, 16) == training[arm]
                assert module.compare_evaluations(evaluations, (16,)) == comparison
        candidate_verified = True
    report = {"training": training, "comparison": comparison,
              "accepted_training_infrastructure_failures": infrastructure,
              "candidate_audit_verified": candidate_verified}
    (root / "verified_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
