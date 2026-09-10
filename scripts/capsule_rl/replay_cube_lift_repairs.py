"""Replay the 17 selected Cube Lift revisions and their P0s without privileged perception.

No model generation or training occurs. Each archived program runs once on its original
training scene, using the existing non-privileged evaluation and source-normalization rules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml

from scripts.capsule_rl.common import atomic_write_json
from scripts.capsule_rl.evaluate_cube_stack import evaluate, read, sha256, verify_replay


def collect_pairs(training_root: Path) -> list[dict]:
    pairs = []
    for stage in ("C16", "C32"):
        for path in sorted((training_root / stage / "training/groups").glob("*.json")):
            assembly = read(path)["assembly"]
            members = [m for m in assembly["group"]["members"]
                       if m["member_type"] == "critique_guided_revision"]
            if not members:
                continue
            if len(members) != 1:
                raise RuntimeError(f"expected one selected revision: {path}")
            member = members[0]
            attempts = [a for a in assembly["repair_attempts"]
                        if a["selected"] and a["revision_program_sample_id"]
                        == member["program_sample_id"]]
            if len(attempts) != 1:
                raise RuntimeError(f"ambiguous selected repair: {path}")
            attempt = attempts[0]
            originals = [r for r in assembly["base_results"]
                         if r["program_sample_id"] == attempt["p0_program_sample_id"]]
            if len(originals) != 1:
                raise RuntimeError(f"missing original failed program: {path}")
            original = originals[0]
            revision = attempt["revision_result"]
            if (
                original["outcome"] not in (
                    "task_failure", "program_error", "program_timeout", "step_budget_exhausted"
                )
                or original["binary_reward"] != 0
                or revision["outcome"] != "success"
                or member["reward"] != 1
                or member["response"] != revision["source"]
                or member["response"] != attempt["revision_source"]
            ):
                raise RuntimeError(f"invalid failed/successful pair: {path}")
            for replay in (original, revision):
                if hashlib.sha256(replay["source"].encode()).hexdigest() != replay["source_sha256"]:
                    raise RuntimeError(f"archived program hash mismatch: {path}")
            pairs.append({
                "pair_id": f"{stage}-seed-{assembly['group']['environment_seed']}",
                "stage": stage,
                "environment_seed": assembly["group"]["environment_seed"],
                "group_path": str(path),
                "group_sha256": sha256(path),
                "original": {key: original[key] for key in
                             ("program_sample_id", "source", "source_sha256", "outcome")},
                "repaired": {key: revision[key] for key in
                             ("program_sample_id", "source", "source_sha256", "outcome")},
            })
    if len(pairs) != 17 or len({p["pair_id"] for p in pairs}) != 17:
        raise RuntimeError(f"expected 17 distinct selected repair pairs, found {len(pairs)}")
    return pairs


def prepare(root: Path, training_root: Path) -> None:
    pairs = collect_pairs(training_root)
    if (root / "protocol.json").exists():
        if read(root / "pairs.json")["pairs"] != pairs:
            raise RuntimeError("archived training inputs changed")
        return
    root.mkdir(parents=True, exist_ok=False)
    reference_path = training_root / "eval_C16/protocol.json"
    reference = read(reference_path)
    config_path = Path(reference["environment_config"])
    if sha256(config_path) != reference["environment_config_sha256"]:
        raise RuntimeError("reference evaluation environment changed")
    config = yaml.safe_load(config_path.read_text())
    if config["env"]["cfg"]["privileged"] or config["env"]["cfg"]["apis"] != ["FrankaControlApi"]:
        raise RuntimeError("replay must use non-privileged high-level perception")
    config["output_dir"] = str(root / "environment_outputs")
    config["record_video"] = False
    (root / "environment.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    atomic_write_json(root / "pairs.json", {"pairs": pairs})
    atomic_write_json(root / "protocol.json", {
        "mode": "archived_cube_lift_p0_vs_revision_nonprivileged_replay",
        "training_root": str(training_root),
        "pairs_sha256": sha256(root / "pairs.json"),
        "reference_protocol_sha256": sha256(reference_path),
        "environment_config": str(root / "environment.yaml"),
        "environment_config_sha256": sha256(root / "environment.yaml"),
        "user_prompt": reference["user_prompt"],
        "total_samples_per_policy": len(pairs),
        "controller_enabled": False,
        "generation_enabled": False,
        "scene_policy": "original training seed, fresh reset for each program",
        "source_sha256": {str(Path(__file__).resolve()): sha256(Path(__file__))},
    })
    for policy in ("original", "repaired"):
        destination = root / "generations" / policy
        destination.mkdir(parents=True)
        identity_path = destination / "identity.json"
        atomic_write_json(identity_path, {
            "protocol_sha256": sha256(root / "protocol.json"),
            "policy": policy, "origin": "archived_program_without_new_generation",
        })
        for ordinal, pair in enumerate(pairs):
            atomic_write_json(destination / f"sample_{ordinal:04d}.json", {
                "ordinal": ordinal, "policy": policy,
                "pair_id": pair["pair_id"], "stage": pair["stage"],
                "environment_seed": pair["environment_seed"],
                "identity_sha256": sha256(identity_path),
                "source": pair[policy]["source"],
                "archived_source_sha256": pair[policy]["source_sha256"],
                "archived_program_sample_id": pair[policy]["program_sample_id"],
                "privileged_outcome": pair[policy]["outcome"],
                "group_sha256": pair["group_sha256"],
            })


def summarize(root: Path) -> dict:
    protocol = read(root / "protocol.json")
    if sha256(root / "pairs.json") != protocol["pairs_sha256"]:
        raise RuntimeError("pair manifest changed")
    rows = []
    transitions: Counter = Counter()
    outcomes = {policy: Counter() for policy in ("original", "repaired")}
    for ordinal in range(protocol["total_samples_per_policy"]):
        original, repaired = (verify_replay(root, policy, ordinal)
                              for policy in ("original", "repaired"))
        for key in ("pair_id", "environment_seed", "initial_state_sha256", "group_sha256"):
            if not original[key] or original[key] != repaired[key]:
                raise RuntimeError(f"pair mismatch at {ordinal}: {key}")
        transitions[f"{original['outcome']} -> {repaired['outcome']}"] += 1
        for policy, record in (("original", original), ("repaired", repaired)):
            outcomes[policy][record["outcome"]] += 1
        rows.append({
            "pair_id": original["pair_id"], "environment_seed": original["environment_seed"],
            "original": original["outcome"], "repaired": repaired["outcome"],
            "original_error": original["error_message"],
            "repaired_error": repaired["error_message"],
            "initial_state_sha256": original["initial_state_sha256"],
        })
    result = {
        "status": "completed", "pairs": len(rows), "replays": len(rows) * 2,
        "outcomes": {key: dict(value) for key, value in outcomes.items()},
        "success_rate_delta": (outcomes["repaired"]["success"]
                               - outcomes["original"]["success"]) / len(rows),
        "transitions": dict(transitions), "records": rows,
        "scope": "selected successful repairs on original scenes; one replay per program; "
                 "tests perception transfer, not unseen-scene or trained-policy generalization",
    }
    atomic_write_json(root / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    prepare(root, args.training_root.resolve())
    for policy in ("original", "repaired"):
        evaluate(root, policy)
    print(json.dumps(summarize(root)), flush=True)


if __name__ == "__main__":
    main()
