"""Audit archived Cube Lift sampling inputs and optionally replay six paired cases.

Run from the remote project with its prepared Python environment. No LLM is loaded,
no optimizer runs, and existing experiment files are read only. --replay requires
the existing local Pyroki endpoint. All new evidence goes under --output.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml

BASELINE = "remote_results/cube_lift_base_vs_lora100_heldout_t07_s25_124_20260828_gate4fix_r02"
TRAIN = "artifacts/cube_lift_capsule_rl_train16_direct_nogates_seeds5_20_20260904_r01"
GROUPS = "outputs/cube_lift_capsule_rl_train16_direct_nogates_seeds5_20_20260904_r02/groups"
PROBE = (
    "artifacts/cube_lift_capsule_rl_prompt_homepose_t07_base_group_probe_seed5_s21_"
    "20260904_r01/base_sanity.json"
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def prompt_record(system: str, user: str) -> dict:
    return {
        "system": system,
        "user": user,
        "user_chars": len(user),
        "system_sha256": digest(system),
        "user_sha256": digest(user),
    }


def archive_audit(root: Path) -> dict:
    baseline = root / BASELINE
    run = baseline / "base/base/run"
    messages = ast.literal_eval((run / "initial_prompt.txt").read_text())
    baseline_prompt = prompt_record(messages[0]["content"], messages[1]["content"][0]["text"])
    train_path = root / TRAIN / "direct_runtime_r02.yaml"
    train = read_yaml(train_path)
    task = json.loads(Path(train["runtime"]["dataset_path"]).read_text().splitlines()[0])
    train_prompt = prompt_record(train["program_service"]["system_prompt"], task["prompt"])
    probe = read_json(root / PROBE)
    probe_config = read_yaml(root / probe["config_path"])
    probe_prompt = prompt_record(
        probe_config["program_service"]["system_prompt"], probe["task_instance"]["prompt"]
    )
    train_verl = read_yaml(Path(train["runtime"]["verl_resolved_config_path"]))
    probe_verl = read_yaml(Path(probe_config["runtime"]["verl_resolved_config_path"]))
    log = (root / TRAIN / "train_r02.log").read_text()
    actual_kwargs = [
        ast.literal_eval(line.split("kwargs: ", 1)[1])
        for line in log.splitlines()
        if "kwargs: {" in line
    ]
    archived = [read_json(p) for p in run.glob("trial_*_result.json")]
    positives = [r for r in archived if r["task_completed"]]
    group_rows = []
    base_results = []
    rejected_revisions = []
    for path in sorted((root / GROUPS).glob("*.json")):
        data = read_json(path)
        assembly = data["assembly"]
        results = assembly["base_results"]
        base_results.extend(results)
        group_rows.append({
            "file": str(path),
            "seed": results[0]["environment_seed"],
            "base_count": len(results),
            "base_successes": sum(r["binary_reward"] for r in results),
            "rewards": data["sequence_rewards"],
            "skipped_actor_update": data["skipped_actor_update"],
        })
        for repair in assembly["repair_attempts"]:
            if repair.get("rejection_reason") == "markdown_fence":
                rejected_revisions.append({
                    "seed": results[0]["environment_seed"],
                    "pt_outcome": repair["pt_result"]["outcome"],
                    "revision_evaluated": repair.get("revision_result") is not None,
                })

    def replay_summary(rows):
        return {
            "count": len(rows),
            "outcomes": dict(Counter(r["outcome"] for r in rows)),
            "observed_success_rejected": [
                r["program_sample_id"] for r in rows
                if r["diagnostics"].get("observed_task_completed") and r["binary_reward"] != 1
            ],
            "unique_sources": len({r["source_sha256"] for r in rows}),
        }

    def rollout(config):
        r = config["actor_rollout_ref"]["rollout"]
        keys = ["temperature", "top_p", "top_k", "repetition_penalty",
                "response_length", "ignore_eos", "n"]
        return {key: r.get(key) for key in keys}

    return {
        "paths": {"baseline": str(baseline), "training": str(train_path), "probe": str(root / PROBE)},
        "prompts": {"baseline": baseline_prompt, "training": train_prompt, "probe": probe_prompt},
        "prompt_diffs": {name: list(difflib.unified_diff(baseline_prompt["user"].splitlines(),
                         p["user"].splitlines(), fromfile="baseline", tofile=name))
                         for name, p in [("training", train_prompt), ("probe", probe_prompt)]},
        "sampling": {
            "baseline_protocol": read_json(baseline / "protocol.json"),
            "model_generation_config": read_json(root / ".codex-downloads/models/Qwen2.5-Coder-7B-Instruct/generation_config.json"),
            "training_verl_path": train["runtime"]["verl_resolved_config_path"],
            "training_resolved_rollout": rollout(train_verl), "training_logged_kwargs": actual_kwargs,
            "probe_resolved_rollout": rollout(probe_verl)},
        "environment_payload_equal": read_yaml(baseline / "eval.yaml")["env"] == read_yaml(root / train["task"]["config_path"])["env"],
        "baseline_outcomes": {"count": len(archived), "successes": len(positives),
            "sandbox_errors": sum(r["sandbox_rc"] != 0 for r in archived),
            "positive_with_program_error": [r["trial"] for r in positives if r["sandbox_rc"] != 0],
            "positive_below_reward_threshold": [r["trial"] for r in positives if r["reward"] < 1.0]},
        "training_outcomes": replay_summary(base_results),
        "training_error_types": dict(Counter(r["error_type"] for r in base_results if r["error_type"])),
        "groups": group_rows, "fenced_revision_rejections": rejected_revisions,
        "initial_probe_outcomes": replay_summary(probe["results"]),
        "probe_failed_sources_with_release_after_close": [i for i, r in enumerate(probe["results"])
            if r["outcome"] == "task_failure" and r["source"].rfind("open_gripper(") > r["source"].rfind("close_gripper(") >= 0],
    }


def config_probe(root: Path, audit: dict) -> dict:
    """Resolve the actual training config and tokenize locally without loading weights."""
    from capx.rl.capsule.server_factory import _load_resolved_verl_config
    from transformers import AutoTokenizer

    paths = {"formal_training": f"{TRAIN}/direct_runtime_r02.yaml"}
    for length in (2048, 4096):
        paths[f"probe_t07_len{length}"] = (
            "artifacts/cube_lift_capsule_rl_prepare_seeds5_24_20260904_"
            f"prompt_homepose_t07_len{length}_r01/capsule_rl.resolved.yaml"
        )
    result = {}
    for label, path in paths.items():
        config = read_yaml(root / path)
        resolved_path = Path(config["runtime"]["verl_resolved_config_path"])
        saved = read_yaml(resolved_path)
        effective = _load_resolved_verl_config(config, root).actor_rollout_ref.rollout
        result[label] = {
            "runtime_config": path,
            "resolved_verl": str(resolved_path),
            "on_disk_response_length": saved["actor_rollout_ref"]["rollout"]["response_length"],
            "effective_response_length": effective.response_length,
            "effective_temperature": effective.temperature,
            "effective_top_p": effective.top_p,
            "effective_top_k": effective.top_k,
            "capsule_revision_response_max_tokens": config["capsule"]["revision_response_max_tokens"],
        }
    tokenizer = AutoTokenizer.from_pretrained(
        root / ".codex-downloads/models/Qwen2.5-Coder-7B-Instruct", local_files_only=True
    )
    result["prompt_token_lengths"] = {}
    for name, prompt in audit["prompts"].items():
        messages = [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ]
        result["prompt_token_lengths"][name] = len(tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        ))
    baseline_run = root / BASELINE / "base/base/run"
    sources = [path.read_text() for path in baseline_run.glob("trial_*/raw_response.sh")]
    probe_sources = [row["source"] for row in read_json(root / PROBE)["results"]]
    for label, rows in (("baseline", sources), ("probe", probe_sources)):
        lengths = [len(tokenizer.encode(source, add_special_tokens=False)) for source in rows]
        result[f"{label}_response_tokens"] = {
            "count": len(lengths), "maximum": max(lengths),
            "over2048": sum(length > 2048 for length in lengths),
        }
    result["baseline_release_after_close_source_count"] = sum(
        source.rfind("open_gripper(") > source.rfind("close_gripper(") >= 0 for source in sources
    )
    return result


def paired_replay(root: Path, output: Path) -> list[dict]:
    from capx.rl.capsule.evaluator import CleanReplayEvaluator, PersistentProcessReplayBackend
    from capx.rl.capsule.schema import TaskInstanceV1
    from capx.rl.capsule.server_factory import YamlEnvironmentFactory
    from capx.utils.launch_utils import _extract_code
    from capx.utils.program_source import normalize_program_source

    probe = read_json(root / PROBE)
    baseline = root / BASELINE
    run = baseline / "base/base/run"

    def archived_source(seed):
        directory = next(p for p in run.glob(f"trial_{seed}_*") if p.is_dir())
        return (directory / "raw_response.sh").read_text()

    valid = probe["results"][14]["source"]
    release = probe["results"][3]["source"]
    cases = [
        ("baseline_success_s25", 25, archived_source(25)),
        ("baseline_failure_s102", 102, archived_source(102)),
        ("initial_probe_success", 5, valid),
        ("initial_probe_lift_then_release", 5, release),
        ("probe_remove_release_only", 5, release.replace(
            "    open_gripper()", "    pass  # diagnostic: omit final release"
        )),
        ("success_then_exception", 5, normalize_program_source(valid)
         + "\nraise ValueError('audit exception after lift')\n"),
    ]
    factory = YamlEnvironmentFactory(str(
        root / "env_configs/cube_lifting/capsule_rl/"
        "franka_robosuite_cube_lift_privileged_clean_replay.yaml"
    ))
    ordinary = YamlEnvironmentFactory(str(baseline / "eval.yaml"))(None)
    backend = PersistentProcessReplayBackend(factory)
    evaluator = CleanReplayEvaluator(backend, timeout_s=180, max_failure_retries=0)
    events = []
    # Observe physical success after each original API call; do not alter actions.
    for api in ordinary._apis.values():
        original_functions = api.functions()

        def wrap(name, fn):
            def observed(*args, **kwargs):
                value = fn(*args, **kwargs)
                events.append({
                    "function": name,
                    "task_completed": bool(ordinary.low_level_env.task_completed()),
                    "reward": float(ordinary.compute_reward()),
                })
                return value

            return observed

        wrapped = {name: wrap(name, fn) for name, fn in original_functions.items()}
        api.functions = lambda functions=wrapped: functions
    results = []
    try:
        for name, seed, source in cases:
            events.clear()
            _, reset = ordinary.reset(seed=seed)
            blocks = _extract_code(source)
            assert blocks == [normalize_program_source(source)]
            _, reward, terminated, truncated, info = ordinary.step(blocks[0])
            task_data = dict(probe["task_instance"])
            task_data.update(environment_seed=seed, initial_state_sha256=reset["initial_state_sha256"])
            task = TaskInstanceV1.from_dict(task_data)
            clean = evaluator.evaluate_program(task, source, seed, program_sample_id=name)
            record = {
                "case": name, "seed": seed, "source": source, "source_sha256": digest(source),
                "ordinary": {
                    "reward": float(reward), "task_completed": info["task_completed"],
                    "sandbox_rc": info["sandbox_rc"], "error_type": info["error_type"],
                    "terminated": terminated, "truncated": truncated,
                    "initial_state_sha256": reset["initial_state_sha256"],
                    "api_events": list(events),
                },
                "clean": clean.to_dict(),
                "worker_pid": backend.worker_pid,
                "same_physical_completion": (
                    info["task_completed"] == clean.diagnostics.get("observed_task_completed")
                ),
                "same_reward": (
                    clean.raw_reward is not None and abs(float(reward) - clean.raw_reward) < 1e-8
                ),
            }
            results.append(record)
            (output / "paired_replay.json").write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps({"case": name, "ordinary_success": info["task_completed"],
                "clean_outcome": clean.outcome.value, "same_reward": record["same_reward"]}), flush=True)
    finally:
        backend.close()
        ordinary.close()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--tokenizer-audit", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    audit_path = args.output / "archive_audit.json"
    if audit_path.exists():
        raise FileExistsError(audit_path)
    audit = archive_audit(args.root)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    if args.tokenizer_audit:
        (args.output / "config_probe.json").write_text(
            json.dumps(config_probe(args.root, audit), indent=2) + "\n"
        )
    print(json.dumps({"baseline": audit["baseline_outcomes"], "training": audit["training_outcomes"],
                      "probe": audit["initial_probe_outcomes"]}), flush=True)
    if args.replay:
        paired_replay(args.root, args.output)


if __name__ == "__main__":
    main()
