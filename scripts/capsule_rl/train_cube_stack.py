"""Prepare and train privileged high-level cube tasks with the Cube Lift Capsule recipe.

Run preparation on the simulator host. Task prompts and initial-state hashes come from real
resets; the model, decoder, group assembler, loss, and LoRA recipe are shared with Cube Lift.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import yaml

from capx.rl.capsule.checkpoint import checkpoint_tree_sha256
from capx.rl.capsule.config_validation import TRAJECTORY_BC_LOSS_MODE, validate_bc_coef
from capx.rl.capsule.repair_manifest import (
    MANIFEST_NAME,
    export_manifest,
    match_manifest,
    read_manifest,
)
from scripts.capsule_rl.common import atomic_write_json, load_and_validate_server_config


def prepare(
    root: Path, model: Path, verl: Path, seeds: tuple[int, ...],
    resume_from: Path | None = None,
    *, task: str = "cube_stack", repair_trigger: str = "all_failed",
    checkpoint_root: Path | None = None,
    controller_model: str | None = None,
    controller_endpoint: str | None = None,
    method: str = "cc-grpo",
    bc_coef: float | None = None,
    record_repair_turns: bool = False,
    repair_manifest: Path | None = None,
) -> None:
    from capx.rl.capsule.server_factory import YamlEnvironmentFactory, load_task_instances

    if task not in ("cube_stack", "cube_restack", "cube_lift", "spill_wipe"):
        raise ValueError(f"unsupported training task: {task}")
    trajectory_bc = method == "grpo-bc-trajectory"
    if trajectory_bc:
        if task not in ("cube_stack", "spill_wipe"):
            raise ValueError("GRPO + BC supports cube_stack and spill_wipe")
        validate_bc_coef(bc_coef)
    elif method != "cc-grpo" or bc_coef is not None:
        raise ValueError("bc_coef is available only with grpo-bc-trajectory")
    if trajectory_bc != (repair_manifest is not None):
        raise ValueError("--repair-manifest is required only for grpo-bc-trajectory")
    if record_repair_turns and method != "cc-grpo":
        raise ValueError("--record-repair-turns applies only to source CC-GRPO")
    if (trajectory_bc or record_repair_turns) and resume_from is not None:
        raise ValueError("paired trajectory experiments must start from the base model")
    manifest = read_manifest(repair_manifest) if repair_manifest is not None else None
    if not seeds or min(seeds) < 0 or len(set(seeds)) != len(seeds):
        raise ValueError("training seeds must be distinct non-negative integers")
    parent = None
    ancestry_seeds = []
    prior_groups = 0
    if resume_from is not None:
        resume_from = resume_from.resolve()
        parent_protocol_path = resume_from / "protocol.json"
        parent_result_path = resume_from / "training_result.json"
        parent_protocol = json.loads(parent_protocol_path.read_text())
        parent_result = json.loads(parent_result_path.read_text())
        if (
            parent_protocol.get("method", "cc-grpo") != method
            or parent_protocol.get("bc_coef") != bc_coef
        ):
            raise ValueError("continuation must retain the parent's method and BC coefficient")
        if (
            parent_result["status"] not in (
                ("completed", "completed_no_updates_all_constant")
                if task == "spill_wipe" else ("completed",)
            )
            or parent_result["completed_group_count"] != parent_protocol["group_count"]
            or parent_protocol["program_service"]["model"] != str(model.resolve())
            or parent_protocol["task"]["environment"] != f"robosuite_{task}"
        ):
            raise ValueError("parent must be a completed run of the same base model")
        ancestry_seeds = parent_protocol["training_seeds"]
        if set(seeds) & set(ancestry_seeds):
            raise ValueError("new training scenes must not overlap ancestor training scenes")
        prior_groups = parent_protocol.get("cumulative_group_count", parent_protocol["group_count"])
        parent = {
            "training_root": str(resume_from),
            "protocol_sha256": hashlib.sha256(parent_protocol_path.read_bytes()).hexdigest(),
            "result_sha256": hashlib.sha256(parent_result_path.read_bytes()).hexdigest(),
            "checkpoint": parent_result["checkpoint"],
            "checkpoint_sha256": parent_result["checkpoint_sha256"],
            "optimizer_step": parent_result["optimizer_step_after"],
        }
    project = Path(__file__).resolve().parents[2]
    config_dir = project / "env_configs/cube_stack/capsule_rl"
    config = yaml.safe_load(
        (config_dir / "franka_robosuite_cube_stack_capsule_critique_grpo.yaml").read_text()
    )
    config["capsule"]["repair_trigger"] = repair_trigger
    if controller_model is not None:
        config["controller_service"]["model"] = controller_model
    if controller_endpoint is not None:
        config["controller_service"]["endpoint"] = controller_endpoint
    if parent is not None and parent_protocol["capsule"].get("repair_trigger", "all_failed") != repair_trigger:
        raise ValueError("continuation must retain the parent's repair trigger")
    lift = yaml.safe_load(
        (
            project
            / ("env_configs/cube_lifting/capsule_rl/franka_robosuite_cube_lift_capsule_smoke.yaml")
        ).read_text()
    )
    config["runtime"].update(
        {
            "run_id": root.name,
            "project_root": str(project),
            "verl_source_path": str(verl.resolve()),
            "verl_resolved_config_path": str(root / "verl.yaml"),
            "dataset_path": str(root / "dataset.seed_resolved.jsonl"),
            "program_model_path": str(model.resolve()),
            "output_dir": str(root / "training"),
        }
    )
    config["task"]["profile"] = "robosuite_cube_stack_privileged"
    if checkpoint_root is not None:
        config["runtime"]["checkpoint_root"] = str(checkpoint_root.resolve())
    if task == "cube_restack":
        config["task"].update({
            "profile": "robosuite_cube_restack_privileged_highlevel",
            "environment": "robosuite_cube_restack",
            "config_path": "env_configs/cube_restack/capsule_rl/"
            "franka_robosuite_cube_restack_privileged_clean_replay.yaml",
        })
    elif task == "cube_lift":
        config["task"] = dict(lift["task"])
    elif task == "spill_wipe":
        config["task"].update({
            "profile": "robosuite_spill_wipe_privileged",
            "environment": "robosuite_spill_wipe",
            "api": "franka_spill_wipe_privileged",
            "config_path": "env_configs/spill_wipe/capsule_rl/"
            "franka_robosuite_spill_wipe_privileged_clean_replay.yaml",
        })
    # Stack repairs replace longer functions: observed complete traces exceed Lift's 8K limit.
    # The worker factory propagates this capacity without truncating the committed history.
    config["capsule"]["revision_input_max_tokens"] = 24576
    config["capsule"]["allow_fenced_revisions"] = True
    config["program_service"] = dict(lift["program_service"])
    config["program_service"]["model"] = str(model.resolve())
    environment = YamlEnvironmentFactory(str(project / config["task"]["config_path"]))(None)
    task_id = "cube-lift" if task == "cube_lift" else f"{task.replace('_', '-')}-red-on-green"
    if task == "spill_wipe":
        task_id = "spill-wipe-brown-spill"
    rows = []
    expected_prompt = None
    try:
        for seed in seeds:
            observation, info = environment.reset(
                seed=seed, options={"capsule_task_state_resolution": True}
            )
            messages = observation["full_prompt"]
            system = messages[0]["content"]
            prompt = messages[1]["content"][0]["text"]
            if system != config["program_service"]["system_prompt"]:
                raise RuntimeError("environment system prompt differs from the Cube Lift recipe")
            if expected_prompt is not None and prompt != expected_prompt:
                raise RuntimeError("task prompt unexpectedly varies across training seeds")
            expected_prompt = prompt
            rows.append(
                {
                    "task_id": task_id,
                    "prompt": prompt,
                    "environment_seed": seed,
                    "initial_state_sha256": info["initial_state_sha256"],
                }
            )
    finally:
        environment.close()
    config["program_service"]["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
    if parent is not None and config["program_service"] != parent_protocol["program_service"]:
        raise ValueError("continuation must retain the parent's Program prompt and sampling recipe")
    model_hash = None
    if manifest is not None or record_repair_turns:
        model_hash = checkpoint_tree_sha256(model.resolve())
        config["runtime"]["base_model_sha256"] = model_hash
    if manifest is not None:
        match_manifest(
            manifest, rows, environment=config["task"]["environment"],
            model_sha256=model_hash, system_prompt=config["program_service"]["system_prompt"],
        )
    root.mkdir(parents=True, exist_ok=False)
    input_files = ["runtime.yaml", "verl.yaml", "dataset.seed_resolved.jsonl"]
    if manifest is not None:
        snapshot = atomic_write_json(root / MANIFEST_NAME, manifest)
        config["runtime"]["repair_manifest_path"] = str(snapshot.resolve())
        config["runtime"]["repair_manifest_file_sha256"] = hashlib.sha256(
            snapshot.read_bytes()
        ).hexdigest()
        input_files.append(MANIFEST_NAME)
    if record_repair_turns:
        config["capsule"]["record_repair_turns"] = True
    (root / "dataset.seed_resolved.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    worker_config = yaml.safe_load(
        (config_dir / "franka_robosuite_cube_stack_capsule_single_a800_verl.yaml").read_text()
    )
    worker_config["trainer"]["total_epochs"] = 1
    worker_config["trainer"]["experiment_name"] = root.name
    if task == "cube_lift":
        # Equal reward-baseline weight and full guided log-probability coefficient.
        for training_config in (config, worker_config):
            training_config["actor_rollout_ref"]["actor"]["policy_loss"][
                "guided_objective"
            ] = "log_probability"
    if trajectory_bc:
        for training_config in (config, worker_config):
            actor = training_config["actor_rollout_ref"]["actor"]
            actor["policy_loss"].update(loss_mode=TRAJECTORY_BC_LOSS_MODE, bc_coef=bc_coef)
            actor["use_kl_loss"] = False
    (root / "verl.yaml").write_text(yaml.safe_dump(worker_config, sort_keys=False))
    config_path = root / "runtime.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    validated = load_and_validate_server_config(config_path, check_runtime_paths=True)
    tasks = load_task_instances(validated)
    atomic_write_json(
        root / "protocol.json",
        {
            "mode": f"privileged_highlevel_{task}_capsule_rl",
            "method": method,
            "bc_coef": bc_coef,
            "record_repair_turns": record_repair_turns,
            "base_model_sha256": model_hash,
            "repair_manifest_sha256": manifest["manifest_sha256"] if manifest else None,
            "project_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=project, text=True
            ).strip(),
            "training_seeds": ancestry_seeds + list(seeds),
            "new_training_seeds": list(seeds),
            "group_count": len(tasks),
            "cumulative_group_count": prior_groups + len(tasks),
            "resume_from": parent,
            "group_size": config["capsule"]["group_size"],
            "repair_trigger": repair_trigger,
            "program_service": config["program_service"],
            "controller_service": config["controller_service"],
            "capsule": config["capsule"],
            "task": config["task"],
            "input_sha256": {
                name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                for name in input_files
            },
        },
    )
    print(json.dumps({"config": str(config_path), "groups": len(tasks)}), flush=True)


def summarize(root: Path, result: dict) -> dict:
    groups = []
    for path in sorted((root / "training/groups").glob("*.json")):
        artifact = json.loads(path.read_text())
        assembly = artifact["assembly"]
        group = assembly["group"]
        groups.append(
            {
                "artifact": str(path),
                "seed": group["environment_seed"],
                "rewards": artifact["sequence_rewards"],
                "base_successes": sum(r["outcome"] == "success" for r in assembly["base_results"]),
                "initial_base_successes": sum(
                    r["outcome"] == "success" for r in assembly["base_results"][:7]
                ),
                "repair_triggered": group["metadata"]["repair_triggered"],
                "guided_success": group["metadata"]["guided_member_selected"],
                "controller_successes": sum(
                    r.get("pt_result") is not None and r["pt_result"]["outcome"] == "success"
                    for r in assembly["repair_attempts"]
                ),
                "repair_statuses": [r["status"] for r in assembly["repair_attempts"]],
                "repair_rejection_reasons": [
                    r["rejection_reason"]
                    for r in assembly["repair_attempts"]
                    if r.get("rejection_reason")
                ],
                "skipped_actor_update": artifact["skipped_actor_update"],
                **({"trajectory_bc": artifact["trajectory_bc"], "loss_metrics": artifact.get(
                    "loss_metrics"
                )} if "trajectory_bc" in artifact else {}),
            }
        )
    if len(groups) != result["completed_group_count"]:
        raise RuntimeError("saved group count disagrees with the completed training run")
    return {
        "training": result,
        "all_initial_base_failed_groups": sum(g["initial_base_successes"] == 0 for g in groups),
        "repair_triggered_groups": sum(g["repair_triggered"] for g in groups),
        "initial_base_success_rate": sum(g["initial_base_successes"] for g in groups) / (7 * len(groups)),
        "controller_successful_repairs": sum(g["controller_successes"] for g in groups),
        "capsule_repaired_groups": sum(g["guided_success"] for g in groups),
        "groups": groups,
    }


def train(root: Path) -> None:
    from capx.rl.capsule.checkpoint import checkpoint_tree_sha256
    from capx.rl.capsule.server_factory import create_trainer

    config = load_and_validate_server_config(root / "runtime.yaml", check_runtime_paths=True)
    if (
        config["actor_rollout_ref"]["actor"]["policy_loss"]["loss_mode"]
        != TRAJECTORY_BC_LOSS_MODE
        and config["capsule"].get("repair_trigger", "all_failed") != "never"
        and not os.environ.get(config["controller_service"]["api_key_env"])
    ):
        raise RuntimeError("set the Controller API key before starting Capsule training")
    protocol = json.loads((root / "protocol.json").read_text())
    for name, expected in protocol["input_sha256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"prepared training input changed: {name}")
    runtime = create_trainer(config)
    parent = protocol.get("resume_from")
    if parent is not None:
        parent_root = Path(parent["training_root"])
        for filename, key in (("protocol.json", "protocol_sha256"), ("training_result.json", "result_sha256")):
            if hashlib.sha256((parent_root / filename).read_bytes()).hexdigest() != parent[key]:
                raise RuntimeError(f"parent training artifact changed: {filename}")
        if checkpoint_tree_sha256(parent["checkpoint"]) != parent["checkpoint_sha256"]:
            raise RuntimeError("parent checkpoint changed before continuation")
        original_starter = runtime.worker_starter

        def restored_starter(config):
            workers = original_starter(config)
            try:
                workers.actor_rollout_wg.load_checkpoint(
                    parent["checkpoint"], del_local_after_load=False
                )
                restored_step = workers.optimizer_step()
                if restored_step != parent["optimizer_step"]:
                    raise RuntimeError("restored optimizer step does not match parent")
                atomic_write_json(root / "restore_evidence.json", {
                    **parent,
                    "restored_optimizer_step": restored_step,
                    "load_contents": ["model", "optimizer", "extra"],
                })
                print(f"Restored parent checkpoint at optimizer step {restored_step}", flush=True)
                return workers
            except BaseException:
                workers.close()
                raise

        runtime.worker_starter = restored_starter
    started = time.monotonic()
    result = runtime.fit()
    if config["capsule"].get("record_repair_turns"):
        result["group_artifact_sha256"] = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((root / "training/groups").glob("*.json"))
        }
    result["training_wall_seconds"] = time.monotonic() - started
    result["cumulative_group_count"] = protocol.get("cumulative_group_count", protocol["group_count"])
    atomic_write_json(root / "training_result.json", result)
    summary = summarize(root, result)
    atomic_write_json(root / "summary.json", summary)
    print(json.dumps(summary), flush=True)


def main(task: str = "cube_stack") -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("cube_stack", "cube_restack", "cube_lift", "spill_wipe"), default=task)
    parser.add_argument("phase", choices=("prepare", "train", "summarize", "export-repair-trajectories"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--verl", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--controller-model")
    parser.add_argument("--controller-endpoint")
    parser.add_argument(
        "--method", choices=("cc-grpo", "grpo-bc-trajectory"),
        help="policy objective for prepare (default: cc-grpo)",
    )
    parser.add_argument("--bc-coef", type=float, help="required with grpo-bc-trajectory")
    parser.add_argument("--record-repair-turns", action="store_true")
    parser.add_argument("--repair-manifest", type=Path)
    parser.add_argument("--repair-trigger", choices=("never", "any_failed", "all_failed"), default="all_failed")
    parser.add_argument("--seeds", default=",".join(map(str, range(5, 21))))
    args = parser.parse_args()
    if args.phase != "prepare" and (
        args.method is not None or args.bc_coef is not None
        or args.record_repair_turns or args.repair_manifest is not None
    ):
        parser.error("method, BC, and trajectory options apply only to prepare")
    root = args.root.resolve()
    if args.phase == "prepare":
        if args.model is None or args.verl is None:
            parser.error("prepare requires --model and --verl")
        prepare(root, args.model, args.verl, tuple(int(s) for s in args.seeds.split(",")),
                args.resume_from, task=args.task, repair_trigger=args.repair_trigger,
                checkpoint_root=args.checkpoint_root, controller_model=args.controller_model,
                controller_endpoint=args.controller_endpoint, method=args.method or "cc-grpo",
                bc_coef=args.bc_coef, record_repair_turns=args.record_repair_turns,
                repair_manifest=args.repair_manifest)
    elif args.phase == "train":
        train(root)
    elif args.phase == "export-repair-trajectories":
        path = atomic_write_json(root / MANIFEST_NAME, export_manifest(root))
        print(path)
    else:
        result = json.loads((root / "training_result.json").read_text())
        atomic_write_json(root / "summary.json", summarize(root, result))


if __name__ == "__main__":
    main()
