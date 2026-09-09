"""Paired ordinary GRPO versus any-failure critique, at 16 and 32 Cube Stack groups."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

import yaml

from scripts.capsule_rl.common import atomic_write_json
from scripts.capsule_rl.evaluate_cube_stack import read, verify_replay


def write_status(path: Path, status: dict) -> None:
    """Replace mutable progress while keeping experiment evidence immutable."""
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def audit_training(root: Path, trigger: str, total: int) -> dict:
    from capx.rl.capsule.group import GroupAssemblyResult
    from capx.rl.capsule.schema import TaskInstanceV1
    from capx.rl.capsule.trainer import validate_group_provenance

    protocol = read(root / "protocol.json")
    result = read(root / "training_result.json")
    if result["status"] != "completed" or result["cumulative_group_count"] != total:
        raise RuntimeError("training did not complete the requested cumulative groups")
    if protocol["capsule"]["repair_trigger"] != trigger:
        raise RuntimeError("wrong training arm")
    if set(protocol["training_seeds"]) & set(range(201, 221)):
        raise RuntimeError("evaluation scenes overlap training ancestry")
    if total == 32:
        restored = read(root / "restore_evidence.json")
        if restored["restored_optimizer_step"] != result["optimizer_step_before"]:
            raise RuntimeError("continuation optimizer state mismatch")
    paths = sorted((root / "training/groups").glob("*.json"))
    if len(paths) != 16 or result["completed_group_count"] != 16:
        raise RuntimeError("each stage must complete exactly 16 groups")
    tasks = [json.loads(line) for line in (root / "dataset.seed_resolved.jsonl").read_text().splitlines()]
    tasks_by_seed = {task["environment_seed"]: task for task in tasks}
    mixed_triggered = 0
    replay_attempts = 0
    for path in paths:
        artifact = read(path)
        assembly = GroupAssemblyResult.from_dict(artifact["assembly"])
        seed = assembly.group.environment_seed
        task_data = tasks_by_seed[seed]
        # The scheduler adds a deterministic collection identity for each epoch/task.
        task_index = protocol["new_training_seeds"].index(seed)
        task = TaskInstanceV1(**task_data,
            **{name: protocol["task"][name] for name in ("environment", "api", "privilege")},
            metadata={
            "capsule_collection_id": f"epoch-00000000:task-{task_index:08d}",
        })
        validate_group_provenance(task, assembly, trigger)
        successes = sum(r.binary_reward == 1.0 for r in assembly.base_results[:7])
        mixed_triggered += int(0 < successes < 7 and bool(assembly.repair_attempts))
        replays = list(assembly.base_results)
        for attempt in assembly.repair_attempts:
            replays.extend(r for r in (attempt.pt_result, attempt.revision_result) if r is not None)
        replay_attempts += sum(r.attempts for r in replays)
    if trigger == "never" and result["controller_usage"]["requests"] != 0:
        raise RuntimeError("ordinary GRPO called the Controller")
    summary = read(root / "summary.json")
    return {
        "training_root": str(root), "checkpoint": result["checkpoint"],
        "checkpoint_sha256": result["checkpoint_sha256"],
        "actor_updates": result["actor_updates"],
        "skipped_actor_updates": result["skipped_actor_updates"],
        "discarded_groups": result["discarded_groups"],
        "mixed_groups_repaired": mixed_triggered,
        "repair_triggered_groups": summary["repair_triggered_groups"],
        "guided_groups": summary["capsule_repaired_groups"],
        "initial_base_success_rate": summary["initial_base_success_rate"],
        "accepted_group_replay_attempts": replay_attempts,
        "program_usage": result["program_usage"],
        "controller_usage": result["controller_usage"],
        "training_wall_seconds": result["training_wall_seconds"],
    }


def compare_evaluations(roots: dict[str, Path]) -> dict:
    comparisons = {}
    for total in (16, 32):
        records = {
            arm: [verify_replay(roots[f"{arm}{total}"], "trained_lora", i) for i in range(80)]
            for arm in ("A", "C")
        }
        for a, c in zip(records["A"], records["C"], strict=True):
            if not a["initial_state_sha256"] or not c["initial_state_sha256"]:
                raise RuntimeError("evaluation is missing physical reset evidence")
            for key in ("ordinal", "environment_seed", "generation_seed", "initial_state_sha256"):
                if a[key] != c[key]:
                    raise RuntimeError(f"A/C evaluation pairing mismatch: {key}")

        counts = {arm: Counter(r["outcome"] for r in rows) for arm, rows in records.items()}
        scene_deltas = [
            sum(int(c["outcome"] == "success") - int(a["outcome"] == "success")
                for a, c in zip(records["A"], records["C"], strict=True)
                if a["environment_seed"] == seed) / 4
            for seed in range(201, 221)
        ]
        rng = random.Random(20260909)
        boot = sorted(sum(rng.choices(scene_deltas, k=20)) / 20 for _ in range(10000))
        comparisons[str(total)] = {
            "A": {"outcomes": dict(counts["A"]), "success_rate": counts["A"]["success"] / 80},
            "C": {"outcomes": dict(counts["C"]), "success_rate": counts["C"]["success"] / 80},
            "success_rate_delta_C_minus_A": sum(scene_deltas) / 20,
            "paired_scene_bootstrap_95_ci": [boot[249], boot[9749]],
            "scope": "one training replicate; interval measures scene uncertainty only",
        }
    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--verl", type=Path, required=True)
    parser.add_argument("--staging-checkpoint-root", type=Path)
    args = parser.parse_args()
    if not args.run_id or Path(args.run_id).name != args.run_id:
        parser.error("run-id must be a directory name")
    if not os.environ.get("CAPX_CONTROLLER_API_KEY"):
        parser.error("CAPX_CONTROLLER_API_KEY is required for arm C")
    project = Path(__file__).resolve().parents[2]
    os.chdir(project)
    root = project / "artifacts" / args.run_id
    root.mkdir(parents=True, exist_ok=True)
    logs = project / "outputs" / args.run_id
    logs.mkdir(parents=True, exist_ok=True)
    os.environ.update({
        "MUJOCO_GL": "egl", "JAX_PLATFORMS": "cpu", "JAX_PLATFORM_NAME": "cpu",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "OMP_NUM_THREADS": "4",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    })
    source_paths = [Path(__file__), project / "capx/rl/capsule/group.py",
                    project / "capx/rl/capsule/trainer.py", project / "capx/rl/capsule/server_factory.py",
                    project / "capx/rl/capsule/controller.py", project / "scripts/capsule_rl/common.py",
                    project / "scripts/capsule_rl/train_cube_stack.py", project / "scripts/capsule_rl/evaluate_cube_stack.py"]
    manifest = {
        "run_id": args.run_id, "model": str(args.model.resolve()), "verl": str(args.verl.resolve()),
        "training_seeds": list(range(5, 37)), "evaluation_seeds": list(range(201, 221)),
        "samples_per_scene": 4, "primary_comparison_groups": 32,
        "source_sha256": {str(p.relative_to(project)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
    }
    manifest_path = root / "experiment.json"
    if manifest_path.exists() and read(manifest_path) != manifest:
        raise RuntimeError("experiment code or inputs changed; choose a new run-id")
    if not manifest_path.exists():
        atomic_write_json(manifest_path, manifest)
    status = {"status": "running", "training": {}, "evaluation": {}}

    def run(module: str, arguments: list[str], label: str, *, arm: str = "C") -> None:
        environment = dict(os.environ)
        if arm == "A":
            environment.pop("CAPX_CONTROLLER_API_KEY", None)
        print(f"Starting {label}", flush=True)
        status["active_phase"] = label
        write_status(root / "status.json", status)
        with (logs / f"{label}.log").open("a") as log:
            subprocess.run([sys.executable, "-m", module, *arguments], env=environment,
                           stdout=log, stderr=subprocess.STDOUT, check=True)

    def ready(port: int) -> bool:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/docs", timeout=2):
                return True
        except OSError:
            return False

    services = []
    try:
        training_roots = {}
        for total in (16, 32):
            for arm, trigger in (("A", "never"), ("C", "any_failed")):
                label = f"{arm}{total}"
                training_root = root / label
                training_roots[label] = training_root
                if not (training_root / "protocol.json").exists():
                    arguments = ["prepare", "--root", str(training_root), "--model", str(args.model),
                                 "--verl", str(args.verl), "--repair-trigger", trigger,
                                 "--seeds", ",".join(map(str, range(5, 21) if total == 16 else range(21, 37)))]
                    if total == 32:
                        arguments += ["--resume-from", str(training_roots[f"{arm}16"])]
                    # Keep one full checkpoint on disk and three in explicit temporary storage.
                    if args.staging_checkpoint_root and label != "A32":
                        arguments += ["--checkpoint-root", str(args.staging_checkpoint_root / args.run_id / label)]
                    run("scripts.capsule_rl.train_cube_stack", arguments, f"{label}_prepare", arm=arm)
                if not (training_root / "training_result.json").exists():
                    runtime = yaml.safe_load((training_root / "runtime.yaml").read_text())["runtime"]
                    storage = Path(runtime.get("checkpoint_root", training_root))
                    storage.mkdir(parents=True, exist_ok=True)
                    if shutil.disk_usage(storage).free < 17 * 1024**3:
                        raise RuntimeError(f"insufficient checkpoint space for {label}")
                    run("scripts.capsule_rl.train_cube_stack", ["train", "--root", str(training_root)], f"{label}_train", arm=arm)
                status["training"][label] = audit_training(training_root, trigger, total)
                write_status(root / "status.json", status)

        evaluation_roots = {}
        # Generate before loading the perception services to keep GPU use bounded.
        for label, training_root in training_roots.items():
            evaluation = root / f"eval_{label}"
            evaluation_roots[label] = evaluation
            if not (evaluation / "protocol.json").exists():
                run("scripts.capsule_rl.evaluate_cube_stack", ["prepare", "--root", str(evaluation),
                    "--training-root", str(training_root), "--seeds", ",".join(map(str, range(201, 221))),
                    "--samples-per-seed", "4"], f"{label}_eval_prepare")
            run("scripts.capsule_rl.evaluate_cube_stack", ["generate", "--root", str(evaluation),
                "--policy", "trained_lora"], f"{label}_generate")
        for service, port in (("sam3", 8114), ("contact_graspnet", 8115), ("pyroki", 8116)):
            if not ready(port):
                with (logs / f"{service}.log").open("a") as log:
                    services.append(subprocess.Popen([sys.executable, "-m", f"capx.serving.launch_{service}_server"],
                        stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL))
                deadline = time.monotonic() + 600
                while not ready(port):
                    if time.monotonic() > deadline:
                        raise RuntimeError(f"{service} readiness timed out")
                    time.sleep(5)
        for label, evaluation in evaluation_roots.items():
            run("scripts.capsule_rl.evaluate_cube_stack", ["evaluate", "--root", str(evaluation),
                "--policy", "trained_lora"], f"{label}_evaluate")
            records = [verify_replay(evaluation, "trained_lora", i) for i in range(80)]
            status["evaluation"][label] = {"successes": sum(r["outcome"] == "success" for r in records), "total": 80}
        comparison = compare_evaluations(evaluation_roots)
        atomic_write_json(root / "comparison.json", comparison)
        status.update({"status": "completed", "active_phase": None, "comparison": comparison})
    except BaseException as error:
        status.update({"status": "failed", "error_type": type(error).__name__})
        raise
    finally:
        write_status(root / "status.json", status)
        for process in services:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    main()
