"""Continue Spill Wipe critique training by 16 groups and evaluate paired scenes."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import yaml

from scripts.capsule_rl.common import atomic_write_json
from scripts.capsule_rl.evaluate_cube_stack import read, sha256
from scripts.capsule_rl.run_cube_stack_ac import audit_training, compare_evaluations, write_status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--baseline-evaluation", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--groups", type=int, choices=(32, 48), default=32)
    args = parser.parse_args()
    total = args.groups
    parent_total = total - 16
    parent_label, label = f"C{parent_total}", f"C{total}"
    new_seeds = list(range(5 + parent_total, 5 + total))
    project = Path(__file__).resolve().parents[2]
    os.chdir(project)
    if Path(args.run_id).name != args.run_id or args.run_id in ("", ".", ".."):
        parser.error("run-id must be a directory name")
    parent, baseline = args.parent.resolve(), args.baseline_evaluation.resolve()
    previous = read(parent / "protocol.json")
    frozen = read(baseline / "protocol.json")
    if (previous["cumulative_group_count"] != parent_total
            or previous["task"]["environment"] != "robosuite_spill_wipe"
            or previous["capsule"]["repair_trigger"] != "any_failed"
            or previous["training_seeds"] != list(range(5, 5 + parent_total))):
        raise ValueError(f"parent must be Spill Wipe {parent_label} with contiguous seeds from 5")
    if (Path(frozen["training_root"]).resolve() != parent
            or frozen["seeds"] != list(range(201, 221))
            or frozen["total_samples_per_policy"] != 80):
        raise ValueError("baseline must be the parent run's frozen 80-sample evaluation")
    audit_training(parent, "any_failed", parent_total)
    root = project / "artifacts" / args.run_id
    logs = project / "outputs" / args.run_id
    root.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    training, evaluation = root / label, root / f"eval_{label}"
    manifest = {"parent": str(parent), "baseline_evaluation": str(baseline),
                "parent_protocol_sha256": sha256(parent / "protocol.json"),
                "parent_result_sha256": sha256(parent / "training_result.json"),
                "baseline_protocol_sha256": sha256(baseline / "protocol.json"),
                "new_training_seeds": new_seeds, "cumulative_groups": total,
                "source_sha256": {name: sha256(project / name) for name in (
                    "scripts/spill_wipe/run_c32.py", "scripts/capsule_rl/train_cube_stack.py",
                    "scripts/capsule_rl/evaluate_cube_stack.py", "capx/rl/capsule/server_factory.py",
                    "capx/rl/capsule/group.py", "capx/rl/capsule/trainer.py")}}
    if (root / "experiment.json").exists():
        if read(root / "experiment.json") != manifest:
            raise RuntimeError("continuation inputs changed; use a fresh run-id")
    else:
        atomic_write_json(root / "experiment.json", manifest)
    os.environ.update({"MUJOCO_GL": "egl", "JAX_PLATFORMS": "cpu", "JAX_PLATFORM_NAME": "cpu",
                       "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "OMP_NUM_THREADS": "4",
                       "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "CAPX_FORCE_STREAMING_CHAT_COMPLETIONS": "0"})
    status = {"status": "running", "phase": "prepare"}

    def run(module: str, arguments: list[str], label: str) -> None:
        status["phase"] = label
        write_status(root / "status.json", status)
        print(label, flush=True)
        with (logs / f"{label}.log").open("a") as log:
            subprocess.run([sys.executable, "-m", module, *arguments], check=True,
                           stdout=log, stderr=subprocess.STDOUT)

    def ready(port: int) -> bool:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/docs", timeout=3):
                return True
        except OSError:
            return False

    owned = []
    try:
        if not ready(8116):
            raise RuntimeError("start Pyroki on port 8116 before continuation")
        if not (training / "training_result.json").exists():
            if not os.environ.get("CAPX_CONTROLLER_API_KEY"):
                raise RuntimeError("CAPX_CONTROLLER_API_KEY is required for critique training")
            parent_checkpoint = Path(read(parent / "training_result.json")["checkpoint"])
            checkpoint_bytes = sum(p.stat().st_size for p in parent_checkpoint.rglob("*") if p.is_file())
            if shutil.disk_usage(project).free < checkpoint_bytes + 1024**3:
                raise RuntimeError("insufficient checkpoint disk space")
            if not (training / "protocol.json").exists():
                runtime = yaml.safe_load((parent / "runtime.yaml").read_text())["runtime"]
                run("scripts.capsule_rl.train_cube_stack", ["prepare", "--task", "spill_wipe",
                    "--root", str(training), "--resume-from", str(parent),
                    "--model", previous["program_service"]["model"],
                    "--verl", runtime["verl_source_path"], "--repair-trigger", "any_failed",
                    "--controller-model", previous["controller_service"]["model"],
                    "--controller-endpoint", previous["controller_service"]["endpoint"],
                    "--seeds", ",".join(map(str, new_seeds))], f"prepare_{label}")
            run("scripts.capsule_rl.train_cube_stack", ["train", "--root", str(training)], f"train_{label}")
        status["training"] = audit_training(training, "any_failed", total)
        restored = read(training / "restore_evidence.json")
        if restored["restored_optimizer_step"] != read(parent / "training_result.json")["optimizer_step_after"]:
            raise RuntimeError("continuation optimizer state differs from parent")
        os.environ.pop("CAPX_CONTROLLER_API_KEY", None)
        if not (evaluation / "protocol.json").exists():
            run("scripts.capsule_rl.evaluate_cube_stack", ["prepare", "--root", str(evaluation),
                "--training-root", str(training), "--seeds", ",".join(map(str, frozen["seeds"])),
                "--samples-per-seed", "4"], "prepare_evaluation")
        run("scripts.capsule_rl.evaluate_cube_stack", ["generate", "--root", str(evaluation),
            "--policy", "trained_lora"], f"generate_{label}")
        if not ready(8114):
            with (logs / "sam3.log").open("a") as log:
                owned.append(subprocess.Popen([sys.executable, "-m", "capx.serving.launch_sam3_server"],
                    stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL))
            deadline = time.monotonic() + 600
            while not ready(8114):
                if time.monotonic() >= deadline:
                    raise RuntimeError("SAM3 readiness timed out")
                time.sleep(5)
        run("scripts.capsule_rl.evaluate_cube_stack", ["evaluate", "--root", str(evaluation),
            "--policy", "trained_lora"], f"evaluate_{label}")
        paired = compare_evaluations({f"A{total}": baseline, f"C{total}": evaluation}, (total,))[str(total)]
        comparison = {parent_label: paired["A"], label: paired["C"],
                      f"delta_{label}_minus_{parent_label}": paired["success_rate_delta_C_minus_A"],
                      "paired_scene_bootstrap_95_ci": paired["paired_scene_bootstrap_95_ci"],
                      "scope": f"adaptive continuation after inspecting {parent_label} on the same evaluation scenes"}
        if not (root / "comparison.json").exists():
            atomic_write_json(root / "comparison.json", comparison)
        elif read(root / "comparison.json") != comparison:
            raise RuntimeError("completed comparison changed")
        status.update(status="completed", phase=None, comparison=comparison)
    except BaseException as error:
        status.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_status(root / "status.json", status)
        for process in owned:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()
