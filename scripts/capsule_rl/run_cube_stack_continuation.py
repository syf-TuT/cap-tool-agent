"""Continue a 32-group Cube Stack run to 64 and 96 groups on SeeTaCloud."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--baseline-evaluation", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    os.chdir(project)
    parent = args.parent.resolve()
    baseline = args.baseline_evaluation.resolve()
    initial = json.loads((parent / "protocol.json").read_text())
    if initial.get("cumulative_group_count", initial["group_count"]) != 32:
        raise ValueError("this launcher starts from a completed 32-group run")
    frozen = json.loads((baseline / "protocol.json").read_text())
    if frozen["seeds"] != list(range(45, 65)) or frozen["total_samples_per_policy"] != 80:
        raise ValueError("baseline must be the frozen 80-sample seed 45-64 evaluation")
    if frozen["base_model"] != initial["program_service"]["model"]:
        raise ValueError("baseline and parent must use the same base model")
    if not os.environ.get("CAPX_CONTROLLER_API_KEY"):
        raise ValueError("CAPX_CONTROLLER_API_KEY must be set in the environment")
    os.environ.update({
        "MUJOCO_GL": "egl", "JAX_PLATFORMS": "cpu", "JAX_PLATFORM_NAME": "cpu",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "OMP_NUM_THREADS": "4",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    })
    log_root = project / "outputs" / args.run_id
    log_root.mkdir(parents=True, exist_ok=False)
    results = {"baseline_summary": str(baseline / "summary.json"), "stages": []}

    def run(module: str, arguments: list[str], log_name: str) -> None:
        print(f"Running {module} {' '.join(arguments)}", flush=True)
        with (log_root / log_name).open("w") as log:
            subprocess.run([sys.executable, "-m", module, *arguments],
                           stdout=log, stderr=subprocess.STDOUT, check=True)

    def service_ready(port: int) -> bool:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/docs", timeout=2):
                return True
        except OSError:
            return False

    for total, seeds in ((64, list(range(37, 45)) + list(range(65, 89))),
                         (96, list(range(89, 121)))):
        root = project / "artifacts" / f"cube_stack_capsule_rl_g{total}_{args.run_id}"
        evaluation = project / "artifacts" / f"cube_stack_nonprivileged_g{total}_{args.run_id}"
        if root.exists() or evaluation.exists():
            raise FileExistsError("use a new run ID; existing stages must not be overwritten")
        if shutil.disk_usage(project).free < 17 * 1024**3:
            raise RuntimeError(f"insufficient free disk for the {total}-group checkpoint")
        run("scripts.capsule_rl.train_cube_stack", [
            "prepare", "--root", str(root), "--model", frozen["base_model"],
            "--verl", str(project / ".codex-downloads/verl-v0.6.1"),
            "--resume-from", str(parent), "--seeds", ",".join(map(str, seeds)),
        ], f"g{total}_prepare.log")
        run("scripts.capsule_rl.train_cube_stack", ["train", "--root", str(root)],
            f"g{total}_train.log")
        result = json.loads((root / "training_result.json").read_text())
        if result["cumulative_group_count"] != total or result["status"] != "completed":
            raise RuntimeError("training did not complete the intended cumulative group count")

        # Preserve the exact frozen protocol and base-policy evidence; only the new
        # trained policy receives a new identity bound to its actual parent chain.
        evaluation.mkdir()
        shutil.copy2(baseline / "protocol.json", evaluation / "protocol.json")
        for folder in ("generations", "evaluation"):
            shutil.copytree(baseline / folder / "base", evaluation / folder / "base")
        run("scripts.capsule_rl.evaluate_cube_stack", [
            "generate", "--root", str(evaluation), "--policy", "trained_lora",
            "--training-root", str(root),
        ], f"g{total}_generate.log")
        owned_services = []
        try:
            for service, port in (("sam3", 8114), ("contact_graspnet", 8115), ("pyroki", 8116)):
                if not service_ready(port):
                    with (log_root / f"g{total}_{service}.log").open("w") as log:
                        process = subprocess.Popen(
                            [sys.executable, "-m", f"capx.serving.launch_{service}_server"],
                            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                        )
                    owned_services.append(process)
                deadline = time.monotonic() + 300
                while not service_ready(port):
                    if time.monotonic() > deadline:
                        raise RuntimeError(f"{service} did not become ready")
                    time.sleep(5)
            run("scripts.capsule_rl.evaluate_cube_stack", [
                "evaluate", "--root", str(evaluation), "--policy", "trained_lora",
            ], f"g{total}_evaluate.log")
            run("scripts.capsule_rl.evaluate_cube_stack", [
                "summarize", "--root", str(evaluation),
            ], f"g{total}_summary.log")
        finally:
            for process in owned_services:
                process.terminate()
            for process in owned_services:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        summary = json.loads((evaluation / "summary.json").read_text())
        results["stages"].append({
            "cumulative_groups": total, "training_root": str(root),
            "evaluation_root": str(evaluation),
            "optimizer_step_after": result["optimizer_step_after"],
            "successes": summary["trained_lora"]["successes"],
            "samples": summary["trained_lora"]["samples"],
            "success_rate": summary["trained_lora"]["success_rate"],
            "delta_vs_base": summary["success_rate_delta"],
        })
        (log_root / "comparison.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(results["stages"][-1]), flush=True)
        parent = root


if __name__ == "__main__":
    main()
