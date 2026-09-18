"""Train privileged Restack from the base model and evaluate transfer to visual Stack."""

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

from scripts.capsule_rl.common import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--verl", type=Path, required=True)
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in (".", ".."):
        parser.error("run-id must be a directory name")
    if not os.environ.get("CAPX_CONTROLLER_API_KEY"):
        raise ValueError("CAPX_CONTROLLER_API_KEY must be set")
    project = Path(__file__).resolve().parents[2]
    os.chdir(project)
    os.environ.update({
        "MUJOCO_GL": "egl", "JAX_PLATFORMS": "cpu", "JAX_PLATFORM_NAME": "cpu",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "OMP_NUM_THREADS": "4",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    })
    logs = project / "outputs" / args.run_id
    logs.mkdir(parents=True, exist_ok=False)
    training = project / "artifacts" / f"cube_restack_capsule_rl_{args.run_id}"
    evaluation = project / "artifacts" / f"cube_stack_transfer_from_restack_{args.run_id}"
    if training.exists() or evaluation.exists():
        raise FileExistsError("use a new run ID")
    if shutil.disk_usage(project).free < 17 * 1024**3:
        raise RuntimeError("at least 17 GiB free is required for a full checkpoint")
    state = {"training_root": str(training), "evaluation_root": str(evaluation)}
    owned = []

    def save_status() -> None:
        staging = logs / "status.pending.json"
        atomic_write_json(staging, state)
        staging.replace(logs / "status.json")

    def run(module: str, arguments: list[str], phase: str) -> None:
        state.update(status="running", phase=phase)
        save_status()
        print(f"Starting {phase}", flush=True)
        with (logs / f"{phase}.log").open("w") as log:
            subprocess.run([sys.executable, "-m", module, *arguments],
                           stdout=log, stderr=subprocess.STDOUT, check=True)

    def ready(port: int) -> bool:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/docs", timeout=2):
                return True
        except OSError:
            return False

    def service(name: str, port: int) -> None:
        if ready(port):
            return
        with (logs / f"{name}.log").open("w") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", f"capx.serving.launch_{name}_server"],
                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            )
        owned.append(process)
        deadline = time.monotonic() + 600
        while not ready(port):
            if process.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError(f"{name} failed readiness; inspect {logs / (name + '.log')}")
            time.sleep(5)

    try:
        service("pyroki", 8116)
        run("scripts.capsule_rl.train_cube_restack", [
            "prepare", "--root", str(training), "--model", str(args.model.resolve()),
            "--verl", str(args.verl.resolve()), "--seeds", ",".join(map(str, range(5, 37))),
        ], "prepare")
        run("scripts.capsule_rl.train_cube_restack", ["train", "--root", str(training)], "train")
        result = json.loads((training / "training_result.json").read_text())
        if (result["status"] != "completed" or result["completed_group_count"] != 32
                or result["optimizer_step_delta"] < 1):
            raise RuntimeError("training did not complete 32 groups with a model update")
        os.environ.pop("CAPX_CONTROLLER_API_KEY", None)
        service("sam3", 8114)
        service("contact_graspnet", 8115)
        run("scripts.capsule_rl.evaluate_cube_stack", [
            "prepare", "--root", str(evaluation), "--training-root", str(training),
            "--seeds", ",".join(map(str, range(45, 65))), "--samples-per-seed", "4",
        ], "evaluation_prepare")
        for policy in ("base", "trained_lora"):
            for phase in ("generate", "evaluate"):
                run("scripts.capsule_rl.evaluate_cube_stack", [
                    phase, "--root", str(evaluation), "--policy", policy,
                ], f"{policy}_{phase}")
        run("scripts.capsule_rl.evaluate_cube_stack", [
            "summarize", "--root", str(evaluation),
        ], "summary")
        state.update(status="completed", summary=str(evaluation / "summary.json"))
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save_status()
        for process in owned:
            process.terminate()
        for process in owned:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()
