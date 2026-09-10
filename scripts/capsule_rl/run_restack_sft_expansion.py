"""Collect 128 positives, select an SFT epoch, then evaluate untouched Restack seeds."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

from scripts.capsule_rl.common import atomic_write_json
from scripts.capsule_rl.evaluate_cube_stack import read
from scripts.capsule_rl.run_cube_stack_ac import write_status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parents[2])
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), Path("scripts/capsule_rl/warmstart_cube_restack.py"),
               Path("scripts/capsule_rl/evaluate_restack_sft.py"),
               Path("scripts/capsule_rl/evaluate_cube_stack.py")]
    protocol = {"target": 128, "budget": 512, "training_seeds": [5, 132],
                "validation_seeds": [201, 220], "test_seeds": [301, 340],
                "epochs": [1, 3, 6], "selection": "validation successes; earlier epoch breaks ties",
                "model": str(args.model.resolve()),
                "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    if (root / "protocol.json").exists():
        if read(root / "protocol.json") != protocol:
            raise RuntimeError("experiment inputs changed; choose a fresh root")
    else:
        atomic_write_json(root / "protocol.json", protocol)
    status = {"status": "running", "validation": {}}

    def run(module: str, arguments: list[str], phase: str, teacher: bool = False) -> None:
        status["phase"] = phase
        write_status(root / "status.json", status)
        environment = dict(os.environ)
        if not teacher:
            environment.pop("CAPX_CONTROLLER_API_KEY", None)
        with (root / f"{phase}.log").open("a") as log:
            subprocess.run([sys.executable, "-u", "-m", module, *arguments], check=True,
                           env=environment, stdout=log, stderr=subprocess.STDOUT)

    try:
        training = root / "training"
        if not (training / "sft_result.json").exists():
            run("scripts.capsule_rl.warmstart_cube_restack", ["--root", str(training),
                "--model", str(args.model.resolve()), "--target", "128", "--budget", "512",
                "--seed-count", "128", "--epochs", "6"], "collect_train", teacher=True)
        for epoch in (1, 3, 6):
            evaluation = root / f"validation_epoch_{epoch}"
            if not (evaluation / "summary.json").exists():
                run("scripts.capsule_rl.evaluate_restack_sft", ["--root", str(evaluation),
                    "--training-root", str(training / f"epoch_{epoch}"), "--sft-only"],
                    f"validation_epoch_{epoch}")
            status["validation"][str(epoch)] = read(evaluation / "summary.json")["results"]["sft_lora"]
        selected = max((1, 3, 6), key=lambda e: (status["validation"][str(e)]["successes"], -e))
        selection = {"epoch": selected, "validation": status["validation"]}
        if not (root / "selection.json").exists():
            atomic_write_json(root / "selection.json", selection)
        elif read(root / "selection.json") != selection:
            raise RuntimeError("selection changed")
        evaluation = root / "test"
        if not (evaluation / "summary.json").exists():
            run("scripts.capsule_rl.evaluate_restack_sft", ["--root", str(evaluation),
                "--training-root", str(training / f"epoch_{selected}"), "--seed-start", "301",
                "--seed-count", "40", "--sft-only"], "test")
        result = read(evaluation / "summary.json")["results"]["sft_lora"]
        status.update(status="completed", phase=None, selected_epoch=selected, test=result,
                      target_observed=result["success_rate"] >= 0.05 and result["successful_scenes"] > 1)
        atomic_write_json(root / "summary.json", status)
    except BaseException as error:
        status.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        write_status(root / "status.json", status)


if __name__ == "__main__":
    main()
