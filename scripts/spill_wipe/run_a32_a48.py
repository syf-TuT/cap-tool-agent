"""Continue Spill Wipe A16 through A32 and A48, evaluating each checkpoint."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--baseline-evaluation", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    parent = args.parent.resolve()
    baseline = args.baseline_evaluation.resolve()
    for total in (32, 48):
        run_id = f"{args.run_id}_a{total}"
        subprocess.run(
            [sys.executable, "-m", "scripts.spill_wipe.run_c32", "--arm", "A",
             "--groups", str(total), "--parent", str(parent),
             "--baseline-evaluation", str(baseline), "--run-id", run_id],
            cwd=project, check=True,
        )
        parent = project / "artifacts" / run_id / f"A{total}"
        baseline = parent.parent / f"eval_A{total}"


if __name__ == "__main__":
    main()
