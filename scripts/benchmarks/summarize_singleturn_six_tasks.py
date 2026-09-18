"""Audit the six-task single-turn benchmark using canonical trial records."""
import argparse
import json
from collections import Counter
from pathlib import Path

TASKS = ("cube_lift", "cube_stack", "cube_restack", "spill_wipe",
         "two_arm_lift", "two_arm_handover")


def summarize(root):
    result = {"root": str(root), "tasks": {}, "complete": True}
    for task in TASKS:
        records = sorted(
            (json.loads(path.read_text()) for path in
             (root / "Qwen2.5-Coder-7B-Instruct" / task).glob("trial_*_result.json")),
            key=lambda row: row["trial"],
        )
        finished = [row for row in records if row["run_outcome"] != "running"]
        successes = [row["trial"] for row in finished
                     if row["run_outcome"] == "finished" and row["task_completed"]
                     and row["sandbox_rc"] == 0]
        complete = [row["trial"] for row in finished] == list(range(1, 21))
        result["complete"] &= complete
        result["tasks"][task] = {
            "finished": len(finished), "successes": len(successes),
            "success_rate": len(successes) / 20 if complete else None,
            "success_seeds": successes,
            "outcomes": dict(Counter(row["run_outcome"] for row in finished)),
            "llm_calls": sum(row["llm"]["call_count"] for row in finished),
            "llm_retries": sum(row["llm"]["retry_count"] for row in finished),
            "program_errors": sum(row["sandbox_rc"] not in (None, 0) for row in finished),
            "records": records,
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = summarize(args.root)
    (args.root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    for data in result["tasks"].values():
        del data["records"]
    print(json.dumps(result, indent=2))
