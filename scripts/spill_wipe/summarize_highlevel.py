"""Summarize canonical per-trial records without treating code execution as task success."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def summarize(root: Path) -> dict:
    result = {"root": str(root), "conditions": {}}
    for condition in ("privileged", "nonprivileged"):
        directory = root / "Qwen2.5-Coder-7B-Instruct" / condition
        records = [json.loads(p.read_text()) for p in directory.glob("trial_*_result.json")]
        records.sort(key=lambda row: row["trial"])
        finished = [row for row in records if row["run_outcome"] != "running"]
        successes = [row for row in finished if row["run_outcome"] == "finished"
                     and row["task_completed"] and row["sandbox_rc"] == 0]
        result["conditions"][condition] = {
            "attempted": len(records), "finished": len(finished),
            "successes": len(successes),
            "success_rate": len(successes) / len(finished) if finished else None,
            "task_completed_count": sum(bool(row["task_completed"]) for row in finished),
            "program_error_count": sum(row["sandbox_rc"] not in (None, 0) for row in finished),
            "llm_call_count": sum(row["llm"]["call_count"] for row in finished),
            "llm_retry_count": sum(row["llm"]["retry_count"] for row in finished),
            "video_total_count": len(list(directory.glob("trial_*/video_combined.mp4"))),
            "mean_reward": sum(row["reward"] or 0 for row in finished) / len(finished)
            if finished else None,
            "outcomes": dict(Counter(row["run_outcome"] for row in finished)),
            "success_seeds": [row["trial"] for row in successes],
            "records": records,
        }
    result["complete"] = all(
        sorted(row["trial"] for row in data["records"]) == list(range(1, 101))
        and data["finished"] == 100 for data in result["conditions"].values()
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    summary = summarize(args.root)
    (args.root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    compact = {"complete": summary["complete"]}
    for condition, data in summary["conditions"].items():
        compact[condition] = {key: value for key, value in data.items() if key != "records"}
    print(json.dumps(compact, indent=2))
