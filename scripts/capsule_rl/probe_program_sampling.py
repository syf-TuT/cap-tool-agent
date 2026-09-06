"""Measure frozen Program sampling through the same VeRL worker and clean replay as training."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from omegaconf import OmegaConf

from capx.rl.capsule.program_protocol import program_sampling
from capx.rl.capsule.server_factory import _apply_chat_template, _load_resolved_verl_config
from scripts.capsule_rl.common import atomic_write_json, load_and_validate_server_config
from scripts.capsule_rl.server_adapter import ConcreteGateRuntime


def run(config_path: Path, output_dir: Path, seeds: tuple[int, ...], samples: int) -> None:
    if samples < 1 or not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("provide distinct seeds and a positive sample count")
    output_dir.mkdir(parents=True, exist_ok=False)
    config = load_and_validate_server_config(config_path, check_runtime_paths=True)
    if not program_sampling(config):
        raise ValueError("this probe requires an explicit program_service sampling protocol")
    runtime = ConcreteGateRuntime(config)
    tasks = {task.environment_seed: task for task in runtime._tasks()}
    if set(seeds) - tasks.keys():
        raise ValueError("probe seeds must be present in runtime.dataset_path")
    project_root = Path(config["runtime"]["project_root"])
    resolved = _load_resolved_verl_config(config, project_root)
    OmegaConf.save(resolved, output_dir / "effective_verl.yaml")
    session = runtime._open_collection_session(tasks[seeds[0]])
    groups = []
    try:
        before = session.workers.optimizer_step()
        if before != 0:
            raise RuntimeError("base sampling probe requires an actor with optimizer step zero")
        prompt = tasks[seeds[0]].prompt
        tokens = _apply_chat_template(
            session.generator.tokenizer, prompt, session.generator.system_prompt
        )
        atomic_write_json(output_dir / "protocol.json", {
            "program_service": dict(config["program_service"]),
            "prompt": prompt,
            "prompt_tokens": tokens,
            "prompt_token_count": len(tokens),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "response_token_limit": session.generator.response_token_limit,
            "effective_rollout": OmegaConf.to_container(resolved.actor_rollout_ref.rollout),
            "worker_sampling": session.workers.program_sampling_evidence,
            "optimizer_step_before": before,
            "verl_provenance": session.workers.verl_provenance(),
        })
        for seed in seeds:
            group = {"seed": seed, "results": []}
            groups.append(group)
            for index in range(samples):
                candidate = session.generator.generate(
                    tasks[seed].prompt, f"{config['runtime']['run_id']}:seed-{seed}:base-{index}"
                )
                result = session.clean_evaluator(tasks[seed], candidate)
                group["results"].append(result.to_dict())
                print(json.dumps({
                    "seed": seed, "sample": index, "outcome": result.outcome.value,
                    "reward": result.binary_reward, "error": result.error_message,
                }), flush=True)
                atomic_write_json(
                    output_dir / f"sample_{seed:03d}_{index:03d}.json",
                    {"seed": seed, "sample": index, "result": result.to_dict()},
                )
            group["outcome_counts"] = dict(Counter(r["outcome"] for r in group["results"]))
            group["successes"] = group["outcome_counts"].get("success", 0)
        after = session.workers.optimizer_step()
        if after != before:
            raise RuntimeError("actor optimizer changed during frozen sampling")
    finally:
        session.close()
    counts = Counter(r["outcome"] for group in groups for r in group["results"])
    payload = {
        "mode": "frozen_program_sampling",
        "config_path": str(config_path.resolve()),
        "samples": samples * len(seeds),
        "successes": counts.get("success", 0),
        "outcome_counts": dict(counts),
        "all_negative_groups": sum(group["successes"] == 0 for group in groups),
        "optimizer_step_before": before,
        "optimizer_step_after": after,
        "ray_release": session.workers.ray_release_evidence(),
        "groups": groups,
    }
    atomic_write_json(output_dir / "summary.json", payload)
    print(json.dumps({k: v for k, v in payload.items() if k != "groups"}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="5,6,7,8")
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()
    run(args.config, args.output_dir, tuple(int(s) for s in args.seeds.split(",")), args.samples)


if __name__ == "__main__":
    main()
