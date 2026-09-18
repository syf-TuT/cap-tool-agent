"""Evaluate SFT Restack and base actors on paired held-out privileged scenes."""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

from scripts.capsule_rl.common import atomic_write_json
from scripts.capsule_rl.evaluate_cube_stack import evaluate, generate, read, sha256, verify_replay
from scripts.capsule_rl.run_cube_stack_ac import write_status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=201)
    parser.add_argument("--seed-count", type=int, default=20)
    parser.add_argument("--sft-only", action="store_true")
    args = parser.parse_args()
    if args.seed_start < 0 or args.seed_count < 1:
        parser.error("require nonnegative seed-start and positive seed-count")
    seeds = list(range(args.seed_start, args.seed_start + args.seed_count))
    total = len(seeds) * 4
    policies = ("sft_lora",) if args.sft_only else ("sft_lora", "base")
    os.chdir(Path(__file__).resolve().parents[2])
    os.environ.update({"MUJOCO_GL": "egl", "JAX_PLATFORMS": "cpu",
                       "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "OMP_NUM_THREADS": "4"})
    os.environ.pop("CAPX_CONTROLLER_API_KEY", None)
    root = args.root.resolve()
    training = args.training_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    config = Path("env_configs/cube_restack/capsule_rl/"
                  "franka_robosuite_cube_restack_privileged_clean_replay.yaml").resolve()
    from capx.rl.capsule.server_factory import YamlEnvironmentFactory

    env = YamlEnvironmentFactory(str(config))(None)
    try:
        observation, _ = env.reset(seed=seeds[0])
        messages = observation["full_prompt"]
    finally:
        env.close()
    protocol = {"mode": "privileged_restack_sft_heldout", "training_root": str(training),
                "base_model": str(Path(read(training / "sft_result.json")["base_model"]).resolve()),
                "environment_config": str(config), "environment_config_sha256": sha256(config),
                "seeds": seeds, "samples_per_seed": 4, "policies": list(policies),
                "total_samples_per_policy": total,
                "generation_seed_offset": 202609100000 + args.seed_start * 4,
                "system_prompt": messages[0]["content"],
                "user_prompt": messages[1]["content"][0]["text"],
                "sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20,
                             "repetition_penalty": 1.1, "max_tokens": 4096},
                "sft_result_sha256": sha256(training / "sft_result.json")}
    if (root / "protocol.json").exists():
        if read(root / "protocol.json") != protocol:
            raise RuntimeError("evaluation protocol changed")
    else:
        atomic_write_json(root / "protocol.json", protocol)
    status = {"status": "running", "results": {}}
    try:
        for policy in policies:
            status["phase"] = policy + "_generate"
            write_status(root / "status.json", status)
            generate(root, policy)
            import torch

            gc.collect()
            torch.cuda.empty_cache()
            status["phase"] = policy + "_replay"
            write_status(root / "status.json", status)
            evaluate(root, policy)
            records = [verify_replay(root, policy, i) for i in range(total)]
            successes = sum(r["outcome"] == "success" for r in records)
            status["results"][policy] = {"successes": successes, "total": total,
                                        "success_rate": successes / total,
                                        "successful_scenes": len({r["environment_seed"] for r in records
                                                                  if r["outcome"] == "success"})}
            write_status(root / "status.json", status)
        for i in range(total if len(policies) == 2 else 0):
            a, b = (verify_replay(root, policy, i) for policy in ("sft_lora", "base"))
            for key in ("initial_state_sha256", "environment_seed", "generation_seed"):
                if a[key] != b[key]:
                    raise RuntimeError("paired evaluation mismatch: " + key)
        status.update(status="completed", phase=None)
        atomic_write_json(root / "summary.json", status)
    except BaseException as error:
        status.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        write_status(root / "status.json", status)


if __name__ == "__main__":
    main()
