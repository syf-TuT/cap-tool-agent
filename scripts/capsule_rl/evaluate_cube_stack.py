"""Paired initial/LoRA evaluation on held-out non-privileged high-level Cube Stack.

Separate generation and replay phases let the actor release GPU memory before perception.
Each program executes once, without Controller assistance. Partial phases can be resumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from uuid import uuid4

import yaml

from scripts.capsule_rl.common import atomic_write_json


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_generation(root: Path, policy: str, ordinal: int) -> tuple[dict, Path]:
    directory = root / "generations" / policy
    identity = read(directory / "identity.json")
    if identity["protocol_sha256"] != sha256(root / "protocol.json"):
        raise RuntimeError("generation protocol changed")
    path = directory / f"sample_{ordinal:04d}.json"
    record = read(path)
    if (
        record["policy"] != policy
        or record["ordinal"] != ordinal
        or record["identity_sha256"] != sha256(directory / "identity.json")
    ):
        raise RuntimeError("generation identity mismatch")
    return record, path


def verify_replay(root: Path, policy: str, ordinal: int) -> dict:
    generated, generation_path = verify_generation(root, policy, ordinal)
    result = read(root / "evaluation" / policy / generation_path.name)
    if result["generation_sha256"] != sha256(generation_path):
        raise RuntimeError("evaluated generation changed")
    for key, value in generated.items():
        if result.get(key) != value:
            raise RuntimeError(f"replay generation identity mismatch: {key}")
    return result


def prepare(root: Path, training_root: Path, seeds: tuple[int, ...], samples: int) -> None:
    from capx.rl.capsule.server_factory import YamlEnvironmentFactory

    training = read(training_root / "protocol.json")
    if not seeds or min(seeds) < 0 or len(set(seeds)) != len(seeds) or samples < 1:
        raise ValueError("provide distinct non-negative seeds and a positive sample count")
    if set(seeds) & set(training["training_seeds"]):
        raise ValueError("evaluation seeds must be held out from training")
    project = Path(__file__).resolve().parents[2]
    config = yaml.safe_load(
        (project / ("env_configs/cube_stack/franka_robosuite_cube_stack.yaml")).read_text()
    )
    config["record_video"] = False
    config["num_workers"] = 1
    config["env"]["cfg"].update({"enable_render": False, "viser_debug": False})
    if config["env"]["cfg"]["privileged"] or config["env"]["cfg"]["apis"] != ["FrankaControlApi"]:
        raise RuntimeError("evaluation must use the non-privileged high-level API")
    root.mkdir(parents=True, exist_ok=False)
    config["output_dir"] = str(root / "environment_outputs")
    config_path = root / "environment.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    environment = YamlEnvironmentFactory(str(config_path))(None)
    try:
        observation, _ = environment.reset(seed=seeds[0])
        messages = observation["full_prompt"]
    finally:
        environment.close()
    atomic_write_json(
        root / "protocol.json",
        {
            "mode": "initial_vs_trained_nonprivileged_highlevel_cube_stack",
            "training_root": str(training_root),
            "training_protocol_sha256": sha256(training_root / "protocol.json"),
            "seeds": list(seeds),
            "samples_per_seed": samples,
            "total_samples_per_policy": len(seeds) * samples,
            "generation_seed_offset": 202609080000,
            "sampling": training["program_service"]["sampling"],
            "base_model": training["program_service"]["model"],
            "environment_config": str(config_path),
            "environment_config_sha256": sha256(config_path),
            "system_prompt": messages[0]["content"],
            "user_prompt": messages[1]["content"][0]["text"],
            "success_rule": "no program error, task_completed, terminal reward >= 1",
            "controller_enabled": False,
        },
    )


def generate(root: Path, policy: str, training_root: Path | None = None) -> None:
    if policy == "base" and training_root is not None:
        raise ValueError("--training-root applies only to trained_lora generation")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    protocol = read(root / "protocol.json")
    destination = root / "generations" / policy
    destination.mkdir(parents=True, exist_ok=True)
    model_path = protocol["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, local_files_only=True
    ).to("cuda:0")
    identity = {
        "base_model": model_path,
        "policy": policy,
        "protocol_sha256": sha256(root / "protocol.json"),
    }
    if policy == "trained_lora":
        from peft import PeftModel

        training_root = (training_root or Path(protocol["training_root"])).resolve()
        training_protocol_path = training_root / "protocol.json"
        training = read(training_protocol_path)
        if (
            training["program_service"]["model"] != protocol["base_model"]
            or training["program_service"]["sampling"] != protocol["sampling"]
            or set(training["training_seeds"]) & set(protocol["seeds"])
        ):
            raise RuntimeError("checkpoint training does not match the frozen evaluation protocol")
        result = read(training_root / "training_result.json")
        adapters = list(Path(result["checkpoint"]).rglob("adapter_config.json"))
        if len(adapters) != 1 or result["optimizer_step_delta"] < 1:
            raise RuntimeError("training must produce exactly one updated LoRA adapter")
        adapter = adapters[0].parent
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
        identity.update(
            {
                "adapter": str(adapter),
                "adapter_config_sha256": sha256(adapter / "adapter_config.json"),
                "adapter_model_sha256": sha256(adapter / "adapter_model.safetensors"),
                "optimizer_steps": result["optimizer_step_delta"],
                "training_root": str(training_root),
                "training_protocol_sha256": sha256(training_protocol_path),
                "training_result_sha256": sha256(training_root / "training_result.json"),
            }
        )
    identity_path = destination / "identity.json"
    if identity_path.exists():
        if read(identity_path) != identity:
            raise RuntimeError("policy identity changed since previous generation")
    else:
        atomic_write_json(identity_path, identity)
    model.eval()
    inputs = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": protocol["system_prompt"]},
            {"role": "user", "content": protocol["user_prompt"]},
        ],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {key: value.to("cuda:0") for key, value in inputs.items()}
    sampling = dict(protocol["sampling"])
    sampling["max_new_tokens"] = sampling.pop("max_tokens")
    for ordinal in range(protocol["total_samples_per_policy"]):
        path = destination / f"sample_{ordinal:04d}.json"
        if path.exists():
            verify_generation(root, policy, ordinal)
            continue
        generation_seed = protocol["generation_seed_offset"] + ordinal
        torch.manual_seed(generation_seed)
        torch.cuda.manual_seed_all(generation_seed)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                **sampling,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        response = output[0, inputs["input_ids"].shape[1] :].tolist()
        record = {
            "ordinal": ordinal,
            "policy": policy,
            "environment_seed": protocol["seeds"][ordinal // protocol["samples_per_seed"]],
            "sample_index": ordinal % protocol["samples_per_seed"],
            "generation_seed": generation_seed,
            "identity_sha256": sha256(identity_path),
            "source": tokenizer.decode(response, skip_special_tokens=True),
            "terminated_with_eos": tokenizer.eos_token_id in response,
            "response_token_count": len(response),
        }
        atomic_write_json(path, record)
        print(json.dumps({k: v for k, v in record.items() if k != "source"}), flush=True)


def evaluate(root: Path, policy: str) -> None:
    import numpy as np

    from capx.rl.capsule.server_factory import YamlEnvironmentFactory
    from capx.utils.program_source import normalize_program_source

    protocol = read(root / "protocol.json")
    config_path = Path(protocol["environment_config"])
    if sha256(config_path) != protocol["environment_config_sha256"]:
        raise RuntimeError("evaluation environment changed after protocol preparation")
    destination = root / "evaluation" / policy
    destination.mkdir(parents=True, exist_ok=True)
    environment = YamlEnvironmentFactory(str(config_path))(None)
    try:
        for ordinal in range(protocol["total_samples_per_policy"]):
            path = destination / f"sample_{ordinal:04d}.json"
            if path.exists():
                verify_replay(root, policy, ordinal)
                continue
            record, generation_path = verify_generation(root, policy, ordinal)
            source = normalize_program_source(record["source"])
            observation, _ = environment.reset(seed=record["environment_seed"])
            if observation["full_prompt"][1]["content"][0]["text"] != protocol["user_prompt"]:
                raise RuntimeError("evaluation prompt changed")
            # Audit the physical reset without exposing privileged state to the policy.
            sim_data = environment.low_level_env.robosuite_env.sim.data
            initial_state = np.round(np.concatenate((sim_data.qpos, sim_data.qvel)), 10)
            initial_hash = hashlib.sha256(initial_state.astype("<f8").tobytes()).hexdigest()
            _, reward, terminated, truncated, info = environment.step(source)
            error = info.get("error_type")
            diagnostics = "\n".join(
                str(info.get(key, "")) for key in ("stdout", "stderr", "error_message")
            )
            pyroki_server_error = (
                "Request to http://127.0.0.1:8116/" in diagnostics
                and "Server Error:" in diagnostics
            )
            if pyroki_server_error or any(
                marker in diagnostics
                for marker in (
                    "Failed to communicate with",
                    "Connection refused",
                    "ConnectionError",
                    "ConnectError",
                    "ReadTimeout",
                    "HTTPConnectionPool",
                    "timed out",
                )
            ):
                failure_dir = root / "infrastructure_failures"
                failure_dir.mkdir(exist_ok=True)
                atomic_write_json(
                    failure_dir / f"{policy}_{ordinal:04d}_{uuid4().hex}.json",
                    {
                        "generation_sha256": sha256(generation_path),
                        "diagnostics": diagnostics,
                    },
                )
                raise RuntimeError(
                    f"service failure during {policy} sample {ordinal}; resume after recovery"
                )
            if error or info.get("sandbox_rc") != 0:
                outcome = "program_error"
            elif info.get("task_completed") and float(reward) >= 1.0 and not truncated:
                outcome = "success"
            else:
                outcome = "task_failure"
            record.update(
                {
                    "outcome": outcome,
                    "raw_reward": float(reward),
                    "task_completed": bool(info.get("task_completed")),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "error_type": error,
                    "error_message": info.get("error_message"),
                    "stdout": info.get("stdout"),
                    "stderr": info.get("stderr"),
                    "generation_sha256": sha256(generation_path),
                    "initial_state_sha256": initial_hash,
                    "executed_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                }
            )
            atomic_write_json(path, record)
            print(
                json.dumps({"policy": policy, "ordinal": ordinal, "outcome": outcome}), flush=True
            )
    finally:
        environment.close()


def summarize(root: Path) -> None:
    protocol = read(root / "protocol.json")
    summary = {"protocol": protocol}
    paired = {}
    for policy in ("base", "trained_lora"):
        records = [
            verify_replay(root, policy, i) for i in range(protocol["total_samples_per_policy"])
        ]
        paired[policy] = records
        counts = Counter(row["outcome"] for row in records)
        summary[policy] = {
            "identity": read(root / "generations" / policy / "identity.json"),
            "samples": len(records),
            "successes": counts["success"],
            "success_rate": counts["success"] / len(records),
            "outcomes": dict(counts),
            "per_seed": {
                str(seed): sum(
                    r["outcome"] == "success" for r in records if r["environment_seed"] == seed
                )
                for seed in protocol["seeds"]
            },
        }
    for base, trained in zip(paired["base"], paired["trained_lora"], strict=True):
        if not base["initial_state_sha256"] or not trained["initial_state_sha256"]:
            raise RuntimeError("paired evaluation is missing physical reset evidence")
        for key in ("ordinal", "environment_seed", "generation_seed", "initial_state_sha256"):
            if base[key] != trained[key]:
                raise RuntimeError(f"paired evaluation mismatch: {key}")
    summary["success_rate_delta"] = (
        summary["trained_lora"]["success_rate"] - summary["base"]["success_rate"]
    )
    atomic_write_json(root / "summary.json", summary)
    print(json.dumps(summary), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "generate", "evaluate", "summarize"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path)
    parser.add_argument("--policy", choices=("base", "trained_lora"))
    parser.add_argument("--seeds", default=",".join(map(str, range(25, 45))))
    parser.add_argument("--samples-per-seed", type=int, default=4)
    args = parser.parse_args()
    if args.phase == "prepare":
        if args.training_root is None:
            parser.error("prepare requires --training-root")
        prepare(
            args.root.resolve(),
            args.training_root.resolve(),
            tuple(int(s) for s in args.seeds.split(",")),
            args.samples_per_seed,
        )
    elif args.phase == "summarize":
        summarize(args.root.resolve())
    else:
        if args.policy is None:
            parser.error("generate/evaluate requires --policy")
        if args.phase == "generate":
            generate(args.root.resolve(), args.policy, args.training_root)
        else:
            evaluate(args.root.resolve(), args.policy)


if __name__ == "__main__":
    main()
