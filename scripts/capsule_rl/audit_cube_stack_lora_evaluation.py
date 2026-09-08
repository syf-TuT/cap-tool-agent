"""Inspect an existing Cube Stack checkpoint and reproduce its evaluation generation."""

import argparse
import contextlib
import difflib
import hashlib
import json
import statistics
from pathlib import Path

import torch
from peft import PeftModel, get_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {}

    def save(stage):
        report["last_completed_stage"] = stage
        (args.output / "audit.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({"stage": stage}), flush=True)

    train = read(args.training_root / "protocol.json")
    evaluation = read(args.evaluation_root / "protocol.json")
    identity = read(args.evaluation_root / "generations/trained_lora/identity.json")
    result = read(args.training_root / "training_result.json")
    checkpoint = Path(result["checkpoint"])
    adapter = checkpoint / "lora_adapter"
    model_path = Path(evaluation["base_model"])
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    saved_tokenizer = AutoTokenizer.from_pretrained(
        checkpoint / "huggingface", local_files_only=True
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    task = json.loads(
        (args.training_root / "dataset.seed_resolved.jsonl").read_text().splitlines()[0]
    )
    prompts = {"training": task["prompt"], "evaluation": evaluation["user_prompt"]}
    system = evaluation["system_prompt"]

    def encode(user, tok=tokenizer):
        return tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=True,
            add_generation_prompt=True,
        )

    prompt_ids = {key: encode(value) for key, value in prompts.items()}
    report["identity"] = {
        "training_root": str(args.training_root),
        "evaluation_root": str(args.evaluation_root),
        "checkpoint": str(checkpoint),
        "base_model": str(model_path),
        "adapter_sha256": digest(adapter / "adapter_model.safetensors"),
        "adapter_hash_matches_evaluation": digest(adapter / "adapter_model.safetensors")
        == identity["adapter_model_sha256"],
        "adapter_config_matches_evaluation": digest(adapter / "adapter_config.json")
        == identity["adapter_config_sha256"],
        "training_result_matches_evaluation": digest(args.training_root / "training_result.json")
        == identity["training_result_sha256"],
        "training_protocol_matches_evaluation": digest(args.training_root / "protocol.json")
        == identity["training_protocol_sha256"],
        "base_model_paths_match": str(model_path) == train["program_service"]["model"],
        "sampling_parameters_match": evaluation["sampling"] == train["program_service"]["sampling"],
        "sampling": evaluation["sampling"],
    }
    report["prompts"] = {
        "system_identical": system == train["program_service"]["system_prompt"],
        "user_identical": prompts["training"] == prompts["evaluation"],
        "task_rules_identical": prompts["training"].split("APIs:")[0]
        == prompts["evaluation"].split("APIs:")[0],
        "token_counts": {k: len(v) for k, v in prompt_ids.items()},
        "checkpoint_tokenizer_ids_match": all(
            encode(p, saved_tokenizer) == encode(p) for p in prompts.values()
        ),
        "chat_template_identical": tokenizer.chat_template == saved_tokenizer.chat_template,
        "diff": list(
            difflib.unified_diff(
                prompts["training"].splitlines(),
                prompts["evaluation"].splitlines(),
                fromfile="training",
                tofile="evaluation",
                lineterm="",
            )
        ),
    }
    (args.output / "prompt_diff.txt").write_text("\n".join(report["prompts"]["diff"]))
    report["versions"] = {"torch": torch.__version__}
    save("metadata")

    weights = load_file(str(adapter / "adapter_model.safetensors"))
    full = torch.load(
        checkpoint / "model_world_size_1_rank_0.pt",
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    checks = []
    for key, value in weights.items():
        full_key = key.replace(".lora_A.weight", ".lora_A.default.weight").replace(
            ".lora_B.weight", ".lora_B.default.weight"
        )
        checks.append(full_key in full and torch.equal(value, full[full_key].to(value.dtype)))
    report["checkpoint_adapter"] = {
        "full_state_tensor_count": len(full),
        "adapter_tensor_count": len(weights),
        "adapter_parameter_count": sum(t.numel() for t in weights.values()),
        "all_adapter_tensors_match_full_checkpoint": all(checks),
        "matched_tensor_count": sum(checks),
        "nonfinite_tensors": [k for k, t in weights.items() if not torch.isfinite(t).all()],
        "zero_B_tensors": [
            k for k, t in weights.items() if ".lora_B." in k and not torch.count_nonzero(t)
        ],
        "sample_full_keys": list(full)[:5],
    }
    save("checkpoint_weights")

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, local_files_only=True
    ).to("cuda:0")
    model.eval()
    base_mismatches = []
    for key, tensor in model.state_dict().items():
        full_key = "base_model.model." + key
        if full_key not in full:
            prefix, suffix = full_key.rsplit(".", 1)
            full_key = prefix + ".base_layer." + suffix
        if full_key not in full or not torch.equal(tensor.cpu(), full[full_key].to(tensor.dtype)):
            base_mismatches.append(key)
    report["checkpoint_adapter"]["base_weight_mismatches"] = base_mismatches
    del full

    def logits(ids):
        inputs = torch.tensor([ids], device="cuda:0")
        with torch.inference_mode():
            return (
                model(input_ids=inputs, attention_mask=torch.ones_like(inputs), logits_to_keep=1)
                .logits[0, -1]
                .float()
                .cpu()
            )

    base_logits = {key: logits(ids) for key, ids in prompt_ids.items()}
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    model.eval()
    loaded = get_peft_model_state_dict(model)
    report["runtime_adapter"] = {
        "active_adapters": model.active_adapters,
        "loaded_tensor_count": len(loaded),
        "all_loaded_weights_match_saved": set(loaded) == set(weights)
        and all(torch.equal(loaded[k].cpu(), v.to(loaded[k].dtype)) for k, v in weights.items()),
        "training_mode": model.training,
        "config": model.peft_config["default"].to_dict(),
        "scales": sorted(
            {
                float(m.scaling["default"])
                for m in model.modules()
                if isinstance(getattr(m, "scaling", None), dict) and "default" in m.scaling
            }
        ),
        "disabled_layer_count": sum(
            bool(m.disable_adapters)
            for m in model.modules()
            if hasattr(m, "disable_adapters") and not callable(m.disable_adapters)
        ),
        "logits": {},
    }
    # Convert PEFT enum/set fields for the JSON record.
    report["runtime_adapter"]["config"] = json.loads(
        json.dumps(
            report["runtime_adapter"]["config"],
            default=lambda v: sorted(v) if isinstance(v, set) else str(v),
        )
    )
    for key, ids in prompt_ids.items():
        active = logits(ids)
        with model.disable_adapter():
            disabled = logits(ids)
        report["runtime_adapter"]["logits"][key] = {
            "disabled_vs_pristine_max_abs": float((disabled - base_logits[key]).abs().max()),
            "enabled_vs_disabled_max_abs": float((active - disabled).abs().max()),
            "enabled_vs_disabled_mean_abs": float((active - disabled).abs().mean()),
        }
    save("live_adapter")

    sampling = dict(evaluation["sampling"])
    sampling["max_new_tokens"] = sampling.pop("max_tokens")
    report["generation_reproductions"] = []
    inputs = torch.tensor([prompt_ids["evaluation"]], device="cuda:0")
    for policy in ("base", "trained_lora"):
        for ordinal in (0, 30, 62):
            old = read(args.evaluation_root / f"generations/{policy}/sample_{ordinal:04d}.json")
            torch.manual_seed(old["generation_seed"])
            torch.cuda.manual_seed_all(old["generation_seed"])
            with (
                model.disable_adapter() if policy == "base" else contextlib.nullcontext(),
                torch.inference_mode(),
            ):
                output = model.generate(
                    input_ids=inputs,
                    attention_mask=torch.ones_like(inputs),
                    **sampling,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            response = output[0, inputs.shape[1] :].tolist()
            source = tokenizer.decode(response, skip_special_tokens=True)
            record = {
                "policy": policy,
                "ordinal": ordinal,
                "source_identical": source == old["source"],
                "response_token_count": len(response),
                "original_response_token_count": old["response_token_count"],
                "source": source,
            }
            report["generation_reproductions"].append(record)
            save(f"reproduce_{policy}_{ordinal}")

    report["teacher_forcing"] = []
    members = []
    for path in sorted((args.training_root / "training/groups").glob("*.json")):
        group = read(path)["assembly"]["group"]
        for member in group["members"]:
            members.append(
                {
                    "seed": group["environment_seed"],
                    "member_type": member["member_type"],
                    "reward": member["reward"],
                    "response": member["response"],
                }
            )
    for i, member in enumerate(members):
        response = tokenizer.encode(member["response"], add_special_tokens=False) + [
            tokenizer.eos_token_id
        ]
        row = {k: v for k, v in member.items() if k != "response"}
        row["response_token_count"] = len(response)
        for prompt_name, ids in prompt_ids.items():
            input_ids = torch.tensor([ids + response], device="cuda:0")
            target = torch.tensor(response, device="cuda:0")
            for policy in ("base", "trained_lora"):
                with (
                    model.disable_adapter() if policy == "base" else contextlib.nullcontext(),
                    torch.inference_mode(),
                ):
                    output = model(
                        input_ids=input_ids,
                        attention_mask=torch.ones_like(input_ids),
                        use_cache=False,
                        logits_to_keep=len(response) + 1,
                    )
                    token_logits = output.logits[0, :-1].float()
                    logp = token_logits.log_softmax(-1).gather(-1, target[:, None]).squeeze(-1)
                    row[f"{prompt_name}_{policy}_mean_logp"] = float(logp.mean())
                del output, token_logits, logp
        report["teacher_forcing"].append(row)
        if (i + 1) % 16 == 0:
            save(f"teacher_forcing_{i + 1}")
    report["teacher_forcing_summary"] = {}
    for label, rows in [
        ("success", [r for r in report["teacher_forcing"] if r["reward"] == 1]),
        ("failure", [r for r in report["teacher_forcing"] if r["reward"] == 0]),
        ("guided", [r for r in report["teacher_forcing"] if r["member_type"] != "base"]),
    ]:
        report["teacher_forcing_summary"][label] = {"count": len(rows)}
        for prompt_name in prompt_ids:
            deltas = [
                r[f"{prompt_name}_trained_lora_mean_logp"] - r[f"{prompt_name}_base_mean_logp"]
                for r in rows
            ]
            report["teacher_forcing_summary"][label][prompt_name] = {
                "mean_logp_delta": statistics.mean(deltas),
                "increased_count": sum(d > 0 for d in deltas),
            }
    save("completed")


if __name__ == "__main__":
    main()
