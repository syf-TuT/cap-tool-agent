"""Collect successful glm-4.7 Restack programs and fit an assistant-only LoRA SFT."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from scripts.capsule_rl.common import atomic_write_json
from scripts.capsule_rl.run_cube_stack_ac import write_status


def collect(root: Path, target: int, budget: int, seed_count: int = 32) -> None:
    from openai import OpenAI

    from capx.rl.capsule.server_factory import YamlEnvironmentFactory
    from capx.utils.program_source import normalize_program_source

    client = OpenAI(api_key=os.environ["CAPX_CONTROLLER_API_KEY"],
                    base_url="https://coding.dashscope.aliyuncs.com/v1", timeout=300)
    config = Path("env_configs/cube_restack/capsule_rl/"
                  "franka_robosuite_cube_restack_privileged_clean_replay.yaml")
    protocol = {"teacher": "glm-4.7", "target": target, "budget": budget,
                "teacher_scene_context": "green initially stacked on red; unstack before restacking",
                "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                "training_seeds": list(range(5, 5 + seed_count)), "temperature": 0.9}
    if (root / "collection_protocol.json").exists():
        if json.loads((root / "collection_protocol.json").read_text()) != protocol:
            raise RuntimeError("collection protocol changed")
    else:
        atomic_write_json(root / "collection_protocol.json", protocol)
    records = root / "samples"
    records.mkdir(exist_ok=True)
    successes = []
    seen = set()
    env = YamlEnvironmentFactory(str(config.resolve()))(None)
    try:
        for ordinal in range(budget):
            path = records / f"{ordinal:04d}.json"
            if path.exists():
                record = json.loads(path.read_text())
            else:
                seed = 5 + ordinal % seed_count
                obs, reset_info = env.reset(seed=seed)
                messages = obs["full_prompt"]
                teacher_messages = messages + [{"role": "user", "content": (
                    "Scene context: initially the green cube is stacked ON TOP OF the red cube. "
                    "The goal is to reverse this stack. Plan the complete unstack-and-restack "
                    "sequence, placing the green cube gently on a clear tabletop location first. "
                    "Query current object poses after moving objects. Return executable Python "
                    "only, including all required imports."
                )}]
                response = client.chat.completions.create(
                    model="glm-4.7", messages=teacher_messages, temperature=0.9,
                    max_tokens=4096, extra_body={"enable_thinking": False})
                raw = response.choices[0].message.content or ""
                source = normalize_program_source(raw)
                _, reward, terminated, truncated, info = env.step(source)
                success = (response.choices[0].finish_reason == "stop"
                           and not info.get("error_type") and info.get("sandbox_rc") == 0
                           and bool(info.get("task_completed")) and float(reward) >= 1
                           and not truncated)
                record = {"ordinal": ordinal, "seed": seed, "messages": messages,
                          "teacher_messages": teacher_messages,
                          "source": source, "teacher": response.model,
                          "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                          "initial_state_sha256": reset_info.get("initial_state_sha256"),
                          "success": success, "reward": float(reward),
                          "terminated": bool(terminated), "truncated": bool(truncated),
                          "error_type": info.get("error_type"),
                          "error_message": info.get("error_message"),
                          "usage": response.usage.model_dump() if response.usage else None}
                atomic_write_json(path, record)
            if record["success"] and record["source_sha256"] not in seen:
                seen.add(record["source_sha256"])
                successes.append(record)
            write_status(root / "status.json", {"phase": "collection", "attempts": ordinal + 1,
                                               "unique_successes": len(successes)})
            print(json.dumps({"attempt": ordinal + 1, "success": record["success"],
                              "unique_successes": len(successes)}), flush=True)
            if len(successes) >= target:
                break
    finally:
        env.close()
    (root / "sft.jsonl").write_text("".join(json.dumps(r) + "\n" for r in successes))
    if len(successes) < target:
        raise RuntimeError(f"only {len(successes)} unique successes; target is {target}")


def train(root: Path, model_path: Path, epochs: int = 3) -> None:
    import random
    import shutil

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(20260910)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    rows = [json.loads(s) for s in (root / "sft.jsonl").read_text().splitlines()]
    examples = []
    for row in rows:
        if not row["success"] or hashlib.sha256(row["source"].encode()).hexdigest() != row["source_sha256"]:
            raise RuntimeError("invalid SFT success provenance")
        messages = []
        for message in row["messages"]:
            content = message["content"]
            if isinstance(content, list):
                content = "\n".join(c["text"] for c in content if c.get("type") == "text")
            messages.append({"role": message["role"], "content": content})
        prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        tokens = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": row["source"]}], tokenize=True)
        if tokens[:len(prefix)] != prefix or len(tokens) > 8192:
            raise RuntimeError("SFT token boundary or sequence length invalid")
        examples.append((tokens, [-100] * len(prefix) + tokens[len(prefix):]))
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                               attn_implementation="sdpa").cuda()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules="all-linear",
                                            task_type="CAUSAL_LM", lora_dropout=0.0))
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-5)
    model.train()
    losses = []
    steps = 0
    for epoch in range(epochs):
        order = list(range(len(examples)))
        random.Random(20260910 + epoch).shuffle(order)
        for start in range(0, len(order), 4):
            batch = order[start:start + 4]
            optimizer.zero_grad()
            for index in batch:
                tokens, labels = examples[index]
                inputs = torch.tensor([tokens], device="cuda")
                loss = model(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                             labels=torch.tensor([labels], device="cuda")).loss
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite SFT loss")
                losses.append(float(loss.detach()))
                (loss / len(batch)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            steps += 1
            write_status(root / "status.json", {"phase": "sft", "epoch": epoch + 1,
                                               "optimizer_steps": steps, "loss": losses[-1]})
        if epoch + 1 in (1, 3, epochs):
            snapshot = root / f"epoch_{epoch + 1}"
            snapshot.mkdir(exist_ok=True)
            model.save_pretrained(snapshot / "lora_adapter")
            tokenizer.save_pretrained(snapshot / "lora_adapter")
            for filename in ("sft.jsonl", "collection_protocol.json"):
                shutil.copyfile(root / filename, snapshot / filename)
            atomic_write_json(snapshot / "sft_result.json", {
                "examples": len(examples), "epochs": epoch + 1, "optimizer_steps": steps,
                "base_model": str(model_path.resolve()),
                "dataset_sha256": hashlib.sha256((root / "sft.jsonl").read_bytes()).hexdigest(),
                "learning_rate": 2e-5, "rank": 16, "alpha": 32,
                "loss_mask": "assistant_only", "adapter": str(snapshot / "lora_adapter")})
    model.save_pretrained(root / "lora_adapter")
    tokenizer.save_pretrained(root / "lora_adapter")
    atomic_write_json(root / "sft_result.json", {"examples": len(examples), "epochs": epochs,
                      "optimizer_steps": steps, "losses": losses, "base_model": str(model_path),
                      "dataset_sha256": hashlib.sha256((root / "sft.jsonl").read_bytes()).hexdigest(),
                      "learning_rate": 2e-5, "rank": 16, "alpha": 32,
                      "loss_mask": "assistant_only", "adapter": str(root / "lora_adapter")})
    write_status(root / "status.json", {"phase": "completed", "examples": len(examples),
                                       "optimizer_steps": steps})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--target", type=int, default=32)
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--seed-count", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    if not 8 <= args.target <= args.budget or not 1 <= args.seed_count <= 196 or args.epochs < 1:
        parser.error("require 8 <= target <= budget, seed-count 1..196, and positive epochs")
    os.environ.update({"MUJOCO_GL": "egl", "JAX_PLATFORMS": "cpu",
                       "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "OMP_NUM_THREADS": "4"})
    args.root.mkdir(parents=True, exist_ok=True)
    collect(args.root, args.target, args.budget, args.seed_count)
    train(args.root, args.model, args.epochs)


if __name__ == "__main__":
    main()
