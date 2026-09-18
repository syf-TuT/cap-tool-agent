"""Score archived repair groups at frozen checkpoints and inspect log-probability gradients.

This is a checkpoint diagnostic, not a reconstruction of unsaved historical gradients.
No generation, simulator rollout, optimizer step, or model-parameter backward pass occurs.
"""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path
from statistics import mean

from scripts.capsule_rl.common import atomic_write_json
from scripts.capsule_rl.evaluate_cube_stack import read, sha256


def build_examples(training: Path, replay: Path, tokenizer) -> list[dict]:
    pairs = read(replay / "pairs.json")["pairs"]
    transferred = {r["pair_id"]: r["repaired"] == "success"
                   for r in read(replay / "summary.json")["records"]}
    evaluation = read(training / "eval_C16/protocol.json")
    examples = []
    for pair in pairs:
        path = Path(pair["group_path"])
        if sha256(path) != pair["group_sha256"]:
            raise RuntimeError(f"source group changed: {path}")
        artifact = read(path)
        members = artifact["assembly"]["group"]["members"]
        protocol = read(training / pair["stage"] / "protocol.json")
        original_ids = tokenizer.encode(pair["original"]["source"], add_special_tokens=False)
        revision_ids = tokenizer.encode(pair["repaired"]["source"], add_special_tokens=False)
        changed = set()
        for tag, _, _, start, end in difflib.SequenceMatcher(
            a=original_ids, b=revision_ids, autojunk=False
        ).get_opcodes():
            if tag != "equal":
                changed.update(range(start, end))
        for index, member in enumerate(members):
            guided = member["member_type"] == "critique_guided_revision"
            is_original = member["program_sample_id"] == pair["original"]["program_sample_id"]
            response = tokenizer.encode(member["response"], add_special_tokens=False)
            response.append(tokenizer.eos_token_id)
            if guided and sum(artifact["guided_token_mask"][index]) != len(response):
                raise RuntimeError("reconstructed guided token count differs from training")
            expected_advantage = member["reward"] - mean(artifact["sequence_rewards"])
            if artifact["sequence_advantages"][index] != expected_advantage:
                raise RuntimeError("archived advantage differs from the reward baseline")
            if guided and member["response"] != pair["repaired"]["source"]:
                raise RuntimeError("selected revision source mismatch")
            role = "repaired" if guided else "ordinary"
            if is_original:
                role = "original"
            prompts = [("privileged", protocol["program_service"]["system_prompt"], member["prompt"])]
            if guided or is_original:
                prompts.append(("nonprivileged", evaluation["system_prompt"], evaluation["user_prompt"]))
            for context, system, prompt in prompts:
                prompt_ids = tokenizer.apply_chat_template(
                    [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                    tokenize=True, add_generation_prompt=True,
                )
                examples.append({
                    "id": f"{pair['pair_id']}:{context}:{index}",
                    "pair_id": pair["pair_id"], "stage": pair["stage"],
                    "context": context, "role": role, "guided": guided,
                    "transfer_success": transferred[pair["pair_id"]],
                    "reward": member["reward"], "advantage": expected_advantage,
                    "source": member["response"], "source_sha256": member["response_sha256"],
                    "prompt_ids": prompt_ids, "response_ids": response,
                    "changed_mask": [guided and i in changed for i in range(len(response))],
                })
    if len(pairs) != 17 or len(examples) != 170:
        raise RuntimeError("expected 17 groups of eight plus 34 nonprivileged-context probes")
    return examples


def loss_diagnostic(log_probs, advantage: float, guided: bool, gamma: float) -> dict:
    import torch

    from capx.rl.capsule.policy_loss import capsule_critique_policy_loss

    # Training uses microbatch=1 and accumulates eight sequence means per update.
    with torch.enable_grad():
        variable = log_probs.detach().float().cpu().clone().unsqueeze(0).requires_grad_(True)
        mask = torch.ones_like(variable, dtype=torch.bool)
        loss, *_ = capsule_critique_policy_loss(
            variable.detach(), variable, torch.full_like(variable, advantage),
            mask, mask if guided else torch.zeros_like(mask), capsule_gamma=gamma,
        )
        gradient, = torch.autograd.grad(loss / 8, variable)
    result = {"pg_logprob_gradient_l1": gradient.abs().sum().item()}
    if guided:
        weight = torch.sigmoid(variable.detach()[0] - torch.log(torch.tensor(gamma)))
        damping = weight * (1 - weight)
        result.update({
            "guided_damping_mean": damping.mean().item(),
            "guided_damping_per_token": damping.tolist(),
            "ordinary_same_advantage_gradient_l1": abs(advantage) / 8,
        })
    return result


def score(model, tokenizer, examples: list[dict], temperature: float, gamma: float) -> list[dict]:
    import torch

    rows = []
    for index, example in enumerate(examples):
        prompt, response = example["prompt_ids"], example["response_ids"]
        ids = torch.tensor([prompt + response], device="cuda:0")
        targets = torch.tensor(response, device="cuda:0")
        with torch.inference_mode():
            output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            logits = output.logits[0, len(prompt) - 1:-1].float()
            selected = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            lp_raw = (selected - torch.logsumexp(logits, dim=-1)).cpu()
            lp_training = (selected / temperature
                           - torch.logsumexp(logits / temperature, dim=-1)).cpu()
            del output, logits, selected
        if not torch.isfinite(lp_raw).all() or not torch.isfinite(lp_training).all():
            raise RuntimeError("non-finite teacher-forced log probabilities")
        changed = torch.tensor(example["changed_mask"], dtype=torch.bool)
        row = {
            "id": example["id"], "mean_nll": -lp_raw.mean().item(),
            "mean_nll_training_temperature": -lp_training.mean().item(),
            "changed_mean_nll": -lp_raw[changed].mean().item() if changed.any() else None,
            "log_probs_raw": lp_raw.tolist(), "log_probs_training_temperature": lp_training.tolist(),
            **loss_diagnostic(lp_training, example["advantage"], example["guided"], gamma),
        }
        if example["guided"]:
            damping = torch.tensor(row["guided_damping_per_token"])
            row["changed_guided_damping_mean"] = damping[changed].mean().item() if changed.any() else None
        rows.append(row)
        if (index + 1) % 20 == 0 or index + 1 == len(examples):
            print(json.dumps({"scored": index + 1, "total": len(examples)}), flush=True)
    return rows


def summarize(root: Path, examples: list[dict]) -> dict:
    scores = {label: {r["id"]: r for r in read(root / f"{label}.json")["scores"]}
              for label in ("base", "A16", "C16", "A32", "C32")}
    result = {}
    for label, rows in scores.items():
        sections = {}
        for context in ("privileged", "nonprivileged"):
            for subset in ("all", "transfer_success", "transfer_failure", "C16", "C32"):
                selected = [e for e in examples if e["guided"] and e["context"] == context
                            and (subset == "all" or subset == e["stage"]
                                 or (subset == "transfer_success" and e["transfer_success"])
                                 or (subset == "transfer_failure" and not e["transfer_success"]))]
                sections[f"{context}/{subset}"] = {
                    "programs": len(selected),
                    "mean_nll": mean(rows[e["id"]]["mean_nll"] for e in selected),
                    "mean_nll_change_from_base": mean(rows[e["id"]]["mean_nll"]
                        - scores["base"][e["id"]]["mean_nll"] for e in selected),
                    "changed_mean_nll": mean(rows[e["id"]]["changed_mean_nll"] for e in selected
                                             if rows[e["id"]]["changed_mean_nll"] is not None),
                    "changed_nll_change_from_base": mean(rows[e["id"]]["changed_mean_nll"]
                        - scores["base"][e["id"]]["changed_mean_nll"] for e in selected
                        if rows[e["id"]]["changed_mean_nll"] is not None),
                    "programs_nll_improved": sum(rows[e["id"]]["mean_nll"]
                        < scores["base"][e["id"]]["mean_nll"] for e in selected),
                    "mean_guided_damping": mean(rows[e["id"]]["guided_damping_mean"] for e in selected),
                    "mean_changed_guided_damping": mean(rows[e["id"]]["changed_guided_damping_mean"]
                        for e in selected if rows[e["id"]]["changed_guided_damping_mean"] is not None),
                }
        shares = []
        for pair_id in {e["pair_id"] for e in examples}:
            group = [e for e in examples if e["pair_id"] == pair_id and e["context"] == "privileged"]
            total = sum(rows[e["id"]]["pg_logprob_gradient_l1"] for e in group)
            guided = sum(rows[e["id"]]["pg_logprob_gradient_l1"] for e in group if e["guided"])
            shares.append(guided / total)
        result[label] = {"sections": sections, "mean_guided_share_of_pg_logprob_l1": mean(shares)}
    payload = {
        "status": "completed", "checkpoints": result, "examples_per_checkpoint": len(examples),
        "limitations": "Frozen-checkpoint teacher forcing on selected training groups. "
        "Log-probability-space PG derivatives exclude KL, parameter Jacobians, gradient clipping, and Adam. "
        "Not historical update gradients or evidence of increased rollout success. "
        "Changed-token masks use token sequence alignment, not semantic edit attribution.",
    }
    atomic_write_json(root / "summary.json", payload)
    return payload


def main() -> None:
    import torch
    import yaml
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    training, replay, root = args.training_root.resolve(), args.replay_root.resolve(), args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    protocol = read(training / "C16/protocol.json")
    worker = yaml.safe_load((training / "C16/verl.yaml").read_text())["actor_rollout_ref"]["actor"]
    if worker["use_dynamic_bsz"] or worker["ppo_micro_batch_size_per_gpu"] != 1:
        raise RuntimeError("diagnostic assumes the recorded one-sequence microbatch recipe")
    model_path = protocol["program_service"]["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    examples = build_examples(training, replay, tokenizer)
    atomic_write_json(root / "examples.json", {"examples": examples})
    adapters = {}
    for label in ("A16", "C16", "A32", "C32"):
        result = read(training / label / "training_result.json")
        configs = list(Path(result["checkpoint"]).rglob("adapter_config.json"))
        if result["status"] != "completed" or len(configs) != 1:
            raise RuntimeError(f"invalid completed checkpoint: {label}")
        adapter = configs[0].parent
        adapters[label] = {"path": str(adapter), "weights_sha256": sha256(adapter / "adapter_model.safetensors"),
                           "training_result_sha256": sha256(training / label / "training_result.json")}
    atomic_write_json(root / "protocol.json", {
        "base_model": model_path, "adapters": adapters, "examples_sha256": sha256(root / "examples.json"),
        "script_sha256": sha256(Path(__file__)), "policy_loss_sha256": sha256(Path(__file__).resolve().parents[2]
            / "capx/rl/capsule/policy_loss.py"), "training_temperature": protocol["program_service"]["sampling"]["temperature"],
        "gamma": protocol["capsule"]["gamma"], "eos_included": True,
        "model_parameters_updated": False,
    })
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                                local_files_only=True).to("cuda:0")
    for label in ("base", "A16", "C16", "A32", "C32"):
        if label == "A16":
            model = PeftModel.from_pretrained(model, adapters[label]["path"], adapter_name=label,
                                              is_trainable=False)
        elif label != "base":
            model.load_adapter(adapters[label]["path"], adapter_name=label, is_trainable=False)
        if label != "base":
            model.set_adapter(label)
        model.eval()
        model.requires_grad_(False)
        print(json.dumps({"checkpoint": label, "status": "scoring"}), flush=True)
        rows = score(model, tokenizer, examples, protocol["program_service"]["sampling"]["temperature"],
                     protocol["capsule"]["gamma"])
        atomic_write_json(root / f"{label}.json", {"scores": rows})
    print(json.dumps(summarize(root, examples)), flush=True)


if __name__ == "__main__":
    main()
