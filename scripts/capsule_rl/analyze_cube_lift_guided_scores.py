"""Separate executable-code edits from formatting in saved checkpoint diagnostic scores."""

from __future__ import annotations

import argparse
import io
import json
import tokenize
from pathlib import Path
from statistics import mean

from capx.utils.program_source import normalize_program_source
from scripts.capsule_rl.common import atomic_write_json
from scripts.capsule_rl.evaluate_cube_stack import read, sha256


def changed_code_mask(example: dict, tokenizer) -> list[bool]:
    source = example["source"]
    code = normalize_program_source(source)
    offset = source.find(code)
    if offset < 0:
        raise RuntimeError("normalized code must be an unchanged substring")
    lines = code.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))
    spans = []
    ignored = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
               tokenize.DEDENT, tokenize.ENDMARKER, tokenize.ENCODING}
    for token in tokenize.generate_tokens(io.StringIO(code).readline):
        if token.type in ignored:
            continue
        start = offset + starts[token.start[0] - 1] + token.start[1]
        end = offset + starts[token.end[0] - 1] + token.end[1]
        spans.append((start, end))
    encoded = tokenizer(source, add_special_tokens=False, return_offsets_mapping=True)
    if encoded["input_ids"] != example["response_ids"][:-1]:
        raise RuntimeError("scored token identities differ from offset tokenization")
    mask = [changed and any(start < right and end > left for left, right in spans)
            for changed, (start, end) in zip(example["changed_mask"][:-1],
                                            encoded["offset_mapping"], strict=True)]
    return mask + [False]  # EOS is not an executable-code edit.


def main() -> None:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    protocol = read(root / "protocol.json")
    if sha256(root / "examples.json") != protocol["examples_sha256"]:
        raise RuntimeError("scoring examples changed")
    tokenizer = AutoTokenizer.from_pretrained(protocol["base_model"], local_files_only=True)
    examples = [e for e in read(root / "examples.json")["examples"] if e["guided"]]
    masks = {e["id"]: changed_code_mask(e, tokenizer) for e in examples}
    labels = ("base", "A16", "C16", "A32", "C32")
    scores = {label: {r["id"]: r for r in read(root / f"{label}.json")["scores"]}
              for label in labels}
    output = {"checkpoints": {}, "changed_code_token_counts": {
        e["id"]: sum(masks[e["id"]]) for e in examples},
        "input_sha256": {f"{label}.json": sha256(root / f"{label}.json") for label in labels},
        "analysis_source_sha256": sha256(Path(__file__)),
        "scope": "Changed response tokens overlapping executable Python lexical tokens; "
                 "excludes outer fences, comments, whitespace-only tokens and EOS; "
                 "teacher-forced fit, not semantic edit attribution or rollout success."}
    for label in labels:
        rows = []
        for e in examples:
            indices = [i for i, active in enumerate(masks[e["id"]]) if active]
            if not indices:
                continue
            row, base = scores[label][e["id"]], scores["base"][e["id"]]
            nll = -mean(row["log_probs_raw"][i] for i in indices)
            base_nll = -mean(base["log_probs_raw"][i] for i in indices)
            rows.append({"id": e["id"], "context": e["context"], "stage": e["stage"],
                         "transfer_success": e["transfer_success"], "nll": nll,
                         "nll_change_from_base": nll - base_nll,
                         "mean_damping": mean(row["guided_damping_per_token"][i] for i in indices)})
        sections = {}
        for context in ("privileged", "nonprivileged"):
            for subset in ("all", "transfer_success", "transfer_failure", "C16", "C32"):
                selected = [r for r in rows if r["context"] == context and
                            (subset == "all" or subset == r["stage"] or
                             (subset == "transfer_success" and r["transfer_success"]) or
                             (subset == "transfer_failure" and not r["transfer_success"]))]
                sections[f"{context}/{subset}"] = {
                    "programs": len(selected),
                    "changed_code_mean_nll": mean(r["nll"] for r in selected) if selected else None,
                    "nll_change_from_base": mean(r["nll_change_from_base"] for r in selected) if selected else None,
                    "programs_improved": sum(r["nll_change_from_base"] < 0 for r in selected),
                    "mean_damping": mean(r["mean_damping"] for r in selected) if selected else None,
                }
        output["checkpoints"][label] = {"sections": sections, "records": rows}
    atomic_write_json(root / "executable_edit_analysis.json", output)
    print(json.dumps({label: value["sections"] for label, value in output["checkpoints"].items()}))


if __name__ == "__main__":
    main()
