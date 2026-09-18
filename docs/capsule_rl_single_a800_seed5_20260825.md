# Capsule-RL single-A800 LoRA seed-5 stopped-run audit (2026-08-25)

## Scope

- Host: one NVIDIA A800 80GB PCIe at `/root/autodl-tmp/cap-x`.
- Actor: materialized `Qwen2.5-Coder-7B-Instruct`, tree SHA-256
  `54b74c0fde823de9581a37c3dc390374d68760a83d8fc9c8b38bfd07e32b2d93`.
- LoRA: rank 16, alpha 32, `all-linear`.
- VeRL: `d62da4950573d7a4b7ef2362337952e7ab59e78d`.
- Controller: llama.cpp build 10516 with the official Q4_K_M GGUF, SHA-256
  `509287f78cb4d4cf6b3843734733b914b2c158e43e22a7f4bf5e963800894d3c`.

This file is a historical record of the stopped local-Controller run. The replacement run uses
the separately documented external Controller contract; the change does not rewrite the facts
or artifacts recorded below.

The run used the committed owned-service launcher and preserved every attempt in
`artifacts/capsule_single_a800/`. Service credentials were generated in memory and were not
written to the repository, configuration, command line, or audit artifacts.

## Validation result

Gate 1, Gate 2, and Gate 3 passed in the first four OOM profiles. Their Gate 4 attempts stopped
while constructing the colocated vLLM engine because its configured total GPU-memory budget had
no remaining KV-cache block after loading the actor. A fifth, cumulative BF16/micro-batch-1/vLLM
0.45 profile subsequently passed Gate 1--4. No optimizer step, adapter, reload smoke, Gate 7
artifact, or `runtime_verified: true` claim was produced.

| Profile | Run ID | Gate 4 result | Failure artifact SHA-256 |
| --- | --- | --- | --- |
| FP32, dynamic batch, vLLM 0.30 | `base_dynamic_fp32-cc6ec2f0a5aa-controller-seed-1` | zero KV cache blocks | `479e2cd443e540444206f6e5c5ecbc964fbefa2c48d86a652c61055eee33cbca` |
| FP32, dynamic batch, vLLM 0.26 | `vllm_util_026-c69961f908f4-controller-seed-1` | zero KV cache blocks | `4fd6114e5aa8aaf1c304a086737526d3ea9840ffa34d45b3f769ca1212d0a4e3` |
| FP32, fixed micro-batch 1 | `fixed_microbatch_1-536c581700bd-controller-seed-1` | zero KV cache blocks | `eb97684d95a50e5ae9c179e230571ffd2a8b3e5e0b73cd10b0aee6ac5ffbd89d` |
| BF16 base, fixed micro-batch 1 | `fsdp_base_bf16-53f8b59edaa7-controller-seed-1` | zero KV cache blocks | `e56092c77c40ea9d4c63fc0dbbbe054d253bbcb98c7a9df8b3b6dec6ff774b2a` |
| BF16 base, fixed micro-batch 1, vLLM 0.45 | `fsdp_base_bf16_vllm_util_045-0036fadc670e-controller-seed-1` | Gate 4 passed; all four traces had zero edits and remained failed | see preserved `gate04_collector.json` |

The final traceback is:

```text
ValueError: No available memory for the cache blocks. Try increasing
`gpu_memory_utilization` when initializing the engine.
```

The launcher classified the first four vLLM initialization conditions as OOM and generated a new
config hash and run ID for every profile. The fifth profile proved that the 0.45 budget could
initialize and run Gate 4 without reducing the 10,240-token caps or changing group size, KL, or
GRPO semantics.

The fifth-profile collector artifact is preserved at:

```text
artifacts/capsule_single_a800/
  fsdp_base_bf16_vllm_util_045-0036fadc670e-controller-seed-1/
  gate04_collector.json
```

All four repair traces contained zero accepted edits. Each trace recorded 12 parse failures, and
its final source was byte-identical to its P0. The Actor P0 was wrapped in a Markdown fence; it was
correctly replayed unchanged as a `SyntaxError` with reward 0, but the old repair-unit discovery
exposed only the whole program. The local Controller's 512-token response was then truncated into
invalid JSON full-program replacements. This was a protocol/repair-transport defect, not a reason
to clean the Actor output before replay.

Gate 5 was started independently, then deliberately stopped once the structural cause was proven;
continuing would have spent hours without creating a valid explicit fence edit. The launcher
cleaned its owned Ray and service processes, the GPU returned idle, and all failure evidence was
retained. No Gate 6, adapter reload, or Gate 7 runtime-verification artifact exists for this run.

The replacement run uses a new config/hash/run ID, explicit fence-open/fence-close deletion units,
an external `qwen3.7-plus` Controller with a 4096-token non-streaming, non-thinking request, and
GPU PyRoKi. It must satisfy the unchanged Gate 1--7 acceptance criteria before any
`runtime_verified: true` claim.

## Tests

- Remote focused Capsule suite after the final launcher fix: 344 passed, 2 warnings.
- Earlier complete Capsule regression suite on the same runtime: 705 passed, 2 warnings.
- Local WSL launcher regression suite after the final fix: 56 passed, 2 warnings.

This stopped run remains a negative audit record and must not be presented as runtime verified.
