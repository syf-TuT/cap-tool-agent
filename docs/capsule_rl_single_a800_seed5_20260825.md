# Capsule-RL single-A800 LoRA seed-5 audit (2026-08-25)

## Scope

- Host: one NVIDIA A800 80GB PCIe at `/root/autodl-tmp/cap-x`.
- Actor: materialized `Qwen2.5-Coder-7B-Instruct`, tree SHA-256
  `54b74c0fde823de9581a37c3dc390374d68760a83d8fc9c8b38bfd07e32b2d93`.
- LoRA: rank 16, alpha 32, `all-linear`.
- VeRL: `d62da4950573d7a4b7ef2362337952e7ab59e78d`.
- Controller: llama.cpp build 10516 with the official Q4_K_M GGUF, SHA-256
  `509287f78cb4d4cf6b3843734733b914b2c158e43e22a7f4bf5e963800894d3c`.

The run used the committed owned-service launcher and preserved every attempt in
`artifacts/capsule_single_a800/`. Service credentials were generated in memory and were not
written to the repository, configuration, command line, or audit artifacts.

## Validation result

Gate 1, Gate 2, and Gate 3 passed in all four OOM profiles. Gate 4 stopped while constructing
the colocated vLLM engine because its configured total GPU-memory budget had no remaining KV
cache block after loading the actor. No collector output, optimizer step, adapter, reload smoke,
Gate 7 artifact, or `runtime_verified: true` claim was produced.

| Profile | Run ID | Gate 4 result | Failure artifact SHA-256 |
| --- | --- | --- | --- |
| FP32, dynamic batch, vLLM 0.30 | `base_dynamic_fp32-cc6ec2f0a5aa-controller-seed-1` | zero KV cache blocks | `479e2cd443e540444206f6e5c5ecbc964fbefa2c48d86a652c61055eee33cbca` |
| FP32, dynamic batch, vLLM 0.26 | `vllm_util_026-c69961f908f4-controller-seed-1` | zero KV cache blocks | `4fd6114e5aa8aaf1c304a086737526d3ea9840ffa34d45b3f769ca1212d0a4e3` |
| FP32, fixed micro-batch 1 | `fixed_microbatch_1-536c581700bd-controller-seed-1` | zero KV cache blocks | `eb97684d95a50e5ae9c179e230571ffd2a8b3e5e0b73cd10b0aee6ac5ffbd89d` |
| BF16 base, fixed micro-batch 1 | `fsdp_base_bf16-53f8b59edaa7-controller-seed-1` | zero KV cache blocks | `e56092c77c40ea9d4c63fc0dbbbe054d253bbcb98c7a9df8b3b6dec6ff774b2a` |

The final traceback is:

```text
ValueError: No available memory for the cache blocks. Try increasing
`gpu_memory_utilization` when initializing the engine.
```

The launcher classified this vLLM initialization condition as OOM, generated a new config hash
and run ID for every profile, cleaned its Ray and service processes, and stopped after exhausting
the specified four-level ladder. It did not reduce the 10,240-token caps or change group size,
KL, or GRPO semantics. At shutdown, the experiment disk retained 83,033 MiB free and no owned
GPU, Ray, controller, actor-identity, or PyRoKi process remained.

## Tests

- Remote focused Capsule suite after the final launcher fix: 344 passed, 2 warnings.
- Earlier complete Capsule regression suite on the same runtime: 705 passed, 2 warnings.
- Local WSL launcher regression suite after the final fix: 56 passed, 2 warnings.

The next experiment must use a new config/hash/run ID. Proceeding requires an explicit profile
change outside the exhausted ladder, such as revising the vLLM memory budget or changing how the
colocated actor is resident during vLLM KV-cache initialization.
