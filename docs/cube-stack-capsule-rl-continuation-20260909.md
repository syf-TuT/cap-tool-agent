# Cube Stack continuation from 32 to 64 and 96 groups

Status: both stages and their 80 evaluations completed and verified. The
96-group model reached 38/80 (47.50%). The orchestration process exited normally.

| Model | Cumulative groups | Non-privileged high-level success |
| --- | ---: | ---: |
| Initial Qwen2.5-Coder-7B-Instruct | 0 | 7/80 (8.75%) |
| Existing LoRA | 32 | 19/80 (23.75%) |
| Continued LoRA | 64 | 24/80 (30.00%) |
| Continued LoRA | 96 | 38/80 (47.50%) |

The 96-group model improves by 17.50 percentage points over 64 groups and
38.75 points over the initial model. Outcomes: 38 successes, 40 task failures,
two program errors. All 80 paired records, physical reset hashes, result identity,
and ancestral seed separation were verified. The restored optimizer step was
63 and the final step was 88: 25 updates and seven constant-reward skips.
Two group attempts were discarded by the existing decode/retokenize round-trip
validation and retried under the existing policy; all 32 scheduled groups completed.
These attempts remain in the training result's discard audit. No evaluation
records were omitted. Results apply to the same 20 scenes with four samples each.

The 64-group model improves by 6.25 percentage points over the 32-group model
and 21.25 points over the initial model. Outcomes: 24 successes, 48 task failures,
eight program errors. All 80 paired generation/replay records and physical reset
hashes were verified, together with the trained identity's result hash and the
absence of ancestral training/evaluation seed overlap. Optimizer step advanced
from 31 to 63 over the 32 added groups. This stage was reported on 2026-09-09.

Each additional stage contains 32 groups of eight training members. The 64-group
stage adds seeds 37–44 and 65–88. The 96-group stage will restore the completed
64-group checkpoint and add seeds 89–120. Ancestor training seeds are retained
in each protocol so evaluation checks exclude every previously trained scene.
The privileged high-level task, Capsule reward/repair rules, rank-16 LoRA, and
constant `1e-5` learning rate remain the same.

The pinned VeRL loader restores model, optimizer, scheduler, and RNG state. The
launcher checks the parent protocol/result/checkpoint hashes and verifies the
restored optimizer step before sampling. The original base model remains the
reference policy through the existing disabled-adapter reference path.

Remote project: `/root/autodl-tmp/cap-x`.
Orchestrator PID at launch: `171878`.
Run ID: `qwen25coder7b_continue32_to96_20260909_r01`.
Thread follow-up `cube-stack-64-96` is paused after both stage results were verified.

- Logs: `outputs/qwen25coder7b_continue32_to96_20260909_r01/`.
- Driver log: `.codex_cube_stack_continue32_to96_20260909.log`.
- Training: `artifacts/cube_stack_capsule_rl_g64_qwen25coder7b_continue32_to96_20260909_r01/`
  and the corresponding `g96` directory.
- Evaluation: `artifacts/cube_stack_nonprivileged_g64_qwen25coder7b_continue32_to96_20260909_r01/`
  and the corresponding `g96` directory.
- Each stage's `restore_evidence.json` identifies its actual parent checkpoint.
- Each completed evaluation has `summary.json`; the log directory receives
  `comparison.json` after each successful stage.

Evaluation retains the byte-identical original protocol for seeds 45–64, four
samples per scene. Existing base generation/replay records are copied and checked
by the existing paired summarizer. Each new LoRA is independently generated and
replayed 80 times without Controller assistance. These are 20 scenes with four
samples each, not 80 independent scene seeds.

Server validation: all three Python entrypoints compiled; targeted Ruff with
`--no-force-exclude` passed. Actual checkpoint loading produced restored step 31.
No unit tests were added or run. Credentials remain in process environments.

To retain space for both new checkpoints, model shards 2 and 3 were copied to
`/root/capx-model-storage/Qwen2.5-Coder-7B-Instruct/`, verified byte-for-byte by
SHA-256, and linked from their original paths. All existing checkpoints remain.
The resulting data-disk free space was 36GB before the new checkpoints.

Reproduce from the remote project with a fresh run ID and the Controller key
already present in `CAPX_CONTROLLER_API_KEY`:

```bash
.venv/bin/python -m scripts.capsule_rl.run_cube_stack_continuation \
  --parent artifacts/cube_stack_capsule_rl_qwen25coder7b_g32_s05_36_20260908_r01 \
  --baseline-evaluation artifacts/cube_stack_nonprivileged_qwen25coder7b_g32_s45_64_20260908_r01 \
  --run-id qwen25coder7b_continue32_to96_NEW_RUN_ID
```
