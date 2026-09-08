# Cube Stack Capsule-RL SeeTaCloud run, 2026-09-08

Status: all requested remote training and evaluation completed. Final training (`r03`) produced
15 optimizer updates and three groups with successful guided repairs. Non-privileged high-level
evaluation measured **8/80 (10%) initially versus 7/80 (8.75%) after training**, a decrease of
1.25 percentage points. This run verified Capsule repair and training but did not improve the
held-out non-privileged success rate.

## Runtime and implementation

- Host workspace: `/root/autodl-tmp/cap-x`, one NVIDIA A800 80GB.
- Synchronized training branch base: `feature/capsule_rl`, commit `6707040`.
- Previous remote uncommitted changes preserved in Git stash
  `before-cube-stack-capsule-rl-20260908` on the previous runtime branch.
- Actor: local Qwen2.5-Coder-7B-Instruct; the Cube Lift Program decoder and LoRA recipe are reused.
- Frozen Controller: `qwen3.7-plus` at `https://coding.dashscope.aliyuncs.com/v1`,
  temperature 0.7, thinking disabled, streaming disabled. Its key is injected only into the
  remote process environment.
- Training preparation: real resets for seeds 5–20; complete environment prompt and per-seed
  initial-state hashes saved under `artifacts/cube_stack_capsule_rl_s05_20_20260908_r01/`.
- The newly added training/evaluation entrypoints passed remote `compileall` and scoped Ruff
  checks. No new unit tests were added.

## Completed frozen privileged sampling

All 16 groups, seeds 5–20 with eight ordinary samples each, completed: **23/128 successes
(17.97%)**, 87 task failures, and 18 program errors. Success counts in seed order were
`1, 1, 1, 2, 0, 2, 3, 2, 2, 2, 1, 1, 1, 1, 3, 0`.

Both **seed 9 and seed 20 produced 0/8 successes**, all task failures without program errors.
Their first seven samples also satisfy the existing Capsule repair trigger. Sampling used
optimizer step 0 before and after, with no restored LoRA adapter. All 128 clean replays completed
in one attempt and matched the prepared physical state hashes. The live worker reported the
declared decoder values, and Ray shutdown completed successfully.

The full audit is `sampling_verification.json`, with `frozen_probe/summary.json`, the effective
VeRL configuration, frozen prompt/decoder protocol, all 128 individual sample records, and
`code_manifest.json` under the training preparation run directory.

## Example all-negative group

Frozen sampling seed 9 produced **0/8 successes**. All eight typed outcomes were
`task_failure`, with no program errors, and shared the same initial-state hash:
`5d6cd20275a7c4188639e7400522d046c2e7370981794b5f3ccdf6228126d5d4`.
The first seven therefore meet the unchanged Capsule repair trigger. Detailed seed-9 evidence is in
`first_all_negative_group.json` and `frozen_probe/sample_009_000.json` through
`frozen_probe/sample_009_007.json` under the training preparation run directory.
These are ordinary model generations; no failure or repair was injected manually.

The first seed-9 program used object-pose quaternions for grasping and placement instead of the
documented grasp-pose API. The failure is an observed baseline result; it is not yet a
Controller repair or a training result.

## Observed repair-capacity failure and correction

The first online attempt (`r01`) completed three groups before being stopped. Seeds 5 and 6
each had one success and updated the actor. Seed 7's first seven ordinary programs all failed;
all four Controller repairs then passed clean replay with reward 1, task completion, and the
same physical reset hash. However, their full revision prompts contained 8,409, 12,762,
17,814, and 16,313 tokens. All exceeded the inherited 8,192-token input limit, so the actor
never generated a guided revision. The ordinary eighth member also failed and the group
correctly skipped its actor update. This is Controller repair success, not guided training success.

The Cube Stack launcher now sets a 24,576-token revision input capacity. Both configuration
validators accept this explicit capacity, and the worker factory propagates it to inference and
batch budgets. Actual reconstruction of the four retained traces passed without truncation.
The resolved worker uses 24,576 prompt plus 4,096 output tokens, totaling 28,672; the remote
model declares a 32,768-token context. The 16 prepared task rows and reset hashes are unchanged.

The second attempt is `artifacts/cube_stack_capsule_rl_s05_20_20260908_r02/`.
Its `capacity_verification.json` records this check. At seed 10, all seven ordinary samples
failed and all four Controller programs passed clean replay, but every actor revision used a
complete Python code fence. The strict raw-Python validator rejected those responses before
replay. This attempt was stopped after nine completed groups and seven actor updates.

The Cube Stack recipe now explicitly accepts a single complete outer Python code fence.
Validation checks the same executable content as clean replay, while preserving the complete
raw response, fence tokens, EOS, response budget, and training mask. It still rejects surrounding
prose, multiple fences, incomplete output, and invalid Python. The strict default for other tasks
is unchanged. Task success and reward rules are unchanged.

The final run is `artifacts/cube_stack_capsule_rl_s05_20_20260908_r03/`.
Both previous attempts retain their groups and `superseded.json` records. The initial-model
evaluation stays frozen, while the trained policy identity binds the `r03` protocol, result,
and checkpoint. Interrupted attempts are not presented as final trained models.

All eight touched Python files passed remote compilation. Scoped Ruff checks found no new
findings compared with `6707040`; ten pre-existing findings remain in shared modules.
The final run's `source_validation.json` records both current and baseline findings. A read-only
review found no blocking issue with raw-token preservation or final-checkpoint identity binding.

## Completed privileged training

The final run completed all 16 scheduled groups, seeds 5–20, with no discarded groups or
discarded attempts. Actual optimizer steps increased from 0 to 15; seed 9's constant-zero
group skipped its update. The 128 training members contain 125 ordinary programs and three
selected guided revisions, with 28 total successes. These on-policy training rewards are not
a frozen-model evaluation score.

The 125 ordinary replays yielded 25 successes, 85 task failures, and 15 program errors.
All 16 Controller replays and nine actor revision replays have complete typed records; every
replay used one attempt and the prepared scene's reset hash. `final_training_verification.json`
also confirms checkpoint file count, optimizer delta, launch-source hashes, and evaluation
binding to the final training result.

| All-seven-failed scene | Controller clean successes / 4 | Actor revision clean successes / 4 | Guided member selected | Actor update |
| --- | ---: | ---: | --- | --- |
| 9 | 0 | 0 | No | Skipped |
| 13 | 3 | 3 | Yes | Yes |
| 16 | 4 | 4 | Yes | Yes |
| 17 | 2 | 2 | Yes | Yes |

Thus three of four naturally triggered repair groups gained a successful eighth member.
All nine successful Controller programs also produced successful actor revisions; there were
no revision format or capacity rejections in the completed run.

The final checkpoint contains 15 files and has aggregate SHA-256
`049682bf056ec0e3e20d857905fc399149efc4b80cad8f203d7f33176395fb49`.
It is under `r03/training/checkpoints/cube_stack_capsule_rl_s05_20_20260908_r03-cf5cfedc2e/final/actor/`.
The evaluation loads its `lora_adapter/`, whose weight SHA-256 is
`edb83aec3fdbb7b38e8ef2c10fd064c0c3c6a15fbb796e20a6f8da1df5d5eab7`.
The trained policy identity binds the actual `r03` protocol, result, and adapter hashes to the
unchanged evaluation protocol. VeRL provenance remained clean at the pinned commit throughout.

## Verified all-failed group repaired during training

In `r03`, seed 13's first seven ordinary samples were all `task_failure`, without program
errors. Three of four repair trajectories produced both a successful Controller program and
a successful actor revision. The first successful revision was selected as the eighth member;
the group rewards were `[0, 0, 0, 0, 0, 0, 0, 1]`, and the actor updated.

The selected Controller program and actor revision each passed one independent clean replay,
with reward 1, task completion, no execution error, and no truncation. All seven failures and
both successful replays shared initial-state hash
`0ea289458696dd3d03cb00367083b2f6626f7096e369541c57eea2ed9d1d00ed`.

The actor's original response includes a complete Python fence. Its stored response, selected
revision, and replay source share SHA-256
`fb78c4c1d19e4a05b72f2e8e2b59a044003eb9f974ff368e7e099d679ed80c31`.
The guided mask has counts `[0, 0, 0, 0, 0, 0, 0, 567]`, exactly covering the raw response
and EOS. Training uses the original task prompt; the repair history is only the revision
generation context. This audit is saved in `r03/first_guided_success_verification.json`.

## Paired non-privileged high-level evaluation

Run: `artifacts/cube_stack_nonprivileged_base_vs_trained_s25_44_20260908_r01/`.

| Policy | Environment seeds | Programs | Successes | Success rate |
| --- | --- | ---: | ---: | ---: |
| Initial model | 25–44, four samples each | 80 | 8 | 10% |
| Trained adapter, 15 optimizer updates | Same frozen protocol | 80 | 7 | 8.75% |

Observed difference: **-1.25 percentage points**. The trained model had 61 task failures and
12 program errors. Among the 80 matched pairs, both policies succeeded twice, only the initial
model succeeded six times, only the trained model succeeded five times, and both failed 67 times.
This small run provides no evidence of improved non-privileged success after training.

The initial model produced 80 EOS-terminated programs. The 72 unsuccessful executions were
54 task failures and 18 program errors. Errors included missing quaternion arguments, incorrect
tuple unpacking, undefined extent variables, and invalid array dimensions. The completed records
contained no HTTP server errors. Repeated resets for each scene had identical physical hashes.
No direct simulator-state access was found in the generated programs.

The trained model also produced 80 EOS-terminated programs. Across all 160 programs, the source
audit found only NumPy imports, no syntax errors, and no direct simulator-state or dynamic-code
access references. The final paired summary verified all scene seeds, generation seeds, and
physical initial-state hashes. The frozen evaluation protocol and initial-model identity hashes
are unchanged, and no infrastructure failures were recorded.

This uses the original single-program Cube Stack task prompt, `privileged: false`, and
`FrankaControlApi`, with SAM3 and Contact-GraspNet providing perception. No Controller assists
evaluation. Success requires a completed terminal stack, reward at least 1, no execution error,
and no exhausted simulator budget. Each matched pair uses the same scene and generation seed.
These are 80 sampled programs on 20 scenes, not 80 independent environment seeds.

The initial-model evidence archive is stored locally at
`remote_results/cube_stack/20260908T0558Z_eval_qwen25coder7b_s25-44/results.tgz`.
SHA-256: `2c69d0e8f92f218fdcd90eb24ac95914ce931d65e79631cce2f2c27e1ef016a4`.
The extracted `base_summary.json`, frozen protocol, generation identities, and evaluation records
are under that run's `data/artifacts/` directory.

See [the reproducible commands](cube-stack-capsule-rl.md) for preparing, training, and evaluating
both policies. The complete local result bundle, including the final LoRA adapter, uses
`remote_results/cube_stack/20260908T0904Z_capsule-rl_qwen25coder7b_s05-20_r03/`.
Its `run.json` records the archive hash and code commit; extracted evidence is under `data/`.
The full 15.9 GB training checkpoint remains on the server. No videos or new unit-test files were
created for this workflow.
