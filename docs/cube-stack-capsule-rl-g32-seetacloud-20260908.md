# Cube Stack privileged Capsule-RL, 32 groups

Status: completed, verified on 2026-09-09. Non-privileged high-level evaluation
improved from **7/80 (8.75%)** initially to **19/80 (23.75%)** with the final LoRA,
an observed increase of **15 percentage points** on this evaluation set.

Remote project: `/root/autodl-tmp/cap-x` on `connect.nma1.seetacloud.com:12547`.
Hardware: one A800 80GB. Initial model: local Qwen2.5-Coder-7B-Instruct.
Training uses the existing privileged high-level recipe, rank-16/alpha-32 LoRA,
learning rate `1e-5`, one epoch, seeds 5–36, 32 groups of eight members.
The first seven are ordinary samples; an all-failed group triggers the existing
Capsule Controller/revision procedure. Controller: qwen3.7-plus.
Credentials are supplied only through the process environment.

Training artifacts:
`artifacts/cube_stack_capsule_rl_qwen25coder7b_g32_s05_36_20260908_r01/`.
Logs: `outputs/cube_stack_capsule_rl_qwen25coder7b_g32_s05_36_20260908/`.
Training PID at launch: `111786`; queued evaluation PID: `115075`.

Preparation completed with `groups: 32`. The first completed group (seed 5)
had rewards `[0, 0, 0, 0, 0, 1, 0, 0]` and did not skip its actor update.
This is training progress, not a frozen-policy evaluation result.

All 32 groups completed with 31 actor updates and one skipped update. There were
68 successful training members out of 256. Four groups triggered repair, and
three selected a successful guided member. No groups or attempts were discarded.
The final checkpoint contains 15 files; aggregate SHA-256:
`28eb5ec1b97ac394d6b7920011674e3b4a444467947a6e717f800338855895140`.

The queued evaluation waits for the training process to exit and requires a
completed 32-group result with positive optimizer-step delta. It then starts
SAM3, Contact-GraspNet, and PyRoKi as needed and evaluates the initial model
and final adapter on held-out seeds 45–64, four samples per seed (80 per policy).
Both use the existing non-privileged high-level evaluator and the same frozen
generation protocol. The evaluator verifies matching physical reset hashes.
No Controller assists evaluation.

Evaluation artifacts:
`artifacts/cube_stack_nonprivileged_qwen25coder7b_g32_s45_64_20260908_r01/`.
Final comparison: `summary.json` in that directory, verified against all 80
saved replay records per policy. The evaluation pipeline has exited.

| Policy | Success | Task failure | Program error | Success rate |
| --- | ---: | ---: | ---: | ---: |
| Initial model | 7 | 55 | 18 | 8.75% |
| Final LoRA | 19 | 50 | 11 | 23.75% |

These are four sampled programs on each of 20 held-out scenes, not 80 independent
scene seeds. The measured improvement is specific to this evaluation set.

The reusable queue launcher is
`scripts/capsule_rl/evaluate_cube_stack_after_training.sh`; its shell syntax
was checked on the server using `bash -n`. No unit tests were added or run.

Reproduce preparation with the existing entrypoint, passing a fresh artifact root,
the local model and pinned VeRL paths, and `--seeds` with all integers 5–36.
Then run `scripts.capsule_rl.train_cube_stack train --root <training-root>`.
Queue evaluation from the project directory:

```bash
bash scripts/capsule_rl/evaluate_cube_stack_after_training.sh \
  <training-pid> <training-root> <new-evaluation-root> <log-root>
```

Do not reuse the prior 16-group run's 10% versus 8.75% result as this run's result.
