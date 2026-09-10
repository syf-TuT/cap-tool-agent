# Cube Lift C16 with undamped guided log probability

Cube Lift preparation now selects `guided_objective: log_probability` in both
the runtime and VeRL actor configurations. Guided tokens use `-A * log(p)`;
the group baseline remains the equal mean of eight rewards. At equal advantage
and averaging, the guided derivative with respect to log probability is `-A`,
matching ordinary PPO at ratio one before clipping. This does not imply equal
parameter-gradient norms or optimizer contributions. Reference KL is unchanged.
Cube Stack and other tasks retain `probability_shaping` by default.

The completed non-privileged evaluation achieved **35/80 (43.75%)**, compared
with the original A16's **41/80 (51.25%)**. The observed difference is **-7.5
percentage points**. Removing attenuation worked mechanically, but this run
did not improve evaluation success.

## Run

Remote project: `/root/autodl-tmp/cap-x`.
Run directory:
`artifacts/cube_lift_c16_logprob_qwen25coder7b_s05_20_20260910_r02`.
Logs use the same slug under `outputs/`.

- Program model: Qwen2.5-Coder-7B-Instruct, initialized from the base model.
- Controller: qwen3.7-plus at the existing DashScope coding endpoint; credential
  supplied through `CAPX_CONTROLLER_API_KEY`, never recorded in this document.
- Training: C arm, `any_failed`, seeds 5–20, 16 groups of eight.
- Evaluation: non-privileged seeds 201–220, four samples per seed, 80 trials.
- Only the guided objective was deliberately changed from the prior C recipe.

The first launch (`r01`) exited during typed actor configuration construction,
before collection or an optimizer update. Adding `guided_objective` to
`CapsulePolicyLossConfig` fixed that integration issue. The complete resolved
actor was instantiated successfully on the server before starting `r02`.

## Training result

Training completed in 6328.32 seconds with 16 actor updates, zero skipped
updates, zero discarded groups, and ten selected guided members.

| Training statistic | Original A16 | New C16 |
| --- | ---: | ---: |
| Positive final members | 68/128 | 68/128 |
| Successes among first seven ordinary candidates | 58/112 | 54/112 |
| Successful eighth members | 10/16 | 14/16 |
| Selected guided members | 0 | 10 |
| Actor updates | 15 | 16 |

The original A16 reference is
`artifacts/cube_lift_ac_qwen25coder7b_g16_32_s05_36_20260909_r02/A16`.
Both runs use the same scenes but learn different policies, so later ordinary
program samples are not held fixed. Guided members occupy the eighth slot;
they are not extra samples beyond the eight-member budget.

Checkpoint tree SHA-256:
`97b3d3a0a94a45f756a95b38577e00061d017b44cc1833d5237f4945226aecc2`.

## Evaluation and verification

All 80 samples completed: 35 successes, 29 task failures, and 16 program errors.
The original A16 and new C16 matched on all 80 environment seeds, generation
seeds, sample indices, and initial-state hashes. Both succeeded on 26 samples;
only A succeeded on 15, only C on nine, and both failed on 30. This is one
training run and does not establish a general effect of the new objective.

The run's existing replay audit passed for all 80 samples. A final artifact
audit verified the equal reward baseline in all 16 training groups and that
the recorded implementation source hashes remained unchanged throughout the run.

Local result bundle:
`remote_results/cube_lift/20260910T1118Z_c16-logprob_qwen25coder7b_s05-20/results.tgz`.
The `data/` directory beside it contains the results, source snapshot,
`verification.json`, and `training_comparison.json`. The archive and all 198
manifest-listed files passed SHA-256 verification after download. Large model
checkpoints are retained on the server and excluded from this bundle.

Archive SHA-256:
`32098a84e388a7e13e8281d089ad3f7c0f0695b87b044f9d95e276b6f0cb1851`.

## Reproduction

Use a fresh run directory and the prepared remote environment. Export
`MUJOCO_GL=egl`, `JAX_PLATFORMS=cpu`, `JAX_PLATFORM_NAME=cpu`,
`XLA_PYTHON_CLIENT_PREALLOCATE=false`, `OMP_NUM_THREADS=4`,
`HF_HUB_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1`.

```bash
RUN=artifacts/<new-run-slug>
.venv/bin/python -m scripts.capsule_rl.train_cube_stack prepare \
  --task cube_lift --root "$RUN/C16" --repair-trigger any_failed \
  --model .codex-downloads/models/Qwen2.5-Coder-7B-Instruct \
  --verl .codex-downloads/verl-v0.6.1 \
  --seeds 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20
.venv/bin/python -m scripts.capsule_rl.train_cube_stack train --root "$RUN/C16"
.venv/bin/python -m scripts.capsule_rl.evaluate_cube_stack prepare \
  --root "$RUN/eval_C16" --training-root "$RUN/C16" \
  --seeds 201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220 \
  --samples-per-seed 4
.venv/bin/python -m scripts.capsule_rl.evaluate_cube_stack generate \
  --root "$RUN/eval_C16" --policy trained_lora
# Ensure SAM3, Contact-GraspNet and Pyroki services are ready on 8114–8116.
.venv/bin/python -m scripts.capsule_rl.evaluate_cube_stack evaluate \
  --root "$RUN/eval_C16" --policy trained_lora
```

Remote autograd verified both objective coefficients, including negative and
zero advantages. The new guided derivative equals `-A / token_count`; the
default objective retains its original sigmoid derivative. The loss and typed
config passed targeted Ruff checks. No test files were added.
