# Spill Wipe C48 continuation

C48 achieved 23/80 (28.75%) evaluation successes, versus C32's 12/80
(15%). The paired difference is +13.75 percentage points; the paired-scene
bootstrap 95% interval is [+1.25, +26.25] points. This is adaptive continuation
on the same evaluation scenes after inspecting C32, not an independent holdout.

| Evaluation | Success | Program error | Task failure |
| --- | ---: | ---: | ---: |
| C32 | 12 | 39 | 29 |
| C48 | 23 | 36 | 21 |

The Qwen2.5-Coder-7B-Instruct LoRA policy resumed C32's model, optimizer,
RNG and scheduler state at optimizer step 28, finishing at step 44.
The frozen glm-5 controller used the any-failure critique rule. Training and
evaluation temperature remained 0.7. The additional 16 groups used seeds
37–52, bringing cumulative training seeds to 5–52. All 16 groups updated the
actor, with no skipped updates or discarded groups. Final training members
included 47 successes out of 128; this is separate from evaluation success.
Training took 13,192 seconds (about 3 hours 40 minutes).

Evaluation used seeds 201–220 with four samples per seed, the same protocol
as C32. Training used the privileged API and evaluation the nonprivileged API.

Run from `/root/autodl-tmp/cap-x`, with `CAPX_CONTROLLER_API_KEY` set securely
and Pyroki available on port 8116:

```bash
.venv/bin/python -m scripts.spill_wipe.run_c32 --groups 48 \
  --parent artifacts/spill_wipe_c32_glm5_qwen7b_20260912/C32 \
  --baseline-evaluation artifacts/spill_wipe_c32_glm5_qwen7b_20260912/eval_C32 \
  --run-id spill_wipe_c48_glm5_qwen7b_20260914
```

Remote results: `artifacts/spill_wipe_c48_glm5_qwen7b_20260914`.
Logs: `outputs/spill_wipe_c48_glm5_qwen7b_20260914`.
Checkpoint: `C48/training/checkpoints/C48-14189a7fdb/final/actor` within the
result directory. Checkpoint SHA-256:
`49263dceb49a7563c0d0b609010d15323c082e694ceb3e6a917bd7301855129e`.

The continuation launcher now accepts `--groups 32` (default) or `--groups 48`,
validates parent ancestry and optimizer restoration, and compares against the
selected parent's evaluation. Validation uses actual remote training and all
80 evaluation records; no additional tests were added.
