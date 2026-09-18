# Cube Restack SFT expansion

## Motivation and completed pilot

The ordinary GRPO pilot completed 16 groups with 0/128 positive members and no
actor updates. The critique arm was stopped at the user's request. A glm-4.7
teacher then collected 32 distinct successful programs in 78 attempts, covering
24 scene seeds. Three epochs of rank-16/alpha-32 LoRA SFT produced 24 optimizer
steps. Assistant tokens alone contributed to the supervised loss.

On privileged Restack seeds 201–220, four samples per seed, that SFT adapter
achieved 1/80 successes, versus the base actor's 0/80. Program errors increased
from 17 to 29. Common errors were undefined variables/imports, array shapes, and
incorrect API use. This pilot does not establish a reliable improvement.

## Expanded protocol

- Teacher: glm-4.7 through the existing DashScope endpoint, credentials from
  `CAPX_CONTROLLER_API_KEY`. Actor: Qwen2.5-Coder-7B-Instruct.
- Collect 128 distinct successful programs, at most 512 attempts, cycling through
  training seeds 5–132. Keep all generation/replay records. Insufficient success
  count stops the pipeline instead of silently training on fewer examples.
- Teacher receives the initial green-on-red scene description and the need to
  unstack before restacking. Actor SFT and evaluation retain the original prompt;
  teacher-only context is recorded separately, never included as SFT input.
- Train from the base actor for six epochs, learning rate 2e-5, effective batch
  four, rank 16, alpha 32, assistant-only loss. Save epochs 1, 3, and 6.
- Select the checkpoint with the most successful replays on validation seeds
  201–220, four samples each. Break ties by the earlier epoch.
- Evaluate the selected checkpoint once on untouched seeds 301–340, four samples
  each (160 trials). Record both successful trials and distinct successful scenes.
  The observed target is at least 5% success across more than one scene; this is
  an empirical target, not a guarantee or a confidence-bound claim.
- No critique or teacher scene hint is used for actor evaluation. The environment
  is privileged Restack, so results do not imply non-privileged visual performance.

## Reproduction

Run from `/root/autodl-tmp/cap-x` with Pyroki available on port 8116:

```bash
.venv/bin/python -u -m scripts.capsule_rl.run_restack_sft_expansion \
  --root artifacts/cube_restack_sft128_glm47_20260910 \
  --model .codex-downloads/models/Qwen2.5-Coder-7B-Instruct
```

Use a fresh root for changed source or inputs. The driver writes immutable input
hashes, per-phase logs, mutable status, checkpoint selection, and the final result.
Collection resumes from saved records. SFT has no optimizer-state resume; after
an interrupted training stage, retain its evidence and use a fresh training root.

The expanded experiment is running. Remote syntax compilation and targeted Ruff
checks passed; no test files or unrelated test suites were added. Final metrics
must be taken from the completed `summary.json`, not inferred from training loss.

The new epoch-save/load and configurable SFT-only evaluation path was also
executed on the server using eight existing successful programs, one SFT epoch,
and four replays on seed 221. All phases completed and the adapter was loaded
successfully. Replay success was 0/4; this is functional verification, not an
effectiveness result. Its independent records are under
`artifacts/cube_restack_sft128_flow_validation_20260910/`. Seed 221 is excluded
from both checkpoint selection and the final test.

## Authorized storage cleanup

The previous Restack experiment was stopped. Only the full model and LoRA weight
files of the 20260909 Cube Lift A16/C16/A32/C32 run were removed, releasing
57.95 GiB. Logs and other records remain; those old checkpoints are no longer
complete/restorable. The remote deletion manifest is
`artifacts/cube_lift_weights_cleanup_20260910.json`.
