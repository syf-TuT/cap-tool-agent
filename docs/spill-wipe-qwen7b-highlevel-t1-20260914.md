# Qwen2.5-Coder-7B-Instruct Spill Wipe evaluation, temperature 1

Completed on 2026-09-14 on the SeeTaCloud A800 80GB server. Each condition
used environment seeds 1–100 and one sampled program per seed.

| High-level primitive condition | Successes | Success rate | Mean reward | Program errors |
|---|---:|---:|---:|---:|
| Privileged | 16/100 | 16% | 0.3635 | 32 |
| Non-privileged | 15/100 | 15% | 0.3380 | 39 |

Success requires `run_outcome == "finished"`, `task_completed == true`, and
`sandbox_rc == 0`. All 200 trials finished, with 100 model calls and zero
retries in each condition. Each condition has 100 combined videos. No
infrastructure failures or trial timeouts occurred. Program errors count as
failures; mean reward includes partial progress.

The observed difference is one percentage point. One sampled program per seed
does not establish a reliable advantage for either condition. The earlier
temperature-0.7 run reported 19% privileged and 14% non-privileged; this run
alone does not establish a temperature effect.

## Protocol

- Model: local `Qwen2.5-Coder-7B-Instruct`, BF16, vLLM 0.8.5.
- Sampling: temperature **1.0**, top-p 0.8, top-k 20, repetition penalty 1.1,
  maximum 4096 output tokens. All 200 vLLM request records confirm temperature 1.0.
- Model server seed: 20260912; one worker, privileged then non-privileged.
  Model sampling seeds were not independently paired per environment seed.
- Privileged config: `env_configs/spill_wipe/franka_robosuite_spill_wipe_privileged.yaml`,
  exposing `FrankaControlSpillWipePrivilegedApi`.
- Non-privileged config: `env_configs/spill_wipe/franka_robosuite_spill_wipe.yaml`,
  exposing `FrankaControlSpillWipeApi`; SAM3 provides object localization.
- Both use Pyroki. No oracle, training, visual feedback, or program repair.
- Remote base commit: `86d8d77232f977365bccf7d90539f68b9036fcef`, with existing
  working-tree changes. The Spill Wipe seed patch is included in the bundle.
- Seed regression validation: `MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false
  .venv/bin/python -m pytest tests/integrations/test_robosuite_spill_wipe_seed.py -q`
  returned **1 passed**, verifying reproducible resets and equality across conditions.

## Reproduction and results

Run from the prepared Linux checkout with free ports 8114, 8116, and 8120 and
a fresh output directory:

```bash
bash scripts/spill_wipe/run_qwen7b_highlevel_t1_s01_100.sh
.venv/bin/python scripts/spill_wipe/summarize_highlevel.py \
  outputs/spill_wipe_highlevel_qwen7b_t1_s01_100_20260914
```

Remote output:
`/root/autodl-tmp/cap-x/outputs/spill_wipe_highlevel_qwen7b_t1_s01_100_20260914`.

Local result directory:
`remote_results/spill_wipe/20260914T0153Z_highlevel-t1_qwen7b_s01-100/`.
The bundle contains protocol, source configs, reproduction scripts, logs,
per-trial JSON records, generated programs, and videos.

Bundle SHA-256:
`618a61b9bffb6cc250a04f2e04c185a2305734cac150553aadaaccc49aca579d`.

Privileged success seeds:
`4, 9, 34, 38, 39, 42, 46, 59, 66, 71, 75, 82, 87, 93, 95, 96`.

Non-privileged success seeds:
`1, 23, 26, 28, 33, 34, 38, 39, 44, 64, 76, 77, 82, 90, 93`.
