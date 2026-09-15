# Qwen2.5-Coder-7B-Instruct: Spill Wipe high-level evaluation

Completed on 2026-09-12 on the SeeTaCloud A800 80GB server. Each condition
used environment seeds 1–100, with one sampled program per seed.

| Condition | Successful trials | Success rate | Mean reward | Program errors |
|---|---:|---:|---:|---:|
| Privileged high-level primitives | 19/100 | 19% | 0.3457 | 45 |
| Non-privileged high-level primitives | 14/100 | 14% | 0.3780 | 31 |

Success requires a finished execution, `task_completed == True`, and
`sandbox_rc == 0`. The task-completion counts alone are also 19 and 14.
Mean reward includes partial task progress. The observed success-rate difference
is 5 percentage points; this single sampling run does not establish a reliable
advantage across repeated model samplings.

All 200 trials finished. Each condition has exactly 100 model calls, zero model
call retries, and 100 recorded videos. There were no infrastructure failures or
trial timeouts in the reported run. No oracle, Controller, training, visual
feedback, or program repair was used. Non-privileged object localization used
SAM3; both conditions used Pyroki for motion planning.

## Protocol and configurations

- Model: local `Qwen2.5-Coder-7B-Instruct`, BF16, served with vLLM 0.8.5.
- Sampling: temperature 0.7, top-p 0.8, top-k 20, repetition penalty 1.1,
  maximum 4096 output tokens. Top-p, top-k, and repetition penalty come from the
  model generation configuration and were verified in vLLM request logs.
- Model server seed: 20260912; one worker, privileged condition followed by
  non-privileged condition. Model generation seeds were not independently paired
  per environment seed.
- Privileged config:
  `env_configs/spill_wipe/franka_robosuite_spill_wipe_privileged.yaml`,
  exposing `FrankaControlSpillWipePrivilegedApi`.
- Non-privileged config:
  `env_configs/spill_wipe/franka_robosuite_spill_wipe.yaml`,
  exposing `FrankaControlSpillWipeApi`.
- Remote base commit: `86d8d77232f977365bccf7d90539f68b9036fcef`, with local
  modifications. The task-specific seed patch and source snapshot are included
  with the results; unrelated working-tree changes were left in place.

Before evaluation, a reset bug was found: the Spill Wipe wrapper updated its own
RNG but did not reseed Robosuite. The wrapper now calls the existing
`reseed_robosuite_owner` helper before resetting. The integration test failed
before this correction and passed afterward. It verifies robot joint state and
spill marker positions for the sequence 1, 2, 1, and equality between privileged
and non-privileged conditions. Validation command:

```bash
MUJOCO_GL=egl .venv/bin/python -m pytest \
  tests/integrations/test_robosuite_spill_wipe_seed.py -q
```

The first model service startup failed before any trial because JAX preallocated
GPU memory. `XLA_PYTHON_CLIENT_PREALLOCATE=false` resolved this startup issue.
These preflight attempts are not included in the 200 trials.

## Reproduction and artifacts

Run from the prepared Linux checkout, with unused service ports 8114, 8116, and
8120 and a fresh output directory:

```bash
bash scripts/spill_wipe/run_qwen7b_highlevel_s01_100.sh
.venv/bin/python scripts/spill_wipe/summarize_highlevel.py \
  outputs/spill_wipe_highlevel_qwen7b_s01_100_20260912
```

Remote output: `/root/autodl-tmp/cap-x/outputs/spill_wipe_highlevel_qwen7b_s01_100_20260912`.

Local result directory:
`remote_results/spill_wipe/20260912T0324Z_highlevel_qwen7b_s01-100/`.
It contains `results.tgz`, `run.json`, and extracted `data/` with `summary.json`,
protocol, logs, generated programs, all 200 videos, model weight/config hashes,
and reproduction files. The result bundle SHA-256 is
`f846474fae5101262f59ac767dfa802f7bd970145d50744592c44ddadc6f30e4`.

Privileged success seeds:
`2, 10, 11, 15, 24, 31, 32, 35, 43, 48, 51, 55, 76, 80, 81, 83, 86, 88, 94`.

Non-privileged success seeds:
`2, 9, 13, 15, 33, 44, 56, 65, 68, 74, 76, 79, 86, 97`.
