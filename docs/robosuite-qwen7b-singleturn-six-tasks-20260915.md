# Qwen2.5-Coder-7B-Instruct: six Robosuite tasks

Completed on 2026-09-15 on the requested SeeTaCloud host, NVIDIA A800 80GB.

| Task | Successes | Success rate | Program errors |
| --- | ---: | ---: | ---: |
| Cube Lift | 7/20 | 35% | 2 |
| Cube Stack | 2/20 | 10% | 6 |
| Cube Restack | 0/20 | 0% | 3 |
| Spill Wipe | 4/20 | 20% | 9 |
| Two Arm Lift | 0/20 | 0% | 8 |
| Two Arm Handover | 0/20 | 0% | 11 |
| Total | 13/120 | 10.83% | 39 |

## Protocol

- Existing CaP-X `capx/envs/launch.py` single-turn code baseline, standard task
  YAMLs and high-level non-privileged APIs. No oracle, Capsule, repair,
  ensemble, or visual feedback to the code model.
- Each task uses environment seeds 1–20, one generated program per seed,
  four simulator workers. Verified 120 finished trial records, each with
  exactly one model call and zero model retries.
- Base model (no adapter): local `Qwen2.5-Coder-7B-Instruct`, BF16,
  vLLM 0.8.5, server seed 20260915, context length 32768.
- Temperature 1.0 (current launch-entry default), maximum output 20480 tokens.
  Model generation defaults supply top_p 0.8, top_k 20, repetition penalty 1.1.
  Individual requests have no explicit sampling seed; scene seed does not
  imply identical generated code across reruns or worker schedules.
- Success requires `run_outcome=finished`, `task_completed=true`, and
  `sandbox_rc=0`. All 20 attempts remain in each denominator, including code
  errors. Zero infrastructure failures or trial timeouts were recorded.
- Videos disabled; generated code, prompts, telemetry and results retained.
- Trial execution: 07:27:11–07:39:13 UTC (15:27:11–15:39:13 UTC+8).

## Source provenance and interpretation

Remote commit: `86d8d77232f977365bccf7d90539f68b9036fcef`, with pre-existing
working-tree changes saved in `source_changes.patch`. These include Restack
settling/state fixes, Spill Wipe and Two Arm Lift seeding fixes, headless
Two Arm Lift rendering, API reset fixes, and Two Arm Lift grasp-coordinate
documentation. No simulator or primitive source was changed for this run.

This is the original single-generation **method on the current checkout**;
it is not a byte-for-byte reproduction of the paper's release environment.
The six task source YAMLs and runtime logs are included in the result bundle.

Success seeds: Cube Lift `4,7,9,11,15,16,20`; Cube Stack `5,14`;
Spill Wipe `3,4,13,15`; all other tasks none.

## Reproduce and inspect

Run inside the prepared remote Linux checkout:

```bash
bash scripts/benchmarks/run_qwen7b_singleturn_six_tasks_s01_20.sh
.venv/bin/python scripts/benchmarks/summarize_singleturn_six_tasks.py \
  outputs/robosuite_singleturn_qwen7b_nonpriv_s01_20_20260915
```

Use a new `RUN` value for a new execution to preserve this result.

Remote output:
`/root/autodl-tmp/cap-x/outputs/robosuite_singleturn_qwen7b_nonpriv_s01_20_20260915`.

Local bundle directory:
`remote_results/robosuite/20260915T0726Z_singleturn_qwen7b_s01-20/`.
The extracted summary is under
`data/outputs/robosuite_singleturn_qwen7b_nonpriv_s01_20_20260915/summary.json`.

Archive SHA-256:
`16c7638e4d04981bd76582e18a5680facd87e9a2851663c36b539d5d5f75d768`.
Local archive hash and 120 extracted trial records were verified.
All temporary experiment services exited and GPU memory returned to 0 MiB.
