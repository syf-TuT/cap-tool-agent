# Qwen2.5-Coder-7B-Instruct: Capsule, maximum 20 steps

Completed 120 attempts on 2026-09-15, 07:42:35–08:16:34 UTC, on the same
SeeTaCloud A800 80GB used for the single-turn baseline.

| Task | Single-turn task success | Capsule final task completion | Capsule strict success |
| --- | ---: | ---: | ---: |
| Cube Lift | 7/20 (35%) | 4/20 (20%) | 1/20 (5%) |
| Cube Stack | 2/20 (10%) | 1/20 (5%) | 0/20 (0%) |
| Cube Restack | 0/20 (0%) | 0/20 (0%) | 0/20 (0%) |
| Spill Wipe | 4/20 (20%) | 6/20 (30%) | 2/20 (10%) |
| Two Arm Lift | 0/20 (0%) | 0/20 (0%) | 0/20 (0%) |
| Two Arm Handover | 0/20 (0%) | 0/20 (0%) | 0/20 (0%) |
| Total | 13/120 (10.83%) | 11/120 (9.17%) | 3/120 (2.50%) |

## Success definitions

Final task completion uses the final `task_completed` flag for finished trials.
Strict success additionally requires `sandbox_rc=0`, matching the previous
report. The existing Capsule implementation keeps its failure flag after an
invalid or failed intermediate action, even if later recovery completes the
task. Eight completed tasks are excluded by that strict criterion. Both
counts are reported without changing the implementation. The baseline was
rechecked: its final task completion and strict success counts are identical.

All 20 requested seeds remain in each denominator. There were 117 finished
trials, two execution exceptions and one trial-budget exhaustion:

- Two Arm Lift seed 13: `TypeError`, `'numpy._ArrayFunctionDispatcher' object is not iterable`.
- Two Arm Handover seed 4: `SyntaxError`, cannot assign to expression, line 9.
- Two Arm Handover seed 15: exceeded the unchanged 450-second trial budget.

These attempts were not replaced. The exception records do not retain complete
Capsule histories; 117 completed trace files were available for step auditing.

## Matched conditions and Capsule settings

- Same base Qwen2.5-Coder-7B-Instruct, no adapter, BF16, vLLM 0.8.5.
- Same high-level non-privileged APIs, environment seeds 1–20 for each task,
  task prompts, perception services, four workers, and no recorded video.
- Temperature 1.0, max output tokens 20480, top_p 0.8, top_k 20,
  repetition penalty 1.1, model context length 32768, model-server seed 20260915.
- Individual requests have no explicit generation seed. Matching scene seeds
  and server seed does not guarantee matched generated programs across methods.
- `agent_mode=capsule`, `capsule_control_mode=llm_step`,
  `max_capsule_steps=20`, semantic groups, max 20 regions per group,
  compact context, require task success before accepting finish,
  checkpoint policy region, rollback policy none, source-region repair feedback.
- Initial program generation is separate from the maximum 20 control steps.
  The same Qwen model generates initial code and control actions.
- Capsule runtime feedback includes reward and task completion, as provided by
  the existing method. Primitive object perception remains non-privileged.
  No visual feedback, image/video differencing, ensemble or oracle was enabled.
- All six generated YAMLs were checked: `env` and `api_servers` equal their
  corresponding previous single-turn source YAMLs. The cube-stack prompt was
  deliberately preserved under the user's instruction to keep other conditions
  unchanged, instead of substituting the usual Capsule prompt-parity prompt.

## Verification and provenance

Remote commit `86d8d77232f977365bccf7d90539f68b9036fcef`.
Both runs' saved source patches have identical SHA-256:
`51df914b6dd2387f0ad2159214c96459f42630bf7fdf95dfead031c27c38a8e2`.
No simulator, primitive or Capsule source was modified for this run.

Verified 120 terminal result records, 2265 model calls, zero model retries,
and no record exceeding 21 calls. Every available completed trace has step IDs
within 1–20. Temporary services exited and GPU memory returned to 0 MiB.

Final task-completed seeds:

- Cube Lift: 6, 7, 9, 14; strict success: 6.
- Cube Stack: 3; strict success: none.
- Spill Wipe: 2, 4, 6, 10, 14, 15; strict success: 6, 10.

## Reproduction and artifacts

Inside the prepared remote Linux checkout:

```bash
bash scripts/benchmarks/run_qwen7b_capsule20_six_tasks_s01_20.sh
.venv/bin/python scripts/benchmarks/summarize_capsule20_six_tasks.py \
  outputs/robosuite_capsule20_qwen7b_nonpriv_s01_20_20260915
```

Set a new `RUN` value before a new execution to preserve these results.

Remote results:
`/root/autodl-tmp/cap-x/outputs/robosuite_capsule20_qwen7b_nonpriv_s01_20_20260915`.

Local bundle:
`remote_results/robosuite/20260915T0742Z_capsule20_qwen7b_s01-20/results.tgz`.
Extracted summary and metadata reside in
`data/outputs/robosuite_capsule20_qwen7b_nonpriv_s01_20_20260915/`
under that directory, as `summary.json` and `run.json`.

Archive SHA-256, verified locally after download:
`ae2a5ef9777683bc3e7650888b2d96c196106b30cdb743c4e49c010d744deb6b`.
Local extracted records independently reproduced the 11 task completions and
three strict successes. Baseline comparison uses
`docs/robosuite-qwen7b-singleturn-six-tasks-20260915.md`.
