# Cube Lift selected repairs: non-privileged paired replay

## Result

All 17 selected successful revisions from the C16/C32 training run were replayed
against their actual original failed programs, on the original scene seeds with
non-privileged perception. No model was trained and no new programs were generated.

| Archived program | Non-privileged success | Task failure | Program error |
| --- | ---: | ---: | ---: |
| Original P0 | 1/17 (5.88%) | 15 | 1 |
| Selected revision | 11/17 (64.71%) | 6 | 0 |

The paired success difference is **+58.82 percentage points**. Ten pairs changed
from failure to success, one succeeded on both sides, six failed on both sides,
and none changed from success to failure. Nine improvements were task failures
becoming successes; one fixed a missing `np` import. These results support transfer
of some repair improvements to non-privileged perception, rather than an effect
limited entirely to privileged execution.

The six revisions that failed non-privileged replay correspond to seeds
6, 10, 12, 32, 33, and 36. All six executed without a program error but failed the
task-completion criterion; the outcome labels alone do not establish the physical
cause. All 17 revisions had previously passed privileged clean replay, by selection.

| Source training stage | Original success | Revision success |
| --- | ---: | ---: |
| C16, 9 pairs | 0/9 | 6/9 |
| C32, 8 pairs | 1/8 | 5/8 |

This diagnosis does not establish that the trained C policy learned these changes.
It narrows the explanation of the prior A/C evaluation tie: lack of any useful
repair data is inconsistent with these paired observations, while incomplete
perception transfer and the strength of the learning signal remain possible
limitations.

## Protocol and limitations

- Source experiment: `cube_lift_ac_qwen25coder7b_g16_32_s05_36_20260909_r02`.
- Extract only the 17 selected `critique_guided_revision` members. Resolve each P0
  through its selected repair attempt's `p0_program_sample_id`; retain raw sources
  and archived source hashes. No unsuccessful or unselected revision is included.
- Use the saved non-privileged Cube Lift evaluation configuration with
  `FrankaControlApi`, SAM3, Contact-GraspNet, and Pyroki. The Controller is unused.
- Execute each program once after a fresh reset to its original training seed.
  The existing evaluation harness normalizes code fences, rejects infrastructure
  failures, and requires a clean execution, task completion, terminal reward >= 1,
  and no truncation for success.
- All 17 pairs have matching physical initial-state hashes. This does not guarantee
  identical stochastic perception/grasp outputs across calls. Originals ran first,
  followed by revisions; this is one replay per program, not repeated estimation.
- These are selected successful repairs on their original scenes, not a random
  sample of generated programs, new-scene generalization, or a policy evaluation.
  The 64.71% rate must not be compared directly with the earlier policy's 62.50%.

## Reproduction and evidence

Prestart the three perception/planning services, then run from the remote project:

```bash
MUJOCO_GL=egl JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 \
.venv/bin/python -u -m scripts.capsule_rl.replay_cube_lift_repairs \
  --training-root artifacts/cube_lift_ac_qwen25coder7b_g16_32_s05_36_20260909_r02 \
  --root artifacts/cube_lift_c_repairs_nonprivileged_original_seeds_20260910_r01
```

Use a new output root for an independent replay. The `generations/` directory
contains archived source inputs for compatibility with the existing replay
harness; its identity explicitly records that no new generation occurred.

Remote validation confirmed all 34 source identities against the 17 archived
training groups, all 17 paired physical states, and every recorded outcome.
The script passed remote syntax and targeted Ruff checks. No tests were added
or test suite run. The two perception services started for this diagnosis were
stopped afterward; the pre-existing Pyroki service was left running.

Local delivery: `remote_results/cube_lift_c_repairs_nonprivileged_original_seeds_20260910_r01/`.
The archive contains 100 input, result, log, selected training-group, and executed
source files. Its SHA-256 is
`e390a7f3de9eab01ec11dfa59dd9dfa5a60b6e51de8451868ad3ae8713ad227d`.
`records/summary.json` contains every pair's outcomes and initial-state hash;
`records/verification.json` records the source and pairing checks.
