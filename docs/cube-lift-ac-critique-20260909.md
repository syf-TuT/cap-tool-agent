# Cube Lift A/C critique ablation, 16 and 32 groups

## Results (completed 2026-09-10)

At 32 cumulative groups, A and C both achieved **50/80 (62.50%)** on the
non-privileged evaluation. At 16 groups, A achieved 41/80 and C achieved 38/80.
This paired training replicate provides no observed evaluation gain from C.

| Checkpoint | Success | Task failure | Program error | C minus A | Paired scene-bootstrap 95% interval |
| --- | ---: | ---: | ---: | ---: | --- |
| A16 | 41/80 (51.25%) | 26 | 13 | — | — |
| C16 | 38/80 (47.50%) | 32 | 10 | -3.75 pp | [-17.5, +10.0] pp |
| A32 | 50/80 (62.50%) | 21 | 9 | — | — |
| C32 | 50/80 (62.50%) | 22 | 8 | 0.00 pp | [-11.25, +11.25] pp |

All 64 accepted training groups had at least one success among the first seven
ordinary programs. Thus the previous `all_failed` trigger would not activate
on any of these recorded groups; this is not a separately trained all-failed arm.
C triggered on 27 mixed-success groups and selected 17 successful revisions.
In C's second stage, the initial seven-program success rate reached 83.04%; five
groups had seven successes and correctly skipped critique.

### Training behavior and cost

The 32-group columns are cumulative across both 16-group stages.

| Metric | A16 | C16 | A32 cumulative | C32 cumulative |
| --- | ---: | ---: | ---: | ---: |
| Repair-triggered mixed groups | 0/16 | 16/16 | 0/32 | 27/32 |
| Guided positive members | 0 | 9 | 0 | 17 |
| Positive final members | 68/128 | 57/128 | 170/256 | 164/256 |
| Actual actor updates | 15 | 16 | 28 | 28 |
| Constant-reward update skips | 1 | 0 | 4 | 4 |
| Discarded collection attempts | 0 | 0 | 0 | 0 |
| Clean-replay attempts in accepted groups | 128 | 193 | 256 | 361 |
| Program generation requests | 128 | 131 | 256 | 265 |
| Program prompt/output tokens | 79,488 / 22,956 | 150,827 / 23,544 | 158,976 / 46,459 | 324,818 / 50,202 |
| Controller requests | 0 | 442 | 0 | 658 |
| Controller prompt/output tokens | 0 / 0 | 999,340 / 81,040 | 0 / 0 | 1,492,091 / 122,330 |
| Training wall time (minutes) | 52.86 | 92.25 | 107.11 | 171.49 |

C used **1.60 times** A's cumulative training wall time and had the same 28
effective actor updates. Costs include training startup, sampling, repair,
updates, and checkpoint saving, but exclude separate preparation and evaluation.
All Controller responses reported usage. Token counts are not monetary charges.
The comparison measures the complete recipes at equal group counts, without
isolating critique text from repair sampling or additional computation.

## Protocol

This experiment reuses the Cube Stack A/C collector, trainer, and paired evaluation
with the Cube Lift task. A uses `repair_trigger=never`: eight ordinary programs and
no Controller calls. C uses `repair_trigger=any_failed`: a known failure among the
first seven ordinary programs triggers the existing repair procedure. Only failed
programs are eligible; the first successful clean-replayed revision occupies slot
eight, otherwise an ordinary program fills that slot. Original failures remain.

Both arms start from Qwen2.5-Coder-7B-Instruct, using rank-16/alpha-32 LoRA,
learning rate 1e-5, temperature 0.7, top-p 0.8, top-k 20, repetition penalty 1.1,
and 4096 output tokens. C uses the frozen qwen3.7-plus Controller, gamma 0.1,
and the same 24576-token revision input capacity as the Stack A/C recipe.

Training uses the privileged high-level Cube Lift clean-replay environment. The
dataset captures the complete prompt and initial-state hash from real resets.
The current 2534-character prompt includes `home_pose`; its SHA-256 is
`5c73be784b7fff8f697b30c090d4cb4a6d8be45a7947351efce17d085ee6676a`.
This differs from the old five-API prompt hash in the smoke YAML; preparation
records the actual environment prompt rather than using that stale hash.

A16/C16 use seeds 5–20. A32/C32 restore their own 16-group model, optimizer,
and extra state, then train seeds 21–36 for 32 cumulative groups per arm.
All four checkpoints use persistent disk storage. A valid stage with only
constant-reward groups may perform zero updates; evaluation accepts its completed
checkpoint and reports the actual update count.

Each checkpoint is evaluated without Controller help in the non-privileged
Cube Lift `FrankaControlApi` environment: seeds 201–220, four matched generations
per scene, 80 trials per checkpoint. Scene, generation seed, and initial physical
state must match between A and C. The 32-group comparison is primary; the paired
scene-bootstrap interval describes scene uncertainty for one training replicate.
Equal group counts do not imply equal sampling or compute costs.

## Reproduction

Run in the remote project `/root/autodl-tmp/cap-x` after Pyroki is ready on port
8116. Provide `CAPX_CONTROLLER_API_KEY` through the process environment.

```bash
.venv/bin/python -u -m scripts.capsule_rl.run_cube_stack_ac \
  --task cube_lift \
  --run-id cube_lift_ac_qwen25coder7b_g16_32_s05_36_20260909_r02 \
  --model /root/autodl-tmp/cap-x/.codex-downloads/models/Qwen2.5-Coder-7B-Instruct \
  --verl /root/autodl-tmp/cap-x/.codex-downloads/verl-v0.6.1
```

The shared launcher and training entrypoint accept `--task cube_lift`; omitted
task arguments preserve Cube Stack defaults. Evaluation selects the environment
from the saved training protocol. Experiment records and checkpoints are under
`artifacts/<run-id>/`, with logs under `outputs/<run-id>/`. Raw outputs and weights
are not committed.

## Actual-flow validation

The three modified Python entrypoints passed remote syntax compilation and
targeted Ruff checks. No test files were added and no test suite was run.
All four training stages and all 320 evaluation records completed. The existing
`audit_training` and `compare_evaluations` checks validate eight-member groups,
repair provenance, held-out seeds, and matched generation/physical reset identity.
A made zero Controller calls. C exercised six single-failure initial groups,
successful repairs, ten exhausted-repair fallbacks, and five all-success groups
that correctly avoided repair. No initial all-failure group occurred.

A32 restored optimizer step 15 from A16 and finished at 28; C32 restored step 16
from C16 and finished at 28. No entire stage had zero updates, so the relaxed
zero-update evaluation eligibility was reviewed but not exercised by this run.
The superseded r01 startup is excluded from all reported results and costs.

Final revalidation passed for all 64 groups, all 320 paired replay records,
all four complete checkpoint tree hashes, and the executed source manifest.
The verification evidence is `records/final_verification.json` in the local
delivery directory below.

## Saved results

Local delivery: `remote_results/cube_lift_ac_qwen25coder7b_g16_32_s05_36_20260909_r02/`.
The archive contains 778 record, log, and executed-source files, excluding model
weights. `archive_manifest.json` records per-file hashes. The archive SHA-256 is
`9eb63d7a5227c75c90b49ba96cbca9cf1182a2a8fb7c68bfff33c01efcf8145d`.
The extracted comparison is `records/comparison.json`; each stage also retains
its protocol, group records, training result, and evaluation records.

All four complete checkpoints remain on the remote persistent disk under
`artifacts/<run-id>/<label>/training/checkpoints/`. Their verified tree hashes are:

| Checkpoint | SHA-256 |
| --- | --- |
| A16 | `5655cd597e9e8519ec314e40236dec85d26bf188618fd122ff72d6dfcb88b16f` |
| C16 | `81ccd0b96e2bcf702355ccafb767ade3abdbcdfc573acc5407d57f3c55e662a1` |
| A32 | `f14e06f6b6c27abef009fea2aba234a0f6a319f5fc943940cebe02165c2e23f6` |
| C32 | `296003d4d392b31c2472515644dcc8f3c5db9f04220fa3da0fbd03959c6ae6aa` |
