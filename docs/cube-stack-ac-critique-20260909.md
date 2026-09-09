# Cube Stack A/C critique ablation, 16 and 32 groups

## Results (completed 2026-09-09)

At the primary 32-group checkpoint, C achieved **25/80 (31.25%)** versus
A's **14/80 (17.50%)**, a **13.75 percentage-point** gain. At 16 groups both
achieved 10/80 (12.50%). All evaluations use non-privileged perception and no
Controller. These are results from one paired training replicate, not evidence
of robustness across independent training seeds.

| Checkpoint | Success | Task failure | Program error | C minus A | Paired scene-bootstrap 95% interval |
| --- | ---: | ---: | ---: | ---: | --- |
| A16 | 10/80 (12.50%) | 52 | 18 | — | — |
| C16 | 10/80 (12.50%) | 56 | 14 | 0.00 pp | [-10.0, +10.0] pp |
| A32 | 14/80 (17.50%) | 56 | 10 | — | — |
| C32 | 25/80 (31.25%) | 47 | 8 | +13.75 pp | [+5.0, +22.5] pp |

Intervals resample the 20 paired scenes, keeping each scene's four generations
together (10,000 bootstrap draws, seed 20260909). They quantify scene uncertainty
conditional on this training pair. This experiment compares the complete A/C
recipes at equal group counts; it does not isolate critique text from additional
sampling, compute, or the increase in effective actor updates. It also does not
compare C directly with the old `all_failed` policy.

### Training behavior and cost

The 32-group columns below are cumulative across both 16-group stages.

| Metric | A16 | C16 | A32 cumulative | C32 cumulative |
| --- | ---: | ---: | ---: | ---: |
| Repair-triggered groups | 0/16 | 16/16 | 0/32 | 32/32 |
| Guided positive members | 0 | 15 | 0 | 31 |
| Mixed-success groups triggering repair | 0 | 14 | 0 | 30 |
| Mixed-success groups receiving a guided positive | 0 | 13 | 0 | 29 |
| Positive final members | 22/128 | 39/128 | 51/256 | 97/256 |
| Actual actor updates | 15 | 16 | 27 | 32 |
| Constant-reward update skips | 1 | 0 | 5 | 0 |
| Discarded collection attempts | 1 | 1 | 2 | 2 |
| Clean-replay attempts in accepted groups | 128 | 219 | 256 | 444 |
| Program generation requests | 130 | 157 | 264 | 320 |
| Program prompt/output tokens | 98,670 / 42,756 | 491,993 / 56,631 | 200,376 / 88,900 | 1,047,386 / 117,153 |
| Controller requests | 0 | 602 | 0 | 1,127 |
| Controller prompt/output tokens | 0 / 0 | 1,922,066 / 239,869 | 0 / 0 | 3,594,141 / 415,263 |
| Training wall time (minutes) | 73.70 | 175.27 | 147.63 | 334.25 |

C used **2.26 times** A's cumulative training wall time. Wall time includes
training-runtime startup, sampling, repair, updates, and checkpoint saving; it
excludes separate preparation, evaluation, and archival. API counters describe
transport calls and returned token usage, not a monetary bill. All Controller
responses reported usage. Generation counters include discarded attempts;
the clean-replay row covers accepted groups only. Each stage discarded one
collection attempt for the existing decode/retokenize round-trip guard and then
successfully recollected its scheduled group.

`status.json` names its mixed-trigger counter `mixed_groups_repaired`; that field
counts mixed groups **triggering** repair, including unsuccessful repair rounds.
The table above separates triggering from actually selecting a guided positive.

### Actual-flow validation

- Revalidated all 64 saved groups with `audit_training` and all 320 replay records
  with `compare_evaluations`. All groups have eight members. Training ancestry
  (5–36) and evaluation scenes (201–220) are disjoint; paired scene, generation
  seed, and physical initial-state hashes match between A and C.
- A made zero Controller calls, including its all-failure groups. C16 seed 5
  triggered on mixed outcomes, kept the original failures, and selected just the
  first successful clean-replayed repair for slot eight. C16 seed 9 exhausted
  four failed repairs and filled slot eight with an ordinary sample.
- A32 loaded model, optimizer, and extra state from A16 at optimizer step 15 and
  ended at 27. C32 restored C16 at step 16 and ended at 32. The saved restore
  evidence matches the parent checkpoint and protocol hashes.
- No accepted C group had exactly one failed initial program, so the two-attempt
  single-P0 branch was reviewed in code but was not exercised by this run.
- Remote syntax compilation passed for all eight changed Python files. Targeted
  Ruff checks used `--no-force-exclude` so the repository's scripts exclusion did
  not suppress them; the two launchers passed. Existing diagnostics elsewhere
  were compared with the pre-change sources, with no new diagnostics. No test
  files were added and no unrelated test suite was run.

## Implementation

This experiment compares ordinary GRPO (A) with any-failure critique (C). Both
start from Qwen2.5-Coder-7B-Instruct and use the same privileged Cube Stack
training recipe: rank-16/alpha-32 LoRA, learning rate `1e-5`, temperature 0.7,
top-p 0.8, top-k 20, repetition penalty 1.1, and 4096 actor output tokens.

## Collection and provenance

`capsule.repair_trigger` supports `never`, `any_failed`, and `all_failed`.
Omitting it retains the previous `all_failed` behavior. The training entrypoint
exposes the setting as `--repair-trigger`; continuation must retain its parent's
setting.

A samples eight ordinary programs and never creates or calls a Controller
client. Its training subprocess has no `CAPX_CONTROLLER_API_KEY`.

C evaluates seven ordinary programs, then triggers repair if at least one has a
known failure. Only failed programs are eligible. The first P0 maximizes partial
reward, and the second maximizes token edit distance from the first, with sample
order breaking ties. At most two P0 programs receive two trajectories each; one
failed program therefore produces two attempts. The first independently
successful revision occupies the eighth position. If no revision succeeds, an
ordinary sample fills that position. The original failures remain in the group;
the eighth ordinary sample never triggers another repair round.

The trainer checks the configured trigger against group provenance, validates
P0 selection and attempt counts, and preserves the clean-replay and guided-token
contracts. No model loss or gamma changes are included.

## Run protocol

Remote project: `/root/autodl-tmp/cap-x`, A800 80 GB. Start PyRoKi on port 8116
before training. Supply the frozen qwen3.7-plus Controller credential through
`CAPX_CONTROLLER_API_KEY` in the process environment.

```bash
.venv/bin/python -u -m scripts.capsule_rl.run_cube_stack_ac \
  --run-id cube_stack_ac_qwen25coder7b_g16_32_s05_36_20260909_r03 \
  --model /root/autodl-tmp/cap-x/.codex-downloads/models/Qwen2.5-Coder-7B-Instruct \
  --verl /root/autodl-tmp/cap-x/.codex-downloads/verl-v0.6.1 \
  --staging-checkpoint-root /dev/shm/capx_ac_checkpoints
```

The launcher runs A16, C16, A32, C32. The first stage uses seeds 5–20;
the second restores model, optimizer, scheduler/RNG extra state and uses seeds
21–36. Each arm trains 32 groups total, with complete checkpoints at 16 and 32.
The optional staging path places A16/C16/C32 checkpoints outside the limited
data disk; A32 remains on disk. Temporary checkpoints must be copied and verified
before the server is stopped. Omit the staging option when disk space is ample.

Evaluation uses non-privileged high-level Cube Stack without Controller help,
seeds 201–220 and four matched generations per scene: 80 programs for each of
four checkpoints. The comparison verifies scene, generation seed, and physical
initial-state identities. The primary comparison is at 32 groups; the 16-group
comparison measures early training behavior. Only one training replicate is run.
The paired scene-bootstrap interval describes scene uncertainty, not training
seed variability.

Each training-stage summary reports its own 16-group costs. For cumulative
32-group costs, add A16+A32 or C16+C32. Actor and Controller token counters,
training wall time, clean-replay counts, repair incidence, guided members, actor
updates, and skipped/discarded groups accompany success rates. Equal groups do
not imply equal computation costs.

Artifacts live under `artifacts/<run-id>/`, logs under `outputs/<run-id>/`.
The source manifest, training protocols, restored optimizer-step evidence,
checkpoint hashes, group records, and evaluation records provide the audit trail.
Raw results and model weights are not committed.

## Downloaded records

The local delivery directory is
`remote_results/cube_stack_ac_qwen25coder7b_g16_32_s05_36_20260909_r03/`.
`results.tar.gz` contains 772 experiment/log files and nine executed source
snapshots; its SHA-256 is
`a239c1a337638685b0e116a74f94c318ba018ea380edd576b28fcb3f104b16e9`.
It is extracted under `data/`. The authoritative comparison is
`data/artifacts/<run-id>/comparison.json`, alongside `status.json`, the frozen
`experiment.json` source manifest, per-stage protocols, and replay evidence.
`results.json` records the archive hash and the 64-group/320-evaluation audit.

Checkpoints are delivered separately under `checkpoints/`, using the format
below. `download_verification.json` records the locally verified archive hashes.
The complete checkpoint tree identities are:

| Checkpoint | SHA-256 |
| --- | --- |
| A16 | `19b409cc95aa975f631457f9e9bfda99d77573e0954bb95265fe3f54ccf9e4ff` |
| C16 | `7946e0c24a1b84f7b4d9461fbb387201890686579b827fb683f5030cd5bd0406` |
| A32 | `994056d3a3ef1ad531182a5cc98d6fde300bda6eb3d0f0d651844ab5ef92e639` |
| C32 | `c62075696e9c0fc4bc4dca48babb8d9b03aa362740155f8c6eaa34b851e42658` |

## Checkpoint archive format

The constrained server disk and SSH throughput require lossless deduplication.
Each checkpoint archive contains `LABEL.model.xdelta`, `LABEL.aux.tar.gz`, and
`LABEL.json`. The JSON records the original model hash, complete checkpoint hash,
archive hashes, and the verified hash of the decoded model. The auxiliary archive
retains optimizer state, RNG/scheduler extra state, adapter files, tokenizer,
configuration, and the original checkpoint manifest.

All four deltas share `reference_model.pt`, whose SHA-256 is
`8daf366974bb7efda3a88b8c709ad909092585958cdfe8980df197124ec57c68`.
This file comes from the existing g96 checkpoint and serves solely as the binary
compression dictionary. A and C training both start from the original Hugging
Face model specified above. `reference.json` records the original four-piece
transfer checksums. The pieces were joined into `reference_model.pt`, its full
hash was verified, and temporary pieces were removed; `reference_verified.json`
retains that verification record.

Restore a checkpoint with `xdelta3` on Linux, replacing `A16` with its label:

```bash
mkdir A16
tar -xzf A16.aux.tar.gz -C A16
xdelta3 -d -s reference_model.pt A16.model.xdelta \
  A16/final/actor/model_world_size_1_rank_0.pt
sha256sum A16/final/actor/model_world_size_1_rank_0.pt
```

Compare the restored file with `model_sha256` in `A16.json`. Each delta is decoded
and hashed on the server before delivery, and downloaded files are checked
against the recorded hashes. The complete checkpoint hash can then be recomputed
with `capx.rl.capsule.checkpoint.checkpoint_tree_sha256` on `A16/final/actor`.
