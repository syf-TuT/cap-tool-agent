# Cube Lift guided-learning checkpoint diagnostic

## Findings

The C policy assigns higher teacher-forced likelihood to some repair edits,
but did not show a larger average
teacher-forced fit improvement than A on these selected programs. The guided
policy-loss derivative is attenuated to approximately one tenth of the
ordinary-token derivative at equal advantage. This motivates a controlled
guided-loss-strength experiment; it does not establish the cause of the A/C
rollout tie or a multiplier that will improve evaluation success.

The diagnostic scored 170 fixed examples at base, A16, C16, A32, and C32:
the 17 repair-bearing training groups' 136 members under their training prompts,
plus their 34 P0/revision programs under the non-privileged evaluation prompt.
No generation, simulation, training, optimizer step, or parameter-gradient
backward pass occurred. Autograd was used only on detached log-probability leaves
to evaluate the existing policy loss.

### Executable repair edits

The table reports mean negative log likelihood (NLL, nats/token; lower is better)
over changed response tokens overlapping executable Python lexical tokens.
It excludes comments, outer fences, whitespace-only tokens, and EOS. Each
program contributes equally to the mean, regardless of its edit length.

| Model | Privileged-prompt edit NLL | Non-privileged-prompt edit NLL | Programs improving over base, non-privileged |
| --- | ---: | ---: | ---: |
| base | 0.900247 | 0.972769 | reference |
| A16 | 0.859953 | 0.922779 | 13/17 |
| C16 | 0.876928 | 0.945163 | 11/17 |
| A32 | 0.803811 | 0.876599 | 9/17 |
| C32 | 0.823926 | 0.896528 | 13/17 |

C32 improves more programs than A32, but A32 has a larger average improvement.
For the 11 revisions that succeeded in the preceding non-privileged replay,
the non-privileged edit NLL is 0.945070 for base, 0.796044 for A32, and 0.834869
for C32. The same ordering persists after excluding comments, fences, and
whitespace-only tokens; lexical edits can still include variable renaming and strings.

Whole-response NLL, including inherited code, comments, fences, and EOS, changes
much less for C32: 0.274718 to 0.273959 under the training prompt, and 0.281822
to 0.282199 under the non-privileged prompt. These small differences should not
be read as changes in task success probability. Across the 17 revisions, the
average response length is 176.71 tokens, with 14.94 changed executable-code
tokens (range 1–39).

### Guided signal strength

The actual worker configuration uses one sequence per microbatch, then
accumulates eight sequence-mean losses per update. For a guided token,

```text
L = -A * p / (p + gamma)
abs(dL / dlogp) = abs(A) * p * gamma / (p + gamma)^2
gamma = 0.1
```

The ordinary local reference uses `old_log_prob = current_log_prob`, hence PPO
ratio = 1, and its coefficient is `abs(A)`. The comparison includes the real
sequence-mean and `/8` accumulation factors. It is not reconstructed historical
PPO importance sampling.

| Frozen model | Mean guided/ordinary coefficient ratio | Guided share of group PG log-probability L1 |
| --- | ---: | ---: |
| base | 9.4700% | 1.3015% |
| C16 | 9.4687% | 1.3010% |
| C32 | 9.4603% | 1.3005% |

The last column is averaged over the 17 repair-bearing groups only. It is the
share of L1 derivatives in log-probability coordinates, not the share of LoRA
parameter gradients or of the optimizer update. KL, parameter Jacobians,
gradient clipping, and Adam are outside this comparison. In particular,
the reciprocal of 9.46% is not a justified loss multiplier.

The archived guided advantage also declines as group success increases:
mean 0.55556 in C16's nine selected groups versus 0.203125 in C32's eight selected
groups. The latter stage therefore has a smaller reward-relative guided signal
even before probability shaping. All these groups had nonzero advantages.

## Interpretation and next experiment

The previous replay demonstrated useful repair programs on their original scenes.
This diagnostic shows increased likelihood of some executable edits, without a mean
advantage over A, and confirms weak local guided coefficients under the current
loss. Together these observations make guided-loss strength worth investigating
before scaling the same recipe to much larger group counts.

A follow-up can hold gamma and the on-policy collection protocol fixed and
compare a small number of explicit guided-loss multipliers against the current
multiplier 1. Select settings on separate validation scenes and verify on fresh
evaluation scenes; do not infer a required multiplier from the derivative ratio.
No such retraining or loss modification is included in this diagnostic.

These are C-selected training programs, not an independent sample. C16 has not
yet trained on the later C32 groups; separate stage subsets are saved. Token
alignment measures lexical edits rather than semantic program changes. All
probabilities are teacher-forced on archived response prefixes, which does not
measure how often a model generates a complete successful program unaided.
There are no saved per-update log probabilities or parameter gradients, so the
diagnostic cannot reconstruct each historical training update or prove causality.

## Reproduction and verification

From `/root/autodl-tmp/cap-x` with the five existing model states available:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 \
.venv/bin/python -u -m scripts.capsule_rl.diagnose_cube_lift_guided_learning \
  --training-root artifacts/cube_lift_ac_qwen25coder7b_g16_32_s05_36_20260909_r02 \
  --replay-root artifacts/cube_lift_c_repairs_nonprivileged_original_seeds_20260910_r01 \
  --root artifacts/cube_lift_guided_learning_checkpoint_diagnostic_20260910_r01

.venv/bin/python -m scripts.capsule_rl.analyze_cube_lift_guided_scores \
  --root artifacts/cube_lift_guided_learning_checkpoint_diagnostic_20260910_r01
```

Use a new root for a new scoring run. The second command only reads saved scores
and tokenizes archived code; it does not load model weights or repeat inference.
Scoring uses bfloat16 model forward passes, float32 probability normalization,
the training chat template, and an appended EOS matching the archived guided
mask length. NLL uses raw temperature-1 probabilities; loss diagnostics use the
training temperature 0.7 without generation-only top-p/top-k filtering.

All 850 score records were verified for source identity, token length, finite
values, and stored local derivatives against the loss formula. The four adapter
hashes were unchanged, and both executed scripts matched their recorded hashes.
The scripts passed remote syntax and targeted Ruff checks. No tests were added
or test suite run.

Results are under remote `artifacts/cube_lift_guided_learning_checkpoint_diagnostic_20260910_r01/`.
The main files are `summary.json`, `executable_edit_analysis.json`, the five
checkpoint score files, `examples.json`, `protocol.json`, and `verification.json`.

Local delivery uses the same run name under `remote_results/`. Its 15-file archive
SHA-256 is `ffc11fb2d1eca55f76d1fb7fead3a564bcf426e69788926bede26a7c7415e2ac`;
`archive_manifest.json` records individual file hashes.
