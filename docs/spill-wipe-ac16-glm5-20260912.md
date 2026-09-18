# Spill Wipe: ordinary GRPO versus any-failure critique

Both arms completed 16 training groups and 80 held-out, non-privileged evaluation
replays on the SeeTaCloud A800 80GB server. The trainable Program model was
Qwen2.5-Coder-7B-Instruct; the frozen Controller was `glm-5`.

| Metric | A: ordinary GRPO | C: critique on any failure |
|---|---:|---:|
| Completed training groups | 16 | 16 |
| Actor optimizer updates | 9 | 12 |
| Skipped constant-reward groups | 7 | 4 |
| Successful final training members / 128 | 12 | 22 |
| Critique-triggered groups | 0 | 16 |
| Groups with a successful guided member | 0 | 9 |
| Controller requests | 0 | 703 |
| Training time | 92.3 min | 198.6 min |
| Evaluation successes / 80 | **10** | **8** |
| Evaluation success rate | **12.5%** | **10.0%** |
| Evaluation program errors | 43 | 43 |

C improved the count of successful training members but did not improve held-out
success in this run. The paired difference C minus A was -2.5 percentage points;
the scene-bootstrap 95% interval was [-15.0, 8.75] percentage points. This is one
training replicate; the interval reflects scene uncertainty, not training-seed
replication. Training success counts include C's guided members and should not
be interpreted as an unbiased comparison of base-policy success rates.

## Protocol

- Training seeds: 5–20, one eight-member group per scene, privileged high-level
  `FrankaControlSpillWipePrivilegedApi`, original task prompt.
- A uses `repair_trigger=never`; C uses `repair_trigger=any_failed` after the first
  seven base samples, following the existing Cube Stack group assembly recipe.
- C used 17 successful Controller repairs across its repair trajectories; nine
  groups ultimately accepted a successful Program-generated guided revision.
- Evaluation seeds: 201–220, four sampled programs per scene per arm, paired
  generation seeds and physical reset hashes. Each program executes once,
  without critique or Controller assistance, using `FrankaControlSpillWipeApi`.
- Binary success requires no program error, task completion, reward >= 1 and no
  truncation. The installed customized Robosuite Wipe reward is the fraction of
  wiped markers. Video recording was disabled, matching the RL evaluation recipe.
- Program sampling: temperature 0.7, top-p 0.8, top-k 20, repetition penalty 1.1,
  maximum 4096 tokens. Sampled response token IDs are preserved for actor loss
  evaluation; decoding and retokenizing can otherwise change action identity.
- VeRL v0.6.1: `d62da4950573d7a4b7ef2362337952e7ab59e78d`.

## Reproduction and verification

Use the prepared Linux checkout. Start Pyroki on port 8116, set
`CAPX_CONTROLLER_API_KEY` in the environment, then run:

```bash
CAPX_RUN_ID=spill_wipe_ac_glm5_qwen7b_g16_20260912_r03 \
  bash scripts/spill_wipe/run_ac16.sh
.venv/bin/python -m scripts.spill_wipe.summarize_ac \
  artifacts/spill_wipe_ac_glm5_qwen7b_g16_20260912_r03
```

Use a fresh run ID for a new experiment. The launcher accepts `CAPX_PROGRAM_MODEL`
and `CAPX_VERL_SOURCE` overrides. No credentials are stored in checked-in files.

Server verification checked deterministic resets for seeds 5, 6, 5 and equality
between privileged and non-privileged scenes; initial hashes include robot state
and spill marker positions. The actual Controller JSON transport was verified.
Final audit passed all group provenance, trigger, optimizer-step, checkpoint hash,
generation identity and paired reset checks, with zero discarded groups and zero
detected infrastructure errors in accepted training diagnostics.

Two startup attempts were excluded: the first selected an incompatible VeRL
directory; the second encountered an absent Pyroki service before any complete
group. The reported r03 run started afresh after service readiness verification.

The experiment ran in an existing dirty checkout. The result bundle preserves
its hashed runtime source and the reviewed commit source separately. The commit
omits unrelated Two Arm Lift and 16/32-stage bookkeeping changes. Its training
audit and evaluation verification were run against the completed server artifacts
and reproduced the same comparison. No additional test suite was added.

Remote run: `/root/autodl-tmp/cap-x/artifacts/spill_wipe_ac_glm5_qwen7b_g16_20260912_r03`.

Local bundle: `remote_results/spill_wipe/20260912T0448Z_ac_glm5-qwen7b_s05-20_r03/`.
It contains training group records, evaluation programs/results, logs, source
snapshots and both LoRA adapters. Full optimizer checkpoints remain on the server;
their verified SHA-256 values are recorded in `verified_summary.json`.
