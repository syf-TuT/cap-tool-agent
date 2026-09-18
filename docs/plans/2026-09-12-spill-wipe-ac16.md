# Spill Wipe A/C GRPO implementation plan

Goal: reuse Cube Stack's privileged high-level GRPO recipe for Spill Wipe,
train each arm for 16 groups, and evaluate both without Controller assistance.

Architecture: extend the existing task profile, trainer and paired evaluator.
A uses `repair_trigger=never`; C uses `repair_trigger=any_failed` with a frozen
`glm-5` Controller. The trainable Program remains Qwen2.5-Coder-7B-Instruct.
Both arms start from the same base model, with training seeds 5–20 and held-out
non-privileged evaluation seeds 201–220, four programs per scene.

1. Add a Spill Wipe clean-replay YAML and task profile using its existing API.
2. Hash robot state and spill marker positions after reset so clean replay and
   paired evaluation can verify the physical initial scene.
3. Extend shared prepare/train/evaluate commands and add a reusable launcher.
4. Validate configuration and seeded resets on the SeeTaCloud host, then run
   both 16-group training jobs and all 160 evaluation replays.
5. Audit group provenance, Controller triggers, checkpoint updates and paired
   scene hashes; download compact results, record metrics and commit this change.

Per user instruction, use direct server verification and add no redundant tests.
Preserve unrelated uncommitted work. Credentials are environment-only and are
excluded from source, reports and result bundles.
