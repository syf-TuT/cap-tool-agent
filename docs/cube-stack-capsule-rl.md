# Privileged Cube Stack Capsule-RL and non-privileged evaluation

`scripts.capsule_rl.train_cube_stack` applies the existing Cube Lift training recipe to
`FrankaPickPlaceCodeEnv` with `FrankaControlPrivilegedApi`. Preparation reads the complete
prompt from the real environment, resets every seed, and freezes the prompt and state hashes.
It uses Qwen2.5-Coder-7B-Instruct, rank-16/alpha-32 LoRA, learning rate `1e-5`, and the pinned
VeRL v0.6.1 worker profile. Program decoding is copied from the Cube Lift protocol:
temperature 0.7, top-p 0.8, top-k 20, repetition penalty 1.1, and 4096 maximum output tokens.
Cube Stack uses a 24,576-token revision input budget because its complete repair trajectories
can exceed the inherited 8,192-token limit. The factory propagates this to the worker budgets:
28,672 tokens including generation, within this model's 32,768-token context. Full committed
repair history is retained; an oversized prompt is still rejected rather than truncated.

Cube Stack also enables `capsule.allow_fenced_revisions`: a single complete outer
`python` code fence is accepted because this actor consistently uses that response format.
Only executable-content validation removes the wrapper, matching the existing replay boundary.
The raw response, fence tokens, EOS, token budget, and guided training mask remain unchanged.
Prose, multiple fences, incomplete output, and invalid Python remain rejected. Other task
profiles keep the existing strict default.

Each training group starts with seven ordinary actor samples. When all seven fail clean replay,
the unchanged Capsule assembler explores two P0 programs with two repair trajectories each.
A successful actor revision is independently replayed and inserted as the eighth member.
Otherwise the eighth member is another ordinary sample. Constant-reward groups skip the actor
update. The saved group JSON contains the original programs, all repair attempts, clean-replay
results, rewards, advantages, and guided-token masks.

Run the following on the prepared SeeTaCloud host. Start the existing PyRoKi service and wait
for `http://127.0.0.1:8116/docs` first. Set `CAPX_CONTROLLER_API_KEY` in the process environment
for the configured frozen qwen3.7-plus Controller. Credentials are not written into run files.

```bash
cd /root/autodl-tmp/cap-x
export MUJOCO_GL=egl JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1

.venv/bin/python -m scripts.capsule_rl.train_cube_stack prepare \
  --root artifacts/cube_stack_capsule_rl_s05_20_RUN_ID \
  --model .codex-downloads/models/Qwen2.5-Coder-7B-Instruct \
  --verl .codex-downloads/verl-v0.6.1
.venv/bin/python -m scripts.capsule_rl.train_cube_stack train \
  --root artifacts/cube_stack_capsule_rl_s05_20_RUN_ID
```

The default training seeds are 5–20: 16 groups, one epoch. Choose a new run directory for each
training attempt. `training_result.json` records the actual optimizer steps and final checkpoint;
`summary.json` reports how many groups triggered repair and how many received a successful
guided member. This entrypoint invokes the existing trainer directly; it does not claim to have
run the separate seven-gate service qualification suite.

For an independent frozen actor sampling audit, use the existing `probe_program_sampling`
entrypoint with the prepared `runtime.yaml`, the same 16 seeds, and `--samples 8`.

`scripts.capsule_rl.evaluate_cube_stack` compares the initial model and final adapter using
the original single-program non-privileged Cube Stack environment and `FrankaControlApi`.
SAM3 and Contact-GraspNet provide perception; neither the Controller nor privileged object
poses are provided to the policy. The default evaluation uses held-out seeds 25–44, four
matched generation seeds per scene, for 80 programs per policy. Training/evaluation seed
overlap is rejected.

If a training attempt is superseded while the initial-model evaluation is already complete,
pass `--training-root NEW_TRAINING_ROOT` to `generate --policy trained_lora`. The evaluator
requires the same base model and decoder and disjoint training/evaluation seeds. It binds the
actual training protocol, result, and adapter hashes into the trained policy identity without
changing the frozen evaluation protocol or reusing an incompatible trained generation.
The final summary includes each policy's actual identity.

```bash
.venv/bin/python -m scripts.capsule_rl.evaluate_cube_stack prepare \
  --root artifacts/cube_stack_comparison_RUN_ID \
  --training-root artifacts/cube_stack_capsule_rl_s05_20_RUN_ID

for policy in base trained_lora; do
  .venv/bin/python -m scripts.capsule_rl.evaluate_cube_stack generate \
    --root artifacts/cube_stack_comparison_RUN_ID --policy "$policy"
  # Start SAM3 (8114), Contact-GraspNet (8115), and PyRoKi (8116) before replay.
  .venv/bin/python -m scripts.capsule_rl.evaluate_cube_stack evaluate \
    --root artifacts/cube_stack_comparison_RUN_ID --policy "$policy"
done
.venv/bin/python -m scripts.capsule_rl.evaluate_cube_stack summarize \
  --root artifacts/cube_stack_comparison_RUN_ID
```

Success requires no program error, a completed task, terminal reward at least 1, and no simulator
budget exhaustion. Replay records preserve console output, source hashes, and an evaluator-only
physical reset fingerprint. The final comparison verifies matching initial states and generation
seeds. Service communication failures stop the phase and remain in `infrastructure_failures/`;
resume after service recovery. Existing generation/replay records are checked against their
protocol and input hashes before they are reused.

The 80 trials share 20 scene seeds, so repeated programs on one scene are not 80 independent
environment samples. Report scene counts and the observed difference without treating a small
run as evidence of general improvement. Real experiment results are recorded separately after
completion. No additional unit tests are introduced for this experiment workflow.
