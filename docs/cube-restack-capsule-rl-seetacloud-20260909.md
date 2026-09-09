# Cube Restack Capsule-RL and transfer to Cube Stack

Status: running on SeeTaCloud; the first complete group is verified. Full training
and the final comparison are pending.

Remote project: `/root/autodl-tmp/cap-x`.
Run ID: `qwen25coder7b_g32_s05_36_20260909_r04`.
Driver PID at launch: `294897`.

The initial Qwen2.5-Coder-7B-Instruct is trained on privileged high-level Cube
Restack, using the existing Cube Stack Capsule recipe: 32 groups of eight,
training scene seeds 5-36, rank-16 LoRA, frozen qwen3.7-plus Controller, and the
existing reward/repair and optimizer settings. No Stack-trained adapter is loaded.

The subsequent evaluation compares initial and Restack-trained policies on
non-privileged high-level **Cube Stack**, as explicitly requested. It uses seeds
45-64, four samples per scene, identical sampling and generation seeds for both
policies, and no Controller. This measures cross-task transfer to Stack; it is not
a Restack held-out success-rate claim. The 80 samples represent 20 scene seeds.

The Restack reset now settles physics, runs a final forward pass, forcibly updates
Robosuite's cached observations, and returns the canonical two-cube/joint state
hash used by clean replay. The old cached-joint angle compensation is removed.
The server confirmed the original cache was stale by up to 2.5cm in cube position.

Validation completed on the server:

- All 32 training seeds were resolved from actual simulator resets.
- A separate 5-6-5 reset sequence verified hashes against freshly updated physical
  simulator observations and joints; the repeated seed matched exactly and the
  different seed differed. This was repeated after the cache-refresh correction.
- All five modified/new Python files passed targeted Ruff; source compilation passed.
- The user-provided Controller key returned HTTP 200 from the configured endpoint.
- The first actual group completed all eight members and four Controller repair
  trajectories. Its rewards were all zero, so `skipped_actor_update` was true,
  as required by the existing constant-reward rule. Repair outcomes were genuine
  task/program failures, without service failures or reset identity mismatch.
  This validates group execution, not an optimizer update or model improvement.
- No unit tests were added. Validation artifacts are under
  `artifacts/cube_restack_capsule_rl_qwen25coder7b_g32_s05_36_20260909_validation/`.

The older generic seed gate refuses the pre-existing model symlinks, so the reset
check was run directly against the simulator without loading a model. Model
symlinks were preserved. The direct evidence is `settled_reset_evidence.json`.

Reproduce with a fresh run ID and `CAPX_CONTROLLER_API_KEY` in the process environment:

```bash
cd /root/autodl-tmp/cap-x
.venv/bin/python -m scripts.capsule_rl.run_cube_restack \
  --run-id qwen25coder7b_g32_s05_36_NEW_RUN_ID \
  --model .codex-downloads/models/Qwen2.5-Coder-7B-Instruct \
  --verl .codex-downloads/verl-v0.6.1
```

Runtime records:

- Driver log: `.codex_restack_train_eval_20260909_r04.log`.
- Phase logs/status: `outputs/qwen25coder7b_g32_s05_36_20260909_r04/`.
- Training: `artifacts/cube_restack_capsule_rl_qwen25coder7b_g32_s05_36_20260909_r04/`.
- Evaluation: `artifacts/cube_stack_transfer_from_restack_qwen25coder7b_g32_s05_36_20260909_r04/`.
- Final comparison: the evaluation directory's `summary.json` after completion.

Storage change explicitly authorized by the user: removed only
`artifacts/cube_stack_capsule_rl_s05_20_20260908_r03/training/checkpoints/cube_stack_capsule_rl_s05_20_20260908_r03-cf5cfedc2e/final/actor/model_world_size_1_rank_0.pt`
(15,393,363,634 bytes). That old run's LoRA, optimizer and logs remain; its full
checkpoint can no longer be restored or pass its original full-tree hash check.
The 32/64/96-group Stack checkpoints remain. Data disk free space became 21GB.
