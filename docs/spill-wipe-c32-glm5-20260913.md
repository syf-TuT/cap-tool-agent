# Spill Wipe C16 to C32 continuation

C32 completed all 16 additional training groups and all 80 paired evaluation
replays. It restored C16's model and optimizer at step 12 and finished at step 28.
The frozen Controller remained `glm-5`; the Program remained
Qwen2.5-Coder-7B-Instruct with the original any-failure critique recipe.

| Evaluation | Successes | Success rate | Program errors | Task failures |
|---|---:|---:|---:|---:|
| C16 | 8/80 | 10.0% | 43 | 29 |
| C32 | 12/80 | 15.0% | 39 | 29 |

C32 improved by 5 percentage points (four additional successes). The paired
scene-bootstrap 95% interval for C32 minus C16 is [-5, 15] percentage points.
This is an adaptive continuation chosen after inspecting C16 on the same scenes,
not an independent final holdout or evidence of a stable improvement.
A32 was not trained. A16's earlier result was 10/80 (12.5%).

The additional stage used seeds 21–36, bringing training ancestry to seeds 5–36
and 32 groups total. All 16 new groups produced actor updates, with no discarded
or constant-reward groups. Their final 128 members included 33 successes.
Critique triggered for every group; 11 groups accepted a successful guided member.
There were 715 Controller calls and 16 successful Controller repair trajectories.
Additional training time was 14,665.6 seconds (244.4 minutes).

Evaluation retained seeds 201–220, four programs per scene, the same generation
seed schedule, non-privileged Spill Wipe APIs, and no Controller assistance.
Group provenance, cumulative ancestry, restored optimizer state, full checkpoint
hash, generation identities and paired physical scene hashes were verified.
Accepted training diagnostics contained no detected service infrastructure errors.
The existing recipe and server execution were used; no redundant tests were added.

Run from the prepared Linux checkout, with Pyroki ready on port 8116 and the
Controller key supplied through `CAPX_CONTROLLER_API_KEY`:

```bash
.venv/bin/python -m scripts.spill_wipe.run_c32 \
  --parent artifacts/spill_wipe_ac_glm5_qwen7b_g16_20260912_r03/C16 \
  --baseline-evaluation artifacts/spill_wipe_ac_glm5_qwen7b_g16_20260912_r03/eval_C16 \
  --run-id spill_wipe_c32_glm5_qwen7b_20260912
```

Use a new run ID for a new continuation. Existing completed phases are verified
and reused when resuming the same inputs. The scheduler remains disabled.

Remote results: `/root/autodl-tmp/cap-x/artifacts/spill_wipe_c32_glm5_qwen7b_20260912`.
The full checkpoint and LoRA adapter remain in
`C32/training/checkpoints/C32-fb44388faf/final/actor` under that directory.
Checkpoint SHA-256: `cfd6953eb877caa76332984633592cedd7a5fd8d055cabbf1272683d4d29b5a1`.
Compact local results and logs:
`remote_results/spill_wipe/20260913T0000Z_c32_glm5-qwen7b_s21-36/`.
The local bundle excludes optimizer/model checkpoint files.
