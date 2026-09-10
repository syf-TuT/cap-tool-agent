# Cube Lift guided coefficient symmetry

Goal: retain equal group reward-baseline weights and remove guided probability
shaping attenuation for Cube Lift only. Use `-A * log(p)` for guided tokens;
ordinary PPO, reference KL, group assembly, and token averaging stay unchanged.
This matches local log-probability coefficients, not parameter-gradient norms.
The shared loss defaults to the existing probability shaping for other tasks.

Implementation and verification:
1. Add an explicit guided objective and propagate it to the VeRL actor.
2. Select log probability only in Cube Lift preparation.
3. Verify actual remote autograd coefficients and configuration propagation;
   add no redundant test files.
4. Train C16 from the base Qwen2.5-Coder-7B-Instruct model, seeds 5–20,
   any-failed repair trigger, existing qwen3.7-plus controller.
5. Evaluate non-privileged seeds 201–220 with four samples each; audit all 80
   outcomes, save a concise result record, and commit only this change.

Baseline reweighting was not selected: its token-dependent weights would change
the group advantage definition. A constant guided multiplier would not remove
the probability-dependent attenuation. Log probability removes it directly.
