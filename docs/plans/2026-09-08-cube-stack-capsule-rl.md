# Cube Stack Capsule-RL implementation and experiment plan

**Goal:** Train the existing privileged high-level Capsule policy on 16 Cube Stack groups,
verify naturally occurring all-negative base groups and successful guided repairs, and compare
the final adapter with its initial model in non-privileged high-level Cube Stack.

**Architecture:** Reuse the Cube Lift Program protocol, frozen Controller, clean replay,
CapsuleGroupAssembler, CapsuleCritiqueRayTrainer, and A800 LoRA worker configuration. Prepare
Cube Stack prompts and initial-state hashes from real resets. Preserve every training group,
repair attempt, checkpoint, and paired evaluation outcome.

**Tech stack:** Robosuite, PyRoKi, VeRL v0.6.1, Qwen2.5-Coder-7B-Instruct, LoRA, SAM3,
Contact-GraspNet. Runtime: SeeTaCloud `/root/autodl-tmp/cap-x`.

1. Preserve the previous remote working state and synchronize `feature/capsule_rl` at `6707040`.
2. Add a Cube Stack preparation/training entrypoint and a paired high-level evaluation entrypoint.
3. Run actual reset preparation for training seeds 5–20 and real sampling/training of 16 groups.
   Retain the existing 7-base-plus-1-guided trigger and all failures; do not engineer failures.
4. Verify all saved group counts, repair clean replays, optimizer updates, and final checkpoint.
5. Evaluate both policies on held-out seeds 25–44 with the same non-privileged environment,
   high-level primitives, prompts, and matched generation seeds. No repair during evaluation.
6. Document measured results and limitations, verify the scoped diff, and commit after the
   remote validation. The user requests actual server validation without redundant new tests.
