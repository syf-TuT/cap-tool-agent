# Cube Restack Capsule-RL Implementation Plan

Goal: Train the initial Qwen2.5-Coder-7B-Instruct on privileged high-level Cube
Restack, then compare initial and trained policies on non-privileged high-level
Cube Stack. This is a cross-task transfer evaluation explicitly requested by the user.

Architecture: Add a Restack task profile and clean-replay YAML. Parameterize the
existing Stack preparation function and expose a thin Restack entrypoint, sharing
the trainer, group assembly, repair, loss, checkpoint, and Stack evaluation code.
Keep Stack as the default and reject continuation across different task identities.

Tech stack: Robosuite, Pyroki, pinned VeRL, Qwen2.5-Coder-7B, rank-16 LoRA.

1. Add profile, configuration, and training entrypoint.
2. Sync source to SeeTaCloud; compile and validate with actual seeded resets.
3. Train 32 groups of eight on seeds 5-36 from the initial model, following the
   existing Stack first-stage recipe. Preserve full checkpoints and prior runs.
4. Run paired Stack evaluation on seeds 45-64 with four samples per seed, identical
   sampling for base and trained policies, no Controller assistance.
5. Verify result identity, physical reset pairing, completed counts and checkpoint
   update evidence; report successes out of 80 and percentage-point difference.

Per user instruction, validation is performed directly on the server; no unrelated
or redundant unit tests are added. Credentials are never saved to these artifacts.
