# Cube Restack A/C 16-group implementation plan

**Goal:** Run ordinary GRPO (A) and any-failure critique (C), 16 groups each,
using the existing Cube Stack training recipe and a frozen glm-5 Controller.

**Architecture:** Reuse the existing privileged Restack clean replay task and
A/C collector. Add explicit group limit, training-only mode, and Controller model
selection to the existing launcher. Keep actor initialization, seeds 5–20,
eight-member groups, reward semantics, and optimizer settings paired.

**Tech stack:** Python, VeRL, Robosuite, PyTorch LoRA, OpenAI-compatible Controller.

1. Extend the existing training preparation and A/C launcher; preserve defaults.
2. Audit final positive members, all-zero groups, and successful Controller repairs.
3. Compile and lint changed files on SeeTaCloud, probe glm-5, and execute both
   actual training arms. Do not add test files or run unrelated suites.
4. Inspect per-group rewards, repair outcomes, and actual actor updates. Report
   zero-positive runs honestly; do not alter rewards or inject oracle samples.
5. Save a concise result record and commit only the scoped code and documentation
   after remote validation. Keep credentials and large checkpoints out of Git.

Restack uses training-only mode because the existing non-privileged evaluator
targets Stack. One new checkpoint may use explicit `/dev/shm` staging when disk
space cannot hold both; this storage is temporary and must be identified in results.
