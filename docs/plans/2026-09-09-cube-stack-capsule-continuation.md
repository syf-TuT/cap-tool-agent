# Cube Stack 64/96-group continuation implementation plan

Goal: continue the completed 32-group LoRA through cumulative 64 and 96 groups,
and report each checkpoint's non-privileged high-level success rate.

Architecture: use the existing Capsule runtime and pinned VeRL checkpoint loader.
Restore model, optimizer, scheduler, and RNG before any new generation. Keep the
original base model as the reference policy. Record parent checkpoint hashes,
the cumulative training seeds/group count, and restored optimizer step.

1. Add optional parent training root to Cube Stack preparation and training.
   Reject changed parent artifacts and overlapping new training seeds. Verify
   restored optimizer step against the parent's completed result on the server.
2. Train 32 additional groups on seeds 37–44 and 65–88 to reach 64. Continue from
   that checkpoint on seeds 89–120 to reach 96. Keep the existing recipe unchanged.
3. Evaluate each stage on seeds 45–64, four samples per scene, with no Controller
   and the original frozen evaluation protocol. Reuse the already-completed base
   records only with byte-identical protocol and verified hashes; keep each LoRA
   evaluation in a separate directory. Check all ancestral seeds for overlap.
4. Validate syntax and execute the actual restored training on SeeTaCloud. Add no
   unrelated tests. Confirm initial restored optimizer step is 31 for the 64-group
   stage and equals the 64-group final step for the 96-group stage.
5. Preserve all LoRA checkpoints and reports. Inspect disk capacity before each
   stage; retain old checkpoint data when arranging additional storage.

Tech stack: existing Python Capsule runtime, VeRL v0.6.1, FSDP, Qwen2.5-Coder-7B,
rank-16 LoRA, Robosuite, SAM3, Contact-GraspNet, PyRoKi, Bash launch orchestration.

The user explicitly authorized continuation and direct server validation.
