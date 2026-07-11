# Remote Results Layout Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Standardize all downloaded remote experiment artifacts under `remote_results/` and move
legacy root-level result folders into that tree without data loss.

**Architecture:** Keep the implementation configuration- and documentation-only. The
repository-local SeeTaCloud transfer skill defines the canonical download and extraction paths,
while `.gitignore` keeps generated artifacts out of version control.

**Tech Stack:** Git ignore rules, Markdown workflow documentation, PowerShell file operations.

---

### Task 1: Record repository ignore rules

**Files:**
- Modify: `.gitignore`

1. Verify that `remote_results/` is already ignored.
2. Add root-scoped ignore rules for `.codex_runs/` and `.codex_sync_*.bundle`.
3. Run `git check-ignore` against representative paths and expect all three classes to match.

### Task 2: Standardize the transfer workflow

**Files:**
- Modify: `.agents/skills/run-seetacloud-capx-video-transfer/SKILL.md`
- Create: `docs/remote-results.md`

1. Replace the flat archive and arbitrary extraction examples with
   `remote_results/<task>/<short-run-id>/results.tgz` and `data/`.
2. Document partial-download renaming, collision handling, and short Windows paths.
3. Add a concise user-facing reference document with layout, naming, and retention guidance.

### Task 3: Preserve and relocate legacy results

**Files:**
- Move ignored local directories only; no tracked files change.

1. Record file counts and total byte sizes for `cl20`, `r1`, `r101418`, `r17r2`, `r4096`, and
   `rr`.
2. Resolve every source and destination path and ensure both are inside the workspace.
3. Move the directories into `remote_results/legacy/`.
4. Recalculate file counts and byte sizes and require exact matches.

### Task 4: Verify the completed layout

**Files:**
- Verify all modified files and local artifact paths.

1. Run `git diff --check` and require a zero exit code.
2. Run `git check-ignore -v` for representative generated paths.
3. Confirm no legacy result directory remains at the repository root.
4. Review `git status --short` and ensure only intended tracked documentation/configuration
   changes remain visible.
