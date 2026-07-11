# Remote Result Naming Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a canonical, validated naming convention and metadata manifest requirement for
downloaded remote experiment runs.

**Architecture:** Extend the tracked remote-results guide with the naming grammar and `run.json`
schema. Update the workspace-local SeeTaCloud transfer skill to construct names from the approved
fields and write metadata after a verified transfer.

**Tech Stack:** Markdown workflow documentation, PowerShell transfer examples, JSON metadata.

---

### Task 1: Document the canonical grammar

**Files:**
- Modify: `docs/remote-results.md`

1. Replace the generic short run id with the approved UTC/method/model/seeds grammar.
2. Document fixed aliases, seed formats, rerun numbering, and the 64-character limit.
3. Explain which mutable or verbose fields must remain out of directory names.

### Task 2: Define run metadata

**Files:**
- Modify: `docs/remote-results.md`

1. Add `run.json` to the canonical layout.
2. Provide a representative manifest with reproducibility fields.
3. Keep secrets and result outcomes out of the manifest template.

### Task 3: Update the transfer workflow

**Files:**
- Modify: `.agents/skills/run-seetacloud-capx-video-transfer/SKILL.md`

1. Construct the local run id from UTC time, stable aliases, seed notation, and optional rerun.
2. Validate the allowed format and length before creating the destination.
3. Write `run.json` after the archive is verified and extracted.

### Task 4: Verify and commit

1. Validate representative accepted and rejected names against the documented grammar.
2. Run `git diff --check`.
3. Review tracked and ignored changes separately.
4. Commit the tracked documentation changes as one scoped commit.
