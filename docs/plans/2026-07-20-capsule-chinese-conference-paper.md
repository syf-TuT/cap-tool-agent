# Capsule Chinese Conference Paper Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Produce an evidence-bounded Markdown draft of a Chinese conference paper for Capsule, together with an evidence matrix, experiment audit, and open-item checklist.

**Architecture:** Build the paper from three traceable layers: the five allowed external papers for domain claims, repository design and implementation files for the Capsule method, and repository experiment artifacts for empirical claims. Draft supporting audit documents first, then write the paper from the stabilized argument chain, and finish with mechanical and evidence-level checks.

**Tech Stack:** Markdown, PowerShell read-only inspection, Git, repository JSON/JSONL/YAML experiment artifacts.

---

### Task 1: Build The Experiment Audit

**Files:**
- Create: `docs/paper/capsule-experiment-audit.md`
- Read: `remote_results/**/run.json`
- Read: `remote_results/**/aggregate*.json`
- Read: `remote_results/**/*.tgz`
- Read: `env_configs/**/*.yaml`

**Step 1: Inventory experiment artifacts**

List each task, run id, Git commit, model, API, privilege mode, seed set, timeout, step budget, retry/selection rule, completion count, reward, and archive location.

**Step 2: Mark comparability**

Classify every candidate comparison as directly comparable, conditionally comparable, or not comparable. Record the exact reason when versions, APIs, timeouts, seeds, retry rules, or result schemas differ.

**Step 3: Separate outcome classes**

Distinguish successful completion, algorithm failure, trial-budget exhaustion, provider failure, experiment-infrastructure failure, and unknown legacy outcomes. Do not count missing or invalid results as task failures.

**Step 4: Record canonical claims**

Write only claims that can be reproduced from repository artifacts. Label legacy selected-latest summaries and missing first-attempt metrics explicitly.

**Step 5: Verify identifiers and numbers**

Run:

```powershell
rg -n "run_id|git_commit|task_completed|success_rate|average_reward|selection_policy|timeout" remote_results
```

Expected: every number in the audit can be traced to a concrete artifact path.

### Task 2: Build The Evidence Matrix

**Files:**
- Create: `docs/paper/capsule-evidence-matrix.md`
- Read: `F:/code/edit/writing-chinese-cs-paper-from-literature/references/evidence-map.md`
- Read: `docs/plans/*capsule*design.md`
- Read: `capx/runtime_control/*.py`
- Read: `capx/envs/trial.py`

**Step 1: Register sources**

Preserve `S01`--`S05` for the five allowed papers. Add separate repository evidence ids for design documents, implementation files, configurations, and experiment artifacts so external work cannot be confused with the user contribution.

**Step 2: Map argument claims**

For background, gap, method, experiment, result, limitation, and conclusion claims, record source id, location, evidence type, strength, applicable conditions, contradictory evidence, and permitted wording.

**Step 3: Map method claims to implementation**

Check code segmentation, effect metadata, execution, tracing, feedback, no-replay guards, recovery validation, syntax recovery, and retry-aware reporting against the current repository implementation.

**Step 4: Record unsupported claims**

Mark any desired but unverified claim as `来源待核实` or `待统一复算`; do not silently omit the gap from the matrix.

### Task 3: Draft The Open-Item Checklist

**Files:**
- Create: `docs/paper/capsule-open-items.md`

**Step 1: List publication metadata gaps**

Record missing authors, affiliations, corresponding author, target venue, page limit, template, anonymous-submission status, and citation style.

**Step 2: List experiment gaps**

Record missing unified baselines, ablations, first-attempt metrics, statistical intervals, hardware details, exact model snapshot, token cost, wall-clock cost, and real-robot validation.

**Step 3: List figure and reproducibility gaps**

Record required figure exports, table recomputation, configuration freeze, code commit tag, artifact manifest, and data-release decision.

### Task 4: Draft The Full Chinese Paper

**Files:**
- Create: `docs/paper/capsule-paper-draft-zh.md`
- Reference: `docs/figures/capsule-runtime-control-architecture.svg`
- Read: `docs/paper/capsule-evidence-matrix.md`
- Read: `docs/paper/capsule-experiment-audit.md`

**Step 1: Write title, placeholders, and keywords**

Use the approved title and explicit placeholders for authors, affiliations, venue, and template. Define Capsule, Code-as-Policy, and effect-bounded forward-only recovery consistently.

**Step 2: Write introduction and related work**

Use the argument chain “closed-loop code policy, physical side effects, recovery gap, research question, Capsule”. Organize related work by mechanism rather than paper order, with citations limited to `S01`--`S05`.

**Step 3: Write problem formulation**

Define program, atomic region, effect-bounded execution unit, physical state, side-effect ledger, historical unit, current-state observation, and admissible recovery. State assumptions and non-goals.

**Step 4: Write method**

Describe runtime boundaries, metadata-only grouping, persistent namespace, automatic forward execution, trace and source-bound feedback, no-replay enforcement, appended recovery, syntax repair, invalid-action handling, and stopping rules.

**Step 5: Write experiments**

Organize by RQ1--RQ4. Report only audited settings and results. Preserve negative results and place incomparable runs in separate rows or subsections.

**Step 6: Write discussion, limitations, and conclusion**

Separate observations from explanations, include alternative explanations, and limit conclusions to Robosuite tasks, available APIs, current model/provider, and audited experiment versions.

**Step 7: Write abstract last**

Compress only claims and numbers already present in the body. Avoid new citations and any unsupported use of “首次”, “全面”, “显著”, or “通用”.

### Task 5: Verify The Deliverables

**Files:**
- Verify: `docs/paper/capsule-paper-draft-zh.md`
- Verify: `docs/paper/capsule-evidence-matrix.md`
- Verify: `docs/paper/capsule-experiment-audit.md`
- Verify: `docs/paper/capsule-open-items.md`

**Step 1: Check placeholders and unsupported claims**

Run:

```powershell
rg -n "来源待核实|待统一复算|待补|TODO|首次|全面|显著|通用" docs/paper
```

Expected: every placeholder is intentional and listed in `capsule-open-items.md`; promotional terms either do not appear or are explicitly qualified.

**Step 2: Check source ids and local links**

Run:

```powershell
rg -n "\[S0[1-5]\]|capsule-runtime-control-architecture" docs/paper
```

Expected: external claims use only `S01`--`S05`, and the architecture figure path resolves.

**Step 3: Check paper structure**

Run:

```powershell
rg -n "^#|^##|^###" docs/paper/capsule-paper-draft-zh.md
```

Expected: title, abstract, introduction, related work, problem definition, method, experiments, results and discussion, limitations, conclusion, and references are present.

**Step 4: Review the diff**

Run:

```powershell
git diff -- docs/paper docs/plans/2026-07-20-capsule-chinese-conference-paper.md
```

Expected: only planned documentation files are changed; unrelated user code changes are absent from the diff scope.
