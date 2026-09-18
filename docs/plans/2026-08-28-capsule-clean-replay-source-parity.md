# Capsule Clean-Replay Source Parity Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make clean replay execute the same canonical Python source as ordinary evaluation and
require every scheduled Capsule seed to produce a complete group before checkpoint publication.

**Architecture:** A new pure source-normalization helper owns the existing Markdown-fence rule.
Both ordinary evaluation and `CleanReplayEvaluator` call it, while replay results retain raw source
identity and record the executed-source hash in diagnostics. The trainer retries only typed group
discards for the same task, and the server persists attempt-level audit evidence before refusing an
exhausted or incomplete run.

**Tech Stack:** Python 3.10+, dataclasses, pytest, YAML, VeRL-facing Capsule trainer adapters.

---

### Task 1: Share the existing program-source normalization rule

**Files:**
- Create: `capx/utils/program_source.py`
- Modify: `capx/utils/launch_utils.py:208`
- Create: `tests/test_program_source.py`

**Step 1: Write failing pure-function tests**

Add tests proving that a ` ```python\n...``` ` response is stripped exactly like the legacy
implementation, an unfenced response is only trimmed, and unsupported fence spellings are not
newly recognized.

**Step 2: Run the tests and verify RED**

Run:

```bash
pytest tests/test_program_source.py -q
```

Expected: collection fails because `capx.utils.program_source` does not exist.

**Step 3: Add the minimal shared helper**

Implement:

```python
def normalize_program_source(content: str) -> str:
    fence_start = "```python\n"
    if fence_start in content:
        content = content[content.find(fence_start) + len(fence_start):]
    if "```" in content:
        content = content[:content.rfind("```")]
    return content.strip()
```

Keep `_extract_code(content)` as a compatibility wrapper returning
`[normalize_program_source(content)]`.

**Step 4: Run the focused tests and verify GREEN**

Run:

```bash
pytest tests/test_program_source.py -q
```

Expected: all source-normalization tests pass.

### Task 2: Execute canonical source while preserving raw replay identity

**Files:**
- Modify: `capx/rl/capsule/evaluator.py:285-564`
- Modify: `tests/test_capsule_evaluator.py`

**Step 1: Write failing clean-replay tests**

Add tests asserting that:

- a fenced raw candidate reaches `_FakeBackend.execute()` without the outer fence;
- `ProgramReplayResultV1.source` and `source_sha256` still identify the raw candidate;
- diagnostics contain raw and executed hashes plus `source_normalized=True`;
- unfenced source is unchanged and reports `source_normalized=False`.

**Step 2: Run the focused tests and verify RED**

Run:

```bash
pytest tests/test_capsule_evaluator.py -q
```

Expected: the backend receives fenced source and the new diagnostics are absent.

**Step 3: Implement the execution boundary**

Import `normalize_program_source`. In `CleanReplayEvaluator.evaluate_program()`, derive one
`executed_source`, pass it to the backend on every retry, and continue passing raw `source` into
typed result construction. Merge these fields into every result's diagnostics:

```python
{
    "raw_source_sha256": source_sha256(source),
    "executed_source_sha256": source_sha256(executed_source),
    "source_normalized": executed_source != source,
}
```

Do not mutate `ProgramCandidate`, response token IDs, or the raw result source.

**Step 4: Run the focused tests and verify GREEN**

Run:

```bash
pytest tests/test_capsule_evaluator.py tests/test_program_source.py -q
```

Expected: both files pass.

### Task 3: Add the strict three-attempt group contract

**Files:**
- Modify: `capx/rl/capsule/compat.py:383-466`
- Modify: `capx/rl/capsule/trainer.py:85-113,366-424,922-950`
- Modify: `tests/test_capsule_config.py`
- Modify: `tests/test_capsule_trainer.py`
- Modify: `env_configs/cube_stack/capsule_rl/franka_robosuite_cube_stack_capsule_critique_grpo.yaml`

**Step 1: Write failing configuration and trainer tests**

Add tests proving:

- `capsule.max_group_attempts` must equal `3` in a formal config;
- two `GroupDiscarded` attempts followed by success produce exactly one result;
- each discard record carries the zero-based group-attempt index and complete replay history;
- three consecutive discards raise a typed `GroupAttemptBudgetExhausted`;
- unexpected exceptions are never retried.

**Step 2: Run the focused tests and verify RED**

Run:

```bash
pytest tests/test_capsule_config.py tests/test_capsule_trainer.py -q
```

Expected: tests fail because the config field, retry loop, attempt index, and exhaustion type do not
exist.

**Step 3: Implement bounded same-seed retries**

Add `capsule.max_group_attempts: 3` to the canonical config and strict compatibility validator.
Add a `GroupAttemptBudgetExhausted` exception and an attempt index to `DiscardedGroupRecord`.
Accept a validated positive `max_group_attempts` in the trainer, defaulting to three for direct unit
construction. Change `fit()` to retry only `GroupDiscarded` for the same task and append no result
until one attempt succeeds. Raise the typed exhaustion error after the third recorded discard.

**Step 4: Run the focused tests and verify GREEN**

Run:

```bash
pytest tests/test_capsule_config.py tests/test_capsule_trainer.py -q
```

Expected: all configuration and trainer tests pass.

### Task 4: Refuse incomplete checkpoint publication and preserve audits

**Files:**
- Modify: `capx/rl/capsule/server_factory.py:1690-1762`
- Modify: `tests/test_capsule_server_factory.py:845-940`

**Step 1: Write failing server-runtime tests**

Add tests proving that the runtime:

- passes `capsule.max_group_attempts` to the trainer;
- catches terminal group-attempt exhaustion, writes every discard attempt, closes resources, and
  never calls checkpoint save;
- refuses any defensive `len(results) != len(scheduled_tasks)` mismatch before checkpoint save;
- reports completed groups separately from discarded group attempts after a recovered retry.

**Step 2: Run the focused tests and verify RED**

Run:

```bash
pytest tests/test_capsule_server_factory.py -q
```

Expected: exhaustion audits and the exact scheduled-group guard are missing.

**Step 3: Implement the server-side publication guard**

Pass the configured attempt limit into the trainer. If `fit()` raises
`GroupAttemptBudgetExhausted`, write the trainer's immutable discard records and raise
`ServerFactoryError` before checkpoint publication. After successful `fit()`, require exact result
count equality. Keep the existing audit file for recovered attempts and add unambiguous result
counters for completed groups and discarded attempts.

**Step 4: Run the focused tests and verify GREEN**

Run:

```bash
pytest tests/test_capsule_server_factory.py -q
```

Expected: all server factory tests pass and mocked checkpoint saves remain uncalled on both failure
paths.

### Task 5: Run local regression and static checks

**Files:**
- Verify all modified Python and YAML files.

**Step 1: Run the Capsule regression suite**

Run in the prepared Linux runtime, not the Windows checkout:

```bash
pytest tests/test_program_source.py tests/test_capsule_evaluator.py \
  tests/test_capsule_config.py tests/test_capsule_trainer.py \
  tests/test_capsule_server_factory.py -q
```

Expected: zero failures.

**Step 2: Run lint on touched Python files**

```bash
ruff check capx/utils/program_source.py capx/utils/launch_utils.py \
  capx/rl/capsule/evaluator.py capx/rl/capsule/compat.py \
  capx/rl/capsule/trainer.py capx/rl/capsule/server_factory.py \
  tests/test_program_source.py tests/test_capsule_evaluator.py \
  tests/test_capsule_config.py tests/test_capsule_trainer.py \
  tests/test_capsule_server_factory.py
```

Expected: zero errors.

### Task 6: Sync and verify on SeeTaCloud

**Files:**
- Sync only the reviewed source, tests, and config to `/root/autodl-tmp/cap-x`.
- Store generated evidence under ignored remote output/artifact directories.

**Step 1: Run remote focused tests**

Use the prepared remote `.venv` and run the five focused test files. Expected: zero failures.

**Step 2: Run a fenced clean-replay smoke test**

Use a deterministic cube-lift seed and the existing replay environment. Evaluate the same valid
program once unfenced and once inside the supported Python fence. Assert identical outcome,
reward, task completion, and executed-source hash, while raw-source hashes differ.

**Step 3: Run a small collection gate**

Run three to five seed-local groups before another full training run. Require:

- no fence-only first-line syntax errors;
- completed group count equals scheduled group count;
- any transient discard records have ordered attempt indexes;
- no checkpoint is produced by a deliberately forced three-discard failure test.

**Step 4: Record verification evidence**

Report exact commands, test counts, smoke outcomes, group counts, discard-attempt counts, and output
paths. Do not claim the previous LoRA checkpoint has been repaired; a new full training run must
start from the base model after this gate passes.
