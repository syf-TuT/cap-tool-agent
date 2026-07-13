# Handover get_observation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expose non-privileged Handover observations through the same public API contract used by other CaP-X tasks.

**Architecture:** Add one thin public API method and register it in `functions()`. Rely on the existing `ApiBase` default to infer the Capsule recovery capability, avoiding task-specific runtime-control logic.

**Tech Stack:** Python 3.12, pytest, CaP-X runtime-control API registry.

---

### Task 1: Public Handover observation API

**Files:**
- Modify: `capx/integrations/franka/handover.py`
- Test: `tests/test_runtime_control_globals.py`

**Step 1: Write the failing test**

Assert that `FrankaHandoverApi.functions()["get_observation"]()` returns the underlying environment
observation and that `recovery_observation_functions()` returns `{"get_observation"}`.

**Step 2: Run test to verify it fails**

Run the focused pytest test in `/home/capx/code/cap-x` under WSL. Expect failure because the public
function is absent.

**Step 3: Write minimal implementation**

Register `get_observation` in `functions()` and implement it as a direct call to
`self._env.get_observation()`.

**Step 4: Run tests to verify they pass**

Run the focused test, then all `tests/test_runtime_control_*.py` tests.

**Step 5: Commit**

Commit the implementation and regression test together after verification.
