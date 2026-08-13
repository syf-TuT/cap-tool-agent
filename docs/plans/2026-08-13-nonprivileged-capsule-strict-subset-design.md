# Non-Privileged Capsule Strict Python Subset Design

## Context

The non-privileged Capsule path currently executes generated Python in the simulator
process. Removing `env` and `APIS` from globals and denying known reflection paths is
not a complete boundary: Python control flow, frame inspection, callable aliases, and
future reflection variants can recover capabilities that a denylist did not anticipate.

The user selected a strict allowlist rather than a separate process sandbox. This design
therefore treats non-privileged Capsule source as a small executable language embedded in
Python syntax, not as arbitrary Python with selected dangerous constructs removed.

## Security Boundary

Every non-privileged Capsule program is validated against the strict subset before any
region or group can execute. This is independent of `capsule_validate_program_contract`;
that setting continues to control the richer Capsule-ready repair protocol for privileged
legacy configurations, but it cannot disable the non-privileged boundary.

Execution globals contain only:

- `__name__`, `INPUTS`, and `RESULT`;
- opaque, traced public API functions;
- an immutable allowlist of safe builtins.

They never contain the low-level environment, API objects, Python's default builtins, or
import machinery.

## Allowed Source

The strict subset allows ordinary literals, containers, arithmetic, comparisons,
conditionals, assignments, indexing, bounded pure control flow, and top-level function
definitions used as pure helpers.

Calls must use a direct `ast.Name` and resolve to one of:

- a public API function exported into the Capsule globals;
- an explicitly allowed safe builtin;
- a statically known helper whose body also satisfies the strict subset and is pure.

Robot side-effect APIs retain the existing Capsule-ready rules: top-level only, no
effectful loop or `try`, and at most one effect per semantic execution group.

## Rejected Source

The validator rejects before execution:

- imports, classes, lambdas, async/yield, global/nonlocal declarations, and dynamic code;
- callable aliases, callable parameters, and calls through attributes or subscripts;
- access to `env`, `APIS`, `__builtins__`, dunder names, frames, modules, or introspection;
- calls to unknown names or helpers that cannot be proven pure;
- function decorators or callable default/annotation expressions;
- existing Capsule-ready violations involving robot effects.

Attribute reads that do not start with `_` may be used as data, but attribute calls are
rejected. This keeps pose/result field access possible without exposing method-based
capability recovery. Programs that need a method operation must use a public API or safe
direct builtin instead.

## Runtime Flow

After every initial segmentation, patch, or append, the runtime computes a combined
analysis containing strict-subset violations, Capsule-ready violations, and executable
effect IDs. The Action LLM receives the violations and can patch source. `run_group`,
`run_region`, and `resume_from_region` are blocked while the selected unit is unsafe.

Auto-forward validates the complete program before its first execution and after every
recovery rewrite. If strict violations remain, it must not run the source; it enters its
existing recovery path or returns a deterministic invalid result.

## Compatibility

Traditional non-Capsule execution is unchanged. Privileged Capsule configurations with
the contract disabled retain their legacy namespace. Non-privileged Capsule behavior is
intentionally stricter because `privileged: false` is now an executable boundary rather
than a prompt convention.

## Verification

Tests must demonstrate that conditional rebinding, frame inspection, imports, direct and
aliased dynamic calls, callable attributes, lambdas, and classes never call the fake
robot API. Positive tests must cover direct public API calls, pure helpers, data attribute
reads, safe builtins, patch-to-compliance, and default privileged compatibility.

