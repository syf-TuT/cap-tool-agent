# Capsule Strict-Subset Local Names and Safe Copy Design

## Context

Non-privileged Capsule execution uses a fail-closed Python subset. A LIBERO seed-4 run
showed that the subset currently rejects two common, data-only Python patterns:

- local names beginning with `_`, including tuple-unpacking placeholders such as `_`;
- zero-argument data copies such as `position.copy()`.

The generated source consequently remained in quarantined repair for all 25 Capsule steps.
No source edit committed and no robot action executed. The existing repair admission rule,
prompt feedback, duplicate-candidate behavior, and step budget are intentionally outside the
scope of this change.

## Goals

- Allow underscore-prefixed local variables and function parameters.
- Allow only zero-argument `value.copy()` attribute calls.
- Preserve private-attribute, sensitive-runtime, dynamic-call, callable-alias, control-flow,
  program-contract, and robot-side-effect protections.
- Keep the prompt's stated strict-subset rules consistent with the validator.
- Regress the exact source shapes observed in the seed-4 failure.

## Non-goals

- Do not change the violation fingerprint or strict sub-multiset repair rule.
- Do not add detailed candidate-violation feedback or duplicate-candidate detection.
- Do not change Capsule budgets, quarantine, lineage, rollback, or execution semantics.
- Do not allow arbitrary attribute methods or parameterized `copy()` calls.
- Do not add a configuration flag.

## Design

### Underscore-prefixed local names

The strict-subset visitor will distinguish ordinary local identifiers from capability and
attribute identifiers.

Local `ast.Name` nodes and function parameters may begin with `_`, including `_`, `_soup`,
and `_value`. Exact sensitive runtime names such as `__builtins__`, `env`, `APIS`, `sys`, and
the existing sensitive-name set remain forbidden regardless of their spelling.

The relaxation does not apply to:

- attribute names such as `obj._private` or `obj.__class__`;
- helper function names such as `def _helper(...):`;
- direct callable targets such as `_helper()`;
- forbidden or protected callable names.

This keeps private capability traversal fail-closed while allowing harmless local naming
conventions.

### Zero-argument `copy()`

The strict-subset call visitor will recognize one safe attribute-call form:

```python
value.copy()
```

The method name must be exactly `copy`, with no positional arguments and no keyword
arguments. The receiver expression is still recursively validated. Consequently,
`obj._private.copy()` and `__builtins__.copy()` remain invalid, while names, subscripts, and
other already-safe data expressions may be copied.

All other attribute calls remain invalid, including `value.tolist()`, `obj._copy()`,
`value.copy(1)`, and `value.copy(order="K")`. Runtime objects are still responsible for
supporting `copy()`; an unsupported receiver produces the existing normal execution failure.

The allowance is intentionally narrow. Classes and imports remain forbidden, public APIs are
trusted capabilities, and arbitrary method dispatch is not enabled.

### Prompt consistency

The strict Capsule prompt will state that attribute calls are forbidden except for
zero-argument `.copy()`. It will continue to require direct public API calls, safe builtins,
proven-pure helpers, and bounded loops.

### Repair and execution behavior

The existing repair fingerprint and admission policy remain unchanged. A remaining-invalid
candidate must still be a proper sub-multiset of the current violation fingerprints. Source
edits remain atomic, and all robot execution remains blocked until the complete source is
valid.

Allowing the two source forms removes their strict-subset diagnostics before repair progress
is compared. The seed-4 candidate can therefore be judged on its remaining program-contract
violations rather than on harmless local naming or data copying.

## Testing

Focused contract tests will prove that:

- `_`, `_soup`, and underscore-prefixed parameters are accepted;
- private attributes, sensitive names, underscore-prefixed helper names, and direct private
  callable targets remain rejected;
- zero-argument `.copy()` on names and subscript expressions is accepted;
- parameterized `.copy()` and every other attribute method remain rejected;
- a private or sensitive receiver cannot be used to reach `.copy()`.

Prompt tests will verify that the documented restriction matches validation behavior.

Trial-loop regressions will use the seed-4 initial source and first repair candidate to verify
that `_` and `.copy()` no longer contribute strict-subset violations, while quarantine and
the unchanged repair-progress rule still govern any remaining contract violations.

Tests and Ruff will run in the prepared WSL project copy, not the Windows checkout. A remote
DeepSeek rerun is optional follow-up validation because model output is nondeterministic and
uses an external API budget.
