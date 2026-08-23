# Capsule Gate Failure Evidence Design

## Goal

Gates 2--6 must leave immutable, machine-readable failure evidence whenever runtime
dispatch, post-execution Git checks, typed verification, publication, or transaction rollback
fails. A failed run must never occupy or imitate the success artifact path.

## Artifact contract

For a requested success artifact `gateNN_name.json`, the adapter records failure evidence at
`gateNN_name.json.failure.json`. The common envelope is:

```json
{
  "schema_version": 1,
  "gate": "seed",
  "passed": false,
  "run_id": "run-01",
  "config_sha256": "... or null when unavailable",
  "git_sha": "... or null when unavailable",
  "exception": {
    "type": "RuntimeError",
    "message": "reset failed",
    "stage": "runtime_dispatch"
  }
}
```

The writer uses exclusive atomic publication and never replaces an existing success, failure,
or log artifact. If rollback also fails, the original exception remains `exception` and the
rollback error is appended as `rollback_exception` with stage `transaction_rollback`.

## Adapter flow

`server_adapter.execute_gate` tracks narrow failure stages across config loading, request
validation, identity hashing, pre/post Git checks, dependency validation, runtime dispatch,
typed verification, and success publication. On a post-dispatch failure it first attempts the
active `GateTransaction` rollback, then writes the independent failure artifact, and finally
re-raises the original exception. Gate 6 therefore retains checkpoint rollback semantics while
also preserving why the gate failed.

## Wrapper flow

`common.run_external_gate` captures child stdout and stderr and publishes each to an immutable
companion log. For staged Gates 2--5, a child adapter failure file beside the unique staging
artifact is hard-linked to the final failure path before any staging cleanup. When a custom
runner exits, omits its artifact, or produces evidence rejected by the wrapper verifier, the
wrapper synthesizes the same failure envelope. Gate 6 publishes directly, so an adapter-created
final failure file is reused and never overwritten.

Cleanup may remove staging success bytes, staging failure bytes, and staging logs only after
equivalent final evidence has been published. It must never delete the sole failure record.

## Tests

Pure tests inject fake runtimes, Git loaders, verifiers, transactions, and subprocess results.
They cover dispatch, post-Git, verifier, rollback, immutable collision, staged promotion,
direct publication, stdout/stderr capture, and failure cleanup. No service, simulator, model,
or optimizer is started.
