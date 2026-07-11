# Remote Results Layout Design

## Goal

Keep artifacts downloaded from remote experiment hosts out of the repository root and make
every transferred run easy to locate and inspect.

## Layout

`remote_results/` is the only local destination for transferred experiment artifacts. Each run
uses a short task and run identifier to limit Windows path lengths:

```text
remote_results/
  <task>/
    <short-run-id>/
      results.tgz
      data/
```

The transferred archive is stored as `results.tgz`, regardless of its remote filename. Extracted
content goes under `data/`. Summary and status files remain inside `data/` in the paths supplied by
the remote archive; transfer verification reports their exact paths.

Existing result directories in the repository root are preserved under
`remote_results/legacy/<old-directory-name>/`.

## Naming and safety

- Use lowercase task names with underscores, for example `cube_stack`.
- Keep the local run id short, for example `20260711_capsule_s1-5`.
- Download to `results.tgz.partial`, verify the copy, and rename it to `results.tgz` before
  extraction.
- Refuse to overwrite a completed run directory unless the operator explicitly chooses a new run
  id or removes the old local copy.
- Never store credentials in scripts, manifests, archives, or documentation.

## Repository behavior

The entire `remote_results/` tree remains ignored by Git. Codex transfer staging files in
`.codex_runs/` and root-level `.codex_sync_*.bundle` files are also ignored so they do not pollute
repository status. The human-readable convention lives in tracked documentation and the
repository-local transfer skill.

## Verification

After a transfer, verify that the archive opens, count expected videos, locate the summary JSON,
and report the final archive and extraction paths. For this layout change, verify ignore behavior
with `git check-ignore` and confirm the old root-level result directories were moved without a
change in file count or total byte size.
