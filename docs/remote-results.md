# Remote experiment results

Store every experiment artifact downloaded from a remote host under `remote_results/`. Do not
download or extract experiment archives directly into the repository root.

Use this layout:

```text
remote_results/
  <task>/
    <short-run-id>/
      results.tgz
      data/
```

For example:

```text
remote_results/cube_stack/20260711_capsule_s1-5/results.tgz
remote_results/cube_stack/20260711_capsule_s1-5/data/
```

Keep `<short-run-id>` compact because archives can contain deeply nested paths. A date, method,
and seed range are usually enough. Put older directories that do not follow the convention under
`remote_results/legacy/` without renaming them.

Download archives as `results.tgz.partial`. After the transfer succeeds, rename the file to
`results.tgz`, validate it with `tar -tzf`, and extract it into `data/`. Do not overwrite an
existing run directory; choose a new short run id for reruns.

`remote_results/` is intentionally ignored by Git. Keep reproducible experiment definitions in
`env_configs/` and report the task, seed range, commit, completion count, average reward, archive
path, and summary path when handing off a transferred run.
