# Remote experiment results

Store every experiment artifact downloaded from a remote host under `remote_results/`. Do not
download or extract experiment archives directly into the repository root.

Use this layout:

```text
remote_results/
  <task>/
    <short-run-id>/
      results.tgz
      run.json
      data/
```

For example:

```text
remote_results/cube_stack/20260711T0614Z_capsule_dsv4flash_s01-05/results.tgz
remote_results/cube_stack/20260711T0614Z_capsule_dsv4flash_s01-05/run.json
remote_results/cube_stack/20260711T0614Z_capsule_dsv4flash_s01-05/data/
```

Use this grammar for `<short-run-id>`:

```text
<YYYYMMDDTHHMMZ>_<method>_<model>_<seeds>[_rNN]
```

- Use a UTC timestamp so names sort chronologically and do not depend on the remote host timezone.
- Use stable lowercase aliases such as `capsule`, `original-mt`, and `dsv4flash`.
- Write one seed as `s17`, a range as `s01-05`, and a mixed set as `s10+14+16-18`.
- Add `_r02`, `_r03`, and so on only when rerunning the same configuration.
- Use only lowercase ASCII letters, digits, hyphens, underscores, and the seed-list plus sign.
- Keep the complete run id at or below 64 characters because archives can contain deeply nested
  paths.

Do not include mutable outcomes such as reward, success rate, or completion state. Keep provider,
full model name, Git commit, token limits, timeouts, and behavior flags in `run.json` instead of the
directory name. Put older directories that do not follow the convention under
`remote_results/legacy/` without renaming them.

A representative `run.json` is:

```json
{
  "schema_version": 1,
  "run_id": "20260711T0614Z_capsule_dsv4flash_s01-05",
  "task": "cube_stack",
  "method": "capsule",
  "model": "deepseek-v4-flash",
  "model_alias": "dsv4flash",
  "provider": "packy",
  "seeds": [1, 2, 3, 4, 5],
  "git_commit": "214b5bc",
  "started_at_utc": "2026-07-11T06:14:00Z",
  "rerun": 1,
  "streaming": true,
  "reasoning": false
}
```

Never put credentials or API keys in `run.json`.

Download archives as `results.tgz.partial`. After the transfer succeeds, rename the file to
`results.tgz`, validate it with `tar -tzf`, and extract it into `data/`. Do not overwrite an
existing run directory; choose a new short run id for reruns.

`remote_results/` is intentionally ignored by Git. Keep reproducible experiment definitions in
`env_configs/` and report the task, seed range, commit, completion count, average reward, archive
path, and summary path when handing off a transferred run.
