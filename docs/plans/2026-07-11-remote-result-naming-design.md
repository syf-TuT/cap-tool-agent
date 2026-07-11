# Remote Result Naming Design

## Goal

Give every downloaded experiment run a short, sortable, and unambiguous local directory name
without encoding the entire experiment configuration in the path.

## Canonical name

Within `remote_results/<task>/`, name each run directory as:

```text
<YYYYMMDDTHHMMZ>_<method>_<model>_<seeds>[_rNN]
```

Examples:

```text
20260711T0614Z_capsule_dsv4flash_s01-05
20260711T0830Z_original-mt_dsv4flash_s01-20
20260711T1015Z_capsule_dsv4flash_s17_r02
20260711T1110Z_capsule_dsv4flash_s10+14+16-18
```

The timestamp is UTC. Method and model identifiers are stable lowercase aliases. Seed notation is
`sNN` for one seed, `sNN-NN` for a continuous range, and `sNN+NN+NN-NN` for a mixed set. `_rNN`
is present only when rerunning the same configuration.

## Boundaries

The task is already represented by the parent directory and is not repeated. Mutable outcomes
such as reward, success rate, and completion state do not belong in the name. Provider, full model
name, Git commit, timeouts, token limits, and feature flags belong in `run.json`.

Run ids use ASCII lowercase letters, digits, hyphens, underscores, and the seed-list plus sign.
They should remain at or below 64 characters to reduce Windows long-path risk.

## Metadata

Each run directory contains a `run.json` with schema version, run id, task, method, full model and
provider names, seeds, Git commit, UTC start time, rerun number, and relevant behavior flags. The
manifest is local experiment metadata and remains ignored along with `remote_results/`.
