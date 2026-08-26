# Cube Lift privileged Capsule replay smoke (2026-08-27)

## Scope and identity

This is readiness evidence for the privileged, high-level Cube Lift Capsule profile. The tested
Windows implementation commit was
`ce988c9e4b6677ad48a1188f602c896262bb165f`. Before verification, 24 scoped implementation,
configuration, and test files were copied from the isolated Windows worktree to the prepared WSL
runtime and a source/destination SHA-256 manifest matched for all 24 files. The runtime-imported
`capx/integrations/__init__.py` was subsequently synchronized from the same tested commit after an
initial PyRoKi-only startup exposed an older, incompatible WSL copy; its source and destination
SHA-256 were both
`5a36330fad26582601a8ae07439943b3250933987eb650cf2821aa9543b1f084`.

The immutable inputs were:

| Input | SHA-256 |
| --- | --- |
| Capsule YAML | `7618719687f1f3aec852182835c138bf23b521e38602191e4f469ff5d9612973` |
| Clean-replay environment YAML | `80c2c07be45238acdce187f39a762dfd8070d86b3617e7c8785b4c0371ba231c` |
| Source-task JSONL | `d189418578d86bcb6faa93e72287ecf3df454a368b84901b6d0a3bdcfb133bb1` |

## Commands and verification

All Python, pytest, Ruff, Robosuite, and PyRoKi commands ran in the prepared WSL project at
`/home/capx/code/cap-x`; no dependency installation or `uv sync` was performed.

Focused regression command:

```powershell
wsl.exe -d Ubuntu-22.04 --cd /home/capx/code/cap-x --exec /usr/bin/env LANG=C.UTF-8 LC_ALL=C.UTF-8 PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin MUJOCO_GL=egl /home/capx/.local/bin/uv run --no-sync python -m pytest -p no:cacheprovider tests/test_capsule_config.py tests/test_capsule_scripts.py tests/test_capsule_main_ppo.py tests/test_capsule_initial_state.py tests/test_capsule_deterministic_reset.py tests/test_robosuite_observation_privilege.py tests/test_capsule_evaluator.py tests/test_capsule_server_adapter.py tests/test_capsule_cube_lift_smoke.py tests/test_capsule_scripts_package.py tests/test_capsule_server_factory.py -q
```

Fresh pre-commit result: `586 passed, 2 warnings in 37.29s`, exit code 0. Both warnings were external PyRoKi
`Cost.create_factory` deprecation warnings.

Scoped Ruff command:

```powershell
wsl.exe -d Ubuntu-22.04 --cd /home/capx/code/cap-x --exec /usr/bin/env LANG=C.UTF-8 LC_ALL=C.UTF-8 PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin /home/capx/.local/bin/uv run --no-sync ruff check capx/rl/capsule/task_profiles.py capx/rl/capsule/compat.py capx/rl/capsule/main_ppo.py capx/rl/capsule/initial_state.py capx/rl/capsule/evaluator.py capx/rl/capsule/server_factory.py capx/envs/simulators/robosuite_cube_lift.py scripts/capsule_rl/common.py scripts/capsule_rl/server_adapter.py scripts/capsule_rl/cube_lift_privileged_replay_smoke.py tests/test_capsule_config.py tests/test_capsule_scripts.py tests/test_capsule_main_ppo.py tests/test_capsule_initial_state.py tests/test_capsule_deterministic_reset.py tests/test_robosuite_observation_privilege.py tests/test_capsule_evaluator.py tests/test_capsule_server_adapter.py tests/test_capsule_cube_lift_smoke.py tests/test_capsule_scripts_package.py tests/test_capsule_server_factory.py
```

Ruff was unavailable in the prepared frozen environment: `uv` returned exit code 1 with
`Failed to spawn: ruff` and `No such file or directory (os error 2)`. Ruff was not installed, so
this record does not claim a successful lint run.

Side-effect-free input validation command:

```powershell
wsl.exe -d Ubuntu-22.04 --cd /home/capx/code/cap-x --exec /usr/bin/env LANG=C.UTF-8 LC_ALL=C.UTF-8 PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin MUJOCO_GL=egl /home/capx/.local/bin/uv run --no-sync python -m scripts.capsule_rl.cube_lift_privileged_replay_smoke --config env_configs/cube_lifting/capsule_rl/franka_robosuite_cube_lift_capsule_smoke.yaml --source-task env_configs/cube_lifting/capsule_rl/cube_lift_capsule_source_tasks.jsonl --validate-only
```

Result: exit code 0, profile `robosuite_cube_lift_privileged_highlevel`, PyRoKi endpoint
`127.0.0.1:8116`, and the three input hashes above. It created no artifact or runtime process.

PyRoKi-only service command:

```powershell
wsl.exe -d Ubuntu-22.04 --cd /home/capx/code/cap-x --exec /usr/bin/env LANG=C.UTF-8 LC_ALL=C.UTF-8 PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin MUJOCO_GL=egl /home/capx/.local/bin/uv run --no-sync capx/serving/launch_servers.py --config-path env_configs/cube_lifting/capsule_rl/franka_robosuite_cube_lift_privileged_clean_replay.yaml --timeout 120 --log-dir artifacts/cube_lift_privileged_smoke_20260827/pyroki_only_r02_logs
```

Port 8116 had no listener before startup. The clean-replay YAML declared exactly one service,
PyRoKi. After startup, an independent `ss -ltnp sport = :8116` check showed one `python3`
listener on `127.0.0.1:8116` with service PID 25. An independent
`curl --fail --silent --show-error --max-time 10 http://127.0.0.1:8116/openapi.json` returned
HTTP success and an OpenAPI document containing only `/ik` and `/plan` paths. The successful child
log is
`artifacts/cube_lift_privileged_smoke_20260827/pyroki_only_r02_logs/pyroki_8116.log`.

An earlier r01 service attempt exited before readiness because the WSL runtime contained a drifted
`capx/integrations/__init__.py` that passed unsupported `config_factory=` to the matching
`register_api()` implementation. No smoke ran and no smoke artifact was created during that
attempt. The mismatch was confirmed by source/destination hashes and a file diff, then corrected
by copying the tested commit's file into the WSL runtime. The failed startup log remains at
`artifacts/cube_lift_privileged_smoke_20260827/pyroki_only_r01_logs/pyroki_8116.log`.

The exact runtime-only remediation and hash check were:

```powershell
wsl.exe -d Ubuntu-22.04 --exec /bin/cp /mnt/f/code/cap-x/.worktrees/cube-lift-privileged-smoke/capx/integrations/__init__.py /home/capx/code/cap-x/capx/integrations/__init__.py
wsl.exe -d Ubuntu-22.04 --exec /usr/bin/env LANG=C.UTF-8 LC_ALL=C.UTF-8 /usr/bin/sha256sum /mnt/f/code/cap-x/.worktrees/cube-lift-privileged-smoke/capx/integrations/__init__.py /home/capx/code/cap-x/capx/integrations/__init__.py
```

Real smoke command:

```powershell
wsl.exe -d Ubuntu-22.04 --cd /home/capx/code/cap-x --exec /usr/bin/env LANG=C.UTF-8 LC_ALL=C.UTF-8 PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin MUJOCO_GL=egl /home/capx/.local/bin/uv run --no-sync python -m scripts.capsule_rl.cube_lift_privileged_replay_smoke --config env_configs/cube_lifting/capsule_rl/franka_robosuite_cube_lift_capsule_smoke.yaml --source-task env_configs/cube_lifting/capsule_rl/cube_lift_capsule_source_tasks.jsonl --seed-sequence 5,6,5 --replay-seed 5 --replays 2 --timeout-s 180 --output artifacts/cube_lift_privileged_smoke_20260827/smoke.json
```

Result: exit code 0 in 21.59 seconds. The JSON artifact was then read back independently and every
invariant listed below was asserted against the persisted data.

## Reset and replay evidence

The reset sequence was exactly `5,6,5`:

| Reset | Seed | Initial-state SHA-256 |
| --- | ---: | --- |
| 1 | 5 | `45a083a869f0da2b5b59fcade4f71eddd74eb2926938309f8c7354c0a80bc3eb` |
| 2 | 6 | `b35dd8569134d7968f1429257af7fdcb7b0ad2848de3d3ea444cefa3e092b6d0` |
| 3 | 5 | `45a083a869f0da2b5b59fcade4f71eddd74eb2926938309f8c7354c0a80bc3eb` |

Thus both seed-5 resets were identical and the seed-6 reset was different.

| Replay | Worker PID | Outcome | Binary reward | Complete | Attempts | State SHA-256 | Replaced | Retry | Fresh namespace | API state cleared | Reset count / confirmed |
| --- | ---: | --- | ---: | --- | ---: | --- | --- | --- | --- | --- | --- |
| 1 | 276 | `success` | 1.0 | true | 1 | `45a083a869f0da2b5b59fcade4f71eddd74eb2926938309f8c7354c0a80bc3eb` | false | false | true | true | 1 / 1 |
| 2 | 276 | `success` | 1.0 | true | 1 | `45a083a869f0da2b5b59fcade4f71eddd74eb2926938309f8c7354c0a80bc3eb` | false | false | true | true | 1 / 1 |

Both calls therefore used one positive, persistent, unreplaced worker PID and reproduced the
probed seed-5 state with clean reset evidence.

The immutable artifact is
`/home/capx/code/cap-x/artifacts/cube_lift_privileged_smoke_20260827/smoke.json`, with SHA-256
`bbb4a5dbfe0f261de7fddb5d59bec0d39f62f173ba02aabaa23cff75f718ca75`. Its
`render_enabled` and `record_video` fields are both false. After the artifact was safely written,
Ctrl-C was sent to the same launcher PTY; the launcher terminated only its PyRoKi child. A final
`ss` check showed no listener on 8116, and `curl` returned connection refused.

## Interpretation boundary

This run used the existing privileged Cube Lift oracle solely for deterministic clean replay. It
did not invoke the perception stack, Program actor sampling, Controller repair, 7+1 group
assembly, Ray, VeRL, Gates 4-7, or an optimizer step. The persisted flags
`program_actor_used`, `controller_used`, `ray_used`, `verl_used`, and `optimizer_used` are all
false. This is environment/profile/replay readiness evidence, not a Capsule training result.
