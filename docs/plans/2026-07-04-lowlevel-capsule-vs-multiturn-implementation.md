# Low-Level Capsule vs Multiturn Benchmark Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a reproducible Robosuite pilot benchmark comparing low-level Capsule against low-level multiturn regeneration.

**Architecture:** Add paired YAML configs for each task/method, then run randomized single-trial jobs on SeeTaCloud with a remote shell runner. A parser aggregates summaries, traces, and response logs into a CSV.

**Tech Stack:** CaP-X YAML configs, Python `LaunchArgs`, Bash `timeout`, Robosuite, Pyroki, DeepSeek via Packy API.

---

### Task 1: Add Low-Level Benchmark Configs

**Files:**
- Create: `env_configs/benchmarks/lowlevel_primitives/cube_stack_mt_regenerate.yaml`
- Create: `env_configs/benchmarks/lowlevel_primitives/cube_stack_capsule.yaml`
- Create: `env_configs/benchmarks/lowlevel_primitives/nut_assembly_mt_regenerate.yaml`
- Create: `env_configs/benchmarks/lowlevel_primitives/nut_assembly_capsule.yaml`

**Steps:**

1. Use privileged low-level APIs only.
2. Disable SAM/contact/VDM/image/video differencing.
3. Set `record_video: true`, `trials: 1`, `num_workers: 1`.
4. Set Capsule configs to `agent_mode: capsule` and `max_capsule_steps: 60`.
5. Set baseline configs to include `multi_turn_prompt` and no `agent_mode`.

### Task 2: Add Remote Single-Run Launcher

**Files:**
- Create: `.codex-upload/benchmark_lowlevel_single.py`

**Steps:**

1. Read `PACKY_API_KEY` from environment or `/root/autodl-tmp/cap-x/.packy_env`.
2. Load a base YAML.
3. Write a temporary YAML with `trials=<trial_id>` and `resume_idx=<trial_id>`.
4. Call `capx.envs.launch.main(LaunchArgs(...))`.
5. Use no secrets in logs or command-line arguments.

### Task 3: Add Remote Randomized Batch Runner

**Files:**
- Create: `.codex-upload/remote_run_lowlevel_benchmark.sh`

**Steps:**

1. Define the 20-run pilot matrix.
2. Randomize with a fixed seed.
3. Run each row with `timeout --kill-after=15s 720s`.
4. Save per-run logs under `.codex_benchmark_lowlevel/logs`.
5. Continue after failed runs and write a manifest TSV.

### Task 4: Add Result Parser

**Files:**
- Create: `.codex-upload/benchmark_lowlevel_collect.py`

**Steps:**

1. Read output directories for every manifest row.
2. Parse `summaries.txt` for success, reward, runtime, regeneration counts.
3. Parse `capsule_trace_trial_*.json` for Capsule action counts.
4. Parse `all_responses.json` for baseline LLM decision counts.
5. Write `benchmark_lowlevel_results.csv`.

### Task 5: Verify Before Running the Full Pilot

**Commands:**

```powershell
wsl.exe -d Ubuntu-22.04 --cd /home/capx/code/cap-x -- .venv/bin/python -c "from capx.envs.launch import LaunchArgs; from capx.utils.launch_utils import _load_config; paths=['env_configs/benchmarks/lowlevel_primitives/cube_stack_mt_regenerate.yaml','env_configs/benchmarks/lowlevel_primitives/cube_stack_capsule.yaml','env_configs/benchmarks/lowlevel_primitives/nut_assembly_mt_regenerate.yaml','env_configs/benchmarks/lowlevel_primitives/nut_assembly_capsule.yaml']; [print(p, _load_config(LaunchArgs(config_path=p))[1]['agent_mode'], _load_config(LaunchArgs(config_path=p))[1]['record_video']) for p in paths]"
```

Expected: all configs load; Capsule configs report `capsule`; multiturn configs report `code`; all report `record_video=True`.
