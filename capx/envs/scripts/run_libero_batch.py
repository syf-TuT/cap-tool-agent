from __future__ import annotations

import copy
import os
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import tyro
import yaml

from capx.envs.launch import LaunchArgs
from capx.envs.launch import main as launch_main

# Force weights_only=False for PyTorch loading of legacy files.
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"


def _load_benchmark_dict() -> Mapping[str, Callable[[], Any]]:
    try:
        from libero import benchmark
    except ImportError as exc:
        raise RuntimeError(
            "LIBERO is not installed; run this command in the dedicated LIBERO environment."
        ) from exc
    return benchmark.get_benchmark_dict()


def _select_task_ids(n_tasks: int, task_ids: list[int] | None) -> list[int]:
    if isinstance(n_tasks, bool) or not isinstance(n_tasks, int) or n_tasks < 0:
        raise ValueError(f"n_tasks must be a non-negative integer, got {n_tasks!r}")
    if task_ids is None:
        return list(range(n_tasks))

    selected: list[int] = []
    seen: set[int] = set()
    for task_id in task_ids:
        if (
            isinstance(task_id, bool)
            or not isinstance(task_id, int)
            or task_id < 0
            or task_id >= n_tasks
        ):
            raise ValueError(
                f"task ID {task_id!r} is outside the valid range [0, {n_tasks})"
            )
        if task_id not in seen:
            selected.append(task_id)
            seen.add(task_id)
    return selected


def _collect_tasks(
    benchmark_dict: Mapping[str, Callable[[], Any]],
    suite_names: list[str],
    task_ids: list[int] | None,
) -> list[tuple[str, int, str, int]]:
    selected_suites: list[tuple[str, Any, list[int]]] = []
    for suite_name in suite_names:
        if suite_name not in benchmark_dict:
            print(f"Warning: Suite '{suite_name}' not found in Libero benchmarks.")
            continue

        task_suite = benchmark_dict[suite_name]()
        num_tasks = task_suite.n_tasks
        print(f"Suite {suite_name} has {num_tasks} tasks.")
        selected_suites.append(
            (suite_name, task_suite, _select_task_ids(num_tasks, task_ids))
        )

    tasks_to_run: list[tuple[str, int, str, int]] = []
    for suite_name, task_suite, selected_task_ids in selected_suites:
        for task_id in selected_task_ids:
            task = task_suite.get_task(task_id)
            task_name = task.name

            try:
                init_states = task_suite.get_task_init_states(task_id)
                assert init_states is not None, f"No initial states found for task {task_name}"
                num_trials = len(init_states)
            except Exception as exc:
                print(
                    f"Warning: Could not determine init states for {task_name}, "
                    f"defaulting to 50. Error: {exc}"
                )
                num_trials = 50

            tasks_to_run.append((suite_name, task_id, task_name, num_trials))

    return tasks_to_run


@dataclass
class LiberoBatchLaunchArgs:
    """Command-line arguments for automated Libero batch execution."""

    # Base configuration file
    base_config_path: str = "env_configs/libero/franka_libero_cap_agent0.yaml"

    # Suites to run
    suites: list[str] = field(
        default_factory=lambda: [
            "libero_object_swap",
            "libero_object_task",
            "libero_goal_swap",
            "libero_goal_task",
            "libero_spatial_swap",
            "libero_spatial_task",
        ]
    )

    # Optional task IDs to run from each selected suite (all tasks by default)
    task_ids: list[int] | None = None

    # Models to run (copied from run_batch.py default)
    models: list[str] = field(
        default_factory=lambda: [
            # "openai/gpt-5.4",
            "google/gemini-3.1-pro-preview"
        ]
    )

    server_url: str = "http://127.0.0.1:8110/chat/completions"  # local server

    # Output directory base
    output_dir: str = "./outputs/libero_batch_run"

    # Other LaunchArgs overrides
    temperature: float = 1.0
    max_tokens: int = 2048 * 10
    reasoning_effort: str = "medium"
    api_key: str | None = None
    use_visual_feedback: bool | None = None
    use_img_differencing: bool | None = None
    use_wrist_camera: bool | None = None
    use_legacy_multi_turn_decision_prompt: bool | None = None
    total_trials: int | None = None
    num_workers: int | None = None
    record_video: bool | None = None
    debug: bool = False
    use_oracle_code: bool | None = None


def _parse_args(cli_args: list[str] | None = None) -> LiberoBatchLaunchArgs:
    return tyro.cli(LiberoBatchLaunchArgs, args=cli_args)


def main(args: LiberoBatchLaunchArgs) -> None:
    # Load base configuration
    if not os.path.exists(args.base_config_path):
        print(f"Error: Base config file not found: {args.base_config_path}")
        sys.exit(1)
        
    with open(args.base_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    benchmark_dict = _load_benchmark_dict()
    
    print(f"Collecting tasks for suites: {args.suites}")
    tasks_to_run = _collect_tasks(benchmark_dict, args.suites, args.task_ids)

    print(f"Total tasks to run: {len(tasks_to_run)}")
    
    total_runs = len(args.models) * len(tasks_to_run)
    print(f"Total experimental runs (models x tasks): {total_runs}")
    
    experiment_idx = 1
    failed_runs = []
    
    for model in args.models:
        print(f"\n{'=' * 80}")
        print(f"Running model: {model}")
        print(f"{'=' * 80}")
        
        for suite_name, task_id, task_name, num_trials in tasks_to_run:
            print(f"\n{'-' * 80}")
            print(f"Running Experiment {experiment_idx}/{total_runs}")
            print(f"Model: {model}")
            print(f"Suite: {suite_name}, Task ID: {task_id}, Task Name: {task_name}")
            print(f"Trials: {num_trials}")
            print(f"{'-' * 80}\n")
            
            # Construct config from base template
            # Deep copy to ensure we don't modify the shared base_config for subsequent runs
            config = copy.deepcopy(base_config)

            # Inject task details
            # We overwrite the 'low_level' entry to be a dictionary definition for FrankaLiberoEnv
            # instead of the string reference (e.g. 'franka_libero_pick_place_low_level')
            
            # Check if env/cfg/privileged exists, use it if so
            is_privileged = config.get("env", {}).get("cfg", {}).get("privileged", False)
            
            config["env"]["cfg"]["low_level"] = {
                "_target_": "capx.envs.simulators.libero.FrankaLiberoEnv",
                "suite_name": suite_name,
                "task_id": task_id,
                "privileged": is_privileged,
                "max_steps": 8000,
                "seed": None,
                "enable_render": True,
                "viser_debug": False
            }
            
            # Set trials count dynamically
            config["trials"] = num_trials
            
            # Customize output directory
            # Structure: output_dir/suite_name/task_name/run
            # launch.py will append model_name automatically to the last component if we rely on its default behavior
            # But we want: output_dir/suite/task/model/run (or similar)
            
            # To get output_dir/suite/task/model/run:
            # We set config["output_dir"] = .../suite/task/run
            # launch.py splits: [..., suite, task, run]
            # Inserts model at -1: [..., suite, task, model, run]
            # Result: .../suite/task/model/run
            
            config_dir_base = os.path.join(os.path.abspath(args.output_dir), suite_name, task_name)
            config["output_dir"] = os.path.join(config_dir_base, "run")
            
            # Create the base directory if it doesn't exist to save the config
            os.makedirs(config_dir_base, exist_ok=True)
            
            # Save config permanently to the run directory (parent of "run" subfolder effectively)
            config_filename = "config.yaml"
            config_path = os.path.join(config_dir_base, config_filename)
            
            with open(config_path, "w") as f:
                yaml.dump(config, f)
            
            # Create LaunchArgs
            launch_args = LaunchArgs(
                config_path=config_path,
                server_url=args.server_url,
                model=model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                reasoning_effort=args.reasoning_effort,
                api_key=args.api_key,
                use_visual_feedback=args.use_visual_feedback,
                use_img_differencing=args.use_img_differencing,
                use_wrist_camera=args.use_wrist_camera,
                use_legacy_multi_turn_decision_prompt=args.use_legacy_multi_turn_decision_prompt,
                total_trials=args.total_trials,
                num_workers=args.num_workers,
                record_video=args.record_video,
                # We do NOT pass output_dir here to avoid the logic that uses config_stem
                output_dir=None, 
                debug=args.debug,
                use_oracle_code=args.use_oracle_code,
            )
            
            try:
                launch_main(launch_args)
            except Exception as e:
                print(f"\nERROR running {suite_name}/{task_name} with model {model}: {e}")
                traceback.print_exc()
                failed_runs.append((model, suite_name, task_name))
            
            experiment_idx += 1

    if failed_runs:
        print(f"\n{'=' * 80}")
        print(f"Batch execution completed with {len(failed_runs)} failures:")
        for model, suite, task in failed_runs:
            print(f"  - model={model}, suite={suite}, task={task}")
        print(f"{'=' * 80}")
        sys.exit(1)
    else:
        print(f"\n{'=' * 80}")
        print("Batch execution completed successfully.")
        print(f"{'=' * 80}")


if __name__ == "__main__":
    main(_parse_args())
