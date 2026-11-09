from __future__ import annotations

import copy
import os
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tyro
import yaml

from capx.envs.launch import LaunchArgs
from capx.envs.launch import main as launch_main

STANDARD_SUITES = ("libero_object", "libero_spatial", "libero_goal")
PRO_SUITES = (
    "libero_object_swap",
    "libero_object_task",
    "libero_goal_swap",
    "libero_goal_task",
    "libero_spatial_swap",
    "libero_spatial_task",
)
ABLATION_CONFIGS = {
    "a1": "franka_libero_a1_one_shot.yaml",
    "a2": "franka_libero_capsule_llm_step_a2_no_patch.yaml",
    "a3": "franka_libero_capsule_llm_step_a3_no_append_recovery.yaml",
    "a4": "franka_libero_capsule_llm_step_a4_full.yaml",
}


@dataclass
class LiberoBatchLaunchArgs:
    """Command-line arguments for automated Libero batch execution."""

    base_config_path: str = "env_configs/libero/franka_libero_cap_agent0.yaml"
    ablation: bool = False
    # Defaults to the three standard suites in ablation mode, or six PRO suites otherwise.
    suites: list[str] | None = None
    # Select these zero-based task indices in each selected suite.
    task_ids: list[int] | None = None
    models: list[str] = field(default_factory=lambda: ["google/gemini-3.1-pro-preview"])
    server_url: str = "http://127.0.0.1:8110/chat/completions"
    output_dir: str = "./outputs/libero_batch_run"
    temperature: float = 1.0
    max_tokens: int = 4096
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


def _load_benchmark_dict() -> Mapping[str, Callable[[], Any]]:
    # LIBERO checkpoints may use the legacy PyTorch serialization format.
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    try:
        from libero import benchmark
    except ImportError as exc:
        raise RuntimeError("LIBERO is unavailable. Install the libero extra.") from exc
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
    benchmarks: Mapping[str, Callable[[], Any]],
    suites: list[str],
    task_ids: list[int] | None,
    *,
    ablation: bool = False,
    total_trials: int | None = None,
) -> list[tuple[str, int, str, int]]:
    tasks = []
    for suite_name in suites:
        if suite_name not in benchmarks:
            if ablation:
                raise ValueError(f"LIBERO suite {suite_name!r} is unavailable")
            print(f"Warning: Suite {suite_name!r} not found in LIBERO benchmarks.")
            continue

        suite = benchmarks[suite_name]()
        for task_id in _select_task_ids(suite.n_tasks, task_ids):
            task_name = suite.get_task(task_id).name
            try:
                num_trials = len(suite.get_task_init_states(task_id))
            except Exception as exc:
                if ablation:
                    raise ValueError(
                        f"Cannot load initial states for {suite_name}/{task_name}"
                    ) from exc
                print(f"Warning: Could not count initial states for {task_name}; using 50. {exc}")
                num_trials = 50

            if ablation:
                if num_trials == 0:
                    raise ValueError(f"No initial states for {suite_name}/{task_name}")
                if total_trials is not None and not 1 <= total_trials <= num_trials:
                    raise ValueError(
                        f"--total-trials must be between 1 and {num_trials} "
                        f"for {suite_name}/{task_name}"
                    )
                num_trials = total_trials if total_trials is not None else num_trials
            tasks.append((suite_name, task_id, task_name, num_trials))
    return tasks


def _load_templates(args):
    paths = {None: Path(args.base_config_path)}
    if args.ablation:
        config_dir = Path(__file__).resolve().parents[3] / "env_configs" / "libero"
        paths = {group: config_dir / filename for group, filename in ABLATION_CONFIGS.items()}
    return {group: yaml.safe_load(path.read_text(encoding="utf-8")) for group, path in paths.items()}


def main(args: LiberoBatchLaunchArgs) -> None:
    suites = args.suites if args.suites is not None else (
        STANDARD_SUITES if args.ablation else PRO_SUITES
    )
    if args.ablation:
        if not suites or any(suite not in STANDARD_SUITES for suite in suites):
            raise ValueError(f"--ablation requires suites from {', '.join(STANDARD_SUITES)}")
        if args.task_ids == []:
            raise ValueError("--task-ids requires at least one task ID")
        if args.use_visual_feedback or args.use_img_differencing or args.use_oracle_code:
            raise ValueError(
                "--ablation requires model-generated code with visual feedback and differencing off"
            )
        model_dirs = {model.replace("/", "_") for model in args.models}
        if not model_dirs or len(model_dirs) != len(args.models):
            raise ValueError("--ablation requires models with distinct output directory names")

    templates = _load_templates(args)
    tasks = _collect_tasks(
        _load_benchmark_dict(), list(dict.fromkeys(suites)), args.task_ids,
        ablation=args.ablation, total_trials=args.total_trials,
    )
    if args.ablation and not tasks:
        raise ValueError("No LIBERO tasks selected for ablation")

    total_runs = len(args.models) * len(tasks) * len(templates)
    print(f"Total experimental runs: {total_runs}")
    failed_runs = []
    for model in args.models:
        for suite_name, task_id, task_name, num_trials in tasks:
            for group, template in templates.items():
                config = copy.deepcopy(template)
                env_config = config["env"]["cfg"]
                task_dir = Path(args.output_dir).resolve() / suite_name / task_name
                if group is None:
                    env_config["low_level"] = {
                        "_target_": "capx.envs.simulators.libero.FrankaLiberoEnv",
                        "privileged": env_config.get("privileged", False),
                        "max_steps": 8000,
                        "seed": None,
                        "enable_render": True,
                        "viser_debug": False,
                    }
                    run_dir = task_dir
                    config_dir = task_dir
                else:
                    run_dir = task_dir / group
                    config_dir = run_dir / model.replace("/", "_")

                env_config["low_level"].update(suite_name=suite_name, task_id=task_id)
                config["trials"] = num_trials
                # The launcher inserts the model name before the final path component.
                config["output_dir"] = (run_dir / "run").as_posix()
                config_dir.mkdir(parents=True, exist_ok=True)
                config_path = config_dir / "config.yaml"
                config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

                label = f"{suite_name}/{task_name}/{group}" if group else f"{suite_name}/{task_name}"
                print(f"Running {label} with {model} ({num_trials} configured trials)")
                launch_args = LaunchArgs(
                    config_path=str(config_path),
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
                    debug=args.debug,
                    use_oracle_code=args.use_oracle_code,
                )
                try:
                    launch_main(launch_args)
                except Exception:
                    traceback.print_exc()
                    failed_runs.append(f"{label} with {model}")

    if failed_runs:
        raise SystemExit("LIBERO batch failed for:\n" + "\n".join(failed_runs))
    print("Batch execution completed successfully.")


def _parse_args(cli_args: list[str] | None = None) -> LiberoBatchLaunchArgs:
    return tyro.cli(LiberoBatchLaunchArgs, args=cli_args)


if __name__ == "__main__":
    main(_parse_args())
