from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from capx.envs.scripts import run_libero_batch


@pytest.mark.parametrize(
    ("n_tasks", "task_ids", "expected"),
    [
        (4, None, [0, 1, 2, 3]),
        (10, [7, 0, 7], [7, 0]),
        (0, None, []),
    ],
)
def test_select_task_ids_preserves_requested_order(
    n_tasks: int, task_ids: list[int] | None, expected: list[int]
) -> None:
    assert run_libero_batch._select_task_ids(n_tasks, task_ids) == expected


@pytest.mark.parametrize("task_ids", [[-1], [10], [0, 11]])
def test_select_task_ids_rejects_out_of_range_ids(task_ids: list[int]) -> None:
    with pytest.raises(ValueError, match="task ID"):
        run_libero_batch._select_task_ids(10, task_ids)


class _FakeSuite:
    def __init__(self, init_states_by_task: dict[int, object]) -> None:
        self.n_tasks = len(init_states_by_task)
        self.init_states_by_task = init_states_by_task
        self.task_requests: list[int] = []
        self.init_state_requests: list[int] = []

    def get_task(self, task_id: int) -> SimpleNamespace:
        self.task_requests.append(task_id)
        return SimpleNamespace(name=f"task-{task_id}")

    def get_task_init_states(self, task_id: int) -> object:
        self.init_state_requests.append(task_id)
        result = self.init_states_by_task[task_id]
        if isinstance(result, Exception):
            raise result
        return result


def test_collect_tasks_limits_each_suite_to_requested_task_ids() -> None:
    suite = _FakeSuite({0: ["state-0"], 1: ["state-1a", "state-1b"]})

    tasks = run_libero_batch._collect_tasks(
        {"libero_object": lambda: suite}, ["libero_object"], [0]
    )

    assert tasks == [("libero_object", 0, "task-0", 1)]
    assert suite.task_requests == [0]
    assert suite.init_state_requests == [0]


def test_collect_tasks_uses_all_tasks_and_preserves_init_state_fallback() -> None:
    suite = _FakeSuite(
        {
            0: ["only-state"],
            1: [],
            2: RuntimeError("states unavailable"),
        }
    )

    tasks = run_libero_batch._collect_tasks(
        {"libero_object": lambda: suite}, ["libero_object"], None
    )

    assert tasks == [
        ("libero_object", 0, "task-0", 1),
        ("libero_object", 1, "task-1", 0),
        ("libero_object", 2, "task-2", 50),
    ]
    assert suite.task_requests == [0, 1, 2]
    assert suite.init_state_requests == [0, 1, 2]


def _write_base_config(path: Path) -> None:
    path.write_text("env:\n  cfg:\n    privileged: false\n", encoding="utf-8")


def test_main_passes_task_filter_and_wrist_camera_to_launch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    base_config_path = tmp_path / "base.yaml"
    _write_base_config(base_config_path)
    suite = _FakeSuite({0: ["state-0"], 1: ["state-1"]})
    launch_calls: list[object] = []
    monkeypatch.setattr(
        run_libero_batch,
        "_load_benchmark_dict",
        lambda: {"libero_object": lambda: suite},
    )
    monkeypatch.setattr(run_libero_batch, "launch_main", launch_calls.append)

    args = run_libero_batch.LiberoBatchLaunchArgs(
        base_config_path=str(base_config_path),
        suites=["libero_object"],
        task_ids=[0],
        models=["test-model"],
        output_dir=str(tmp_path / "outputs"),
        use_wrist_camera=True,
    )
    run_libero_batch.main(args)

    assert suite.task_requests == [0]
    assert len(launch_calls) == 1
    assert launch_calls[0].use_wrist_camera is True


def test_main_validates_task_ids_before_creating_output_or_launching(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    base_config_path = tmp_path / "base.yaml"
    _write_base_config(base_config_path)
    output_dir = tmp_path / "outputs"
    suite = _FakeSuite({0: ["state-0"]})
    launch_calls: list[object] = []
    monkeypatch.setattr(
        run_libero_batch,
        "_load_benchmark_dict",
        lambda: {"libero_object": lambda: suite},
    )
    monkeypatch.setattr(run_libero_batch, "launch_main", launch_calls.append)

    args = run_libero_batch.LiberoBatchLaunchArgs(
        base_config_path=str(base_config_path),
        suites=["libero_object"],
        task_ids=[1],
        models=["test-model"],
        output_dir=str(output_dir),
    )
    with pytest.raises(ValueError, match="task ID"):
        run_libero_batch.main(args)

    assert not output_dir.exists()
    assert launch_calls == []
