import pytest

from capx.envs.launch import LaunchArgs
from capx.envs.scripts.run_batch import BatchLaunchArgs
from capx.envs.scripts.run_libero_batch import LiberoBatchLaunchArgs


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(LaunchArgs(config_path="config.yaml"), id="launch"),
        pytest.param(BatchLaunchArgs(), id="batch"),
        pytest.param(LiberoBatchLaunchArgs(), id="libero-batch"),
    ],
)
def test_cli_experiment_entry_points_default_to_4096_tokens(args: object) -> None:
    assert getattr(args, "max_tokens") == 4096
