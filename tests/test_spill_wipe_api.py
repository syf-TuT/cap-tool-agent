import pytest

from capx.integrations.franka.spill_wipe import FrankaControlSpillWipeApi
from capx.integrations.franka.spill_wipe_privileged import FrankaControlSpillWipePrivilegedApi


@pytest.mark.parametrize(
    "api_cls",
    (FrankaControlSpillWipeApi, FrankaControlSpillWipePrivilegedApi),
)
def test_spill_wipe_apis_declare_object_pose_as_recovery_observation_function(api_cls):
    api = object.__new__(api_cls)

    assert api.recovery_observation_functions() == {"get_object_pose"}
