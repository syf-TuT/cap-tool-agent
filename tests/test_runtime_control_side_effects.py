from capx.runtime_control.side_effects import collect_side_effect_calls


class _FakeApi:
    def __init__(self, names):
        self._names = names

    def side_effect_functions(self):
        return set(self._names)


class _SensingOnlyApi:
    """An API that declares no side effects (all sensing/compute)."""

    def side_effect_functions(self):
        return set()


class _LegacyApi:
    """An older API that predates the declaration method."""


def test_collect_returns_empty_for_no_apis():
    assert collect_side_effect_calls([]) == set()


def test_collect_gathers_declared_side_effect_names():
    apis = [_FakeApi({"goto_pose", "close_gripper"})]

    assert collect_side_effect_calls(apis) == {"goto_pose", "close_gripper"}


def test_collect_merges_across_multiple_apis():
    apis = [_FakeApi({"goto_pose"}), _FakeApi({"open_gripper", "goto_home_joint_position"})]

    assert collect_side_effect_calls(apis) == {
        "goto_pose",
        "open_gripper",
        "goto_home_joint_position",
    }


def test_collect_skips_apis_without_declaration_method():
    apis = [_FakeApi({"goto_pose"}), _LegacyApi(), _SensingOnlyApi()]

    assert collect_side_effect_calls(apis) == {"goto_pose"}
