import capx.runtime_control as runtime_control
from capx.runtime_control.contract import (
    ProgramContractViolation,
    analyze_capsule_program_contract,
)
from capx.runtime_control.normalizer import segment_python_code_groups
from capx.runtime_control.segmenter import segment_python_code


SIDE_EFFECTS = {"goto_pose", "open_gripper", "close_gripper"}


def _analyze(source: str) -> list[ProgramContractViolation]:
    regions = segment_python_code(source)
    groups = segment_python_code_groups(
        source,
        regions,
        side_effect_calls=SIDE_EFFECTS,
    )
    return analyze_capsule_program_contract(
        source,
        regions,
        groups,
        side_effect_calls=SIDE_EFFECTS,
    )


def test_allows_pure_helpers_and_one_side_effect_per_top_level_group():
    source = """\
def offset(p):
    return p + 0.1
target = offset(0.2)
goto_pose(target, [1, 0, 0, 0])
verified = True
open_gripper()
"""

    assert _analyze(source) == []


def test_rejects_direct_and_transitive_effectful_helpers():
    source = """\
def inner():
    close_gripper()
def outer():
    inner()
outer()
"""

    violations = _analyze(source)

    assert {
        violation.helper_name
        for violation in violations
        if violation.code == "effectful_helper"
    } == {"inner", "outer"}


def test_rejects_effectful_loops_and_try_blocks():
    source = """\
for item in items:
    open_gripper()
try:
    close_gripper()
except RuntimeError:
    recovered = False
"""

    violations = _analyze(source)

    assert [
        violation.code
        for violation in violations
        if violation.code == "effectful_control_flow"
    ] == ["effectful_control_flow", "effectful_control_flow"]


def test_control_flow_resolves_transitive_helper_effects_and_cycles():
    source = """\
def first():
    second()
def second():
    first()
    open_gripper()
while ready:
    first()
"""

    violations = _analyze(source)

    assert {
        violation.helper_name
        for violation in violations
        if violation.code == "effectful_helper"
    } == {"first", "second"}
    assert any(
        violation.code == "effectful_control_flow"
        and violation.side_effect_calls == ("open_gripper",)
        for violation in violations
    )


def test_nested_definition_body_is_not_an_executed_helper_effect():
    source = """\
def outer():
    def nested():
        close_gripper()
    return True
outer()
"""

    assert _analyze(source) == []


def test_counts_repeated_side_effect_occurrences_within_a_group():
    source = """\
goto_pose([0, 0, 0], [1, 0, 0, 0])
goto_pose([1, 0, 0], [1, 0, 0, 0])
"""

    violations = _analyze(source)

    repeated = [
        violation
        for violation in violations
        if violation.code == "multiple_effects_in_group"
    ]
    assert len(repeated) == 1
    assert repeated[0].side_effect_calls == ("goto_pose", "goto_pose")


def test_violation_to_dict_uses_public_serialization_shape():
    violation = ProgramContractViolation(
        code="effectful_helper",
        message="Helper 'move' can execute a robot side effect",
        start_line=2,
        end_line=4,
        region_ids=("region_1",),
        group_ids=("group_1",),
        side_effect_calls=("goto_pose",),
        helper_name="move",
    )

    assert violation.to_dict() == {
        "code": "effectful_helper",
        "message": "Helper 'move' can execute a robot side effect",
        "source_span": {"start_line": 2, "end_line": 4},
        "region_ids": ["region_1"],
        "group_ids": ["group_1"],
        "side_effect_calls": ["goto_pose"],
        "helper_name": "move",
    }


def test_contract_types_are_exported_from_runtime_control_package():
    assert runtime_control.ProgramContractViolation is ProgramContractViolation
    assert (
        runtime_control.analyze_capsule_program_contract
        is analyze_capsule_program_contract
    )
