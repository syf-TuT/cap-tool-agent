import pytest

import capx.runtime_control as runtime_control
from capx.runtime_control.contract import (
    ProgramContractAnalysis,
    ProgramContractViolation,
    analyze_capsule_program_contract,
    analyze_capsule_program_contract_details,
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


def test_contract_details_mark_helper_calls_but_not_pure_definitions_effectful():
    source = """\
def move():
    close_gripper()
move()
"""
    regions = segment_python_code(source)
    groups = segment_python_code_groups(
        source,
        regions,
        max_regions_per_group=1,
        side_effect_calls=SIDE_EFFECTS,
    )

    analysis = analyze_capsule_program_contract_details(
        source,
        regions,
        groups,
        side_effect_calls=SIDE_EFFECTS,
    )

    assert analysis.effectful_region_ids == ("region_2",)
    assert analysis.effectful_group_ids == ("group_2",)


def test_contract_details_include_definition_time_and_class_body_effects():
    source = """\
@register(open_gripper())
def prepare(value=close_gripper()):
    pass
class Bad:
    goto_pose([0, 0, 0], [1, 0, 0, 0])
"""
    regions = segment_python_code(source)
    groups = segment_python_code_groups(
        source,
        regions,
        max_regions_per_group=1,
        side_effect_calls=SIDE_EFFECTS,
    )

    analysis = analyze_capsule_program_contract_details(
        source,
        regions,
        groups,
        side_effect_calls=SIDE_EFFECTS,
    )

    assert analysis.effectful_region_ids == tuple(
        region.region_id for region in regions
    )
    assert analysis.effectful_group_ids == tuple(group.group_id for group in groups)


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


def test_rejects_effectful_comprehensions_and_transitive_helper_calls():
    source = """\
def close_twice():
    close_gripper()
values = [close_twice() for _ in range(2)]
pending = (open_gripper() for _ in range(2))
"""

    violations = _analyze(source)

    comprehension_violations = [
        violation
        for violation in violations
        if violation.code == "effectful_control_flow"
    ]
    assert [item.start_line for item in comprehension_violations] == [
        3,
        4,
    ]
    assert comprehension_violations[0].side_effect_calls == ("close_gripper",)
    assert comprehension_violations[1].side_effect_calls == ("open_gripper",)


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
    assert runtime_control.ProgramContractAnalysis is ProgramContractAnalysis
    assert runtime_control.ProgramContractViolation is ProgramContractViolation
    assert (
        runtime_control.analyze_capsule_program_contract
        is analyze_capsule_program_contract
    )
    assert (
        runtime_control.analyze_capsule_program_contract_details
        is analyze_capsule_program_contract_details
    )


def test_duplicate_helper_definitions_do_not_hide_an_effectful_definition():
    source = """\
def move():
    close_gripper()
move()
def move():
    pass
"""

    violations = _analyze(source)

    effectful_helpers = [
        violation
        for violation in violations
        if violation.code == "effectful_helper"
    ]
    assert len(effectful_helpers) == 1
    assert effectful_helpers[0].helper_name == "move"
    assert effectful_helpers[0].start_line == 1
    assert effectful_helpers[0].side_effect_calls == ("close_gripper",)


def test_called_nested_helper_propagates_effect_to_its_top_level_helper():
    source = """\
def outer():
    def nested():
        close_gripper()
    nested()
outer()
"""

    violations = _analyze(source)

    assert {
        violation.helper_name
        for violation in violations
        if violation.code == "effectful_helper"
    } == {"outer"}


def test_function_definition_time_expressions_count_effect_occurrences():
    source = """\
@register(open_gripper())
def prepare(a=close_gripper(), *, target=goto_pose([0, 0, 0], [1, 0, 0, 0])):
    pass
"""

    violations = _analyze(source)

    repeated = [
        violation
        for violation in violations
        if violation.code == "multiple_effects_in_group"
    ]
    assert len(repeated) == 1
    assert repeated[0].side_effect_calls == (
        "open_gripper",
        "close_gripper",
        "goto_pose",
    )


def test_transitive_helper_effect_expansion_is_saturated():
    definitions = ["def helper_0():\n    goto_pose(target, orientation)"]
    for index in range(1, 11):
        definitions.append(
            f"def helper_{index}():\n"
            f"    helper_{index - 1}()\n"
            f"    helper_{index - 1}()"
        )
    source = "\n".join([*definitions, "helper_10()", ""])

    violations = _analyze(source)

    assert any(
        violation.code == "multiple_effects_in_group"
        and violation.side_effect_calls == ("goto_pose", "goto_pose")
        for violation in violations
    )
    assert all(len(violation.side_effect_calls) <= 2 for violation in violations)


def test_pure_helper_calls_do_not_hide_a_later_direct_effect():
    source = """\
def p1():
    return 1
def p2():
    return 2
def bad():
    p1()
    p2()
    close_gripper()
bad()
"""

    violations = _analyze(source)

    assert any(
        violation.code == "effectful_helper"
        and violation.helper_name == "bad"
        and violation.side_effect_calls == ("close_gripper",)
        for violation in violations
    )


def test_class_body_effects_count_as_definition_time_occurrences():
    source = """\
class Bad:
    open_gripper()
    close_gripper()
"""

    violations = _analyze(source)

    assert any(
        violation.code == "multiple_effects_in_group"
        and violation.side_effect_calls == ("open_gripper", "close_gripper")
        for violation in violations
    )


def test_function_annotation_effects_count_as_definition_time_occurrences():
    source = """\
def prepare(target: open_gripper()) -> close_gripper():
    pass
"""

    violations = _analyze(source)

    assert any(
        violation.code == "multiple_effects_in_group"
        and violation.side_effect_calls == ("open_gripper", "close_gripper")
        for violation in violations
    )


def test_pure_helper_dag_uses_one_bounded_summary_pass(monkeypatch):
    original = runtime_control.contract._compute_definition_effect_summaries
    summary_passes = 0

    def counted_summary_pass(*args, **kwargs):
        nonlocal summary_passes
        summary_passes += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        runtime_control.contract,
        "_compute_definition_effect_summaries",
        counted_summary_pass,
    )
    definitions = ["def helper_0():\n    return 0"]
    for index in range(1, 31):
        definitions.append(
            f"def helper_{index}():\n"
            f"    helper_{index - 1}()\n"
            f"    helper_{index - 1}()"
        )

    assert _analyze("\n".join([*definitions, "helper_30()", ""])) == []
    assert summary_passes == 1


def test_unreachable_nested_effectful_loop_is_not_reported():
    source = """\
def outer():
    def nested():
        for item in items:
            close_gripper()
    return True
outer()
"""

    violations = _analyze(source)

    assert all(
        violation.code != "effectful_control_flow"
        for violation in violations
    )
    assert all(
        violation.helper_name != "outer"
        for violation in violations
        if violation.code == "effectful_helper"
    )


def test_inner_local_binding_shadows_effectful_enclosing_binding():
    source = """\
def outer():
    def move():
        close_gripper()
    def inner():
        def move():
            return True
        move()
    inner()
outer()
"""

    violations = _analyze(source)

    assert all(
        violation.helper_name != "outer"
        for violation in violations
        if violation.code == "effectful_helper"
    )


def test_reachable_local_effect_cycle_propagates_with_a_bounded_summary():
    source = """\
def outer():
    def first():
        second()
    def second():
        first()
        close_gripper()
    first()
outer()
"""

    violations = _analyze(source)

    outer = next(
        violation
        for violation in violations
        if violation.code == "effectful_helper"
        and violation.helper_name == "outer"
    )
    assert outer.side_effect_calls == ("close_gripper",)


def test_rejects_simple_and_chained_side_effect_aliases():
    source = """\
alias = close_gripper
again = alias
again()
"""

    analysis = analyze_capsule_program_contract_details(
        source,
        segment_python_code(source),
        segment_python_code_groups(
            source,
            segment_python_code(source),
            side_effect_calls=SIDE_EFFECTS,
        ),
        side_effect_calls=SIDE_EFFECTS,
    )

    assert any(
        violation.code == "aliased_effect_call"
        and violation.side_effect_calls == ("close_gripper",)
        for violation in analysis.violations
    )
    assert analysis.effectful_region_ids
    assert analysis.effectful_group_ids


def test_lambda_alias_effect_is_only_executable_when_called():
    uncalled_source = "alias = lambda: close_gripper()\nvalue = 1\n"
    uncalled_regions = segment_python_code(uncalled_source)
    uncalled_groups = segment_python_code_groups(
        uncalled_source,
        uncalled_regions,
        side_effect_calls=SIDE_EFFECTS,
    )
    uncalled = analyze_capsule_program_contract_details(
        uncalled_source,
        uncalled_regions,
        uncalled_groups,
        side_effect_calls=SIDE_EFFECTS,
    )

    called_source = "alias = lambda: (open_gripper(), close_gripper())\nalias()\n"
    called_regions = segment_python_code(called_source)
    called_groups = segment_python_code_groups(
        called_source,
        called_regions,
        side_effect_calls=SIDE_EFFECTS,
    )
    called = analyze_capsule_program_contract_details(
        called_source,
        called_regions,
        called_groups,
        side_effect_calls=SIDE_EFFECTS,
    )

    assert uncalled.effectful_region_ids == ()
    assert uncalled.effectful_group_ids == ()
    assert any(
        violation.code == "aliased_effect_call"
        for violation in called.violations
    )
    assert any(
        violation.code == "multiple_effects_in_group"
        for violation in called.violations
    )
    assert called.effectful_region_ids
    assert called.effectful_group_ids


@pytest.mark.parametrize(
    "source",
    [
        'globals()["close_gripper"]()\n',
        'locals()["close_gripper"]()\n',
        'eval("close_gripper()")\n',
        'exec("close_gripper()")\n',
        'getattr(tool, function_name)()\n',
        'APIS["robot"].close_gripper()\n',
        'close_gripper.__wrapped__()\n',
    ],
)
def test_rejects_dynamic_runtime_effect_access(source):
    regions = segment_python_code(source)
    groups = segment_python_code_groups(
        source,
        regions,
        side_effect_calls=SIDE_EFFECTS,
    )

    analysis = analyze_capsule_program_contract_details(
        source,
        regions,
        groups,
        side_effect_calls=SIDE_EFFECTS,
    )

    assert any(
        violation.code in {"dynamic_effect_call", "forbidden_runtime_access"}
        for violation in analysis.violations
    )
    assert analysis.effectful_region_ids == tuple(
        region.region_id for region in regions
    )
    assert analysis.effectful_group_ids == tuple(group.group_id for group in groups)


def test_alias_effect_in_loop_is_effectful_control_flow():
    source = """\
alias = close_gripper
for _ in range(2):
    alias()
"""

    violations = _analyze(source)

    assert any(
        violation.code == "effectful_control_flow"
        and violation.side_effect_calls == ("close_gripper",)
        for violation in violations
    )


def test_rebinding_alias_to_pure_callable_uses_current_binding():
    source = """\
alias = close_gripper
alias = len
alias([])
"""

    assert _analyze(source) == []


def test_uncertain_callable_binding_fails_closed():
    source = """\
alias = close_gripper if should_close else len
alias()
"""
    regions = segment_python_code(source)
    groups = segment_python_code_groups(
        source,
        regions,
        side_effect_calls=SIDE_EFFECTS,
    )

    analysis = analyze_capsule_program_contract_details(
        source,
        regions,
        groups,
        side_effect_calls=SIDE_EFFECTS,
    )

    assert any(
        violation.code == "dynamic_effect_call"
        for violation in analysis.violations
    )
    assert analysis.effectful_region_ids
    assert analysis.effectful_group_ids


def test_class_body_callable_alias_fails_closed():
    source = """\
class Unsafe:
    alias = close_gripper
    alias()
"""
    regions = segment_python_code(source)
    groups = segment_python_code_groups(
        source,
        regions,
        side_effect_calls=SIDE_EFFECTS,
    )

    analysis = analyze_capsule_program_contract_details(
        source,
        regions,
        groups,
        side_effect_calls=SIDE_EFFECTS,
    )

    assert any(
        violation.code == "dynamic_effect_call"
        for violation in analysis.violations
    )
    assert analysis.effectful_region_ids
    assert analysis.effectful_group_ids


def test_callable_parameter_invocation_fails_closed():
    source = """\
def invoke(fn):
    fn()
invoke(close_gripper)
"""
    regions = segment_python_code(source)
    groups = segment_python_code_groups(
        source,
        regions,
        side_effect_calls=SIDE_EFFECTS,
    )

    analysis = analyze_capsule_program_contract_details(
        source,
        regions,
        groups,
        side_effect_calls=SIDE_EFFECTS,
    )

    assert any(
        violation.code == "dynamic_effect_call"
        for violation in analysis.violations
    )
    assert analysis.effectful_region_ids
    assert analysis.effectful_group_ids


@pytest.mark.parametrize(
    "source",
    [
        "close_gripper.__closure__[0].cell_contents()\n",
        "close_gripper.__self__\n",
        "close_gripper.__globals__\n",
        "close_gripper.__func__\n",
        "close_gripper.__code__\n",
        "close_gripper.__dict__\n",
        "close_gripper.__getattribute__('__closure__')\n",
        "vars(close_gripper)\n",
        "dir(close_gripper)\n",
        "builtins.vars(close_gripper)\n",
        "inspect.getclosurevars(close_gripper)\n",
        "gc.get_referrers(close_gripper)\n",
    ],
)
def test_private_callable_introspection_fails_closed(source):
    regions = segment_python_code(source)
    groups = segment_python_code_groups(
        source,
        regions,
        side_effect_calls=SIDE_EFFECTS,
    )

    analysis = analyze_capsule_program_contract_details(
        source,
        regions,
        groups,
        side_effect_calls=SIDE_EFFECTS,
    )

    assert any(
        violation.code == "forbidden_runtime_access"
        for violation in analysis.violations
    )
    assert analysis.effectful_region_ids == tuple(
        region.region_id for region in regions
    )
    assert analysis.effectful_group_ids == tuple(group.group_id for group in groups)


def test_effectful_class_method_marks_class_definition_and_attribute_call():
    source = """\
class Unsafe:
    def act(self):
        close_gripper()
obj = Unsafe()
obj.act()
"""
    regions = segment_python_code(source)
    groups = segment_python_code_groups(
        source,
        regions,
        max_regions_per_group=1,
        side_effect_calls=SIDE_EFFECTS,
    )

    analysis = analyze_capsule_program_contract_details(
        source,
        regions,
        groups,
        side_effect_calls=SIDE_EFFECTS,
    )

    assert any(
        violation.code == "effectful_helper"
        and violation.helper_name == "Unsafe.act"
        for violation in analysis.violations
    )
    assert analysis.effectful_region_ids == (
        regions[0].region_id,
        regions[-1].region_id,
    )
    assert analysis.effectful_group_ids == (
        groups[0].group_id,
        groups[-1].group_id,
    )


def test_pure_class_method_remains_allowed():
    source = """\
class Safe:
    def size(self, values):
        return len(values)
"""

    assert _analyze(source) == []


def test_ordinary_object_attributes_and_methods_remain_allowed():
    source = "value = obj.shape\nobj.measure()\n"

    assert _analyze(source) == []


@pytest.mark.parametrize(
    "source,expected_effects",
    [
        ("(lambda: close_gripper())()\n", ("close_gripper",)),
        (
            "(lambda: (open_gripper(), close_gripper()))()\n",
            ("open_gripper", "close_gripper"),
        ),
        ("list(map(lambda _: close_gripper(), [1]))\n", ("close_gripper",)),
    ],
)
def test_invoked_or_passed_effectful_lambda_fails_closed(source, expected_effects):
    regions = segment_python_code(source)
    groups = segment_python_code_groups(
        source,
        regions,
        side_effect_calls=SIDE_EFFECTS,
    )

    analysis = analyze_capsule_program_contract_details(
        source,
        regions,
        groups,
        side_effect_calls=SIDE_EFFECTS,
    )

    assert any(
        violation.code == "dynamic_effect_call"
        and violation.side_effect_calls == expected_effects[:2]
        for violation in analysis.violations
    )
    assert analysis.effectful_region_ids
    assert analysis.effectful_group_ids


@pytest.mark.parametrize(
    "source",
    [
        "(alias,) = (close_gripper,)\nalias()\n",
        "[alias] = [close_gripper]\nalias()\n",
        "(alias := close_gripper)()\n",
        "first, *rest = [close_gripper]\nrest[0]()\n",
    ],
)
def test_destructured_named_or_uncertain_alias_fails_closed(source):
    regions = segment_python_code(source)
    groups = segment_python_code_groups(
        source,
        regions,
        side_effect_calls=SIDE_EFFECTS,
    )

    analysis = analyze_capsule_program_contract_details(
        source,
        regions,
        groups,
        side_effect_calls=SIDE_EFFECTS,
    )

    assert analysis.violations
    assert analysis.effectful_region_ids
    assert analysis.effectful_group_ids


def test_unresolved_starred_assignment_containing_effect_callable_is_rejected():
    source = "first, *rest = [close_gripper]\n"
    regions = segment_python_code(source)
    groups = segment_python_code_groups(
        source,
        regions,
        side_effect_calls=SIDE_EFFECTS,
    )

    analysis = analyze_capsule_program_contract_details(
        source,
        regions,
        groups,
        side_effect_calls=SIDE_EFFECTS,
    )

    assert any(
        violation.code == "dynamic_effect_call"
        for violation in analysis.violations
    )
    assert analysis.effectful_region_ids
    assert analysis.effectful_group_ids
