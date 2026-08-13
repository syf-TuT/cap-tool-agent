import pytest

import capx.runtime_control as runtime_control
from capx.runtime_control.contract import (
    ProgramContractAnalysis,
    ProgramContractViolation,
    STRICT_CAPSULE_MAX_STATIC_ITERATIONS,
    STRICT_CAPSULE_SAFE_BUILTINS,
    analyze_capsule_program_contract,
    analyze_capsule_program_contract_details,
    analyze_capsule_strict_subset,
    preflight_capsule_strict_source,
)
from capx.runtime_control.normalizer import segment_python_code_groups
from capx.runtime_control.segmenter import segment_python_code


SIDE_EFFECTS = {"goto_pose", "open_gripper", "close_gripper"}
PUBLIC_API_CALLS = {*SIDE_EFFECTS, "detect_object", "get_ee_pose"}


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


def _strict_analyze(source: str) -> list[ProgramContractViolation]:
    regions = segment_python_code(source)
    groups = segment_python_code_groups(
        source,
        regions,
        side_effect_calls=SIDE_EFFECTS,
    )
    return analyze_capsule_strict_subset(
        source,
        regions,
        groups,
        public_api_calls=PUBLIC_API_CALLS,
        side_effect_calls=SIDE_EFFECTS,
    )


def test_strict_subset_allows_bounded_python_and_direct_capabilities():
    source = """\
def clamp(value, lower=0, upper=1):
    return min(max(value, lower), upper)
def transform(values):
    return values[0] + values[1]
values = [1, 2, 3]
adjusted = [clamp(item + 0.25) for item in (1, 2, 3)]
combined = transform(adjusted)
mapping = {"first": adjusted[0]}
position = pose.position
index = 0
for item in range(3):
    index += item
if any(value > 0 for value in (1, 2, 3)):
    mapping["handled"] = True
object_info = detect_object("bowl")
ee_pose = get_ee_pose()
close_gripper()
"""

    assert _strict_analyze(source) == []


def test_strict_subset_allows_each_safe_builtin_as_a_direct_call():
    source = """\
absolute = abs(-1)
all_true = all([True, True])
any_true = any([False, True])
truth = bool(1)
mapping = dict([("a", 1)])
pairs = list(enumerate([1, 2]))
decimal = float(1)
integer = int(1.2)
size = len(pairs)
listed = list((1, 2))
largest = max(1, 2)
smallest = min(1, 2)
sequence = range(2)
backwards = reversed([1, 2])
rounded = round(1.25, 1)
unique = set([1, 1])
ordered = sorted([2, 1])
text = str(integer)
total = sum([1, 2])
packed = tuple([1, 2])
paired = zip([1], [2])
print(text)
"""

    assert _strict_analyze(source) == []
    assert {
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "float",
        "int",
        "len",
        "list",
        "max",
        "min",
        "print",
        "range",
        "reversed",
        "round",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
    } <= STRICT_CAPSULE_SAFE_BUILTINS


@pytest.mark.parametrize(
    "source",
    [
        "import sys\n",
        "from sys import version\n",
        "class Unsafe:\n    pass\n",
        "value = lambda item: item\n",
        "async def helper():\n    return 1\n",
        "def helper():\n    yield 1\n",
        "def helper():\n    yield from values\n",
        "def helper():\n    global state\n",
        "def helper():\n    nonlocal state\n",
        "with resource:\n    value = 1\n",
    ],
)
def test_strict_subset_rejects_syntax_outside_the_language(source):
    violations = _strict_analyze(source)

    assert violations
    assert {violation.code for violation in violations} == {
        "strict_subset_violation"
    }
    assert all(violation.message for violation in violations)


@pytest.mark.parametrize(
    "source,reason",
    [
        ("@register\ndef helper():\n    return 1\n", "decorator"),
        ("def helper(value=make_default()):\n    return value\n", "default"),
        ("def helper(value: int):\n    return value\n", "annotation"),
        ("def outer():\n    def nested():\n        return 1\n", "nested"),
    ],
)
def test_strict_subset_rejects_unsafe_helper_definitions(source, reason):
    violations = _strict_analyze(source)

    assert any(reason in violation.message.lower() for violation in violations)


def test_strict_subset_allows_literal_helper_defaults():
    source = """\
def configure(count=2, labels=("a", "b"), options=None):
    return count + len(labels) if options is None else 0
result = configure()
"""

    assert _strict_analyze(source) == []


def test_strict_subset_allows_direct_safe_exception_construction():
    source = 'raise ValueError("invalid target")\n'

    assert _strict_analyze(source) == []


@pytest.mark.parametrize(
    "source",
    [
        "obj.measure()\n",
        "callbacks[0]()\n",
        "(lambda: 1)()\n",
        "unknown_call()\n",
        "np.array([1, 2])\n",
        "def invoke(fn):\n    return fn()\ninvoke(callback)\n",
    ],
)
def test_strict_subset_rejects_non_direct_or_unknown_calls(source):
    violations = _strict_analyze(source)

    assert any(
        violation.code == "strict_subset_violation"
        and "call" in violation.message.lower()
        for violation in violations
    )


@pytest.mark.parametrize(
    "source",
    [
        "alias = close_gripper\nalias()\n",
        "alias = close_gripper if condition else len\nalias()\n",
        "(alias,) = (close_gripper,)\nalias()\n",
        "[alias] = [close_gripper]\nalias()\n",
        "(alias := close_gripper)()\n",
        "close_gripper = callback\nclose_gripper()\n",
        "def helper():\n    return 1\nhelper = callback\nhelper()\n",
    ],
)
def test_strict_subset_rejects_callable_aliases_and_rebinding(source):
    violations = _strict_analyze(source)

    assert any(
        violation.code == "strict_subset_violation"
        and any(
            reason in violation.message.lower()
            for reason in ("callable", "call", "rebind")
        )
        for violation in violations
    )


@pytest.mark.parametrize(
    "source",
    [
        "value = env\n",
        "APIS = {}\n",
        "value = __builtins__\n",
        "value = frame\n",
        "value = pose._position\n",
        "pose.position = target\n",
        "_hidden = 1\n",
        "sys._getframe()\n",
    ],
)
def test_strict_subset_rejects_private_and_sensitive_runtime_access(source):
    violations = _strict_analyze(source)

    assert violations
    assert all(
        violation.code == "strict_subset_violation"
        for violation in violations
    )


@pytest.mark.parametrize(
    "forbidden_name",
    [
        "eval",
        "exec",
        "globals",
        "locals",
        "getattr",
        "setattr",
        "vars",
        "dir",
        "open",
        "compile",
        "__import__",
        "breakpoint",
        "input",
        "help",
        "type",
        "object",
        "super",
    ],
)
def test_strict_subset_rejects_forbidden_builtin_calls(forbidden_name):
    violations = _strict_analyze(f"{forbidden_name}()\n")

    assert len(violations) == 1
    assert forbidden_name in violations[0].message


def test_strict_subset_violations_are_deterministic_and_bind_source_units():
    source = "alias = close_gripper\nalias()\n"

    first = _strict_analyze(source)
    second = _strict_analyze(source)

    assert first == second
    assert all(violation.region_ids for violation in first)
    assert all(violation.group_ids for violation in first)
    assert [
        (violation.start_line, violation.end_line, violation.message)
        for violation in first
    ] == sorted(
        (violation.start_line, violation.end_line, violation.message)
        for violation in first
    )


def test_strict_subset_api_is_exported_from_runtime_control_package():
    assert (
        runtime_control.analyze_capsule_strict_subset
        is analyze_capsule_strict_subset
    )
    assert (
        runtime_control.STRICT_CAPSULE_SAFE_BUILTINS
        is STRICT_CAPSULE_SAFE_BUILTINS
    )


@pytest.mark.parametrize(
    "source",
    [
        "if condition:\n    def helper():\n        return 1\n",
        "for item in values:\n    def helper():\n        return item\n",
        "try:\n    def helper():\n        return 1\nexcept ValueError:\n    pass\n",
        "with resource:\n    def helper():\n        return 1\n",
    ],
)
def test_strict_subset_rejects_helpers_not_directly_in_module_body(source):
    violations = _strict_analyze(source)

    assert any(
        "direct module" in violation.message.lower()
        for violation in violations
    )


@pytest.mark.parametrize(
    "use_helper",
    [
        "ordered = sorted([2, 1], key=helper)\n",
        "alias = helper\n",
    ],
)
def test_strict_subset_rejects_top_level_helper_as_callable_value(use_helper):
    source = "def helper(value):\n    return value\n" + use_helper

    violations = _strict_analyze(source)

    assert any(
        "callable 'helper'" in violation.message.lower()
        for violation in violations
    )


@pytest.mark.parametrize(
    "source",
    [
        "def unsafe():\n    close_gripper()\nunsafe()\n",
        "def unsafe():\n    close_gripper()\nvalue = 1\n",
        (
            "def unsafe():\n"
            "    close_gripper()\n"
            "def outer():\n"
            "    unsafe()\n"
            "outer()\n"
        ),
    ],
)
def test_strict_subset_rejects_direct_and_transitive_effectful_helpers(source):
    violations = _strict_analyze(source)

    assert any(
        "unsafe helper" in violation.message.lower()
        or "side effect" in violation.message.lower()
        for violation in violations
    )


@pytest.mark.parametrize(
    "source",
    [
        "def recurse():\n    recurse()\nvalue = 1\n",
        (
            "def first():\n"
            "    second()\n"
            "def second():\n"
            "    first()\n"
            "value = 1\n"
        ),
        (
            "def unsafe_recurse():\n"
            "    close_gripper()\n"
            "    unsafe_recurse()\n"
            "value = 1\n"
        ),
    ],
)
def test_strict_subset_rejects_recursive_helpers_even_when_uncalled(source):
    violations = _strict_analyze(source)

    assert any(
        "recursive helper" in violation.message.lower()
        for violation in violations
    )


def test_strict_subset_allows_proven_pure_helper_chains_and_public_queries():
    source = """\
def normalize(value):
    return round(abs(value), 2)
def locate(name):
    return detect_object(name)
def prepare(value, name):
    return (normalize(value), locate(name))
result = prepare(-1.25, "bowl")
"""

    assert _strict_analyze(source) == []


@pytest.mark.parametrize("callable_name", ["sorted", "min", "max"])
def test_strict_subset_rejects_observation_method_as_key_callback(callable_name):
    source = f"result = {callable_name}(values, key=observation.tofile)\n"

    violations = _strict_analyze(source)

    assert any(
        "callback" in violation.message.lower()
        for violation in violations
    )


def test_strict_subset_rejects_dynamic_keyword_callback_bypass():
    source = 'result = sorted(values, **{"key": observation.tofile})\n'

    violations = _strict_analyze(source)

    assert any(
        "dynamic keyword" in violation.message.lower()
        for violation in violations
    )


def test_strict_subset_allows_numeric_ordering_without_callbacks():
    source = """\
ordered = sorted([3, 1, 2])
smallest = min(3, 1, 2)
largest = max([3, 1, 2])
unchanged = sorted([3, 1, 2], key=None)
"""

    assert _strict_analyze(source) == []


def test_strict_subset_allows_statically_bounded_loops_and_comprehensions():
    source = """\
total = 0
for item in [1, 2, 3]:
    total += item
for index, item in enumerate((4, 5)):
    total += index + item
for left, right in zip(range(3), reversed([1, 2, 3])):
    total += left + right
for countdown in range(3, -1, -1):
    total += countdown
grid = [left + right for left in range(90) for right in range(90)]
"""

    assert _strict_analyze(source) == []


@pytest.mark.parametrize(
    "source",
    [
        "for item in observations:\n    value = item\n",
        "for item in range(limit):\n    value = item\n",
        "for item in range(10 ** 100):\n    value = item\n",
        "for item in range(10001):\n    value = item\n",
        "for item in range(0, 2, 0):\n    value = item\n",
        "values = [item for item in observations]\n",
        "while False:\n    value = 1\n",
    ],
)
def test_strict_subset_rejects_unbounded_control_flow(source):
    violations = _strict_analyze(source)

    assert any(
        "bounded control flow" in violation.message.lower()
        for violation in violations
    )


@pytest.mark.parametrize(
    "source",
    [
        "for outer in range(101):\n    for inner in range(100):\n        value = inner\n",
        "grid = [(x, y) for x in range(101) for y in range(100)]\n",
    ],
)
def test_strict_subset_rejects_nested_iteration_products_over_budget(source):
    violations = _strict_analyze(source)

    assert any(
        "iteration budget" in violation.message.lower()
        for violation in violations
    )


def test_strict_subset_resets_iteration_budget_for_sequential_loops():
    source = """\
for first in range(5000):
    value = first
for second in range(5000):
    value = second
"""

    assert _strict_analyze(source) == []


def test_strict_subset_rejects_excessive_helper_count_without_recursing():
    source = "\n".join(
        f"def helper_{index}():\n    return {index}"
        for index in range(1200)
    )

    violations = _strict_analyze(source)

    assert len(violations) == 1
    assert violations[0].code == "strict_subset_violation"
    assert "helper limit" in violations[0].message.lower()
    assert violations[0].start_line == 1
    assert violations[0].end_line == 2400


def test_strict_subset_helper_limit_violation_binds_all_source_units():
    source = "\n".join(
        f"def helper_{index}():\n    return {index}"
        for index in range(257)
    )
    regions = segment_python_code(source)
    groups = segment_python_code_groups(
        source,
        regions,
        side_effect_calls=SIDE_EFFECTS,
    )

    violations = analyze_capsule_strict_subset(
        source,
        regions,
        groups,
        public_api_calls=PUBLIC_API_CALLS,
        side_effect_calls=SIDE_EFFECTS,
    )

    assert len(violations) == 1
    assert violations[0].region_ids == tuple(
        region.region_id for region in regions
    )
    assert violations[0].group_ids == tuple(
        group.group_id for group in groups
    )


def test_strict_subset_allows_pure_helper_chain_at_complexity_limit():
    definitions = ["def helper_0(value):\n    return value"]
    for index in range(1, 256):
        definitions.append(
            f"def helper_{index}(value):\n"
            f"    return helper_{index - 1}(value)"
        )
    source = "\n".join([*definitions, "result = helper_255(1)", ""])

    assert _strict_analyze(source) == []


def test_strict_subset_reports_async_comprehension_at_its_own_line():
    source = "async def collect():\n    return [item async for item in stream]\n"

    violations = _strict_analyze(source)

    violation = next(
        violation
        for violation in violations
        if "asynchronous comprehensions" in violation.message.lower()
    )
    assert (violation.start_line, violation.end_line) == (2, 2)


@pytest.mark.parametrize(
    "source",
    [
        "values = list(range(1000000000))\n",
        "values = set(range(10001))\n",
        "values = tuple(range(limit))\n",
        "values = dict(observations)\n",
        "values = sorted(range(10001))\n",
        "value = sum(range(10 ** 100))\n",
        "value = min(observations)\n",
        "value = max(range(10001))\n",
        "value = all(observations)\n",
        "value = any(observations)\n",
        "value = sum(observations, 10)\n",
    ],
)
def test_strict_subset_rejects_unbounded_iterable_consumers(source):
    violations = _strict_analyze(source)

    assert any(
        "iterable consumer" in violation.message.lower()
        for violation in violations
    )


def test_strict_subset_allows_finite_consumers_and_scalar_min_max():
    source = """\
listed = list(range(10))
unique = set([1, 1, 2])
packed = tuple((1, 2))
mapping = dict([("a", 1), ("b", 2)])
mapping_with_keywords = dict(a=1, b=2)
empty_mapping = dict()
ordered = sorted(range(10))
total = sum(range(10))
smallest = min(3, 2, 1)
largest = max(1, 2, 3)
all_true = all((True, True))
any_true = any([False, True])
"""

    assert _strict_analyze(source) == []


def test_strict_subset_allows_one_consumer_at_exact_total_budget():
    source = "ordered = sorted(range(10000))\n"

    assert _strict_analyze(source) == []


@pytest.mark.parametrize(
    "source",
    [
        (
            "def work():\n"
            "    for first in range(10000):\n"
            "        value = first\n"
            "    for second in range(10000):\n"
            "        value = second\n"
            "work()\n"
        ),
        (
            "def work():\n"
            "    for item in range(10000):\n"
            "        value = item\n"
            "work()\n"
            "for item in range(10000):\n"
            "    value = item\n"
        ),
    ],
)
def test_strict_subset_rejects_sequential_work_over_total_budget(source):
    violations = _strict_analyze(source)

    assert any(
        "static compute budget" in violation.message.lower()
        for violation in violations
    )


def test_strict_subset_rejects_exponentially_reused_helper_dag():
    definitions = ["def helper_0():\n    return 1"]
    for index in range(1, 40):
        definitions.append(
            f"def helper_{index}():\n"
            f"    return helper_{index - 1}() + helper_{index - 1}()"
        )
    source = "\n".join([*definitions, "result = helper_39()", ""])

    violations = _strict_analyze(source)

    assert any(
        "static compute budget" in violation.message.lower()
        for violation in violations
    )


def test_strict_subset_allows_small_total_helper_and_loop_cost():
    source = """\
def normalize(value):
    return abs(value)
def work():
    total = 0
    for item in range(20):
        total += normalize(item)
    return total
first = work()
second = work()
"""

    assert _strict_analyze(source) == []
    assert STRICT_CAPSULE_MAX_STATIC_ITERATIONS == 10_000


def test_strict_subset_reports_over_budget_helper_even_when_uncalled():
    source = """\
def expensive():
    first = list(range(6000))
    second = list(range(6000))
value = 1
"""

    violations = _strict_analyze(source)

    assert any(
        violation.helper_name == "expensive"
        and "static compute budget" in violation.message.lower()
        for violation in violations
    )


def test_strict_subset_preflight_rejects_helper_flood_before_segmentation():
    source = "\n".join(
        f"def helper_{index}():\n    return {index}"
        for index in range(1200)
    )

    violations = preflight_capsule_strict_source(source)

    assert len(violations) == 1
    assert violations[0].code == "strict_subset_violation"
    assert "helper limit" in violations[0].message.lower()
    assert violations[0].region_ids == ()
    assert violations[0].group_ids == ()


def test_strict_subset_preflight_counts_conditional_helper_flood_iteratively():
    source = "\n".join(
        f"if condition_{index}:\n    def helper_{index}():\n        return {index}"
        for index in range(300)
    )

    violations = preflight_capsule_strict_source(source)

    assert len(violations) == 1
    assert "helper limit" in violations[0].message.lower()


def test_strict_subset_preflight_api_is_exported():
    assert (
        runtime_control.preflight_capsule_strict_source
        is preflight_capsule_strict_source
    )
    assert (
        runtime_control.STRICT_CAPSULE_MAX_STATIC_ITERATIONS
        == STRICT_CAPSULE_MAX_STATIC_ITERATIONS
    )


def test_strict_subset_counts_for_iterable_construction_cost():
    source = "for item in list(range(10000)):\n    value = item\n"

    violations = _strict_analyze(source)

    assert any(
        "static compute budget" in violation.message.lower()
        for violation in violations
    )


def test_strict_subset_counts_nested_iterable_construction_per_prefix():
    source = (
        "pairs = [(outer, inner) "
        "for outer in range(100) "
        "for inner in list(range(100))]\n"
    )

    violations = _strict_analyze(source)

    assert any(
        "static compute budget" in violation.message.lower()
        for violation in violations
    )


def test_strict_subset_counts_outer_comprehension_iterable_construction():
    source = "copied = [item for item in [value for value in range(10000)]]\n"

    violations = _strict_analyze(source)

    assert any(
        "static compute budget" in violation.message.lower()
        for violation in violations
    )


def test_strict_subset_multiplies_exact_budget_helper_inside_comprehension():
    source = """\
def exact_budget():
    for item in range(10000):
        value = item
values = [exact_budget() for item in range(10000)]
"""

    violations = _strict_analyze(source)

    assert any(
        "module exceeds static compute budget" in violation.message.lower()
        for violation in violations
    )


def test_strict_subset_composes_generator_and_consumer_costs():
    source = "total = sum((item for item in range(6000)))\n"

    violations = _strict_analyze(source)

    assert any(
        "static compute budget" in violation.message.lower()
        for violation in violations
    )


def test_strict_subset_allows_small_nested_iterable_construction():
    source = """\
pairs = [
    (outer, inner)
    for outer in range(10)
    for inner in list(range(10))
]
total = sum((item for item in range(100)))
"""

    assert _strict_analyze(source) == []


@pytest.mark.parametrize(
    "source",
    [
        "try:\n    value = 1\nexcept ValueError:\n    value = 2\n",
        "assert condition\n",
        "with resource:\n    value = 1\n",
        "match value:\n    case 1:\n        result = True\n",
        "del values[0]\n",
    ],
)
def test_strict_subset_rejects_unmodeled_statements(source):
    violations = _strict_analyze(source)

    assert any(
        "not allowed" in violation.message.lower()
        for violation in violations
    )


@pytest.mark.parametrize(
    "source",
    [
        "raise\n",
        "raise ValueError\n",
        'raise ValueError("bad") from cause\n',
        'raise Exception("bad")\n',
    ],
)
def test_strict_subset_rejects_unsafe_raise_forms(source):
    violations = _strict_analyze(source)

    assert any(
        "raise" in violation.message.lower()
        for violation in violations
    )


@pytest.mark.parametrize(
    "assignment",
    [
        "values[exact_budget()] = 1",
        "first, values[exact_budget()] = (1, 2)",
        "values[exact_budget()]: int = 1",
        "values[exact_budget()] += 1",
    ],
)
def test_strict_subset_counts_assignment_target_evaluation(assignment):
    source = (
        "def exact_budget():\n"
        "    for item in range(10000):\n"
        "        value = item\n"
        f"{assignment}\n"
        "marker = 1\n"
    )

    violations = _strict_analyze(source)

    assert any(
        "module exceeds static compute budget" in violation.message.lower()
        for violation in violations
    )


def test_strict_subset_allows_simple_subscript_and_annotated_assignments():
    source = """\
values = [1, 2]
values[0] = 3
first, values[1] = (4, 5)
values[0]: int = 6
values[1] += 1
(count := 2)
"""

    assert _strict_analyze(source) == []


@pytest.mark.parametrize("source", ["break\n", "continue\n", "return 1\n"])
def test_strict_subset_rejects_loop_or_function_control_outside_scope(source):
    violations = _strict_analyze(source)

    assert any(
        "outside" in violation.message.lower()
        for violation in violations
    )


def test_strict_subset_counts_for_target_cost_per_iteration():
    source = """\
def exact_budget():
    for item in range(10000):
        value = item
values = [0]
for values[exact_budget()] in range(1):
    pass
"""

    violations = _strict_analyze(source)

    assert any(
        "module exceeds static compute budget" in violation.message.lower()
        for violation in violations
    )


def test_strict_subset_counts_comprehension_target_cost_per_iteration():
    source = """\
def exact_budget():
    for item in range(10000):
        value = item
values = [0]
result = [values[0] for values[exact_budget()] in range(1)]
"""

    violations = _strict_analyze(source)

    assert any(
        "module exceeds static compute budget" in violation.message.lower()
        for violation in violations
    )


def test_strict_subset_allows_zero_cost_name_loop_targets():
    source = """\
total = 0
for item in range(9900):
    total = item
for first, second in [(1, 2), (3, 4)]:
    total = first + second
"""

    assert _strict_analyze(source) == []


def test_strict_subset_allows_small_constant_subscript_loop_target():
    source = """\
values = [0]
for values[0] in range(10):
    pass
result = [values[0] for values[0] in range(10)]
"""

    assert _strict_analyze(source) == []


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


@pytest.mark.parametrize(
    "source",
    [
        'runner = eval\nrunner("close_gripper()")\n',
        'runner = exec\nrunner("close_gripper()")\n',
        'runner = globals\nrunner()["close_gripper"]()\n',
        "runner = locals\nrunner()\n",
        "runner = vars\nrunner(close_gripper)\n",
        "runner = dir\nrunner(close_gripper)\n",
        'runner = __import__\nrunner("inspect")\n',
        'r = eval\ns = r\ns("close_gripper()")\n',
        'ga = getattr\nga(close_gripper, "__self__")\n',
        '(runner := eval)("close_gripper()")\n',
        '(runner,) = (exec,)\nrunner("close_gripper()")\n',
    ],
)
def test_forbidden_runtime_callable_aliases_propagate(source):
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
    assert analysis.effectful_region_ids
    assert analysis.effectful_group_ids


def test_forbidden_callable_alias_can_be_rebound_to_pure_callable():
    source = "runner = eval\nrunner = len\nrunner([])\n"

    assert _analyze(source) == []


def test_uncalled_forbidden_callable_alias_is_not_executable():
    source = "runner = eval\nvalue = 1\n"

    assert _analyze(source) == []
