from pathlib import Path

import capx.envs.trial as trial_module
import capx.runtime_control.prompts as prompt_module


REMOVED_RUNTIME_SYMBOLS = (
    "_run_capsule_auto_forward_loop",
    "_coerce_terminal_append_recovery_action",
    "_terminal_python_payload",
    "_validate_recovery_action",
    "_group_index_by_id",
    "_first_group_index_starting_after_line",
    "_group_index_for_region",
    "_insert_recovery_source_after_line",
    "build_capsule_recovery_prompt",
    "build_capsule_terminal_recovery_prompt",
    "append_recovery_insert_after_line",
)


def test_auto_forward_runtime_is_removed():
    removed_names = REMOVED_RUNTIME_SYMBOLS[:8]

    assert not [name for name in removed_names if hasattr(trial_module, name)]


def test_recovery_only_prompt_builders_are_removed():
    removed_names = (
        "build_capsule_recovery_prompt",
        "build_capsule_terminal_recovery_prompt",
    )

    assert not [name for name in removed_names if hasattr(prompt_module, name)]


def test_removed_runtime_symbols_have_no_active_references():
    project_root = Path(__file__).resolve().parents[1]
    offenders = []
    for relative_root in ("capx", "tests", "env_configs"):
        for path in (project_root / relative_root).rglob("*"):
            if path == Path(__file__).resolve() or path.suffix not in {
                ".py",
                ".yaml",
                ".yml",
            }:
                continue
            text = path.read_text(encoding="utf-8")
            for symbol in REMOVED_RUNTIME_SYMBOLS:
                if symbol in text:
                    offenders.append(f"{path.relative_to(project_root)}: {symbol}")

    assert not offenders
