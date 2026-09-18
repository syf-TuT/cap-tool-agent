from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from capx.rl.capsule.compat import (
    PINNED_VERL_SHA,
    VeRLCompatibilityError,
    check_verl_compatibility,
)


def _write_pinned_surface(root: Path, *, include_slot: bool = True) -> None:
    core = root / "verl" / "trainer" / "ppo" / "core_algos.py"
    actor = root / "verl" / "workers" / "actor" / "dp_actor.py"
    role = root / "verl" / "workers" / "roles" / "actor.py"
    fsdp = root / "verl" / "workers" / "fsdp_workers.py"
    for path in (core, actor, role, fsdp):
        path.parent.mkdir(parents=True, exist_ok=True)
    core.write_text(
        "POLICY_LOSS_REGISTRY = {}\n"
        "def register_policy_loss(name):\n    return lambda fn: fn\n"
        "def compute_policy_loss_vanilla(old_log_prob, log_prob, advantages, "
        "response_mask, loss_agg_mode='token-mean', config=None, "
        "rollout_is_weights=None):\n    return None\n",
        encoding="utf-8",
    )
    slot_source = (
        "rollout_is_weights = data.batch.get('rollout_is_weights')\n"
        if include_slot
        else "return None\n"
    )
    actor.write_text(
        "class DataParallelPPOActor:\n"
        "    def compute_log_prob(self, data, calculate_entropy=False):\n        return None\n"
        "    def update_policy(self, data):\n"
        f"        {slot_source}",
        encoding="utf-8",
    )
    role.write_text(
        "class ActorRolloutRefWorker:\n"
        "    def compute_log_prob(self, data):\n        return None\n"
        "    def update_actor(self, data):\n        return None\n",
        encoding="utf-8",
    )
    fsdp.write_text(
        "class ActorRolloutRefWorker:\n"
        "    def compute_ref_log_prob(self, data):\n        return None\n",
        encoding="utf-8",
    )


def _mock_git(
    monkeypatch: pytest.MonkeyPatch,
    sha: str = PINNED_VERL_SHA,
    *,
    status: str = "",
) -> list[object]:
    calls: list[object] = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if command[-2:] == ["rev-parse", "HEAD"]:
            stdout = f"{sha}\n"
        elif command[-2:] == ["rev-parse", "--show-toplevel"]:
            stdout = f"{command[2]}\n"
        else:
            stdout = status
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr("capx.rl.capsule.compat.subprocess.run", fake_run)
    return calls


def test_compatibility_check_is_read_only_and_matches_pinned_surface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CAPX_GIT_ENV_SENTINEL", "preserved")
    _write_pinned_surface(tmp_path)
    calls = _mock_git(monkeypatch)

    report = check_verl_compatibility(tmp_path)

    assert report.compatible is True
    assert report.actual_sha == PINNED_VERL_SHA
    assert report.rollout_is_slot == "rollout_is_weights"
    command, kwargs = calls[0]
    assert command == ["git", "-C", str(tmp_path.resolve()), "rev-parse", "HEAD"]
    assert kwargs.get("shell", False) is False
    assert kwargs["capture_output"] is True
    for _command, call_kwargs in calls:
        assert call_kwargs["env"]["GIT_OPTIONAL_LOCKS"] == "0"
        assert call_kwargs["env"]["CAPX_GIT_ENV_SENTINEL"] == "preserved"


def test_compatibility_rejects_tracked_changes_in_pinned_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_pinned_surface(tmp_path)
    _mock_git(monkeypatch, status=" M verl/workers/actor/dp_actor.py\n")

    with pytest.raises(VeRLCompatibilityError) as caught:
        check_verl_compatibility(tmp_path)

    assert caught.value.code == "dirty_source"


def test_compatibility_rejects_untracked_files_in_pinned_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_pinned_surface(tmp_path)
    _mock_git(monkeypatch, status="?? untracked_patch.py\n")

    with pytest.raises(VeRLCompatibilityError) as caught:
        check_verl_compatibility(tmp_path)

    assert caught.value.code == "dirty_source"


def test_uninitialized_source_has_a_typed_error(tmp_path: Path) -> None:
    missing = tmp_path / "not-initialized"
    with pytest.raises(VeRLCompatibilityError) as caught:
        check_verl_compatibility(missing)
    assert caught.value.code == "uninitialized_source"


def test_sha_mismatch_is_rejected_before_source_is_imported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_pinned_surface(tmp_path)
    _mock_git(monkeypatch, "0" * 40)

    with pytest.raises(VeRLCompatibilityError) as caught:
        check_verl_compatibility(tmp_path)

    assert caught.value.code == "sha_mismatch"
    assert PINNED_VERL_SHA in str(caught.value)


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("missing_slot", "missing_rollout_is_slot"),
        ("bad_signature", "signature_mismatch"),
    ],
)
def test_required_slot_and_signature_are_checked_without_importing_verl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected_code: str,
) -> None:
    _write_pinned_surface(tmp_path, include_slot=mutation != "missing_slot")
    if mutation == "bad_signature":
        core = tmp_path / "verl" / "trainer" / "ppo" / "core_algos.py"
        core.write_text(
            "POLICY_LOSS_REGISTRY = {}\n"
            "def register_policy_loss(name):\n    return lambda fn: fn\n"
            "def compute_policy_loss_vanilla(old_log_prob):\n    return None\n",
            encoding="utf-8",
        )
    _mock_git(monkeypatch)

    with pytest.raises(VeRLCompatibilityError) as caught:
        check_verl_compatibility(tmp_path)

    assert caught.value.code == expected_code
