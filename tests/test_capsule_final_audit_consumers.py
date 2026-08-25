from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.capsule_rl import analyze_artifacts, materialize_resolved_dataset
from scripts.capsule_rl.common import ConfigValidationError
from test_capsule_scripts import (
    _materialization_gate7_audit,
    _resolved_task,
    _server_config,
)


def test_materializer_rejects_final_audit_that_does_not_match_reproduction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _server_config(tmp_path)
    audit_path = _materialization_gate7_audit(config_path)
    reproduced = json.loads(audit_path.read_text(encoding="utf-8"))
    reproduced["minimum_mem_available_mib"] += 1
    calls: list[Path] = []

    def forged_reproduction(directory: str | Path, **_kwargs: object) -> dict[str, object]:
        calls.append(Path(directory).resolve())
        return deepcopy(reproduced)

    monkeypatch.setattr(
        analyze_artifacts,
        "finalize_runtime_audit",
        forged_reproduction,
    )

    with pytest.raises(ConfigValidationError, match="do not equal.*reproduced"):
        materialize_resolved_dataset.materialize(
            config_path=config_path,
            gate7_audit_path=audit_path,
            output_dir=tmp_path / "reproduction-mismatch",
            validate_only=True,
        )

    assert calls == [audit_path.parent.resolve()]


def test_materializer_guard_rejects_candidate_a_to_b_to_a_during_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _server_config(tmp_path)
    audit_path = _materialization_gate7_audit(config_path)
    audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
    candidate_path = audit_path.parent / "gate07_audit.candidate.json"
    trusted_candidate = candidate_path.read_bytes()
    destination = tmp_path / "candidate-race-output"
    monkeypatch.setattr(
        analyze_artifacts,
        "finalize_runtime_audit",
        lambda *_args, **_kwargs: deepcopy(audit_payload),
    )

    def racing_resolver(_config: object):
        candidate_path.write_bytes(b'{"attacker":true}\n')
        candidate_path.write_bytes(trusted_candidate)
        return (_resolved_task(),)

    with pytest.raises(ConfigValidationError, match="guarded runtime input changed"):
        materialize_resolved_dataset.materialize(
            config_path=config_path,
            gate7_audit_path=audit_path,
            output_dir=destination,
            validate_only=False,
            task_resolver=racing_resolver,
        )

    assert not destination.exists()


def test_materializer_guard_rejects_resolved_profile_a_to_b_to_a_during_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _server_config(tmp_path)
    audit_path = _materialization_gate7_audit(config_path)
    audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
    profile_path = audit_path.parent / "resolved" / "verl.yaml"
    trusted_profile = profile_path.read_bytes()
    destination = tmp_path / "resolved-profile-race-output"
    monkeypatch.setattr(
        analyze_artifacts,
        "finalize_runtime_audit",
        lambda *_args, **_kwargs: deepcopy(audit_payload),
    )

    def racing_resolver(_config: object):
        profile_path.write_bytes(b"capsule_runtime:\n  oom_profile: attacker\n")
        profile_path.write_bytes(trusted_profile)
        return (_resolved_task(),)

    with pytest.raises(ConfigValidationError, match="guarded runtime input changed"):
        materialize_resolved_dataset.materialize(
            config_path=config_path,
            gate7_audit_path=audit_path,
            output_dir=destination,
            validate_only=False,
            task_resolver=racing_resolver,
        )

    assert not destination.exists()
