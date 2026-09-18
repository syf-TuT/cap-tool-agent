from __future__ import annotations

import json
from pathlib import Path

import pytest

from capx.rl.capsule.checkpoint import AtomicCheckpointClaim, checkpoint_tree_sha256


def test_checkpoint_claim_stages_and_publishes_manifest_without_overwrite(
    tmp_path: Path,
) -> None:
    claim_root = tmp_path / "run-01"
    checkpoint = claim_root / "global_step_1" / "actor"

    with AtomicCheckpointClaim(checkpoint, claim_root=claim_root) as claim:
        evidence = claim.publish(
            lambda staging: (
                staging.mkdir(parents=True),
                (staging / "state.bin").write_bytes(b"state"),
            ),
            optimizer_step_before=0,
            optimizer_step_after=1,
        )

    assert evidence.path == checkpoint.resolve()
    assert evidence.file_count == 1
    assert evidence.sha256 == checkpoint_tree_sha256(checkpoint)
    manifest = json.loads(evidence.manifest_path.read_text(encoding="utf-8"))
    assert manifest["optimizer_step_delta"] == 1
    assert manifest["checkpoint_sha256"] == evidence.sha256
    assert not list(claim_root.glob(".staging-*"))
    assert not (claim_root / ".capsule_checkpoint_claim").exists()

    with pytest.raises(FileExistsError):
        with AtomicCheckpointClaim(checkpoint, claim_root=claim_root):
            raise AssertionError("an existing claim root must fail first")


def test_failed_checkpoint_save_releases_owned_claim_for_safe_retry(tmp_path: Path) -> None:
    claim_root = tmp_path / "run-01"
    checkpoint = claim_root / "global_step_1" / "actor"

    with pytest.raises(RuntimeError, match="save failed"):
        with AtomicCheckpointClaim(checkpoint, claim_root=claim_root) as claim:
            claim.publish(
                lambda _staging: (_ for _ in ()).throw(RuntimeError("save failed")),
                optimizer_step_before=0,
                optimizer_step_after=1,
            )

    assert not claim_root.exists()
    with AtomicCheckpointClaim(checkpoint, claim_root=claim_root):
        pass
    assert not claim_root.exists()


def test_checkpoint_tree_rejects_a_symlink_used_as_the_root(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "state.bin").write_bytes(b"checkpoint")
    checkpoint_link = tmp_path / "checkpoint-link"
    checkpoint_link.symlink_to(checkpoint, target_is_directory=True)

    with pytest.raises(ValueError, match="checkpoint path must not be a symlink"):
        checkpoint_tree_sha256(checkpoint_link)
