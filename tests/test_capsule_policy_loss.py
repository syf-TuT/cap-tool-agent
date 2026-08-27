from __future__ import annotations

import math
from collections.abc import Iterator, Mapping

import pytest

torch = pytest.importorskip("torch")

from capx.rl.capsule.policy_loss import (  # noqa: E402
    CAPSULE_CRITIQUE_LOSS_MODE,
    capsule_critique_policy_loss,
    map_guided_token_mask_to_verl_slot,
    validate_capsule_training_config,
    verl_capsule_critique_policy_loss,
)


def vanilla_reference(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    *,
    clip_ratio=0.2,
    clip_ratio_low=None,
    clip_ratio_high=None,
    clip_ratio_c=3.0,
):
    clip_ratio_low = clip_ratio if clip_ratio_low is None else clip_ratio_low
    clip_ratio_high = clip_ratio if clip_ratio_high is None else clip_ratio_high
    negative_approx_kl = torch.clamp(log_prob - old_log_prob, -20.0, 20.0)
    ratio = torch.exp(negative_approx_kl)
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - clip_ratio_low, 1 + clip_ratio_high
    )
    clipped = torch.maximum(pg_losses1, pg_losses2)
    dual_clipped = torch.minimum(-advantages * clip_ratio_c, clipped)
    losses = torch.where(advantages < 0, dual_clipped, clipped)
    mask = response_mask.to(losses.dtype)
    return (losses * mask).sum() / mask.sum()


def test_guided_loss_matches_probability_shaping_formula() -> None:
    old = torch.tensor([[123.0]])
    logp = torch.tensor([[math.log(0.3)]], requires_grad=True)
    advantage = torch.tensor([[2.0]])
    response_mask = torch.tensor([[True]])
    guided_mask = torch.tensor([[True]])

    loss, clipfrac, ppo_kl, lower_clipfrac = capsule_critique_policy_loss(
        old,
        logp,
        advantage,
        response_mask,
        guided_mask,
        capsule_gamma=0.1,
    )

    assert loss.item() == pytest.approx(-2.0 * 0.3 / (0.3 + 0.1))
    assert clipfrac.item() == 0.0
    assert ppo_kl.item() == 0.0
    assert lower_clipfrac.item() == 0.0


def test_guided_low_and_high_log_probability_have_finite_gradients() -> None:
    logp = torch.tensor([[-100.0, 100.0]], requires_grad=True)
    loss, *_ = capsule_critique_policy_loss(
        torch.zeros_like(logp),
        logp,
        torch.ones_like(logp),
        torch.tensor([[True, True]]),
        torch.tensor([[True, True]]),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logp.grad).all()


def test_padding_has_zero_gradient() -> None:
    logp = torch.tensor([[-1.0, -1.0]], requires_grad=True)
    loss, *_ = capsule_critique_policy_loss(
        torch.zeros_like(logp),
        logp,
        torch.ones_like(logp),
        torch.tensor([[True, False]]),
        torch.tensor([[True, False]]),
    )
    loss.backward()
    assert logp.grad[0, 0] != 0
    assert logp.grad[0, 1] == 0


def test_guided_tokens_are_independent_of_old_log_probability() -> None:
    args = (
        torch.tensor([[-0.4, -0.7]]),
        torch.tensor([[1.0, -1.0]]),
        torch.tensor([[True, True]]),
        torch.tensor([[True, True]]),
    )
    first = capsule_critique_policy_loss(torch.tensor([[0.0, 0.0]]), *args)
    second = capsule_critique_policy_loss(torch.tensor([[1e9, -1e9]]), *args)
    for left, right in zip(first, second, strict=True):
        torch.testing.assert_close(left, right)


def test_all_base_matches_pinned_verl_vanilla_value_and_gradient() -> None:
    old = torch.tensor([[-0.1, -0.8, -1.2], [-0.5, -0.2, -2.0]])
    initial_logp = torch.tensor([[-0.2, -0.1, -1.5], [-1.0, 0.3, -1.7]])
    advantages = torch.tensor([[1.5, -0.5, 2.0], [-1.0, 0.7, -2.5]])
    response_mask = torch.tensor([[True, True, False], [True, True, True]])
    guided_mask = torch.zeros_like(response_mask)

    actual_logp = initial_logp.clone().requires_grad_()
    actual, *_ = capsule_critique_policy_loss(
        old,
        actual_logp,
        advantages,
        response_mask,
        guided_mask,
        clip_ratio=0.2,
        clip_ratio_low=0.1,
        clip_ratio_high=0.3,
        clip_ratio_c=3.0,
    )
    actual.backward()

    expected_logp = initial_logp.clone().requires_grad_()
    expected = vanilla_reference(
        old,
        expected_logp,
        advantages,
        response_mask,
        clip_ratio=0.2,
        clip_ratio_low=0.1,
        clip_ratio_high=0.3,
        clip_ratio_c=3.0,
    )
    expected.backward()

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_logp.grad, expected_logp.grad)


@pytest.mark.parametrize(
    ("response_mask", "guided_mask", "message"),
    [
        (torch.ones(1, 2), torch.zeros(1, 2, dtype=torch.bool), "response_mask"),
        (torch.ones(1, 2, dtype=torch.bool), torch.zeros(1, 2), "guided_token_mask"),
        (
            torch.tensor([[True, False]]),
            torch.tensor([[False, True]]),
            "subset",
        ),
        (
            torch.ones(1, 2, dtype=torch.bool),
            torch.zeros(2, 1, dtype=torch.bool),
            "shape",
        ),
    ],
)
def test_masks_are_strictly_validated(response_mask, guided_mask, message) -> None:
    tensor = torch.zeros(1, 2)
    with pytest.raises((TypeError, ValueError), match=message):
        capsule_critique_policy_loss(
            tensor,
            tensor,
            tensor,
            response_mask,
            guided_mask,
        )


def test_verl_slot_is_explicitly_mapped_and_numeric_is_weights_are_rejected() -> None:
    response_mask = torch.tensor([[True, True]])
    guided_mask = torch.tensor([[False, True]])
    batch = {
        "response_mask": response_mask,
        "guided_token_mask": guided_mask,
    }

    mapped = map_guided_token_mask_to_verl_slot(batch)

    assert mapped["guided_token_mask"] is guided_mask
    assert mapped["rollout_is_weights"] is guided_mask

    tensor = torch.zeros(1, 2)
    with pytest.raises(TypeError, match="guided_token_mask"):
        verl_capsule_critique_policy_loss(
            tensor,
            tensor,
            tensor,
            response_mask,
            config={"clip_ratio": 0.2},
            rollout_is_weights=torch.ones(1, 2),
        )


def test_verl_wrapper_fails_closed_when_guided_mask_slot_is_missing() -> None:
    tensor = torch.zeros(1, 2)
    response_mask = torch.ones(1, 2, dtype=torch.bool)

    with pytest.raises(ValueError, match="guided_token_mask.*rollout_is_weights"):
        verl_capsule_critique_policy_loss(
            tensor,
            tensor,
            tensor,
            response_mask,
            config={"clip_ratio": 0.2},
            rollout_is_weights=None,
        )


def test_verl_wrapper_reads_nested_gamma_and_returns_the_four_tuple() -> None:
    old = torch.tensor([[0.0]])
    logp = torch.tensor([[math.log(0.4)]])
    advantages = torch.tensor([[1.5]])
    response_mask = torch.tensor([[True]])
    guided_mask = torch.tensor([[True]])
    config = {
        "clip_ratio": 0.2,
        "clip_ratio_low": None,
        "clip_ratio_high": None,
        "clip_ratio_c": 3.0,
        "policy_loss": {"capsule_gamma": 0.2},
    }

    actual = verl_capsule_critique_policy_loss(
        old,
        logp,
        advantages,
        response_mask,
        config=config,
        rollout_is_weights=guided_mask,
    )
    expected = capsule_critique_policy_loss(
        old,
        logp,
        advantages,
        response_mask,
        guided_mask,
        capsule_gamma=0.2,
    )

    assert len(actual) == 4
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right)


def test_verl_wrapper_reads_nested_gamma_from_verl_style_mapping() -> None:
    class VerlStyleActorConfig(Mapping[str, object]):
        def __init__(self) -> None:
            self.clip_ratio = 0.2
            self.clip_ratio_low = None
            self.clip_ratio_high = None
            self.clip_ratio_c = 3.0
            self.policy_loss = {"capsule_gamma": 0.2}

        def __getitem__(self, key: str) -> object:
            return getattr(self, key)

        def __iter__(self) -> Iterator[str]:
            return iter(vars(self))

        def __len__(self) -> int:
            return len(vars(self))

    old = torch.tensor([[0.0]])
    logp = torch.tensor([[math.log(0.4)]])
    advantages = torch.tensor([[1.5]])
    response_mask = torch.tensor([[True]])
    guided_mask = torch.tensor([[True]])

    actual = verl_capsule_critique_policy_loss(
        old,
        logp,
        advantages,
        response_mask,
        config=VerlStyleActorConfig(),
        rollout_is_weights=guided_mask,
    )
    expected = capsule_critique_policy_loss(
        old,
        logp,
        advantages,
        response_mask,
        guided_mask,
        capsule_gamma=0.2,
    )

    assert len(actual) == 4
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right)


@pytest.mark.parametrize(
    "algorithm",
    [
        {"rollout_is": True, "rollout_is_threshold": None},
        {"rollout_is": False, "rollout_is_threshold": 2.0},
    ],
)
def test_config_validator_disables_standard_rollout_importance_sampling(algorithm) -> None:
    with pytest.raises(ValueError, match="rollout importance sampling"):
        validate_capsule_training_config({"algorithm": algorithm})


def test_config_validator_accepts_disabled_rollout_importance_sampling() -> None:
    validate_capsule_training_config(
        {"algorithm": {"rollout_is": False, "rollout_is_threshold": None}}
    )
    assert CAPSULE_CRITIQUE_LOSS_MODE == "capsule_critique"


def test_config_validator_reports_missing_algorithm_without_key_error() -> None:
    with pytest.raises(ValueError, match="requires an algorithm section"):
        validate_capsule_training_config({})
