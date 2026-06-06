"""Unit tests for the continuous (tanh-squashed Gaussian) policy helpers.

Covers the continuous-mode requirements: bounded sampled actions, the
Jacobian-corrected log-probability, base-Normal entropy, log-std clamping, and
the finite / non-degenerate guarantees (no NaN/Inf at extreme means or log-stds).
"""

from __future__ import annotations

import math

import torch
from quantumnematode.brain.arch._policy import (
    CONTINUOUS_ACTION_DIM,
    CONTINUOUS_LOG_STD_MAX,
    CONTINUOUS_LOG_STD_MIN,
    clamp_continuous_log_std,
    continuous_evaluate_tanh_gaussian,
    continuous_sample_tanh_gaussian,
)

# Physical action bounds: speed in [0, 1], turn in [-pi, pi].
_LOW: torch.Tensor = torch.tensor([0.0, -math.pi])
_HIGH: torch.Tensor = torch.tensor([1.0, math.pi])


class TestBoundedSampling:
    """Sampled actions always lie within the configured physical bounds."""

    def test_sampled_action_within_bounds(self) -> None:
        """Many draws from a unit Gaussian stay within speed/turn bounds."""
        torch.manual_seed(0)
        mean = torch.zeros(CONTINUOUS_ACTION_DIM)
        log_std = torch.zeros(CONTINUOUS_ACTION_DIM)
        for _ in range(500):
            action, _, _, _ = continuous_sample_tanh_gaussian(mean, log_std, _LOW, _HIGH)
            # Compare against the (float32) bound tensors: at tanh saturation the
            # action equals the bound exactly in float32, whereas math.pi is
            # float64 and differs in the last ulp.
            assert torch.all(action >= _LOW)
            assert torch.all(action <= _HIGH)

    def test_extreme_mean_stays_bounded_and_finite(self) -> None:
        """A saturating mean keeps the action within bounds and everything finite."""
        mean = torch.tensor([100.0, -100.0])
        log_std = torch.zeros(CONTINUOUS_ACTION_DIM)
        action, log_prob, entropy, pre_tanh = continuous_sample_tanh_gaussian(
            mean,
            log_std,
            _LOW,
            _HIGH,
        )
        for tensor in (action, log_prob, entropy, pre_tanh):
            assert torch.isfinite(tensor).all()
        assert torch.all(action >= _LOW)
        assert torch.all(action <= _HIGH)


class TestJacobianLogProb:
    """The log-probability carries the tanh + affine change-of-variables terms."""

    def test_matches_manual_change_of_variables(self) -> None:
        """Helper log-prob equals an independent manual change-of-variables computation."""
        mean = torch.tensor([[0.2, -0.5]])
        log_std = torch.tensor([[0.0, -0.3]])
        pre_tanh = torch.tensor([[0.4, 0.1]])
        log_probs, _ = continuous_evaluate_tanh_gaussian(mean, log_std, pre_tanh, _LOW, _HIGH)

        base = torch.distributions.Normal(mean, torch.exp(log_std))
        half_range = (_HIGH - _LOW) / 2.0
        manual = (
            base.log_prob(pre_tanh).sum(-1)
            - torch.log(half_range).sum(-1)
            - torch.log(1.0 - torch.tanh(pre_tanh) ** 2).sum(-1)
        )
        assert torch.allclose(log_probs, manual, atol=1e-5)

    def test_sample_logprob_matches_evaluate(self) -> None:
        """Rollout sample log-prob/entropy match the batched update re-scoring."""
        torch.manual_seed(3)
        mean = torch.tensor([0.1, 0.2])
        log_std = torch.tensor([-0.1, 0.0])
        _, log_prob, entropy, pre_tanh = continuous_sample_tanh_gaussian(
            mean,
            log_std,
            _LOW,
            _HIGH,
        )
        log_probs, mean_entropy = continuous_evaluate_tanh_gaussian(
            mean.unsqueeze(0),
            log_std.unsqueeze(0),
            pre_tanh.unsqueeze(0),
            _LOW,
            _HIGH,
        )
        assert torch.allclose(log_prob, log_probs.squeeze(0), atol=1e-6)
        # Single sample → batch-mean entropy equals the per-sample entropy.
        assert torch.allclose(entropy, mean_entropy, atol=1e-6)

    def test_evaluate_is_differentiable(self) -> None:
        """The update path back-propagates finite gradients to mean and log-std."""
        mean = torch.tensor([[0.1, -0.2]], requires_grad=True)
        log_std = torch.tensor([[0.0, 0.0]], requires_grad=True)
        pre_tanh = torch.tensor([[0.3, -0.1]])
        log_probs, mean_entropy = continuous_evaluate_tanh_gaussian(
            mean,
            log_std,
            pre_tanh,
            _LOW,
            _HIGH,
        )
        (log_probs.sum() + mean_entropy).backward()
        assert mean.grad is not None
        assert log_std.grad is not None
        assert torch.isfinite(mean.grad).all()
        assert torch.isfinite(log_std.grad).all()


class TestEntropy:
    """Base-Normal entropy behaves monotonically in the log-std."""

    def test_entropy_increases_with_log_std(self) -> None:
        """A wider Gaussian has greater (base-Normal) entropy."""
        mean = torch.zeros(CONTINUOUS_ACTION_DIM)
        _, _, entropy_narrow, _ = continuous_sample_tanh_gaussian(
            mean,
            torch.full((CONTINUOUS_ACTION_DIM,), -1.0),
            _LOW,
            _HIGH,
        )
        _, _, entropy_wide, _ = continuous_sample_tanh_gaussian(
            mean,
            torch.full((CONTINUOUS_ACTION_DIM,), 1.0),
            _LOW,
            _HIGH,
        )
        assert float(entropy_wide) > float(entropy_narrow)


class TestLogStdClamp:
    """Log-std clamping keeps the distribution finite and non-degenerate."""

    def test_clamp_bounds(self) -> None:
        """Clamp maps out-of-range log-stds onto the configured bounds."""
        clamped = clamp_continuous_log_std(torch.tensor([-100.0, 0.0, 100.0]))
        assert float(clamped[0]) == CONTINUOUS_LOG_STD_MIN
        assert float(clamped[1]) == 0.0
        assert float(clamped[2]) == CONTINUOUS_LOG_STD_MAX

    def test_extreme_log_std_stays_finite(self) -> None:
        """Pathological log-std inputs are clamped, so outputs remain finite."""
        mean = torch.zeros(CONTINUOUS_ACTION_DIM)
        for raw_log_std in (-1.0e3, 1.0e3):
            log_std = torch.full((CONTINUOUS_ACTION_DIM,), raw_log_std)
            action, log_prob, entropy, pre_tanh = continuous_sample_tanh_gaussian(
                mean,
                log_std,
                _LOW,
                _HIGH,
            )
            for tensor in (action, log_prob, entropy, pre_tanh):
                assert torch.isfinite(tensor).all()


class TestDeterminism:
    """Sampling is reproducible under a fixed torch seed."""

    def test_seeded_sample_reproducible(self) -> None:
        """Same seed → identical action, pre-squash sample, and log-prob."""
        mean = torch.tensor([0.0, 0.0])
        log_std = torch.zeros(CONTINUOUS_ACTION_DIM)

        torch.manual_seed(42)
        action_a, log_prob_a, _, pre_a = continuous_sample_tanh_gaussian(
            mean,
            log_std,
            _LOW,
            _HIGH,
        )
        torch.manual_seed(42)
        action_b, log_prob_b, _, pre_b = continuous_sample_tanh_gaussian(
            mean,
            log_std,
            _LOW,
            _HIGH,
        )
        assert torch.equal(action_a, action_b)
        assert torch.equal(pre_a, pre_b)
        assert torch.equal(log_prob_a, log_prob_b)
