"""Unit tests for the shared action-policy helpers (`brain/arch/_policy.py`).

Covers the discrete-mode scenarios at
``openspec/changes/add-continuous-2d-and-action-heads/specs/continuous-action-policy/spec.md``:
the helpers must reproduce the per-brain inline numerics exactly (byte-equivalent
migration). Continuous (tanh-squashed Gaussian) helpers are tested when added by
the continuous-action-heads work.
"""

from __future__ import annotations

import numpy as np
import torch
from quantumnematode.brain.arch._policy import (
    categorical_evaluate_torch,
    categorical_sample_numpy,
    categorical_sample_torch,
    ppo_clip_policy_loss,
)


class TestCategoricalSampleTorch:
    """The torch-backend sampler must match the inline MLP-PPO / connectome-PPO path."""

    def test_matches_inline_categorical_under_same_seed(self) -> None:
        logits = torch.tensor([0.5, -1.2, 0.3, 0.9])

        # Inline reference (the pre-refactor per-brain code).
        torch.manual_seed(123)
        probs_ref = torch.softmax(logits, dim=-1)
        dist_ref = torch.distributions.Categorical(probs_ref)
        action_ref = int(dist_ref.sample().item())
        log_prob_ref = dist_ref.log_prob(torch.tensor(action_ref))
        entropy_ref = dist_ref.entropy()

        # Helper, same RNG state.
        torch.manual_seed(123)
        action, log_prob, entropy, probs = categorical_sample_torch(logits)

        assert action == action_ref
        assert torch.equal(probs, probs_ref)
        assert torch.equal(log_prob, log_prob_ref)
        assert torch.equal(entropy, entropy_ref)

    def test_outputs_finite_and_action_in_range(self) -> None:
        torch.manual_seed(0)
        logits = torch.randn(4)
        action, log_prob, entropy, probs = categorical_sample_torch(logits)
        assert 0 <= action < 4
        assert torch.isfinite(log_prob)
        assert torch.isfinite(entropy)
        assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-6)


class TestCategoricalEvaluateTorch:
    """The torch-backend batch evaluator must match the inline PPO-update path."""

    def test_matches_inline_batch_logprob_and_entropy(self) -> None:
        logits = torch.tensor([[0.5, -1.2, 0.3, 0.9], [0.1, 0.2, -0.5, 0.0]])
        actions = torch.tensor([2, 0])

        probs_ref = torch.softmax(logits, dim=-1)
        dist_ref = torch.distributions.Categorical(probs_ref)
        log_probs_ref = dist_ref.log_prob(actions)
        entropy_ref = dist_ref.entropy().mean()

        log_probs, entropy = categorical_evaluate_torch(logits, actions)

        assert torch.equal(log_probs, log_probs_ref)
        assert torch.equal(entropy, entropy_ref)


class TestCategoricalSampleNumpy:
    """The numpy-backend sampler must match the inline LSTM-PPO / CfC-PPO path."""

    def test_matches_inline_rng_choice_and_manual_logprob(self) -> None:
        probs = np.array([0.1, 0.2, 0.3, 0.4])

        # Inline reference.
        rng_ref = np.random.default_rng(42)
        n_actions = len(probs)
        action_ref = int(rng_ref.choice(n_actions, p=probs))
        log_prob_ref = float(np.log(probs[action_ref]))

        # Helper, same seed → same RNG stream.
        rng = np.random.default_rng(42)
        action, log_prob = categorical_sample_numpy(probs, rng)

        assert action == action_ref
        assert log_prob == log_prob_ref

    def test_rng_stream_is_consumed_identically(self) -> None:
        probs = np.array([0.25, 0.25, 0.25, 0.25])
        rng_ref = np.random.default_rng(7)
        ref = [int(rng_ref.choice(4, p=probs)) for _ in range(5)]
        rng = np.random.default_rng(7)
        got = [categorical_sample_numpy(probs, rng)[0] for _ in range(5)]
        assert got == ref


class TestPPOClipPolicyLoss:
    """The shared clipped surrogate must match the inline per-brain term."""

    def test_matches_inline_surrogate(self) -> None:
        new_log_probs = torch.tensor([-0.5, -1.0, -0.2])
        old_log_probs = torch.tensor([-0.7, -0.9, -0.3])
        advantages = torch.tensor([1.0, -2.0, 0.5])
        clip_epsilon = 0.2

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
        loss_ref = -torch.min(surr1, surr2).mean()

        loss = ppo_clip_policy_loss(new_log_probs, old_log_probs, advantages, clip_epsilon)
        assert torch.equal(loss, loss_ref)
