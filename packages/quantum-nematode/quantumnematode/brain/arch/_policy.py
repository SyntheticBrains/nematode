"""Shared action-policy helpers for PPO-family brains.

This module factors the action sampling, log-probability, entropy, and PPO
surrogate terms out of the per-brain copy-paste (MLP-PPO, LSTM-PPO, CfC-PPO,
connectome-PPO each re-implemented these independently). The helpers preserve
each brain's *exact* numerics so the migration onto this module is
byte-equivalent.

The brains use two different RNG streams, and the migration preserves each:

- MLP-PPO and connectome-PPO use ``torch.distributions.Categorical`` end to end
  → ``categorical_sample_torch`` (rollout) + ``categorical_evaluate_torch``
  (update). Byte-exact.
- LSTM-PPO and CfC-PPO sample via ``numpy`` ``rng.choice`` (kept inline so the
  sampled-action trajectory is byte-identical) but route their log-probability
  and entropy through ``categorical_logprob_entropy_torch`` (rollout, per given
  action) and ``categorical_evaluate_torch`` (update). This replaces their manual
  ``log(softmax)`` / ``-sum(p*log p)`` with torch's numerically-stabler equivalents;
  measured deviation on the taken action is ~1e-7 (float32 round-off).

The continuous (tanh-squashed Gaussian) policy operations live alongside the
discrete ones: ``continuous_sample_tanh_gaussian`` (rollout — returns the bounded
``(speed, turn)`` action, a Jacobian-corrected log-prob, the base-Normal entropy,
and the pre-squash sample to store) and ``continuous_evaluate_tanh_gaussian``
(update — re-scores the stored pre-squash samples under the current parameters).
``clamp_continuous_log_std`` keeps the Gaussian finite and non-degenerate.

Leaf-module discipline: import these helpers *directly* from the brain modules.
Do NOT re-export them through ``brain/arch/__init__.py`` (which imports the brain
modules and would invert the dependency / risk an import cycle), following the
``_ppo_buffer`` / ``_brain`` precedent.
"""

from __future__ import annotations

import math

import torch
from torch.nn import functional

__all__ = [
    "CONTINUOUS_ACTION_DIM",
    "CONTINUOUS_LOG_STD_MAX",
    "CONTINUOUS_LOG_STD_MIN",
    "categorical_evaluate_torch",
    "categorical_logprob_entropy_torch",
    "categorical_sample_torch",
    "clamp_continuous_log_std",
    "continuous_evaluate_tanh_gaussian",
    "continuous_sample_tanh_gaussian",
    "ppo_clip_policy_loss",
]

# Continuous (tanh-squashed Gaussian) action dimension: ``(speed, turn)``.
CONTINUOUS_ACTION_DIM: int = 2

# Log-std clamp range. Keeps the Gaussian strictly positive and bounded so
# sampling / log-prob / entropy stay finite (no collapse to a delta, no blow-up).
CONTINUOUS_LOG_STD_MIN: float = -5.0
CONTINUOUS_LOG_STD_MAX: float = 2.0

_LOG_2: float = math.log(2.0)


def categorical_sample_torch(
    logits: torch.Tensor,
    *,
    device: torch.device | str | None = None,
) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample a discrete action from ``Categorical(softmax(logits))`` (torch backend).

    Mirrors the MLP-PPO / connectome-PPO sampling path exactly:
    ``probs = softmax(logits); dist = Categorical(probs); a = dist.sample();
    log_prob = dist.log_prob(a); entropy = dist.entropy()``.

    Args:
        logits: Raw action logits (1-D over the action set).
        device: Device for the sampled-action tensor used in ``log_prob``;
            defaults to ``logits.device``.

    Returns
    -------
        ``(action_idx, log_prob, entropy, probs)``.
    """
    probs = torch.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)
    action = int(dist.sample().item())
    target_device = device if device is not None else logits.device
    log_prob = dist.log_prob(torch.tensor(action, device=target_device))
    entropy = dist.entropy()
    return action, log_prob, entropy, probs


def categorical_evaluate_torch(
    logits: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-evaluate batch log-probs and mean entropy for a PPO update (torch backend).

    Mirrors the MLP-PPO / connectome-PPO update path exactly:
    ``dist = Categorical(softmax(logits)); dist.log_prob(actions); dist.entropy().mean()``.

    Args:
        logits: Batched action logits, shape ``(batch, n_actions)``.
        actions: Batched action indices, shape ``(batch,)``.

    Returns
    -------
        ``(log_probs, mean_entropy)`` where ``log_probs`` has shape ``(batch,)``.
    """
    probs = torch.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)
    return dist.log_prob(actions), dist.entropy().mean()


def categorical_logprob_entropy_torch(
    logits: torch.Tensor,
    action: int,
    *,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Torch ``Categorical`` log-prob + entropy for a GIVEN action (no sampling).

    Used by LSTM-PPO and CfC-PPO, which sample the action index via their inline
    ``numpy`` ``rng.choice`` (kept verbatim so the trajectory is byte-identical),
    then compute the log-probability and entropy here via torch — replacing their
    manual ``log(softmax)`` / ``-sum(p*log p)`` with the numerically-stabler torch
    equivalents (shared with the update path). Differentiable: usable inside the
    BPTT update loop as well as at rollout time.

    Args:
        logits: Raw action logits (1-D over the action set).
        action: The (already-sampled or stored) action index to score.
        device: Device for the action tensor; defaults to ``logits.device``.

    Returns
    -------
        ``(log_prob, entropy, probs)``.
    """
    probs = torch.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)
    target_device = device if device is not None else logits.device
    log_prob = dist.log_prob(torch.tensor(action, device=target_device))
    entropy = dist.entropy()
    return log_prob, entropy, probs


def ppo_clip_policy_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float,
) -> torch.Tensor:
    """Clipped PPO surrogate policy loss (shared across brains).

    Mirrors the per-brain term exactly:
    ``ratio = exp(new - old); -min(ratio*A, clamp(ratio, 1-e, 1+e)*A).mean()``.

    Args:
        new_log_probs: Log-probs under the current policy, shape ``(batch,)``.
        old_log_probs: Log-probs stored at rollout time, shape ``(batch,)``.
        advantages: Advantage estimates, shape ``(batch,)``.
        clip_epsilon: PPO clip range.

    Returns
    -------
        Scalar policy loss (already negated and mean-reduced).
    """
    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
    return -torch.min(surr1, surr2).mean()


# ──────────────────────────────────────────────────────────────────────────────
# Continuous mode: tanh-squashed diagonal Gaussian
# ──────────────────────────────────────────────────────────────────────────────
#
# The continuous action is ``(speed, turn)``: a sample ``u`` is drawn from a
# diagonal ``Normal(mean, std)``, squashed with ``tanh`` to ``[-1, 1]`` per dim,
# then affine-rescaled to the physical bounds (``speed ∈ [0, max]``,
# ``turn ∈ [-π, π]``). The log-probability carries the change-of-variables
# (log-det-Jacobian) correction for BOTH the tanh squash and the affine rescale,
# so the reported value is the density of the *squashed, rescaled* action.
#
# PPO needs to re-score the taken action under updated parameters. Rather than
# invert the squash (``atanh`` is unstable near ±1), the sampler returns the
# pre-squash ``u`` (``pre_tanh``); the buffer stores it, and the update re-scores
# via ``continuous_evaluate_tanh_gaussian(mean, log_std, pre_tanh, …)`` — ``u`` is
# a deterministic function of the action, so it is policy-parameter-independent.


def clamp_continuous_log_std(log_std: torch.Tensor) -> torch.Tensor:
    """Clamp a log-std tensor to ``[CONTINUOUS_LOG_STD_MIN, CONTINUOUS_LOG_STD_MAX]``."""
    return torch.clamp(log_std, CONTINUOUS_LOG_STD_MIN, CONTINUOUS_LOG_STD_MAX)


def _affine_center_half_range(
    action_low: torch.Tensor,
    action_high: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the ``(center, half_range)`` of the affine map ``[-1, 1] → [low, high]``."""
    center = (action_low + action_high) / 2.0
    half_range = (action_high - action_low) / 2.0
    return center, half_range


def _tanh_log_det_jacobian(pre_tanh: torch.Tensor) -> torch.Tensor:
    """Return summed ``log|d tanh(u)/du| = log(1 - tanh(u)^2)`` over the action dims.

    Uses the numerically stable identity ``2 * (log 2 - u - softplus(-2u))`` to
    avoid the ``log(0)`` that ``log(1 - tanh(u)**2)`` hits for large ``|u|``.
    """
    return (2.0 * (_LOG_2 - pre_tanh - functional.softplus(-2.0 * pre_tanh))).sum(-1)


def _tanh_gaussian_log_prob(
    base: torch.distributions.Normal,
    pre_tanh: torch.Tensor,
    half_range: torch.Tensor,
) -> torch.Tensor:
    """Jacobian-corrected log-prob of the squashed, rescaled action.

    ``log p(action) = sum log N(u; mean, std) - sum log(half_range)
    - sum log(1 - tanh(u)^2)``
    (the affine term is constant in the parameters but is included so the reported
    value is the true density of the bounded action).
    """
    log_prob_base = base.log_prob(pre_tanh).sum(-1)
    affine_correction = torch.log(half_range).sum(-1)
    return log_prob_base - affine_correction - _tanh_log_det_jacobian(pre_tanh)


def continuous_sample_tanh_gaussian(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample a bounded continuous action from a tanh-squashed Gaussian.

    Parameters
    ----------
    mean : torch.Tensor
        Gaussian mean per action dim, shape ``(…, action_dim)``.
    log_std : torch.Tensor
        Gaussian log-std per action dim (clamped internally).
    action_low : torch.Tensor
        Lower action bound per dim (e.g. ``[0, -1]``).
    action_high : torch.Tensor
        Upper action bound per dim (e.g. ``[1, 1]``).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ``(action, log_prob, entropy, pre_tanh)`` — ``action`` is the bounded
        ``(speed, turn)`` vector; ``log_prob`` is Jacobian-corrected; ``entropy``
        is the base-Normal differential entropy (the squashed entropy has no
        closed form — standard PPO surrogate); ``pre_tanh`` is the pre-squash
        sample to store for the update.
    """
    std = torch.exp(clamp_continuous_log_std(log_std))
    base = torch.distributions.Normal(mean, std)
    pre_tanh = base.sample()
    center, half_range = _affine_center_half_range(action_low, action_high)
    action = center + half_range * torch.tanh(pre_tanh)
    log_prob = _tanh_gaussian_log_prob(base, pre_tanh, half_range)
    entropy = base.entropy().sum(-1)
    return action, log_prob, entropy, pre_tanh


def continuous_evaluate_tanh_gaussian(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    pre_tanh: torch.Tensor,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-score stored pre-squash samples for a PPO update (tanh-Gaussian).

    Mirrors ``categorical_evaluate_torch`` for the continuous mode: given the
    stored ``pre_tanh`` samples and the *current* policy parameters, return the
    Jacobian-corrected log-probs and the mean base-Normal entropy.

    Parameters
    ----------
    mean : torch.Tensor
        Batched Gaussian mean, shape ``(batch, action_dim)``.
    log_std : torch.Tensor
        Batched (or broadcastable) log-std (clamped internally).
    pre_tanh : torch.Tensor
        Stored pre-squash samples, shape ``(batch, action_dim)``.
    action_low : torch.Tensor
        Lower action bound per dim.
    action_high : torch.Tensor
        Upper action bound per dim.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(log_probs, mean_entropy)`` where ``log_probs`` has shape ``(batch,)``.
    """
    std = torch.exp(clamp_continuous_log_std(log_std))
    base = torch.distributions.Normal(mean, std)
    _, half_range = _affine_center_half_range(action_low, action_high)
    log_probs = _tanh_gaussian_log_prob(base, pre_tanh, half_range)
    mean_entropy = base.entropy().sum(-1).mean()
    return log_probs, mean_entropy
