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

The continuous (tanh-squashed Gaussian) policy operations are added by the
continuous-action-heads work and live alongside these.

Leaf-module discipline: import these helpers *directly* from the brain modules.
Do NOT re-export them through ``brain/arch/__init__.py`` (which imports the brain
modules and would invert the dependency / risk an import cycle), following the
``_ppo_buffer`` / ``_brain`` precedent.
"""

from __future__ import annotations

import torch

__all__ = [
    "categorical_evaluate_torch",
    "categorical_logprob_entropy_torch",
    "categorical_sample_torch",
    "ppo_clip_policy_loss",
]


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
