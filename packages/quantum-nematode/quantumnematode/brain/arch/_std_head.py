"""State-dependent log-std head for continuous policy modes (roadmap D7).

A minimal linear head mapping a brain's trunk feature to per-action-dim
``log_std``. The weight and bias are built **directly as zero
``nn.Parameter``s** — construction consumes **no RNG draws** (the change's
review B1: ``nn.Linear``-then-zero would overwrite values but not rewind
the torch generator, shifting every subsequent draw). Zero parameters mean
``log_std ≡ 0 → std = 1`` for every state at step 0, bit-for-bit the
state-independent parameter's init, so "on-at-init ≡ off-at-init" is a run
property.

Leaf module by design (imports only from ``._policy``); the shared
tanh-Gaussian helpers are shape-generic over ``log_std`` and are not
touched by D7.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Mapping

from quantumnematode.brain.arch._policy import (
    CONTINUOUS_ACTION_DIM,
    clamp_continuous_log_std,
)


class StateDependentLogStdHead(nn.Module):
    """Zero-initialised, RNG-free linear head: trunk feature → ``log_std``.

    Output shape mirrors the input's leading dims: ``(in,) → (2,)``,
    ``(B, in) → (B, 2)`` — matching what each brain passes as ``mean`` to
    the shared samplers/evaluators.
    """

    def __init__(self, in_features: int, device: torch.device) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.zeros(CONTINUOUS_ACTION_DIM, in_features, device=device),
        )
        self.bias = nn.Parameter(torch.zeros(CONTINUOUS_ACTION_DIM, device=device))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute per-state ``log_std`` from the trunk feature."""
        return nn.functional.linear(features, self.weight, self.bias)


def clamped_log_std_stats(log_std: torch.Tensor) -> tuple[float, float]:
    """Mean and max of the clamped log-std batch (the D7 ceiling monitor).

    ``torch.clamp`` has zero gradient outside ``[-5, 2]``, so a state whose
    head output drifts past the +2 ceiling receives no restoring gradient
    through the std path — a per-state stuck-at-max-entropy trap the single
    shared parameter could not exhibit. Mode-on runs record these stats per
    update; the pre-registered response to ceiling-pinning is the bounded
    entropy-only repair pass (change design § Risks).
    """
    clamped = clamp_continuous_log_std(log_std.detach())
    return clamped.mean().item(), clamped.max().item()


def raise_on_std_mode_mismatch(
    components: Mapping[str, object],
    *,
    state_dependent: bool,
) -> None:
    """Reject cross-mode weight loads with a descriptive error (review S3).

    A mode-on brain fed a mode-off file (or vice versa) must not
    AttributeError or silently skip the std component — the loaded policy's
    exploration would be wrong with no signal.
    """
    if state_dependent and "log_std" in components:
        msg = (
            "Weight file carries a state-independent 'log_std' component but this "
            "brain runs continuous_std_mode: state_dependent. Std modes must match "
            "between the saved weights and the loading config."
        )
        raise ValueError(msg)
    if not state_dependent and "log_std_head" in components:
        msg = (
            "Weight file carries a state-dependent 'log_std_head' component but this "
            "brain runs continuous_std_mode: state_independent. Std modes must match "
            "between the saved weights and the loading config."
        )
        raise ValueError(msg)
