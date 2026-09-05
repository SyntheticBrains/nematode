"""State-dependent log-std head for continuous policy modes.

A minimal linear head mapping a brain's trunk feature to per-action-dim
``log_std``. The weight and bias are built **directly as zero
``nn.Parameter``s** — construction consumes **no RNG draws**
(``nn.Linear``-then-zero would overwrite values but not rewind the torch
generator, shifting every subsequent draw). Zero parameters mean
``log_std ≡ 0 → std = 1`` for every state at step 0, bit-for-bit the
state-independent parameter's init, so "on-at-init ≡ off-at-init" is a run
property.

Leaf module by design (imports only from ``._policy``); the shared
tanh-Gaussian helpers are shape-generic over ``log_std`` and are left
untouched.
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

if TYPE_CHECKING:
    from quantumnematode.brain.arch._brain import BrainHistoryData

# Weight-component names for the two std mechanisms (also keyed on by the
# cross-mode load guard below).
STD_HEAD_COMPONENT = "log_std_head"
STD_PARAM_COMPONENT = "log_std"
# Telemetry keys for the ceiling monitor when routed through a rule report.
LOG_STD_CLAMPED_MEAN_KEY = "log_std_clamped_mean"
LOG_STD_CLAMPED_MAX_KEY = "log_std_clamped_max"


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
    """Mean and max of the clamped log-std batch (the std ceiling monitor).

    ``torch.clamp`` has zero gradient outside ``[-5, 2]``, so a state whose
    head output drifts past the +2 ceiling receives no restoring gradient
    through the std path — a per-state stuck-at-max-entropy trap the single
    shared parameter could not exhibit. Mode-on runs record these stats per
    update so ceiling-pinning is visible in telemetry rather than
    discoverable only through a degenerate policy.
    """
    clamped = clamp_continuous_log_std(log_std.detach())
    return clamped.mean().item(), clamped.max().item()


def raise_on_std_mode_mismatch(
    components: Mapping[str, object],
    *,
    state_dependent: bool,
) -> None:
    """Reject cross-mode weight loads with a descriptive error.

    A mode-on brain fed a mode-off file (or vice versa) must not
    AttributeError or silently skip the std component — the loaded policy's
    exploration would be wrong with no signal.
    """
    training_state = components.get("training_state")
    saved_state = getattr(training_state, "state", None)
    saved_mode = saved_state.get("continuous_std_mode") if isinstance(saved_state, dict) else None
    if saved_mode is not None and (saved_mode == "state_dependent") != state_dependent:
        msg = (
            f"Weight file was saved with continuous_std_mode: {saved_mode} but this "
            "brain runs the other std mode. Std modes must match between the saved "
            "weights and the loading config (component subsets included)."
        )
        raise ValueError(msg)
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


def std_parameters(
    *,
    state_dependent: bool,
    log_std_head: nn.Module | None = None,
    log_std: nn.Parameter | None = None,
) -> list[nn.Parameter]:
    """Parameters of the active std mechanism (empty for discrete brains).

    Computed once at construction and reused for both the optimiser and the
    grad-clip lists, so the two can never silently diverge.
    """
    if state_dependent:
        if log_std_head is None:
            msg = "state-dependent std requires a log_std_head"
            raise ValueError(msg)
        return list(log_std_head.parameters())
    if log_std is not None:
        return [log_std]
    return []


def record_clamped_log_std(
    history_data: BrainHistoryData,
    log_stds: list[torch.Tensor],
) -> None:
    """Append the ceiling-monitor stats for one update (no-op on empty input).

    ``log_stds`` holds the final training epoch's detached per-minibatch (or
    per-step) log-std tensors, so the recorded value reflects the near-final
    weights of the update rather than being diluted by stale earlier-epoch
    evaluations.
    """
    if not log_stds:
        return
    ls_mean, ls_max = clamped_log_std_stats(torch.cat([t.reshape(-1) for t in log_stds]))
    history_data.log_std_clamped_mean.append(ls_mean)
    history_data.log_std_clamped_max.append(ls_max)
