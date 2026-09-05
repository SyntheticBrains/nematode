"""Frozen pre-generalisation reference of the three-factor update.

Copied VERBATIM from ``ConnectomeThreeFactorRule.step`` immediately before
the rule was rewritten to read the plastic-topology seam, rewritten only as
a free function over the connectome topology and explicit hyperparameters.
It names the connectome's attributes directly -- that is the point: the
generic rule must reproduce this arithmetic bit for bit on the connectome,
and this is what it is compared against.

Do NOT modernise this file to track the live implementation -- its entire
value is that it does not move.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from quantumnematode.brain.arch.connectome_ppo import ConnectomeTopology


def legacy_three_factor_step(  # noqa: PLR0913 -- mirrors the rule's hyperparameters
    topo: ConnectomeTopology,
    *,
    reward: float,
    baseline: float,
    plasticity_rate: float,
    weight_decay: float,
    weight_bound: float,
    baseline_rate: float,
    freeze_updates: bool,
    modulated: bool,
) -> tuple[float, float, float, float]:
    """One pre-generalisation update; returns (delta, baseline, mean_abs_delta, saturated)."""
    with torch.no_grad():
        delta = reward - baseline
        baseline += baseline_rate * delta
        modulator = delta if modulated else 1.0

        weights = topo.w_chem.data
        before = weights.detach().clone()

        if freeze_updates:
            pass
        else:
            update = plasticity_rate * modulator * topo.activity_traces
            update -= plasticity_rate * weight_decay * weights
            update = topo.apply_weight_mask(update)
            weights.add_(update)
            weights.clamp_(-weight_bound, weight_bound)

        mean_abs_delta = (weights - before).abs().mean().item()
        edge_count = int(topo.m_chem.sum().item())
        if edge_count == 0:  # pragma: no cover
            saturated = 0.0
        else:
            at_bound = ((weights.abs() >= weight_bound) & topo.m_chem).sum().item()
            saturated = at_bound / edge_count
    return delta, baseline, mean_abs_delta, saturated
