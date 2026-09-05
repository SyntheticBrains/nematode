"""Configuration shared by every brain that offers the plasticity rules.

"Matched rule" is a claim that two arms were trained with the same
hyperparameters. That claim only survives if there is one place those
numbers are defined: two hand-copied blocks of defaults are equal until
someone edits one of them, and nothing would announce the divergence.
Every brain that can select a plasticity rule inherits these fields, their
bounds, and the trace-pairing validator from here.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

LearningRuleName = Literal["ppo", "three_factor", "hebbian"]

# Rules that read the eligibility trace, and so require it to be enabled.
PLASTIC_RULES = frozenset({"three_factor", "hebbian"})
# Plastic rules that apply no neuromodulatory factor — the ablation floors.
# A new plastic rule must be classified here deliberately rather than
# inheriting a default: whether an update is gated by reward is the property
# the panel's floors are built to isolate, so getting it silently wrong would
# mislabel an arm rather than break it.
UNMODULATED_RULES = frozenset({"hebbian"})


class PlasticityConfigMixin(BaseModel):
    """Rule selection, plasticity hyperparameters, and the trace substrate.

    Mixed into a brain's config class ahead of ``BrainConfig`` in the base
    list. Pydantic composes the two models' fields and runs both classes'
    validators, so a brain inheriting this gets the pairing check below in
    addition to its own.
    """

    # ── Learning rule ────────────────────────────────────────
    # ``ppo`` is the default and is byte-identical to builds predating this
    # field. ``three_factor`` selects reward-modulated Hebbian plasticity;
    # ``hebbian`` is the same update with the neuromodulatory factor
    # removed — plain co-activity learning, the ablation that separates
    # "learned something" from "learned something from reward". Both
    # plastic modes read the eligibility trace, hence the pairing check.
    learning_rule: LearningRuleName = "ppo"

    # ── Activity-trace substrate (opt-in) ────────────────────
    # Per-weight eligibility traces accumulated during rollout forwards.
    # Off by default: trace-off builds are byte-identical, and while no
    # rule consumes the traces, trace-ON training is bit-identical too.
    # ``trace_decay`` is bounded to [0, 1) at load time — a decay >= 1 is a
    # divergent accumulator.
    enable_activity_traces: bool = False
    trace_decay: float = Field(default=0.9, ge=0.0, lt=1.0)

    # ── Three-factor hyperparameters (ignored under ``ppo``) ─
    # Bounds are load-time rather than advisory: a non-positive rate makes
    # the rule a no-op, a decay outside [0, 1) either does nothing or
    # inverts weights each step, and a non-positive bound collapses every
    # weight to zero.
    plasticity_rate: float = Field(default=0.01, gt=0.0)
    plasticity_weight_decay: float = Field(default=0.001, ge=0.0, lt=1.0)
    # The bound must clear the initialisation it is applied to on EVERY
    # substrate that shares it. The connectome's chemical weights start at
    # N(0, 1/sqrt(chemical in-degree)) with a tail near |w| ~ 1.5-1.7; the
    # MLP's orthogonal layers sit near 0.6. A bound of 1.0 would clamp a
    # handful of connectome synapses on the very first update — silently
    # starting that arm from a different substrate than the frozen baseline
    # it is compared against. 3.0 is roughly ten connectome initial standard
    # deviations: ample room for growth, no contact with any starting weight.
    plasticity_weight_bound: float = Field(default=3.0, gt=0.0)
    plasticity_baseline_rate: float = Field(default=0.01, gt=0.0, le=1.0)

    # ── Paired-control freeze ────────────────────────────────
    # Run everything -- rollouts, telemetry, bookkeeping -- but never write
    # a weight. Honoured by every rule on every brain that inherits this, so
    # a "frozen" arm means the same thing wherever it appears: a control that
    # quietly kept learning would be indistinguishable from a plastic arm in
    # its config and very different in its results.
    freeze_updates: bool = False

    @model_validator(mode="after")
    def _validate_rule_pairing(self) -> PlasticityConfigMixin:
        """Reject a plasticity rule with no eligibility trace to read.

        The three-factor update is proportional to the trace, so without
        one every update is identically zero: training would appear to run
        and change nothing. Failing at load is the difference between a
        typo and a silently wasted campaign.
        """
        if self.learning_rule in PLASTIC_RULES and not self.enable_activity_traces:
            msg = (
                f"learning_rule={self.learning_rule!r} requires "
                "enable_activity_traces=true: the update is proportional to the "
                "eligibility trace, so without one every weight update would be "
                "identically zero."
            )
            raise ValueError(msg)
        return self
