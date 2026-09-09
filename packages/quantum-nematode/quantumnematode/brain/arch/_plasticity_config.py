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
# Consolidation mechanisms. A mechanism slows or stops the rule writing where it has
# already written, which the minimal rule does not do at any rate: its update magnitude
# is set by the normalised modulator and trace, not by how good the policy is. Selected
# one at a time rather than composed, since two brakes at once give a result attributable
# to neither.
ConsolidationName = Literal["none", "anchor", "rigidity", "oracle"]

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

    # ── Substrate-invariant scaling (opt-in) ────────────────
    # Two independent switches, both off by default so the raw rule is
    # byte-identical. With the modulator normalised the third factor is
    # tanh(delta / sigma), sigma a bias-corrected running RMS of the raw
    # prediction error: bounded, sign-preserving, and no longer dominated by
    # the one terminal penalty that dwarfs every other step. With the trace
    # normalised the Hebbian term is divided by rho, a bias-corrected running
    # RMS of each plastic tensor's trace over its edge set, so the rate means
    # the same root-mean-square step per unit modulator on a sparse recurrent
    # connectome and on a dense feedforward stack whose traces differ by
    # orders of magnitude. The scale rate is the EMA rate of both estimators;
    # the floor sits under both before division.
    plasticity_normalise_modulator: bool = False
    plasticity_normalise_trace: bool = False
    plasticity_scale_rate: float = Field(default=0.01, gt=0.0, le=1.0)
    plasticity_scale_floor: float = Field(default=1e-6, gt=0.0)

    # ── Consolidation (opt-in) ───────────────────────────────
    # ``anchor`` gives every plastic tensor a slow moving average of its own
    # weights and adds a restoring term toward it, so a steady departure
    # builds a force against itself; an anchor rate of zero holds the anchor
    # at the weights the rule started from. ``rigidity`` gives every synapse a
    # protective variable that grows where a positive modulator met a large
    # trace and divides the rate by ``1 + strength * c``, so synapses reward
    # has repeatedly written become progressively harder to write. ``oracle``
    # scales the rate by how far a trailing episode-success rate sits below a
    # reference: it reads the environment's success flag rather than the
    # reward stream, so it is a diagnostic bound on what a quality-gated brake
    # could do and not a mechanism an animal could host.
    #
    # Each mechanism's parameters default to values that leave it inert, and
    # selecting a mechanism that its own parameters leave inert is rejected
    # below: a config that names a brake and does not brake would be read as
    # evidence about the brake.
    plasticity_consolidation: ConsolidationName = "none"
    plasticity_anchor_rate: float = Field(default=0.0, ge=0.0, lt=1.0)
    plasticity_anchor_stiffness: float = Field(default=0.0, ge=0.0)
    plasticity_rigidity_growth: float = Field(default=0.0, ge=0.0)
    plasticity_rigidity_decay: float = Field(default=0.0, ge=0.0, lt=1.0)
    plasticity_rigidity_strength: float = Field(default=0.0, ge=0.0)
    # 1.0 never closes the gate: a run that selects the oracle pins its own
    # reference, rather than inheriting one panel's comparator from a default.
    plasticity_oracle_reference: float = Field(default=1.0, gt=0.0, le=1.0)
    plasticity_oracle_rate: float = Field(default=0.01, gt=0.0, le=1.0)

    # ── Homeostatic incoming-norm scaling (opt-in) ───────────
    # After each plastic update every unit's incoming plastic weights are
    # rescaled, over the edge set only, to the norm they had when the rule
    # was built. Off by default; on, the substrate cannot run away onto the
    # magnitude bound, and the decay term is undone by the rescale.
    plasticity_homeostasis: bool = False

    # ── Initial action noise ─────────────────────────────────
    # The state-independent continuous log-std starts here on every brain
    # that offers the plasticity rules. Zero (an action std of 1.0) is the
    # historical value and is byte-identical. Under a plastic rule the
    # parameter never trains, so this is the noise the arm explores with for
    # its whole run; under the gradient rule it trains from here. Ignored in
    # discrete mode; rejected below when the state-dependent std head is
    # selected, since that head computes its log-std from the hidden state.
    initial_log_std: float = 0.0

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
        std_mode = getattr(self, "continuous_std_mode", "state_independent")
        if self.initial_log_std != 0.0 and std_mode == "state_dependent":
            msg = (
                "initial_log_std has no effect under the state-dependent std head, whose "
                "log-std is a function of the hidden state; a non-zero value would silently "
                "do nothing."
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _validate_consolidation(self) -> PlasticityConfigMixin:
        """Reject a consolidation mechanism its own parameters leave inert.

        A config that names a brake and does not brake is worse than one
        that names none: its runs would be read as evidence about the
        mechanism. Each mechanism is inert when the parameter it multiplies
        through is zero.
        """
        inert = {
            "anchor": ("plasticity_anchor_stiffness",),
            "rigidity": ("plasticity_rigidity_growth", "plasticity_rigidity_strength"),
            "oracle": (),
        }.get(self.plasticity_consolidation, ())
        zeroed = [name for name in inert if getattr(self, name) == 0.0]
        if zeroed:
            msg = (
                f"plasticity_consolidation={self.plasticity_consolidation!r} is inert with "
                f"{', '.join(f'{name}=0' for name in zeroed)}: the mechanism would be selected "
                "and never act, and its runs would be read as evidence about it."
            )
            raise ValueError(msg)
        return self
