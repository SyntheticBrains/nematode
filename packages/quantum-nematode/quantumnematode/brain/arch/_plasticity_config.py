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

from quantumnematode.brain.arch._node_noise_schedule import NodeNoiseSchedule

LearningRuleName = Literal["ppo", "three_factor", "hebbian"]
# Consolidation mechanisms. A mechanism slows or stops the rule writing where it has
# already written, which the minimal rule does not do at any rate: its update magnitude
# is set by the normalised modulator and trace, not by how good the policy is. Selected
# one at a time rather than composed, since two brakes at once give a result attributable
# to neither.
ConsolidationName = Literal["none", "anchor", "rigidity", "oracle"]
# Decorrelating terms. The minimal rule potentiates where pre- and post-synaptic
# activity agree in sign and depresses where they disagree, so on a network that is
# mostly excitatory the dominant loop is positive feedback. These oppose it: one by
# learning with the opposite sign at synapses whose transmitter is inhibitory, the
# other by the classic normalisation that needs no identity at all.
DecorrelationName = Literal["none", "anti_hebbian_inhibitory", "oja"]
# Where the neuromodulatory third factor reaches. "global" broadcasts one scalar to every
# plastic synapse, which is what every panel so far has run. "pathway" applies it only where
# the wiring carries a modulatory signal -- synapses whose post-synaptic neuron receives
# chemical input from an aminergic neuron -- and leaves the unmodulated Hebbian term
# elsewhere, so the arm is an interpolation between two floors the panels already measured.
ThirdFactorRouting = Literal["global", "pathway"]
# What the eligibility trace carries. "hebbian" is pre x post: the co-activity every panel
# ran, which correlates reward surprise with the network's ordinary activity.
# "node_perturbation" is pre x xi, the part of a unit's output it actually varied, which is
# what makes the update an estimate of the reward gradient rather than reinforced
# correlation.
EligibilityMode = Literal["hebbian", "node_perturbation"]

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
    # 1.0 leaves the gate open at every trailing estimate below it, so a run that
    # selects the oracle pins its own reference rather than inheriting one panel's
    # comparator from a default. An estimate saturated at 1.0 does close it, which
    # is what an unbroken run of successes should mean at any reference.
    plasticity_oracle_reference: float = Field(default=1.0, gt=0.0, le=1.0)
    plasticity_oracle_rate: float = Field(default=0.01, gt=0.0, le=1.0)

    # ── Eligibility (opt-in) ─────────────────────────────────
    # Off by default: "hebbian" is the trace every registered result used. Under
    # "node_perturbation" each plastic unit's PRE-activation is perturbed by
    # N(0, node_noise^2) and the trace carries that perturbation instead of the
    # unit's activity, so the rule estimates the reward gradient over what the
    # network could have done differently. The perturbation changes the forward
    # pass -- the unit must act on it, or the eligibility describes a
    # counterfactual the network never took -- so an arm using it is a new arm.
    plasticity_eligibility: EligibilityMode = "hebbian"
    plasticity_node_noise: float = Field(default=0.0, ge=0.0)

    # The perturbation scale may decay over episodes rather than stay fixed. It has two
    # incompatible jobs: as a probe it must be large enough to move behaviour, or the
    # eligibility is indistinguishable from noise; as jitter on a competent policy it is
    # damage. Nothing requires one value to serve both, so ``plasticity_node_noise`` becomes
    # the INITIAL scale and these two give it a geometric decay to a floor:
    #
    #     sigma(e) = sigma_final                                if e >= anneal_episodes
    #     sigma(e) = sigma_0 * (sigma_final / sigma_0) ** (e / anneal_episodes)
    #
    # Both unset leaves the scale constant and every arm bit-identical. The floor must be
    # positive: the substrates decide at forward time whether they are perturbing by testing
    # the scale against zero, and a scale that reached zero would return the trace to
    # pre x post -- the Hebbian eligibility -- rather than silencing it.
    plasticity_node_noise_final: float | None = Field(default=None, ge=0.0)
    plasticity_node_noise_anneal_episodes: int | None = Field(default=None, gt=0)

    # ── Third-factor routing (opt-in) ────────────────────────
    # Off by default: "global" is the broadcast scalar every panel has run. "pathway" needs a
    # substrate that derives an instructive pathway, so it is refused on the dense yardstick and
    # wherever transmitter identities are absent, and it is refused with the unmodulated rule,
    # where the modulator is already 1.0 everywhere and routing would be a no-op wearing the
    # name of an arm.
    third_factor: ThirdFactorRouting = "global"

    # ── Decorrelation (opt-in) ───────────────────────────────
    # ``anti_hebbian_inhibitory`` negates the Hebbian term at synapses whose
    # grounded sign is inhibitory, so co-activity strengthens what such a
    # synapse does rather than unwinding it; the term's magnitude is unchanged,
    # so the variant redirects the update rather than resizing it. It keys on
    # transmitter identity and is refused where the signs were drawn rather than
    # grounded -- negating an arbitrary sign would be negating a coin flip.
    # ``oja`` subtracts the classic normalisation term, opposing growth in
    # proportion to how active a post-synaptic unit is and how large the weight
    # already is; it needs no identity and so runs on any substrate.
    plasticity_decorrelation: DecorrelationName = "none"
    plasticity_oja_coefficient: float = Field(default=0.0, ge=0.0)

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
    def _validate_eligibility(self) -> PlasticityConfigMixin:
        """Reject an eligibility mode with nothing to build a trace from.

        With a zero perturbation the trace is identically zero, and the arm would look like a
        rule that learns nothing when it is a rule that was never given anything to learn
        from. Whether the substrate can perturb at all is the rule's check, not this one:
        this mixin is shared with substrates whose topologies differ.
        """
        if self.plasticity_eligibility == "node_perturbation" and self.plasticity_node_noise == 0.0:
            msg = (
                "plasticity_eligibility='node_perturbation' requires a positive "
                "plasticity_node_noise: with no perturbation the eligibility is identically "
                "zero, and the arm would look like a rule that learns nothing rather than one "
                "given nothing to learn from."
            )
            raise ValueError(msg)
        return self

    def node_noise_schedule(self) -> NodeNoiseSchedule | None:
        """Build the perturbation schedule this config describes, or ``None`` for a fixed scale."""
        if self.plasticity_node_noise_final is None:
            return None
        if self.plasticity_node_noise_anneal_episodes is None:
            return None
        return NodeNoiseSchedule(
            initial=self.plasticity_node_noise,
            final=self.plasticity_node_noise_final,
            episodes=self.plasticity_node_noise_anneal_episodes,
        )

    @model_validator(mode="after")
    def _validate_node_noise_schedule(self) -> PlasticityConfigMixin:
        """Reject a perturbation schedule that is half stated, inverted, or reaches zero.

        Either bound alone leaves the shape undefined, and a floor at or above the initial
        scale is not a decay. A floor of zero is refused for a substrate reason: perturbing
        is decided at forward time by testing the scale against zero, and where a topology is
        not perturbing the trace carries post-synaptic activity, so a schedule reaching zero
        would silently restore the Hebbian eligibility instead of silencing the arm.
        """
        final = self.plasticity_node_noise_final
        episodes = self.plasticity_node_noise_anneal_episodes
        if (final is None) != (episodes is None):
            msg = (
                "plasticity_node_noise_final and plasticity_node_noise_anneal_episodes must "
                "be set together: one alone does not define a schedule."
            )
            raise ValueError(msg)
        if final is None:
            return self
        if final == 0.0:
            msg = (
                "plasticity_node_noise_final must be positive: a scale of zero does not "
                "silence the eligibility, it returns it to pre-synaptic times post-synaptic "
                "activity, which is the eligibility the perturbation replaces."
            )
            raise ValueError(msg)
        if final >= self.plasticity_node_noise:
            msg = (
                f"plasticity_node_noise_final ({final}) must be below plasticity_node_noise "
                f"({self.plasticity_node_noise}): the schedule is a decay."
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _validate_third_factor_routing(self) -> PlasticityConfigMixin:
        """Reject routing a modulator that is already constant.

        Under the unmodulated rule the third factor is 1.0 at every synapse, so routing it
        changes nothing; a config that names the routed arm and runs the plain Hebbian floor
        would have its runs read as evidence about routing.
        """
        if self.third_factor == "pathway" and self.learning_rule in UNMODULATED_RULES:
            msg = (
                f"third_factor='pathway' has no effect under learning_rule="
                f"{self.learning_rule!r}: that rule applies no neuromodulatory factor, so the "
                "routed arm would be the unmodulated Hebbian floor under another name."
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _validate_decorrelation(self) -> PlasticityConfigMixin:
        """Reject a decorrelating term its own coefficient leaves inert.

        The grounded-signs requirement of ``anti_hebbian_inhibitory`` is not
        checked here: this mixin is shared with substrates that have no
        transmitter identities at all, so the check belongs where the identities
        do.
        """
        if self.plasticity_decorrelation == "oja" and self.plasticity_oja_coefficient == 0.0:
            msg = (
                "plasticity_decorrelation='oja' is inert with plasticity_oja_coefficient=0: "
                "the term would be selected and never act, and its runs would be read as "
                "evidence about it."
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
