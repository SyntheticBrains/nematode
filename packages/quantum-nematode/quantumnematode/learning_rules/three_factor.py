"""Reward-modulated Hebbian plasticity over any plastic topology.

The update is the minimal three-factor form::

    dw = eta * delta * E - eta * lambda_w * w ,   delta = r - b

with the three factors being pre-synaptic activity and post-synaptic
activity (jointly, through the eligibility trace ``E`` the topology
accumulates) and a global neuromodulatory signal ``δ``. Nothing else
enters: no backward pass, no optimiser, no value head, no per-synapse
error signal. That locality is the whole point — it is the property a
gradient method cannot claim, and the reason this rule family can be
asked whether a wiring diagram is legible to a learner the animal could
plausibly host.

Two choices in here are load-bearing enough to state at the top.

**The modulator is a prediction error, not a reward.** ``b`` is an
exponential moving average of observed reward, so ``δ`` measures
surprise. Without it, a task whose rewards are predominantly one-signed
— which is the normal case once step and proximity penalties are in play
— would drive every eligible synapse in one direction regardless of
which behaviour earned the reward, making the rule a decay term with
extra arithmetic.

**Synapse signs are deliberately unconstrained.** Forbidding a synapse
from crossing between excitatory and inhibitory would be the right
restriction if signs carried neurotransmitter identity. In this
substrate they do not: initial chemical weights are drawn from a
zero-mean distribution, so each sign is an arbitrary draw. Clamping it
would preserve noise and would stop the rule from correcting a synapse
whose initial sign was simply wrong.

Two optional scalings make the rate mean the same thing everywhere. Raw,
``eta`` is an absolute step in units that differ by orders of magnitude
between substrates (a dense layer's trace is far smaller per weight than
a sparse recurrent matrix's) and between steps (a terminal penalty can be
two hundred times an ordinary prediction error). With the modulator
normalised the third factor is ``tanh(delta / sigma) - c``, ``sigma`` a
running RMS of the prediction error and ``c`` the running mean of the
compressed value: bounded, sign-preserving, and zero-mean, which a
bounded function of a skewed prediction error is not on its own. With
the trace normalised the Hebbian term is divided by ``rho``, a running
RMS of each tensor's trace over its edge set, so ``eta`` is the
root-mean-square step per unit modulator on every substrate. Both are
off by default and, off, the rule is bit-identical to the raw form.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch

from quantumnematode.brain.arch._rule import RuleStepReport

if TYPE_CHECKING:
    from quantumnematode.brain.arch._brain import BrainHistoryData
    from quantumnematode.brain.arch._topology import BrainTopology, PlasticTopology

# Telemetry keys, shared with the brain that records them.
PREDICTION_ERROR_KEY = "plasticity_prediction_error"
BASELINE_KEY = "plasticity_baseline"
MEAN_ABS_DELTA_KEY = "plasticity_mean_abs_delta"
SATURATED_FRACTION_KEY = "plasticity_saturated_fraction"
MODULATOR_KEY = "plasticity_modulator"
MODULATOR_SCALE_KEY = "plasticity_modulator_scale"
MODULATOR_CENTRE_KEY = "plasticity_modulator_centre"
TRACE_SCALE_KEY = "plasticity_trace_scale"
NORM_DRIFT_KEY = "plasticity_norm_drift"
RATE_MULTIPLIER_KEY = "plasticity_rate_multiplier"
ANCHOR_DEPARTURE_KEY = "plasticity_anchor_departure"
RIGIDITY_KEY = "plasticity_rigidity"
DECORRELATION_SHARE_KEY = "plasticity_decorrelation_share"
INSTRUCTED_FRACTION_KEY = "plasticity_instructed_fraction"
INSTRUCTED_SHARE_KEY = "plasticity_instructed_share"


@dataclass
class ThreeFactorBatch:
    """Experience for one plasticity step.

    A single scalar: the reward earned by the action just taken. The
    eligibility it gates lives on the topology, which is why this carries
    so much less than the PPO rule's batch.
    """

    reward: float


@dataclass(frozen=True)
class ScalingOptions:
    """The rule's substrate-invariant scaling switches.

    Both switches off reproduces the raw rule bit for bit. ``scale_rate`` is
    the EMA rate of both running scales; ``scale_floor`` sits under both
    before any division.
    """

    normalise_modulator: bool = False
    normalise_trace: bool = False
    scale_rate: float = 0.01
    scale_floor: float = 1e-6

    def __post_init__(self) -> None:
        """Hold a direct construction to the bounds the brain configs enforce at load."""
        # The brain configs bound these at load; a direct construction must be
        # held to the same bounds, since a zero rate divides the bias correction
        # by zero and a non-positive floor cannot floor anything.
        if not 0.0 < self.scale_rate <= 1.0:
            msg = f"scale_rate must be in (0, 1], got {self.scale_rate}"
            raise ValueError(msg)
        if self.scale_floor <= 0.0:
            msg = f"scale_floor must be positive, got {self.scale_floor}"
            raise ValueError(msg)


@dataclass(frozen=True)
class ConsolidationOptions:
    """The rule's consolidation mechanism and its parameters.

    ``mechanism`` selects one brake or none; the parameters belonging to the
    others are ignored. ``none`` reproduces the rule without consolidation bit
    for bit and allocates nothing.

    ``anchor`` holds a slow moving average of each plastic tensor's own weights
    and adds a restoring term toward it, so a steady departure builds a force
    against itself. ``rigidity`` grows a per-synapse protective variable where a
    positive modulator met a large trace and divides the Hebbian rate by
    ``1 + strength * c``. ``oracle`` scales the rate by how far a trailing
    episode-success rate sits below ``oracle_reference``; it reads a flag the
    environment supplies rather than the reward stream the rule observes, so it
    bounds what a quality-gated brake could do and is not a mechanism a nervous
    system could host.
    """

    mechanism: str = "none"
    anchor_rate: float = 0.0
    anchor_stiffness: float = 0.0
    rigidity_growth: float = 0.0
    rigidity_decay: float = 0.0
    rigidity_strength: float = 0.0
    oracle_reference: float = 1.0
    oracle_rate: float = 0.01

    def __post_init__(self) -> None:
        """Hold a direct construction to the bounds the brain configs enforce at load."""
        if self.mechanism not in {"none", "anchor", "rigidity", "oracle"}:
            msg = f"unknown consolidation mechanism {self.mechanism!r}"
            raise ValueError(msg)
        if not 0.0 <= self.anchor_rate < 1.0:
            msg = f"anchor_rate must be in [0, 1), got {self.anchor_rate}"
            raise ValueError(msg)
        if not 0.0 <= self.rigidity_decay < 1.0:
            msg = f"rigidity_decay must be in [0, 1), got {self.rigidity_decay}"
            raise ValueError(msg)
        if not 0.0 < self.oracle_reference <= 1.0:
            msg = f"oracle_reference must be in (0, 1], got {self.oracle_reference}"
            raise ValueError(msg)
        if not 0.0 < self.oracle_rate <= 1.0:
            msg = f"oracle_rate must be in (0, 1], got {self.oracle_rate}"
            raise ValueError(msg)
        for name in ("anchor_stiffness", "rigidity_growth", "rigidity_strength"):
            if getattr(self, name) < 0.0:
                msg = f"{name} must be non-negative, got {getattr(self, name)}"
                raise ValueError(msg)


@dataclass(frozen=True)
class DecorrelationOptions:
    """The rule's decorrelating term and its coefficient.

    The minimal update potentiates a synapse where pre- and post-synaptic
    activity agree in sign and depresses it where they disagree, so on a network
    that is mostly excitatory the dominant loop is positive feedback, checked
    only by the bound and whatever normalisation is on. These oppose it.

    ``anti_hebbian_inhibitory`` negates the Hebbian term wherever the synapse's
    grounded sign is inhibitory, leaving grounded excitatory and ungrounded
    synapses alone. Co-activity then strengthens what such a synapse does rather
    than unwinding it. The magnitude is untouched, so the variant redirects the
    update rather than resizing it, and it keys on transmitter identity: without
    grounded signs there is no inhibitory synapse to key on.

    ``oja`` subtracts ``eta * gamma * y**2 * w``, the classic normalisation:
    growth is opposed in proportion to how active the post-synaptic unit is and
    how large the weight already is. It needs no identity and runs anywhere.
    """

    mechanism: str = "none"
    oja_coefficient: float = 0.0

    def __post_init__(self) -> None:
        """Hold a direct construction to the bounds the brain configs enforce at load."""
        if self.mechanism not in {"none", "anti_hebbian_inhibitory", "oja"}:
            msg = f"unknown decorrelation mechanism {self.mechanism!r}"
            raise ValueError(msg)
        if not math.isfinite(self.oja_coefficient):
            # Checked before the comparisons below: NaN fails every ordering test, so a
            # NaN coefficient would pass both of them and then poison every weight it
            # touched on the first step.
            msg = f"oja_coefficient must be finite, got {self.oja_coefficient}"
            raise ValueError(msg)
        if self.oja_coefficient < 0.0:
            msg = f"oja_coefficient must be non-negative, got {self.oja_coefficient}"
            raise ValueError(msg)
        if self.mechanism == "oja" and self.oja_coefficient == 0.0:
            msg = "oja_coefficient must be positive when the oja term is selected"
            raise ValueError(msg)


class _RunningScale:
    """A bias-corrected running root-mean-square.

    Adam-style correction: dividing the moving average by ``1 - (1 - r)^t``
    after ``t`` observations makes the first observation count fully and
    every later estimate a properly weighted average, instead of one
    anchored at zero for the first hundred steps. ``scale_for`` returns the
    scale a step should be measured against -- the estimate from BEFORE
    that step is absorbed, or the observation itself the first time.
    """

    def __init__(self, rate: float, floor: float) -> None:
        self.rate = rate
        self.floor = floor
        self._ema = 0.0
        self._count = 0

    @property
    def count(self) -> int:
        return self._count

    def current(self) -> float | None:
        """Return the bias-corrected RMS estimate, or ``None`` before any observation."""
        if self._count == 0:
            return None
        corrected = self._ema / (1.0 - (1.0 - self.rate) ** self._count)
        return math.sqrt(corrected)

    def scale_for(self, mean_square: float) -> float:
        """Return the floored scale to measure this observation against, before absorbing it."""
        estimate = self.current()
        value = math.sqrt(mean_square) if estimate is None else estimate
        return max(value, self.floor)

    def absorb(self, mean_square: float) -> None:
        self._ema = (1.0 - self.rate) * self._ema + self.rate * mean_square
        self._count += 1

    def floored(self) -> float:
        """Return the current estimate floored, or the floor itself before any observation."""
        estimate = self.current()
        return self.floor if estimate is None else max(estimate, self.floor)


class _RunningMean:
    """A bias-corrected running mean from a zero prior.

    The same Adam-style correction as the scales, but the prior is zero rather
    than the first observation: a prediction error is zero-mean a priori, so
    before anything has been seen the centre is zero and the first compressed
    step passes through uncentred.
    """

    def __init__(self, rate: float) -> None:
        self.rate = rate
        self._ema = 0.0
        self._count = 0

    @property
    def count(self) -> int:
        return self._count

    def current(self) -> float:
        """Return the bias-corrected mean, or zero before any observation."""
        if self._count == 0:
            return 0.0
        return self._ema / (1.0 - (1.0 - self.rate) ** self._count)

    def absorb(self, value: float) -> None:
        self._ema = (1.0 - self.rate) * self._ema + self.rate * value
        self._count += 1


def _incoming_norms(weight: torch.Tensor, mask: torch.Tensor, axis: int) -> torch.Tensor:
    """Per-unit norm of the incoming plastic weights over the edge set only."""
    return (weight.detach() * mask.to(weight.dtype)).norm(dim=axis)


def _pooled_drift(drifts: list[tuple[float, int]]) -> float:
    """Pool per-tensor (sum, count) drifts into one mean over every unit; NaN if none."""
    total = sum(count for _, count in drifts)
    if total == 0:
        return math.nan
    return sum(value for value, _ in drifts) / total


class ThreeFactorRule:
    """Reward-modulated Hebbian plasticity on whatever a topology declares plastic.

    Satisfies the ``LearningRule`` Protocol. Owns its hyperparameters and
    a little scalar state (the reward baseline and, when scaling is on, the
    running scales); it owns no optimiser, no critic, and no experience
    buffer.

    The rule reads its substrate through the ``PlasticTopology`` seam --
    the aligned lists of plastic weights, eligibility traces, and edge
    masks -- and never names a substrate's own attributes. That is what
    lets one implementation drive a sparse recurrent connectome and a
    dense feedforward MLP identically, which is what "matched rule" has
    to mean for the comparison between them to be honest.

    Only the seam's plastic weights change. Everything else a substrate
    carries -- sensory gains, a motor readout, biases, action-noise
    parameters -- is left at its initial value, so a difference between
    two arms trained under this rule is attributable to what was plastic
    and how it was wired, not to a differently-fitted periphery.
    """

    def __init__(  # noqa: PLR0913 — mirrors the hyperparameters it caches
        self,
        topology: PlasticTopology,
        *,
        plasticity_rate: float,
        weight_decay: float,
        weight_bound: float,
        baseline_rate: float,
        freeze_updates: bool,
        modulated: bool,
        device: torch.device,
        scaling: ScalingOptions | None = None,
        consolidation: ConsolidationOptions | None = None,
        decorrelation: DecorrelationOptions | None = None,
        pathway_masks: list[torch.Tensor] | None = None,
        homeostasis: bool = False,
        synapse_signs: list[torch.Tensor] | None = None,
        enforce_signs: bool = True,
        eligibility: str = "hebbian",
    ) -> None:
        self._topology = topology
        self.plasticity_rate = plasticity_rate
        self.weight_decay = weight_decay
        self.weight_bound = weight_bound
        self.baseline_rate = baseline_rate
        # Paired-control branch: run the rule's bookkeeping but never write a
        # weight. Honoured here as well as in the gradient rule so the flag
        # means the same thing whichever rule is selected — a "frozen" arm
        # that quietly kept learning would be indistinguishable from a
        # plastic one in its config and very different in its results.
        self.freeze_updates = freeze_updates
        # Whether the neuromodulatory third factor is applied. Off gives the
        # ablation floor: plain co-activity learning, dw = eta * E, with the
        # reward stream observed but never used. It is the comparison that
        # separates "this arm learned something" from "this arm learned
        # something FROM REWARD" — without it, an advantage attributable to
        # the wiring's correlation structure under any Hebbian process would
        # be indistinguishable from reward-driven learning.
        self.modulated = modulated
        self.device = device
        self.scaling = scaling if scaling is not None else ScalingOptions()
        # Consolidation. One brake or none; ``none`` allocates nothing and adds
        # no operation, so the default path is the rule without this feature.
        self.consolidation = consolidation if consolidation is not None else ConsolidationOptions()
        # The decorrelating term, if any. ``none`` adds no operation.
        self.decorrelation = decorrelation if decorrelation is not None else DecorrelationOptions()

        # Running estimate of the task's reward level. Persists across
        # episodes: it describes the task, not one episode, and resetting
        # it per episode would make every episode's opening steps register
        # as surprising regardless of behaviour.
        self.baseline = 0.0
        # The running scales, also persistent across episodes for the same
        # reason. Allocated regardless of the switches (they are cheap) but
        # only ever advanced when their switch is on, so an off switch adds
        # no operation to the raw path.
        self._modulator_scale = _RunningScale(self.scaling.scale_rate, self.scaling.scale_floor)
        self._modulator_centre = _RunningMean(self.scaling.scale_rate)
        self._trace_scales = [
            _RunningScale(self.scaling.scale_rate, self.scaling.scale_floor)
            for _ in topology.plastic_weights
        ]
        # Homeostatic incoming-norm scaling: each unit's incoming plastic
        # weights are held at the norm they had when the rule was built, over
        # the edge set only. This is the multiplicative normalisation of a
        # neuron's synaptic budget -- synaptic scaling -- and it is what stops
        # Hebbian positive feedback from running the weights onto the bound.
        # Targets are captured here, so they are whatever norm the substrate
        # carried at construction (its initialisation, for a freshly built brain).
        # Consolidation state, allocated only for the mechanism selected.
        # The anchor starts at the weights the rule was built over: with the
        # anchor rate at zero that is where it stays, which is the limiting
        # case of the mechanism rather than a separate one.
        self._anchors: list[torch.Tensor] = []
        self._rigidity: list[torch.Tensor] = []
        # Trailing episode-success rate for the oracle gate. Starts at zero, so
        # a run's opening episodes are ungated while the estimate warms up.
        self._success_rate = 0.0
        if self.consolidation.mechanism == "anchor":
            with torch.no_grad():
                self._anchors = [w.detach().clone() for w in topology.plastic_weights]
        elif self.consolidation.mechanism == "rigidity":
            self._rigidity = [torch.zeros_like(w) for w in topology.plastic_weights]

        self.homeostasis = homeostasis
        # Dale's law. One signed matrix per plastic tensor, aligned with the seam's weights:
        # +1 excitatory, -1 inhibitory, 0 for a synapse whose source implies no sign. Empty
        # when enforcement is off, which is the default and leaves every update untouched.
        # One signed matrix per plastic tensor when the substrate's signs are grounded,
        # aligned with the seam's weights: +1 excitatory, -1 inhibitory, 0 where the source
        # implies no sign. Two independent consumers -- Dale's law, which constrains where a
        # weight may go, and the anti-Hebbian variant, which reads the same identities to
        # decide which way an update points -- so the signs are held whenever they exist and
        # ``enforce_signs`` says whether the first of them acts.
        self._synapse_signs: list[torch.Tensor] = list(synapse_signs) if synapse_signs else []
        self.enforce_signs = enforce_signs
        # The instructive pathway, one boolean mask per plastic tensor when the third factor is
        # routed and empty when it is broadcast. Handed in like the sign vector rather than read
        # off a substrate, so the rule still names no substrate. Where the mask is false the
        # modulator is replaced by 1.0 -- the unmodulated Hebbian term, the panel's other
        # registered floor -- so a routed arm is an interpolation between two measured arms.
        self._pathway_masks: list[torch.Tensor] = list(pathway_masks) if pathway_masks else []
        self.instructed_fraction = math.nan
        if self._pathway_masks:
            on = sum(int(m.sum().item()) for m in self._pathway_masks)
            edges = sum(int(m.sum().item()) for m in topology.plastic_masks)
            self.instructed_fraction = on / edges if edges else 0.0
        # The eligibility mode the topology's trace was built under. The rule does not build
        # the trace -- the topology does -- so this is a check that the substrate is actually
        # perturbing, not a switch: a mode selected over a topology that cannot perturb would
        # read a trace of ordinary co-activity while reporting itself a gradient estimator.
        self.eligibility = eligibility
        if eligibility == "node_perturbation" and not topology.plastic_perturbations:
            msg = (
                "eligibility='node_perturbation' needs a topology that perturbs its units: "
                "this one exposes no perturbations, so the trace would carry ordinary "
                "co-activity while the rule reported itself a gradient estimator."
            )
            raise ValueError(msg)
        if self.decorrelation.mechanism == "anti_hebbian_inhibitory" and not self._synapse_signs:
            msg = (
                "anti_hebbian_inhibitory needs grounded synapse signs: without them there is "
                "no inhibitory synapse to key on, only an arbitrary draw to negate."
            )
            raise ValueError(msg)
        # Pre-computed once: -1 where a synapse is grounded inhibitory, +1 elsewhere. The
        # variant is one elementwise product, so selecting it costs a multiply per step.
        self._hebbian_signs: list[torch.Tensor] = []
        if self.decorrelation.mechanism == "anti_hebbian_inhibitory":
            with torch.no_grad():
                self._hebbian_signs = [
                    torch.where(signs < 0, -torch.ones_like(w), torch.ones_like(w))
                    for signs, w in zip(self._synapse_signs, topology.plastic_weights, strict=True)
                ]
        self._fan_in_axes: list[int] = []
        self._norm_targets: list[torch.Tensor] = []
        if homeostasis:
            self._fan_in_axes = list(topology.plastic_fan_in_axes)
            with torch.no_grad():
                for w, mask, axis in zip(
                    topology.plastic_weights,
                    topology.plastic_masks,
                    self._fan_in_axes,
                    strict=True,
                ):
                    self._norm_targets.append(_incoming_norms(w, mask, axis))

    @property
    def modulator_scale(self) -> _RunningScale:
        """The running scale of the prediction error (advances only when its switch is on)."""
        return self._modulator_scale

    @property
    def modulator_centre(self) -> _RunningMean:
        """The running mean the compressed modulator is centred by (advances only when on)."""
        return self._modulator_centre

    @property
    def trace_scales(self) -> list[_RunningScale]:
        """One running trace scale per plastic tensor (advance only when the switch is on)."""
        return self._trace_scales

    @property
    def norm_targets(self) -> list[torch.Tensor]:
        """Per plastic tensor, each unit's target incoming norm (empty when homeostasis is off)."""
        return self._norm_targets

    def _norm_drift(
        self,
        index: int,
        weight: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[float, int]:
        """Return the summed relative norm deviation and the count of units with a target.

        Returned as a sum and a count rather than a mean so the telemetry pools every unit
        across tensors equally: a mean of per-tensor means would let a two-unit output layer
        weigh as much as a sixty-four-unit hidden layer.
        """
        target = self._norm_targets[index]
        has_target = target > 0
        count = int(has_target.sum().item())
        if count == 0:
            return 0.0, 0
        norms = _incoming_norms(weight, mask, self._fan_in_axes[index])
        return float((norms[has_target] / target[has_target] - 1.0).abs().sum().item()), count

    def _rescale_incoming(self, index: int, weight: torch.Tensor, mask: torch.Tensor) -> None:
        """Return each unit's incoming norm to its target, touching masked entries only."""
        axis = self._fan_in_axes[index]
        target = self._norm_targets[index]
        norms = _incoming_norms(weight, mask, axis)
        # Divide by the actual norm whenever there is one, however small, so the
        # target is restored exactly. A unit whose incoming weights are all zero
        # has no direction to scale along; inventing one would write weights no
        # learning update produced, so it is left as it is.
        restorable = (target > 0) & (norms > 0)
        safe_norms = torch.where(restorable, norms, torch.ones_like(norms))
        factor = torch.where(restorable, target / safe_norms, torch.ones_like(norms))
        # Broadcast the per-unit factor along the fan-in axis, and apply it on
        # the edge set only: off-edge entries are multiplied by exactly one.
        factor = factor.unsqueeze(axis)
        weight.data.mul_(1.0 + mask.to(weight.dtype) * (factor - 1.0))

    def _modulator(self, delta: float) -> tuple[float, float, float]:
        """Return the third factor and the scale and centre it was measured against (NaN if off)."""
        if not self.scaling.normalise_modulator:
            return (delta if self.modulated else 1.0), math.nan, math.nan
        # The scale is estimated BEFORE this step's error is absorbed, the same
        # convention as the baseline: a surprising step is scored as surprising,
        # not against a scale it has already inflated. The first observation
        # counts fully (the bias-corrected estimate is then that observation).
        squared = delta * delta
        sigma = self._modulator_scale.scale_for(squared)
        self._modulator_scale.absorb(squared)
        compressed = math.tanh(delta / sigma)
        # Compression breaks the zero mean the baseline gave delta: a bounded
        # function maps rare large negatives and frequent moderate positives to
        # the same +-1, so the frequent side wins and the modulator acquires a
        # constant offset -- a reward-blind Hebbian drive. Subtracting the
        # running mean of the compressed value restores the zero mean a
        # prediction error must have. Zero prior, pre-update value.
        centre = self._modulator_centre.current()
        self._modulator_centre.absorb(compressed)
        # Computed and reported even when unmodulated, so both arms record the
        # scale and centre of the surprise they saw and only one records having
        # used it.
        modulator = (compressed - centre) if self.modulated else 1.0
        return modulator, sigma, centre

    def _trace_divisors(
        self,
        traces: list[torch.Tensor],
        masks: list[torch.Tensor],
    ) -> tuple[list[float] | None, float]:
        """Per-tensor trace scales for this step (None when off) and their mean for telemetry."""
        if not self.scaling.normalise_trace:
            return None, math.nan
        divisors: list[float] = []
        for trace, mask, scale in zip(traces, masks, self._trace_scales, strict=True):
            # Over the edge set only: off-edge zeros would dilute a sparse
            # substrate's scale by its sparsity.
            on_edges = trace[mask.to(torch.bool)]
            mean_square = float(on_edges.square().mean().item()) if on_edges.numel() else 0.0
            if mean_square > 0.0:
                divisors.append(scale.scale_for(mean_square))
                scale.absorb(mean_square)
            else:
                # An all-zero trace (every episode's first step, by the trace's
                # design) neither updates the estimate nor counts toward its
                # correction; the Hebbian term it would scale is zero anyway.
                divisors.append(scale.floored())
        return divisors, float(sum(divisors) / len(divisors)) if divisors else math.nan

    def step(
        self,
        topology: BrainTopology,
        batch: Any,  # noqa: ANN401 — LearningRule Protocol shape (rule-specific batch)
    ) -> RuleStepReport:
        """Apply one reward-modulated Hebbian update.

        Runs entirely under ``torch.no_grad()``: this rule has no backward
        pass, and the trace it reads was accumulated outside the autograd
        graph, so nothing here should be recording operations.
        """
        if topology is not self._topology:
            msg = (
                "ThreeFactorRule.step received a topology other than the one it "
                "was constructed over. The rule reads that topology's eligibility "
                "traces, so updating a different one would apply credit assigned "
                "from unrelated activity — construct a new rule for a new topology."
            )
            raise ValueError(msg)
        topo = self._topology
        if not topo.enable_activity_traces:
            msg = (
                "ThreeFactorRule requires activity traces, but the topology has "
                "none allocated. Without a trace the update is identically zero "
                "and training would silently do nothing."
            )
            raise ValueError(msg)

        reward = cast("ThreeFactorBatch", batch).reward

        with torch.no_grad():
            # Third factor: reward surprise, evaluated BEFORE the baseline
            # absorbs this reward, so a step is scored against what was
            # expected of it rather than against a level it has already
            # shifted.
            delta = reward - self.baseline
            self.baseline += self.baseline_rate * delta
            modulator, modulator_scale, modulator_centre = self._modulator(delta)

            weights = topo.plastic_weights
            traces = topo.eligibility_traces
            masks = topo.plastic_masks
            # The trace scales advance even under a freeze, like the baseline:
            # a frozen arm must report what the plastic arm would, or the two
            # stop being comparable step for step.
            divisors, trace_scale = self._trace_divisors(traces, masks)
            # Snapshot before writing: the reported weight change must be
            # what the weights ACTUALLY did, not what the update proposed.
            # Once entries reach the magnitude bound the clamp discards the
            # proposal entirely, and reporting the proposal would show a
            # healthy learning signal for a rule that has become a constant
            # function — defeating the one telemetry meant to detect exactly
            # that.
            befores = [w.detach().clone() for w in weights]
            drifts: list[tuple[float, int]] = []

            # Oracle gate, if selected: one factor on the whole update, so at
            # the reference success rate nothing is written at all.
            multiplier = self._rate_multiplier()
            rate = self.plasticity_rate * multiplier

            rate_multiplier, anchor_departure, rigidity_mean = self._consolidation_telemetry(
                weights,
                masks,
                multiplier,
            )
            instructed_share = self._instructed_share(weights, traces, masks, modulator, rate)
            decorrelation_share = self._decorrelation_share(
                weights,
                traces,
                masks,
                modulator,
                rate,
                divisors,
            )

            if not self.freeze_updates:
                for index, (w, trace, mask) in enumerate(zip(weights, traces, masks, strict=True)):
                    update = self._proposed_update(
                        index,
                        w,
                        trace,
                        modulator,
                        rate,
                        divisors[index] if divisors is not None else None,
                    )
                    # The trace is already on the edge set, but the decay
                    # term is not: it is proportional to the weights, which
                    # may hold values off the allowed edges (the connectome's
                    # soft-prior mode). Masking keeps the rule from writing
                    # anywhere the substrate says there is no synapse; on a
                    # dense substrate the mask is all-true and this is a
                    # no-op by construction.
                    update = update * mask.to(update.dtype)
                    w.data.add_(update)
                    # Dale's law, before the homeostatic rescale so the rescale acts on
                    # permitted weights; it multiplies by a positive per-unit factor and
                    # cannot reintroduce a forbidden sign, and the clamp still comes last.
                    self._project_signs(index, w)
                    if self.homeostasis:
                        # Drift is measured after the update and before the
                        # rescale; the clamp comes last so the bound always holds.
                        drifts.append(self._norm_drift(index, w, mask))
                        self._rescale_incoming(index, w, mask)
                    w.data.clamp_(-self.weight_bound, self.weight_bound)
                    self._advance_consolidation(
                        index,
                        w,
                        trace,
                        mask,
                        modulator,
                        divisors[index] if divisors is not None else None,
                    )
            elif self.homeostasis:
                # Nothing is written under a freeze; the drift of the unchanged
                # weights is still reported so the arms stay comparable.
                drifts = [
                    self._norm_drift(index, w, mask)
                    for index, (w, mask) in enumerate(zip(weights, masks, strict=True))
                ]
            if self.freeze_updates:
                # Consolidation state advances under a freeze, like the
                # baseline and the running scales: a frozen arm must report
                # what the plastic arm would. No weight is touched by it.
                for index, (w, trace, mask) in enumerate(zip(weights, traces, masks, strict=True)):
                    self._advance_consolidation(
                        index,
                        w,
                        trace,
                        mask,
                        modulator,
                        divisors[index] if divisors is not None else None,
                    )

            # Under a freeze nothing is written at all — not the Hebbian
            # term, not the decay, not the clamp. A clamp alone would still
            # edit a weight that started outside the bound, which is
            # precisely the silent substrate change a frozen control exists
            # to avoid. Reporting continues, so the control stays comparable
            # step-for-step against the plastic arm.

            # Effective change, aggregated over every plastic entry of every
            # tensor. Reduced in-tensor rather than as a Python-float ratio so
            # a single-tensor substrate reports bit-for-bit what it did
            # before the rule became generic.
            deltas = [
                (w.detach() - b).abs().reshape(-1) for w, b in zip(weights, befores, strict=True)
            ]
            mean_abs_delta = torch.cat(deltas).mean().item()
            # Saturated fraction over real synapses only: off-edge entries are
            # not synapses and would dilute the signal by the sparsity of the
            # substrate. Dense substrates count every entry.
            edge_count = sum(int(m.sum().item()) for m in masks)
            if edge_count == 0:  # pragma: no cover — a substrate with no plastic entries
                saturated = 0.0
            else:
                at_bound = sum(
                    int(((w.detach().abs() >= self.weight_bound) & m).sum().item())
                    for w, m in zip(weights, masks, strict=True)
                )
                saturated = at_bound / edge_count

        return RuleStepReport(
            extra={
                PREDICTION_ERROR_KEY: delta,
                BASELINE_KEY: self.baseline,
                MEAN_ABS_DELTA_KEY: mean_abs_delta,
                SATURATED_FRACTION_KEY: saturated,
                MODULATOR_KEY: modulator,
                MODULATOR_SCALE_KEY: modulator_scale,
                MODULATOR_CENTRE_KEY: modulator_centre,
                TRACE_SCALE_KEY: trace_scale,
                NORM_DRIFT_KEY: _pooled_drift(drifts),
                RATE_MULTIPLIER_KEY: rate_multiplier,
                ANCHOR_DEPARTURE_KEY: anchor_departure,
                RIGIDITY_KEY: rigidity_mean,
                DECORRELATION_SHARE_KEY: decorrelation_share,
                INSTRUCTED_FRACTION_KEY: self.instructed_fraction,
                INSTRUCTED_SHARE_KEY: instructed_share,
            },
        )

    def _proposed_update(  # noqa: PLR0913 — one term per factor of the update
        self,
        index: int,
        weight: torch.Tensor,
        trace: torch.Tensor,
        modulator: float,
        rate: float,
        divisor: float | None,
    ) -> torch.Tensor:
        """Compute the unmasked weight change for one plastic tensor.

        Hebbian term over the eligibility trace, plus decay toward zero — decay
        is what keeps unreinforced synapses from holding whatever a transient
        correlation put there — plus, when the anchor is selected, a restoring
        force toward it.
        """
        if self._rigidity:
            # Rigidity divides the Hebbian rate per synapse, at the value from
            # before this step's growth: a step is not charged for the rigidity
            # it is about to create.
            hebbian_rate = rate / (
                1.0 + self.consolidation.rigidity_strength * self._rigidity[index]
            )
        else:
            hebbian_rate = rate
        if self._pathway_masks:
            # Routed: the modulator reaches the instructed synapses; everywhere else the third
            # factor is 1.0, which is the unmodulated rule's update at those synapses.
            gated = torch.where(
                self._pathway_masks[index],
                torch.full_like(trace, modulator),
                torch.ones_like(trace),
            )
            update = hebbian_rate * gated * trace
        else:
            update = hebbian_rate * modulator * trace
        if divisor is not None:
            update = update / divisor
        if self._hebbian_signs:
            # Negated at grounded inhibitory synapses, so co-activity strengthens what such
            # a synapse does rather than unwinding it. Elementwise +-1, so the term's
            # magnitude is untouched: the variant redirects the update, never resizes it.
            update = update * self._hebbian_signs[index]
        update = update - rate * self.weight_decay * weight
        if self.decorrelation.mechanism == "oja":
            # Growth opposed in proportion to how active the post-synaptic unit is and how
            # large the weight already is. Broadcast along the post-synaptic axis, which is
            # the one the fan-in axis is not.
            post = self._post_activity(index)
            update = update - rate * self.decorrelation.oja_coefficient * post.square() * weight
        if self._anchors:
            # Unlike decay, this is not a uniform per-unit shrink, so the
            # homeostatic rescale cancels only its radial part and leaves the
            # change of direction it made.
            update = update - (
                rate * self.consolidation.anchor_stiffness * (weight - self._anchors[index])
            )
        return update

    def _consolidation_telemetry(
        self,
        weights: list[torch.Tensor],
        masks: list[torch.Tensor],
        multiplier: float,
    ) -> tuple[float, float, float]:
        """Measure the rate multiplier applied, the anchor departure and the protective variable.

        Measured on the state the step is taken at rather than the state it
        leaves behind — the same pre-update convention the modulator scale and
        the trace scales follow, and the only one under which the reported
        multiplier is the multiplier actually applied. NaN where a mechanism is
        not selected, as the scaling telemetry reports an estimator that never
        ran.
        """
        if self._anchors:
            departure = self._edge_mean(
                [(w.detach() - a).abs() for w, a in zip(weights, self._anchors, strict=True)],
                masks,
            )
            return multiplier, departure, math.nan
        if self._rigidity:
            divisors = [
                1.0 / (1.0 + self.consolidation.rigidity_strength * c) for c in self._rigidity
            ]
            return (
                self._edge_mean(divisors, masks),
                math.nan,
                self._edge_mean(self._rigidity, masks),
            )
        return multiplier, math.nan, math.nan

    def _instructed_share(
        self,
        weights: list[torch.Tensor],
        traces: list[torch.Tensor],
        masks: list[torch.Tensor],
        modulator: float,
        rate: float,
    ) -> float:
        """How the step's Hebbian magnitude split between the instructed set and the rest.

        Measured on the effective update, like the decorrelation share. Reported as the whole
        under a broadcast third factor, since every synapse is then instructed in the only sense
        that applies. This describes the split; that the *modulated* part reaches only the
        instructed set is asserted by test against the global and unmodulated rules, entry for
        entry, not by this number.
        """
        if not self._pathway_masks:
            return 1.0
        del weights
        total = 0.0
        part = 0.0
        for index, (trace, mask) in enumerate(zip(traces, masks, strict=True)):
            edges = mask.to(torch.bool)
            gated = torch.where(
                self._pathway_masks[index],
                torch.full_like(trace, modulator),
                torch.ones_like(trace),
            )
            magnitude = (rate * gated * trace).abs()
            total += float(magnitude[edges].sum().item())
            part += float(magnitude[self._pathway_masks[index] & edges].sum().item())
        return part / total if total > 0.0 else 0.0

    def _decorrelation_share(  # noqa: PLR0913 — the factors the update itself is built from
        self,
        weights: list[torch.Tensor],
        traces: list[torch.Tensor],
        masks: list[torch.Tensor],
        modulator: float,
        rate: float,
        divisors: list[float] | None,
    ) -> float:
        """Share of the update's absolute magnitude the decorrelating term accounts for.

        Measured on the components the update is actually built from — the same
        rate, modulator, rigidity divisor and trace scale ``_proposed_update``
        applies — because a ratio of raw traces to a scaled Oja term compares
        quantities in different units and would misreport how much of the step
        each accounts for.

        Under the anti-Hebbian variant the magnitude is unchanged by construction,
        so this is the share of it that was *redirected*: the negated subset's
        weight in the Hebbian term's total. Under the Oja term it is that term's
        own contribution to the two together. Zero when no term is selected, and
        zero when the step carries no effective modulation at all.
        """
        mechanism = self.decorrelation.mechanism
        if mechanism == "none":
            return 0.0
        hebbian_total = 0.0
        part = 0.0
        for index, (w, trace, mask) in enumerate(zip(weights, traces, masks, strict=True)):
            edges = mask.to(torch.bool)
            hebbian_rate = (
                rate / (1.0 + self.consolidation.rigidity_strength * self._rigidity[index])
                if self._rigidity
                else rate
            )
            hebbian = (hebbian_rate * modulator * trace).abs()
            if divisors is not None:
                hebbian = hebbian / divisors[index]
            hebbian_total += float(hebbian[edges].sum().item())
            if mechanism == "anti_hebbian_inhibitory":
                part += float(hebbian[(self._hebbian_signs[index] < 0) & edges].sum().item())
            else:
                oja = (
                    (rate * self.decorrelation.oja_coefficient)
                    * self._post_activity(index).square()
                    * w.detach().abs()
                )
                part += float(oja[edges].sum().item())
        denominator = hebbian_total + (part if mechanism == "oja" else 0.0)
        return part / denominator if denominator > 0.0 else 0.0

    def _post_activity(self, index: int) -> torch.Tensor:
        """Shape this tensor's post-synaptic activity to broadcast along its post axis.

        The seam gives one vector per plastic tensor, indexed along the axis the
        fan-in axis is not; a 2-D weight therefore needs it as a row when the
        fan-in axis is 0 and as a column when it is 1.
        """
        post = self._topology.plastic_post_activities[index]
        fan_in = self._topology.plastic_fan_in_axes[index]
        return post.reshape(1, -1) if fan_in == 0 else post.reshape(-1, 1)

    def _rate_multiplier(self) -> float:
        """Global rate factor from the oracle gate; ``1.0`` under every other mechanism."""
        if self.consolidation.mechanism != "oracle":
            return 1.0
        ratio = self._success_rate / self.consolidation.oracle_reference
        return float(min(max(1.0 - ratio, 0.0), 1.0))

    def _advance_consolidation(  # noqa: PLR0913 — the step it is advanced from
        self,
        index: int,
        weight: torch.Tensor,
        trace: torch.Tensor,
        mask: torch.Tensor,
        modulator: float,
        divisor: float | None,
    ) -> None:
        """Advance the selected mechanism's state from the step just taken.

        After the write, for the same reason the running scales are estimated
        before it: each quantity is measured against the step it actually saw.
        Advanced under a freeze as well, like the baseline and the scales, so a
        frozen arm reports what the plastic arm would.
        """
        if self._anchors:
            anchor = self._anchors[index]
            anchor.add_(self.consolidation.anchor_rate * (weight.detach() - anchor))
        elif self._rigidity:
            growth = trace.abs() * (self.consolidation.rigidity_growth * max(modulator, 0.0))
            if divisor is not None:
                # On the trace as the update saw it, so a pinned growth rate is
                # the same root-mean-square growth on every substrate.
                growth = growth / divisor
            rigidity = self._rigidity[index]
            rigidity.mul_(1.0 - self.consolidation.rigidity_decay)
            rigidity.add_(growth * mask.to(growth.dtype))

    def _edge_mean(self, values: list[torch.Tensor], masks: list[torch.Tensor]) -> float:
        """Mean of a per-synapse quantity over the edge set, NaN when there is none."""
        if not values:
            return math.nan
        total = 0.0
        count = 0
        for value, mask in zip(values, masks, strict=True):
            edges = mask.to(torch.bool)
            on_edges = int(edges.sum().item())
            if on_edges:
                total += float(value[edges].sum().item())
                count += on_edges
        return total / count if count else math.nan

    def _project_signs(self, index: int, weight: torch.Tensor) -> None:
        """Clamp each grounded synapse to its sign; ungrounded synapses are left alone.

        A no-op unless Dale's law is enforced: the signs are held whenever the
        substrate grounds them, because a decorrelating variant reads the same
        identities without constraining any weight.
        """
        if not self.enforce_signs:
            return
        if not self._synapse_signs:
            return
        signs = self._synapse_signs[index]
        weight.data = torch.where(
            signs > 0,
            weight.data.clamp_min(0.0),
            torch.where(signs < 0, weight.data.clamp_max(0.0), weight.data),
        )

    def reset_state(self) -> None:
        """Return every running quantity to its prior; re-anchor homeostasis to the current weights.

        The baseline, the modulator scale, the trace scales and the modulator centre
        restart from their priors, the topology's eligibility traces are cleared, and the
        homeostatic norm targets are recomputed from the weights as they are now. Used
        when weights are loaded into a brain: a warm start begins from better weights,
        not from another run's reward statistics and not from the norms of weights it no
        longer has.
        """
        self.baseline = 0.0
        self._modulator_scale = _RunningScale(self.scaling.scale_rate, self.scaling.scale_floor)
        self._trace_scales = [
            _RunningScale(self.scaling.scale_rate, self.scaling.scale_floor)
            for _ in self._trace_scales
        ]
        self._modulator_centre = _RunningMean(self.scaling.scale_rate)
        self._topology.reset_traces()
        # A warm start begins the perturbation schedule again: the loaded policy is explored
        # from the initial scale, not from wherever the previous run's schedule had reached.
        self._topology.reset_schedule()
        # Consolidation state follows the weights the rule now starts from. An
        # anchor left at a previous substrate's values would pull a loaded
        # policy toward weights it no longer has, with a force proportional to
        # the distance between them -- the failure the homeostatic targets
        # below had before they were re-anchored here.
        if self._anchors:
            with torch.no_grad():
                self._anchors = [w.detach().clone() for w in self._topology.plastic_weights]
        if self._rigidity:
            self._rigidity = [torch.zeros_like(w) for w in self._topology.plastic_weights]
        self._success_rate = 0.0
        # The homeostatic targets are re-anchored to the weights the rule now starts from.
        # Left at their construction values they would drag a loaded policy back to the
        # incoming norms of the random initialisation on the first step.
        if self.homeostasis:
            with torch.no_grad():
                self._norm_targets = [
                    _incoming_norms(w, mask, axis)
                    for w, mask, axis in zip(
                        self._topology.plastic_weights,
                        self._topology.plastic_masks,
                        self._fan_in_axes,
                        strict=True,
                    )
                ]

    def observe_episode(self, *, success: bool | None) -> None:
        """Absorb one episode's success flag into the oracle's trailing estimate.

        Ignored under every other mechanism, and by a rule that never sees a
        flag. The flag is a property of the task's scoring rather than of the
        reward stream this rule observes, which is why the mechanism that reads
        it is a bound on what a quality-gated brake could do and not a
        mechanism a nervous system could host.
        """
        if self.consolidation.mechanism != "oracle" or success is None:
            return
        rate = self.consolidation.oracle_rate
        self._success_rate += rate * (float(success) - self._success_rate)

    @property
    def success_rate(self) -> float:
        """The oracle's trailing episode-success estimate; zero under every other mechanism."""
        return self._success_rate

    def reset_episode(self) -> None:
        """No per-episode rule state to clear.

        The eligibility trace is topology-owned and reset there; the
        baseline and the running scales are deliberately retained across
        episodes. Kept as a documented no-op for the ``LearningRule``
        lifecycle.
        """


def record_plasticity_report(
    history_data: BrainHistoryData,
    report: RuleStepReport,
    reward: float,
) -> None:
    """Append one plasticity step's telemetry to a brain's history.

    Shared by every brain that hosts the rule, so the keys -- and what they
    mean -- cannot drift between the arms a panel compares. Also records
    the reward, matching where the gradient path records it.
    """
    extra = report.extra
    history_data.plasticity_prediction_error.append(extra[PREDICTION_ERROR_KEY])
    history_data.plasticity_baseline.append(extra[BASELINE_KEY])
    history_data.plasticity_mean_abs_delta.append(extra[MEAN_ABS_DELTA_KEY])
    history_data.plasticity_saturated_fraction.append(extra[SATURATED_FRACTION_KEY])
    history_data.plasticity_modulator.append(extra[MODULATOR_KEY])
    history_data.plasticity_modulator_scale.append(extra[MODULATOR_SCALE_KEY])
    history_data.plasticity_modulator_centre.append(extra[MODULATOR_CENTRE_KEY])
    history_data.plasticity_trace_scale.append(extra[TRACE_SCALE_KEY])
    history_data.plasticity_norm_drift.append(extra[NORM_DRIFT_KEY])
    history_data.plasticity_rate_multiplier.append(extra[RATE_MULTIPLIER_KEY])
    history_data.plasticity_anchor_departure.append(extra[ANCHOR_DEPARTURE_KEY])
    history_data.plasticity_rigidity.append(extra[RIGIDITY_KEY])
    history_data.plasticity_decorrelation_share.append(extra[DECORRELATION_SHARE_KEY])
    history_data.plasticity_instructed_fraction.append(extra[INSTRUCTED_FRACTION_KEY])
    history_data.plasticity_instructed_share.append(extra[INSTRUCTED_SHARE_KEY])
    history_data.rewards.append(reward)


# The connectome was the first substrate this rule drove, and existing code
# imports it under that name. The name now describes only where it started.
ConnectomeThreeFactorRule = ThreeFactorRule
