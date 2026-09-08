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
        homeostasis: bool = False,
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
        self.homeostasis = homeostasis
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

            if not self.freeze_updates:
                for index, (w, trace, mask) in enumerate(zip(weights, traces, masks, strict=True)):
                    # Hebbian term over the eligibility trace, plus decay
                    # toward zero. Decay is what keeps unreinforced synapses
                    # from holding whatever a transient correlation put there.
                    update = self.plasticity_rate * modulator * trace
                    if divisors is not None:
                        update = update / divisors[index]
                    update -= self.plasticity_rate * self.weight_decay * w
                    # The trace is already on the edge set, but the decay
                    # term is not: it is proportional to the weights, which
                    # may hold values off the allowed edges (the connectome's
                    # soft-prior mode). Masking keeps the rule from writing
                    # anywhere the substrate says there is no synapse; on a
                    # dense substrate the mask is all-true and this is a
                    # no-op by construction.
                    update = update * mask.to(update.dtype)
                    w.data.add_(update)
                    if self.homeostasis:
                        # Drift is measured after the update and before the
                        # rescale; the clamp comes last so the bound always holds.
                        drifts.append(self._norm_drift(index, w, mask))
                        self._rescale_incoming(index, w, mask)
                    w.data.clamp_(-self.weight_bound, self.weight_bound)
            elif self.homeostasis:
                # Nothing is written under a freeze; the drift of the unchanged
                # weights is still reported so the arms stay comparable.
                drifts = [
                    self._norm_drift(index, w, mask)
                    for index, (w, mask) in enumerate(zip(weights, masks, strict=True))
                ]
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
            },
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
    history_data.rewards.append(reward)


# The connectome was the first substrate this rule drove, and existing code
# imports it under that name. The name now describes only where it started.
ConnectomeThreeFactorRule = ThreeFactorRule
