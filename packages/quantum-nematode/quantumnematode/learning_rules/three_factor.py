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
"""

from __future__ import annotations

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


@dataclass
class ThreeFactorBatch:
    """Experience for one plasticity step.

    A single scalar: the reward earned by the action just taken. The
    eligibility it gates lives on the topology, which is why this carries
    so much less than the PPO rule's batch.
    """

    reward: float


class ThreeFactorRule:
    """Reward-modulated Hebbian plasticity on whatever a topology declares plastic.

    Satisfies the ``LearningRule`` Protocol. Owns its hyperparameters and
    a single scalar of state (the reward baseline); it owns no optimiser,
    no critic, and no experience buffer.

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

        # Running estimate of the task's reward level. Persists across
        # episodes: it describes the task, not one episode, and resetting
        # it per episode would make every episode's opening steps register
        # as surprising regardless of behaviour.
        self.baseline = 0.0

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
            # Computed and reported even when unmodulated, so both arms record
            # what the reward stream was doing and only one records having used
            # it. That makes the ablation visible in a run's own telemetry
            # rather than inferable only from its configuration — which matters
            # for an arm whose entire job is to be a trustworthy reference.
            modulator = delta if self.modulated else 1.0

            weights = topo.plastic_weights
            traces = topo.eligibility_traces
            masks = topo.plastic_masks
            # Snapshot before writing: the reported weight change must be
            # what the weights ACTUALLY did, not what the update proposed.
            # Once entries reach the magnitude bound the clamp discards the
            # proposal entirely, and reporting the proposal would show a
            # healthy learning signal for a rule that has become a constant
            # function — defeating the one telemetry meant to detect exactly
            # that.
            befores = [w.detach().clone() for w in weights]

            if not self.freeze_updates:
                for w, trace, mask in zip(weights, traces, masks, strict=True):
                    # Hebbian term over the eligibility trace, plus decay
                    # toward zero. Decay is what keeps unreinforced synapses
                    # from holding whatever a transient correlation put there.
                    update = self.plasticity_rate * modulator * trace
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
                    w.data.clamp_(-self.weight_bound, self.weight_bound)
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
            },
        )

    def reset_episode(self) -> None:
        """No per-episode rule state to clear.

        The eligibility trace is topology-owned and reset there; the
        baseline is deliberately retained across episodes. Kept as a
        documented no-op for the ``LearningRule`` lifecycle.
        """


def record_plasticity_report(
    history_data: BrainHistoryData,
    report: RuleStepReport,
    reward: float,
) -> None:
    """Append one plasticity step's telemetry to a brain's history.

    Shared by every brain that hosts the rule, so the four keys -- and what
    they mean -- cannot drift between the arms a panel compares. Also records
    the reward, matching where the gradient path records it.
    """
    extra = report.extra
    history_data.plasticity_prediction_error.append(extra[PREDICTION_ERROR_KEY])
    history_data.plasticity_baseline.append(extra[BASELINE_KEY])
    history_data.plasticity_mean_abs_delta.append(extra[MEAN_ABS_DELTA_KEY])
    history_data.plasticity_saturated_fraction.append(extra[SATURATED_FRACTION_KEY])
    history_data.rewards.append(reward)


# The connectome was the first substrate this rule drove, and existing code
# imports it under that name. The name now describes only where it started.
ConnectomeThreeFactorRule = ThreeFactorRule
