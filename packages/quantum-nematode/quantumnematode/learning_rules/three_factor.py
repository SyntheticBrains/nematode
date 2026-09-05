"""Reward-modulated Hebbian plasticity over a connectome topology.

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
    from quantumnematode.brain.arch._topology import BrainTopology
    from quantumnematode.brain.arch.connectome_ppo import ConnectomeTopology

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


class ConnectomeThreeFactorRule:
    """Reward-modulated Hebbian plasticity on a connectome's chemical synapses.

    Satisfies the ``LearningRule`` Protocol. Owns its hyperparameters and
    a single scalar of state (the reward baseline); it owns no optimiser,
    no critic, and no experience buffer.

    Only ``w_chem`` changes. Sensory-projection gains, the motor readout,
    and action-noise parameters are left at their initial values, so any
    difference between two wiring variants trained under this rule is
    attributable to the wiring rather than to a differently-fitted
    decoder.
    """

    def __init__(  # noqa: PLR0913 — mirrors the hyperparameters it caches
        self,
        topology: ConnectomeTopology,
        *,
        plasticity_rate: float,
        weight_decay: float,
        weight_bound: float,
        baseline_rate: float,
        device: torch.device,
    ) -> None:
        self._topology = topology
        self.plasticity_rate = plasticity_rate
        self.weight_decay = weight_decay
        self.weight_bound = weight_bound
        self.baseline_rate = baseline_rate
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
                "ConnectomeThreeFactorRule.step received a topology other than "
                "the one it was constructed over. The rule reads that topology's "
                "eligibility trace, so updating a different one would apply "
                "credit assigned from unrelated activity — construct a new rule "
                "for a new topology."
            )
            raise ValueError(msg)
        topo = self._topology
        if not topo.enable_activity_traces:
            msg = (
                "ConnectomeThreeFactorRule requires activity traces, but the "
                "topology has none allocated. Without a trace the update is "
                "identically zero and training would silently do nothing."
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

            weights = topo.w_chem.data
            # Hebbian term over the eligibility trace, plus decay toward
            # zero. Decay is what keeps unreinforced synapses from holding
            # whatever a transient correlation put there.
            update = self.plasticity_rate * delta * topo.activity_traces
            update -= self.plasticity_rate * self.weight_decay * weights
            # The trace is already masked, but the decay term is not: it is
            # proportional to the weights, which under the soft-prior mask
            # mode may hold values off the wild-type edge set. Projecting
            # keeps the rule from writing anywhere the topology says there
            # is no synapse.
            update = topo.apply_weight_mask(update)

            weights.add_(update)
            weights.clamp_(-self.weight_bound, self.weight_bound)

            mean_abs_delta = update.abs().mean().item()
            # Fraction measured over real synapses only: the off-edge zeros
            # are not saturated, and counting them would dilute the signal
            # by the connectome's sparsity and hide a saturating run.
            edge_count = int(topo.m_chem.sum().item())
            if edge_count == 0:  # pragma: no cover — a topology with no synapses
                saturated = 0.0
            else:
                at_bound = ((weights.abs() >= self.weight_bound) & topo.m_chem).sum().item()
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
