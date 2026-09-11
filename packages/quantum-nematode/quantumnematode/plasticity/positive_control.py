"""A positive control for the reward-modulated three-factor rule.

Seven registered results asked whether a connectome's wiring is legible to the minimal
three-factor rule, and none of them established that the rule learns anything: the matched-rule
MLP yardstick sat at chance and the rule destroyed a competent policy on a dense feedforward
network within three episodes. "The wiring is not legible to this rule" and "this rule does not
learn" predict the same null, and only a task with a known answer separates them.

This module is that task. It is deliberately the smallest thing reward-modulated Hebbian
learning is supposed to solve, with both the optimum and the cue-blind floor in closed form:

- a cue ``c`` is drawn uniformly from ``K`` one-hot alternatives;
- the policy emits an unsquashed scalar action ``a = mu(c) + eps``, ``eps ~ N(0, sigma^2)``;
- reward is ``-(a - t(c))^2`` for a fixed per-cue target ``t(c)``.

The targets never appear in the observation, so the association is discoverable only from
reward. A policy that ignores the cue earns at best ``-Var[t] - sigma^2`` (the exploration noise
costs every policy the same, so it sits on both sides); a policy with ``mu(c) = t(c)`` earns
``-sigma^2``. The gap between them is exactly ``Var[t]``, and it can only be closed by using the
cue.

The task also supports a **delay** between the scored action and the reward. At delay ``D`` the
cue is shown, the scored action is taken, ``D`` further steps run against a constant filler
observation, and the reward arrives once at the end. Nothing about what is scored changes, and
neither do the bounds -- they depend on the targets and the exploration noise alone -- so a
delayed arm is directly comparable with an undelayed one and ``D = 0`` is the undelayed control.

The delay exists because the rule's eligibility decays, and a one-step task cannot see that: the
control resets the trace every trial, which is what removes the horizon confound from the
undelayed question and what makes it useless for the horizon itself. What a delay imposes is not
decay but **dilution** -- the credited step's share of the trace falls as later steps add their
own terms -- and dilution is what survives a rule that normalises its trace.

Pure functions and a dataclass: no environment, no runner, no connectome, no action head. What
is under test is the rule, its eligibility and its modulator, and everything else that could
explain a null is removed rather than controlled for.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_CUES = 4
"""One-hot alternatives. Four is enough to need the cue and small enough to stay analytic."""


@dataclass(frozen=True)
class ContextualAssociation:
    """The control's task: cue in, action out, reward on the squared error.

    ``targets`` are fixed at construction and spread across the action range. They are not part
    of the observation -- that is what makes reward the only route to them.
    """

    targets: np.ndarray

    @classmethod
    def default(cls, n_cues: int = DEFAULT_CUES) -> ContextualAssociation:
        """Build the default task: targets spread evenly over the range an action explores."""
        return cls(targets=np.linspace(-1.0, 1.0, n_cues))

    @property
    def n_cues(self) -> int:
        """Count the alternatives a cue is drawn from."""
        return int(self.targets.size)

    def sample_cue(self, rng: np.random.Generator) -> int:
        """Draw a cue index uniformly, independently of every previous trial."""
        return int(rng.integers(self.n_cues))

    def observation(self, cue: int) -> np.ndarray:
        """Return the one-hot cue, which carries no information about that cue's target."""
        out = np.zeros(self.n_cues, dtype=np.float32)
        out[cue] = 1.0
        return out

    def filler(self) -> np.ndarray:
        """Return the observation shown between the scored action and a delayed reward.

        Uniform over the cue channels, so it is identical on every trial and says nothing about
        which cue was shown. Three properties matter and each is load-bearing:

        - **Nonzero.** Where a topology's plastic layer takes the observation as its pre-synaptic
          input, a zero input adds nothing to the eligibility trace. The trace at reward time
          would then be a pure scalar multiple of the credited step's, and a rule that normalises
          its trace by a running RMS divides exactly that out -- the delay would measure nothing
          at any length.
        - **Constant.** Anything varying with the cue would leak the association into the steps
          that follow the scored action.
        - **The same width as a cue.** A separate "no cue" channel would widen the observation and
          change the network's initialisation, so a delay of zero would no longer reproduce the
          undelayed control.
        """
        return np.full(self.n_cues, 1.0 / self.n_cues, dtype=np.float32)

    def reward(self, cue: int, action: float) -> float:
        """Score an action: ``-(a - t(c))^2``, zero at the target and falling away quadratically."""
        return -float((action - self.targets[cue]) ** 2)

    def credited_share(self, decay: float, delay: int) -> float:
        """Return the credited step's share of the trace when a delayed reward arrives.

        The scored step contributes ``decay ** delay`` by the time the modulator lands; each
        intervening step contributes its own term, which is what *dilutes* the credited one. A
        rule that normalises its trace rescales the whole sum, so the share -- not the magnitude --
        is what a delay actually changes, and it is what this control measures.
        """
        credited = decay**delay
        intervening = sum(decay**step for step in range(delay))
        return credited / (credited + intervening)

    def cue_blind_floor(self, noise: float) -> float:
        """Return the best expected reward available without using the cue.

        A constant mean ``a0`` earns ``-E[(a0 - t)^2] - sigma^2``, minimised at ``a0 = E[t]``,
        which leaves the variance of the targets plus the exploration cost. Nothing that ignores
        the cue beats it, which is what makes it a floor rather than a baseline.
        """
        return -float(np.var(self.targets)) - noise**2

    def optimum(self, noise: float) -> float:
        """Return the best expected reward available at all.

        Achieved by ``mu(c) = t(c)``, which pays only the exploration noise.
        """
        return -(noise**2)

    def gap(self, noise: float) -> float:
        """Return the floor-to-optimum gap -- exactly ``Var[t]``, since the noise term cancels."""
        return self.optimum(noise) - self.cue_blind_floor(noise)
