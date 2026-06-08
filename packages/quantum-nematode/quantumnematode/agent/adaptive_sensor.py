"""Adaptive-threshold / biphasic chemosensory sensor.

Models *C. elegans* chemosensory adaptation (Kato et al. 2014 *Neuron*; Levy &
Bargmann 2020 *Neuron* "hyper-Weber"): the chemosensory signal is coded **relative
to a slowly-tracked background**, not as a static squash. A leaky-integrator
background ``B_t = (1 - alpha) * B_{t-1} + alpha * C_t`` (following STAM's
exponential-decay pattern) feeds a biphasic relative readout. Plain
log-concentration is a documented, under-powered baseline special case.

The readout form is config-selectable; the choice also fixes the **channel
interaction** (a behavioural contract, not a free tuning):

- ``fold_change`` — reshapes the **derivative / turning** channel:
  ``(dC/dt) / (C + eps) ~ d(log C)/dt``. The regularising ``+ eps`` is required —
  raw ``(dC/dt)/C`` is singular at the common ``C ~ 0`` regime far from a source.
  The strength (magnitude) channel is left non-adaptive.
- ``contrast`` — reshapes the **magnitude / strength** channel:
  ``(C - B) / (C + B + eps)``. The derivative channel is left non-adaptive.
- ``log`` — the log-concentration baseline ``log(1 + C)`` on the magnitude channel
  (the documented ablation comparator).

This sensor applies to chemosensory channels only (food chemotaxis; pheromone / CO2
where active) — never thermosensory or mechanosensory channels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Config-selectable readout modes.
READOUT_FOLD_CHANGE = "fold_change"
READOUT_CONTRAST = "contrast"
READOUT_LOG = "log"


class LeakyIntegrator:
    """Exponential-decay running background ``B_t = (1-alpha)*B_{t-1} + alpha*C_t``.

    Follows STAM's decay pattern. Seeded on the first sample so the background
    starts at the first observed concentration (no spurious initial transient).
    """

    def __init__(self, alpha: float) -> None:
        self._alpha = float(alpha)
        self._background: float | None = None

    def update(self, value: float) -> float:
        """Fold a new sample into the background and return the updated background."""
        if self._background is None:
            self._background = float(value)
        else:
            self._background = (1.0 - self._alpha) * self._background + self._alpha * float(value)
        return self._background

    @property
    def background(self) -> float:
        """The current tracked background (0.0 before the first sample)."""
        return self._background if self._background is not None else 0.0

    def reset(self) -> None:
        """Clear the background (e.g. at episode start)."""
        self._background = None


def fold_change_readout(derivative: float, concentration: float, eps: float) -> float:
    """Derivative-channel fold-change ``(dC/dt) / (C + eps) ~ d(log C)/dt``."""
    return derivative / (concentration + eps)


def magnitude_contrast_readout(concentration: float, background: float, eps: float) -> float:
    """Instantaneous biphasic contrast ``(C - B) / (C + B + eps)`` in ``[-1, 1]``."""
    return (concentration - background) / (concentration + background + eps)


def log_concentration_readout(concentration: float) -> float:
    """Log-concentration baseline ``log(1 + C)`` (documented under-powered special case)."""
    return math.log1p(max(0.0, concentration))


@dataclass
class AdaptiveChemosensor:
    """Per-channel adaptive readout over a leaky-integrator background.

    Holds the background state for one chemosensory channel across timesteps; call
    :meth:`adapt` once per step with the raw concentration and STAM-derived
    derivative. The configured ``readout`` fixes which channel is reshaped (see the
    module docstring).
    """

    readout: str = READOUT_FOLD_CHANGE
    alpha: float = 0.1
    eps: float = 1e-3
    _integrator: LeakyIntegrator = field(init=False)
    last_readout: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Initialise the per-channel background tracker."""
        self._integrator = LeakyIntegrator(self.alpha)

    @property
    def background(self) -> float:
        """The current tracked background concentration (0.0 before the first sample).

        Public accessor delegating to the leaky integrator so callers (e.g. a
        renderer snapshot builder) never reach into the private integrator state.
        """
        return self._integrator.background

    def reset(self) -> None:
        """Reset the background tracker (episode boundary)."""
        self._integrator.reset()
        self.last_readout = None

    def adapt(self, concentration: float, derivative: float) -> tuple[float, float]:
        """Update the background and return the adapted ``(strength, derivative)``.

        Exactly one channel is reshaped per the configured ``readout``; the other
        retains its non-adaptive value. The reshaped channel's value is also cached
        in :attr:`last_readout` for read-only inspection (e.g. rendering).
        """
        background = self._integrator.update(concentration)
        if self.readout == READOUT_FOLD_CHANGE:
            readout = fold_change_readout(derivative, concentration, self.eps)
            self.last_readout = readout
            return concentration, readout
        if self.readout == READOUT_CONTRAST:
            readout = magnitude_contrast_readout(concentration, background, self.eps)
            self.last_readout = readout
            return readout, derivative
        if self.readout == READOUT_LOG:
            readout = log_concentration_readout(concentration)
            self.last_readout = readout
            return readout, derivative
        # Unknown readout: behave as a no-op (defensive; config is Literal-validated).
        # Clear the cached readout so a stale value can't leak into a snapshot.
        self.last_readout = None
        return concentration, derivative
