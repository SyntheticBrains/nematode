"""Step-input adaptation-transient validation for the adaptive chemosensory sensor.

Drives a concentration **step input** to the sensor in isolation and measures the
**adaptation transient** (a peak response on the step, then relaxation toward
baseline as the background tracker catches up), and quantifies **Weber-like
invariance** of the peak to the absolute background level — the hyper-Weber
signature (Kato et al. 2014; Levy & Bargmann 2020). Both are reported **against the
log-concentration baseline**, which has no background tracking and therefore neither
relaxes nor stays background-invariant.
"""

from __future__ import annotations

from dataclasses import dataclass

from quantumnematode.agent.adaptive_sensor import (
    READOUT_FOLD_CHANGE,
    AdaptiveChemosensor,
)


@dataclass
class StepInputResult:
    """Transient of one readout to a single concentration step."""

    readout: str
    background: float
    step: float
    transient: list[float]  # response channel over time (pre-step + post-step)
    pre_steps: int
    peak: float  # max |response| in the post-step window
    relaxation_ratio: float  # |final| / |peak| — low ⇒ strong adaptation (relaxes back)


def _response_channel(sensor: AdaptiveChemosensor, strength: float, derivative: float) -> float:
    """Return the channel the configured readout reshapes (carries the adaptive signal)."""
    return derivative if sensor.readout == READOUT_FOLD_CHANGE else strength


def run_step_input(  # noqa: PLR0913
    readout: str,
    *,
    background: float = 0.1,
    step: float = 0.2,
    pre_steps: int = 20,
    post_steps: int = 60,
    alpha: float = 0.1,
    eps: float = 1e-3,
) -> StepInputResult:
    """Drive a ``background → background + step`` concentration series through the sensor.

    The derivative ``dC/dt`` is the finite difference of the concentration series.
    Returns the response-channel transient plus the peak and the relaxation ratio
    (``|final| / |peak|``; ``≈ 0`` means the response relaxes back to baseline —
    adaptation; ``≈ 1`` means a sustained, non-adapting shift).
    """
    sensor = AdaptiveChemosensor(readout=readout, alpha=alpha, eps=eps)
    series = [background] * pre_steps + [background + step] * post_steps
    transient: list[float] = []
    prev_c = background
    for c in series:
        derivative = c - prev_c
        prev_c = c
        strength, deriv_out = sensor.adapt(c, derivative)
        transient.append(_response_channel(sensor, strength, deriv_out))

    post = [abs(v) for v in transient[pre_steps:]]
    peak = max(post, default=0.0)
    final = abs(transient[-1]) if transient else 0.0
    relaxation_ratio = (final / peak) if peak > 0 else 0.0
    return StepInputResult(
        readout=readout,
        background=background,
        step=step,
        transient=transient,
        pre_steps=pre_steps,
        peak=peak,
        relaxation_ratio=relaxation_ratio,
    )


def weber_invariance_peaks(  # noqa: PLR0913
    readout: str,
    *,
    backgrounds: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4),
    relative_step: float = 2.0,
    pre_steps: int = 20,
    post_steps: int = 60,
    alpha: float = 0.1,
    eps: float = 1e-3,
) -> list[float]:
    """Peak response to the **same relative** step (``C → relative_step · C``) at each background.

    A Weber / hyper-Weber sensor returns ~constant peaks across absolute background
    levels; a non-adaptive (log) readout does not.
    """
    peaks: list[float] = []
    for background in backgrounds:
        step = background * (relative_step - 1.0)
        result = run_step_input(
            readout,
            background=background,
            step=step,
            pre_steps=pre_steps,
            post_steps=post_steps,
            alpha=alpha,
            eps=eps,
        )
        peaks.append(result.peak)
    return peaks


def weber_spread(peaks: list[float]) -> float:
    """Relative spread of the Weber peaks (``(max - min) / mean``); ~0 ⇒ invariant."""
    if not peaks:
        return 0.0
    mean = sum(peaks) / len(peaks)
    if mean == 0.0:
        return 0.0
    return (max(peaks) - min(peaks)) / mean
