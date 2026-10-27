"""Measure a simulated track the way a real tracker measures a worm.

A real tracker reports a noisy midbody position and loses the worm at times: Scheer's loses most
seconds of an omega turn, and Chen's breaks every track into fragments. A simulated point worm has
neither, so statistics computed on its exact positions are not comparable with real ones. Before
any window is cut, every simulated track therefore passes through a :class:`TrackerModel`:

- each 1-s position gains independent normal noise of ``jitter_mm`` per axis;
- tracking is then lost in runs: at each tracked second a loss starts with ``p_omega`` if the
  simulated primitive that second is an omega and ``p_other`` otherwise, and lasts a length drawn
  from that context's measured runs (``runs_omega_s``, ``runs_other_s``). Without runs, a start
  loses that second alone;
- where ``fragment_lengths_s`` is given, the track is cut into fragments of lengths drawn from it,
  one lost second marking each cut, as a tracker that loses the worm's identity does.

Every draw comes from the animal's own stream, ``default_rng([seed, stream_key])``, so an animal is
measured the same way alone or in any population. Real and imported tracks are never passed here.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from quantumnematode.report.behaviour import BehaviourStep

LAWN_STREAM = 2033
CHEMOTAXIS_STREAM = 2037


@dataclass(frozen=True)
class TrackerModel:
    """One tracker's measurement: position noise, lost seconds and fragmentation."""

    jitter_mm: float
    p_omega: float = 0.0
    p_other: float = 0.0
    runs_omega_s: tuple[int, ...] | None = None
    runs_other_s: tuple[int, ...] | None = None
    fragment_lengths_s: tuple[int, ...] | None = None
    stream_key: int = LAWN_STREAM

    def to_dict(self) -> dict[str, object]:
        """Return the model as recorded with the bands."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, record: dict[str, object]) -> TrackerModel:
        """Return a model from its recorded form."""
        fields = {f.name for f in dataclasses.fields(cls)}
        values = {k: v for k, v in record.items() if k in fields}
        for key in ("runs_omega_s", "runs_other_s", "fragment_lengths_s"):
            if values.get(key) is not None:
                values[key] = tuple(int(v) for v in values[key])  # type: ignore[union-attr]
        return cls(**values)  # type: ignore[arg-type]


def lost_seconds(
    primitives: Sequence[str | None],
    model: TrackerModel,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return which seconds the tracker loses.

    At each tracked second a loss starts with ``p_omega`` in an omega and ``p_other`` elsewhere,
    and lasts a length drawn from that context's runs; without runs it lasts that second alone.
    """
    n = len(primitives)
    lost = np.zeros(n, dtype=bool)
    draws = rng.random(n)
    t = 0
    while t < n:
        omega = primitives[t] == "omega"
        if draws[t] < (model.p_omega if omega else model.p_other):
            runs = model.runs_omega_s if omega else model.runs_other_s
            length = 1 if runs is None else int(runs[rng.integers(len(runs))])
            lost[t : t + length] = True
            t += length
        else:
            t += 1
    return lost


def fragment_cuts(n: int, lengths: Sequence[int], rng: np.random.Generator) -> np.ndarray:
    """Return the seconds at which a track of ``n`` seconds is cut, fragments drawn from lengths."""
    cuts = []
    t = 0
    pool = np.asarray(lengths, dtype=int)
    while True:
        t += max(1, int(pool[rng.integers(len(pool))]))
        if t >= n:
            break
        cuts.append(t)
        t += 1
    return np.array(cuts, dtype=int)


def measure(
    steps: Sequence[BehaviourStep],
    seed: int,
    model: TrackerModel,
) -> list[BehaviourStep]:
    """Return a simulated animal's steps as the tracker would report them.

    Positions gain noise, lost seconds and fragment cuts have NaN positions, and every other field
    (lawn contact, primitive, phase) is kept, since those are the simulator's ground truth that the
    benchmark reads only where real tracks give the same.
    """
    rng = np.random.default_rng([seed, model.stream_key])
    n = len(steps)
    noise = rng.normal(0.0, model.jitter_mm, size=(n, 2))
    lost = lost_seconds([s.primitive for s in steps], model, rng)
    if model.fragment_lengths_s is not None:
        lost[fragment_cuts(n, model.fragment_lengths_s, rng)] = True
    out = []
    for i, s in enumerate(steps):
        if lost[i]:
            out.append(dataclasses.replace(s, x=math.nan, y=math.nan))
        else:
            out.append(dataclasses.replace(s, x=s.x + noise[i, 0], y=s.y + noise[i, 1]))
    return out


def jitter_from_steps(step_mm: np.ndarray) -> float:
    """Return the per-axis noise SD implied by 1-s steps of a still worm.

    The step between two noisy readings of one point is the norm of a 2-D normal of SD
    ``sigma * sqrt(2)`` per axis, whose mean is ``sigma * sqrt(pi)``.
    """
    steps = np.asarray(step_mm, dtype=float)
    steps = steps[np.isfinite(steps)]
    return float(np.mean(steps) / math.sqrt(math.pi))
