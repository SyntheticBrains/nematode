"""Measure a simulated track the way a real tracker measures a worm.

A real tracker reports a noisy midbody position and loses the worm at times: Scheer's loses most
seconds of an omega turn, and Chen's breaks every track into fragments. A simulated point worm has
neither, so statistics computed on its exact positions are not comparable with real ones. Before
any window is cut, every simulated track therefore passes through a :class:`TrackerModel`:

- each 1-s position gains noise: with a ``noise_library``, the real tracker's readings of a still
  worm, resampled (:class:`NoiseLibrary`); otherwise independent normal noise of ``jitter_mm`` per
  axis;
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
import functools
import hashlib
import math
import operator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from quantumnematode.benchmark.references import BENCHMARK_DATA

if TYPE_CHECKING:
    from collections.abc import Sequence

    from quantumnematode.report.behaviour import BehaviourStep

LAWN_STREAM = 2033
CHEMOTAXIS_STREAM = 2037
UM_PER_MM = 1000.0


SPLICE_CANDIDATES = 50


@dataclass(frozen=True)
class NoiseLibrary:
    """The real tracker's readings of a still worm: stretches of offsets from their own mean.

    Stretch ``k`` is ``lengths[k]`` consecutive 1-s offsets in ``x_mm`` and ``y_mm``. A noise track
    starts at a random stretch's first offset and runs along it. Where a stretch ends, the track is
    spliced: of :data:`SPLICE_CANDIDATES` stretches drawn uniformly with replacement, it continues
    from the one holding the offset nearest the track's last, at the offset after that one. So the
    track never jumps further than the nearest match, and every step after a splice is a real step.
    """

    x_mm: np.ndarray
    y_mm: np.ndarray
    lengths: np.ndarray

    @functools.cached_property
    def starts(self) -> np.ndarray:
        """Return each stretch's first index into the offsets, computed once."""
        return np.concatenate(([0], np.cumsum(self.lengths)[:-1]))

    def _splice(self, x: float, y: float, rng: np.random.Generator) -> tuple[int, int]:
        """Return the index to continue from and its stretch's end, nearest the offset (x, y).

        A stretch's usable offsets are all but its last. They are laid end to end in the order the
        candidates were drawn, so the first minimum is the first in candidate-then-offset order.
        """
        ks = rng.integers(len(self.lengths), size=SPLICE_CANDIDATES)
        starts, lengths = self.starts[ks], self.lengths[ks]
        usable = lengths - 1
        ends = np.cumsum(usable)
        index = np.arange(int(ends[-1])) + np.repeat(starts - (ends - usable), usable)
        distance = np.hypot(self.x_mm[index] - x, self.y_mm[index] - y)
        best = int(np.argmin(distance))
        row = int(np.searchsorted(ends, best, side="right"))
        return int(index[best]) + 1, int(starts[row] + lengths[row])

    def draw(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Return an ``(n, 2)`` noise track drawn from ``rng``."""
        out = np.empty((n, 2))
        k = int(rng.integers(len(self.lengths)))
        j, end = int(self.starts[k]), int(self.starts[k] + self.lengths[k])
        t = 0
        while t < n:
            take = min(end - j, n - t)
            out[t : t + take, 0] = self.x_mm[j : j + take]
            out[t : t + take, 1] = self.y_mm[j : j + take]
            t += take
            if t < n:
                j, end = self._splice(out[t - 1, 0], out[t - 1, 1], rng)
        return out


def save_noise_library(path: Path, x_mm: np.ndarray, y_mm: np.ndarray, lengths: np.ndarray) -> None:
    """Write a noise library, its offsets in whole micrometres as the Scheer store keeps them.

    The read caches are cleared, so a library regenerated in this process is read anew.
    """
    np.savez_compressed(
        path,
        x_um=np.round(np.asarray(x_mm) * UM_PER_MM).astype(np.int32),
        y_um=np.round(np.asarray(y_mm) * UM_PER_MM).astype(np.int32),
        lengths=np.asarray(lengths, dtype=np.int32),
    )
    _read_library.cache_clear()
    _digest.cache_clear()


def _stamp(path: Path) -> tuple[str, int, int]:
    """Return a file's path, size and modification time: a rewritten file has a new stamp."""
    status = path.stat()
    return str(path), status.st_size, status.st_mtime_ns


@functools.cache
def _read_library(stamp: tuple[str, int, int]) -> NoiseLibrary:
    with np.load(stamp[0]) as store:
        return NoiseLibrary(
            x_mm=store["x_um"].astype(float) / UM_PER_MM,
            y_mm=store["y_um"].astype(float) / UM_PER_MM,
            lengths=store["lengths"].astype(int),
        )


def load_noise_library(name: str) -> NoiseLibrary:
    """Return a vendored noise library by its file name under the benchmark data.

    It is read once per version of the file: rewriting the file is seen within the same process.
    """
    return _read_library(_stamp(BENCHMARK_DATA / name))


def file_sha256(path: Path) -> str:
    """Return a file's SHA-256, as recorded beside a model that names it."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@functools.cache
def _digest(stamp: tuple[str, int, int]) -> str:
    return file_sha256(Path(stamp[0]))


def _library_sha256(name: str) -> str:
    return _digest(_stamp(BENCHMARK_DATA / name))


def checked_noise_library(model: TrackerModel) -> NoiseLibrary:
    """Return a model's noise library, refusing one whose file differs from the recorded digest.

    Raises
    ------
    ValueError
        If the model records a digest and the library's file no longer matches it, so a seed's
        noise cannot change silently under a registered model.
    """
    name = model.noise_library
    if name is None:
        msg = "the model names no noise library"
        raise ValueError(msg)
    if model.noise_sha256 is not None and _library_sha256(name) != model.noise_sha256:
        msg = f"{name} does not match the digest its model records ({model.noise_sha256})"
        raise ValueError(msg)
    return load_noise_library(name)


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
    noise_library: str | None = None
    noise_sha256: str | None = None

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
    if model.noise_library is not None:
        noise = checked_noise_library(model).draw(n, rng)
    else:
        noise = rng.normal(0.0, model.jitter_mm, size=(n, 2))
    lost = lost_seconds([s.primitive for s in steps], model, rng)
    if model.fragment_lengths_s is not None:
        lost[fragment_cuts(n, model.fragment_lengths_s, rng)] = True
    if not steps:
        return []
    # Each measured step is rebuilt by its constructor, positionally: the same fields and the same
    # __init__ as dataclasses.replace. The noisy positions are added as arrays, element for
    # element the same sums, and iterate as the same numpy floats.
    cls = type(steps[0])
    names = _init_fields(cls)
    ix, iy = names.index("x"), names.index("y")
    fields = operator.attrgetter(*names)
    xs = np.array([s.x for s in steps]) + noise[:, 0]
    ys = np.array([s.y for s in steps]) + noise[:, 1]
    out = []
    for s, gone, x, y in zip(steps, lost.tolist(), xs, ys, strict=True):
        values = list(fields(s))
        if gone:
            values[ix], values[iy] = math.nan, math.nan
        else:
            values[ix], values[iy] = x, y
        out.append(cls(*values))
    return out


@functools.cache
def _init_fields(cls: type) -> list[str]:
    """Return a dataclass's constructor fields, in order."""
    return [f.name for f in dataclasses.fields(cls) if f.init]


def jitter_from_steps(step_mm: np.ndarray) -> float:
    """Return the per-axis noise SD implied by 1-s steps of a still worm.

    The step between two noisy readings of one point is the norm of a 2-D normal of SD
    ``sigma * sqrt(2)`` per axis, whose mean is ``sigma * sqrt(pi)``.
    """
    steps = np.asarray(step_mm, dtype=float)
    steps = steps[np.isfinite(steps)]
    return float(np.mean(steps) / math.sqrt(math.pi))
