"""Locomotor primitives for the continuous point worm, driven through start hazards.

Under ``continuous.locomotion: primitives`` the worm moves only by:

- **runs**: one step at a controller-set speed in ``[0, v_max]`` and curvature in
  ``[-omega_max, omega_max]``; a pause is a run at zero speed;
- **reversal bouts**: backing at a fixed speed along an unchanged heading for a drawn duration; a
  reversal the caller marks as starting on a lawn draws the on-food speed and duration, shorter and
  slower than off food. Optionally its speed instead follows the worm's recent run speed, as a real
  worm's does on a lawn: interpolated between measured knots on the mean run speed of its last few
  seconds of runs;
- **omega and delta bouts**: a drawn heading change toward the worm's ventral side (omega) or dorsal
  side (delta), at a constant rate over a drawn duration, without translation; a worm crawls on one
  side, so which way is ventral is drawn once per episode.

A controller drives them through five numbers in ``[-1, 1]`` each step: run speed and curvature,
mapped linearly, and a start hazard for each bout kind, mapped log-linearly onto the configured
range per second. While no bout runs, a bout starts with probability ``1 - exp(-L dt)``, ``L`` the
hazards' sum, its kind drawn in proportion to its hazard: competing hazards, so constant hazards
make bout starts a Poisson process. A start is drawn only at the beginning of a step with no bout
running; a bout's last step may be partial, and nothing starts within it. During a bout the
controller's outputs move nothing.

The fixed kinematics are read from ``data/locomotion/primitives.json``, every value with its source:
a published one, or for a value measured from deposited tracks, the script and the record it wrote.
"""

from __future__ import annotations

import bisect
import itertools
import json
import math
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
KINEMATICS_PATH = _PROJECT_ROOT / "data" / "locomotion" / "primitives.json"

PrimitiveKind = Literal["run", "reversal", "omega", "delta"]
PRIMITIVE_KINDS: tuple[PrimitiveKind, ...] = ("run", "reversal", "omega", "delta")
BOUT_KINDS: tuple[PrimitiveKind, ...] = ("reversal", "omega", "delta")
PRIMITIVES_ACTION_DIM = 5
"""Run speed, run curvature, then a start hazard for each of reversal, omega and delta."""

# A bout whose remaining time falls below this has ended; it absorbs float residue from summing
# whole steps against a drawn duration.
_END_TOLERANCE_S = 1e-9
# Rejection draws for a truncated normal before falling back to clipping; with the vendored bounds
# a draw is accepted with probability above 0.9, so the fallback is never reached in practice.
_MAX_REJECTIONS = 1000


@dataclass(frozen=True)
class GammaDuration:
    """A gamma-distributed duration, in seconds, given by its mean and standard deviation."""

    mean_s: float
    sd_s: float

    def sample(self, rng: np.random.Generator) -> float:
        """Draw one duration."""
        shape = (self.mean_s / self.sd_s) ** 2
        return float(rng.gamma(shape, self.sd_s**2 / self.mean_s))


@dataclass(frozen=True)
class TruncatedNormalAngle:
    """A heading change's magnitude, in radians, normal and truncated to ``[low, high]``."""

    mean_rad: float
    sd_rad: float
    low_rad: float
    high_rad: float

    def sample(self, rng: np.random.Generator) -> float:
        """Draw one magnitude by rejection."""
        for _ in range(_MAX_REJECTIONS):
            value = float(rng.normal(self.mean_rad, self.sd_rad))
            if self.low_rad <= value <= self.high_rad:
                return value
        return min(self.high_rad, max(self.low_rad, self.mean_rad))


@dataclass(frozen=True)
class SpeedCoupling:
    """An on-food reversal's speed as a function of the worm's recent run speed.

    Each knot pairs a mean recent run speed with real worms' mean reversal speed at it. The speed is
    interpolated linearly between knots and is flat beyond the outermost. The recent run speed is
    the mean commanded run speed over the worm's last ``window_s`` of run steps.
    """

    knots_before_mm_per_s: tuple[float, ...]
    knots_speed_mm_per_s: tuple[float, ...]
    window_s: float

    def window_steps(self, step_seconds: float) -> int:
        """Return how many run steps the window holds at a step of ``step_seconds``."""
        return max(1, round(self.window_s / step_seconds))

    def speed(self, mean_run_speed_mm_per_s: float) -> float:
        """Return a reversal's speed for a worm's mean recent run speed."""
        xs, ys = self.knots_before_mm_per_s, self.knots_speed_mm_per_s
        v = mean_run_speed_mm_per_s
        if v <= xs[0]:
            return ys[0]
        if v >= xs[-1]:
            return ys[-1]
        k = bisect.bisect_right(xs, v)
        return ys[k - 1] + (ys[k] - ys[k - 1]) * (v - xs[k - 1]) / (xs[k] - xs[k - 1])


@dataclass(frozen=True)
class PrimitiveKinematics:
    """The primitives' fixed kinematics, in mm, seconds and radians.

    The ``reversal_*`` values hold off food; the ``on_food_reversal_*`` values hold for a reversal
    that starts on a lawn. ``on_food_reversal_speed_coupling`` replaces the on-food speed when a
    reversal's speed follows the worm's run speed.
    """

    v_max_mm_per_s: float
    omega_max_rad_per_s: float
    reversal_speed_mm_per_s: float
    reversal_duration: GammaDuration
    on_food_reversal_speed_mm_per_s: float
    on_food_reversal_duration: GammaDuration
    on_food_reversal_speed_coupling: SpeedCoupling
    omega_angle: TruncatedNormalAngle
    omega_duration: GammaDuration
    delta_angle: TruncatedNormalAngle
    delta_duration: GammaDuration
    hazard_min_per_s: float
    hazard_max_per_s: float


def _gamma(entry: dict[str, object]) -> GammaDuration:
    if entry.get("distribution") != "gamma":
        msg = f"expected a gamma duration, got {entry.get('distribution')!r}"
        raise ValueError(msg)
    mean, sd = float(entry["mean"]), float(entry["sd"])  # type: ignore[arg-type]
    if mean <= 0.0 or sd <= 0.0:
        msg = f"a duration's mean and sd must be positive, got {mean} and {sd}"
        raise ValueError(msg)
    return GammaDuration(mean_s=mean, sd_s=sd)


def _angle(entry: dict[str, object], side: str) -> TruncatedNormalAngle:
    if entry.get("distribution") != "truncated_normal" or entry.get("side") != side:
        msg = f"expected a truncated-normal {side} angle, got {entry}"
        raise ValueError(msg)
    mean, sd = float(entry["mean"]), float(entry["sd"])  # type: ignore[arg-type]
    low, high = float(entry["low"]), float(entry["high"])  # type: ignore[arg-type]
    if not 0.0 <= low <= mean <= high or sd <= 0.0:
        msg = f"an angle needs 0 <= low <= mean <= high and a positive sd, got {entry}"
        raise ValueError(msg)
    return TruncatedNormalAngle(
        mean_rad=math.radians(mean),
        sd_rad=math.radians(sd),
        low_rad=math.radians(low),
        high_rad=math.radians(high),
    )


def _coupling(entry: dict[str, object] | None) -> SpeedCoupling:
    if entry is None:
        msg = "the kinematics lack 'on_food_reversal_speed_coupling'"
        raise ValueError(msg)
    before = tuple(float(v) for v in entry["knots_before_mm_per_s"])  # type: ignore[attr-defined]
    speed = tuple(float(v) for v in entry["knots_speed_mm_per_s"])  # type: ignore[attr-defined]
    window = float(entry["window_s"])  # type: ignore[arg-type]
    finite = all(math.isfinite(v) for v in (*before, *speed, window))
    if not (
        finite
        and len(before) == len(speed) >= 2  # noqa: PLR2004 - a line needs two knots
        and before[0] >= 0.0
        and all(a < b for a, b in itertools.pairwise(before))
        and min(speed) > 0.0
        and window > 0.0
    ):
        msg = (
            "a reversal speed coupling needs at least two finite knots of equal length, the speeds "
            "before non-negative and strictly increasing, the reversal speeds positive and a "
            f"positive window, got {entry}"
        )
        raise ValueError(msg)
    return SpeedCoupling(knots_before_mm_per_s=before, knots_speed_mm_per_s=speed, window_s=window)


_STATUSES = ("verified", "figure", "measured")


def _check_measured(key: str, entry: dict[str, object]) -> None:
    """Refuse a measured value unless it names its script and record, and both exist."""
    for name in ("script", "record"):
        value = entry.get(name)
        if not isinstance(value, str) or not (_PROJECT_ROOT / value).is_file():
            msg = f"measured kinematics entry {key!r} needs an existing {name}, got {value!r}"
            raise ValueError(msg)


@cache
def load_kinematics(path: Path = KINEMATICS_PATH) -> PrimitiveKinematics:
    """Load and check the primitives' kinematics.

    Raises
    ------
    ValueError
        If an entry lacks its source, condition or status, a measured entry lacks the script or
        record it came from, or a value is out of range.
    """
    data: dict[str, dict[str, object]] = json.loads(path.read_text())
    for key, entry in data.items():
        if key == "description":
            continue
        missing = [field for field in ("source", "condition", "status") if field not in entry]
        if missing or entry["status"] not in _STATUSES:
            msg = f"kinematics entry {key!r} lacks {missing or 'a valid status'}"
            raise ValueError(msg)
        if entry["status"] == "measured":
            _check_measured(key, entry)
    hazards = data["hazard_range_per_s"]
    low, high = float(hazards["low"]), float(hazards["high"])  # type: ignore[arg-type]
    if not 0.0 < low < high:
        msg = f"the hazard range must satisfy 0 < low < high, got {low} and {high}"
        raise ValueError(msg)
    speeds = [
        float(data[key]["value"])  # type: ignore[arg-type]
        for key in (
            "v_max_mm_per_s",
            "omega_max_deg_per_s",
            "reversal_speed_mm_per_s",
            "on_food_reversal_speed_mm_per_s",
        )
    ]
    if min(speeds) <= 0.0:
        msg = f"speeds and turn rates must be positive, got {speeds}"
        raise ValueError(msg)
    return PrimitiveKinematics(
        v_max_mm_per_s=speeds[0],
        omega_max_rad_per_s=math.radians(speeds[1]),
        reversal_speed_mm_per_s=speeds[2],
        reversal_duration=_gamma(data["reversal_duration_s"]),
        on_food_reversal_speed_mm_per_s=speeds[3],
        on_food_reversal_duration=_gamma(data["on_food_reversal_duration_s"]),
        on_food_reversal_speed_coupling=_coupling(data.get("on_food_reversal_speed_coupling")),
        omega_angle=_angle(data["omega_angle_deg"], "ventral"),
        omega_duration=_gamma(data["omega_duration_s"]),
        delta_angle=_angle(data["delta_angle_deg"], "dorsal"),
        delta_duration=_gamma(data["delta_duration_s"]),
        hazard_min_per_s=low,
        hazard_max_per_s=high,
    )


@dataclass(frozen=True)
class PrimitiveCommand:
    """A controller's five outputs in physical units."""

    speed_mm_per_s: float
    curvature_rad_per_s: float
    hazards_per_s: tuple[float, float, float]


def map_action(raw: Sequence[float], kinematics: PrimitiveKinematics) -> PrimitiveCommand:
    """Map a controller's five numbers in ``[-1, 1]`` onto speed, curvature and start hazards.

    Each number is clipped to ``[-1, 1]``. Speed is ``v_max (x + 1) / 2``; curvature is
    ``omega_max x``; each hazard is log-linear between the configured range's ends.
    """
    if len(raw) != PRIMITIVES_ACTION_DIM:
        msg = f"a primitives action has {PRIMITIVES_ACTION_DIM} entries, got {len(raw)}"
        raise ValueError(msg)
    x = [min(1.0, max(-1.0, float(v))) for v in raw]
    log_low = math.log(kinematics.hazard_min_per_s)
    log_span = math.log(kinematics.hazard_max_per_s) - log_low
    hazards = tuple(math.exp(log_low + (v + 1.0) / 2.0 * log_span) for v in x[2:])
    return PrimitiveCommand(
        speed_mm_per_s=kinematics.v_max_mm_per_s * (x[0] + 1.0) / 2.0,
        curvature_rad_per_s=kinematics.omega_max_rad_per_s * x[1],
        hazards_per_s=(hazards[0], hazards[1], hazards[2]),
    )


def raw_for_hazard(rate_per_s: float, kinematics: PrimitiveKinematics) -> float:
    """Return the number in ``[-1, 1]`` that maps to a start hazard of ``rate_per_s``."""
    log_low = math.log(kinematics.hazard_min_per_s)
    log_span = math.log(kinematics.hazard_max_per_s) - log_low
    rate = min(kinematics.hazard_max_per_s, max(kinematics.hazard_min_per_s, rate_per_s))
    return 2.0 * (math.log(rate) - log_low) / log_span - 1.0


def raw_for_speed(speed_mm_per_s: float, kinematics: PrimitiveKinematics) -> float:
    """Return the number in ``[-1, 1]`` that maps to a run speed of ``speed_mm_per_s``."""
    return min(1.0, max(-1.0, 2.0 * speed_mm_per_s / kinematics.v_max_mm_per_s - 1.0))


@dataclass
class Bout:
    """A running reversal, omega or delta, and what it still has to do."""

    kind: PrimitiveKind
    remaining_s: float
    # Signed displacement along the heading per second (reversal) and heading change per second
    # (omega, delta).
    speed_mm_per_s: float = 0.0
    turn_rad_per_s: float = 0.0


@dataclass
class PrimitiveState:
    """One worm's primitive state across an episode's steps.

    ``run_speeds`` holds the commanded speeds of its latest run steps, oldest first, kept only when
    a reversal's speed follows the run speed.
    """

    ventral_side: int
    bout: Bout | None = None
    run_speeds: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class PrimitiveStep:
    """What one step did: the primitive that moved the worm, its displacement and heading change."""

    kind: PrimitiveKind
    displacement_mm: float
    heading_change_rad: float


def new_state(rng: np.random.Generator) -> PrimitiveState:
    """Start a worm's episode: draw which side, left (+1) or right (-1), is ventral."""
    return PrimitiveState(ventral_side=1 if rng.random() < 0.5 else -1)  # noqa: PLR2004


def mean_run_speed(window: Sequence[float]) -> float:
    """Return a run-speed window's mean, summed oldest first, the one way both runners take it."""
    return sum(window) / len(window)


def recent_run_speed(window: Sequence[float], steps: int) -> float | None:
    """Return the mean of a window's last ``steps`` speeds, or None until it holds that many."""
    return mean_run_speed(window[-steps:]) if len(window) >= steps else None


def push_run_speed(window: list[float], speed_mm_per_s: float, steps: int) -> None:
    """Add a run step's commanded speed to a window, keeping its last ``steps``, in place."""
    window.append(speed_mm_per_s)
    del window[:-steps]


def _start_bout(  # noqa: PLR0913 - the bout, the worm, kinematics, RNG and two conditions
    kind: PrimitiveKind,
    state: PrimitiveState,
    kinematics: PrimitiveKinematics,
    rng: np.random.Generator,
    *,
    on_food: bool = False,
    run_speed_mean: float | None = None,
) -> Bout:
    """Draw a new bout; a reversal takes its condition's kinematics, fixed for the whole bout.

    ``run_speed_mean``, if given, is the worm's mean recent run speed: a reversal on food then backs
    at the coupling's speed for it rather than the pooled on-food speed. It changes no draw.
    """
    if kind == "reversal":
        if on_food:
            duration = kinematics.on_food_reversal_duration
            speed = (
                kinematics.on_food_reversal_speed_mm_per_s
                if run_speed_mean is None
                else kinematics.on_food_reversal_speed_coupling.speed(run_speed_mean)
            )
        else:
            duration, speed = kinematics.reversal_duration, kinematics.reversal_speed_mm_per_s
        return Bout(kind=kind, remaining_s=duration.sample(rng), speed_mm_per_s=-speed)
    if kind == "omega":
        duration = kinematics.omega_duration.sample(rng)
        angle = kinematics.omega_angle.sample(rng) * state.ventral_side
    else:
        duration = kinematics.delta_duration.sample(rng)
        angle = -kinematics.delta_angle.sample(rng) * state.ventral_side
    return Bout(kind=kind, remaining_s=duration, turn_rad_per_s=angle / duration)


def step_primitives(  # noqa: PLR0913 - the worm, its action, the step, kinematics, RNG, food
    state: PrimitiveState,
    raw_action: Sequence[float],
    step_seconds: float,
    kinematics: PrimitiveKinematics,
    rng: np.random.Generator,
    *,
    on_food: bool = False,
    follows_run_speed: bool = False,
) -> PrimitiveStep:
    """Advance one worm by one step under the primitives, updating its state in place.

    While no bout runs, one uniform draw decides whether a bout starts and, if one does, a second
    decides its kind; a new bout then draws its duration and, for a turn, its angle, and runs from
    this step. Otherwise the step is a run at the controller's speed and curvature. ``on_food`` is
    whether the worm is on a lawn as the step starts; it sets a new reversal's kinematics. With
    ``follows_run_speed``, each run step's speed enters the worm's window, and a reversal on food
    backs at the coupling's speed for the window's mean once it is full.
    """
    steps = kinematics.on_food_reversal_speed_coupling.window_steps(step_seconds)
    if state.bout is None:
        command = map_action(raw_action, kinematics)
        total = sum(command.hazards_per_s)
        if rng.random() < 1.0 - math.exp(-total * step_seconds):
            pick = rng.random() * total
            cumulative = 0.0
            kind: PrimitiveKind = BOUT_KINDS[-1]
            for candidate, hazard in zip(BOUT_KINDS, command.hazards_per_s, strict=True):
                cumulative += hazard
                if pick < cumulative:
                    kind = candidate
                    break
            state.bout = _start_bout(
                kind,
                state,
                kinematics,
                rng,
                on_food=on_food,
                run_speed_mean=(
                    recent_run_speed(state.run_speeds, steps) if follows_run_speed else None
                ),
            )
        else:
            if follows_run_speed:
                push_run_speed(state.run_speeds, command.speed_mm_per_s, steps)
            return PrimitiveStep(
                kind="run",
                displacement_mm=command.speed_mm_per_s * step_seconds,
                heading_change_rad=command.curvature_rad_per_s * step_seconds,
            )
    bout = state.bout
    elapsed = min(step_seconds, bout.remaining_s)
    bout.remaining_s -= elapsed
    if bout.remaining_s <= _END_TOLERANCE_S:
        state.bout = None
    return PrimitiveStep(
        kind=bout.kind,
        displacement_mm=bout.speed_mm_per_s * elapsed,
        heading_change_rad=bout.turn_rad_per_s * elapsed,
    )


def running_kind(state: PrimitiveState | None) -> PrimitiveKind:
    """Return the primitive a worm is in at the start of its next step: its bout's, else a run."""
    if state is None or state.bout is None:
        return "run"
    return state.bout.kind
