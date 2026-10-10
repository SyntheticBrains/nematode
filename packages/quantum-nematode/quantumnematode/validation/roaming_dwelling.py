"""Roaming and dwelling from speed and angular speed, read alike on real and simulated worms.

Roaming and dwelling are the two persistent states of a worm on food: fast and straight, or slow and
turning (Ben Arous et al. 2009; Flavell et al. 2013). They are classified in three steps.

1. **Measures.** A track is sampled at the simulation's step, 5 worm-seconds. Each step has a speed,
   its displacement over its duration, and a turn, the angle between its displacement and the
   previous step's, the three-point angle real-worm trackers use. A window of two steps, 10 seconds,
   has the mean speed (mm/s) and the mean turn per second (degrees/s).
2. **A model.** Each window's ``(log speed, angular speed)`` is scored under a two-state hidden
   Markov model with Gaussian emissions, fitted on real worms' tracks measured exactly as here
   against the authors' own labels, and decoded by Viterbi within each run of windows on food.
   Off food a worm searches and disperses rather than roaming or dwelling, so off-food windows are
   not classified.
3. **The authors' own form**, a line ``speed * slope > angular speed`` smoothed by their categorical
   model (Flavell et al. 2013; Scheer & Bargmann 2023), is kept beside it: at the 5 s step a
   dwelling worm's turn is close to random, and the line reproduces the authors' labels less well.

Real and simulated tracks go through the same functions with the same slope and model: nothing is
refitted on simulated data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

STEP_SECONDS = 5.0
WINDOW_STEPS = 2
OFF_FOOD = -1
DWELLING = 0
ROAMING = 1

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
REFERENCE_HMM_PATH = _PROJECT_ROOT / "data" / "roaming_dwelling" / "reference_hmm.json"
CALIBRATED_HMM_PATH = _PROJECT_ROOT / "data" / "roaming_dwelling" / "calibration_retry.json"


def step_measures(
    x: np.ndarray,
    y: np.ndarray,
    step_seconds: float = STEP_SECONDS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return each step's speed (mm/s) and turn (degrees) from positions sampled once per step.

    Step ``i`` runs from sample ``i`` to ``i + 1``. Its turn is the angle between its displacement
    and step ``i - 1``'s: NaN for the first step, and wherever either displacement is zero.
    """
    dx, dy = np.diff(np.asarray(x, dtype=float)), np.diff(np.asarray(y, dtype=float))
    length = np.hypot(dx, dy)
    speed = length / step_seconds
    turn = np.full(len(dx), np.nan)
    if len(dx) > 1:
        dot = dx[1:] * dx[:-1] + dy[1:] * dy[:-1]
        norms = length[1:] * length[:-1]
        with np.errstate(invalid="ignore", divide="ignore"):
            cosine = np.where(norms > 0, dot / norms, np.nan)
        turn[1:] = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return speed, turn


def window_measures(
    x: np.ndarray,
    y: np.ndarray,
    step_seconds: float = STEP_SECONDS,
    window_steps: int = WINDOW_STEPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return each window's mean speed (mm/s) and angular speed (degrees/s).

    Windows are consecutive, non-overlapping runs of ``window_steps`` steps; a trailing partial
    window is dropped. A window's angular speed is its steps' mean turn per second, ignoring NaN
    turns; NaN if every turn in it is.
    """
    speed, turn = step_measures(x, y, step_seconds)
    n = len(speed) // window_steps
    speed_w = speed[: n * window_steps].reshape(n, window_steps).mean(axis=1)
    turns = turn[: n * window_steps].reshape(n, window_steps)
    with np.errstate(invalid="ignore"):
        counts = np.sum(~np.isnan(turns), axis=1)
        sums = np.nansum(turns, axis=1)
        angular = np.where(counts > 0, sums / np.maximum(counts, 1) / step_seconds, np.nan)
    return speed_w, angular


def roaming_observations(speed: np.ndarray, angular: np.ndarray, slope: float) -> np.ndarray:
    """Return 1 where ``speed * slope > angular speed`` (a roaming observation), else 0.

    A window whose angular speed is NaN (no measurable turn) is judged on speed alone against a
    zero turn, so a straight run counts as roaming and a stationary one as dwelling.
    """
    return (np.asarray(speed) * slope > np.nan_to_num(np.asarray(angular), nan=0.0)).astype(int)


def _viterbi(
    log_pi0: np.ndarray,
    log_transitions: np.ndarray,
    log_likelihood: np.ndarray,
) -> np.ndarray:
    """Return the most likely state sequence given each step's log-likelihood per state."""
    n, k = log_likelihood.shape
    if n == 0:
        return np.zeros(0, dtype=int)
    score = np.empty((n, k))
    back = np.zeros((n, k), dtype=int)
    score[0] = log_pi0 + log_likelihood[0]
    for t in range(1, n):
        candidates = score[t - 1][:, None] + log_transitions
        back[t] = np.argmax(candidates, axis=0)
        score[t] = candidates[back[t], np.arange(k)] + log_likelihood[t]
    states = np.empty(n, dtype=int)
    states[-1] = int(np.argmax(score[-1]))
    for t in range(n - 1, 0, -1):
        states[t - 1] = back[t, states[t]]
    return states


@dataclass(frozen=True)
class CategoricalHMM:
    """A two-state hidden Markov model over binary observations, in log space.

    ``log_pi0[k]`` is the log probability of starting in state ``k``; ``log_transitions[j, k]`` of
    moving from ``j`` to ``k``; ``log_emissions[k, c]`` of state ``k`` emitting category ``c``.
    State 1 is roaming and state 0 dwelling.
    """

    log_pi0: np.ndarray
    log_transitions: np.ndarray
    log_emissions: np.ndarray

    def viterbi(self, observations: np.ndarray) -> np.ndarray:
        """Return the most likely state sequence for a run of binary observations."""
        obs = np.asarray(observations, dtype=int)
        return _viterbi(self.log_pi0, self.log_transitions, self.log_emissions[:, obs].T)

    def sample(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Draw ``n`` steps of states and observations from the model."""
        pi0, trans, emit = (
            np.exp(a) for a in (self.log_pi0, self.log_transitions, self.log_emissions)
        )
        states = np.empty(n, dtype=int)
        obs = np.empty(n, dtype=int)
        states[0] = rng.choice(len(pi0), p=pi0 / pi0.sum())
        for t in range(n):
            if t:
                row = trans[states[t - 1]]
                states[t] = rng.choice(len(row), p=row / row.sum())
            probs = emit[states[t]]
            obs[t] = rng.choice(len(probs), p=probs / probs.sum())
        return states, obs


@dataclass(frozen=True)
class Classifier:
    """A calibrated line and a smoothing model: everything needed to read a track's states."""

    slope: float
    hmm: CategoricalHMM

    def states(self, speed: np.ndarray, angular: np.ndarray, on_food: np.ndarray) -> np.ndarray:
        """Return each window's state: ``ROAMING``, ``DWELLING``, or ``OFF_FOOD``.

        The model decodes each contiguous run of on-food windows on its own, so a run's states do
        not depend on what happened before the worm left food.
        """
        on = np.asarray(on_food, dtype=bool)
        result = np.full(len(on), OFF_FOOD, dtype=int)
        obs = roaming_observations(speed, angular, self.slope)
        for start, stop in on_food_runs(on):
            result[start:stop] = self.hmm.viterbi(obs[start:stop])
        return result


SPEED_FLOOR_MM_S = 0.001


def window_features(speed: np.ndarray, angular: np.ndarray) -> np.ndarray:
    """Return each window's ``(log(speed + 0.001 mm/s), angular speed)``, a missing turn as 0."""
    return np.column_stack(
        [
            np.log(np.asarray(speed, dtype=float) + SPEED_FLOOR_MM_S),
            np.nan_to_num(np.asarray(angular, dtype=float), nan=0.0),
        ],
    )


@dataclass(frozen=True)
class GaussianHMM:
    """A two-state hidden Markov model with Gaussian emissions over window features.

    State 1 is roaming and state 0 dwelling. ``means`` is ``(2, d)`` and ``covariances``
    ``(2, d, d)``.
    """

    log_pi0: np.ndarray
    log_transitions: np.ndarray
    means: np.ndarray
    covariances: np.ndarray

    def log_likelihood(self, features: np.ndarray) -> np.ndarray:
        """Return each window's log-density under each state's Gaussian, ``(n, 2)``."""
        x = np.atleast_2d(np.asarray(features, dtype=float))
        out = np.empty((len(x), len(self.means)))
        for k, (mean, cov) in enumerate(zip(self.means, self.covariances, strict=True)):
            diff = x - mean
            inverse = np.linalg.inv(cov)
            _, logdet = np.linalg.slogdet(cov)
            mahalanobis = np.einsum("ni,ij,nj->n", diff, inverse, diff)
            out[:, k] = -0.5 * (mahalanobis + logdet + x.shape[1] * np.log(2 * np.pi))
        return out

    def viterbi(self, features: np.ndarray) -> np.ndarray:
        """Return the most likely state sequence for a run of windows' features."""
        if len(features) == 0:
            return np.zeros(0, dtype=int)
        return _viterbi(self.log_pi0, self.log_transitions, self.log_likelihood(features))

    @classmethod
    def fit_labelled(cls, runs: list[tuple[np.ndarray, np.ndarray]]) -> GaussianHMM:
        """Fit from labelled runs of ``(features, states)``, with no iterative search.

        Each state's mean and full covariance come from its windows; the transition probabilities
        are counted from consecutive windows within each run; the initial distribution is the
        states' shares.
        """
        features = np.vstack([f for f, _ in runs])
        states = np.concatenate([s for _, s in runs])
        k = 2
        means = np.array([features[states == j].mean(axis=0) for j in range(k)])
        covariances = np.array([np.cov(features[states == j].T) for j in range(k)])
        counts = np.zeros((k, k))
        for _, s in runs:
            np.add.at(counts, (s[:-1], s[1:]), 1)
        transitions = counts / counts.sum(axis=1, keepdims=True)
        shares = np.bincount(states, minlength=k) / len(states)
        return cls(
            log_pi0=np.log(shares),
            log_transitions=np.log(transitions),
            means=means,
            covariances=covariances,
        )


@dataclass(frozen=True)
class GaussianClassifier:
    """Read a track's states from its windows' features with a Gaussian-emission model."""

    hmm: GaussianHMM

    def states(self, speed: np.ndarray, angular: np.ndarray, on_food: np.ndarray) -> np.ndarray:
        """Return each window's state: ``ROAMING``, ``DWELLING``, or ``OFF_FOOD``."""
        on = np.asarray(on_food, dtype=bool)
        result = np.full(len(on), OFF_FOOD, dtype=int)
        features = window_features(speed, angular)
        for start, stop in on_food_runs(on):
            result[start:stop] = self.hmm.viterbi(features[start:stop])
        return result


def on_food_runs(on_food: np.ndarray) -> list[tuple[int, int]]:
    """Return ``(start, stop)`` index pairs of each contiguous run of True values."""
    on = np.concatenate([[False], np.asarray(on_food, dtype=bool), [False]])
    edges = np.flatnonzero(np.diff(on.astype(int)))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True)]


def load_reference_hmm(path: Path = REFERENCE_HMM_PATH) -> CategoricalHMM:
    """Load the vendored reference model's parameters (see its provenance beside it)."""
    data = json.loads(path.read_text())
    return CategoricalHMM(
        log_pi0=np.asarray(data["log_pi0"], dtype=float),
        log_transitions=np.asarray(data["log_transitions"], dtype=float),
        log_emissions=np.asarray(data["log_emissions"], dtype=float),
    )


def load_calibrated_hmm(path: Path = CALIBRATED_HMM_PATH) -> GaussianHMM:
    """Load the Gaussian-emission model calibrated on real worms at the 5 s step."""
    params = json.loads(path.read_text())["parameters"]
    return GaussianHMM(
        log_pi0=np.asarray(params["log_pi0"], dtype=float),
        log_transitions=np.asarray(params["log_transitions"], dtype=float),
        means=np.asarray(params["means"], dtype=float),
        covariances=np.asarray(params["covariances"], dtype=float),
    )


def agreement(predicted: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Return the share of windows that agree and Cohen's kappa, over windows both classified."""
    p, lab = np.asarray(predicted), np.asarray(labels)
    keep = (p != OFF_FOOD) & (lab != OFF_FOOD)
    p, lab = p[keep], lab[keep]
    if p.size == 0:
        return {"n": 0.0, "accuracy": float("nan"), "kappa": float("nan")}
    accuracy = float(np.mean(p == lab))
    chance = float(np.mean(p == ROAMING) * np.mean(lab == ROAMING)) + float(
        np.mean(p == DWELLING) * np.mean(lab == DWELLING),
    )
    kappa = (accuracy - chance) / (1.0 - chance) if chance < 1.0 else float("nan")
    return {"n": float(p.size), "accuracy": accuracy, "kappa": float(kappa)}


def calibrate_slope(
    tracks: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    hmm: CategoricalHMM,
    candidates: np.ndarray,
) -> tuple[float, float]:
    """Return the candidate slope whose decoded states best agree with the labels, and its kappa.

    Each track is ``(speed, angular speed, on_food, labels)`` per window, labels as ``ROAMING`` /
    ``DWELLING`` / ``OFF_FOOD``. Agreement is Cohen's kappa pooled over every track's windows.
    """
    best_slope, best_kappa = float(candidates[0]), -np.inf
    for slope in candidates:
        classifier = Classifier(slope=float(slope), hmm=hmm)
        predicted = np.concatenate([classifier.states(s, a, on) for s, a, on, _ in tracks])
        labels = np.concatenate([lab for *_, lab in tracks])
        kappa = agreement(predicted, labels)["kappa"]
        if kappa > best_kappa:
            best_slope, best_kappa = float(slope), kappa
    return best_slope, float(best_kappa)
