"""C.1's mechanistic baseline: klinokinesis, the edge response, search, and per-animal streams.

Covers the mechanistic-baseline spec's scenarios "Klinokinesis lowers hazards up-gradient", "Leaving
food raises reversals", "Search decays off food" and "An animal is the same alone and in a batch".
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
import torch
from quantumnematode.env.primitives import load_kinematics
from quantumnematode.evaluation.batched import BatchedSenses
from quantumnematode.evaluation.controllers import (
    MECHANISTIC_PARAMETERS,
    BatchedMechanistic,
)

_K = load_kinematics()
_BASE = {
    "a0": math.log(0.002),
    "a_intake": 0.0,
    "a_satiety": 0.0,
    "b0": math.log(0.01),
    "b_intake": 0.0,
    "b_satiety": 0.0,
    "v_dwell": 0.02,
    "v_roam": 0.07,
    "sigma_dwell": 40.0,
    "sigma_roam": 30.0,
    "kappa_weathervane": 10.0,
    "r_dwell": 0.05,
    "r_roam": 0.05,
    "beta_reversal": 1.0,
    "edge": 2.0,
    "o_dwell": 0.005,
    "o_roam": 0.005,
    "beta_omega": 1.0,
    "search_amplitude": 4.0,
    "search_tau_s": 300.0,
}


def _senses(n: int, *, contact: float = 1.0, derivative: float = 0.0) -> BatchedSenses:
    zeros = np.zeros(n)
    return BatchedSenses(
        concentration=zeros,
        lateral=zeros,
        derivative=np.full(n, derivative),
        food_contact=np.full(n, contact),
        satiety_fraction=np.full(n, 0.5),
        locomotor=np.zeros(n, dtype=np.int64),
        stam=np.zeros((n, 1)),
        moved=1.0,
        intake=zeros,
        max_intake=None,
        lateral_scale=1.0,
        derivative_scale=1.0,
    )


def _gens(seeds: list[int]) -> list[torch.Generator]:
    return [torch.Generator().manual_seed(s) for s in seeds]


def _rate(raw: float) -> float:
    """Invert the primitives' log-linear hazard map."""
    log_low = math.log(_K.hazard_min_per_s)
    span = math.log(_K.hazard_max_per_s) - log_low
    return math.exp(log_low + (raw + 1.0) / 2.0 * span)


def _reversal(controller: BatchedMechanistic, senses: BatchedSenses) -> float:
    return _rate(float(controller.act(senses, _gens([1]))[0, 2]))


def test_the_parameters_are_twenty_and_ordered() -> None:
    """C.1 has 20 parameters, and a vector builds the same controller as a mapping."""
    assert len(MECHANISTIC_PARAMETERS) == 20
    vector = np.array([_BASE[name] for name in MECHANISTIC_PARAMETERS])
    assert BatchedMechanistic.from_vector(vector).p == BatchedMechanistic(_BASE).p
    with pytest.raises(ValueError, match="missing"):
        BatchedMechanistic({"a0": 0.0})


def test_klinokinesis_lowers_hazards_up_gradient() -> None:
    """A positive dC/dt lowers the reversal hazard below its value at zero."""
    flat = BatchedMechanistic(_BASE)
    up = BatchedMechanistic(_BASE)
    assert _reversal(up, _senses(1, derivative=2.0)) < _reversal(flat, _senses(1))


def test_leaving_food_raises_reversals() -> None:
    """A fall in food contact multiplies the reversal hazard by exp(edge x the fall)."""
    controller = BatchedMechanistic({**_BASE, "search_amplitude": 0.0})
    before = _reversal(controller, _senses(1, contact=1.0))
    controller.previous_contact = np.array([1.0])
    after = _rate(float(controller.act(_senses(1, contact=0.5), _gens([1]))[0, 2]))
    assert after / before == pytest.approx(math.exp(_BASE["edge"] * 0.5), rel=1e-6)


def test_search_decays_off_food() -> None:
    """Off food the hazards carry 1 + A exp(-t/tau), falling as time off food grows."""
    # Switching rates so low the animal stays dwelling throughout.
    pinned = {**_BASE, "edge": 0.0, "a0": math.log(1e-12), "b0": math.log(1e-3)}
    controller = BatchedMechanistic(pinned)
    gens = _gens([1])
    controller.act(_senses(1, contact=1.0), gens)
    rates = [_rate(float(controller.act(_senses(1, contact=0.0), gens)[0, 2])) for _ in range(600)]
    assert controller.state is not None
    assert controller.state[0] == 0
    assert rates[0] > rates[299] > rates[599]
    expected = _BASE["r_dwell"] * (1.0 + _BASE["search_amplitude"] * math.exp(-600.0 / 300.0))
    assert rates[-1] == pytest.approx(expected, rel=1e-6)


def test_an_animal_is_the_same_alone_and_in_a_batch() -> None:
    """Each animal draws only from its own generator."""
    alone, crowd = BatchedMechanistic(_BASE), BatchedMechanistic(_BASE)
    gens_alone, gens_crowd = _gens([7]), _gens([3, 7, 11])
    for t in range(200):
        contact = 1.0 if t < 100 else 0.0
        a = alone.act(_senses(1, contact=contact), gens_alone)
        c = crowd.act(_senses(3, contact=contact), gens_crowd)
        assert np.array_equal(a[0], c[1])


def test_mean_action_is_refused() -> None:
    """The baseline needs each animal's generator."""
    with pytest.raises(ValueError, match="noise is None"):
        BatchedMechanistic(_BASE).act(_senses(2), None)


def _steer_senses(n: int, contact: float) -> BatchedSenses:
    senses = _senses(n, contact=contact)
    return dataclasses.replace(senses, lateral=np.linspace(-1.0, 1.0, n))


def _steering(params: dict[str, float], contact: float, seeds: list[int]) -> np.ndarray:
    controller = BatchedMechanistic(params)
    gens = _gens(seeds)
    return np.stack([controller.act(_steer_senses(len(seeds), contact), gens) for _ in range(20)])


def test_an_off_food_gain_changes_steering_only_off_food() -> None:
    """Without kappa_off_food the model is unchanged; with it, only food-free steps steer by it."""
    seeds = [3, 4, 5, 6]
    same = {**_BASE, "kappa_off_food": _BASE["kappa_weathervane"]}
    other = {**_BASE, "kappa_off_food": -_BASE["kappa_weathervane"]}
    for contact in (1.0, 0.0):
        np.testing.assert_array_equal(
            _steering(_BASE, contact, seeds),
            _steering(same, contact, seeds),
        )
    np.testing.assert_array_equal(_steering(_BASE, 1.0, seeds), _steering(other, 1.0, seeds))
    base, flipped = _steering(_BASE, 0.0, seeds), _steering(other, 0.0, seeds)
    # Off food only the run curvature (column 1) changes; speed and the three hazards do not.
    np.testing.assert_array_equal(base[..., [0, 2, 3, 4]], flipped[..., [0, 2, 3, 4]])
    assert not np.array_equal(base[..., 1], flipped[..., 1])


def test_off_food_kinematics_change_runs_only_off_food() -> None:
    """Without the off-food speed and noise the model is unchanged; with them, only food-free steps.

    On food every column is the state's; off food the speed and curvature columns change, and the
    three hazards do not.
    """
    seeds = [3, 4, 5, 6]
    off = {**_BASE, "v_off_food": 0.15, "sigma_off_food": 55.0}
    np.testing.assert_array_equal(_steering(_BASE, 1.0, seeds), _steering(off, 1.0, seeds))
    base, moved = _steering(_BASE, 0.0, seeds), _steering(off, 0.0, seeds)
    np.testing.assert_array_equal(base[..., 2:], moved[..., 2:])
    assert not np.array_equal(base[..., 0], moved[..., 0])
    assert not np.array_equal(base[..., 1], moved[..., 1])


def test_a_speed_spread_varies_speed_by_window_only() -> None:
    """Without the spread the model is unchanged; with it, speed varies across 10-s windows only.

    Within a window the speed column holds; across windows it moves; curvature and the hazards
    draw exactly what they drew without the spread, since the spread's normal adds no draw.
    """
    seeds = [3, 4, 5, 6]
    spread = {**_BASE, "speed_log_sd_dwell": 0.5, "speed_log_sd_roam": 0.5}
    base, varied = _steering(_BASE, 1.0, seeds), _steering(spread, 1.0, seeds)
    np.testing.assert_array_equal(base[..., 1:], varied[..., 1:])
    # Steps 0-9 share one window and 10-19 the next (each animal stays in its state here).
    first, second = varied[:10, :, 0], varied[10:, :, 0]
    assert np.all(first == first[0])
    assert np.all(second == second[0])
    assert not np.array_equal(first[0], second[0])
    with pytest.raises(ValueError, match="come together"):
        BatchedMechanistic({**_BASE, "speed_log_sd_dwell": 0.5})


def test_an_on_food_dwelling_rate_changes_only_dwelling_on_food() -> None:
    """Without r_dwell_on_food the model is unchanged; with it, dwelling on food reverses at it.

    Covers the mechanistic-baseline scenarios "Off food nothing changes" and "On food only dwelling
    reversals change": off food every column is the state's; on food only the reversal hazard of
    dwelling animals changes, scaled from r_dwell to r_dwell_on_food; roaming keeps r_roam.
    """
    seeds = list(range(40))
    on = {**_BASE, "r_dwell_on_food": 0.2}
    np.testing.assert_array_equal(_steering(_BASE, 0.0, seeds), _steering(on, 0.0, seeds))
    base, moved = _steering(_BASE, 1.0, seeds), _steering(on, 1.0, seeds)
    np.testing.assert_array_equal(base[..., [0, 1, 3, 4]], moved[..., [0, 1, 3, 4]])
    inside = (np.abs(base[..., 2]) < 1) & (np.abs(moved[..., 2]) < 1)
    rate = np.vectorize(_rate)
    ratio = rate(moved[..., 2][inside]) / rate(base[..., 2][inside])
    # Dwelling steps scale by 0.2 / r_dwell; roaming steps are unchanged; both occur here.
    assert np.all(np.isclose(ratio, 0.2 / _BASE["r_dwell"]) | np.isclose(ratio, 1.0))
    assert np.isclose(ratio, 0.2 / _BASE["r_dwell"]).any()
    assert np.isclose(ratio, 1.0).any()
