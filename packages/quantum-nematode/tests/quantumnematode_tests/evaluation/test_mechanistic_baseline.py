"""C.1's mechanistic baseline: klinokinesis, the edge response, search, and per-animal streams.

Covers the mechanistic-baseline spec's scenarios "Klinokinesis lowers hazards up-gradient", "Leaving
food raises reversals", "Search decays off food" and "An animal is the same alone and in a batch".
"""

from __future__ import annotations

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
