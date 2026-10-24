"""The batched controllers and the evaluator's surface: what each accepts and refuses.

Covers the batched-evaluation spec's scenarios "Two animals release independently" and "Random lawn
placement is refused" (with the surface's other refusals).
"""

from __future__ import annotations

import pytest
import torch
from quantumnematode.evaluation.batched import BatchedEvaluator, UnsupportedConfigError
from quantumnematode.evaluation.controllers import BatchedConnectome, BatchedMLP

from . import _parity as p


def test_two_animals_release_independently() -> None:
    """Each row's release equals one state's, and the single-state result is unchanged."""
    _, brain, _ = p.build("connectome", "uniform_lawn_flavell")
    topology = brain.topology
    torch.manual_seed(5001)
    hidden = torch.rand(2, topology.n_neurons) * 2.0 - 1.0
    rows = topology.modulator_release_batched(hidden)
    for i in range(2):
        single = topology.modulator_release(hidden[i])
        assert rows[i].tolist() == pytest.approx(list(single), abs=1e-6)
    assert topology.modulator_release(hidden[0]) == pytest.approx(
        (
            float(hidden[0, topology._serotonin_release_indices].clamp_min(0.0).mean()),
            float(hidden[0, topology._pdf_release_indices].clamp_min(0.0).mean()),
        ),
    )


def _with(config, **changes):  # type: ignore[no-untyped-def]
    environment = config.environment
    sensing = environment.get_sensing_config().model_copy(update=changes.pop("sensing", {}))
    satiety = config.satiety.model_copy(update=changes.pop("satiety", {}))
    return config.model_copy(
        update={
            "environment": environment.model_copy(update={"sensing": sensing}),
            "satiety": satiety,
        },
    )


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"satiety": {"starvation_ends_episode": True}}, "starvation_ends_episode"),
        ({"sensing": {"adaptive_chemosensor_readout": "contrast"}}, "fold-change"),
        ({"sensing": {"stam_enabled": False}}, "short-term memory"),
    ],
)
def test_the_surface_refuses_what_it_does_not_reproduce(change: dict, match: str) -> None:  # type: ignore[type-arg]
    """Each refusal names the setting outside the surface."""
    config, _, batched = p.build("constant", "scheer_lawn")
    with pytest.raises(UnsupportedConfigError, match=match):
        BatchedEvaluator(_with(config, **change), batched, [5001])


def test_a_state_dependent_noise_brain_is_refused() -> None:
    """The batched sampler gives every animal the same noise scale."""
    _, brain, _ = p.build("mlp", "scheer_lawn")
    brain.config = brain.config.model_copy(update={"continuous_std_mode": "state_dependent"})
    with pytest.raises(ValueError, match="noise depends on the state"):
        BatchedMLP(brain)


def test_a_leaky_connectome_is_refused() -> None:
    """The batched forward settles from zero each step, which leaky dynamics does not."""
    _, brain, _ = p.build("connectome", "scheer_lawn")
    brain.config = brain.config.model_copy(update={"dynamics": "leaky"})
    with pytest.raises(ValueError, match="settling"):
        BatchedConnectome(brain)
