"""Signed speed bounds on every continuous brain, their agreement with the environment, and the
Emmons 2024 connectome source.

Covers the continuous-action-policy requirement "Signed speed bounds that agree with the
environment" (every continuous brain reads the shared bounds; a disagreeing brain and environment
are refused; signed speed needs continuous actions) and the connectome-ppo-brain requirement
"Emmons 2024 as a connectome source" (Emmons differs from Cook in four gap pairs only).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from quantumnematode.brain.arch import (
    ConnectomePPOBrain,
    ConnectomePPOBrainConfig,
)
from quantumnematode.brain.arch._policy import continuous_action_bounds
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.utils.config_loader import SimulationConfig, load_simulation_config

_REPO = Path(__file__).resolve().parents[6]
_HARD350 = (
    _REPO / "configs/scenarios/foraging/"
    "connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_hard350.yml"
)


class TestBounds:
    def test_unsigned_bounds_are_unchanged(self) -> None:
        low, high = continuous_action_bounds(signed_speed=False, device=torch.device("cpu"))
        assert low.tolist() == [0.0, -1.0]
        assert high.tolist() == [1.0, 1.0]

    def test_signed_bounds_open_the_speed(self) -> None:
        low, high = continuous_action_bounds(signed_speed=True, device=torch.device("cpu"))
        assert low.tolist() == [-1.0, -1.0]
        assert high.tolist() == [1.0, 1.0]


def _brain(name: str, *, signed: bool) -> Any:
    from quantumnematode.brain.arch.cfc_ppo import CfCBrainConfig, CfCPPOBrain
    from quantumnematode.brain.arch.lstmppo import LSTMPPOBrain, LSTMPPOBrainConfig
    from quantumnematode.brain.arch.mlpppo import MLPPPOBrain, MLPPPOBrainConfig
    from quantumnematode.brain.arch.transformer_ppo import (
        TransformerPPOBrain,
        TransformerPPOBrainConfig,
    )

    kinds: dict[str, tuple[Any, Any]] = {
        "connectome": (ConnectomePPOBrain, ConnectomePPOBrainConfig),
        "mlp": (MLPPPOBrain, MLPPPOBrainConfig),
        "lstm": (LSTMPPOBrain, LSTMPPOBrainConfig),
        "cfc": (CfCPPOBrain, CfCBrainConfig),
        "transformer": (TransformerPPOBrain, TransformerPPOBrainConfig),
    }
    brain_type, config_type = kinds[name]
    extra: dict[str, Any] = {} if name == "connectome" else {"sensory_modules": ["food_chemotaxis"]}
    config = config_type(seed=0, action_mode="continuous", signed_speed=signed, **extra)
    return brain_type(config=config, device=DeviceType.CPU)


@pytest.mark.parametrize("name", ["connectome", "mlp", "lstm", "cfc", "transformer"])
class TestEveryContinuousBrain:
    def test_signed_speed_widens_the_speed_bound(self, name: str) -> None:
        brain = _brain(name, signed=True)
        assert brain._action_low.tolist() == [-1.0, -1.0]
        assert brain._action_high.tolist() == [1.0, 1.0]

    def test_unsigned_speed_keeps_the_old_bound(self, name: str) -> None:
        brain = _brain(name, signed=False)
        assert brain._action_low.tolist() == [0.0, -1.0]


class TestAgreement:
    def _raw(self) -> dict[str, Any]:
        return yaml.safe_load(_HARD350.read_text())

    def test_a_matched_pair_loads(self) -> None:
        raw = self._raw()
        raw["environment"]["continuous"]["allow_reversal"] = True
        raw["brain"]["config"]["signed_speed"] = True
        SimulationConfig.model_validate(raw)

    def test_a_signed_brain_in_a_clamping_environment_is_refused(self) -> None:
        raw = self._raw()
        raw["brain"]["config"]["signed_speed"] = True
        with pytest.raises(
            ValueError, match="allow_reversal is False but brain.config.signed_speed"
        ):
            SimulationConfig.model_validate(raw)

    def test_a_reversing_environment_with_an_unsigned_brain_is_refused(self) -> None:
        raw = self._raw()
        raw["environment"]["continuous"]["allow_reversal"] = True
        with pytest.raises(
            ValueError, match="allow_reversal is True but brain.config.signed_speed"
        ):
            SimulationConfig.model_validate(raw)

    def test_the_committed_cells_are_unchanged(self) -> None:
        config = load_simulation_config(str(_HARD350))
        assert config.brain is not None
        assert config.brain.config.signed_speed is False

    def test_signed_speed_needs_continuous_actions(self) -> None:
        with pytest.raises(ValueError, match="requires action_mode: continuous"):
            ConnectomePPOBrainConfig(signed_speed=True)


class TestEmmonsSource:
    _GAP_PAIRS = (("ALML", "BDUL"), ("ALMR", "BDUR"), ("BDUL", "PLML"), ("BDUR", "PLMR"))

    def _topologies(self) -> tuple[Any, Any]:
        container = load_simulation_config(str(_HARD350)).brain
        assert container is not None
        assert isinstance(container.config, ConnectomePPOBrainConfig)
        built = []
        for source in ("cook_2019_hermaphrodite", "emmons_2024_hermaphrodite"):
            config = container.config.model_copy(update={"seed": 3, "connectome_source": source})
            built.append(ConnectomePPOBrain(config=config, device=DeviceType.CPU).topology)
        return built[0], built[1]

    def test_cook_is_the_default(self) -> None:
        assert ConnectomePPOBrainConfig().connectome_source == "cook_2019_hermaphrodite"

    def test_emmons_differs_from_cook_in_four_gap_pairs_only(self) -> None:
        cook, emmons = self._topologies()
        assert torch.equal(cook.m_chem, emmons.m_chem)
        assert torch.equal(cook.w_chem, emmons.w_chem)
        index = {name: i for i, name in enumerate(cook.neuron_names)}
        differs = {
            (min(a, b), max(a, b)) for a, b in torch.nonzero(cook.g_gap != emmons.g_gap).tolist()
        }
        named = {(min(index[a], index[b]), max(index[a], index[b])) for a, b in self._GAP_PAIRS}
        # Gap normalisation divides each entry by its endpoints' gap degrees, so a pair that gains
        # a junction also rescales that neuron's other entries; every changed entry must touch one
        # of the four pairs' neurons.
        touched = {i for pair in named for i in pair}
        assert named <= differs
        assert all(a in touched or b in touched for a, b in differs)

    def test_an_unknown_source_is_refused_at_construction(self) -> None:
        config = ConnectomePPOBrainConfig(seed=0).model_copy(
            update={"connectome_source": "witvliet_2021"},
        )
        with pytest.raises(ValueError, match="Unsupported connectome_source"):
            ConnectomePPOBrain(config=config, device=DeviceType.CPU)
