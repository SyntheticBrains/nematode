"""Per-connection synapse signs on the connectome brain.

Covers the connectome-ppo-brain requirement "Per-connection synapse signs": a measured inhibitory
connection is inhibitory; magnitudes match the random-sign build; the models differ only where the
table corrects the rule; a null's signs travel with its edges; the atlas model is unchanged on a
null; PPO holds per-connection signs. Also the requirement "Sign overrides excluded by
citation": excluding the López-Cruz overrides rebuilds the earlier signs, an unknown or misplaced
key is refused, and the configs behind recorded runs exclude them.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pytest
import torch
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.arch.connectome_ppo import ConnectomePPOBrain, ConnectomePPOBrainConfig
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.connectome.loader import load_cook_2019_hermaphrodite
from quantumnematode.connectome.neurons import NEURON_CLASSIFICATION
from quantumnematode.connectome.neurotransmitters import sign_for
from quantumnematode.connectome.signs import per_connection_signs
from quantumnematode.utils.config_loader import load_simulation_config

_REPO = Path(__file__).resolve().parents[6]
_T35 = (
    _REPO
    / "configs/scenarios/thermal_foraging"
    / "connectomeppo_small_continuous2d_thermal_klinotaxis_t35.yml"
)
_SEED = 7


def _brain(**overrides: object) -> ConnectomePPOBrain:
    container = load_simulation_config(str(_T35)).brain
    assert container is not None
    assert isinstance(container.config, ConnectomePPOBrainConfig)
    config = container.config.model_copy(update={"seed": _SEED, **overrides})
    return ConnectomePPOBrain(config=config, device=DeviceType.CPU)


def _tensor(brain: ConnectomePPOBrain, name: str) -> torch.Tensor:
    value = getattr(brain.topology, name)
    assert isinstance(value, torch.Tensor)
    return value.detach()


def _edge_signs(brain: ConnectomePPOBrain) -> dict[tuple[str, str], int]:
    """Read each existing edge's sign back from the topology, keyed by name."""
    names = brain.topology.neuron_names
    mask, sign = _tensor(brain, "m_chem"), _tensor(brain, "chem_sign")
    return {(names[i], names[j]): int(sign[i, j]) for i, j in (mask != 0).nonzero().tolist()}


@pytest.fixture(scope="module")
def wild() -> dict[str, ConnectomePPOBrain]:
    """Build the wild type at one seed under each sign model."""
    return {model: _brain(synapse_signs=model) for model in ("random", "atlas", "per_connection")}


class TestWildType:
    def test_a_measured_inhibitory_connection_is_inhibitory(
        self,
        wild: dict[str, ConnectomePPOBrain],
    ) -> None:
        """Scenario "A measured inhibitory connection is inhibitory": AWC -> AIY flips."""
        per, atlas = _edge_signs(wild["per_connection"]), _edge_signs(wild["atlas"])
        for edge in (("AWCL", "AIYL"), ("AWCR", "AIYR")):
            assert per[edge] == -1
            assert atlas[edge] == 1

    def test_magnitudes_match_the_random_sign_build(
        self,
        wild: dict[str, ConnectomePPOBrain],
    ) -> None:
        """Scenario "Magnitudes match the random-sign build"."""
        assert torch.equal(
            _tensor(wild["per_connection"], "w_chem").abs(),
            _tensor(wild["random"], "w_chem").abs(),
        )

    def test_the_models_differ_only_where_the_table_corrects_the_rule(
        self,
        wild: dict[str, ConnectomePPOBrain],
    ) -> None:
        """Scenario "The models differ only where the table corrects the rule": the 358."""
        per, atlas = _edge_signs(wild["per_connection"]), _edge_signs(wild["atlas"])
        differ = {edge for edge in per if per[edge] != atlas[edge]}
        table = per_connection_signs(load_cook_2019_hermaphrodite())
        corrected = {
            edge
            for edge, entry in table.items()
            if entry.sign and entry.sign != (sign_for(NEURON_CLASSIFICATION[edge[0]][1]) or 0)
        }
        assert differ == corrected
        assert len(differ) == 358
        assert sum(1 for s in per.values() if s) == sum(1 for s in atlas.values() if s) == 3176


def _outgoing(signs: dict[tuple[str, str], int]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = defaultdict(list)
    for (pre, _post), sign in signs.items():
        out[pre].append(sign)
    return {pre: sorted(values) for pre, values in out.items()}


class TestNulls:
    @pytest.mark.parametrize("wiring", ["rewired_chemical_only", "rewired_degree_preserving"])
    def test_a_null_s_signs_travel_with_its_edges(
        self,
        wild: dict[str, ConnectomePPOBrain],
        wiring: str,
    ) -> None:
        """Scenario "A null's signs travel with its edges": outgoing sign multisets are kept."""
        null = _brain(synapse_signs="per_connection", wiring=wiring)
        null_signs = _edge_signs(null)
        assert set(null_signs) != set(_edge_signs(wild["per_connection"]))
        assert _outgoing(null_signs) == _outgoing(_edge_signs(wild["per_connection"]))

    def test_the_atlas_model_is_unchanged_on_a_null(self) -> None:
        """Scenario "The atlas model is unchanged on a null": every edge signed by its pre."""
        null = _brain(synapse_signs="atlas", wiring="rewired_chemical_only")
        by_rule = {
            edge: sign_for(NEURON_CLASSIFICATION[edge[0]][1]) or 0 for edge in _edge_signs(null)
        }
        assert _edge_signs(null) == by_rule


class TestEnforcement:
    def test_ppo_holds_per_connection_signs(self) -> None:
        """Scenario "PPO holds per-connection signs": no grounded weight crosses zero."""
        brain = _brain(
            synapse_signs="per_connection",
            enforce_synapse_signs=True,
            learning_rate=0.05,
            rollout_buffer_size=16,
            num_minibatches=2,
            num_epochs=2,
        )
        signs = _tensor(brain, "chem_sign")
        torch.manual_seed(0)
        for step in range(32):
            params = BrainParams(
                food_concentration=0.1 * (step % 5),
                food_lateral_gradient=0.05 * (step % 3) - 0.05,
                food_dconcentration_dt=0.01 * step,
                temperature=20.0 + step % 4,
                temperature_lateral_gradient=0.1,
                temperature_ddt=0.01,
                cultivation_temperature=20.0,
            )
            brain.run_brain(
                params,
                reward=None,
                input_data=None,
                top_only=False,
                top_randomize=False,
            )
            brain.learn(params, reward=float(step % 3) - 1.0, episode_done=False)
        weights = _tensor(brain, "w_chem")
        assert bool((weights[signs > 0] >= 0).all())
        assert bool((weights[signs < 0] <= 0).all())

    def test_enforcing_random_signs_is_still_refused(self) -> None:
        with pytest.raises(ValueError, match="requires synapse_signs='atlas' or 'per_connection'"):
            ConnectomePPOBrainConfig(enforce_synapse_signs=True)

    def test_a_checkpoint_from_another_sign_model_is_refused(
        self,
        wild: dict[str, ConnectomePPOBrain],
    ) -> None:
        with pytest.raises(ValueError, match="synapse-sign model"):
            wild["per_connection"].load_weight_components(
                wild["atlas"].get_weight_components(),
            )


_LAWN = "foraging/connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_lawns_"
_THERMAL = "thermal_foraging/connectomeppo_small_continuous2d_thermal_klinotaxis_"
# The per-connection configs behind recorded runs, which keep the table they ran with.
_RECORDED = (
    *(f"{_LAWN}modulated_perconn{v}.yml" for v in ("", "_frozen", "_pdfr1", "_satiety", "_tph1")),
    *(
        f"{_THERMAL}{null}per_connection_signs{v}_t35.yml"
        for null in ("", "rewired_chemical_null_")
        for v in ("", "_frozen")
    ),
)


class TestSignOverrideExclude:
    def test_excluding_the_lopez_cruz_overrides_rebuilds_the_earlier_signs(
        self,
        wild: dict[str, ConnectomePPOBrain],
    ) -> None:
        """Scenario "Excluding the López-Cruz overrides reproduces the earlier brain"."""
        excluded = _edge_signs(
            _brain(synapse_signs="per_connection", sign_override_exclude=["lopezcruz2019"]),
        )
        earlier = per_connection_signs(
            load_cook_2019_hermaphrodite(),
            exclude_citations=("lopezcruz2019",),
        )
        assert excluded == {edge: earlier[edge].sign for edge in excluded}
        full = _edge_signs(wild["per_connection"])
        differ = {edge for edge in full if full[edge] != excluded[edge]}
        assert len(differ) == 16
        assert {(full[e], excluded[e]) for e in differ} == {(-1, 1)}

    @pytest.mark.parametrize(
        ("changes", "match"),
        [
            (
                {"synapse_signs": "per_connection", "sign_override_exclude": ["nobody2099"]},
                "nobody",
            ),
            ({"synapse_signs": "atlas", "sign_override_exclude": ["lopezcruz2019"]}, "requires"),
            ({"sign_override_exclude": ["lopezcruz2019"]}, "requires"),
        ],
    )
    def test_an_unknown_or_misplaced_key_is_refused(
        self,
        changes: dict[str, object],
        match: str,
    ) -> None:
        """Scenario "An unknown or misplaced key is refused"."""
        with pytest.raises(ValueError, match=match):
            ConnectomePPOBrainConfig(**changes)  # type: ignore[arg-type]

    def test_a_config_copied_past_its_validators_is_refused_at_construction(self) -> None:
        """``model_copy`` skips validators, so the brain repeats the check when it is built."""
        container = load_simulation_config(str(_T35)).brain
        assert container is not None
        assert isinstance(container.config, ConnectomePPOBrainConfig)
        config = container.config.model_copy(
            update={"seed": _SEED, "synapse_signs": "atlas", "sign_override_exclude": ["x"]},
        )
        with pytest.raises(ValueError, match="requires synapse_signs='per_connection'"):
            ConnectomePPOBrain(config=config, device=DeviceType.CPU)

    @pytest.mark.parametrize("config", _RECORDED)
    def test_the_configs_behind_recorded_runs_exclude_them(self, config: str) -> None:
        container = load_simulation_config(str(_REPO / "configs/scenarios" / config)).brain
        assert container is not None
        assert isinstance(container.config, ConnectomePPOBrainConfig)
        assert container.config.sign_override_exclude == ["lopezcruz2019"]


class TestChemicalWeightScale:
    def test_unit_scale_is_the_unscaled_build(self, wild: dict[str, ConnectomePPOBrain]) -> None:
        scaled = _brain(synapse_signs="atlas", chemical_weight_scale=1.0)
        assert torch.equal(_tensor(scaled, "w_chem"), _tensor(wild["atlas"], "w_chem"))

    def test_a_scale_multiplies_every_weight_and_keeps_the_signs(
        self,
        wild: dict[str, ConnectomePPOBrain],
    ) -> None:
        scaled = _brain(synapse_signs="per_connection", chemical_weight_scale=0.42)
        base = _tensor(wild["per_connection"], "w_chem")
        assert torch.allclose(_tensor(scaled, "w_chem"), base * 0.42)
        assert torch.equal(
            _tensor(scaled, "chem_sign"),
            _tensor(wild["per_connection"], "chem_sign"),
        )
