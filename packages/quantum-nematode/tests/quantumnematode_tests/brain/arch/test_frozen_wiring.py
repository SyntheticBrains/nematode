"""A frozen-wiring learner, and the boundary null's boundary under the body drive.

Covers the connectome-ppo-brain requirements "A frozen-wiring learner" (the wiring is read, not
written; off is byte-identical) and "The boundary null's body-drive boundary" (the boundary covers
the cells the body reads).
"""

from __future__ import annotations

import pytest
import torch
from quantumnematode.agent import QuantumNematodeAgent, RewardConfig, SatietyConfig
from quantumnematode.brain.arch.connectome_ppo import (
    ConnectomePPOBrain,
    ConnectomePPOBrainConfig,
    boundary_neurons,
)
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.connectome.loader import (
    load_emmons_2024_hermaphrodite,
    load_emmons_2024_neuromuscular,
)
from quantumnematode.env.continuous_2d import Continuous2DEnvironment, Continuous2DParams
from quantumnematode.utils.config_loader import SensingConfig


def _brain(**overrides: object) -> ConnectomePPOBrain:
    config = ConnectomePPOBrainConfig(
        seed=0,
        action_mode="continuous",
        action_space="body_drive",
        connectome_source="emmons_2024_hermaphrodite",
        **overrides,  # type: ignore[arg-type]
    )
    return ConnectomePPOBrain(config=config, device=DeviceType.CPU)


class TestFrozenWiring:
    def test_off_keeps_the_chemical_weights_first(self) -> None:
        """Off is byte-identical: the chemical weights lead the learnable parameters as before."""
        topo = _brain().topology
        assert topo.learnable_parameters[0] is topo.w_chem

    def test_the_wiring_is_read_not_written(self) -> None:
        """A PPO update leaves the chemical weights at their draw and trains the rest."""
        brain = _brain(freeze_wiring=True, rollout_buffer_size=16, num_minibatches=2, num_epochs=2)
        topo = brain.topology
        assert not any(p is topo.w_chem for p in topo.learnable_parameters)
        w_chem = topo.w_chem.detach().clone()
        gains = topo.body_drive_log_gain.detach().clone()
        env = Continuous2DEnvironment(
            continuous=Continuous2DParams(
                world_size_mm=20.0,
                allow_reversal=True,
                body_model="kinematic",
            ),
            seed=0,
        )
        agent = QuantumNematodeAgent(
            brain=brain,
            env=env,
            satiety_config=SatietyConfig(initial_satiety=100.0),
            sensing_config=SensingConfig(),
        )
        agent.run_episode(RewardConfig(), max_steps=40)
        assert torch.equal(topo.w_chem, w_chem)
        assert topo.w_chem.grad is None
        assert not torch.equal(topo.body_drive_log_gain, gains)

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            (
                {"learning_rule": "three_factor", "enable_activity_traces": True},
                "requires learning_rule='ppo'",
            ),
            ({"freeze_updates": True}, "redundant under freeze_updates"),
        ],
    )
    def test_refusals(self, overrides: dict[str, object], match: str) -> None:
        settings: dict[str, object] = {"action_mode": "continuous", "freeze_wiring": True}
        settings.update(overrides)
        with pytest.raises(ValueError, match=match):
            ConnectomePPOBrainConfig(**settings)  # type: ignore[arg-type]


class TestBodyDriveBoundary:
    def test_the_boundary_is_every_cell_the_body_reads(self) -> None:
        """Under the body drive the motor side is every cell with a junction, motor classes too."""
        connectome = load_emmons_2024_hermaphrodite()
        _, pooled = boundary_neurons(connectome)
        _, body = boundary_neurons(connectome, body_drive=True)
        junction_cells = {j.pre for j in load_emmons_2024_neuromuscular()}
        assert body == frozenset(junction_cells)
        assert pooled <= body
        assert len(body) == 162

    def test_edges_into_the_cells_the_body_reads_are_the_wild_types(self) -> None:
        """The boundary null holds every chemical edge into a cell the body drive reads."""
        wild = _brain().topology
        null = _brain(wiring="rewired_boundary_held").topology
        _, body = boundary_neurons(load_emmons_2024_hermaphrodite(), body_drive=True)
        columns = torch.tensor([wild._idx[c] for c in sorted(body)])
        assert torch.equal(wild.m_chem[:, columns], null.m_chem[:, columns])
        assert not torch.equal(wild.m_chem, null.m_chem)
