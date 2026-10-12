"""D.1's lawn-cell configs are the generator's, and load as the design sets them.

Covers the patchy-lawns change's cell: lawns in place of point food, signed speed with turns of up
to half a revolution per step, no reward term that favours a state, and the three arms' modules.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
from quantumnematode.utils.config_loader import load_simulation_config

_REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO / "scripts" / "campaigns"))

import generate_lawn_configs as gen  # noqa: E402  # pyright: ignore[reportMissingImports]


@pytest.mark.parametrize("arm", gen.ARMS)
def test_the_committed_lawn_configs_are_the_generators(arm: str) -> None:
    """Each config on disk is what the generator writes, and it loads as the design sets it."""
    path, text = gen.derive(arm)
    assert path.read_text() == text
    config = load_simulation_config(str(path))
    assert config.max_steps == gen.MAX_STEPS
    assert config.environment is not None
    assert config.environment.foraging is not None
    assert config.environment.foraging.food_model == "lawns"
    continuous = config.environment.continuous
    assert continuous is not None
    assert continuous.allow_reversal
    assert continuous.max_turn_rad == pytest.approx(math.pi, abs=1e-5)
    assert config.reward is not None
    for key in gen.ZERO_SHAPING:
        assert getattr(config.reward, key) == 0
    assert config.brain is not None
    brain = config.brain.config
    assert getattr(brain, "signed_speed", False)
    assert getattr(brain, "entropy_coef", None) == gen.ENTROPY_COEF
    modules = [str(m) for m in getattr(brain, "sensory_modules", [])]
    assert ("internal_state" in modules) is (arm != "blind")
    assert getattr(brain, "freeze_updates", False) is (arm == "internal_frozen")


@pytest.mark.parametrize("cell", [c for c in gen.CELLS if c != "patch"])
@pytest.mark.parametrize("arm", gen.LEAVING_ARMS)
def test_the_committed_leaving_configs_are_the_generators(arm: str, cell: str) -> None:
    """Each leaving cell is D.1's cell with its own lawns, and loads."""
    path, text = gen.derive(arm, cell)
    assert path.read_text() == text
    config = load_simulation_config(str(path))
    assert config.environment is not None
    assert config.environment.foraging is not None
    lawns = config.environment.foraging.lawns
    assert lawns is not None
    assert lawns.radius_mm == gen.CELLS[cell]["radius_mm"]
    assert lawns.count == gen.CELLS[cell]["count"]
    assert lawns.start_on_lawn


@pytest.mark.parametrize("arm", gen.CONNECTOME_ARMS)
def test_the_committed_connectome_lawn_configs_are_the_generators(arm: str) -> None:
    """Each connectome arm is the connectome on D.1's lawn cell, at its entropy, and loads."""
    path, text = gen.derive_connectome(arm)
    assert path.read_text() == text
    config = load_simulation_config(str(path))
    assert config.max_steps == gen.MAX_STEPS
    assert config.brain is not None
    assert config.brain.name == "connectomeppo"
    brain = config.brain.config
    level = arm.removeprefix("connectome_").removesuffix("_frozen")
    assert getattr(brain, "entropy_coef", None) == gen.CONNECTOME_ENTROPY[level]
    assert getattr(brain, "freeze_updates", False) is arm.endswith("_frozen")
    assert getattr(brain, "connectome_source", None) == "emmons_2024_hermaphrodite"
    assert config.environment is not None
    assert config.environment.foraging is not None
    assert config.environment.foraging.food_model == "lawns"
    patch = load_simulation_config(str(gen.FORAGING / f"{gen.stem('internal')}.yml"))
    assert patch.environment is not None
    assert config.environment.foraging == patch.environment.foraging
    assert config.reward == patch.reward


@pytest.mark.parametrize("arm", list(gen.MODULATED_ARMS))
def test_the_committed_modulated_configs_are_the_generators(arm: str) -> None:
    """Each modulated arm is stage 0's entropy-0.005 connectome with the field on, and loads."""
    path, text = gen.derive_modulated(arm)
    assert path.read_text() == text
    config = load_simulation_config(str(path))
    assert config.brain is not None
    brain = config.brain.config
    assert getattr(brain, "modulators", False) is True
    for key, value in gen.MODULATED_ARMS[arm].items():
        assert getattr(brain, key) == value
    stage0 = load_simulation_config(str(gen.derive_connectome("connectome_e005")[0]))
    assert stage0.brain is not None
    expected = stage0.brain.config.model_copy(
        update={"modulators": True, **gen.MODULATED_ARMS[arm]},
    )
    assert brain == expected
    assert config.environment == stage0.environment
    assert config.reward == stage0.reward


def test_an_acute_arm_runs_its_trained_arm_s_config() -> None:
    """``modulated+tph1`` has the intact arm's stem: its config and its training log."""
    for arm in gen.ACUTE_ARMS:
        assert gen.stem(arm) == gen.stem(arm.split("+")[0])
    assert gen.stem("modulated_atlas+pdfr1") == gen.stem("modulated_atlas")


@pytest.mark.parametrize("arm", gen.BODY_ARMS)
def test_the_body_demo_configs_are_the_generators(arm: str) -> None:
    """Each body demo arm is D.1's lawn arm through the kinematic body, and loads."""
    path, text = gen.derive_body(arm)
    assert path.read_text() == text
    config = load_simulation_config(str(path))
    assert config.environment is not None
    assert config.environment.continuous is not None
    assert config.environment.foraging is not None
    assert config.environment.continuous.body_model == "kinematic"
    assert config.environment.foraging.food_model == "lawns"
    assert config.brain is not None
    assert config.brain.config.action_space == "body_drive"
    assert getattr(config.brain.config, "freeze_updates", False) is arm.endswith("_frozen")
