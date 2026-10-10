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
    modules = [str(m) for m in getattr(brain, "sensory_modules", [])]
    assert ("internal_state" in modules) is (arm != "blind")
    assert getattr(brain, "freeze_updates", False) is (arm == "internal_frozen")
