"""G.0's configs: every arm is its committed parent with the listed brain keys alone changed.

Covers the connectome-ppo-brain requirements "Gap-junction weight options" and "Gap-outlier wirings"
through the configs that run them: each arm loads, differs from its parent only where it says, and
builds a brain.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path
from typing import Any

import pytest
from quantumnematode.brain.arch.connectome_ppo import ConnectomePPOBrain, ConnectomePPOBrainConfig
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.utils.config_loader import load_simulation_config

_root = Path(__file__).resolve()
while _root != _root.parent and not (_root / "scripts" / "campaigns").is_dir():
    _root = _root.parent
sys.path.insert(0, str(_root / "scripts" / "campaigns"))

import generate_gap_operating_point_configs as gen  # noqa: E402  # pyright: ignore[reportMissingImports]

_THERMAL = _root / "configs" / "scenarios" / "thermal_foraging"


@functools.cache
def _loaded(stem: str) -> dict[str, Any]:
    return load_simulation_config(str(_THERMAL / f"{stem}.yml")).model_dump()


def test_there_are_28_arms() -> None:
    """Four arms at two levels, two at three, two at seven profile levels."""
    assert len(gen.CHILDREN) == 4 * 2 + 2 * 3 + 2 * 7


@pytest.mark.parametrize("child", sorted(gen.CHILDREN))
def test_each_arm_is_committed_and_differs_from_its_parent_in_the_listed_keys_alone(
    child: str,
) -> None:
    """Each arm's file is the generator's output and its parent with only the listed changes."""
    path, text = gen.derive(child)
    assert path.read_text() == text
    parent, changes = gen.CHILDREN[child]
    assert changes["connectome_source"] == "emmons_2024_hermaphrodite"
    got, expected = _loaded(child), dict(_loaded(parent))
    expected["brain"] = {**expected["brain"]}
    expected["brain"]["config"] = {**expected["brain"]["config"], **changes}
    assert got == expected


@pytest.mark.parametrize("child", sorted(gen.CHILDREN))
def test_each_arm_builds_a_brain(child: str) -> None:
    """No arm names a combination the brain refuses."""
    container = load_simulation_config(str(_THERMAL / f"{child}.yml")).brain
    assert container is not None
    assert isinstance(container.config, ConnectomePPOBrainConfig)
    config = container.config.model_copy(update={"seed": 9851})
    ConnectomePPOBrain(config=config, device=DeviceType.CPU)


def test_the_frozen_floors_freeze_and_the_learners_learn() -> None:
    """Every frozen arm comes from the frozen parent and says so in its name."""
    for child, (parent, _) in gen.CHILDREN.items():
        frozen = _loaded(child)["brain"]["config"]["freeze_updates"]
        assert frozen is (parent == gen.FROZEN_PARENT), child
        assert child.endswith("_frozen_t35") is frozen, child
