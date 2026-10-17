"""G.0-A's selection-pilot configs: each is its committed parent with the listed keys alone changed.

Covers the connectome-ppo-brain requirement "Ohmic gap-junction coupling in the settling map"
through the configs that run it: every arm loads, differs from its parent only where it says,
and builds.
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

import generate_gap_substrate_configs as gen  # noqa: E402  # pyright: ignore[reportMissingImports]


@functools.cache
def _loaded(path: Path) -> dict[str, Any]:
    return load_simulation_config(str(path)).model_dump()


def test_each_cell_has_eleven_arms_all_on_disk() -> None:
    """Two models, two wirings, learn and frozen; the additive wild type; the MLP and its floor."""
    for arms in gen.ARMS.values():
        assert len(arms) == 11
        assert all(gen.path_of(stem).is_file() for stem in arms.values())


@pytest.mark.parametrize("child", sorted(gen.CHILDREN))
def test_each_config_is_its_parent_with_the_listed_keys_alone(child: str) -> None:
    """Each generated file is the generator's output and its parent with only the listed changes."""
    path, text = gen.derive(child)
    assert path.read_text() == text
    directory, parent, changes = gen.CHILDREN[child]
    got, expected = _loaded(path), dict(_loaded(directory / f"{parent}.yml"))
    expected["brain"] = {**expected["brain"]}
    expected["brain"]["config"] = {**expected["brain"]["config"], **changes}
    assert got == expected


@pytest.mark.parametrize(
    "stem",
    sorted(s for arms in gen.ARMS.values() for s in arms.values() if s.startswith("connectome")),
)
def test_each_connectome_arm_builds_on_emmons(stem: str) -> None:
    """No arm names a combination the brain refuses, and every one is on Emmons 2024."""
    container = load_simulation_config(str(gen.path_of(stem))).brain
    assert container is not None
    assert isinstance(container.config, ConnectomePPOBrainConfig)
    assert container.config.connectome_source == "emmons_2024_hermaphrodite"
    ConnectomePPOBrain(
        config=container.config.model_copy(update={"seed": 9871}),
        device=DeviceType.CPU,
    )


def test_the_frozen_arms_freeze_and_the_learners_learn() -> None:
    """Every frozen arm says so in its name and in its config."""
    for arms in gen.ARMS.values():
        for role, stem in arms.items():
            frozen = _loaded(gen.path_of(stem))["brain"]["config"]["freeze_updates"]
            assert frozen is role.endswith("_frozen"), stem
