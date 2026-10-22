"""The strength-normalised gap model's configs: each is its parent with the listed keys alone.

Covers `add-gap-model-question` task 3 and the connectome-ppo-brain requirement "Strength-normalised
gap junctions" through the configs that run it: every arm loads, differs from its parent only where
it says, freezes where its name says, and builds on Emmons 2024 under strength normalisation.
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

import generate_gap_model_configs as gen  # noqa: E402  # pyright: ignore[reportMissingImports]


@functools.cache
def _loaded(path: Path) -> dict[str, Any]:
    return load_simulation_config(str(path)).model_dump()


@pytest.mark.parametrize("role", sorted(gen.ARMS))
def test_each_config_is_its_parent_with_the_listed_keys_alone(role: str) -> None:
    """Each generated file is the generator's output and its parent with only the listed changes."""
    child, parent, changes = gen.ARMS[role]
    path, text = gen.derive(child, parent, changes)
    assert path.read_text() == text
    got, expected = _loaded(path), dict(_loaded(gen.THERMAL / f"{parent}.yml"))
    expected["brain"] = {**expected["brain"]}
    expected["brain"]["config"] = {**expected["brain"]["config"], **changes}
    assert got == expected


@pytest.mark.parametrize("role", sorted(gen.ARMS))
def test_each_arm_builds_under_strength_and_freezes_where_named(role: str) -> None:
    """Every arm is on Emmons with count weights under strength normalisation, and builds."""
    child, _, _ = gen.ARMS[role]
    container = load_simulation_config(str(gen.THERMAL / f"{child}.yml")).brain
    assert container is not None
    assert isinstance(container.config, ConnectomePPOBrainConfig)
    assert container.config.connectome_source == "emmons_2024_hermaphrodite"
    assert container.config.gap_normalisation == "strength"
    assert container.config.gap_weight_mode == "count"
    assert container.config.freeze_updates is role.endswith("_frozen")
    ConnectomePPOBrain(
        config=container.config.model_copy(update={"seed": 9905}),
        device=DeviceType.CPU,
    )
