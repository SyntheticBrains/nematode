"""The body contrast's configs on log counts: each is 092's body config with two keys alone changed.

Covers `add-body-logcount-contrast` design Decision A: every arm loads, differs from its committed
parent only in the connectome source and the gap weight mode, freezes where its name says, and builds
on Emmons 2024 with log-compressed counts.
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

import generate_body_logcount_configs as gen  # noqa: E402  # pyright: ignore[reportMissingImports]


@functools.cache
def _loaded(path: Path) -> dict[str, Any]:
    return load_simulation_config(str(path)).model_dump()


@pytest.mark.parametrize("role", sorted(gen.ARMS))
def test_each_config_is_its_parent_with_the_two_keys_alone(role: str) -> None:
    child, parent = gen.ARMS[role]
    path, text = gen.derive(child, parent)
    assert path.read_text() == text
    got, expected = _loaded(path), dict(_loaded(gen.THERMAL / f"{parent}.yml"))
    expected["brain"] = {**expected["brain"]}
    expected["brain"]["config"] = {**expected["brain"]["config"], **gen.DELTA}
    assert got == expected


@pytest.mark.parametrize("role", sorted(gen.ARMS))
def test_each_arm_freezes_where_its_name_says_and_builds(role: str) -> None:
    child, _ = gen.ARMS[role]
    container = load_simulation_config(str(gen.THERMAL / f"{child}.yml")).brain
    assert container is not None
    assert isinstance(container.config, ConnectomePPOBrainConfig)
    assert container.config.connectome_source == "emmons_2024_hermaphrodite"
    assert container.config.gap_weight_mode == "log_count"
    assert container.config.freeze_updates is role.endswith("_frozen")
    ConnectomePPOBrain(
        config=container.config.model_copy(update={"seed": 9895}),
        device=DeviceType.CPU,
    )
