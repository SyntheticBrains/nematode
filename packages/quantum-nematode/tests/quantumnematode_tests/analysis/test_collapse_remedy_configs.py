"""M.1(b)'s configs: each is one of 092's body configs with the listed keys alone changed.

Covers the learning-rules requirement "PPO stabilisers and the placebo shuffle" through the configs
that run it: every arm loads, differs from its parent only where it says, and carries the placebo.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path
from typing import Any

import pytest
from quantumnematode.utils.config_loader import load_simulation_config

_root = Path(__file__).resolve()
while _root != _root.parent and not (_root / "scripts" / "campaigns").is_dir():
    _root = _root.parent
sys.path.insert(0, str(_root / "scripts" / "campaigns"))

import generate_collapse_remedy_configs as gen  # noqa: E402  # pyright: ignore[reportMissingImports]


@functools.cache
def _loaded(stem: str) -> dict[str, Any]:
    return load_simulation_config(str(gen.THERMAL / f"{stem}.yml")).model_dump()


def test_four_rounds_on_three_learners() -> None:
    """The placebo and three remedies, each on the wild type, the gap-only null and the MLP."""
    assert len(gen.CHILDREN) == 12
    assert list(gen.ROUNDS) == ["placebo", "klstop", "wclip", "adam"]


@pytest.mark.parametrize("child", sorted(gen.CHILDREN))
def test_each_config_is_its_parent_with_the_listed_keys_alone(child: str) -> None:
    """Each generated file is the generator's output and its parent with only the listed changes."""
    path, text = gen.derive(child)
    assert path.read_text() == text
    parent, changes = gen.CHILDREN[child]
    got, expected = _loaded(child), dict(_loaded(parent))
    expected["brain"] = {**expected["brain"]}
    expected["brain"]["config"] = {**expected["brain"]["config"], **changes}
    assert got == expected
    assert got["brain"]["config"]["rollout_shuffle_stream"] is True
