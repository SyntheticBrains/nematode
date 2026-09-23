"""A.1's control re-read at depth 6: the eight new arms, the stem map and the seed band.

The re-read's claim is that it is A.1's design with one thing moved -- the settling depth -- and on
seeds nothing else has used. Both halves of that are checkable without a campaign: every new arm is
loaded through the real config loader and compared with its A.1 parent, and the seed band is checked
against every band spent before it.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path
from typing import Any

import pytest
from quantumnematode.utils.config_loader import load_simulation_config

_REPO = Path(__file__).resolve().parents[5]
_ANALYSIS = _REPO / "scripts" / "analysis"
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

import init_sharing_reread as rr  # noqa: E402  # pyright: ignore[reportMissingImports]
import operating_point_surface as ops  # noqa: E402  # pyright: ignore[reportMissingImports]

_CONFIGS = _REPO / "configs" / "scenarios" / "foraging"

# The eight arms this re-read generated; the four edge_order arms at depth 6 are A.2's own.
_NEW = sorted(stem for stem, (_, mode) in rr.ARM_BY_STEM.items() if mode != "edge_order")


@functools.cache
def _brain_config(stem: str) -> dict[str, Any]:
    brain = load_simulation_config(str(_CONFIGS / f"{stem}.yml")).brain
    if brain is None or brain.config is None:
        msg = f"{stem} carries no brain config"
        raise AssertionError(msg)
    return brain.config.model_dump()


def test_the_panel_is_a_1s_design_at_one_depth() -> None:
    """Twelve stems -- four arms by three draw modes -- and every one a committed config."""
    assert len(rr.ARM_BY_STEM) == 12
    missing = [s for s in rr.ARM_BY_STEM if not (_CONFIGS / f"{s}.yml").is_file()]
    assert not missing, missing


@pytest.mark.parametrize("stem", _NEW, ids=lambda s: s[-36:])
def test_each_new_arm_differs_from_its_a1_parent_in_depth_alone(stem: str) -> None:
    """Loaded through the real loader, each new arm moves only `forward_pass_depth`, to 6."""
    parent = stem.removesuffix(f"_d{rr.DEPTH}")
    child, base = _brain_config(stem), _brain_config(parent)
    differing = {k for k in set(child) | set(base) if child.get(k) != base.get(k)}
    assert differing == {"forward_pass_depth"}, f"{stem} differs in {sorted(differing)}"
    assert child["forward_pass_depth"] == rr.DEPTH


def test_the_seeds_are_fresh() -> None:
    """The 32 re-read seeds overlap no band burnt before them, A.2's pilot and panels included."""
    spent = set(ops.BURNT_SEEDS) | set(ops.PILOT_SEEDS)
    for seeds in ops.SEEDS_BY_HALF.values():
        spent |= set(seeds)
    assert not set(rr.SEEDS) & spent
    assert len(rr.SEEDS) == 32
