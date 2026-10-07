"""C.0's validation: its configs and its floor gate.

Covers the continuous-2d-environment requirement "Signed speed" and the connectome-ppo-brain
requirement "Emmons 2024 as a connectome source" through the configs that run them.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path
from typing import Any

import pytest
from quantumnematode.utils.config_loader import load_simulation_config

_root = Path(__file__).resolve()
while _root != _root.parent and not (_root / "scripts" / "analysis").is_dir():
    _root = _root.parent
sys.path.insert(0, str(_root / "scripts" / "analysis"))
sys.path.insert(0, str(_root / "scripts" / "campaigns"))

import body_prerequisites as bp  # noqa: E402  # pyright: ignore[reportMissingImports]
import generate_body_prerequisite_configs as gen  # noqa: E402  # pyright: ignore[reportMissingImports]

_FORAGING = _root / "configs" / "scenarios" / "foraging"


@functools.cache
def _loaded(stem: str) -> dict[str, Any]:
    return load_simulation_config(str(_FORAGING / f"{stem}.yml")).model_dump()


class TestConfigs:
    def test_every_learner_config_is_generated(self) -> None:
        stems = {s for pair in bp.LEARNERS.values() for s in pair}
        assert stems == set(gen.CHILDREN)
        for stem in stems:
            assert (_FORAGING / f"{stem}.yml").is_file()

    @pytest.mark.parametrize("child", sorted(gen.CHILDREN))
    def test_each_differs_from_its_parent_in_the_listed_keys_alone(self, child: str) -> None:
        parent, changes = gen.CHILDREN[child]
        got, expected = _loaded(child), _loaded(parent)
        expected = {**expected, "environment": {**expected["environment"]}}
        expected["environment"]["continuous"] = {
            **expected["environment"]["continuous"],
            "allow_reversal": True,
        }
        expected["brain"] = {**expected["brain"]}
        expected["brain"]["config"] = {
            **expected["brain"]["config"],
            "signed_speed": True,
            **changes,
        }
        assert got == expected

    def test_the_connectome_arms_are_on_emmons(self) -> None:
        for stem in bp.LEARNERS["connectome"]:
            assert _loaded(stem)["brain"]["config"]["connectome_source"] == (
                "emmons_2024_hermaphrodite"
            )

    def test_the_seeds_are_unused_elsewhere(self) -> None:
        import across_step_control as asc  # pyright: ignore[reportMissingImports]

        used = (
            set(asc.FIRST_PILOT_SEEDS)
            | set(asc.PILOT_SEEDS)
            | set(asc.MLP_SEEDS)
            | {s for spec in asc.CELLS.values() for s in spec["seeds"]}
        )
        assert not set(bp.SEEDS) & used


class TestGate:
    def test_a_learner_above_its_floor_passes(self) -> None:
        learn = {s: 60.0 + s % 3 for s in bp.SEEDS}
        frozen = dict.fromkeys(bp.SEEDS, 0.0)
        assert bp.gate(learn, frozen)["verdict"] == "passes"

    def test_a_learner_at_its_floor_fails(self) -> None:
        learn = {s: float(s % 2) for s in bp.SEEDS}
        frozen = {s: float((s + 1) % 2) for s in bp.SEEDS}
        assert bp.gate(learn, frozen)["verdict"] == "fails"

    def test_missing_seeds_read_incomplete(self) -> None:
        learn = dict.fromkeys(bp.SEEDS[:5], 60.0)
        frozen = dict.fromkeys(bp.SEEDS, 0.0)
        assert bp.gate(learn, frozen)["verdict"] == "incomplete"

    def test_no_runs_do_not_validate(self, tmp_path: Path) -> None:
        assert bp.score([tmp_path])["validated"] is False
