"""The measured-prior pilot: its configs, seeds, levels reaching the brain, and selection.

Covers the architecture-comparison-protocol requirements "A pin is chosen on the learner's own gate,
never on the contrast it will carry" (the selection is a function of the gates alone, including the
cases where a level's null fails its floor or both arms saturate) and "A swept level is shown to
reach the learner it is set on" (every level changes the constructed chemical weights, and each
level's frozen floor is built from the same weights as its learning arm).

The two-key delta is verified through the real configuration loader, not by diffing text: a
generated file is its parent's keys re-emitted, and what matters is what a run actually gets.
"""

# pyright: reportPrivateUsage=false
from __future__ import annotations

import functools
import inspect
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from quantumnematode.brain.arch.connectome_ppo import ConnectomePPOBrain, ConnectomePPOBrainConfig
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.utils.config_loader import load_simulation_config

_REPO = Path(__file__).resolve().parents[5]
_ANALYSIS = _REPO / "scripts" / "analysis"
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

import init_sharing_reread as rr  # noqa: E402  # pyright: ignore[reportMissingImports]
import measured_prior_pilot as mp  # noqa: E402  # pyright: ignore[reportMissingImports]
import operating_point_surface as ops  # noqa: E402  # pyright: ignore[reportMissingImports]

_CONFIGS = _REPO / "configs" / "scenarios" / "foraging"
_SEED = 5

_NEW = [
    (half, arm, suffix)
    for half in mp.HALVES
    for suffix in mp.ALL_LEVELS
    if suffix != mp.RANDOM
    for arm in mp.ARMS
]


@functools.cache
def _loaded(stem: str) -> dict[str, Any]:
    """Load the whole simulation config as a run gets it, through the real loader."""
    return load_simulation_config(str(_CONFIGS / f"{stem}.yml")).model_dump()


@functools.cache
def _w_chem(stem: str) -> torch.Tensor:
    """Build one arm's brain at one seed and return its chemical weights."""
    container = load_simulation_config(str(_CONFIGS / f"{stem}.yml")).brain
    assert container is not None
    assert isinstance(container.config, ConnectomePPOBrainConfig)
    cfg = container.config.model_copy(update={"seed": _SEED})
    return ConnectomePPOBrain(config=cfg, device=DeviceType.CPU).topology.w_chem.detach().clone()


class TestTheConfigs:
    def test_every_arm_at_every_level_is_a_committed_config(self) -> None:
        """Seven levels by four arms by two learners, every one on disk."""
        assert len(mp.ARM_BY_STEM) == 7 * 4 * 2
        missing = [s for s in mp.ARM_BY_STEM if not (_CONFIGS / f"{s}.yml").is_file()]
        assert not missing, missing

    @pytest.mark.parametrize(("half", "arm", "suffix"), _NEW, ids=str)
    def test_each_arm_differs_from_its_parent_in_its_levels_keys_alone(
        self,
        half: str,
        arm: str,
        suffix: str,
    ) -> None:
        """Through the real loader only the level's keys move, and nothing outside the brain."""
        child = _loaded(mp.stem_for(half, arm, suffix))
        parent = _loaded(mp.PARENTS[half][arm])
        c_brain, p_brain = child["brain"]["config"], parent["brain"]["config"]
        differing = {k for k in set(c_brain) | set(p_brain) if c_brain.get(k) != p_brain.get(k)}
        keys = mp.level_keys(suffix)
        assert differing == set(keys) - {k for k, v in keys.items() if p_brain.get(k) == v}
        for key, value in keys.items():
            assert c_brain[key] == value
        assert {k: v for k, v in child.items() if k != "brain"} == {
            k: v for k, v in parent.items() if k != "brain"
        }

    def test_the_ppo_half_runs_under_the_fan_in_draw_and_the_reading_half_does_not(self) -> None:
        """PPO takes the shared initialisation; the reading learner stays where A.2 swept it."""
        for arm in mp.ARMS:
            assert _loaded(mp.PARENTS["ppo"][arm])["brain"]["config"]["weight_draw"] == (
                "per_neuron_fanin"
            )
            assert _loaded(mp.PARENTS["reading"][arm])["brain"]["config"]["weight_draw"] == (
                "edge_order"
            )

    def test_the_default_multiplier_is_never_written(self) -> None:
        """At 1.0 the multiplier is a no-op key, and under the sign-only prior it is refused."""
        for suffix in mp.ALL_LEVELS:
            assert mp.level_keys(suffix).get("measured_weight_scale") != 1.0


class TestTheSeeds:
    def test_the_bands_are_fresh(self) -> None:
        """No pilot seed has been spent by any earlier panel, pilot or re-read."""
        spent = set(ops.BURNT_SEEDS) | set(ops.PILOT_SEEDS) | set(rr.SEEDS)
        for seeds in ops.SEEDS_BY_HALF.values():
            spent |= set(seeds)
        for seeds in mp.SEEDS_BY_HALF.values():
            assert not set(seeds) & spent

    def test_the_bands_are_disjoint_and_below_b1c(self) -> None:
        """The two learners do not share a seed, and neither reaches B.1c's panel."""
        ppo, reading = (set(mp.SEEDS_BY_HALF[h]) for h in ("ppo", "reading"))
        assert not ppo & reading
        assert len(ppo) == len(reading) == 8
        assert max(ppo | reading) < mp.B1C_FIRST_SEED


class TestEveryLevelReachesTheBrain:
    @pytest.mark.parametrize("half", mp.HALVES)
    def test_each_level_builds_different_chemical_weights(self, half: str) -> None:
        """Every level's wild-type learning arm is constructed on weights no other level has."""
        built = {s: _w_chem(mp.stem_for(half, "wt_learn", s)) for s in mp.ALL_LEVELS}
        for i, a in enumerate(mp.ALL_LEVELS):
            for b in mp.ALL_LEVELS[i + 1 :]:
                assert not torch.equal(built[a], built[b]), f"{half}: {a} and {b} build alike"

    @pytest.mark.parametrize("half", mp.HALVES)
    @pytest.mark.parametrize("wiring", ["wt", "rn"])
    def test_each_levels_floor_is_built_from_its_learning_arms_weights(
        self,
        half: str,
        wiring: str,
    ) -> None:
        """A floor at the wrong level would gate a level against another substrate."""
        for suffix in mp.ALL_LEVELS:
            learn = _w_chem(mp.stem_for(half, f"{wiring}_learn", suffix))
            frozen = _w_chem(mp.stem_for(half, f"{wiring}_frozen", suffix))
            assert torch.equal(learn, frozen), f"{half}/{wiring}/{suffix}"


def _gates(
    *,
    wt: float = 0.1,
    rn: float = 0.1,
    saturated: bool = False,
) -> dict[str, Any]:
    """Build a gate record shaped like ``learning_gates``' output."""
    return {
        "wt": {"vs_floor": {"ci_lo": wt}},
        "rn": {"vs_floor": {"ci_lo": rn}},
        "gate_passes": wt > 0.0 and rn > 0.0,
        "saturated": saturated,
    }


def _all(**overrides: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Pass every level, with some levels replaced."""
    return {s: overrides.get(s, _gates()) for s in mp.ALL_LEVELS}


_FAIL = _gates(wt=-0.1, rn=-0.1)


class TestTheSelection:
    def test_the_default_is_chosen_when_it_passes(self) -> None:
        """1.0 wins whenever it passes, whatever else does."""
        out = mp.select(_all())
        assert (out["branch"], out["chosen_level"], out["chosen_multiplier"]) == (
            "selected",
            "m1",
            1.0,
        )

    def test_the_nearest_passing_level_on_the_log_scale_otherwise(self) -> None:
        """Without 1.0, 0.5 and 2.0 are equidistant on the log scale and the smaller wins."""
        out = mp.select(_all(m1=_FAIL))
        assert out["chosen_level"] == "m05"
        out = mp.select(_all(m1=_FAIL, m05=_FAIL))
        assert out["chosen_level"] == "m2"
        out = mp.select(_all(m1=_FAIL, m05=_FAIL, m2=_FAIL))
        assert out["chosen_level"] == "m025"

    def test_a_level_whose_null_fails_its_floor_is_not_chosen(self) -> None:
        """The wild type learning is not enough: the contrast needs both arms."""
        out = mp.select(_all(m1=_gates(rn=-0.1)))
        assert out["chosen_level"] == "m05"
        assert out["arms_below_floor"] == {"m1": ["rn"]}

    def test_a_saturated_level_is_not_chosen(self) -> None:
        """Two arms tied at the ceiling cannot carry the contrast."""
        out = mp.select(_all(m1=_gates(saturated=True)))
        assert out["chosen_level"] == "m05"

    def test_pathway_unlearnable(self) -> None:
        """Random learns; no measured level and not the sign-only prior do."""
        failing = dict.fromkeys((*mp.MULTIPLIER_LEVELS, mp.SIGN_LEVEL), _FAIL)
        out = mp.select(_all(**failing))
        assert out["branch"] == "pathway_unlearnable"
        assert out["chosen_level"] is None

    def test_the_lee_branch_reads_the_wild_type_alone(self) -> None:
        """A wild type that learns with a broken null is not the pathway failing."""
        broken_null = _gates(rn=-0.1)
        levels = dict.fromkeys((*mp.MULTIPLIER_LEVELS, mp.SIGN_LEVEL), broken_null)
        out = mp.select(_all(**levels))
        assert out["branch"] == "no_level_passes"

    def test_magnitude_is_the_obstacle(self) -> None:
        """The sign-only prior passes and no multiplier does."""
        out = mp.select(_all(**dict.fromkeys(mp.MULTIPLIER_LEVELS, _FAIL)))
        assert out["branch"] == "magnitude_obstacle"
        assert out["chosen_level"] is None

    def test_a_random_level_that_fails_makes_the_pilot_uninformative(self) -> None:
        """No measured failure can be read against a substrate that does not learn at all."""
        out = mp.select(_all(random=_FAIL))
        assert out["branch"] == "uninformative"
        assert out["chosen_level"] is None

    def test_the_selection_cannot_see_the_gap(self) -> None:
        """It takes the gates and nothing else, so the wiring gap has no way in."""
        assert list(inspect.signature(mp.select).parameters) == ["gates_by_level"]


def _levels_with_gaps(gaps: dict[str, tuple[float, float, float]]) -> dict[str, Any]:
    """Build levels carrying a primary-metric gap of ``(mean, ci_lo, ci_hi)`` each."""
    metric = ops.UNCENSORED_METRIC
    return {
        s: {
            "metric_choice": {"primary_metric": metric},
            metric: {"wiring_gap": {"gap_mean": m, "test": {"ci_lo": lo, "ci_hi": hi}}},
        }
        for s, (m, lo, hi) in gaps.items()
    }


class TestSignMovement:
    def test_a_level_past_zero_on_the_other_side_is_recorded(self) -> None:
        """The random level reads positive; a level whose interval sits below zero has moved."""
        gaps = dict.fromkeys(mp.ALL_LEVELS, (0.1, 0.05, 0.15))
        gaps["m4"] = (-0.2, -0.3, -0.1)
        gaps["m2"] = (-0.05, -0.2, 0.1)
        out = mp.sign_movement(_levels_with_gaps(gaps))
        assert out["random_side"] == 1
        assert out["random_side_basis"] == "interval"
        assert out["levels_moved"] == ["m4"]

    def test_an_unresolved_random_level_takes_the_side_of_its_mean(self) -> None:
        """Said so in the record, since the side then rests on a point estimate."""
        gaps = dict.fromkeys(mp.ALL_LEVELS, (-0.02, -0.1, 0.05))
        gaps["sign"] = (0.2, 0.1, 0.3)
        out = mp.sign_movement(_levels_with_gaps(gaps))
        assert (out["random_side"], out["random_side_basis"]) == (-1, "mean")
        assert out["levels_moved"] == ["sign"]


class TestTheManifest:
    def _campaign(self, tmp_path: Path, half: str, seeds: tuple[int, ...]) -> Path:
        logs = tmp_path / "logs"
        logs.mkdir()
        for suffix in mp.ALL_LEVELS:
            for arm in mp.ARMS:
                for seed in seeds:
                    (logs / f"{mp.stem_for(half, arm, suffix)}-seed{seed}.log").write_text("")
        return tmp_path

    def test_a_complete_panel_is_accepted_in_a_2s_line_format(self, tmp_path: Path) -> None:
        """Every line reads ``<arm> <level> <seed> <log>``, the format the shared gate reads."""
        seeds = mp.SEEDS_BY_HALF["ppo"][:2]
        campaign = self._campaign(tmp_path, "ppo", seeds)
        manifest = mp.build_manifest(campaign, tmp_path / "m.txt", "ppo", seeds)
        mp.require_complete(manifest, "ppo", seeds)
        lines = manifest.read_text().split("\n")[:-1]
        assert len(lines) == len(mp.ALL_LEVELS) * 4 * 2
        arm, suffix, seed, _ = lines[0].split()
        assert arm in mp.ARMS
        assert suffix in mp.ALL_LEVELS
        assert int(seed) in seeds

    def test_a_missing_run_is_refused(self, tmp_path: Path) -> None:
        """The instruments only warn on a gap; the pilot refuses one."""
        seeds = mp.SEEDS_BY_HALF["reading"][:2]
        campaign = self._campaign(tmp_path, "reading", seeds)
        victim = mp.stem_for("reading", "rn_frozen", "m4")
        (campaign / "logs" / f"{victim}-seed{seeds[0]}.log").unlink()
        manifest = mp.build_manifest(campaign, tmp_path / "m.txt", "reading", seeds)
        with pytest.raises(mp.PilotError, match="m4/rn_frozen"):
            mp.require_complete(manifest, "reading", seeds)

    def test_a_log_from_another_panel_is_refused(self, tmp_path: Path) -> None:
        """A stray config in the campaign is an error, not a silently dropped row."""
        logs = tmp_path / "logs"
        logs.mkdir()
        (logs / "connectomeppo_something_else-seed113.log").write_text("")
        with pytest.raises(mp.PilotError, match="does not have"):
            mp.build_manifest(tmp_path, tmp_path / "m.txt", "ppo", (113,))


def test_the_gate_takes_each_levels_own_floor(tmp_path: Path) -> None:
    """The shared gate reads the floor the pilot names, not A.2's centre rule."""
    manifest = tmp_path / "m.txt"
    manifest.write_text("")
    out = ops.learning_gates(manifest, "ppo", (113,), "m4", floor_level="m4")
    assert out["floor_level"] == "m4"
