"""A.6's panel: its configs, seeds, manifest, gates, verdicts and family.

Covers the architecture-comparison-protocol requirement "A null states every structural property it
does not preserve": the chemical-only null's configs change the wiring and nothing else, its frozen
floors hold the wild type's gap junctions, and the verdict map reads a move as the held properties'
jointly, attributing a gap only where one exists.
"""

# pyright: reportPrivateUsage=false
from __future__ import annotations

import functools
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
import measured_prior_contrast as mc  # noqa: E402  # pyright: ignore[reportMissingImports]
import measured_prior_pilot as mp  # noqa: E402  # pyright: ignore[reportMissingImports]
import null_strength_control as nsc  # noqa: E402  # pyright: ignore[reportMissingImports]
import operating_point_surface as ops  # noqa: E402  # pyright: ignore[reportMissingImports]

_CONFIGS = _REPO / "configs" / "scenarios" / "foraging"
_SEED = 3


@functools.cache
def _loaded(stem: str) -> dict[str, Any]:
    """Load the whole simulation config as a run gets it, through the real loader."""
    return load_simulation_config(str(_CONFIGS / f"{stem}.yml")).model_dump()


@functools.cache
def _topology(stem: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one arm's brain at one seed and return its chemical mask and gap buffer."""
    container = load_simulation_config(str(_CONFIGS / f"{stem}.yml")).brain
    assert container is not None
    assert isinstance(container.config, ConnectomePPOBrainConfig)
    cfg = container.config.model_copy(update={"seed": _SEED})
    topology = ConnectomePPOBrain(config=cfg, device=DeviceType.CPU).topology
    return topology.m_chem.clone(), topology.g_gap.clone()


class TestTheConfigs:
    def test_every_arm_is_a_committed_config(self) -> None:
        """Two learners, two levels, the wild-type arms shared: every stem on disk."""
        missing = [s for s in nsc.LEVELS_BY_STEM if not (_CONFIGS / f"{s}.yml").is_file()]
        assert not missing, missing
        assert len(nsc.LEVELS_BY_STEM) == 2 * 6
        assert len(nsc.NEW_ARMS) == 4

    def test_the_reused_arms_are_block_v_and_a2s_centre(self) -> None:
        """The wild-type and full-null arms are the committed configs, not new copies."""
        assert nsc.STEMS["reading"][nsc.FULL] == ops.CENTRE_STEMS["reading"]
        assert nsc.STEMS["ppo"][nsc.FULL] == ops.CENTRE_STEMS["ppo"]

    @pytest.mark.parametrize("stem", sorted(nsc.NEW_ARMS), ids=lambda s: s[-45:])
    def test_each_new_arm_differs_from_its_parent_in_the_wiring_alone(self, stem: str) -> None:
        """Through the real loader only ``wiring`` moves, and nothing outside the brain."""
        _, parent = nsc.NEW_ARMS[stem]
        child, base = _loaded(stem), _loaded(parent)
        c_brain, p_brain = child["brain"]["config"], base["brain"]["config"]
        differing = {k for k in set(c_brain) | set(p_brain) if c_brain.get(k) != p_brain.get(k)}
        assert differing == {"wiring"}
        assert c_brain["wiring"] == "rewired_chemical_only"
        assert p_brain["wiring"] == "rewired_degree_preserving"
        assert {k: v for k, v in child.items() if k != "brain"} == {
            k: v for k, v in base.items() if k != "brain"
        }

    @pytest.mark.parametrize("half", nsc.HALVES)
    def test_the_chemical_null_holds_the_wild_types_gap_junctions(self, half: str) -> None:
        """Learning and frozen arms alike: the wild type's g_gap, and a different chemical mask."""
        wild_mask, wild_gap = _topology(nsc.STEMS[half][nsc.FULL]["wt_learn"])
        for arm in ("rn_learn", "rn_frozen"):
            mask, gap = _topology(nsc.STEMS[half][nsc.CHEMICAL][arm])
            assert torch.equal(gap, wild_gap)
            assert not torch.equal(mask, wild_mask)
        learn = _topology(nsc.STEMS[half][nsc.CHEMICAL]["rn_learn"])
        frozen = _topology(nsc.STEMS[half][nsc.CHEMICAL]["rn_frozen"])
        assert torch.equal(learn[0], frozen[0])


class TestTheSeeds:
    def test_the_bands_are_fresh_and_disjoint(self) -> None:
        """No seed spent by an earlier panel, and none shared between learners."""
        spent = set(ops.BURNT_SEEDS) | set(ops.PILOT_SEEDS) | set(rr.SEEDS)
        for seeds in (
            *ops.SEEDS_BY_HALF.values(),
            *mp.SEEDS_BY_HALF.values(),
            *mc.SEEDS_BY_HALF.values(),
        ):
            spent |= set(seeds)
        ppo, reading = (set(nsc.SEEDS_BY_HALF[h]) for h in ("ppo", "reading"))
        assert not (ppo | reading) & spent
        assert not ppo & reading
        assert (len(ppo), len(reading)) == (32, 48)


class TestTheManifest:
    def _campaign(self, tmp_path: Path, half: str, seeds: tuple[int, ...]) -> Path:
        logs = tmp_path / "logs"
        logs.mkdir()
        for stem, (log_half, _, _) in nsc.LEVELS_BY_STEM.items():
            if log_half != half:
                continue
            for seed in seeds:
                (logs / f"{stem}-seed{seed}.log").write_text("")
        return tmp_path

    def test_a_wild_type_run_serves_both_levels(self, tmp_path: Path) -> None:
        """Six runs per seed become eight manifest lines: the wild type once under each level."""
        seeds = nsc.SEEDS_BY_HALF["ppo"][:2]
        campaign = self._campaign(tmp_path, "ppo", seeds)
        manifest = nsc.build_manifest(campaign, tmp_path / "m.txt", "ppo", seeds)
        mp.require_complete(manifest, "ppo", seeds, nsc.LEVELS)
        lines = manifest.read_text().split("\n")[:-1]
        assert len(lines) == 2 * 4 * len(seeds)
        wt = [ln for ln in lines if ln.startswith("wt_learn ") and f" {seeds[0]} " in ln]
        assert sorted(ln.split()[1] for ln in wt) == sorted(nsc.LEVELS)

    def test_a_missing_chemical_null_run_is_refused(self, tmp_path: Path) -> None:
        """The instruments only warn on a gap; the panel refuses one."""
        seeds = nsc.SEEDS_BY_HALF["reading"][:2]
        campaign = self._campaign(tmp_path, "reading", seeds)
        victim = nsc.STEMS["reading"][nsc.CHEMICAL]["rn_frozen"]
        (campaign / "logs" / f"{victim}-seed{seeds[0]}.log").unlink()
        manifest = nsc.build_manifest(campaign, tmp_path / "m.txt", "reading", seeds)
        with pytest.raises(mp.PilotError, match="chemical/rn_frozen"):
            mp.require_complete(manifest, "reading", seeds, nsc.LEVELS)


def _gates(*, wt: float = 0.1, rn: float = 0.1, saturated: bool = False) -> dict[str, Any]:
    return {
        "wt": {"vs_floor": {"ci_lo": wt}},
        "rn": {"vs_floor": {"ci_lo": rn}},
        "gate_passes": wt > 0.0 and rn > 0.0,
        "saturated": saturated,
    }


def _interaction(mean: float, lo: float, hi: float, q: float) -> dict[str, Any]:
    return {"interaction_mean": mean, "test": {"ci_lo": lo, "ci_hi": hi, "bh_q": q}}


_READABLE = {level: _gates() for level in nsc.LEVELS}
_NO_MOVE = _interaction(0.001, -0.01, 0.012, 0.6)
_WT_AHEAD = {"ci_lo": 0.03, "ci_hi": 0.09}
_NULL_AHEAD = {"ci_lo": -0.3, "ci_hi": -0.1}
_SPANS_ZERO = {"ci_lo": -0.02, "ci_hi": 0.05}


class TestTheVerdicts:
    def test_no_move_with_a_ppo_gap_is_chemical(self) -> None:
        """The wild type still leads the chemical null, and the gap does not move."""
        out = nsc.read_learner("ppo", _READABLE, _NO_MOVE, _WT_AHEAD)
        assert (out["state"], out["verdict"]) == ("no_move", "chemical")

    def test_no_move_without_a_gap_attributes_nothing(self) -> None:
        """A gap that is not there cannot be said to live in the chemical wiring."""
        out = nsc.read_learner("ppo", _READABLE, _NO_MOVE, _SPANS_ZERO)
        assert out["verdict"] == "no_gap_to_attribute"

    def test_the_reading_learners_gap_is_on_the_nulls_side(self) -> None:
        """Its reference effect is negative, so the gate looks below zero, not above."""
        assert nsc.read_learner("reading", _READABLE, _NO_MOVE, _NULL_AHEAD)["verdict"] == (
            "chemical"
        )
        assert nsc.read_learner("reading", _READABLE, _NO_MOVE, _WT_AHEAD)["verdict"] == (
            "no_gap_to_attribute"
        )

    @pytest.mark.parametrize(
        ("interaction", "verdict"),
        [
            (_interaction(-0.08, -0.12, -0.05, 0.001), "gap_or_autapse"),
            (_interaction(0.08, 0.05, 0.12, 0.001), "amplified"),
            (_interaction(0.01, 0.005, 0.02, 0.01), "below_minimum"),
            (_interaction(0.02, -0.02, 0.06, 0.4), "unresolved"),
        ],
        ids=["toward-null", "toward-wild-type", "below", "unresolved"],
    )
    def test_the_other_states(self, interaction: dict[str, Any], verdict: str) -> None:
        """A move is attributed to the held properties jointly, whichever way it goes."""
        assert nsc.read_learner("ppo", _READABLE, interaction, _WT_AHEAD)["verdict"] == verdict

    @pytest.mark.parametrize(
        "bad",
        [_gates(rn=-0.1), _gates(saturated=True)],
        ids=["null-below-floor", "saturated"],
    )
    def test_an_unreadable_level_gives_no_verdict(self, bad: dict[str, Any]) -> None:
        """A broken null or two arms at the ceiling give no verdict, whatever the gap says."""
        gates = {nsc.FULL: _gates(), nsc.CHEMICAL: bad}
        assert nsc.read_learner("ppo", gates, _NO_MOVE, _WT_AHEAD)["verdict"] == "unreadable"

    def test_a_drifted_reading_half_is_void(self) -> None:
        """B.1c's drift rule, reused: a reading learner that wrote its matrix has no verdict."""
        out = nsc.read_learner("reading", _READABLE, _NO_MOVE, _NULL_AHEAD)
        drift = {"void": True, "obligation_applies": True}
        assert mc.honour_drift("reading", out, drift)["verdict"] == "void"

    def test_the_minimum_is_two_thirds_of_each_committed_effect(self) -> None:
        """A.1's edge-order effect for PPO, A.2's centre for the reading learner."""
        assert nsc.minimum("ppo") == pytest.approx(2 / 3 * 0.06098958333333333)
        assert nsc.minimum("reading") == pytest.approx(2 / 3 * 0.2105)


def test_the_family_is_exactly_the_two_primary_interactions() -> None:
    """One interaction per learner on the primary metric, corrected together."""
    results = {
        half: {"interactions": {nsc.PRIMARY_METRIC: {"test": {"wilcoxon_p": p}}}}
        for half, p in zip(nsc.HALVES, (0.01, 0.2), strict=True)
    }
    nsc.correct_family(results, nsc.PRIMARY_METRIC)
    qs = sorted(results[h]["interactions"][nsc.PRIMARY_METRIC]["test"]["bh_q"] for h in nsc.HALVES)
    # Folded two-sided: 0.02 and 0.4; BH over two: 0.04 and 0.4.
    assert qs == pytest.approx([0.04, 0.4])
