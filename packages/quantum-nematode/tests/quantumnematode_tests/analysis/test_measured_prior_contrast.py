"""The wiring x weight-prior 2x3: its configs, seeds, shuffled arms, states and verdicts.

Covers the architecture-comparison-protocol requirement "A measured-weight positive is read against
its placement-shuffled control": the verdict map reports legibility, or hiding, only when both
the measured-versus-random and the measured-versus-shuffled interactions move the gap the same way,
and a distribution effect otherwise, in either direction.

The configs are checked through the real loader, the shuffled arms by constructing their brains, and
the classification and verdict map on synthetic statistics, since those two tables are what the
registration fixed and a campaign cannot be run to test them.
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
from quantumnematode.connectome.loader import load_cook_2019_hermaphrodite
from quantumnematode.connectome.measured_weights import coverage, measured_weights
from quantumnematode.utils.config_loader import load_simulation_config

_REPO = Path(__file__).resolve().parents[5]
_ANALYSIS = _REPO / "scripts" / "analysis"
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

import init_sharing_reread as rr  # noqa: E402  # pyright: ignore[reportMissingImports]
import measured_prior_contrast as mc  # noqa: E402  # pyright: ignore[reportMissingImports]
import measured_prior_pilot as mp  # noqa: E402  # pyright: ignore[reportMissingImports]
import operating_point_surface as ops  # noqa: E402  # pyright: ignore[reportMissingImports]

_CONFIGS = _REPO / "configs" / "scenarios" / "foraging"
_SEED = 7


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
    def test_every_arm_is_a_committed_config(self) -> None:
        """Three levels by four arms by two learners, every one on disk."""
        assert len(mc.ARM_BY_STEM) == 3 * 4 * 2
        missing = [s for s in mc.ARM_BY_STEM if not (_CONFIGS / f"{s}.yml").is_file()]
        assert not missing, missing

    def test_the_random_and_measured_levels_are_the_pilots_configs(self) -> None:
        """Only the shuffled level is new; the other two are B.1b's arms, reused unchanged."""
        new = mc.ARM_BY_STEM.keys() - mp.ARM_BY_STEM.keys()
        assert new == {mp.stem_for(h, a, mc.SHUFFLED) for h in mc.HALVES for a in mc.ARMS}
        assert len(new) == 8

    @pytest.mark.parametrize("half", mc.HALVES)
    @pytest.mark.parametrize("arm", mc.ARMS)
    def test_each_shuffled_arm_differs_from_its_parent_in_the_prior_alone(
        self,
        half: str,
        arm: str,
    ) -> None:
        """Through the real loader only ``weight_prior`` moves, and nothing outside the brain."""
        child = _loaded(mp.stem_for(half, arm, mc.SHUFFLED))
        parent = _loaded(mp.PARENTS[half][arm])
        c_brain, p_brain = child["brain"]["config"], parent["brain"]["config"]
        differing = {k for k in set(c_brain) | set(p_brain) if c_brain.get(k) != p_brain.get(k)}
        assert differing == {"weight_prior"}
        assert c_brain["weight_prior"] == "measured_shuffled"
        assert c_brain["measured_weight_scale"] == 1.0
        assert {k: v for k, v in child.items() if k != "brain"} == {
            k: v for k, v in parent.items() if k != "brain"
        }

    def test_the_pilots_own_panel_does_not_contain_the_shuffled_level(self) -> None:
        """Adding it there would make the committed pilot campaign read as incomplete."""
        assert mc.SHUFFLED not in mp.ALL_LEVELS
        assert all(level != mc.SHUFFLED for _, _, level in mp.ARM_BY_STEM.values())


class TestTheShuffledArms:
    @pytest.mark.parametrize("half", mc.HALVES)
    def test_same_values_on_the_covered_edges_placed_differently(self, half: str) -> None:
        """The shuffled wild type holds the measured wild type's covered multiset, elsewhere."""
        covered = sorted(coverage(measured_weights(), load_cook_2019_hermaphrodite()).covered)
        measured = _w_chem(mp.stem_for(half, "wt_learn", mc.MEASURED))
        shuffled = _w_chem(mp.stem_for(half, "wt_learn", mc.SHUFFLED))
        brain_stem = mp.stem_for(half, "wt_learn", mc.MEASURED)
        container = load_simulation_config(str(_CONFIGS / f"{brain_stem}.yml")).brain
        assert container is not None
        assert isinstance(container.config, ConnectomePPOBrainConfig)
        idx = ConnectomePPOBrain(
            config=container.config.model_copy(update={"seed": _SEED}),
            device=DeviceType.CPU,
        ).topology._idx
        # Placed values are value x the post neuron's 1/sqrt(in-degree), so compare per post neuron.
        by_post_m: dict[str, list[float]] = {}
        by_post_s: dict[str, list[float]] = {}
        for pre, post in covered:
            by_post_m.setdefault(post, []).append(float(measured[idx[pre], idx[post]]))
            by_post_s.setdefault(post, []).append(float(shuffled[idx[pre], idx[post]]))
        in_degree = measured.ne(0).sum(dim=0)
        unscaled_m = sorted(
            v * float(in_degree[idx[p]]) ** 0.5 for p, vs in by_post_m.items() for v in vs
        )
        unscaled_s = sorted(
            v * float(in_degree[idx[p]]) ** 0.5 for p, vs in by_post_s.items() for v in vs
        )
        assert unscaled_m == pytest.approx(unscaled_s, rel=1e-4)
        assert not torch.equal(measured, shuffled)

    @pytest.mark.parametrize("half", mc.HALVES)
    @pytest.mark.parametrize("wiring", ["wt", "rn"])
    def test_the_shuffled_floor_is_built_from_its_learning_arms_weights(
        self,
        half: str,
        wiring: str,
    ) -> None:
        """A floor at another level would gate the shuffled arm against another substrate."""
        learn = _w_chem(mp.stem_for(half, f"{wiring}_learn", mc.SHUFFLED))
        frozen = _w_chem(mp.stem_for(half, f"{wiring}_frozen", mc.SHUFFLED))
        assert torch.equal(learn, frozen)


class TestTheSeeds:
    def test_the_bands_are_fresh_and_disjoint(self) -> None:
        """No seed spent by an earlier panel, pilot or re-read, and none shared between learners."""
        spent = set(ops.BURNT_SEEDS) | set(ops.PILOT_SEEDS) | set(rr.SEEDS)
        for seeds in (*ops.SEEDS_BY_HALF.values(), *mp.SEEDS_BY_HALF.values()):
            spent |= set(seeds)
        ppo, reading = (set(mc.SEEDS_BY_HALF[h]) for h in ("ppo", "reading"))
        assert not (ppo | reading) & spent
        assert not ppo & reading
        assert (len(ppo), len(reading)) == (32, 48)
        assert min(ppo | reading) == mp.B1C_FIRST_SEED


class TestTheClassification:
    _MIN = 0.04

    @pytest.mark.parametrize(
        ("stats", "state"),
        [
            ((0.06, 0.02, 0.10, 0.01), "move_wt"),
            ((-0.06, -0.10, -0.02, 0.01), "move_null"),
            ((0.02, 0.01, 0.03, 0.01), "below"),
            ((-0.02, -0.03, -0.01, 0.01), "below"),
            ((0.0, -0.02, 0.02, 0.5), "no_move"),
            ((0.02, -0.02, 0.06, 0.5), "unresolved"),
            ((0.02, 0.01, 0.03, 0.2), "unresolved"),
            ((0.02, -0.01, 0.05, 0.01), "unresolved"),
        ],
        ids=[
            "toward-wild-type",
            "toward-null",
            "below-positive",
            "below-negative",
            "no-move",
            "spans-the-minimum",
            "interval-excludes-zero-q-not-significant",
            "q-significant-interval-spans-zero",
        ],
    )
    def test_each_state(self, stats: tuple[float, float, float, float], state: str) -> None:
        """Every registered state, including the three ways to be unresolved."""
        mean, lo, hi, q = stats
        assert mc.classify(mean, lo, hi, q, self._MIN) == state

    def test_the_minimum_is_two_thirds_of_the_committed_effect_on_each_draw(self) -> None:
        """PPO's reference is A.1's fan-in effect; the reading learner's is A.2's centre."""
        assert mc.minimum("ppo") == pytest.approx(2 / 3 * 0.055020833)
        assert mc.minimum("reading") == pytest.approx(2 / 3 * 0.2105)


class TestTheVerdictMap:
    @pytest.mark.parametrize(
        ("measured", "placement", "verdict"),
        [
            ("move_wt", "move_wt", "legible"),
            ("move_wt", "no_move", "value_distribution_wt"),
            ("move_wt", "move_null", "value_distribution_wt"),
            ("move_wt", "unresolved", "value_distribution_wt"),
            ("move_null", "move_null", "hides"),
            ("move_null", "no_move", "value_distribution_null"),
            ("move_null", "move_wt", "value_distribution_null"),
            ("no_move", "no_move", "null"),
            ("no_move", "move_wt", "placement_only"),
            ("no_move", "move_null", "placement_only"),
            ("no_move", "below", "null_placement_unresolved"),
            ("no_move", "unresolved", "null_placement_unresolved"),
            ("below", "move_wt", "below_minimum"),
            ("unresolved", "move_wt", "unresolved"),
        ],
    )
    def test_every_row(self, measured: str, placement: str, verdict: str) -> None:
        """The table fixed at registration, including both value-distribution directions."""
        assert mc.verdict(measured, placement) == verdict


def _gates(*, wt: float = 0.1, rn: float = 0.1, saturated: bool = False) -> dict[str, Any]:
    return {
        "wt": {"vs_floor": {"ci_lo": wt}},
        "rn": {"vs_floor": {"ci_lo": rn}},
        "gate_passes": wt > 0.0 and rn > 0.0,
        "saturated": saturated,
    }


def _interaction(mean: float, lo: float, hi: float, q: float) -> dict[str, Any]:
    return {"interaction_mean": mean, "test": {"ci_lo": lo, "ci_hi": hi, "bh_q": q}}


class TestTheGates:
    _MOVE = _interaction(0.2, 0.1, 0.3, 0.001)

    def test_a_wild_type_that_cannot_learn_the_measured_prior_is_the_lee_branch(self) -> None:
        """Read before any interaction, and closes the learner rather than reading a gap."""
        gates = {mc.RANDOM: _gates(), mc.MEASURED: _gates(wt=-0.1), mc.SHUFFLED: _gates()}
        out = mc.read_learner("ppo", gates, {"measured": self._MOVE, "placement": self._MOVE})
        assert out["verdict"] == "lee_unlearnable"

    @pytest.mark.parametrize(
        "bad",
        [_gates(rn=-0.1), _gates(saturated=True)],
        ids=["null-below-floor", "saturated"],
    )
    def test_a_level_that_cannot_carry_the_contrast_is_unreadable(self, bad: dict) -> None:
        """A broken null or two arms at the ceiling give no verdict, whatever the gap says."""
        gates = {mc.RANDOM: _gates(), mc.MEASURED: _gates(), mc.SHUFFLED: bad}
        out = mc.read_learner("reading", gates, {"measured": self._MOVE, "placement": self._MOVE})
        assert out["verdict"] == "unreadable"

    def test_readable_gates_reach_the_verdict(self) -> None:
        """With every level readable, both states are read against the learner's own minimum."""
        gates = {level: _gates() for level in mc.LEVELS}
        out = mc.read_learner("reading", gates, {"measured": self._MOVE, "placement": self._MOVE})
        assert out["states"] == {"measured": "move_wt", "placement": "move_wt"}
        assert out["verdict"] == "legible"


def test_the_primary_family_is_exactly_the_four_registered_interactions() -> None:
    """Two contrasts by two learners on the primary metric, corrected together and nothing else."""
    results = {
        half: {
            "interactions": {
                mc.PRIMARY_METRIC: {
                    name: {"test": {"wilcoxon_p": p}}
                    for (name, _, _), p in zip(mc.CONTRASTS, (0.01, 0.2), strict=True)
                },
            },
        }
        for half in mc.HALVES
    }
    mc.correct_family(results, mc.PRIMARY_METRIC)
    qs = [
        results[half]["interactions"][mc.PRIMARY_METRIC][name]["test"]["bh_q"]
        for half in mc.HALVES
        for name, _, _ in mc.CONTRASTS
    ]
    assert len(qs) == 4
    # Two-sided folding doubles 0.01 and 0.2 to 0.02 and 0.4; BH over four gives 0.04 and 0.4.
    assert sorted(qs) == pytest.approx([0.04, 0.04, 0.4, 0.4])
