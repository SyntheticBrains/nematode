"""M.1(a)'s diagnosis on synthetic telemetry: pairs, identity, the envelope and the causes.

Covers the training-telemetry requirement "Per-update PPO telemetry" as the diagnosis reads it: each
collapsed run against its intact twin on the same seed, the departure rule, every cause's pattern,
leave-one-out for the intact side, and the indicated / not indicated / mixed rule.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_root = Path(__file__).resolve()
while _root != _root.parent and not (_root / "scripts" / "analysis").is_dir():
    _root = _root.parent
sys.path.insert(0, str(_root / "scripts" / "analysis"))

import collapse_diagnosis as cd  # noqa: E402  # pyright: ignore[reportMissingImports]

_H = 3000
_ONSET = 2500


class TestPairs:
    def test_eight_pairs_each_with_the_other_wiring(self) -> None:
        assert len(cd.PAIRS) == 8
        for pair in cd.PAIRS:
            assert {pair.collapsed, pair.intact} == {cd.WT, cd.GAP_ONLY}
        assert sum(p.collapsed == cd.WT for p in cd.PAIRS) == 2

    def test_24_runs_none_twice(self) -> None:
        assert len(cd.RUNS) == len(set(cd.RUNS)) == 24
        assert {s for stem, s in cd.RUNS if stem == cd.MLP} == set(cd.MLP_SEEDS)

    def test_every_run_has_a_committed_plateau(self) -> None:
        committed = cd.committed_plateaus()
        assert all(run in committed for run in cd.RUNS)


class TestIdentity:
    def test_a_matching_plateau_is_identical_and_a_missing_run_is_not(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        stem, seed = cd.RUNS[0]
        lines = ["Run: 1 Status: FAILED Eaten: 0/20"] * 6 + [
            "Run: 9 Status: SUCCESS Eaten: 20/20",
        ] * 2
        (tmp_path / f"{stem}-seed{seed}.log").write_text("\n".join(lines) + "\n")
        table = dict.fromkeys(cd.RUNS, (12.5, 1))
        table[stem, seed] = (100.0, 1)
        monkeypatch.setattr(cd, "committed_plateaus", lambda: table)
        result = cd.identity([tmp_path])
        assert result["runs"][f"{stem}-seed{seed}"]["identical"] is True
        assert result["all_identical"] is False

    def test_a_different_plateau_is_not_identical(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        stem, seed = cd.RUNS[0]
        (tmp_path / f"{stem}-seed{seed}.log").write_text("Run: 1 Status: SUCCESS Eaten: 20/20\n")
        monkeypatch.setattr(cd, "committed_plateaus", lambda: dict.fromkeys(cd.RUNS, (99.9, 1)))
        assert cd.identity([tmp_path])["runs"][f"{stem}-seed{seed}"]["identical"] is False


class TestSeries:
    def test_smoothing_is_a_trailing_mean_that_skips_unseen_episodes(self) -> None:
        series = np.array([np.nan, 1.0, 3.0, 5.0])
        out = cd.smoothed(series, window=2)
        assert np.isnan(out[0])
        assert list(out[1:]) == [1.0, 2.0, 4.0]

    def test_updates_average_within_an_episode_and_carry_forward(self) -> None:
        base = {
            "weight_norms": {"topology.w": 3.0, "topology.v": 4.0, "critic.w": 100.0},
            "saturated_fraction": 0.1,
            "srank": 50,
            "adv_std": 1.0,
            "grad_norm": 0.4,
            "approx_kl": 0.01,
            "clip_fraction": 0.1,
            "explained_variance": 0.5,
            "value_loss": 1.0,
            "log_std": -0.5,
            "action_mean_large_fraction": 0.0,
            "entropy": 1.0,
            "critic_grad": 2.0,
            "actor_grad": 1.0,
        }
        lines = [
            {**base, "episode": 1, "adv_std": 1.0},
            {**base, "episode": 1, "adv_std": 3.0},
            {**base, "episode": 3, "adv_std": 5.0},
        ]
        series = cd.episode_series(lines, horizon=4)
        assert list(series["adv_std"]) == [2.0, 2.0, 5.0, 5.0]
        assert series["weight_norm"][0] == pytest.approx(5.0)
        assert series["critic_over_actor_grad"][0] == pytest.approx(2.0)

    def test_alignment_pads_before_the_start(self) -> None:
        window = cd.aligned(np.arange(10.0), onset=5, lead=6)
        assert np.isnan(window[:2]).all()
        assert list(window[2:]) == [0.0, 1.0, 2.0, 3.0]


class TestDeparture:
    def test_a_long_stretch_above_the_envelope_departs_with_its_lead(self) -> None:
        mean, sd = np.zeros(1500), np.ones(1500)
        series = np.zeros(1500)
        series[-300:] = 5.0
        assert cd.departure(series, mean, sd, +1) == 300
        assert cd.departure(series, mean, sd, -1) is None

    def test_a_short_stretch_does_not(self) -> None:
        mean, sd = np.zeros(1500), np.ones(1500)
        series = np.zeros(1500)
        series[-150:] = 5.0
        assert cd.departure(series, mean, sd, +1) is None


def _run(rng: np.random.Generator, *, grow: bool = False) -> dict[str, np.ndarray]:
    """Build a synthetic run's per-episode statistics; ``grow`` adds weight growth before onset."""
    out = {name: 1.0 + 0.01 * rng.standard_normal(_H) for name in cd.STATISTICS}
    if grow:
        ramp = np.clip((np.arange(_H) - (_ONSET - 1200)) / 400, 0, None)
        out["weight_norm"] = out["weight_norm"] + ramp
        out["saturated_fraction"] = out["saturated_fraction"] + ramp
        out["srank"] = out["srank"] - ramp
    return out


class TestCauses:
    def _panel(self, growing: int) -> dict[str, object]:
        rng = np.random.default_rng(0)
        seeds = list(range(8))
        collapsed = {s: _run(rng, grow=s < growing) for s in seeds}
        intact = {s: _run(rng) for s in seeds}
        return cd.diagnose_series(collapsed, intact, dict.fromkeys(seeds, _ONSET))

    def test_a_pattern_in_every_collapse_is_indicated(self) -> None:
        result = self._panel(growing=8)
        weight = result["causes"]["weight_growth"]  # type: ignore[index]
        assert weight["collapsed_departures"] == 8
        assert weight["intact_departures"] <= 1
        assert weight["reading"] == "indicated"
        assert result["indicated"] == ["weight_growth"]  # type: ignore[index]

    def test_a_pattern_in_half_the_collapses_is_mixed(self) -> None:
        assert self._panel(growing=4)["causes"]["weight_growth"]["reading"] == "mixed"  # type: ignore[index]

    def test_a_pattern_in_none_is_not_indicated(self) -> None:
        result = self._panel(growing=0)
        assert all(c["reading"] == "not_indicated" for c in result["causes"].values())  # type: ignore[union-attr]

    def test_every_registered_cause_is_read(self) -> None:
        assert set(self._panel(growing=8)["causes"]) == set(cd.CAUSES)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("collapsed", "intact", "reading"),
        [(6, 1, "indicated"), (6, 2, "mixed"), (5, 0, "mixed"), (2, 0, "not_indicated")],
    )
    def test_the_thresholds(self, collapsed: int, intact: int, reading: str) -> None:
        assert cd.classify(collapsed, 8, intact, 8) == reading

    def test_stale_moments_need_the_gradient_first(self) -> None:
        assert cd._ordered("stale_second_moments", {"grad_norm": 400, "approx_kl": 300})
        assert not cd._ordered("stale_second_moments", {"grad_norm": 200, "approx_kl": 300})
        assert cd._ordered("weight_growth", {})
