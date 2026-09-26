"""A.6's gap-only split: its configs, pairing, manifest, identity check, verdicts and breakdown.

Covers the connectome-ppo-brain scenario "The gap-held null pairs exactly with the
degree-preserving null" on the committed configs, and the architecture-comparison-protocol
requirement "A committed baseline is reused only under a parsed-field identity check": the
comparator fails a run that differs in a single ``Run:`` line or in its final chemical matrix.
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

import gap_split as gs  # noqa: E402  # pyright: ignore[reportMissingImports]
import measured_prior_pilot as mp  # noqa: E402  # pyright: ignore[reportMissingImports]
import null_strength_control as nsc  # noqa: E402  # pyright: ignore[reportMissingImports]

_CONFIGS = _REPO / "configs" / "scenarios" / "foraging"
_SEED = 4


@functools.cache
def _loaded(stem: str) -> dict[str, Any]:
    """Load the whole simulation config as a run gets it, through the real loader."""
    return load_simulation_config(str(_CONFIGS / f"{stem}.yml")).model_dump()


@functools.cache
def _topology(stem: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build one arm at one seed and return its chemical mask, chemical weights and gap buffer."""
    container = load_simulation_config(str(_CONFIGS / f"{stem}.yml")).brain
    assert container is not None
    assert isinstance(container.config, ConnectomePPOBrainConfig)
    cfg = container.config.model_copy(update={"seed": _SEED})
    top = ConnectomePPOBrain(config=cfg, device=DeviceType.CPU).topology
    return top.m_chem.clone(), top.w_chem.detach().clone(), top.g_gap.clone()


class TestTheConfigs:
    def test_every_arm_is_a_committed_config(self) -> None:
        """Three levels over two learners, the wild-type arms shared, every stem on disk."""
        missing = [s for s in gs.LEVELS_BY_STEM if not (_CONFIGS / f"{s}.yml").is_file()]
        assert not missing, missing
        assert len(gs.LEVELS_BY_STEM) == 2 * 8
        assert len(gs.NEW_ARMS) == 4

    def test_a6s_arms_are_reused_as_they_are(self) -> None:
        """The full and chemical levels are A.6's own stems; only the gap-held arms are new."""
        for half in gs.HALVES:
            assert gs.STEMS[half][gs.FULL] == nsc.STEMS[half][nsc.FULL]
            assert gs.STEMS[half][gs.CHEMICAL] == nsc.STEMS[half][nsc.CHEMICAL]
        assert set(gs.LEVELS_BY_STEM) - set(nsc.LEVELS_BY_STEM) == set(gs.NEW_ARMS)

    @pytest.mark.parametrize("stem", sorted(gs.NEW_ARMS), ids=lambda s: s[-45:])
    def test_each_new_arm_differs_from_its_parent_in_the_wiring_alone(self, stem: str) -> None:
        """Through the real loader only ``wiring`` moves, and nothing outside the brain."""
        _, parent = gs.NEW_ARMS[stem]
        child, base = _loaded(stem), _loaded(parent)
        c_brain, p_brain = child["brain"]["config"], base["brain"]["config"]
        differing = {k for k in set(c_brain) | set(p_brain) if c_brain.get(k) != p_brain.get(k)}
        assert differing == {"wiring"}
        assert c_brain["wiring"] == "rewired_gap_junctions_held"
        assert {k: v for k, v in child.items() if k != "brain"} == {
            k: v for k, v in base.items() if k != "brain"
        }

    @pytest.mark.parametrize("half", gs.HALVES)
    @pytest.mark.parametrize("arm", ["rn_learn", "rn_frozen"])
    def test_the_gap_held_arm_pairs_exactly_with_its_parent(self, half: str, arm: str) -> None:
        """The current null's chemical mask and weights, and the wild type's gap junctions."""
        held = _topology(gs.STEMS[half][gs.GAP_HELD][arm])
        full = _topology(gs.STEMS[half][gs.FULL][arm])
        wild = _topology(gs.STEMS[half][gs.FULL]["wt_learn"])
        assert torch.equal(held[0], full[0])
        assert torch.equal(held[1], full[1])
        assert torch.equal(held[2], wild[2])
        assert not torch.equal(held[2], full[2])

    def test_the_seeds_are_a6s(self) -> None:
        """The split reuses A.6's bands by design: the pairing needs the same seeds."""
        assert gs.SEEDS_BY_HALF == nsc.SEEDS_BY_HALF
        assert gs.IDENTITY_SEED == {"ppo": 305, "reading": 337}


class TestTheManifest:
    def _campaigns(self, tmp_path: Path, half: str, seeds: tuple[int, ...]) -> tuple[Path, Path]:
        a6, split = tmp_path / "a6" / "logs", tmp_path / "split" / "logs"
        a6.mkdir(parents=True)
        split.mkdir(parents=True)
        for stem, (h, _, _) in gs.LEVELS_BY_STEM.items():
            if h != half:
                continue
            target = split if stem in gs.NEW_ARMS else a6
            for seed in seeds:
                (target / f"{stem}-seed{seed}.log").write_text("")
        return a6.parent, split.parent

    def test_both_campaigns_fill_every_level(self, tmp_path: Path) -> None:
        """A.6's runs and the new ones together complete all three levels."""
        seeds = gs.SEEDS_BY_HALF["ppo"][:2]
        dirs = self._campaigns(tmp_path, "ppo", seeds)
        manifest = gs.build_manifest(dirs, tmp_path / "m.txt", "ppo", seeds)
        mp.require_complete(manifest, "ppo", seeds, gs.LEVELS)
        lines = manifest.read_text().split("\n")[:-1]
        assert len(lines) == 3 * 4 * len(seeds)

    def test_a_missing_gap_held_run_is_refused(self, tmp_path: Path) -> None:
        """The new campaign's gap is caught, not scored around."""
        seeds = gs.SEEDS_BY_HALF["reading"][:2]
        a6, split = self._campaigns(tmp_path, "reading", seeds)
        victim = gs.STEMS["reading"][gs.GAP_HELD]["rn_learn"]
        (split / "logs" / f"{victim}-seed{seeds[1]}.log").unlink()
        manifest = gs.build_manifest((a6, split), tmp_path / "m.txt", "reading", seeds)
        with pytest.raises(mp.PilotError, match="gap_held/rn_learn"):
            mp.require_complete(manifest, "reading", seeds, gs.LEVELS)


_LOG = "setup\nRun: 1 Status: SUCCESS Steps: 90 Eaten: 10/10\nRun: 2 Status: FAIL Eaten: 4/10\n"


class TestTheIdentityComparator:
    def _pair(self, tmp_path: Path, rerun_text: str) -> tuple[Path, Path]:
        a, b = tmp_path / "a.log", tmp_path / "b.log"
        a.write_text(_LOG)
        b.write_text(rerun_text)
        return a, b

    def test_identical_runs_pass(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Same Run: lines and the same final chemical matrix."""
        monkeypatch.setattr(gs, "_final_w_chem", lambda _log: torch.ones(3, 3))
        a, b = self._pair(tmp_path, "different preamble\n" + _LOG.split("\n", 1)[1])
        assert gs.compare_runs(a, b)["identical"]

    def test_one_changed_run_line_fails(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A single episode's foods differing is a failed check."""
        monkeypatch.setattr(gs, "_final_w_chem", lambda _log: torch.ones(3, 3))
        a, b = self._pair(tmp_path, _LOG.replace("Eaten: 4/10", "Eaten: 5/10"))
        out = gs.compare_runs(a, b)
        assert not out["identical"]
        assert out["first_difference"] == 1

    def test_a_changed_final_w_chem_fails(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Identical episodes are not enough: the drift check reads the weights too."""
        a, b = self._pair(tmp_path, _LOG)
        weights = {a: torch.ones(3, 3), b: torch.ones(3, 3) * 1.0001}
        monkeypatch.setattr(gs, "_final_w_chem", lambda log: weights[log])
        out = gs.compare_runs(a, b)
        assert out["run_lines_equal"]
        assert not out["identical"]

    def test_missing_weights_fail(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A run whose weights cannot be found is not licensed by its episodes alone."""
        monkeypatch.setattr(gs, "_final_w_chem", lambda _log: None)
        a, b = self._pair(tmp_path, _LOG)
        assert not gs.compare_runs(a, b)["identical"]


def _gates(*, rn: float = 0.1, saturated: bool = False) -> dict[str, Any]:
    return {
        "wt": {"vs_floor": {"ci_lo": 0.1}},
        "rn": {"vs_floor": {"ci_lo": rn}},
        "gate_passes": rn > 0.0,
        "saturated": saturated,
    }


def _interaction(mean: float, lo: float, hi: float, q: float) -> dict[str, Any]:
    return {"interaction_mean": mean, "test": {"ci_lo": lo, "ci_hi": hi, "bh_q": q}}


_READABLE = {level: _gates() for level in gs.LEVELS}


class TestTheVerdicts:
    @pytest.mark.parametrize(
        ("interaction", "verdict"),
        [
            (_interaction(-0.025, -0.035, -0.02, 0.01), "gap_junctions"),
            (_interaction(-0.01, -0.015, -0.005, 0.01), "partial"),
            (_interaction(0.001, -0.01, 0.012, 0.6), "not_gap_junctions"),
            (_interaction(0.025, 0.02, 0.035, 0.01), "opposite"),
            (_interaction(-0.01, -0.03, 0.01, 0.4), "unresolved"),
        ],
        ids=["gap-junctions", "partial", "not-gap-junctions", "opposite", "unresolved"],
    )
    def test_every_row_on_ppos_minimum(self, interaction: dict[str, Any], verdict: str) -> None:
        """PPO's minimum is 2/3 of A.6's committed move, 0.0185."""
        assert gs.read_learner("ppo", _READABLE, interaction)["verdict"] == verdict

    def test_the_minimum_is_two_thirds_of_a6s_move(self) -> None:
        """Read off Logbook 074's committed interactions."""
        assert gs.minimum("ppo") == pytest.approx(2 / 3 * 0.027791666666666662)
        assert gs.minimum("reading") == pytest.approx(2 / 3 * 0.09859027777777779)

    @pytest.mark.parametrize("bad", [_gates(rn=-0.1), _gates(saturated=True)])
    def test_an_unreadable_level_gives_no_verdict(self, bad: dict[str, Any]) -> None:
        """Including the new gap-held level."""
        gates = {**_READABLE, gs.GAP_HELD: bad}
        out = gs.read_learner("ppo", gates, _interaction(-0.03, -0.04, -0.02, 0.01))
        assert out["verdict"] == "unreadable"


class TestTheBreakdown:
    def test_a6s_committed_move_is_read_seed_by_seed(self, tmp_path: Path) -> None:
        """The reproduction check passes on A.6's own figures and fails on a changed one."""
        committed = tmp_path / "a6.csv"
        committed.write_text(
            "half,seed,interaction_auc_success\nppo,305,-0.010000\nppo,306,0.020000\n",
        )
        assert gs.reproduces_a6("ppo", {305: -0.01, 306: 0.02}, committed)["reproduced"]
        out = gs.reproduces_a6("ppo", {305: -0.01, 306: 0.021}, committed)
        assert (out["reproduced"], out["mismatched"]) == (False, [306])

    def test_the_terms_sum_to_a6s_move(self) -> None:
        """gap(held) - gap(full) plus gap(chemical) - gap(held) is gap(chemical) - gap(full)."""
        full, held, chem = 0.05, 0.03, 0.02
        assert (held - full) + (chem - held) == pytest.approx(chem - full)


def test_the_family_is_the_two_primaries() -> None:
    """One gap-junction interaction per learner, corrected together."""
    results = {
        half: {"interactions": {gs.PRIMARY_METRIC: {"test": {"wilcoxon_p": p}}}}
        for half, p in zip(gs.HALVES, (0.01, 0.2), strict=True)
    }
    nsc.correct_family(results, gs.PRIMARY_METRIC)
    qs = sorted(results[h]["interactions"][gs.PRIMARY_METRIC]["test"]["bh_q"] for h in gs.HALVES)
    assert qs == pytest.approx([0.04, 0.4])
