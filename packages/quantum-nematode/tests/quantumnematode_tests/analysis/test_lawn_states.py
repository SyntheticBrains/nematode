"""D.1's analysis: intake plateaus and the learning gate, windows on lawns, bouts and depletion.

Covers the patchy-lawns change's readings: a learning arm passes when its intake plateau beats its
floor's, paired by seed; a window is on a lawn only when all of its positions are; bout durations
count complete bouts within on-lawn runs; the cell's density is recovered from intake.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Self

import numpy as np
import pytest
from quantumnematode.report.dtypes import BehaviourStep
from quantumnematode.validation import roaming_dwelling as rd

_REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO / "scripts" / "analysis"))
sys.path.insert(0, str(_REPO / "scripts" / "campaigns"))

import lawn_states as ls  # noqa: E402  # pyright: ignore[reportMissingImports]


def _log(path: Path, intakes: list[float]) -> Path:
    lines = [
        f"Run: {i + 1:<3} Status: SUCCESS Reason: max_steps            Steps: 720    "
        f"Reward:   10.00  Satiety: 50.0   Intake: {v:<8.3f} "
        for i, v in enumerate(intakes)
    ]
    path.write_text("\n".join(["noise", *lines, "tail"]) + "\n")
    return path


def test_the_plateau_is_the_final_quarter_of_intake(tmp_path: Path) -> None:
    """Eight episodes' intakes: the plateau is the mean of the last two."""
    log = _log(tmp_path / "run.log", [1, 1, 1, 1, 1, 1, 4, 6])
    assert ls.intake_plateau(log) == pytest.approx(5.0)
    assert ls.intake_plateau(tmp_path / "missing.log") is None


def test_the_learning_gate_pairs_by_seed() -> None:
    """Above the floor on every seed passes; level with it does not."""
    floor = {s: 3.0 + 0.1 * s for s in range(8)}
    assert ls.learning_gate({s: v + 5.0 for s, v in floor.items()}, floor)["passes"]
    assert not ls.learning_gate(dict(floor), floor)["passes"]


def _steps(xs: list[float], on: list[bool], intake: list[float]) -> list[BehaviourStep]:
    return [
        BehaviourStep(
            step=i,
            x=x,
            y=0.0,
            heading_rad=0.0,
            concentration=0.0,
            dc_dt=0.0,
            grad_dir=0.0,
            grad_strength=0.0,
            on_lawn=o,
            intake=b,
            satiety=1.0,
        )
        for i, (x, o, b) in enumerate(zip(xs, on, intake, strict=True))
    ]


def test_a_window_is_on_a_lawn_only_when_all_its_positions_are() -> None:
    """Leaving the lawn at the last sample takes that window off the lawn."""
    xs = [0.0, 0.5, 1.0, 1.5, 2.0]
    steps = _steps(xs, [True, True, True, True, False], [0.0, 0.1, 0.1, 0.05, 0.0])
    classifier = rd.GaussianClassifier(hmm=rd.load_calibrated_hmm())
    episode = ls.episode_windows(steps, classifier, intake_fraction=0.1, quality=1.0)
    assert episode.states[0] != rd.OFF_FOOD
    assert episode.states[1] == rd.OFF_FOOD
    assert episode.density[0] == pytest.approx(1.0)
    assert episode.density[1] == pytest.approx(0.5)


def test_bouts_count_only_complete_bouts_within_lawn_runs() -> None:
    """Edge bouts are open and left out; a bout ending in an off-lawn window is left out too."""
    d, r, off = rd.DWELLING, rd.ROAMING, rd.OFF_FOOD
    states = np.array([d, d, r, r, r, d, d, off, r, r])
    assert ls.bout_durations(states, rd.ROAMING) == [30.0]
    assert ls.bout_durations(states, rd.DWELLING) == []


def test_run_readings_pool_the_episodes() -> None:
    """Roaming share on lawns, and roaming where grazed against where fresh."""
    d, r, off = rd.DWELLING, rd.ROAMING, rd.OFF_FOOD
    episode = ls.Episode(
        states=np.array([d, r, r, off]),
        density=np.array([0.9, 0.2, 0.3, np.nan]),
    )
    out = ls.run_readings([episode])
    assert out["on_lawn_windows"] == 3
    assert out["roaming_fraction"] == pytest.approx(2 / 3)
    assert out["roaming_where_grazed"] == 1.0
    assert out["roaming_where_fresh"] == 0.0


def _run(arm: str, seed: int, roaming: float) -> dict[str, object]:
    return {
        "arm": arm,
        "seed": seed,
        "roaming_fraction": roaming,
        "n_roaming_bouts": 1,
        "n_dwelling_bouts": 1,
        "roaming_where_grazed": 0.5,
        "roaming_where_fresh": 0.4,
    }


class TestPanel:
    def test_a_learner_that_dwells_well_above_its_floor_dwells(self) -> None:
        """Dwelling share 0.6 above a floor of none, on every seed, reads ``dwells``."""
        runs = [_run("internal", s, 0.4 - 0.01 * (s % 3)) for s in range(16)]
        runs += [_run("internal_frozen", s, 1.0) for s in range(16)]
        out = ls.read_panel({"passes": True}, runs)
        assert out["state"] == "move_wt"
        assert out["verdict"] == "dwells"

    def test_a_learner_level_with_its_floor_does_not(self) -> None:
        """No difference on any seed reads ``no_dwelling_at_minimum``."""
        runs = [_run("internal", s, 0.9 + 0.001 * (s % 2)) for s in range(16)]
        runs += [_run("internal_frozen", s, 0.9 + 0.001 * ((s + 1) % 2)) for s in range(16)]
        assert ls.read_panel({"passes": True}, runs)["verdict"] == "no_dwelling_at_minimum"

    def test_a_failed_learning_gate_is_unreadable(self) -> None:
        """Without the gate, nothing is classified."""
        assert ls.read_panel({"passes": False}, [])["verdict"] == "unreadable"

    def test_the_readings_beside(self) -> None:
        """Bouts of both states are counted per arm; depletion is grazed minus fresh."""
        runs = [
            _run(arm, s, 0.5) for arm in ("internal", "blind", "internal_frozen") for s in (1, 2)
        ]
        out = ls.beside(runs)
        assert out["seeds_with_both_bouts"] == {"internal": 2, "blind": 2}
        assert out["roaming_grazed_minus_fresh"]["internal"][1] == pytest.approx(0.1)


def test_the_csv_has_a_row_per_run_with_its_plateau(tmp_path: Path) -> None:
    """Each evaluated run is a row, carrying its training intake plateau and its action noise."""
    result = {
        "gates": {"internal": {"learn": {"1": 3.0}, "floor": {"1": 1.0}}},
        "runs": [
            _run("internal", 1, 0.25) | {"action_log_std": [-3.0, -2.0]},
            _run("internal_frozen", 1, 1.0) | {"action_log_std": None},
        ],
    }
    lines = ls.write_csv(result, tmp_path / "per-run.csv").read_text().splitlines()
    assert len(lines) == 3
    header = lines[0].split(",")
    first = dict(zip(header, lines[1].split(","), strict=True))
    assert first["arm"] == "internal"
    assert first["intake_plateau"] == "3.0"
    assert first["dwelling_share"] == "0.75"
    assert first["log_std_speed"] == "-3.0"


class TestLeaving:
    def test_an_episode_records_its_exits(self) -> None:
        """Three samples on, then a full window off: one exit at that window."""
        xs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
        on = [True, True, True, False, False, False, False]
        steps = _steps(xs, on, [0.0] * 7)
        classifier = rd.GaussianClassifier(hmm=rd.load_calibrated_hmm())
        episode = ls.episode_windows(steps, classifier, intake_fraction=0.02, quality=1.0)
        assert episode.exits == (2,)

    def test_leaving_readings_pool_the_exits(self) -> None:
        """A band of roaming before an exit, against a run that mostly dwells."""
        d, r, off = rd.DWELLING, rd.ROAMING, rd.OFF_FOOD
        states = np.array([d] * 42 + [r] * 15 + [d] * 3 + [off] * 4)
        episode = ls.Episode(states=states, density=np.full(64, np.nan), exits=(60,))
        out = ls.leaving_readings([episode])
        assert out["exits"] == 1
        assert out["band_roaming"] == 1.0
        assert out["band_enrichment"] == pytest.approx(0.75)

    def test_dwelling_ends_by_the_spots_density(self) -> None:
        """Dwelling ends into roaming on a grazed spot and persists on a fresh one."""
        d, r = rd.DWELLING, rd.ROAMING
        episode = ls.Episode(
            states=np.array([d, d, r, d, d]),
            density=np.array([0.9, 0.2, 0.2, 0.9, 0.9]),
        )
        out = ls.dwelling_ends([episode])
        assert out["dwell_end_grazed"] == 1.0
        assert out["dwell_end_fresh"] == 0.0

    def _runs(self, enrichment: float, exits: int = 10) -> list[dict[str, object]]:
        return [
            {
                "arm": "internal",
                "seed": s,
                "exits": exits,
                "roaming_fraction": 0.4,
                "band_enrichment": enrichment + 0.01 * (s % 3),
            }
            for s in range(16)
        ]

    def test_a_coupling_above_the_minimum_leaves_from_roaming(self) -> None:
        """Enrichment of about 0.3 on every seed, minimum 0.1, reads ``leaves_from_roaming``."""
        out = ls.read_leaving(
            {"passes": True},
            self._runs(0.3),
            minimum=0.1,
            min_exits=3,
            min_seeds=8,
        )
        assert out["verdict"] == "leaves_from_roaming"

    def test_too_few_leaving_seeds_is_unreadable(self) -> None:
        """Seeds that leave less than the threshold are left out; too few left is unreadable."""
        out = ls.read_leaving(
            {"passes": True},
            self._runs(0.3, exits=1),
            minimum=0.1,
            min_exits=3,
            min_seeds=8,
        )
        assert out["verdict"] == "unreadable"
        assert len(out["seeds_too_few_exits"]) == 16


def test_a_seed_with_one_state_is_not_read() -> None:
    """A seed that never roams leaves while dwelling; it is reported and left out."""
    runs = [
        {"arm": "internal", "seed": s, "exits": 10, "roaming_fraction": 0.0, "band_enrichment": 0.0}
        for s in range(16)
    ]
    out = ls.read_leaving({"passes": True}, runs, minimum=0.1, min_exits=3, min_seeds=8)
    assert out["verdict"] == "unreadable"
    assert len(out["seeds_one_state"]) == 16


def test_every_leaving_cell_gets_the_registered_reading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI reads the leaving panel for any leaving cell, not only the one named ``leaving``."""
    learn = {s: 2.0 + 0.1 * s for s in range(4)}
    floor = dict.fromkeys(range(4), 0.5)

    def fake_plateaus(
        _logs: object,
        arm: str,
        seeds: tuple[int, ...],
        _cell: str,
    ) -> dict[int, float]:
        return {s: (floor if arm.endswith("_frozen") else learn)[s] for s in seeds}

    def fake_evaluate(job: tuple[str, int, list[str], int, str]) -> dict[str, object]:
        arm, seed, *_ = job
        return {
            "arm": arm,
            "seed": seed,
            "exits": 20,
            "roaming_fraction": 0.5,
            "band_enrichment": 0.0,
        }

    class _Pool:
        def __init__(self, **_: object) -> None: ...
        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_: object) -> None: ...
        def map(self, fn: object, jobs: list[object]) -> list[object]:
            return [fake_evaluate(j) for j in jobs]  # type: ignore[arg-type]

    monkeypatch.setattr(ls, "plateaus", fake_plateaus)
    monkeypatch.setattr(ls, "ProcessPoolExecutor", _Pool)
    out = tmp_path / "panel.json"
    ls.main(
        [
            "--logs",
            str(tmp_path),
            "--cell",
            "leaving15x4",
            "--seeds",
            "0",
            "1",
            "2",
            "3",
            "--leaving-minimum",
            "0.1",
            "--min-exits",
            "10",
            "--min-seeds",
            "4",
            "--out",
            str(out),
        ],
    )
    import json

    assert "leaving" in json.loads(out.read_text())


def test_a_named_learner_is_read_against_its_floor(tmp_path: Path) -> None:
    """Scenario "A connectome arm is read against its floor": the same reading, other arm names."""
    runs = [_run("connectome_e005", s, 0.3) for s in range(16)]
    runs += [_run("connectome_e005_frozen", s, 1.0) for s in range(16)]
    out = ls.read_panel({"passes": True}, runs, "connectome_e005", "connectome_e005_frozen", 0.3)
    assert out["minimum"] == 0.3
    assert out["verdict"] == "dwells"
    mlp = tmp_path / "d1.json"
    import json

    mlp.write_text(json.dumps({"runs": [_run("internal", s, 0.5) for s in range(16)]}))
    pairing = ls.against_mlp(runs, "connectome_e005", mlp)
    assert pairing["test"]["mean_delta"] == pytest.approx(0.2)


def test_action_noise_is_found_for_either_learner() -> None:
    """MLP-PPO keeps log_std at the top level; the connectome keeps it in its topology."""
    import torch

    assert ls.log_std_of({"log_std": {"log_std": torch.tensor([-3.0, -2.0])}}) == [-3.0, -2.0]
    assert ls.log_std_of({"topology": {"log_std": torch.tensor([-5.0, 1.0])}}) == [-5.0, 1.0]
    assert ls.log_std_of({"value": {}}) is None


# ── B.3's knockout readings ─────────────────────────────────────────────────────────────────────


def test_holm_steps_down_and_stays_monotone() -> None:
    """The smallest p is doubled, the next taken as is, and no adjusted q falls below an earlier."""
    assert ls.holm({"a": 0.01, "b": 0.04}) == pytest.approx({"a": 0.02, "b": 0.04})
    assert ls.holm({"a": 0.03, "b": 0.02}) == pytest.approx({"a": 0.04, "b": 0.04})


def _knockout_runs(tph1_roaming: float, pdfr1_roaming: float) -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    for s in range(16):
        jitter = 0.01 * (s % 3)
        runs += [
            _run("modulated", s, 0.3 + jitter),
            _run("modulated_frozen", s, 1.0),
            _run("modulated+tph1", s, tph1_roaming + jitter + 0.002 * (s % 2)),
            _run("modulated+pdfr1", s, pdfr1_roaming + jitter + 0.002 * (s % 2)),
        ]
    return runs


class TestKnockouts:
    def test_both_mutant_directions_reproduced(self) -> None:
        """Without serotonin it dwells less; without PDFR-1 it dwells more: both read positive."""
        out = ls.read_knockouts(
            {"passes": True},
            _knockout_runs(0.7, 0.0),
            minimum=0.2,
            measure="dwelling",
        )
        assert out["readable"]
        assert out["tph1"]["verdict"] == "mutant_direction"
        assert out["pdfr1"]["verdict"] == "mutant_direction"
        assert out["tph1"]["reading"]["test"]["mean_delta"] == pytest.approx(0.4, abs=0.01)
        assert out["pdfr1"]["reading"]["test"]["mean_delta"] == pytest.approx(0.3, abs=0.01)

    def test_the_opposite_direction_is_named(self) -> None:
        """A tph-1 knockout that dwells more moves against the mutant."""
        out = ls.read_knockouts(
            {"passes": True},
            _knockout_runs(0.0, 0.7),
            minimum=0.2,
            measure="dwelling",
        )
        assert out["tph1"]["verdict"] == "opposite_direction"
        assert out["pdfr1"]["verdict"] == "opposite_direction"

    def test_on_speed_the_mutant_directions_reverse(self) -> None:
        """Without serotonin it moves faster, without PDFR-1 slower: both read positive on speed."""
        runs = _knockout_runs(0.7, 0.0)
        speeds = {"modulated": 0.03, "modulated+tph1": 0.06, "modulated+pdfr1": 0.01}
        runs = [
            r | {"on_lawn_speed": speeds[str(r["arm"])] + 0.001 * (int(r["seed"]) % 3)}  # type: ignore[call-overload]
            if r["arm"] in speeds
            else r
            for r in runs
        ]
        out = ls.read_knockouts({"passes": True}, runs, minimum=0.01)
        assert out["measure"] == "speed"
        assert out["tph1"]["reading"]["test"]["mean_delta"] == pytest.approx(0.03)
        assert out["pdfr1"]["reading"]["test"]["mean_delta"] == pytest.approx(0.02)
        assert out["tph1"]["verdict"] == "mutant_direction"
        assert out["pdfr1"]["verdict"] == "mutant_direction"

    def test_significant_dwelling_below_the_stage0_minimum_passes_the_gate(self) -> None:
        """The gate asks for a dwelling state, not stage 0's size: a significant +0.1 passes."""
        runs = _knockout_runs(0.7, 0.0)
        runs = [
            r | {"roaming_fraction": 0.9 - 0.001 * (int(r["seed"]) % 3)}  # type: ignore[call-overload]
            if r["arm"] == "modulated"
            else r
            for r in runs
        ]
        out = ls.read_knockouts({"passes": True}, runs, minimum=0.01, measure="dwelling")
        assert out["dwelling_gate"]["state"] == "below"
        assert out["readable"]

    def test_a_knockout_with_no_paired_seed_leaves_the_panel_unreadable(self) -> None:
        """Holm needs both readings: without the pdfr-1 arm, neither is read, and it says why."""
        runs = [r for r in _knockout_runs(0.7, 0.0) if r["arm"] != "modulated+pdfr1"]
        out = ls.read_knockouts({"passes": True}, runs, minimum=0.2, measure="dwelling")
        assert out["verdict"] == "unreadable"
        assert out["missing_knockouts"] == ["pdfr1"]
        assert "tph1" not in out

    def test_an_intact_network_that_does_not_dwell_is_unreadable(self) -> None:
        """The dwelling gate comes first: no state to shift, no knockout reading."""
        runs = _knockout_runs(0.7, 0.0)
        runs = [r | {"roaming_fraction": 1.0} if r["arm"] == "modulated" else r for r in runs]
        out = ls.read_knockouts({"passes": True}, runs, minimum=0.2)
        assert out["verdict"] == "unreadable"
        assert "tph1" not in out


def test_concentrations_are_read_by_state() -> None:
    """Each window carries its steps' mean concentrations; they are pooled per state."""
    windows = [
        ls.Episode(
            states=np.array([rd.ROAMING, rd.DWELLING, rd.OFF_FOOD]),
            density=np.full(3, np.nan),
            concentrations=np.array([[0.1, 0.5], [0.3, 0.2], [9.0, 9.0]]),
        ),
    ]
    out = ls.concentrations_by_state(windows)
    assert out == pytest.approx(
        {
            "serotonin_roaming": 0.1,
            "pdf_roaming": 0.5,
            "serotonin_dwelling": 0.3,
            "pdf_dwelling": 0.2,
        },
    )


def test_on_lawn_speed_pools_the_on_lawn_windows() -> None:
    """Speed is averaged over on-lawn windows only; off-lawn travel does not count."""
    episodes = [
        ls.Episode(
            states=np.array([rd.ROAMING, rd.DWELLING, rd.OFF_FOOD]),
            density=np.full(3, np.nan),
            speed=np.array([0.04, 0.02, 0.5]),
        ),
    ]
    assert ls.on_lawn_speed(episodes) == pytest.approx(0.03)
    assert ls.on_lawn_speed([]) is None


def test_learned_strengths_are_read_from_the_weights(tmp_path: Path) -> None:
    """The four pathway strengths are the softplus of the saved values; None without the field."""
    import torch

    path = tmp_path / "final.pt"
    torch.save({"topology": {"modulator_rho": torch.zeros(4)}}, path)
    strengths = ls.modulator_strengths(path)
    assert strengths is not None
    assert list(strengths) == list(ls.MODULATOR_PATHWAYS)
    assert strengths["mod1"] == pytest.approx(float(np.log(2.0)))
    torch.save({"topology": {}}, path)
    assert ls.modulator_strengths(path) is None


def test_an_acute_arm_evaluates_the_trained_weights_with_its_knockout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``modulated+pdfr1`` reads the intact arm's log and weights, with the knockout overridden."""
    weights = tmp_path / "final.pt"
    log = tmp_path / f"{ls.gen.stem('modulated')}-seed7.log"
    log.write_text("")
    seen: dict[str, object] = {}

    def fake_capture(
        config: Path,
        seed: int,
        found: Path | None,
        episodes: int,
        overrides: dict[str, object] | None = None,
    ) -> tuple[object, ...]:
        seen.update(config=config, seed=seed, weights=found, overrides=overrides)
        return [], [], type("L", (), {"quality": (1.0, 1.0), "intake_fraction": 0.02})(), None

    monkeypatch.setattr(ls.bc, "final_weights", lambda _log: weights)
    monkeypatch.setattr(ls, "capture", fake_capture)
    monkeypatch.setattr(ls, "modulator_strengths", lambda _w: None)
    monkeypatch.setattr(ls, "action_log_std", lambda _w: None)
    monkeypatch.setattr(ls, "run_readings", lambda _windows: {})
    ls.evaluate_run(("modulated+pdfr1", 7, [str(tmp_path)], 2, "patch"))
    assert seen["weights"] == weights
    assert seen["overrides"] == {"modulator_knockout": "pdfr1"}
    assert seen["config"] == ls.gen.FORAGING / f"{ls.gen.stem('modulated')}.yml"
