"""B v2's controls: the round trip's denoiser, the blind fit's stopping rule and the verdict.

Covers the benchmark-controls spec's scenarios "The known exploit still passes" and "One unlucky N2
unit" (the verdict's checks), and the measurement-model round trip on toy tracks.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from quantumnematode.benchmark import grading as g
from quantumnematode.benchmark import measurement as mm

_REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO / "scripts" / "analysis"))
import benchmark_v2_controls as v2  # noqa: E402  # pyright: ignore[reportMissingImports]


def test_denoise_fills_interior_gaps_and_keeps_the_ends_missing() -> None:
    """Interior gaps are interpolated; seconds before the first and after the last fix stay NaN."""
    x = np.array([np.nan, 0.0, 1.0, np.nan, 3.0, 4.0, np.nan])
    sx, _ = v2.denoise(x, x)
    assert math.isnan(sx[0])
    assert math.isnan(sx[-1])
    assert np.isfinite(sx[1:-1]).all()
    # A straight line is unchanged by the centred mean away from the ends.
    assert sx[3] == pytest.approx(2.0)


def test_round_trip_on_a_toy_track_recovers_the_raw_value() -> None:
    """A noisy straight track, denoised and measured again, keeps its speed within the noise."""
    rng = np.random.default_rng(1)
    n = 600
    true_x, true_y = 0.05 * np.arange(n, dtype=float), np.zeros(n)
    model = mm.TrackerModel(jitter_mm=0.005)
    raw_x = true_x + rng.normal(0, model.jitter_mm, n)
    raw_y = true_y + rng.normal(0, model.jitter_mm, n)
    sx, sy = v2.denoise(raw_x, raw_y)
    mx, my = v2._positions(mm.measure(v2._as_steps(sx, sy), seed=3, model=model))
    raw_speed = np.nanmean(np.hypot(np.diff(raw_x), np.diff(raw_y)))
    trip_speed = np.nanmean(np.hypot(np.diff(mx), np.diff(my)))
    assert trip_speed == pytest.approx(raw_speed, rel=0.02)


def test_z_rows_pass_within_one_scale() -> None:
    """A round-tripped value passes within one band scale of the raw value."""
    bands = {"a": g.Band(0.0, 2.0), "b": g.Band(0.0, 2.0)}
    rows = v2._z_rows({"a": 1.0, "b": 1.0}, {"a": 2.5, "b": 3.5}, bands)
    assert rows["a"]["passes"] is True
    assert rows["b"]["passes"] is False
    assert rows["b"]["z"] == pytest.approx(1.25)


def test_a_wide_sigma_plateau_does_not_stop_the_blind_fit() -> None:
    """The plateau stops a blind restart only once sigma is at most sigma0."""
    flat = [5.0] * 40
    assert v2.plateau_stop(flat, v2.BLIND_SIGMA0) is True
    assert v2.plateau_stop(flat, 0.3) is False


def test_blind_simulation_indices_do_not_collide() -> None:
    """Each restart, generation and member has its own simulation index."""
    ks = {
        v2.blind_k(r, n, m)
        for r in range(2)
        for n in range(1, v2.BLIND_V2_CAP + 1)
        for m in range(v2.BLIND_POPSIZE)
    }
    assert len(ks) == 2 * v2.BLIND_V2_CAP * v2.BLIND_POPSIZE


def _parts(**changes: Any) -> tuple[dict, dict, dict, dict]:
    negatives = {
        n: {"fails_wild_type": True, "fails_chemotaxis": n in v2.OPEN_LOOP}
        for n in (*v2.OPEN_LOOP, "untrained_mlp", "untrained_connectome")
    }
    oracle = {"round_trip": {"passes": True}, "reachability": {"passes": True}}
    record = {"chemotaxis_leave_one_out": {"passes": True}}
    read = {
        "n2_control": {"passes": True},
        "negatives": negatives,
        "c1_answer": {"passes_wild_type": False},
    }
    blind = {"passes_wild_type": False}
    for key, value in changes.items():
        if key == "c1_passes":
            read["c1_answer"]["passes_wild_type"] = value
        elif key == "open_loop_passes_chemotaxis":
            negatives["hmm"]["fails_chemotaxis"] = not value
        elif key == "reachability":
            oracle["reachability"] = value
    return oracle, record, read, blind


def test_every_check_holding_is_valid() -> None:
    """With every registered reading in place, B v2 is VALID."""
    assert v2.verdict(*_parts())["verdict"] == "VALID"


def test_the_known_exploit_passing_voids_v2() -> None:
    """Covers "The known exploit still passes": C.1's v1 answer passing v2 makes it VOID."""
    out = v2.verdict(*_parts(c1_passes=True))
    assert out["verdict"] == "VOID"
    assert out["checks"]["c1_answer_fails"] is False


def test_an_open_loop_worm_passing_chemotaxis_voids_v2() -> None:
    """A gradient-blind negative inside the chemotaxis family makes B v2 VOID."""
    assert v2.verdict(*_parts(open_loop_passes_chemotaxis=True))["verdict"] == "VOID"


def test_an_unrun_reachability_oracle_voids_v2() -> None:
    """When the round trip fails, reachability is not run and B v2 is VOID."""
    assert v2.verdict(*_parts(reachability="not run"))["verdict"] == "VOID"


def test_gap_fill_changes_no_present_reading() -> None:
    """Gap-filling interpolates interior gaps only; every present reading is kept exactly."""
    x = np.array([np.nan, 0.3, np.nan, 0.9, 1.0, np.nan])
    fx, _ = v2.gap_fill(x, x)
    assert math.isnan(fx[0])
    assert math.isnan(fx[-1])
    assert fx[2] == pytest.approx(0.6)
    np.testing.assert_array_equal(fx[[1, 3, 4]], x[[1, 3, 4]])


def test_noise_statistics_of_a_still_worm_scale_with_jitter() -> None:
    """A still worm's step median grows with the jitter; it reads as a finite onset rate."""
    rng = np.random.default_rng(2)

    def stretches(sigma: float) -> list[tuple[np.ndarray, np.ndarray]]:
        return [tuple(rng.normal(0.0, sigma, size=(2, 60))) for _ in range(50)]

    low, high = v2.noise_statistics(stretches(0.002)), v2.noise_statistics(stretches(0.008))
    assert high["step_median_mm"] == pytest.approx(4 * low["step_median_mm"], rel=0.1)
    assert high["paused_window_speed_median"] > low["paused_window_speed_median"]
    assert math.isfinite(low["onsets_per_paused_min"])


def test_the_library_never_sees_the_check() -> None:
    """Covers "The library never sees the check": the two halves of the band dates are disjoint."""
    from quantumnematode.benchmark import references as refs

    library, check = v2.split_band_dates(refs.load_scheer_tracks())
    assert len(library) == v2.LIBRARY_DATES
    assert not set(library) & set(check)
    assert len(library) + len(check) == 41


def test_the_permutation_gate_brackets_zero_and_passes_a_match() -> None:
    """Covers "A still simulated worm reads like a still real worm": the gate's range and read."""
    rng = np.random.default_rng(4)
    per_date = {
        d: {
            "onsets": int(rng.integers(50, 150)),
            "seconds": 3000,
            "speeds": rng.gamma(2.0, 0.002, 300),
        }
        for d in range(41)
    }
    gate = v2.permutation_gate(per_date)
    for name in v2.GATED:
        low, high = gate["ranges"][name]
        assert low < 0 < high
    real = [tuple(rng.normal(0, 0.005, size=(2, 30))) for _ in range(200)]
    same = v2.gated_check(real, real, gate)
    assert same["passes"] is True
    shifted = v2.gated_check(real, [(x * 4, y * 4) for x, y in real], gate)
    assert shifted["passes"] is False


def test_the_reachability_worm_spans_its_box_around_the_matched_values() -> None:
    """Covers "The matched worm misses with no noise": the box's centre is B.4's matched worm."""
    import benchmark_controls as bc  # pyright: ignore[reportMissingImports]

    states = json.loads(bc.PARAMS_PATH.read_text())["states"]
    centre = v2.reach_parameters(np.full(len(v2.REACH_FREE), 0.5))
    low = v2.reach_parameters(np.zeros(len(v2.REACH_FREE)))
    for state, name in v2.REACH_FREE:
        assert centre[state][name] == pytest.approx(states[state][name])
        assert low[state][name] == pytest.approx(states[state][name] / v2.REACH_BOX)
