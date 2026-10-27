#!/usr/bin/env python
r"""B v2's probes and controls: the measurements behind the revision, its oracles and its verdict.

``probe`` re-reads, into a committed record, every number the revision was designed from:

- **Scheer, band cohorts:** per-state 10-s window speed (median and 90th percentile) and angular
  speed by recording date, against C.1's v1 answer; detected on-lawn reversals per minute for the
  real worms, B.4's matched open-loop worm and C.1; the tracker's jitter (from paused seconds) and
  its lost seconds by kind of second, with the lengths of real missing runs;
- **Chen 2025:** the vendored field against the concentrations the authors record along their
  tracks, the pixel scale, the tracker's jitter (raw positions against the authors' smoothed track)
  and the fragment-length distribution of the standard-protocol recordings;
- **the blind worm on v1** (``--blind``): C.1 with every sensing term pinned off, fitted by sep-CMA
  against v1's D and graded on five populations.

``oracle`` (refuses until launch Part A is committed): the measurement-model round trip on both
rigs (band-side tracks denoised, measured again, every new statistic within |z| <= 1 of the raw
value), then the reachability oracle (B.4's matched open-loop worm, 40 measured populations).

``read`` (refuses until Part B and the bands it binds are committed; runs once): the fresh N2
control (4 units), B.4's six negatives on v2, C.1's v1 answer on five populations, and W's minimum
on both family sets.

``blind`` (refuses until Part B is committed; resumable): the blind worm fitted against v2's D, two
sep-CMA restarts with the sigma guard, the best graded on five populations.

``verdict``: VALID only when every registered check holds, into ``controls_v2.json``.

Usage::

    uv run python scripts/analysis/benchmark_v2_controls.py probe \
        --chen-archive <Chen_learning_2024.zip> [--blind] [--out <probes.json>]
    uv run python scripts/analysis/benchmark_v2_controls.py oracle --chen-archive <zip>
    uv run python scripts/analysis/benchmark_v2_controls.py read
    uv run python scripts/analysis/benchmark_v2_controls.py blind
    uv run python scripts/analysis/benchmark_v2_controls.py verdict
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_controls as bc
import benchmark_grade as bg
import npe_recovery as nr
from quantumnematode.benchmark import grading as g
from quantumnematode.benchmark import measurement as mm
from quantumnematode.benchmark import readouts as ro
from quantumnematode.benchmark import references as refs
from quantumnematode.validation import roaming_dwelling as rd

DESCRIPTION = "B v2's probes and controls: the measurements behind the revision and its verdict."
_REPO = Path(__file__).resolve().parents[2]
SUPPORT = _REPO / "docs" / "experiments" / "logbooks" / "supporting" / "115-benchmark-v2"
PROBES_PATH = SUPPORT / "probes.json"
C1_FIT = _REPO / "docs" / "experiments" / "logbooks" / "supporting" / "113-c1-fit" / "fit.json"

# The planning probes' seeds, re-used so their numbers are reproduced (recorded in the seed table).
SPEED_PROBE_BASE = 143_000_000
REVERSAL_PROBE_BASE = 144_000_000
BLIND_FIT_BASE = 140_000_000
BLIND_GRADE_BASE = 141_000_000
BLIND_CMA_SEED = 9331
ANIMALS = 200
MIN_STATE_WINDOWS = 30
STILL_STEP_MM = 0.005  # a 1-s step below this leaves the heading undefined


def _state_quantiles(rows: list[tuple[int, float, float]], state: int) -> list[float]:
    r = np.array([(a, b) for s, a, b in rows if s == state])
    if len(r) == 0:
        return [float("nan")] * 3
    return [
        float(np.median(r[:, 0])),
        float(np.quantile(r[:, 0], 0.9)),
        float(np.nanmedian(r[:, 1])),
    ]


def window_rows(
    x: np.ndarray,
    y: np.ndarray,
    on: np.ndarray,
    clf: rd.GaussianClassifier,
) -> list[tuple[int, float, float]]:
    """Return each on-lawn classified window's state, 10-s speed and angular speed."""
    w = g.window_inputs(x, y, on)
    states = clf.states(w.speed, w.angular, w.on)
    return [
        (int(s), float(sp), float(an))
        for s, sp, an, o in zip(states, w.speed, w.angular, w.on, strict=True)
        if o and s in (rd.ROAMING, rd.DWELLING) and np.isfinite(sp)
    ]


def onlawn_reversals(x: np.ndarray, y: np.ndarray, on: np.ndarray) -> tuple[int, int]:
    """Return detected reversal onsets on the lawn, and the valid on-lawn seconds.

    The probe's own count, at the detector's default floor, as the planning probes read it.
    """
    onsets = ro.reversal_onsets(x, y, g.OFF_FOOD_DETECTOR)
    valid = np.isfinite(np.diff(x)) & np.isfinite(np.diff(y)) & on[1:] & on[:-1]
    kept = [o for o in onsets if int(o) < len(on) and on[int(o)]]
    return len(kept), int(valid.sum())


def _simulate_small_lawn(
    controller: Any,  # noqa: ANN401 - any batched controller
    base: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    from quantumnematode.evaluation.batched import BatchedEvaluator

    config = nr._shared()["configs"]["small_lawn"]
    traj = BatchedEvaluator(config, controller, [base + i for i in range(ANIMALS)]).run()
    assert traj.assay is not None  # noqa: S101 - an assay config records its timing
    analysis = tuple(traj.assay["analysis_steps"])  # type: ignore[arg-type]
    out = []
    for i in range(ANIMALS):
        steps = bg.window("small_lawn", traj.to_behaviour_steps(i), analysis)
        out.append(
            (
                np.array([s.x for s in steps]),
                np.array([s.y for s in steps]),
                np.array([bool(s.on_lawn) for s in steps]),
            ),
        )
    return out


def _c1_controller() -> Any:  # noqa: ANN401 - a controller
    from quantumnematode.evaluation.controllers import BatchedMechanistic

    fit = json.loads(C1_FIT.read_text())
    z = fit["restarts"][fit["chosen_restart"]]["answer"]
    return BatchedMechanistic(nr.parameters_from_unit(np.asarray(z)))


def interior_runs(missing: np.ndarray, omega: np.ndarray) -> dict[str, Any]:
    """Return the tracker's interior losses: start rates and lengths, by the starting second's kind.

    A run touching a track's first or last second is the worm not yet found or already lost, not a
    loss the tracker made, so it is left out. A second is at risk while it is tracked.
    """
    lengths: dict[str, list[int]] = {"omega": [], "other": []}
    at_risk = {"omega": 0, "other": 0}
    n = missing.shape[1]
    for m, o in zip(missing, omega, strict=True):
        i = 0
        while i < n:
            kind = "omega" if o[i] else "other"
            if not m[i]:
                at_risk[kind] += 1
                i += 1
                continue
            j = i
            while j < n and m[j]:
                j += 1
            if i > 0 and j < n:
                lengths[kind].append(j - i)
            i = j
    return {
        kind: {
            "runs": len(lengths[kind]),
            "start_rate_per_s": len(lengths[kind]) / at_risk[kind],
            "length_median_s": float(np.median(lengths[kind])) if lengths[kind] else 0.0,
            "length_p90_s": float(np.quantile(lengths[kind], 0.9)) if lengths[kind] else 0.0,
            "lengths_s": sorted(lengths[kind]),
        }
        for kind in lengths
    }


def scheer_probe(clf: rd.GaussianClassifier, bands: dict[str, Any]) -> dict[str, Any]:
    """Return the Scheer probes: per-state kinematics, reversals and the tracker."""
    tracks = refs.load_scheer_tracks()
    arm = bands["arms"]["small_lawn"]["arm"]
    rows = tracks.arm(arm)
    rows = rows[tracks.side[rows] == refs.BANDS]
    by_date: dict[int, list[tuple[int, float, float]]] = {}
    rev: dict[int, list[int]] = {}
    for i in rows:
        on = g.real_on_lawn(tracks.x[i], tracks.y[i], tracks.boundary_radius[i])
        by_date.setdefault(int(tracks.date[i]), []).extend(
            window_rows(tracks.x[i], tracks.y[i], on, clf),
        )
        n, s = onlawn_reversals(tracks.x[i], tracks.y[i], on)
        acc = rev.setdefault(int(tracks.date[i]), [0, 0])
        acc[0] += n
        acc[1] += s
    states = {"dwelling": rd.DWELLING, "roaming": rd.ROAMING}
    real = {
        name: np.array(
            [
                _state_quantiles(r, s)
                for r in by_date.values()
                if sum(1 for t, _, _ in r if t == s) >= MIN_STATE_WINDOWS
            ],
        )
        for name, s in states.items()
    }
    c1_rows: list[tuple[int, float, float]] = []
    c1_rev = [0, 0]
    for x, y, on in _simulate_small_lawn(_c1_controller(), SPEED_PROBE_BASE):
        c1_rows.extend(window_rows(x, y, on, clf))
    for x, y, on in _simulate_small_lawn(_c1_controller(), REVERSAL_PROBE_BASE):
        n, s = onlawn_reversals(x, y, on)
        c1_rev[0] += n
        c1_rev[1] += s
    params = json.loads(bc.PARAMS_PATH.read_text())
    hmm_rev = [0, 0]
    for x, y, on in _simulate_small_lawn(bc.hazard_controller("hmm", params), REVERSAL_PROBE_BASE):
        n, s = onlawn_reversals(x, y, on)
        hmm_rev[0] += n
        hmm_rev[1] += s
    kinematics = {}
    for name, s in states.items():
        mu, sd = real[name].mean(axis=0), real[name].std(axis=0, ddof=1)
        c1 = np.array(_state_quantiles(c1_rows, s))
        kinematics[name] = {
            label: {
                "real_mean": float(mu[k]),
                "real_sd_dates": float(sd[k]),
                "dates": len(real[name]),
                "C1": float(c1[k]),
                "z": float((c1[k] - mu[k]) / sd[k]),
            }
            for k, label in enumerate(("window_speed_median", "window_speed_p90", "angular_median"))
        }
    rates = np.array([60.0 * n / s for n, s in rev.values() if s > 0])
    mu, sd = float(rates.mean()), float(rates.std(ddof=1))
    reversal = {
        "real_per_min": mu,
        "real_sd_dates": sd,
        "dates": len(rates),
        "matched_hmm_per_min": 60.0 * hmm_rev[0] / hmm_rev[1],
        "C1_per_min": 60.0 * c1_rev[0] / c1_rev[1],
    }
    reversal["matched_hmm_z"] = (reversal["matched_hmm_per_min"] - mu) / sd
    reversal["C1_z"] = (reversal["C1_per_min"] - mu) / sd
    # The tracker: jitter from paused seconds, lost seconds by kind, missing-run lengths.
    n = tracks.x.shape[1]

    def per_second(name: str, *, every: bool) -> np.ndarray:
        f = tracks.flag(name)[rows][:, : 3 * n].reshape(len(rows), n, 3)
        return f.all(axis=2) if every else f.any(axis=2)

    paused = per_second("pause", every=True)
    run = paused[:, :-1] & paused[:, 1:]
    steps = np.hypot(np.diff(tracks.x[rows]), np.diff(tracks.y[rows]))[run]
    missing = ~np.isfinite(tracks.x[rows])
    omega = per_second("omega", every=False)
    tracker = {
        "paused_step_median_mm": float(np.nanmedian(steps)),
        "jitter_mm": mm.jitter_from_steps(steps),
        "p_missing_omega_second": float(missing[omega].mean()),
        "p_missing_not_omega_second": float(missing[~omega].mean()),
        "interior_runs": interior_runs(missing, omega),
    }
    return {"kinematics": kinematics, "reversal": reversal, "tracker": tracker}


def _archive_mat(archive: Path, member: str) -> dict[str, Any]:
    import scipy.io as sio

    with zipfile.ZipFile(archive) as z:
        return sio.loadmat(io.BytesIO(z.read(member)), squeeze_me=True, struct_as_record=False)


def chen_probe(archive: Path) -> dict[str, Any]:
    """Return the Chen probes: the field's registration, the tracker's jitter and fragments."""
    chen = refs.load_chen_tracks()
    with np.load(refs.CHEN_TRACKS_PATH) as store:
        scale = float(store["mm_per_px"])
    data = _archive_mat(archive, "temp/data4plots/Data_nai_test2.mat")["Data"]
    xs, ys, dcs = [], [], []
    for d in data:
        xy, dc = np.atleast_2d(d.xy), np.atleast_1d(d.dc)
        k = min(xy.shape[1], len(dc))
        xs.append(xy[0, :k] * scale)
        ys.append(xy[1, :k] * scale)
        dcs.append(dc[:k])
    x, y, dc = (np.concatenate(v) for v in (xs, ys, dcs))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(dc)
    field = chen.concentration(x[ok], y[ok])
    raw = _archive_mat(archive, "temp/track_data/rawTracks_N2nai.mat")["Tracks"]
    residual = []
    for t in raw:
        p = np.atleast_2d(t.Path)
        sx, sy = np.atleast_1d(t.SmoothX), np.atleast_1d(t.SmoothY)
        if len(sx) == len(p):
            residual.extend((p[:, 0] - sx).tolist())
            residual.extend((p[:, 1] - sy).tolist())
    standard = [r for r, p in enumerate(chen.recording_protocol) if p == "standard"]
    lengths = chen.fragment_length_s[np.isin(chen.fragment_recording, standard)]
    return {
        "field_registration": {
            "samples": int(ok.sum()),
            "correlation": float(np.corrcoef(field, dc[ok])[0, 1]),
            "median_abs_difference": float(np.median(np.abs(field - dc[ok]))),
        },
        "mm_per_px": scale,
        "jitter_mm": float(np.std(residual) * scale),
        "fragment_length_s": {
            "fragments": len(lengths),
            "median": float(np.median(lengths)),
            "p90": float(np.quantile(lengths, 0.9)),
        },
    }


def blind_probe() -> dict[str, Any]:
    """Fit the blind worm against v1's D, as the planning probe did, and grade it on five populations."""
    from concurrent.futures import ProcessPoolExecutor

    import c1_fit as c1
    import cma
    import cma_recovery as cr
    from quantumnematode.evaluation.controllers import MECHANISTIC_PARAMETERS

    pinned = {
        "a_intake": 0.5,
        "a_satiety": 0.5,
        "b_intake": 0.5,
        "b_satiety": 0.5,
        "kappa_weathervane": 0.5,
        "beta_reversal": 0.0,
        "beta_omega": 0.0,
        "edge": 0.0,
        "search_amplitude": 0.0,
        "search_tau_s": 0.5,
    }
    free = [p for p in MECHANISTIC_PARAMETERS if p not in pinned]

    def full(z: np.ndarray) -> list[float]:
        out = dict(pinned)
        out.update(zip(free, map(float, z), strict=True))
        return [out[p] for p in MECHANISTIC_PARAMETERS]

    es = cma.CMAEvolutionStrategy(
        [0.5] * len(free),
        0.2,
        {
            "CMA_diagonal": True,
            "bounds": [0, 1],
            "popsize": 16,
            "seed": BLIND_CMA_SEED,
            "verbose": -9,
        },
    )
    medians: list[float] = []
    k = 0
    with ProcessPoolExecutor(max_workers=16, initializer=nr._worker_setup) as pool:
        for _ in range(1, 151):
            population = es.ask()
            records = list(
                pool.map(
                    nr.simulate,
                    range(k, k + 16),
                    [full(x) for x in population],
                    [BLIND_FIT_BASE] * 16,
                ),
            )
            k += 16
            scores = [c1.objective(nr._finite(r)) for r in records]
            es.tell(population, scores)
            medians.append(float(np.median(scores)))
            if cr.plateau_reached(medians):
                break
        z = full(np.clip(es.result.xfavorite, 0, 1))
        grades = list(pool.map(c1.grade_answer, range(5), [z] * 5, [BLIND_GRADE_BASE] * 5))
    passes = [c1.population_passes(gr["family_passes"]) for gr in grades]
    return {
        "stop_generation": len(medians),
        "population_passes": passes,
        "passes_wild_type": c1.wild_type_verdict(passes),
        "population_D": [gr["D"] for gr in grades],
        "family_passes": [gr["family_passes"] for gr in grades],
    }


def _probe(args: argparse.Namespace) -> int:
    bands = bg.load_bands()
    clf = bg.classifier(bands)
    record: dict[str, Any] = {
        "description": "The measurements B v2 was designed from, re-read by a committed script.",
        "seeds": {
            "speed_probe": SPEED_PROBE_BASE,
            "reversal_probe": REVERSAL_PROBE_BASE,
            "blind_fit": BLIND_FIT_BASE,
            "blind_grade": BLIND_GRADE_BASE,
            "blind_cma": BLIND_CMA_SEED,
        },
        "scheer": scheer_probe(clf, bands),
        "chen": chen_probe(args.chen_archive),
    }
    if args.blind:
        record["blind_on_v1"] = blind_probe()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(bg._finite(record), indent=2, allow_nan=False) + "\n")
    print(json.dumps(bg._finite({k: v for k, v in record.items() if k != "description"}), indent=2))
    return 0


# ---------------------------------------------------------------------------------------------
# Registered settings for the oracles and the controls
# ---------------------------------------------------------------------------------------------

LAUNCH_PATH = SUPPORT / "launch.md"
BANDS_V2_PATH = _REPO / "data" / "benchmark" / "bands_v2.json"
BANDS_V2_RECORD = SUPPORT / "bands_v2_record.json"
ORACLE_PATH = SUPPORT / "oracle.json"
ORACLE_AMENDED_PATH = SUPPORT / "oracle_amendment1.json"
READ_PATH = SUPPORT / "read.json"
BLIND_PATH = SUPPORT / "blind_v2.json"
CONTROLS_PATH = SUPPORT / "controls_v2.json"
BLIND_STATE = _REPO / "build" / "benchmark-v2" / "blind"
CONTROLS_V1 = _REPO / "docs" / "experiments" / "logbooks" / "supporting" / "110-benchmark-controls"

SMOOTH_S = 5
STORE_UM_MM = 1e-3  # the store keeps positions in whole micrometres
ROUND_TRIP_BAR = 1.0
ROUND_TRIP_BASE = 154_000_000
JITTER_BASE = 155_000_000
PAUSED_MIN_S = 10
JITTER_SCALE_N = 200
JITTER_GATED = {
    "onsets_per_paused_min": "reversals_per_on_lawn_min",
    "paused_window_speed_median": "dwelling_speed_median",
    "paused_window_speed_p90": "dwelling_speed_p90",
}
ORACLE_BASE = 150_000_000
ORACLE_POPULATIONS = 40
ORACLE_MEAN_Z = 1.0
ORACLE_SHARE = 0.9
ORACLE_STATISTICS = tuple(s for s in g.KINEMATIC_STATISTICS if s != "roaming_survival_5min")
N2_ARM = "N2_od2_Data"
N2_SPLIT_SEED = 2035
N2_BAND_SEED = 2042
N2_ALLOWANCE = 1
W_SPLITS = 1000
W_FITTED_SEED = 2034
W_SEARCH_SEED = 2036
W_SIZE_STEP = 25
W_QUANTILE = 95
OPEN_LOOP = ("hmm", "poisson", "random_walk", "correlated_walk")
GRADE_POPULATIONS = 5
BLIND_V2_CMA_SEEDS = (9401, 9402)
BLIND_V2_FIT_BASE = 151_000_000
BLIND_V2_GRADE_BASE = 152_000_000
C1_V2_GRADE_BASE = 153_000_000
BLIND_V2_CAP = 210
BLIND_POPSIZE = 16
BLIND_SIGMA0 = 0.2
BLIND_PINNED = {
    "a_intake": 0.5,
    "a_satiety": 0.5,
    "b_intake": 0.5,
    "b_satiety": 0.5,
    "kappa_weathervane": 0.5,
    "beta_reversal": 0.0,
    "beta_omega": 0.0,
    "edge": 0.0,
    "search_amplitude": 0.0,
    "search_tau_s": 0.5,
}


def _committed(*, bands: bool) -> bool:
    """Return whether Part A (and, with ``bands``, Part B with the bands file) is committed."""
    import cma_recovery as cr

    bound = (BANDS_V2_PATH, BANDS_V2_RECORD, NOISE_PATH, REACH_PATH) if bands else ()
    return cr.launch_committed(bound=bound, launch_path=LAUNCH_PATH)


# ---------------------------------------------------------------------------------------------
# The measurement-model round trip
# ---------------------------------------------------------------------------------------------


def denoise(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a 1-s track's denoised positions: interior gaps interpolated, a centred 5-s mean.

    The seconds before the worm is first found and after it is last seen stay missing; every
    second between is a position, so the measurement model alone decides which go missing.
    """
    out = []
    for v in (np.asarray(x, dtype=float), np.asarray(y, dtype=float)):
        present = np.flatnonzero(np.isfinite(v))
        filled = np.full(len(v), np.nan)
        if len(present):
            a, b = present[0], present[-1] + 1
            t = np.arange(a, b)
            filled[a:b] = np.interp(t, present, v[present])
            kernel = np.ones(SMOOTH_S)
            seg = filled[a:b]
            total = np.convolve(seg, kernel, mode="same")
            count = np.convolve(np.ones(len(seg)), kernel, mode="same")
            filled[a:b] = total / count
        out.append(filled)
    return out[0], out[1]


def _as_steps(x: np.ndarray, y: np.ndarray, primitive: np.ndarray | None = None) -> list[Any]:
    from quantumnematode.report.behaviour import BehaviourStep

    return [
        BehaviourStep(
            step=t,
            x=float(x[t]),
            y=float(y[t]),
            heading_rad=0.0,
            concentration=0.0,
            dc_dt=0.0,
            grad_dir=0.0,
            grad_strength=0.0,
            primitive=None if primitive is None else ("omega" if primitive[t] else "run"),
        )
        for t in range(len(x))
    ]


def _positions(steps: list[Any]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([s.x for s in steps], dtype=float),
        np.array([s.y for s in steps], dtype=float),
    )


def _z_rows(
    raw: dict[str, float],
    trip: dict[str, float],
    scales: dict[str, g.Band],
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for name, value in raw.items():
        if name not in scales:
            # Computed and reported beside the graded statistics; there is no band to read it on.
            rows[name] = {"raw": value, "round_trip": trip[name], "reported": True}
            continue
        z = (trip[name] - value) / scales[name].scale
        rows[name] = {
            "raw": value,
            "round_trip": trip[name],
            "scale": scales[name].scale,
            "z": z,
            "passes": bool(abs(z) <= ROUND_TRIP_BAR),
        }
    return rows


def scheer_round_trip(bands: dict[str, Any]) -> dict[str, Any]:
    """Return the lawn rig's round trip: band-side PD1074 tracks, denoised and measured again."""
    import benchmark_bands as bb

    tracks = refs.load_scheer_tracks()
    clf = bg.classifier(bands)
    model = mm.TrackerModel.from_dict(bands["measurement"]["lawn"])
    rows = bb.band_rows(tracks, bb.WILD_TYPE)
    n = tracks.x.shape[1]
    omega = tracks.flag("omega")[rows][:, : 3 * n].reshape(len(rows), n, 3).any(axis=2)
    raw_k, trip_k = np.zeros(g.KINEMATIC_FIELDS), np.zeros(g.KINEMATIC_FIELDS)
    raw_l, trip_l = np.zeros(len(g.TALLY_FIELDS)), np.zeros(len(g.TALLY_FIELDS))
    for j, i in enumerate(rows):
        x, y, radius = tracks.x[i], tracks.y[i], tracks.boundary_radius[i]
        on = g.real_on_lawn(x, y, radius)
        w = g.animal_windows(x, y, on, clf)
        raw_k += g.kinematic_tally(w, x, y, on)
        raw_l += g.tally(w)
        sx, sy = denoise(x, y)
        measured = mm.measure(_as_steps(sx, sy, omega[j]), ROUND_TRIP_BASE + int(i), model)
        mx, my = _positions(measured)
        mon = g.real_on_lawn(mx, my, radius)
        mw = g.animal_windows(mx, my, mon, clf)
        trip_k += g.kinematic_tally(mw, mx, my, mon)
        trip_l += g.tally(mw)
    kin = _z_rows(
        g.kinematic_statistics(raw_k),
        g.kinematic_statistics(trip_k),
        bg.family_bands_v2(bands, "kinematics", len(rows)),
    )
    lawn = _z_rows(
        g.lawn_statistics(raw_l),
        g.lawn_statistics(trip_l),
        bg.family_bands_v2(bands, "small_lawn", len(rows)),
    )
    return {
        "animals": len(rows),
        "denoiser": f"interior gaps interpolated, then a centred {SMOOTH_S}-s moving average",
        "kinematics": kin,
        "small_lawn_reported": lawn,
        "passes": all(r["passes"] for r in kin.values()),
    }


def gap_fill(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a 1-s track with its interior gaps interpolated and nothing else changed."""
    out = []
    for v in (np.asarray(x, dtype=float), np.asarray(y, dtype=float)):
        present = np.flatnonzero(np.isfinite(v))
        filled = np.full(len(v), np.nan)
        if len(present):
            a, b = present[0], present[-1] + 1
            filled[a:b] = np.interp(np.arange(a, b), present, v[present])
        out.append(filled)
    return out[0], out[1]


def _band_rows_and_omega(tracks: refs.ScheerTracks) -> tuple[np.ndarray, np.ndarray]:
    import benchmark_bands as bb

    rows = bb.band_rows(tracks, bb.WILD_TYPE)
    n = tracks.x.shape[1]
    omega = tracks.flag("omega")[rows][:, : 3 * n].reshape(len(rows), n, 3).any(axis=2)
    return rows, omega


def loss_round_trip(bands: dict[str, Any]) -> dict[str, Any]:
    """Return the loss model's round trip: gap-filled band-side tracks through the losses alone.

    Gap-filling leaves every real reading in place, so the real tracker's own noise stays and no
    jitter is added; the losses are drawn as the lawn model draws them, on the real omega flags.
    """
    tracks = refs.load_scheer_tracks()
    clf = bg.classifier(bands)
    lawn = mm.TrackerModel.from_dict(bands["measurement"]["lawn"])
    losses = mm.TrackerModel(
        jitter_mm=0.0,
        p_omega=lawn.p_omega,
        p_other=lawn.p_other,
        runs_omega_s=lawn.runs_omega_s,
        runs_other_s=lawn.runs_other_s,
        stream_key=lawn.stream_key,
    )
    rows, omega = _band_rows_and_omega(tracks)
    raw, trip = np.zeros(g.KINEMATIC_FIELDS), np.zeros(g.KINEMATIC_FIELDS)
    for j, i in enumerate(rows):
        x, y, radius = tracks.x[i], tracks.y[i], tracks.boundary_radius[i]
        on = g.real_on_lawn(x, y, radius)
        raw += g.kinematic_tally(g.animal_windows(x, y, on, clf), x, y, on)
        fx, fy = gap_fill(x, y)
        measured = mm.measure(_as_steps(fx, fy, omega[j]), ROUND_TRIP_BASE + int(i), losses)
        mx, my = _positions(measured)
        mon = g.real_on_lawn(mx, my, radius)
        trip += g.kinematic_tally(g.animal_windows(mx, my, mon, clf), mx, my, mon)
    rows_z = _z_rows(
        g.kinematic_statistics(raw),
        g.kinematic_statistics(trip),
        bg.family_bands_v2(bands, "kinematics", len(rows)),
    )
    return {
        "animals": len(rows),
        "applied": "the losses alone, on gap-filled tracks",
        "kinematics": rows_z,
        "passes": all(r["passes"] for r in rows_z.values() if "passes" in r),
    }


def paused_stretches(
    tracks: refs.ScheerTracks,
    rows: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return every band-side run of paused, tracked seconds at least :data:`PAUSED_MIN_S` long.

    A second is paused when the authors flag all three of its frames as a pause.
    """
    n = tracks.x.shape[1]
    paused = tracks.flag("pause")[rows][:, : 3 * n].reshape(len(rows), n, 3).all(axis=2)
    out = []
    for j, i in enumerate(rows):
        x, y = tracks.x[i], tracks.y[i]
        keep = paused[j] & np.isfinite(x) & np.isfinite(y)
        t = 0
        while t < n:
            if not keep[t]:
                t += 1
                continue
            u = t
            while u < n and keep[u]:
                u += 1
            if u - t >= PAUSED_MIN_S:
                out.append((np.asarray(x[t:u], dtype=float), np.asarray(y[t:u], dtype=float)))
            t = u
    return out


def noise_statistics(stretches: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, float]:
    """Return a still worm's readings: 1-s steps, detected onsets and 10-s window speeds."""
    steps, speeds = [], []
    onsets, seconds = 0, 0
    for x, y in stretches:
        steps.append(np.hypot(np.diff(x), np.diff(y)))
        onsets += len(
            ro.reversal_onsets(x, y, g.OFF_FOOD_DETECTOR, floor=g.ON_LAWN_REVERSAL_FLOOR_MM),
        )
        seconds += len(x) - 1
        w = g.window_inputs(x, y, np.ones(len(x), dtype=bool))
        speeds.append(w.speed[np.isfinite(w.speed)])
    step, speed = np.concatenate(steps), np.concatenate(speeds)
    return {
        "step_median_mm": float(np.median(step)),
        "step_p90_mm": float(np.quantile(step, 0.9)),
        "onsets_per_paused_min": 60.0 * onsets / seconds,
        "paused_window_speed_median": float(np.median(speed)),
        "paused_window_speed_p90": float(np.quantile(speed, 0.9)),
    }


def jitter_check(bands: dict[str, Any]) -> dict[str, Any]:
    """Return the jitter check: the real tracker's readings of a still worm against the model's.

    On the authors' paused stretches the worm is still, so its readings are the tracker's noise.
    Each stretch's model counterpart is its mean position held still, with the lawn model's
    jitter. The gated readings are judged on the kinematics bands' scales at the graded size,
    since a still worm is where the noise weighs most on what is graded.
    """
    tracks = refs.load_scheer_tracks()
    lawn = mm.TrackerModel.from_dict(bands["measurement"]["lawn"])
    jitter = mm.TrackerModel(jitter_mm=lawn.jitter_mm, stream_key=lawn.stream_key)
    rows, _ = _band_rows_and_omega(tracks)
    real = paused_stretches(tracks, rows)
    model = []
    for k, (x, y) in enumerate(real):
        still_x, still_y = np.full(len(x), float(np.mean(x))), np.full(len(y), float(np.mean(y)))
        model.append(_positions(mm.measure(_as_steps(still_x, still_y), JITTER_BASE + k, jitter)))
    a, b = noise_statistics(real), noise_statistics(model)
    scales = bg.family_bands_v2(bands, "kinematics", JITTER_SCALE_N)
    rows_z = {}
    for name, value in a.items():
        row: dict[str, Any] = {"real": value, "model": b[name]}
        if name in JITTER_GATED:
            scale = scales[JITTER_GATED[name]].scale
            row.update(
                scale_of=JITTER_GATED[name],
                scale=scale,
                z=(b[name] - value) / scale,
                passes=bool(abs(b[name] - value) / scale <= ROUND_TRIP_BAR),
            )
        rows_z[name] = row
    return {
        "stretches": len(real),
        "paused_seconds": int(sum(len(x) for x, _ in real)),
        "minimum_stretch_s": PAUSED_MIN_S,
        "scale_n": JITTER_SCALE_N,
        "readings": rows_z,
        "passes": all(r["passes"] for r in rows_z.values() if "passes" in r),
    }


def chen_smoothed(
    archive: Path,
) -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Return each standard-protocol fragment's raw and the authors' smoothed positions at 1 s, mm.

    The vendored fragments are the raw paths; the authors' smoothing is in their raw-track file,
    matched to each fragment by its path. The raw positions are re-derived here and must equal the
    vendored ones, to the store's micrometre.
    """
    import extract_benchmark_references as ex

    chen = refs.load_chen_tracks()
    with np.load(refs.CHEN_TRACKS_PATH) as store:
        scale = float(store["mm_per_px"])
    raw = np.atleast_1d(_archive_mat(archive, "temp/track_data/rawTracks_N2nai.mat")["Tracks"])
    smooth = {
        np.atleast_2d(t.Path).astype(float).tobytes(): (
            np.atleast_1d(t.SmoothX),
            np.atleast_1d(t.SmoothY),
        )
        for t in raw
    }
    naive = _archive_mat(archive, ex.CHEN_TRACKS_MEMBER)["track_learn_all"][ex.CHEN_NAIVE]
    vendored = chen.fragments()
    out = []
    f = 0
    for r, recording in enumerate(naive):
        for t in np.atleast_1d(recording):
            p = np.atleast_2d(t.Path).astype(float)
            time_s = np.atleast_1d(t.Time).astype(float)
            _, xy = ex.resample_fragment(p, time_s)
            if len(xy) < 2:
                continue
            rec, _, vx, vy = vendored[f]
            f += 1
            if chen.recording_protocol[r] != "standard":
                continue
            if rec != r or not (
                np.allclose(vx, xy[:, 0] * scale, rtol=0, atol=STORE_UM_MM)
                and np.allclose(vy, xy[:, 1] * scale, rtol=0, atol=STORE_UM_MM)
            ):
                msg = f"fragment {f - 1} does not match the vendored store"
                raise RuntimeError(msg)
            sx, sy = smooth[p.tobytes()]
            _, sxy = ex.resample_fragment(np.column_stack([sx, sy]).astype(float), time_s)
            out.append((r, vx, vy, sxy[:, 0] * scale, sxy[:, 1] * scale))
    return out


def chen_round_trip(bands: dict[str, Any], archive: Path) -> dict[str, Any]:
    """Return Chen's rig's round trip: the authors' smoothed fragments with the model's jitter.

    The fragments' cuts are the real ones, so only the jitter is applied; the model's fragment
    lengths are the real distribution by construction.
    """
    from quantumnematode.benchmark import chemotaxis as ch

    chen = refs.load_chen_tracks()
    model = mm.TrackerModel.from_dict(bands["measurement"]["chemotaxis"])
    jitter_only = mm.TrackerModel(jitter_mm=model.jitter_mm, stream_key=model.stream_key)
    fragments = chen_smoothed(archive)
    raw = [(vx, vy) for _, vx, vy, _, _ in fragments]
    trip = []
    for k, (_, _, _, sx, sy) in enumerate(fragments):
        mx, my = _positions(mm.measure(_as_steps(sx, sy), ROUND_TRIP_BASE + k, jitter_only))
        trip.append((mx, my))
    rows = _z_rows(
        ch.chemotaxis_statistics(raw, chen.concentration),
        ch.chemotaxis_statistics(trip, chen.concentration),
        bg.family_bands_v2(bands, "chemotaxis", 0),
    )
    return {
        "fragments": len(fragments),
        "denoiser": "the authors' SmoothX/SmoothY, resampled to 1 s as the raw paths are",
        "applied": "jitter only; the fragments are the real ones",
        "chemotaxis": rows,
        "passes": all(r["passes"] for r in rows.values()),
    }


# ---------------------------------------------------------------------------------------------
# The reachability oracle
# ---------------------------------------------------------------------------------------------


def _oracle_population(
    p: int,
    base: int = ORACLE_BASE,
    path: Path = bg.BANDS_V2_PATH,
) -> dict[str, float]:
    """Return the reachability worm's kinematic statistics on one measured population.

    The worm is the two-state open-loop worm fitted through the measurement model; the worm
    matched on per-state mean speeds and hazards misses the kinematics with no noise at all.
    """
    del path  # the bands are read where the worm was fitted, from the vendored file
    worm = json.loads(REACH_PATH.read_text())
    seeds = [base + ANIMALS * p + i for i in range(ANIMALS)]
    return _measured_small_lawn(reach_controller(worm["states"]), seeds)


def reachability_oracle(
    bands: dict[str, Any],
    workers: int,
    path: Path = bg.BANDS_V2_PATH,
) -> dict[str, Any]:
    """Return the reachability oracle: 40 measured populations of the matched open-loop worm."""
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=workers, initializer=nr._worker_setup) as pool:
        values = list(
            pool.map(
                _oracle_population,
                range(ORACLE_POPULATIONS),
                [ORACLE_BASE] * ORACLE_POPULATIONS,
                [path] * ORACLE_POPULATIONS,
            ),
        )
    band = bg.family_bands_v2(bands, "kinematics", ANIMALS)
    bar = float(bands["bars"]["wild_type"])
    rows = {}
    for name in ORACLE_STATISTICS:
        z = np.array([g.z_score(v[name], band[name]) for v in values])
        mean, share = float(np.mean(z)), float(np.mean(np.abs(z) <= bar))
        rows[name] = {
            "mean_z": mean,
            "share_passing": share,
            "z": z.tolist(),
            "passes": bool(abs(mean) <= ORACLE_MEAN_Z and share >= ORACLE_SHARE),
        }
    survival = np.array(
        [g.z_score(v["roaming_survival_5min"], band["roaming_survival_5min"]) for v in values],
    )
    return {
        "populations": ORACLE_POPULATIONS,
        "animals": ANIMALS,
        "seed_base": ORACLE_BASE,
        "statistics": rows,
        "reported_roaming_survival_mean_z": float(np.mean(survival)),
        "passes": all(r["passes"] for r in rows.values()),
    }


def _oracle(args: argparse.Namespace) -> int:
    """Run the oracles as amended: the losses, the jitter and Chen's round trip, then reachability.

    Part A's original round trip (``scheer_round_trip``, a 5-s mean as the denoiser) failed and is
    kept in ``oracle.json``; the amendment's readings go to ``oracle_amendment1.json``.
    """
    if not _committed(bands=False):
        print(
            f"{LAUNCH_PATH} (Part A) is not committed and clean; nothing is run.",
            file=sys.stderr,
        )
        return 1
    if ORACLE_AMENDED_PATH.exists():
        print(f"{ORACLE_AMENDED_PATH} exists: the amended oracle is read once.", file=sys.stderr)
        return 1
    bands = bg.load_bands_v2(args.bands)
    record: dict[str, Any] = {
        "description": "B v2's oracles as amended: the losses, the jitter, Chen's round trip, "
        "then reachability.",
    }
    rigs = {
        "lawn_losses": loss_round_trip(bands),
        "lawn_jitter": jitter_check(bands),
        "chemotaxis": chen_round_trip(bands, args.chen_archive),
    }
    record["round_trip"] = {**rigs, "passes": all(r["passes"] for r in rigs.values())}
    if record["round_trip"]["passes"]:
        record["reachability"] = reachability_oracle(bands, args.workers, args.bands)
        record["passes"] = bool(record["reachability"]["passes"])
    else:
        record["reachability"] = "not run: the round trip failed, so the measurement model is wrong"
        record["passes"] = False
    ORACLE_AMENDED_PATH.write_text(json.dumps(bg._finite(record), indent=2, allow_nan=False) + "\n")
    print(json.dumps(bg._finite(record), indent=2)[:6000])
    return 0 if record["passes"] else 1


# ---------------------------------------------------------------------------------------------
# The positive control: fresh N2
# ---------------------------------------------------------------------------------------------


def n2_unread_dates(tracks: refs.ScheerTracks) -> list[int]:
    """Return N2's held-out dates that no earlier read has touched: all but tph-1's paired dates."""
    controls = json.loads((CONTROLS_V1 / "controls.json").read_text())
    read = set(controls["positive_controls"]["units"]["mutant/tph-1"]["paired_dates"])
    rows = tracks.arm(N2_ARM)
    held = rows[tracks.side[rows] == refs.HELD_OUT]
    return sorted({int(d) for d in tracks.date[held]} - read)


def _tallies_by_date(
    tracks: refs.ScheerTracks,
    rows: np.ndarray,
    clf: rd.GaussianClassifier,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    lawn: dict[int, list[np.ndarray]] = {}
    kin: dict[int, list[np.ndarray]] = {}
    for i in rows:
        x, y = tracks.x[i], tracks.y[i]
        on = g.real_on_lawn(x, y, tracks.boundary_radius[i])
        w = g.animal_windows(x, y, on, clf)
        lawn.setdefault(int(tracks.date[i]), []).append(g.tally(w))
        kin.setdefault(int(tracks.date[i]), []).append(g.kinematic_tally(w, x, y, on))
    stack = lambda d: {k: np.vstack(v) for k, v in sorted(d.items())}  # noqa: E731
    return stack(lawn), stack(kin)


def n2_control(bands: dict[str, Any]) -> dict[str, Any]:
    """Grade the 4 N2 units: two halves of the unread held-out dates, on two families each."""
    import benchmark_bands as bb

    tracks = refs.load_scheer_tracks()
    clf = bg.classifier(bands)
    bar = float(bands["bars"]["wild_type"])
    rows = tracks.arm(N2_ARM)
    band_lawn, band_kin = _tallies_by_date(tracks, rows[tracks.side[rows] == refs.BANDS], clf)
    unread = n2_unread_dates(tracks)
    held = rows[np.isin(tracks.date[rows], unread) & (tracks.side[rows] == refs.HELD_OUT)]
    held_lawn, held_kin = _tallies_by_date(tracks, held, clf)
    order = np.random.default_rng(N2_SPLIT_SEED).permutation(len(unread))
    halves = [
        sorted(unread[k] for k in order[: len(unread) // 2]),
        sorted(unread[k] for k in order[len(unread) // 2 :]),
    ]
    v1_kappa = json.loads(bb.RECORD_PATH.read_text())["calibration"]["kappa"]
    families = {
        "small_lawn": (band_lawn, held_lawn, g.LAWN_STATISTICS, g.lawn_statistics, v1_kappa, 0),
        "kinematics": (
            band_kin,
            held_kin,
            g.KINEMATIC_STATISTICS,
            g.kinematic_statistics,
            bands["kinematics_kappa"],
            1,
        ),
    }
    units = {}
    for h, dates in enumerate(halves):
        for family, (band, heldout, names, statistic, kappa, index) in families.items():
            tallies = np.vstack([heldout[d] for d in dates])
            n = len(tallies)
            nb = bb.lawn_bands(
                band,
                names,
                n,
                np.random.default_rng([N2_BAND_SEED, index, n]),
                kappa=kappa,
                statistic=statistic,
            )
            readings = g.grade(statistic(tallies.sum(axis=0)), nb, bar)
            units[f"{family}/half{h}"] = {
                "dates": dates,
                "n": n,
                "statistics": bc._reading_dict(readings),
                "passes": all(r.passes for r in readings.values()),
            }
    record = json.loads(BANDS_V2_RECORD.read_text())["checks"]
    rates = [1.0 - record[u.split("/")[0]]["joint_pass_rate"] for u in units]
    failed = [u for u, r in units.items() if not r["passes"]]
    return {
        "arm": N2_ARM,
        "band_dates": len(band_lawn),
        "unread_dates": len(unread),
        "split_seed": N2_SPLIT_SEED,
        "units": units,
        "failed": failed,
        "allowance": N2_ALLOWANCE,
        "p_more_than_allowance": bc.allowance_tail(rates, N2_ALLOWANCE),
        "passes": len(failed) <= N2_ALLOWANCE,
        "_halves": halves,
        "_held_lawn": held_lawn,
        "_held_kin": held_kin,
    }


# ---------------------------------------------------------------------------------------------
# The negatives and C.1's answer, through the measurement model
# ---------------------------------------------------------------------------------------------


def _negative_v2(
    name: str,
    bands: dict[str, Any],
    animals: list[int] = bc.READ_ANIMALS,
    networks: dict[str, list[int]] = bc.READ_NETWORKS,
) -> dict[str, Any]:
    """Run and grade one of B.4's negatives on v2: measured, with the chemotaxis assay."""
    from quantumnematode.evaluation.controllers import BatchedConnectome, BatchedMLP

    params = json.loads(bc.PARAMS_PATH.read_text())
    if not name.startswith("untrained_"):
        populations, chemotaxis = bg.simulate_v2(
            "mlp",
            lambda _path: bc.hazard_controller(name, params),
            animals,
            bands,
        )
    else:
        arm = "mlp" if name == "untrained_mlp" else "connectome"
        seeds = networks[name]
        per = len(animals) // len(seeds)
        populations, chemotaxis = {}, []
        for j, seed in enumerate(seeds):
            block = animals[j * per : (j + 1) * per]

            def build(path: Path, seed: int = seed) -> Any:  # noqa: ANN401 - a controller
                brain, _, _ = bc.harness.build_brain(path, seed)
                return BatchedMLP(brain) if arm == "mlp" else BatchedConnectome(brain)  # type: ignore[arg-type]

            pops, chem = bg.simulate_v2(arm, build, block, bands)
            for family, part in pops.items():
                populations.setdefault(family, []).extend(part)
            chemotaxis.extend(chem)
    graded = bg.grade_populations_v2(populations, chemotaxis, bands)
    graded["fails_wild_type"] = not graded["wild_type_passes"]
    graded["fails_chemotaxis"] = not graded["family_passes"]["chemotaxis"]
    return graded


def _mechanistic_grade(population: int, z: list[float], base: int) -> dict[str, Any]:
    """Grade a C.1 parameter point on one v2 population of 200 animals."""
    from quantumnematode.evaluation.controllers import BatchedMechanistic

    bands = bg.load_bands_v2()
    seeds = [base + ANIMALS * population + i for i in range(ANIMALS)]
    params = nr.parameters_from_unit(np.asarray(z))
    populations, chemotaxis = bg.simulate_v2(
        "mlp",
        lambda _path: BatchedMechanistic(params),
        seeds,
        bands,
    )
    return bg._finite(bg.grade_populations_v2(populations, chemotaxis, bands))


def _graded_answer(z: list[float], base: int, pool: Any) -> dict[str, Any]:  # noqa: ANN401 - a pool
    import c1_fit as c1

    grades = list(
        pool.map(
            _mechanistic_grade,
            range(GRADE_POPULATIONS),
            [z] * GRADE_POPULATIONS,
            [base] * GRADE_POPULATIONS,
        ),
    )
    passes = [c1.population_passes(gr["family_passes"]) for gr in grades]
    return {
        "seed_base": base,
        "population_passes": passes,
        "passes_wild_type": c1.wild_type_verdict(passes),
        "population_D": [gr["D"] for gr in grades],
        "family_passes": [gr["family_passes"] for gr in grades],
    }


# ---------------------------------------------------------------------------------------------
# W's minimum, on both family sets
# ---------------------------------------------------------------------------------------------


def _difference(
    a: dict[str, float],
    b: dict[str, float],
    scales: dict[str, float],
) -> dict[str, g.Reading]:
    out = {}
    for s, scale in scales.items():
        band = g.Band(0.0, scale)
        out[s] = g.Reading(a[s] - b[s], band, g.z_score(a[s] - b[s], band), passes=True)
    return out


def _mean_of(
    rows: list[dict[str, float]],
    picked: Any,  # noqa: ANN401 - indices
    names: tuple[str, ...],
) -> dict[str, float]:
    return {s: float(np.nanmean([rows[j][s] for j in picked])) for s in names}


def w_minimum_v2(bands: dict[str, Any], n2: dict[str, Any]) -> dict[str, Any]:
    """Return W's minimum on the fitted and on the held-out families: each the 95th percentile."""
    import itertools

    import benchmark_bands as bb

    family_map = bg._v2_family_map(bands)
    held_lawn, held_kin = n2["_held_lawn"], n2["_held_kin"]
    dates = sorted(held_lawn)
    builder = bg._builder()
    rng = np.random.default_rng(W_FITTED_SEED)
    fitted = []
    for _ in range(W_SPLITS):
        order = rng.permutation(len(dates))
        sides = [
            [dates[k] for k in order[: len(dates) // 2]],
            [dates[k] for k in order[len(dates) // 2 :]],
        ]
        lawn = [np.vstack([held_lawn[d] for d in side]) for side in sides]
        kin = [np.vstack([held_kin[d] for d in side]) for side in sides]
        size = max(W_SIZE_STEP, round(min(len(lawn[0]), len(lawn[1])) / W_SIZE_STEP) * W_SIZE_STEP)
        small, kinematic = builder.small_lawn(size), bg._kinematic_bands_at(size)
        readings = {
            "small_lawn": _difference(
                g.lawn_statistics(lawn[0].sum(axis=0)),
                g.lawn_statistics(lawn[1].sum(axis=0)),
                {s: small[s].scale for s in g.LAWN_STATISTICS},
            ),
            "kinematics": _difference(
                g.kinematic_statistics(kin[0].sum(axis=0)),
                g.kinematic_statistics(kin[1].sum(axis=0)),
                {s: kinematic[s].scale for s in g.KINEMATIC_STATISTICS},
            ),
        }
        fitted.append(g.distance(readings, ["small_lawn", "kinematics"], family_map))
    lopezcruz = refs.load_lopezcruz()
    batches = bb.lopezcruz_batches(lopezcruz)
    split = json.loads(bb.LOPEZCRUZ_SPLIT_PATH.read_text())["sides"]
    band_batches = [
        bb.search_statistics(batches[b]) for b, side in sorted(split.items()) if side == refs.BANDS
    ]
    search_sd = {
        s: g.scale_of(np.array([b[s] for b in band_batches], dtype=float))
        for s in g.LOCAL_SEARCH_STATISTICS
    }
    record = json.loads(BANDS_V2_RECORD.read_text())
    recordings = [
        r["statistics"]
        for r in record["chemotaxis_recordings"].values()
        if r["protocol"] == "standard"
    ]
    from quantumnematode.benchmark import chemotaxis as ch

    chem_sd = {
        s: g.scale_of(np.array([r[s] for r in recordings], dtype=float))
        for s in ch.CHEMOTAXIS_STATISTICS
    }
    k_chem = len(recordings) // 2
    chem_splits = []
    for picked in itertools.combinations(range(len(recordings)), k_chem):
        if 0 not in picked:
            continue  # each 3-against-3 split once
        rest = [j for j in range(len(recordings)) if j not in picked]
        chem_splits.append(
            _difference(
                _mean_of(recordings, picked, ch.CHEMOTAXIS_STATISTICS),
                _mean_of(recordings, rest, ch.CHEMOTAXIS_STATISTICS),
                {s: v / math.sqrt(k_chem) for s, v in chem_sd.items()},
            ),
        )
    rng = np.random.default_rng(W_SEARCH_SEED)
    k_search = len(band_batches) // 2
    held_out = []
    search_only = []
    for _ in range(W_SPLITS):
        order = rng.permutation(len(band_batches))
        search = _difference(
            _mean_of(band_batches, order[:k_search], g.LOCAL_SEARCH_STATISTICS),
            _mean_of(band_batches, order[k_search:], g.LOCAL_SEARCH_STATISTICS),
            {s: v / math.sqrt(k_search) for s, v in search_sd.items()},
        )
        search_only.append(g.distance({"local_search": search}, ["local_search"], family_map))
        held_out.extend(
            g.distance(
                {"local_search": search, "chemotaxis": chem},
                ["local_search", "chemotaxis"],
                family_map,
            )
            for chem in chem_splits
        )
    chem_only = [g.distance({"chemotaxis": c}, ["chemotaxis"], family_map) for c in chem_splits]
    return {
        "fitted": {
            "families": ["small_lawn", "kinematics"],
            "source": f"halves of N2's {len(dates)} unread held-out dates, on PD1074's v2 band scales",
            "splits": W_SPLITS,
            "seed": W_FITTED_SEED,
            "median": float(np.median(fitted)),
            "minimum_95th": float(np.percentile(fitted, W_QUANTILE)),
        },
        "held_out": {
            "families": ["local_search", "chemotaxis"],
            "source": (
                f"halves of López-Cruz's {len(band_batches)} band batches ({W_SPLITS} splits) crossed "
                f"with every 3-against-3 split of Chen's {len(recordings)} recordings ({len(chem_splits)})"
            ),
            "seed": W_SEARCH_SEED,
            "pairs": len(held_out),
            "median": float(np.median(held_out)),
            "minimum_95th": float(np.percentile(held_out, W_QUANTILE)),
            "reported_local_search_95th": float(np.percentile(search_only, W_QUANTILE)),
            "reported_chemotaxis_95th": float(np.percentile(chem_only, W_QUANTILE)),
        },
    }


def _read(args: argparse.Namespace) -> int:
    from concurrent.futures import ProcessPoolExecutor

    if not _committed(bands=True):
        print(
            f"{LAUNCH_PATH} with Part B, and the bands it binds, are not committed and clean; "
            "nothing held-out is read.",
            file=sys.stderr,
        )
        return 1
    if READ_PATH.exists():
        print(f"{READ_PATH} exists: the read is made once.", file=sys.stderr)
        return 1
    bands = bg.load_bands_v2()
    n2 = n2_control(bands)
    record: dict[str, Any] = {
        "description": "B v2's read: fresh N2, the negatives, C.1's answer, W.",
    }
    record["w_minimum"] = w_minimum_v2(bands, n2)
    record["n2_control"] = {k: v for k, v in n2.items() if not k.startswith("_")}
    print(
        "n2",
        json.dumps(bg._finite({k: record["n2_control"][k] for k in ("failed", "passes")})),
        flush=True,
    )
    record["negatives"] = {}
    for name in bc.NEGATIVES:
        graded = _negative_v2(name, bands)
        record["negatives"][name] = graded
        print(
            name,
            json.dumps(bg._finite({k: graded[k] for k in ("family_passes", "D", "D_fit")})),
            flush=True,
        )
    fit = json.loads(C1_FIT.read_text())
    z = fit["restarts"][fit["chosen_restart"]]["answer"]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=nr._worker_setup) as pool:
        record["c1_answer"] = {
            "restart": fit["chosen_restart"],
            **_graded_answer(z, C1_V2_GRADE_BASE, pool),
        }
    READ_PATH.write_text(json.dumps(bg._finite(record), indent=2, allow_nan=False) + "\n")
    return 0


# ---------------------------------------------------------------------------------------------
# The blind worm, fitted against v2
# ---------------------------------------------------------------------------------------------


def _blind_full(z: np.ndarray) -> list[float]:
    from quantumnematode.evaluation.controllers import MECHANISTIC_PARAMETERS

    free = [p for p in MECHANISTIC_PARAMETERS if p not in BLIND_PINNED]
    out = dict(BLIND_PINNED)
    out.update(zip(free, map(float, z), strict=True))
    return [out[p] for p in MECHANISTIC_PARAMETERS]


def _blind_simulate(k: int, z: list[float]) -> float:
    """Return v2's D for the blind worm at a free-parameter point on simulation k; the cap if none."""
    graded = _mechanistic_grade(k, _blind_full(np.asarray(z)), BLIND_V2_FIT_BASE)
    return float(g.Z_CAP) if graded["D"] is None else float(graded["D"])


def blind_k(restart: int, generation: int, member: int) -> int:
    """Return a blind-fit simulation's index: 10,000 per restart, 16 per generation."""
    return 10_000 * restart + BLIND_POPSIZE * (generation - 1) + member


def plateau_stop(medians: list[float], sigma: float) -> bool:
    """Return whether the plateau rule stops a blind fit: it holds and sigma is at most sigma0.

    A restart stuck at a wide sigma has not converged, however flat its medians.
    """
    import cma_recovery as cr

    return cr.plateau_reached(medians) and sigma <= BLIND_SIGMA0


def _cma_restart(  # noqa: PLR0913 - one restart's registered settings
    state_dir: Path,
    restart: int,
    cma_seed: int,
    dimension: int,
    simulate: Any,  # noqa: ANN401 - a picklable (k, z) -> objective
    cap: int,
    pool: Any,  # noqa: ANN401 - a pool
    tag: str,
) -> dict[str, Any]:
    """Run, or resume, one sep-CMA restart from the box centre; state saved every generation.

    Simulation indices are :func:`blind_k`'s. The plateau stops a restart only once sigma is at
    most sigma0; otherwise it runs to the cap.
    """
    import pickle

    import cma
    import cma_recovery as cr

    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / f"restart{restart}.pkl"
    history_path = state_dir / f"restart{restart}.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else None
    if history is not None and history["stop"] is not None:
        return history
    if history is not None and state_path.exists():
        checkpoint = pickle.loads(state_path.read_bytes())  # noqa: S301 - this run's own state
        if checkpoint["generation"] != history["generation"]:
            msg = f"{tag} restart {restart}: checkpoint and history disagree on the generation"
            raise RuntimeError(msg)
        es = checkpoint["strategy"]
    else:
        es = cma.CMAEvolutionStrategy(
            [0.5] * dimension,
            BLIND_SIGMA0,
            {
                "CMA_diagonal": True,
                "bounds": [0, 1],
                "popsize": BLIND_POPSIZE,
                "seed": cma_seed,
                "verbose": -9,
            },
        )
        history = {
            "restart": restart,
            "cma_seed": cma_seed,
            "generation": 0,
            "medians": [],
            "sigmas": [],
            "answers": [],
            "stop": None,
        }
    while history["stop"] is None:
        generation = history["generation"] + 1
        population = es.ask()
        ks = [blind_k(restart, generation, m) for m in range(BLIND_POPSIZE)]
        scores = list(pool.map(simulate, ks, [list(map(float, x)) for x in population]))
        es.tell(population, scores)
        history["medians"].append(float(np.median(scores)))
        history["sigmas"].append(float(es.sigma))
        history["answers"].append(np.clip(es.result.xfavorite, 0, 1).tolist())
        if plateau_stop(history["medians"], float(es.sigma)):
            history["stop"] = {"generation": generation, "by": "plateau"}
        elif generation >= cap:
            history["stop"] = {"generation": generation, "by": "cap"}
        history["generation"] = generation
        cr._write_bytes(state_path, pickle.dumps({"generation": generation, "strategy": es}))
        cr._write_json(history_path, history)
        print(
            f"{tag} restart {restart} gen {generation}: median {history['medians'][-1]:.3f}, "
            f"sigma {history['sigmas'][-1]:.3f}",
            flush=True,
        )
    return history


def _blind_restart(restart: int, pool: Any) -> dict[str, Any]:  # noqa: ANN401 - a pool
    return _cma_restart(
        BLIND_STATE,
        restart,
        BLIND_V2_CMA_SEEDS[restart],
        len(_blind_free()),
        _blind_simulate,
        BLIND_V2_CAP,
        pool,
        "blind",
    )


def _blind_free() -> list[str]:
    from quantumnematode.evaluation.controllers import MECHANISTIC_PARAMETERS

    return [p for p in MECHANISTIC_PARAMETERS if p not in BLIND_PINNED]


def _blind(args: argparse.Namespace) -> int:
    from concurrent.futures import ProcessPoolExecutor

    import cma_recovery as cr

    if not _committed(bands=True):
        print(
            f"{LAUNCH_PATH} with Part B is not committed and clean; the fit is not run.",
            file=sys.stderr,
        )
        return 1
    with ProcessPoolExecutor(max_workers=args.workers, initializer=nr._worker_setup) as pool:
        histories = [_blind_restart(r, pool) for r in range(len(BLIND_V2_CMA_SEEDS))]
        best = min(range(len(histories)), key=lambda r: cr.final_objective(histories[r]["medians"]))
        z = _blind_full(np.asarray(histories[best]["answers"][-1]))
        graded = _graded_answer(z, BLIND_V2_GRADE_BASE, pool)
    record = {
        "description": "The blind worm fitted against v2's D, graded on five populations.",
        "free": _blind_free(),
        "pinned": BLIND_PINNED,
        "restarts": [
            {k: h[k] for k in ("restart", "cma_seed", "stop", "medians", "sigmas")}
            for h in histories
        ],
        "best_restart": best,
        "answer": z,
        **graded,
    }
    BLIND_PATH.write_text(json.dumps(bg._finite(record), indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            bg._finite(
                {
                    k: record[k]
                    for k in (
                        "best_restart",
                        "population_passes",
                        "passes_wild_type",
                        "population_D",
                    )
                },
            ),
        ),
    )
    return 0


# ---------------------------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------------------------


def verdict(
    oracle: dict[str, Any],
    record: dict[str, Any],
    read: dict[str, Any],
    blind: dict[str, Any],
) -> dict[str, Any]:
    """Return B v2's validity from its registered readings: VALID only if every check holds."""
    negatives = read["negatives"]
    checks = {
        "round_trip": bool(oracle["round_trip"]["passes"]),
        "reachability": bool(
            isinstance(oracle["reachability"], dict) and oracle["reachability"]["passes"],
        ),
        "n2_control": bool(read["n2_control"]["passes"]),
        "chemotaxis_leave_one_out": bool(record["chemotaxis_leave_one_out"]["passes"]),
        "negatives_fail_wild_type": all(negatives[n]["fails_wild_type"] for n in bc.NEGATIVES),
        "open_loop_fail_chemotaxis": all(negatives[n]["fails_chemotaxis"] for n in OPEN_LOOP),
        "blind_worm_fails": not blind["passes_wild_type"],
        "c1_answer_fails": not read["c1_answer"]["passes_wild_type"],
    }
    return {"checks": checks, "verdict": "VALID" if all(checks.values()) else "VOID"}


def _verdict(_args: argparse.Namespace) -> int:
    parts = [
        json.loads(p.read_text())
        for p in (ORACLE_AMENDED2_PATH, BANDS_V2_RECORD, READ_PATH, BLIND_PATH)
    ]
    out = {
        "description": "B v2's verdict, from the oracle, the bands record, the read and the blind fit.",
        **verdict(*parts),
        "w_minimum": parts[2]["w_minimum"],
    }
    CONTROLS_PATH.write_text(json.dumps(bg._finite(out), indent=2, allow_nan=False) + "\n")
    print(json.dumps(out["checks"], indent=2), "\nverdict:", out["verdict"])
    return 0


# ---------------------------------------------------------------------------------------------
# The empirical noise library, its gate and the reachability worm
# ---------------------------------------------------------------------------------------------

NOISE_NAME = "scheer_tracker_noise.npz"
NOISE_PATH = refs.BENCHMARK_DATA / NOISE_NAME
NOISE_PILOT_PATH = SUPPORT / "noise_pilot.json"
REACH_PATH = SUPPORT / "reachability_worm.json"
ORACLE_AMENDED2_PATH = SUPPORT / "oracle_amendment2.json"
REACH_STATE = _REPO / "build" / "benchmark-v2" / "reach"
SPLIT_SEED = 2043
LIBRARY_DATES = 21
HALVINGS = 1000
HALVING_SEED = 2044
GATE_QUANTILES = (2.5, 97.5)
NOISE_TRACK_S = 2700
CHECK_BASE = 156_000_000
PILOT_BASE = 157_000_000
REACH_FIT_BASE = 158_000_000
REACH_CHECK_BASE = 158_900_000
REACH_CMA_SEEDS = (9411, 9412)
REACH_CAP = 100
REACH_BOX = 4.0
REACH_CHECKS = 5
REACH_BAR = 1.0
# The still-worm readings given permutation ranges, and those of them gated: the window speeds bias
# the graded dwelling speeds; the onset rate biases no graded statistic, so it is reported.
NOISE_READINGS = ("onsets_per_paused_min", "paused_window_speed_median", "paused_window_speed_p90")
GATED = ("paused_window_speed_median", "paused_window_speed_p90")
REACH_FREE = (
    ("dwelling", "speed_mm_per_s"),
    ("roaming", "speed_mm_per_s"),
    ("dwelling", "curvature_sd_deg_per_s"),
    ("roaming", "curvature_sd_deg_per_s"),
    ("dwelling", "reversal_hazard_per_s"),
    ("roaming", "reversal_hazard_per_s"),
    ("dwelling", "omega_hazard_per_s"),
    ("roaming", "omega_hazard_per_s"),
    ("dwelling", "speed_log_sd"),
    ("roaming", "speed_log_sd"),
    ("both", "reversal_speed_coupling"),
)
COUPLING_MAX = 3.0
# The within-state speed spread has no matched value to centre on: its box is linear from 0.
SPREAD_MAX = 1.5


def split_band_dates(tracks: refs.ScheerTracks) -> tuple[list[int], list[int]]:
    """Return PD1074's band dates split into the library half (21) and the check half (20)."""
    rows, _ = _band_rows_and_omega(tracks)
    dates = sorted({int(d) for d in tracks.date[rows]})
    order = np.random.default_rng(SPLIT_SEED).permutation(len(dates))
    library = sorted(dates[k] for k in order[:LIBRARY_DATES])
    check = sorted(dates[k] for k in order[LIBRARY_DATES:])
    return library, check


def _stretches_on(
    tracks: refs.ScheerTracks,
    dates: list[int],
) -> list[tuple[np.ndarray, np.ndarray]]:
    rows, _ = _band_rows_and_omega(tracks)
    return paused_stretches(tracks, rows[np.isin(tracks.date[rows], dates)])


def build_library(tracks: refs.ScheerTracks) -> dict[str, Any]:
    """Write the library half's paused stretches' offsets from their own means; return its facts."""
    library, check = split_band_dates(tracks)
    stretches = _stretches_on(tracks, library)
    x = np.concatenate([sx - sx.mean() for sx, _ in stretches])
    y = np.concatenate([sy - sy.mean() for _, sy in stretches])
    lengths = np.array([len(sx) for sx, _ in stretches])
    mm.save_noise_library(NOISE_PATH, x, y, lengths)
    return {
        "library_dates": library,
        "check_dates": check,
        "stretches": len(stretches),
        "seconds": int(lengths.sum()),
        "length_median_s": float(np.median(lengths)),
        "sha256": mm.file_sha256(NOISE_PATH),
    }


def model_stretches(
    real: list[tuple[np.ndarray, np.ndarray]],
    library: mm.NoiseLibrary,
    base: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return each real stretch's model counterpart: a still point's noise, cut from a full track.

    Stretch k's track is the small-lawn assay's length, drawn at ``default_rng([base + k, 2033])``,
    and its counterpart is cut at a uniform point from the same stream, so stretches are joined at
    the rate a simulated track sees.
    """
    out = []
    for k, (x, _) in enumerate(real):
        rng = np.random.default_rng([base + k, mm.LAWN_STREAM])
        track = library.draw(NOISE_TRACK_S, rng)
        start = int(rng.integers(0, NOISE_TRACK_S - len(x) + 1))
        out.append((track[start : start + len(x), 0], track[start : start + len(x), 1]))
    return out


def per_date_noise(tracks: refs.ScheerTracks) -> dict[int, dict[str, Any]]:
    """Return each band date's still-worm readings' parts: onsets, seconds, window speeds, steps."""
    rows, _ = _band_rows_and_omega(tracks)
    out = {}
    for d in sorted({int(v) for v in tracks.date[rows]}):
        stretches = paused_stretches(tracks, rows[tracks.date[rows] == d])
        onsets, seconds, speeds = 0, 0, []
        for x, y in stretches:
            onsets += len(
                ro.reversal_onsets(x, y, g.OFF_FOOD_DETECTOR, floor=g.ON_LAWN_REVERSAL_FLOOR_MM),
            )
            seconds += len(x) - 1
            w = g.window_inputs(x, y, np.ones(len(x), dtype=bool))
            speeds.append(w.speed[np.isfinite(w.speed)])
        out[d] = {"onsets": onsets, "seconds": seconds, "speeds": np.concatenate(speeds)}
    return out


def _pooled(parts: list[dict[str, Any]]) -> dict[str, float]:
    speed = np.concatenate([p["speeds"] for p in parts])
    return {
        "onsets_per_paused_min": 60.0
        * sum(p["onsets"] for p in parts)
        / sum(p["seconds"] for p in parts),
        "paused_window_speed_median": float(np.median(speed)),
        "paused_window_speed_p90": float(np.quantile(speed, 0.9)),
    }


def permutation_gate(per_date: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Return each gated reading's central 95% of real-half differences over random date halvings."""
    dates = sorted(per_date)
    rng = np.random.default_rng(HALVING_SEED)
    diffs: dict[str, list[float]] = {k: [] for k in NOISE_READINGS}
    for _ in range(HALVINGS):
        order = rng.permutation(len(dates))
        a = _pooled([per_date[dates[k]] for k in order[:LIBRARY_DATES]])
        b = _pooled([per_date[dates[k]] for k in order[LIBRARY_DATES:]])
        for k in NOISE_READINGS:
            diffs[k].append(a[k] - b[k])
    return {
        "halvings": HALVINGS,
        "seed": HALVING_SEED,
        "ranges": {
            k: [float(np.percentile(v, q)) for q in GATE_QUANTILES] for k, v in diffs.items()
        },
        "sd": {k: float(np.std(v, ddof=1)) for k, v in diffs.items()},
    }


def gated_check(
    real: list[tuple[np.ndarray, np.ndarray]],
    model: list[tuple[np.ndarray, np.ndarray]],
    gate: dict[str, Any],
) -> dict[str, Any]:
    """Return the model's still-worm readings against the real ones, each gated by its range."""
    a, b = noise_statistics(real), noise_statistics(model)
    rows: dict[str, Any] = {}
    for name, value in a.items():
        row: dict[str, Any] = {"real": value, "model": b[name], "difference": b[name] - value}
        if name in NOISE_READINGS:
            low, high = gate["ranges"][name]
            row["range"] = [low, high]
            if name in GATED:
                row["passes"] = bool(low <= b[name] - value <= high)
        rows[name] = row
    return {
        "stretches": len(real),
        "readings": rows,
        "passes": all(r["passes"] for r in rows.values() if "passes" in r),
    }


def reach_parameters(z: np.ndarray) -> dict[str, dict[str, float]]:
    """Return the reachability worm's per-state parameters.

    Speeds, curvature SDs and hazards are log-uniform on [θ/4, 4θ] around B.4's matched values;
    each state's speed spread (a log SD, per 10-s window) is uniform on [0, :data:`SPREAD_MAX`],
    and the reversal hazard's coupling to the window's speed draw on [0, :data:`COUPLING_MAX`].
    """
    states = json.loads(bc.PARAMS_PATH.read_text())["states"]
    out = {s: dict(v) for s, v in states.items()}
    for (state, name), u in zip(REACH_FREE, np.asarray(z, dtype=float), strict=True):
        if name == "speed_log_sd":
            out[state][name] = float(SPREAD_MAX * u)
        elif name == "reversal_speed_coupling":
            out[state] = {name: float(COUPLING_MAX * u)}
        else:
            out[state][name] = float(states[state][name] * REACH_BOX ** (2.0 * u - 1.0))
    return out


def reach_controller(states: dict[str, dict[str, float]]) -> Any:  # noqa: ANN401 - a controller
    """Return the two-state open-loop worm with B.4's transitions and these per-state parameters."""
    from quantumnematode.env.primitives import load_kinematics
    from quantumnematode.evaluation.controllers import BatchedStateHazards

    params = json.loads(bc.PARAMS_PATH.read_text())
    minimum = load_kinematics().hazard_min_per_s
    order = [states["dwelling"], states["roaming"]]
    return BatchedStateHazards(
        np.asarray(params["hmm"]["transitions_per_10_s"]),
        tuple(s["speed_mm_per_s"] for s in order),  # type: ignore[arg-type]
        tuple(s["curvature_sd_deg_per_s"] for s in order),  # type: ignore[arg-type]
        np.array([[s["reversal_hazard_per_s"], s["omega_hazard_per_s"], minimum] for s in order]),
        window_steps=round(params["hmm"]["window_seconds"]),
        speed_log_sd=tuple(s.get("speed_log_sd", 0.0) for s in order),  # type: ignore[arg-type]
        reversal_speed_coupling=states.get("both", {}).get("reversal_speed_coupling"),
    )


def _measured_small_lawn(controller: Any, seeds: list[int]) -> dict[str, float]:  # noqa: ANN401
    from quantumnematode.evaluation.batched import BatchedEvaluator

    bands = bg.load_bands_v2()
    model = mm.TrackerModel.from_dict(bands["measurement"]["lawn"])
    traj = BatchedEvaluator(nr._shared()["configs"]["small_lawn"], controller, seeds).run()
    assert traj.assay is not None  # noqa: S101 - an assay config records its timing
    analysis = tuple(traj.assay["analysis_steps"])  # type: ignore[arg-type]
    animals = [
        bg.window("small_lawn", mm.measure(traj.to_behaviour_steps(i), seed, model), analysis)  # type: ignore[arg-type]
        for i, seed in enumerate(seeds)
    ]
    return bg.kinematic_values(animals, bg.classifier(bands))


def reach_rms(values: dict[str, float]) -> float:
    """Return the RMS z of the five gated kinematics statistics at n = 200."""
    band = bg.family_bands_v2(bg.load_bands_v2(), "kinematics", ANIMALS)
    z = [g.z_score(values[s], band[s]) for s in ORACLE_STATISTICS]
    return float(np.sqrt(np.mean(np.square(z))))


def _reach_simulate(k: int, z: list[float], base: int = REACH_FIT_BASE) -> float:
    seeds = [base + ANIMALS * k + i for i in range(ANIMALS)]
    return reach_rms(_measured_small_lawn(reach_controller(reach_parameters(np.asarray(z))), seeds))


FLOOR_GRID_MM = tuple(round(0.01 * k, 2) for k in range(1, 11))
FLOOR_SHARE = 0.01


def floor_rule(tracks: refs.ScheerTracks, dates: list[int]) -> dict[str, Any]:
    """Return the on-lawn reversal floor's rule on these dates, and the floor it chooses.

    It is the smallest floor on the grid at which the real tracker's still-worm onset rate is
    under :data:`FLOOR_SHARE` of the real on-lawn rate, both read at that floor.
    """
    rows, _ = _band_rows_and_omega(tracks)
    rows = rows[np.isin(tracks.date[rows], dates)]
    stills = paused_stretches(tracks, rows)
    grid = []
    for floor in FLOOR_GRID_MM:
        onsets = sum(
            len(ro.reversal_onsets(x, y, g.OFF_FOOD_DETECTOR, floor=floor)) for x, y in stills
        )
        still = 60.0 * onsets / sum(len(x) - 1 for x, _ in stills)
        n_on = n_s = 0
        for i in rows:
            x, y = tracks.x[i], tracks.y[i]
            on = g.real_on_lawn(x, y, tracks.boundary_radius[i])
            kept = ro.reversal_onsets(x, y, g.OFF_FOOD_DETECTOR, floor=floor)
            n_on += sum(1 for o in kept if int(o) < len(on) and on[int(o)])
            n_s += int((np.isfinite(np.diff(x)) & np.isfinite(np.diff(y)) & on[1:] & on[:-1]).sum())
        lawn = 60.0 * n_on / n_s
        grid.append({"floor_mm": floor, "still_per_min": still, "on_lawn_per_min": lawn})
    chosen = next(
        (r["floor_mm"] for r in grid if r["still_per_min"] < FLOOR_SHARE * r["on_lawn_per_min"]),
        None,
    )
    return {"share": FLOOR_SHARE, "grid": grid, "chosen_mm": chosen}


def _noise_library_stage(_args: argparse.Namespace) -> int:
    if not _committed(bands=False):
        print(f"{LAUNCH_PATH} is not committed and clean; nothing is built.", file=sys.stderr)
        return 1
    facts = build_library(refs.load_scheer_tracks())
    print(json.dumps(facts, indent=2))
    return 0


def _noise_pilot(args: argparse.Namespace) -> int:
    """Run the pilot: the draw in-sample, the permutation gate, the reachability worm's fit."""
    from concurrent.futures import ProcessPoolExecutor

    tracks = refs.load_scheer_tracks()
    library_dates, _ = split_band_dates(tracks)
    library = mm.load_noise_library(NOISE_NAME)
    per_date = per_date_noise(tracks)
    gate = permutation_gate(per_date)
    real = _stretches_on(tracks, library_dates)
    floor = floor_rule(tracks, library_dates)
    print(json.dumps(floor, indent=2), flush=True)
    if floor["chosen_mm"] != g.ON_LAWN_REVERSAL_FLOOR_MM:
        msg = f"the floor rule chooses {floor['chosen_mm']}, not {g.ON_LAWN_REVERSAL_FLOOR_MM}"
        raise RuntimeError(msg)
    record: dict[str, Any] = {
        "description": "The empirical-noise pilot, on the library half and discarded seeds.",
        "library_sha256": mm.file_sha256(NOISE_PATH),
        "reversal_floor": floor,
        "gate": gate,
        "gaussian_misses_at_the_0.01_floor": {
            k: v["model"] - v["real"]
            for k, v in json.loads(ORACLE_AMENDED_PATH.read_text())["round_trip"]["lawn_jitter"][
                "readings"
            ].items()
            if k in GATED
        },
        "in_sample": gated_check(real, model_stretches(real, library, PILOT_BASE), gate),
    }
    print(
        json.dumps(
            bg._finite(
                {k: record[k] for k in ("gate", "gaussian_misses_at_the_0.01_floor", "in_sample")},
            ),
            indent=2,
        ),
        flush=True,
    )
    if not record["in_sample"]["passes"]:
        record["stop"] = "the draw fails in-sample: the joins distort the noise"
        NOISE_PILOT_PATH.write_text(json.dumps(bg._finite(record), indent=2) + "\n")
        return 1
    with ProcessPoolExecutor(max_workers=args.workers, initializer=nr._worker_setup) as pool:
        histories = [
            _cma_restart(
                REACH_STATE,
                r,
                REACH_CMA_SEEDS[r],
                len(REACH_FREE),
                _reach_simulate,
                REACH_CAP,
                pool,
                "reach",
            )
            for r in range(len(REACH_CMA_SEEDS))
        ]
        import cma_recovery as cr

        best = min(range(len(histories)), key=lambda r: cr.final_objective(histories[r]["medians"]))
        z = histories[best]["answers"][-1]
        checks = list(
            pool.map(
                _reach_simulate,
                range(REACH_CHECKS),
                [z] * REACH_CHECKS,
                [REACH_CHECK_BASE] * REACH_CHECKS,
            ),
        )
    worm = {
        "description": "The reachability worm: B.4's two-state structure, fitted through the "
        "measurement model on discarded seeds.",
        "free": [f"{s}/{n}" for s, n in REACH_FREE],
        "box": f"log-uniform on [θ/{REACH_BOX:g}, {REACH_BOX:g}θ] around B.4's matched values; "
        f"each state's speed spread uniform on [0, {SPREAD_MAX:g}]; the reversal coupling on "
        f"[0, {COUPLING_MAX:g}]",
        "restarts": [
            {k: h[k] for k in ("restart", "cma_seed", "stop", "medians", "sigmas")}
            for h in histories
        ],
        "best_restart": best,
        "answer": z,
        "states": reach_parameters(np.asarray(z)),
        "check_rms_z": checks,
        "check_median_rms_z": float(np.median(checks)),
        "reaches": bool(np.median(checks) <= REACH_BAR),
    }
    REACH_PATH.write_text(json.dumps(bg._finite(worm), indent=2) + "\n")
    record["reachability_worm"] = {
        k: worm[k] for k in ("best_restart", "check_rms_z", "check_median_rms_z", "reaches")
    }
    NOISE_PILOT_PATH.write_text(json.dumps(bg._finite(record), indent=2) + "\n")
    print(json.dumps(bg._finite(record["reachability_worm"]), indent=2))
    return 0 if worm["reaches"] else 1


def _oracle2(args: argparse.Namespace) -> int:
    """Run the oracles as amended twice: the losses, the jitter on the check half, Chen, reachability."""
    import cma_recovery as cr

    bound = (BANDS_V2_PATH, BANDS_V2_RECORD, NOISE_PATH, REACH_PATH, NOISE_PILOT_PATH)
    if not cr.launch_committed(bound=bound, launch_path=LAUNCH_PATH):
        print(
            f"{LAUNCH_PATH} with amendment 2 and the files it binds is not committed and clean.",
            file=sys.stderr,
        )
        return 1
    if ORACLE_AMENDED2_PATH.exists():
        print(f"{ORACLE_AMENDED2_PATH} exists: the oracle is read once.", file=sys.stderr)
        return 1
    bands = bg.load_bands_v2()
    tracks = refs.load_scheer_tracks()
    _, check_dates = split_band_dates(tracks)
    gate = json.loads(NOISE_PILOT_PATH.read_text())["gate"]
    real = _stretches_on(tracks, check_dates)
    library = mm.load_noise_library(NOISE_NAME)
    rigs = {
        "lawn_losses": loss_round_trip(bands),
        "lawn_jitter": {
            "check_dates": check_dates,
            **gated_check(real, model_stretches(real, library, CHECK_BASE), gate),
        },
        "chemotaxis": chen_round_trip(bands, args.chen_archive),
    }
    record: dict[str, Any] = {
        "description": "B v2's oracles as amended twice: the losses, the jitter on the check half, "
        "Chen's round trip, then reachability with the fitted worm.",
        "round_trip": {**rigs, "passes": all(r["passes"] for r in rigs.values())},
    }
    if record["round_trip"]["passes"]:
        record["reachability"] = reachability_oracle(bands, args.workers)
        record["passes"] = bool(record["reachability"]["passes"])
    else:
        record["reachability"] = "not run: a measurement check failed"
        record["passes"] = False
    ORACLE_AMENDED2_PATH.write_text(
        json.dumps(bg._finite(record), indent=2, allow_nan=False) + "\n",
    )
    print(json.dumps(bg._finite(record), indent=2)[:6000])
    return 0 if record["passes"] else 1


def main(argv: list[str] | None = None) -> int:
    """CLI: the probes, the oracles, the read, the blind fit and the verdict."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    sub = parser.add_subparsers(dest="stage", required=True)
    probe = sub.add_parser("probe")
    probe.add_argument("--chen-archive", type=Path, required=True)
    probe.add_argument("--blind", action="store_true")
    probe.add_argument("--out", type=Path, default=PROBES_PATH)
    oracle = sub.add_parser("oracle")
    oracle.add_argument("--chen-archive", type=Path, required=True)
    oracle.add_argument("--workers", type=int, default=16)
    oracle.add_argument("--bands", type=Path, default=bg.BANDS_V2_PATH)
    for name in ("read", "blind"):
        sub.add_parser(name).add_argument("--workers", type=int, default=16)
    sub.add_parser("verdict")
    sub.add_parser("noise-library")
    sub.add_parser("noise-pilot").add_argument("--workers", type=int, default=16)
    oracle2 = sub.add_parser("oracle2")
    oracle2.add_argument("--chen-archive", type=Path, required=True)
    oracle2.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    stages = {
        "probe": _probe,
        "oracle": _oracle,
        "read": _read,
        "blind": _blind,
        "verdict": _verdict,
        "noise-library": _noise_library_stage,
        "noise-pilot": _noise_pilot,
        "oracle2": _oracle2,
    }
    return stages[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())
