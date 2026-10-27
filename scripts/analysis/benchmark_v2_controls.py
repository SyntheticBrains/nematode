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

Usage::

    uv run python scripts/analysis/benchmark_v2_controls.py probe \
        --chen-archive <Chen_learning_2024.zip> [--blind] [--out <probes.json>]
"""

from __future__ import annotations

import argparse
import io
import json
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
    """Return detected reversal onsets on the lawn, and the valid on-lawn seconds."""
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


def main(argv: list[str] | None = None) -> int:
    """CLI: the probes; the oracles and the controls follow in later stages."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    sub = parser.add_subparsers(dest="stage", required=True)
    probe = sub.add_parser("probe")
    probe.add_argument("--chen-archive", type=Path, required=True)
    probe.add_argument("--blind", action="store_true")
    probe.add_argument("--out", type=Path, default=PROBES_PATH)
    args = parser.parse_args(argv)
    return _probe(args)


if __name__ == "__main__":
    sys.exit(main())
