#!/usr/bin/env python
r"""The benchmark's controls: held-out real worms must pass, and negatives must fail.

``pilot``
    Run every negative at n = 64 on pilot seeds, graded on the vendored bands. No held-out value is
    read. The result goes into the launch record as description.

``read``
    The registered reading. It refuses to run unless the launch record
    (``docs/experiments/logbooks/supporting/110-benchmark-controls/launch.md``) is committed and
    unchanged. Then it:

    - grades the 15 positive-control units on held-out data, each at its own size:
      - the OD 2 wild type on the small lawn and on the uniform well;
      - each held-out López-Cruz batch, on its statistics and on presence;
      - each mutant's held-out Δ against its same-date wild type;
    - runs the six negatives at n = 200 on animal seeds 5101-5300;
    - reads W's minimum;
    - writes the verdict: VALID when at most 2 units fail and every negative fails the wild type,
      VOID otherwise.

**The negatives**, with parameters from ``negatives_params.json`` (band cohorts only):

- the open-loop HMM worm (``BatchedStateHazards``), with the authors' transitions every 10 s;
- the Poisson worm (``BatchedCurvatureWalk``, AR(1) curvature, pooled hazards);
- the random walk (uniform curvature, minimum hazards) and the correlated walk (AR(1) curvature,
  minimum hazards), both at the pooled speed;
- the untrained MLP and connectome, 8 networks x 25 animals each (8 x 8 in the pilot).

Usage::

    uv run python scripts/analysis/benchmark_controls.py pilot --out <pilot.json>
    uv run python scripts/analysis/benchmark_controls.py read --out <controls.json>
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import quantumnematode.brain  # noqa: F401  - registers the brains the configs name
from quantumnematode.benchmark import grading as g
from quantumnematode.benchmark import references as refs
from quantumnematode.env.primitives import load_kinematics
from quantumnematode.evaluation.batched import BatchedEvaluator
from quantumnematode.evaluation.controllers import (
    BatchedConnectome,
    BatchedCurvatureWalk,
    BatchedMLP,
    BatchedStateHazards,
)
from quantumnematode.utils.config_loader import load_simulation_config

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_bands as bb
import benchmark_grade as bg
import body_kinematics_eval as harness

DESCRIPTION = "The benchmark's controls: held-out real worms must pass, and negatives must fail."
_REPO = Path(__file__).resolve().parents[2]
SUPPORT = _REPO / "docs" / "experiments" / "logbooks" / "supporting" / "110-benchmark-controls"
LAUNCH_PATH = SUPPORT / "launch.md"
PARAMS_PATH = SUPPORT / "negatives_params.json"

ALLOWANCE = 2
UNIT_FAILURE_RATES = {
    "small_lawn": 0.075,
    "uniform_lawn": 0.05,
    "local_search": 0.038,
    "mutant": 0.05,
}
NEGATIVES = (
    "hmm",
    "poisson",
    "random_walk",
    "correlated_walk",
    "untrained_mlp",
    "untrained_connectome",
)
READ_ANIMALS = list(range(5101, 5301))
READ_NETWORKS = {
    "untrained_mlp": list(range(5301, 5309)),
    "untrained_connectome": list(range(5309, 5317)),
}
PILOT_ANIMALS = list(range(9201, 9265))
PILOT_NETWORKS = {
    "untrained_mlp": list(range(9265, 9273)),
    "untrained_connectome": list(range(9273, 9281)),
}
W_SPLITS = 1000
W_SEED = 2037
W_SIZE_STEP = 25


# ---------------------------------------------------------------------------------------------
# The negatives
# ---------------------------------------------------------------------------------------------


def hazard_controller(name: str, params: dict[str, Any]) -> Any:  # noqa: ANN401 - a controller
    """Return one non-learning negative's batched controller from the matched parameters."""
    minimum = load_kinematics().hazard_min_per_s
    pooled = params["pooled"]
    if name == "hmm":
        states = [params["states"]["dwelling"], params["states"]["roaming"]]
        return BatchedStateHazards(
            np.asarray(params["hmm"]["transitions_per_10_s"]),
            tuple(s["speed_mm_per_s"] for s in states),  # type: ignore[arg-type]
            tuple(s["curvature_sd_deg_per_s"] for s in states),  # type: ignore[arg-type]
            np.array(
                [[s["reversal_hazard_per_s"], s["omega_hazard_per_s"], minimum] for s in states],
            ),
            window_steps=round(params["hmm"]["window_seconds"]),
        )
    if name == "poisson":
        return BatchedCurvatureWalk(
            pooled["speed_mm_per_s"],
            (pooled["reversal_hazard_per_s"], pooled["omega_hazard_per_s"], minimum),
            "ar1",
            pooled["curvature_sd_deg_per_s"],
            pooled["heading_persistence"],
        )
    if name == "random_walk":
        return BatchedCurvatureWalk(
            pooled["speed_mm_per_s"],
            (minimum, minimum, minimum),
            "uniform",
        )
    if name == "correlated_walk":
        return BatchedCurvatureWalk(
            pooled["speed_mm_per_s"],
            (minimum, minimum, minimum),
            "ar1",
            pooled["curvature_sd_deg_per_s"],
            pooled["heading_persistence"],
        )
    msg = f"unknown hazard negative {name!r}"
    raise ValueError(msg)


def _assay_path(arm: str, family: str) -> Path:
    return bg.FORAGING / f"{bg.ARMS[arm]}_assay_{bg.PROTOCOLS[family]}.yml"


def _window(family: str, trajectories: Any, animals: int) -> list[Any]:  # noqa: ANN401
    analysis = tuple(trajectories.assay["analysis_steps"])
    return [bg.window(family, trajectories.to_behaviour_steps(i), analysis) for i in range(animals)]


def negative_populations(
    name: str,
    params: dict[str, Any],
    animals: list[int],
    networks: dict[str, list[int]],
) -> dict[str, list[Any]]:
    """Run one negative on the three wild-type assays; return each family's windowed animals."""
    populations: dict[str, list[Any]] = {}
    for family in bg.PROTOCOLS:
        if name.startswith("untrained_"):
            arm = "mlp" if name == "untrained_mlp" else "connectome"
            path = _assay_path(arm, family)
            config = load_simulation_config(str(path))
            seeds = networks[name]
            per = len(animals) // len(seeds)
            population: list[Any] = []
            for j, seed in enumerate(seeds):
                brain, _, _ = harness.build_brain(path, seed)
                controller = (
                    BatchedMLP(brain)  # type: ignore[arg-type]
                    if arm == "mlp"
                    else BatchedConnectome(brain)  # type: ignore[arg-type]
                )
                block = animals[j * per : (j + 1) * per]
                population += _window(
                    family,
                    BatchedEvaluator(config, controller, block).run(),
                    len(block),
                )
            populations[family] = population
        else:
            config = load_simulation_config(str(_assay_path("mlp", family)))
            trajectories = BatchedEvaluator(config, hazard_controller(name, params), animals).run()
            populations[family] = _window(family, trajectories, len(animals))
    return populations


def grade_negatives(
    animals: list[int],
    networks: dict[str, list[int]],
) -> dict[str, Any]:
    """Run and grade every negative; return each one's grading and its run time."""
    params = json.loads(PARAMS_PATH.read_text())
    bands = bg.load_bands()
    out: dict[str, Any] = {}
    for name in NEGATIVES:
        start = time.perf_counter()
        populations = negative_populations(name, params, animals, networks)
        graded = bg.grade_populations(populations, bands)
        graded["fails_wild_type"] = not graded["wild_type_passes"]
        graded["seconds"] = time.perf_counter() - start
        out[name] = graded
        print(name, json.dumps({k: graded[k] for k in ("family_passes", "D", "D_fit")}), flush=True)
    return out


# ---------------------------------------------------------------------------------------------
# The positive controls (held-out data, read only in ``read``)
# ---------------------------------------------------------------------------------------------


def held_out_rows(tracks: refs.ScheerTracks, arm: str) -> np.ndarray:
    """Return an arm's held-out rows."""
    rows = tracks.arm(arm)
    return rows[tracks.side[rows] == refs.HELD_OUT]


def held_out_tallies(
    tracks: refs.ScheerTracks,
    arm: str,
    classifier: Any,  # noqa: ANN401 - the benchmark's classifier
    *,
    uniform: bool = False,
) -> dict[int, np.ndarray]:
    """Return each held-out date's animals' tallies."""
    out: dict[int, list[np.ndarray]] = {}
    for i in held_out_rows(tracks, arm):
        lawn = g.real_on_lawn(tracks.x[i], tracks.y[i], tracks.boundary_radius[i], uniform=uniform)
        windows = g.animal_windows(tracks.x[i], tracks.y[i], lawn, classifier)
        out.setdefault(int(tracks.date[i]), []).append(g.tally(windows))
    return {d: np.vstack(v) for d, v in sorted(out.items())}


def _reading_dict(readings: dict[str, g.Reading]) -> dict[str, Any]:
    return {
        name: {
            "value": r.value,
            "centre": r.band.centre,
            "scale": r.band.scale,
            "z": r.z,
            "passes": r.passes,
        }
        for name, r in readings.items()
    }


def positive_controls(builder: bb.BandBuilder, bands: dict[str, Any]) -> dict[str, Any]:
    """Grade the 15 held-out units."""
    tracks = builder.tracks
    clf = builder._classifier
    bar = float(bands["bars"]["wild_type"])
    units: dict[str, Any] = {}

    wild = held_out_tallies(tracks, bb.WILD_TYPE, clf)
    n_wild = int(sum(len(v) for v in wild.values()))
    values = g.lawn_statistics(np.vstack(list(wild.values())).sum(axis=0))
    readings = g.grade(values, builder.small_lawn(n_wild), bar)
    units["small_lawn"] = {"n": n_wild, "dates": len(wild), "statistics": _reading_dict(readings)}
    units["small_lawn"]["passes"] = all(r.passes for r in readings.values())

    uniform = held_out_tallies(tracks, bb.UNIFORM, clf, uniform=True)
    n_uniform = int(sum(len(v) for v in uniform.values()))
    values = g.lawn_statistics(np.vstack(list(uniform.values())).sum(axis=0))
    readings = g.grade(values, builder.uniform_lawn(n_uniform), bar)
    units["uniform_lawn"] = {
        "n": n_uniform,
        "dates": len(uniform),
        "statistics": _reading_dict(readings),
    }
    units["uniform_lawn"]["passes"] = all(r.passes for r in readings.values())

    lopezcruz = refs.load_lopezcruz()
    batches = bb.lopezcruz_batches(lopezcruz)
    sides = json.loads(bb.LOPEZCRUZ_SPLIT_PATH.read_text())["sides"]
    search_bands = bg.family_bands(bands, "local_search", 0)
    for batch in sorted(b for b, side in sides.items() if side == refs.HELD_OUT):
        row = batches[batch]
        readings = g.grade(bb.search_statistics(row), search_bands, bar)
        presence = bb.batch_presence(row)
        units[f"local_search/{batch}"] = {
            "n": row["n"],
            "statistics": _reading_dict(readings),
            "presence": presence,
            "passes": all(r.passes for r in readings.values()) and bool(presence["present"]),
        }

    for name, (arm, control) in bb.MUTANTS.items():
        mutant = held_out_tallies(tracks, arm, clf)
        wild_control = held_out_tallies(tracks, control, clf)
        dates = sorted(set(mutant) & set(wild_control))
        m = g.lawn_statistics(np.vstack([mutant[d] for d in dates]).sum(axis=0))
        w = g.lawn_statistics(np.vstack([wild_control[d] for d in dates]).sum(axis=0))
        n_mutant = int(sum(len(mutant[d]) for d in dates))
        delta = {s: m[s] - w[s] for s in g.MUTANT_LAWN_STATISTICS}
        mutant_bands, _ = builder.mutant(name, n_mutant)
        readings = g.grade(delta, mutant_bands, float(bands["bars"][name]), directions=True)
        units[f"mutant/{name}"] = {
            "n": n_mutant,
            "paired_dates": dates,
            "unpaired_held_out_dates": sorted(set(mutant) - set(dates)),
            "statistics": {
                **_reading_dict(readings),
                **{
                    s: {**_reading_dict(readings)[s], "direction": mutant_bands[s].direction}
                    for s in readings
                },
            },
            "passes": all(r.passes for r in readings.values()),
        }
    return units


def allowance_tail(rates: list[float], allowance: int) -> float:
    """Return P(more than ``allowance`` failures) under independent units with these rates."""
    dist = np.zeros(len(rates) + 1)
    dist[0] = 1.0
    for q in rates:
        nxt = dist * (1 - q)
        nxt[1:] += dist[:-1] * q
        dist = nxt
    return float(1.0 - np.cumsum(dist)[allowance])


# ---------------------------------------------------------------------------------------------
# W's minimum
# ---------------------------------------------------------------------------------------------


def w_minimum(builder: bb.BandBuilder) -> dict[str, Any]:
    """Read the distance two halves of the held-out wild type show by chance; its 95th percentile."""
    clf = builder._classifier
    wild = held_out_tallies(builder.tracks, bb.WILD_TYPE, clf)
    dates = list(wild)
    record = json.loads(bb.RECORD_PATH.read_text())
    batch_stats = list(record["lopezcruz_band_batches"].values())
    batch_sd = {
        s: g.scale_of(np.array([b[s] for b in batch_stats], dtype=float))
        for s in g.LOCAL_SEARCH_STATISTICS
    }
    lopezcruz = refs.load_lopezcruz()
    batches = bb.lopezcruz_batches(lopezcruz)
    sides = json.loads(bb.LOPEZCRUZ_SPLIT_PATH.read_text())["sides"]
    held = [
        bb.search_statistics(batches[b])
        for b, side in sorted(sides.items())
        if side == refs.HELD_OUT
    ]
    families = ["small_lawn", "local_search"]
    rng = np.random.default_rng(W_SEED)
    distances = []
    for _ in range(W_SPLITS):
        order = rng.permutation(len(dates))
        half = len(dates) // 2
        a = np.vstack([wild[dates[k]] for k in order[:half]])
        b = np.vstack([wild[dates[k]] for k in order[half:]])
        size = max(W_SIZE_STEP, round(min(len(a), len(b)) / W_SIZE_STEP) * W_SIZE_STEP)
        scales = builder.small_lawn(size)
        sa, sb = g.lawn_statistics(a.sum(axis=0)), g.lawn_statistics(b.sum(axis=0))
        readings: dict[str, dict[str, g.Reading]] = {"small_lawn": {}, "local_search": {}}
        for s in g.LAWN_STATISTICS:
            z = g.z_score(sa[s] - sb[s], g.Band(0.0, scales[s].scale))
            readings["small_lawn"][s] = g.Reading(sa[s] - sb[s], scales[s], z, passes=True)
        order_lc = rng.permutation(len(held))
        k = len(held) // 2
        for s in g.LOCAL_SEARCH_STATISTICS:
            ma = float(np.nanmean([held[j][s] for j in order_lc[:k]]))
            mb = float(np.nanmean([held[j][s] for j in order_lc[k:]]))
            scale = batch_sd[s] / math.sqrt(k)
            z = g.z_score(ma - mb, g.Band(0.0, scale))
            readings["local_search"][s] = g.Reading(ma - mb, g.Band(0.0, scale), z, passes=True)
        distances.append(g.distance(readings, families))
    values = np.asarray(distances)
    return {
        "splits": W_SPLITS,
        "seed": W_SEED,
        "families": families,
        "size_step": W_SIZE_STEP,
        "uniform_lawn": "left out: one held-out date",
        "median": float(np.median(values)),
        "minimum_95th": float(np.percentile(values, 95)),
    }


# ---------------------------------------------------------------------------------------------
# The CLI
# ---------------------------------------------------------------------------------------------


def _git(*args: str) -> subprocess.CompletedProcess[bytes]:
    """Run git in the repository with fixed arguments."""
    return subprocess.run(  # noqa: S603 - fixed git arguments, no user input
        ["git", *args],  # noqa: S607 - git on the PATH
        cwd=_REPO,
        capture_output=True,
        check=False,
    )


def launch_committed() -> bool:
    """Return whether the launch record is committed and unchanged, and its parameters with it.

    The matched negatives' parameters must be byte-identical to their version in the commit that
    last touched the launch record, and unchanged in the working tree, so the negatives read are
    the ones registered.
    """
    launch = str(LAUNCH_PATH.relative_to(_REPO))
    params = str(PARAMS_PATH.relative_to(_REPO))
    commit = _git("log", "-1", "--format=%H", "--", launch).stdout.decode().strip()
    if not commit:
        return False
    if _git("diff", "--quiet", "HEAD", "--", launch, params).returncode != 0:
        return False
    registered = _git("show", f"{commit}:{params}")
    return registered.returncode == 0 and registered.stdout == PARAMS_PATH.read_bytes()


def _finite(value: Any) -> Any:  # noqa: ANN401 - any JSON-shaped value
    if isinstance(value, dict):
        return {str(k): _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


def main(argv: list[str] | None = None) -> int:
    """CLI: the pilot, or the registered reading."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("pilot", "read"):
        sub.add_parser(mode).add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode == "pilot":
        record: dict[str, Any] = {
            "description": "Pilot: the negatives on band data only, as description.",
            "animals": [PILOT_ANIMALS[0], PILOT_ANIMALS[-1]],
            "networks": PILOT_NETWORKS,
            "negatives": grade_negatives(PILOT_ANIMALS, PILOT_NETWORKS),
        }
    else:
        if not launch_committed():
            print(
                f"{LAUNCH_PATH} is not committed and clean; nothing held-out is read.",
                file=sys.stderr,
            )
            return 1
        bands = bg.load_bands()
        builder = bb.BandBuilder()
        units = positive_controls(builder, bands)
        failed = [name for name, unit in units.items() if not unit["passes"]]
        rates = [
            UNIT_FAILURE_RATES["mutant" if u.startswith("mutant/") else u.split("/")[0]]
            for u in units
        ]
        negatives = grade_negatives(READ_ANIMALS, READ_NETWORKS)
        passing_negatives = [n for n, r in negatives.items() if not r["fails_wild_type"]]
        controls_pass = len(failed) <= ALLOWANCE
        record = {
            "description": DESCRIPTION,
            "launch": str(LAUNCH_PATH.relative_to(_REPO)),
            "positive_controls": {
                "units": units,
                "failed": failed,
                "allowance": ALLOWANCE,
                "p_more_than_allowance": allowance_tail(rates, ALLOWANCE),
                "pass": controls_pass,
            },
            "negatives": negatives,
            "negatives_passing": passing_negatives,
            "w_minimum": w_minimum(builder),
            "verdict": "VALID" if controls_pass and not passing_negatives else "VOID",
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_finite(record), indent=2, allow_nan=False) + "\n")
    if args.mode == "read":
        print(
            "verdict:",
            record["verdict"],
            "| failed units:",
            record["positive_controls"]["failed"],
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
