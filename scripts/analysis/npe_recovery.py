#!/usr/bin/env python
r"""Parameter recovery for C.1's mechanistic baseline by neural posterior estimation.

The positive control D38 asks of each fitting method: generate the benchmark's statistics at
known parameters, and check the fit gets them back.

**The model** is :class:`quantumnematode.evaluation.controllers.BatchedMechanistic`, 20 parameters.
NPE works on a unit box, ``z`` in [0, 1]^20, mapped to each parameter by its prior: uniform, or
log-uniform. No López-Cruz value sets a prior.

**A simulation** turns z into the controller and runs it at 200 animals through the batched
evaluator on the three wild-type assays (Scheer's small lawn and uniform well, López-Cruz's food
removal). The grader returns the 11 wild-type statistics and local search's presence bound.
Simulation k's animals are ``1_000_000 + 200 k + i``, or ``2_000_000 + 200 k + i`` in the pilot.

**The summary** is those 12 numbers. Each is z-scored over the training simulations where it is
defined; an undefined entry is set to -5, with a 0/1 flag beside it: 24 inputs.

**The gates** (registered in ``supporting/111-npe-recovery/launch.md``), for each of two flows:

- **calibrated:** over 100 held-out truths, the pooled central-90% interval coverage of the 20
  parameters lies in [0.80, 0.97];
- **informative:** for 20 of those truths, simulations at 20 posterior draws reproduce the truth's
  statistics with a median, over truths, of the median D at most 1.42 (W's minimum). D is the
  weighted RMS over the 11 statistics of ``(s_draw - s_truth) / scale``, the scale the bands' at
  n = 200, capped at 10. A statistic undefined in both counts 0, and in one only, the cap.

Subcommands::

    uv run python scripts/analysis/npe_recovery.py pilot --out <pilot.json>
    uv run python scripts/analysis/npe_recovery.py simulate --workers 16
    uv run python scripts/analysis/npe_recovery.py train --out <recovery.json>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

DESCRIPTION = "Parameter recovery for C.1's mechanistic baseline by neural posterior estimation."
_REPO = Path(__file__).resolve().parents[2]
SUPPORT = _REPO / "docs" / "experiments" / "logbooks" / "supporting" / "111-npe-recovery"
SIMS_PATH = _REPO / "build" / "npe-recovery" / "simulations.jsonl"

ANIMALS = 200
SEED_BASE = 1_000_000
PILOT_SEED_BASE = 2_000_000
PRIOR_SEED = 5401
FLOW_SEEDS = (5402, 5403)
PRIOR_PREDICTIVE_SEED = 5404
TRAINING = 10_000
TRUTHS = 100
PREDICTIVE_TRUTHS = 20
PREDICTIVE_DRAWS = 20
POSTERIOR_SAMPLES = 1000
COVERAGE_LEVEL = 0.90
COVERAGE_GATE = (0.80, 0.97)
INFORMATIVE_GATE = 1.42
SENTINEL = -5.0
NEAR_WILD_TYPE = 25
PILOT_SIMULATIONS = 64

# Each parameter's prior: ("uniform", low, high) or ("log", low, high), the log one uniform in ln.
# The switching intercepts are logs of rates per second, so their prior is uniform in [ln 1e-4,
# ln 0.1].
PRIOR: dict[str, tuple[str, float, float]] = {
    "a0": ("uniform", math.log(1e-4), math.log(0.1)),
    "a_intake": ("uniform", -5.0, 5.0),
    "a_satiety": ("uniform", -5.0, 5.0),
    "b0": ("uniform", math.log(1e-4), math.log(0.1)),
    "b_intake": ("uniform", -5.0, 5.0),
    "b_satiety": ("uniform", -5.0, 5.0),
    "v_dwell": ("uniform", 0.005, 0.20),
    "v_roam": ("uniform", 0.005, 0.20),
    "sigma_dwell": ("uniform", 5.0, 60.0),
    "sigma_roam": ("uniform", 5.0, 60.0),
    "kappa_weathervane": ("uniform", -50.0, 50.0),
    "r_dwell": ("log", 1e-3, 0.5),
    "r_roam": ("log", 1e-3, 0.5),
    "beta_reversal": ("uniform", 0.0, 5.0),
    "edge": ("uniform", 0.0, 5.0),
    "o_dwell": ("log", 1e-3, 0.5),
    "o_roam": ("log", 1e-3, 0.5),
    "beta_omega": ("uniform", 0.0, 5.0),
    "search_amplitude": ("uniform", 0.0, 20.0),
    "search_tau_s": ("log", 10.0, 3000.0),
}


def parameters_from_unit(z: np.ndarray) -> dict[str, float]:
    """Map a point of the unit box to C.1's parameters through each prior."""
    from quantumnematode.evaluation.controllers import MECHANISTIC_PARAMETERS

    out = {}
    for value, name in zip(np.asarray(z, dtype=float), MECHANISTIC_PARAMETERS, strict=True):
        kind, low, high = PRIOR[name]
        if kind == "log":
            out[name] = float(math.exp(math.log(low) + value * (math.log(high) - math.log(low))))
        else:
            out[name] = float(low + value * (high - low))
    return out


def prior_draws() -> np.ndarray:
    """Return every registered prior draw: the training simulations, then the truths."""
    return np.random.default_rng(PRIOR_SEED).random((TRAINING + TRUTHS, 20))


# ---------------------------------------------------------------------------------------------
# The simulator
# ---------------------------------------------------------------------------------------------

_STATE: dict[str, Any] = {}


def _worker_setup() -> None:
    """One thread per worker process: the pool parallelises across simulations."""
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    import torch

    torch.set_num_threads(1)


def _shared() -> dict[str, Any]:
    if not _STATE:
        import benchmark_grade as bg
        import quantumnematode.brain  # noqa: F401  - registers the brains the configs name
        from quantumnematode.utils.config_loader import load_simulation_config

        _STATE["bg"] = bg
        _STATE["bands"] = bg.load_bands()
        _STATE["configs"] = {
            family: load_simulation_config(
                str(bg.FORAGING / f"{bg.ARMS['mlp']}_assay_{protocol}.yml"),
            )
            for family, protocol in bg.PROTOCOLS.items()
        }
    return _STATE


def simulate(k: int, z: list[float], seed_base: int = SEED_BASE) -> dict[str, Any]:
    """Run simulation k at the unit-box point z; return its statistics and presence bound."""
    from quantumnematode.evaluation.batched import BatchedEvaluator
    from quantumnematode.evaluation.controllers import BatchedMechanistic

    shared = _shared()
    bg = shared["bg"]
    seeds = [seed_base + ANIMALS * k + i for i in range(ANIMALS)]
    start = time.perf_counter()
    populations = {}
    for family, config in shared["configs"].items():
        controller = BatchedMechanistic(parameters_from_unit(np.asarray(z)))
        trajectories = BatchedEvaluator(config, controller, seeds).run()
        assert trajectories.assay is not None  # noqa: S101 - every assay config records its timing
        analysis = tuple(trajectories.assay["analysis_steps"])  # type: ignore[arg-type]
        populations[family] = [
            bg.window(family, trajectories.to_behaviour_steps(i), analysis) for i in range(ANIMALS)
        ]
    graded = bg.grade_populations(populations, shared["bands"])
    stats = {
        f"{family}/{name}": reading["value"]
        for family, readings in graded["families"].items()
        for name, reading in readings.items()
    }
    return {
        "k": k,
        "z": list(z),
        "statistics": stats,
        "presence": graded["local_search_presence"]["lower_bound"],
        "D": graded["D"],
        "seconds": time.perf_counter() - start,
    }


def _finite(value: Any) -> Any:  # noqa: ANN401 - any JSON-shaped value
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


def run_pool(jobs: list[tuple[int, list[float]]], workers: int, path: Path, seed_base: int) -> None:
    """Run jobs not yet in ``path``, appending each result as one JSON line when it completes."""
    done = (
        {json.loads(line)["k"] for line in path.read_text().splitlines()}
        if path.exists()
        else set()
    )
    todo = [(k, z) for k, z in jobs if k not in done]
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"{len(done)} done, {len(todo)} to run on {workers} workers", flush=True)
    with (
        ProcessPoolExecutor(max_workers=workers, initializer=_worker_setup) as pool,
        path.open("a") as out,
    ):
        futures = [pool.submit(simulate, k, z, seed_base) for k, z in todo]
        for count, future in enumerate(as_completed(futures), start=1):
            out.write(json.dumps(_finite(future.result()), allow_nan=False) + "\n")
            out.flush()
            if count % 100 == 0:
                print(f"{count}/{len(todo)}", flush=True)


def load_simulations(path: Path = SIMS_PATH) -> dict[int, dict[str, Any]]:
    """Return every completed simulation, by k."""
    return {r["k"]: r for r in (json.loads(line) for line in path.read_text().splitlines())}


# ---------------------------------------------------------------------------------------------
# The summary, the gates and the readings
# ---------------------------------------------------------------------------------------------


def statistic_names() -> list[str]:
    """Return the 11 statistics' keys and the presence bound's, in the summary's order."""
    from quantumnematode.benchmark import grading as g

    return [f"{f}/{s}" for f, stats in g.FAMILIES.items() for s in stats] + ["presence"]


def raw_summary(record: dict[str, Any]) -> np.ndarray:
    """Return a simulation's 12 values, NaN where undefined."""
    names = statistic_names()
    by_name = {**record["statistics"], "presence": record["presence"]}
    return np.array([np.nan if by_name[n] is None else float(by_name[n]) for n in names])


def standardiser(training: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return each column's mean and SD over the training rows where it is defined."""
    mean = np.nanmean(training, axis=0)
    sd = np.nanstd(training, axis=0)
    return mean, np.where(sd > 0, sd, 1.0)


def summarise(raw: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Z-score, set undefined entries to the sentinel, and append a 0/1 flag per statistic."""
    raw = np.atleast_2d(raw)
    undefined = ~np.isfinite(raw)
    z = np.where(undefined, SENTINEL, (raw - mean) / sd)
    return np.hstack([z, undefined.astype(float)])


def coverage(samples: np.ndarray, truth: np.ndarray, level: float = COVERAGE_LEVEL) -> np.ndarray:
    """Return, per parameter, whether the truth lies in the samples' central ``level`` interval."""
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(samples, [tail, 1.0 - tail], axis=0)
    return (truth >= low) & (truth <= high)


def contraction(samples: np.ndarray) -> np.ndarray:
    """Return 1 - posterior SD / prior SD per parameter, on the unit box (prior SD 1/sqrt(12))."""
    return 1.0 - samples.std(axis=0) * math.sqrt(12.0)


def predictive_distance(
    truth: dict[str, Any],
    draw: dict[str, Any],
    scales: dict[str, float],
) -> float:
    """Return D between two simulations' 11 statistics on the bands' scales."""
    from quantumnematode.benchmark import grading as g

    readings: dict[str, dict[str, Any]] = {f: {} for f in g.FAMILIES}
    for family, stats in g.FAMILIES.items():
        for s in stats:
            key = f"{family}/{s}"
            a, b = truth["statistics"][key], draw["statistics"][key]
            if a is None and b is None:
                z = 0.0
            elif a is None or b is None:
                z = g.Z_CAP
            else:
                z = g.z_score(float(b) - float(a), g.Band(0.0, scales[key]))
            readings[family][s] = g.Reading(0.0, g.Band(0.0, scales[key]), z, passes=True)
    return g.distance(readings, list(g.FAMILIES))


def band_scales() -> dict[str, float]:
    """Return each statistic's band scale at n = 200."""
    import benchmark_grade as bg
    from quantumnematode.benchmark import grading as g

    bands = bg.load_bands()["bands"]["200"]
    return {
        f"{f}/{s}": float(bands[f][s]["scale"]) for f, stats in g.FAMILIES.items() for s in stats
    }


def posterior_predictive_jobs(
    flow: int,
    truth: int,
    draws: np.ndarray,
) -> list[tuple[int, list[float]]]:
    """Return the simulation jobs for one (flow, truth) pair's posterior draws."""
    base = TRAINING + TRUTHS + flow * PREDICTIVE_TRUTHS * PREDICTIVE_DRAWS
    return [
        (base + truth * PREDICTIVE_DRAWS + d, list(map(float, draws[d]))) for d in range(len(draws))
    ]


def prior_predictive_jobs() -> list[tuple[int, list[float]]]:
    """Return the prior-predictive jobs: 20 prior draws per predictive truth."""
    base = TRAINING + TRUTHS + len(FLOW_SEEDS) * PREDICTIVE_TRUTHS * PREDICTIVE_DRAWS
    draws = np.random.default_rng(PRIOR_PREDICTIVE_SEED).random(
        (PREDICTIVE_TRUTHS * PREDICTIVE_DRAWS, 20),
    )
    return [(base + j, list(map(float, draws[j]))) for j in range(len(draws))]


def train_flows(theta: np.ndarray, x: np.ndarray) -> list[Any]:
    """Train one NPE posterior per registered seed (sbi's neural-spline flow)."""
    import torch
    from sbi.inference import NPE
    from sbi.utils import BoxUniform

    prior = BoxUniform(low=torch.zeros(20), high=torch.ones(20))
    posteriors = []
    for seed in FLOW_SEEDS:
        torch.manual_seed(seed)
        inference = NPE(prior=prior, density_estimator="nsf")
        inference.append_simulations(
            torch.as_tensor(theta, dtype=torch.float32),
            torch.as_tensor(x, dtype=torch.float32),
        )
        estimator = inference.train()
        posteriors.append(inference.build_posterior(estimator))
    return posteriors


def sample(posterior: Any, x: np.ndarray, n: int, seed: int) -> np.ndarray:  # noqa: ANN401
    """Draw n posterior samples on the unit box at summary x."""
    import torch

    torch.manual_seed(seed)
    return posterior.sample(
        (n,),
        x=torch.as_tensor(x, dtype=torch.float32),
        show_progress_bars=False,
    ).numpy()


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0915 - three stages, one record
    """CLI: the pilot, the simulations, and the training with its gates."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    sub = parser.add_subparsers(dest="stage", required=True)
    pilot = sub.add_parser("pilot")
    pilot.add_argument("--out", type=Path, required=True)
    pilot.add_argument("--workers", type=int, default=16)
    sim = sub.add_parser("simulate")
    sim.add_argument("--workers", type=int, default=16)
    train = sub.add_parser("train")
    train.add_argument("--out", type=Path, required=True)
    train.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    draws = prior_draws()

    if args.stage == "pilot":
        path = _REPO / "build" / "npe-recovery" / "pilot.jsonl"
        jobs = [(k, list(map(float, draws[k]))) for k in range(PILOT_SIMULATIONS)]
        start = time.perf_counter()
        run_pool(jobs, args.workers, path, PILOT_SEED_BASE)
        wall = time.perf_counter() - start
        records = load_simulations(path)
        seconds = np.array([r["seconds"] for r in records.values()])
        total = TRAINING + TRUTHS + 3 * PREDICTIVE_TRUTHS * PREDICTIVE_DRAWS
        per_wall = wall / PILOT_SIMULATIONS
        undefined = np.mean([np.isnan(raw_summary(r)).any() for r in records.values()])
        record = {
            "simulations": PILOT_SIMULATIONS,
            "workers": args.workers,
            "seconds_per_simulation_median": float(np.median(seconds)),
            "seconds_per_simulation_wall": per_wall,
            "projected_hours": total * per_wall / 3600.0,
            "simulations_registered": total,
            "share_with_an_undefined_statistic": float(undefined),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record, indent=2))
        return 0

    if args.stage == "simulate":
        jobs = [(k, list(map(float, draws[k]))) for k in range(TRAINING + TRUTHS)]
        run_pool(jobs, args.workers, SIMS_PATH, SEED_BASE)
        return 0

    sims = load_simulations()
    missing = [k for k in range(TRAINING + TRUTHS) if k not in sims]
    if missing:
        print(f"{len(missing)} simulations missing; run simulate first", file=sys.stderr)
        return 1
    raw = np.vstack([raw_summary(sims[k]) for k in range(TRAINING + TRUTHS)])
    mean, sd = standardiser(raw[:TRAINING])
    x = summarise(raw, mean, sd)
    theta = draws
    posteriors = train_flows(theta[:TRAINING], x[:TRAINING])

    flows: list[dict[str, Any]] = []
    predictive: list[tuple[int, list[float]]] = []
    samples_by_flow = []
    for f, posterior in enumerate(posteriors):
        covered, contracted, per_truth = [], [], []
        flow_samples = []
        for j in range(TRUTHS):
            s = sample(posterior, x[TRAINING + j], POSTERIOR_SAMPLES, FLOW_SEEDS[f] * 1000 + j)
            flow_samples.append(s)
            c = coverage(s, theta[TRAINING + j])
            covered.append(c)
            contracted.append(contraction(s))
            per_truth.append(float(c.mean()))
            if j < PREDICTIVE_TRUTHS:
                predictive += posterior_predictive_jobs(f, j, s[:PREDICTIVE_DRAWS])
        samples_by_flow.append(flow_samples)
        covered_arr, contracted_arr = np.array(covered), np.array(contracted)
        flows.append(
            {
                "seed": FLOW_SEEDS[f],
                "pooled_coverage": float(covered_arr.mean()),
                "coverage_by_parameter": covered_arr.mean(axis=0).tolist(),
                "contraction_by_parameter": contracted_arr.mean(axis=0).tolist(),
            },
        )
    run_pool(predictive + prior_predictive_jobs(), args.workers, SIMS_PATH, SEED_BASE)
    sims = load_simulations()
    scales = band_scales()
    from quantumnematode.evaluation.controllers import MECHANISTIC_PARAMETERS

    truth_d = np.array(
        [
            sims[TRAINING + j]["D"] if sims[TRAINING + j]["D"] is not None else np.inf
            for j in range(TRUTHS)
        ],
    )
    near = np.argsort(truth_d)[:NEAR_WILD_TYPE]
    prior_jobs = prior_predictive_jobs()
    prior_d = [
        float(
            np.median(
                [
                    predictive_distance(
                        sims[TRAINING + j],
                        sims[prior_jobs[j * PREDICTIVE_DRAWS + d][0]],
                        scales,
                    )
                    for d in range(PREDICTIVE_DRAWS)
                ],
            ),
        )
        for j in range(PREDICTIVE_TRUTHS)
    ]
    for f, flow in enumerate(flows):
        medians = []
        for j in range(PREDICTIVE_TRUTHS):
            jobs = posterior_predictive_jobs(f, j, samples_by_flow[f][j][:PREDICTIVE_DRAWS])
            medians.append(
                float(
                    np.median(
                        [predictive_distance(sims[TRAINING + j], sims[k], scales) for k, _ in jobs],
                    ),
                ),
            )
        near_cov = np.array([coverage(samples_by_flow[f][j], theta[TRAINING + j]) for j in near])
        flow["informative_median_D"] = float(np.median(medians))
        flow["informative_D_by_truth"] = medians
        flow["prior_predictive_median_D"] = float(np.median(prior_d))
        flow["near_wild_type_coverage"] = float(near_cov.mean())
        flow["calibrated"] = COVERAGE_GATE[0] <= flow["pooled_coverage"] <= COVERAGE_GATE[1]
        flow["informative"] = flow["informative_median_D"] <= INFORMATIVE_GATE
        flow["passes"] = bool(flow["calibrated"] and flow["informative"])
    record = {
        "description": DESCRIPTION,
        "launch": str((SUPPORT / "launch.md").relative_to(_REPO)),
        "parameters": list(MECHANISTIC_PARAMETERS),
        "prior": {k: list(v) for k, v in PRIOR.items()},
        "training": TRAINING,
        "truths": TRUTHS,
        "flows": flows,
        "passes": all(f["passes"] for f in flows),
        "summary_standardiser": {
            "mean": mean.tolist(),
            "sd": sd.tolist(),
            "names": statistic_names(),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_finite(record), indent=2, allow_nan=False) + "\n")
    print(
        json.dumps({k: record[k] for k in ("passes",)}),
        [
            {
                k: f[k]
                for k in (
                    "seed",
                    "pooled_coverage",
                    "informative_median_D",
                    "prior_predictive_median_D",
                    "passes",
                )
            }
            for f in flows
        ],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
