"""Why NPE is not used to fit C.1: the prior's coverage near the wild type, read after F.0a.

Description after F.0a's registered gates, which stand. It reads F.0a's 10,000 prior simulations
(``build/npe-recovery/simulations.jsonl``) and re-simulates a few of them:

- **the region:** how far the prior's worms sit from the bands, and how many qualify as truths near
  the wild type;
- **the floor near the wild type:** the 32 prior worms nearest the bands, each re-simulated twice
  with fresh animals;
- **a nearest-neighbour posterior:** for each of those 32, the 20 nearest training simulations in
  statistic space, re-simulated, and the gain fraction (prior D - posterior D) / (prior D - floor);
- **truncation:** the share of the prior box a round-1 flow's highest-density region (quantile
  1e-4) keeps, at the band cohorts' statistics and at 8 near-wild-type worms.

Seeds: re-simulations at animals ``20_000_000``, ``24_000_000`` and ``28_000_000 + 200·k + i``;
the flow's torch seed 9301; the prior sample for the box share 9302.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import npe_recovery as nr

_REPO = Path(__file__).resolve().parents[2]
DESCRIPTION = (
    "Why NPE is not used to fit C.1: the prior's coverage near the wild type, read after F.0a."
)
OUT = nr.SUPPORT / "region_check.json"
BANDS_RECORD = (
    _REPO
    / "docs"
    / "experiments"
    / "logbooks"
    / "supporting"
    / "109-benchmark-grading"
    / "bands_record.json"
)
NEAR = 32
NEIGHBOURS = 20
TRUNCATION_TRUTHS = 8
FIRST_REPEAT_BASE = 20_000_000
SECOND_REPEAT_BASE = 24_000_000
NEIGHBOUR_BASE = 28_000_000
FLOW_SEED = 9301
BOX_SEED = 9302
BOX_SAMPLES = 100_000
QUANTILE = 1e-4
LOW_FLOOR = 1.5


def band_distance(record: dict[str, Any], bands: dict[str, Any], families: list[str]) -> float:
    """Return a simulation's benchmark D to the bands at n = 200; undefined counts the cap."""
    from quantumnematode.benchmark import grading as g

    readings: dict[str, dict[str, Any]] = {}
    for f in families:
        readings[f] = {}
        for s in g.FAMILIES[f]:
            v = record["statistics"][f"{f}/{s}"]
            band = g.Band(bands[f][s]["centre"], bands[f][s]["scale"])
            z = g.Z_CAP if v is None else g.z_score(float(v), band)
            value = math.nan if v is None else float(v)
            readings[f][s] = g.Reading(value, band, z, passes=True)
    return g.distance(readings, families)


def gain_fraction(prior: float, posterior: float, floor: float) -> float:
    """Return the share of the reachable gain over the prior: 1 at the floor, 0 at the prior."""
    return (prior - posterior) / (prior - floor)


def band_observation() -> np.ndarray:
    """Return the band cohorts' 12 raw values: the 11 statistics' centres and median presence."""
    import benchmark_grade as bg
    from quantumnematode.benchmark import grading as g

    b200 = bg.load_bands()["bands"]["200"]
    presence = json.loads(BANDS_RECORD.read_text())["lopezcruz_band_batch_presence"]
    centres = [b200[f][s]["centre"] for f, stats in g.FAMILIES.items() for s in stats]
    return np.array([*centres, float(np.median([v["lower_bound"] for v in presence.values()]))])


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {f"{q:g}": float(np.quantile(values, q)) for q in (0.001, 0.01, 0.05, 0.5)}


def _truncation(sims: dict[int, Any], near: list[int]) -> dict[str, Any]:
    """Train a round-1 flow on the prior simulations and read its region's share of the box."""
    import torch
    from sbi.inference import NPE
    from sbi.utils import BoxUniform, get_density_thresholder

    draws = nr.prior_draws()
    raw = np.vstack([nr.raw_summary(sims[k]) for k in range(nr.TRAINING)])
    mean, sd = nr.standardiser(raw)
    x = nr.summarise(raw, mean, sd)
    torch.manual_seed(FLOW_SEED)
    inference = NPE(
        prior=BoxUniform(low=torch.zeros(20), high=torch.ones(20)),
        density_estimator="nsf",
    )
    inference.append_simulations(
        torch.as_tensor(draws[: nr.TRAINING], dtype=torch.float32),
        torch.as_tensor(x, dtype=torch.float32),
    )
    posterior = inference.build_posterior(inference.train(show_train_summary=False))
    box = torch.rand(BOX_SAMPLES, 20, generator=torch.Generator().manual_seed(BOX_SEED))
    observations = {"bands": nr.summarise(band_observation(), mean, sd)[0]}
    observations |= {f"training_{k}": x[k] for k in near[:TRUNCATION_TRUTHS]}
    out = {}
    for name, xo in observations.items():
        p = posterior.set_default_x(torch.as_tensor(xo, dtype=torch.float32))
        keep = get_density_thresholder(
            p,
            quantile=QUANTILE,
            num_samples_to_estimate_support=BOX_SAMPLES,
        )
        out[name] = float(keep(box).numpy().astype(bool).mean())
    truths = [v for k, v in out.items() if k != "bands"]
    return {"box_share_kept": out, "median_over_near_wild_type": float(np.median(truths))}


def main(argv: list[str] | None = None) -> int:
    """CLI: read the region, re-simulate, and write the record."""
    from quantumnematode.benchmark import grading as g

    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)

    import benchmark_grade as bg

    bands = bg.load_bands()["bands"]["200"]
    sims = nr.load_simulations()
    scales = nr.band_scales()
    ks = range(nr.TRAINING)
    d = np.array([band_distance(sims[k], bands, list(g.FAMILIES)) for k in ks])
    d_fit = np.array([band_distance(sims[k], bands, list(g.FIT_FAMILIES)) for k in ks])
    defined = np.array([all(v is not None for v in sims[k]["statistics"].values()) for k in ks])
    presence = np.array(
        [np.nan if sims[k]["presence"] is None else sims[k]["presence"] for k in ks],
    )
    early_late = np.array(
        [sims[k]["statistics"]["local_search/early_late_ratio"] or np.nan for k in ks],
    )
    region = {
        "D_quantiles": _quantiles(d),
        "D_fit_quantiles": _quantiles(d_fit),
        "within_D": {str(t): int((d <= t).sum()) for t in (3, 4, 5)},
        "median_early_late_within_D_4": float(np.nanmedian(early_late[d <= 4])),
        "pool_D5_presence1.5_defined": int(((d <= 5) & (presence > 1.5) & defined).sum()),
        "placement_D4.5_presence2_defined": int(((d <= 4.5) & (presence > 2) & defined).sum()),
    }

    near = [int(k) for k in np.argsort(d)[:NEAR]]
    neighbours, jobs = [], []
    for t, k in enumerate(near):
        dk = np.array(
            [np.inf if i == k else nr.predictive_distance(sims[k], sims[i], scales) for i in ks],
        )
        order = np.argsort(dk)[:NEIGHBOURS]
        prior = float(np.median([dk[i] for i in range(nr.TRAINING) if i != k][:200]))
        neighbours.append({"prior": prior, "nearest_own_D": float(np.median(dk[order]))})
        jobs += [(t * NEIGHBOURS + n, sims[int(i)]["z"]) for n, i in enumerate(order)]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=nr._worker_setup) as pool:
        first = list(
            pool.map(nr.simulate, near, [sims[k]["z"] for k in near], [FIRST_REPEAT_BASE] * NEAR),
        )
        second = list(
            pool.map(nr.simulate, near, [sims[k]["z"] for k in near], [SECOND_REPEAT_BASE] * NEAR),
        )
        resim = list(
            pool.map(
                nr.simulate,
                [j[0] for j in jobs],
                [j[1] for j in jobs],
                [NEIGHBOUR_BASE] * len(jobs),
            ),
        )

    gamma = "local_search/gamma_per_min"
    rows = []
    for t, k in enumerate(near):
        a, b = nr._finite(first[t]), nr._finite(second[t])
        floor = nr.predictive_distance(sims[k], a, scales)
        post = float(
            np.median(
                [
                    nr.predictive_distance(sims[k], nr._finite(resim[t * NEIGHBOURS + n]), scales)
                    for n in range(NEIGHBOURS)
                ],
            ),
        )
        flips = (a["statistics"][gamma] is None) != (b["statistics"][gamma] is None) or (
            sims[k]["statistics"][gamma] is None
        ) != (a["statistics"][gamma] is None)
        rows.append(
            {
                "training_simulation": k,
                "D_to_bands": float(d[k]),
                "floor": floor,
                "floor_between_repeats": nr.predictive_distance(a, b, scales),
                "gamma_flips": bool(flips),
                "prior_D": neighbours[t]["prior"],
                "nearest_own_D": neighbours[t]["nearest_own_D"],
                "neighbour_posterior_D": post,
                "gain": gain_fraction(neighbours[t]["prior"], post, floor),
            },
        )
    low = [r for r in rows if r["floor"] < LOW_FLOOR]
    summary = {
        "floor_median": float(np.median([r["floor"] for r in rows])),
        "floor_median_gamma_stable": float(
            np.median([r["floor"] for r in rows if not r["gamma_flips"]]),
        ),
        "floor_median_gamma_flips": float(
            np.median([r["floor"] for r in rows if r["gamma_flips"]]),
        ),
        "low_floor_truths": len(low),
        "low_floor": {
            key: float(np.median([r[key] for r in low]))
            for key in ("floor", "prior_D", "nearest_own_D", "neighbour_posterior_D", "gain")
        },
        "low_floor_gain_at_least_0.75": sum(r["gain"] >= 0.75 for r in low),
    }
    record = {
        "description": DESCRIPTION,
        "seeds": {
            "repeats": [FIRST_REPEAT_BASE, SECOND_REPEAT_BASE],
            "neighbours": NEIGHBOUR_BASE,
            "flow": FLOW_SEED,
            "box": BOX_SEED,
        },
        "region": region,
        "near_wild_type": summary,
        "truncation": _truncation(sims, near),
        "truths": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(nr._finite(record), indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "truths"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
