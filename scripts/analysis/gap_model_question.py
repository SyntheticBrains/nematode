#!/usr/bin/env python
"""Two principled gap models, each read for a wiring effect: strength-normalised and ohmic coupling.

On log-compressed counts and on unit weights the wild type shows no lead over its gap-only null. Both of
those corrections shrink each pair's weight and leave the normaliser, ``sqrt(d_i d_j)`` with ``d`` the
binary gap degree, which is what let a few high-count pairs couple far above 1. This module reads the
wiring contrast under two models that change something else:

* **strength** (``gap_normalisation: strength``, count weights): each pair is divided by
  ``sqrt(s_i s_j)``, ``s`` the neuron's total gap weight, so every coupling is at most 1;
* **ohmic** (``gap_coupling: ohmic`` at ``gap_weight_scale: 0.03``): current in proportion to the
  potential difference, at the conductance scale calibrated on untrained brains.

**The panel**: the thermal cell at target 35 on Emmons 2024, 3,000 episodes, seeds 4201-4264, each
model's wild type and gap-only null learning and frozen (512 runs).

**The readings**: ``gap(strength)`` and ``gap(ohmic)`` on ``auc_success``, wild type - null, paired by
seed, with 80% bootstrap intervals; BH-FDR across the two, on the two-sided Wilcoxon p; each classified
by ``mc.classify`` at m = 0.0577. Gates per model: each learning arm beats its frozen floor, and the two
do not both plateau at or above 90%.

**What the pair does to the STOP statement**: either ``placement`` qualifies it (the effect depends on
how electrical synapses are modelled); both ``no_wiring_effect`` leave it standing; anything else is a
dated decision. An ohmic null is a null at weak coupling: at x0.03 the median neuron keeps 93% of its own
drive.

Usage::

    uv run python scripts/analysis/gap_model_question.py plan
    uv run python scripts/analysis/gap_model_question.py probe --out probe.json
    uv run python scripts/analysis/gap_model_question.py preflight --logs <strength pilot> \
        --logs <ohmic evidence> --out preflight.json
    uv run python scripts/analysis/gap_model_question.py table --out table.json
    uv run python scripts/analysis/gap_model_question.py score --logs <panel logs> --out-dir <scratch> \
        --out panel.json --csv per-seed.csv
"""

# pyright: reportPrivateUsage=false
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
for _path in (_HERE, _HERE.parent / "campaigns"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import gap_operating_point as gop  # noqa: E402  # pyright: ignore[reportMissingImports]
import gap_substrate_selection as gs  # noqa: E402  # pyright: ignore[reportMissingImports]
import gate_preflight as gp  # noqa: E402  # pyright: ignore[reportMissingImports]
import generate_gap_model_configs as gen  # noqa: E402  # pyright: ignore[reportMissingImports]
import logcount_contrast as lc  # noqa: E402  # pyright: ignore[reportMissingImports]
import measured_prior_contrast as mc  # noqa: E402  # pyright: ignore[reportMissingImports]
import measured_prior_pilot as mp  # noqa: E402  # pyright: ignore[reportMissingImports]
import operating_point_surface as ops  # noqa: E402  # pyright: ignore[reportMissingImports]
import thermal_split as ts  # noqa: E402  # pyright: ignore[reportMissingImports]

HALF = "ppo"
CELL = ts.CELL
MODELS: tuple[str, ...] = ("strength", "ohmic")
SEEDS: tuple[int, ...] = tuple(range(4201, 4265))
PILOT_SEEDS: tuple[int, ...] = tuple(range(9905, 9913))
SMOKE_SEEDS: tuple[int, ...] = (9913, 9914)
# The ohmic model's gate evidence: the substrate-selection pilot, on the same configs.
OHMIC_EVIDENCE_SEEDS: tuple[int, ...] = tuple(range(9871, 9887))
MINIMUM = gop.CARRIED_MINIMUM
REFERENCE_FRACTION = gs.REFERENCE_FRACTION
PRIMARY_METRIC = ops.UNCENSORED_METRIC
BESIDE_METRIC = ops.CENSORED_METRIC
METRICS: tuple[str, ...] = (PRIMARY_METRIC, BESIDE_METRIC)
EPISODES = 3000
COMMITTED_BANDS: tuple[tuple[int, int], ...] = (
    (2401, 2464),
    (3901, 3964),
    (4001, 4128),
    (513, 576),
    (9801, 9900),
)

_OHMIC = gs.gen.ARMS["thermal"]
STEMS: dict[str, dict[str, str]] = {
    "strength": {role: child for role, (child, _, _) in gen.ARMS.items()},
    "ohmic": {
        "wt_learn": _OHMIC["ohmic_wt_learn"],
        "wt_frozen": _OHMIC["ohmic_wt_frozen"],
        "rn_learn": _OHMIC["ohmic_gaponly_learn"],
        "rn_frozen": _OHMIC["ohmic_gaponly_frozen"],
    },
}
# The untrained probe's reference brains, by the stems whose configs they derive from.
_PROBE_STEMS: dict[str, str] = {
    "count": _OHMIC["count_wt_learn"],
    "log": _OHMIC["log_wt_learn"],
    "ohmic": _OHMIC["ohmic_wt_learn"],
    "strength": STEMS["strength"]["wt_learn"],
}


class QuestionError(ValueError):
    """The panel's runs cannot be read as registered."""


# ── The verdict map ──────────────────────────────────────────────────────────────────────────
def verdict(readable: bool, gap: dict[str, Any], minimum: float = MINIMUM) -> dict[str, Any]:  # noqa: FBT001
    """One model's verdict: gates first, then the state of its corrected gap."""
    if not readable:
        return {"verdict": "unreadable", "why": "a learning arm fails its floor, or both saturate"}
    test = gap["test"]
    mean = float(gap["gap_mean"])
    state = mc.classify(mean, test["ci_lo"], test["ci_hi"], test["bh_q"], minimum)
    out: dict[str, Any] = {"state": state, "verdict": lc.VERDICTS[state]}
    if state == "below":
        out["direction"] = "wild_type" if mean > 0.0 else "null"
    return out


def joint(verdicts: dict[str, str]) -> dict[str, Any]:
    """Return what the pair of verdicts does to the STOP statement."""
    placed = sorted(m for m, v in verdicts.items() if v == "placement")
    if placed:
        return {
            "stop_statement": "qualified",
            "why": "the wiring effect depends on how electrical synapses are modelled",
            "to_the_body": placed,
        }
    if all(v == "no_wiring_effect" for v in verdicts.values()):
        return {
            "stop_statement": "stands",
            "why": (
                "four corrections show no wiring effect: unit weights, log counts, strength "
                "normalisation, and ohmic coupling at its calibrated (weak) conductance"
            ),
            "to_the_body": [],
        }
    return {
        "stop_statement": "dated_decision",
        "why": "neither branch the map names",
        "to_the_body": [],
    }


# ── Manifest and scoring ─────────────────────────────────────────────────────────────────────
def build_manifest(log_dirs: list[Path], path: Path, seeds: tuple[int, ...] = SEEDS) -> Path:
    """Write ``<arm> <level> <seed> <log>`` for every panel run that exited 0 under its own config."""
    seen: dict[tuple[str, str, int], Path] = {}
    lines: list[str] = []
    for arm, level, seed, log in gp.evidence(STEMS, log_dirs):
        status = log.with_suffix(".exit")
        if seed not in seeds or not status.is_file() or status.read_text().strip() != "0":
            continue
        expected = f"{log.stem.rpartition('-seed')[0]}.yml"
        ran = lc._ran_config(log)
        if ran != expected:
            msg = f"{log} exited 0 but records config {ran!r}, not {expected!r}"
            raise QuestionError(msg)
        prior = seen.setdefault((arm, level, seed), log)
        if prior != log:
            msg = f"{arm} {level} seed {seed} has two runs: {prior} and {log}"
            raise QuestionError(msg)
        lines.append(f"{arm} {level} {seed} {log.resolve()}")
    path.write_text("\n".join(lines) + "\n")
    return path


def require_complete(manifest: Path, seeds: tuple[int, ...] = SEEDS) -> None:
    """Refuse a partial panel: every arm of both models on every seed."""
    have: dict[tuple[str, str], set[int]] = {}
    for raw in manifest.read_text().splitlines():
        if raw.strip():
            arm, level, seed, _ = raw.split()
            have.setdefault((arm, level), set()).add(int(seed))
    missing = [
        f"{level}/{arm}: {sorted(set(seeds) - have.get((arm, level), set()))}"
        for level in MODELS
        for arm in gp.ARMS
        if set(seeds) - have.get((arm, level), set())
    ]
    if missing:
        msg = "the gap-model panel is incomplete: " + "; ".join(missing)
        raise QuestionError(msg)


def overlapping_bands(seeds: tuple[int, ...] = SEEDS) -> list[tuple[int, int]]:
    """Return the committed seed bands the panel's seeds touch; empty when they are disjoint."""
    return [(lo, hi) for lo, hi in COMMITTED_BANDS if any(lo <= s <= hi for s in seeds)]


def read(
    gates: dict[str, dict[str, Any]],
    gaps: dict[str, dict[str, Any]],
    minimum: float = MINIMUM,
) -> dict[str, Any]:
    """Each model's verdict, then the joint outcome; the gaps carry the family-corrected q."""
    per_model = {m: verdict(mp.level_passes(gates[m]), gaps[m], minimum) for m in MODELS}
    return {
        "minimum": minimum,
        "models": per_model,
        "joint": joint({m: v["verdict"] for m, v in per_model.items()}),
    }


def score(log_dirs: list[Path], out_dir: Path) -> dict[str, Any]:
    """Score both models, correct the family of two, gate each model, and read."""
    if overlapping_bands():
        msg = f"the panel's seeds overlap committed bands {overlapping_bands()}"
        raise QuestionError(msg)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(log_dirs, out_dir / "manifest-gap-model.txt")
    require_complete(manifest)
    reports = {
        m: ops.score_level(manifest, HALF, m, out_dir / f"tmp-{m}", cell=CELL) for m in MODELS
    }
    gaps = {metric: {m: ops.wiring_gap(reports[m], metric) for m in MODELS} for metric in METRICS}
    for metric in METRICS:
        ops.apply_family_correction(
            {m: {metric: {"interaction": gaps[metric][m]}} for m in MODELS},
        )
    gates = {m: ops.learning_gates(manifest, HALF, SEEDS, m, floor_level=m) for m in MODELS}
    return {
        "cell": CELL,
        "seeds": list(SEEDS),
        "primary_metric": PRIMARY_METRIC,
        "beside_metric": BESIDE_METRIC,
        "gates": gates,
        "gaps": gaps,
        "collapse": (
            f"not read: no run here exceeds {EPISODES} episodes, the horizon below which the "
            "collapse statistic does not apply"
        ),
        "reading": read(gates, gaps[PRIMARY_METRIC]),
    }


def write_csv(result: dict[str, Any], path: Path) -> Path:
    """One row per seed: each model's plateaus and floors, and its gap on both metrics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["seed"]
    for m in MODELS:
        header += [f"{m}_{c}" for c in ("wt_plateau", "wt_floor", "rn_plateau", "rn_floor")]
        header += [f"{m}_gap_{metric}" for metric in METRICS]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        for seed in result["seeds"]:
            row: list[Any] = [seed]
            for m in MODELS:
                for wiring in ("wt", "rn"):
                    per_seed = result["gates"][m][wiring]["per_seed"].get(seed, {})
                    row += [gop._fmt(per_seed.get("learn")), gop._fmt(per_seed.get("floor"))]
                row += [gop._fmt(result["gaps"][x][m]["per_seed"].get(seed)) for x in METRICS]
            writer.writerow(row)
    return path


# ── Before launch ────────────────────────────────────────────────────────────────────────────
def probe(seeds: tuple[int, ...] = PILOT_SEEDS) -> dict[str, Any]:
    """Untrained operating points; the strength model runs only if it keeps half the gap-off spread."""
    import sign_scale_calibration as ssc  # pyright: ignore[reportMissingImports]

    inputs = ssc.inputs()

    def reading(stem: str, **update: object) -> dict[str, float]:
        values = [
            ssc.measure(gs._connectome_config(stem, s).model_copy(update=update), inputs)
            for s in seeds
        ]
        return {
            "saturated": float(np.mean([v[0] for v in values])),
            "action_spread": float(np.mean([v[1] for v in values])),
        }

    out: dict[str, Any] = {
        "seeds": [seeds[0], seeds[-1]],
        "reference_fraction": REFERENCE_FRACTION,
        "gap_off": reading(_PROBE_STEMS["count"], enable_gap_junctions=False),
        **{name: reading(stem) for name, stem in _PROBE_STEMS.items()},
    }
    floor = REFERENCE_FRACTION * out["gap_off"]["action_spread"]
    out["floor"] = floor
    out["launch"] = out["strength"]["action_spread"] >= floor
    return out


def preflight(log_dirs: list[Path]) -> dict[str, Any]:
    """Each model's floor and saturation gates on runs of its own configs, on disjoint seeds."""
    return gp.preflight(STEMS, log_dirs)


def _raw_strengths(stem: str, seed: int) -> dict[str, float]:
    """Each named neuron's gap strength on a wiring, recovered from its degree-normalised matrix."""
    from quantumnematode.brain.arch.connectome_ppo import ConnectomePPOBrain
    from quantumnematode.brain.arch.dtypes import DeviceType

    config = gs._connectome_config(stem, seed).model_copy(update={"gap_normalisation": "degree"})
    topology = ConnectomePPOBrain(config=config, device=DeviceType.CPU).topology
    gap = topology.g_gap.double().numpy()
    degree = (gap != 0).sum(axis=0).astype(np.float64)
    raw = gap * np.sqrt(np.outer(degree, degree))
    names = list(topology.neuron_names)
    return {n: float(raw[names.index(n)].sum()) for n in ("ALA", "CANL", "AVAL")}


def table(seeds: tuple[int, ...] = SEEDS) -> dict[str, Any]:
    """D24's matched statistics for both models' wirings, with the strength model's normalisers."""
    out: dict[str, Any] = {}
    for model in MODELS:
        wt, rn = STEMS[model]["wt_learn"], STEMS[model]["rn_learn"]
        per_seed = {seed: gop.matched_statistics(gop._brain(rn, seed)) for seed in seeds}
        out[model] = {
            wt: {
                "seed_independent": True,
                "stats": gop.matched_statistics(gop._brain(wt, seeds[0])),
            },
            rn: {
                "seed_independent": False,
                "per_seed": per_seed,
                "summary": gop._summary(per_seed),
            },
        }
    strengths = {seed: _raw_strengths(STEMS["strength"]["rn_learn"], seed) for seed in seeds}
    out["strength"]["normalisers"] = {
        "wild_type": _raw_strengths(STEMS["strength"]["wt_learn"], seeds[0]),
        "gap_only_median": {
            n: float(np.median([strengths[s][n] for s in seeds])) for n in ("ALA", "CANL", "AVAL")
        },
    }
    return out


def plan() -> dict[str, Any]:
    """Return the panel's configs and seeds."""
    stems = [stem for model in MODELS for stem in STEMS[model].values()]
    return {
        "configs": [f"configs/scenarios/thermal_foraging/{s}.yml" for s in stems],
        "seeds": [SEEDS[0], SEEDS[-1]],
        "runs": len(stems) * len(SEEDS),
        "episodes": EPISODES,
        "overlapping_bands": overlapping_bands(),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="command", required=True)
    pl = sub.add_parser("plan", help="the panel's configs and seeds")
    pr = sub.add_parser("probe", help="the strength model's untrained operating point")
    pf = sub.add_parser("preflight", help="each model's gates on runs of its own configs")
    pf.add_argument("--logs", type=Path, action="append", required=True)
    tb = sub.add_parser("table", help="D24's matched statistics on the panel's draws")
    sc = sub.add_parser("score", help="read the panel")
    sc.add_argument("--logs", type=Path, action="append", required=True)
    sc.add_argument("--out-dir", type=Path, required=True)
    sc.add_argument("--csv", type=Path, help="write the per-seed CSV here")
    for parser in (pl, pr, pf, tb, sc):
        parser.add_argument("--out", type=Path, help="write the JSON here")
    args = ap.parse_args(argv)
    ok = True
    if args.command == "plan":
        result = plan()
        ok = not result["overlapping_bands"]
    elif args.command == "probe":
        result = probe()
        ok = bool(result["launch"])
    elif args.command == "preflight":
        result = preflight(args.logs)
        ok = bool(result["launch"])
    elif args.command == "table":
        result = table()
    else:
        result = score(args.logs, args.out_dir)
        if args.csv:
            write_csv(result, args.csv)
    payload = json.dumps(result, indent=2, sort_keys=True, default=str) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
    print(payload)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
