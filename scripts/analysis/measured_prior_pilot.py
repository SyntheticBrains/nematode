#!/usr/bin/env python
"""The measured-prior pilot: sign versus magnitude, and the unit-scale multiplier swept.

B.1c's 2x3 cannot register on a multiplier nobody has swept, and the measured weights may leave the
klinotaxis pathway unlearnable. This pilot reads both, on both learners, at pilot scale:

* **PPO** under the per-neuron fan-in draw, which is the shared initialisation B.1c's PPO arm runs
  under; its parents are A.1's committed ``_fanin`` configs.
* **The reading learner** (``readout_only``) at A.2's reading centre, under the edge-order draw A.2
  swept it at. It freezes the chemical matrix, so the prior is the substrate it reads rather than an
  initialisation.

Each learner runs seven levels -- the random prior, the sign-only prior, and the measured prior at
five multipliers -- on both wirings, learning and frozen. **Every level changes the substrate before
any learning, so every level is gated against its own frozen floor.**

**The multiplier is chosen on the learner's own gate and never on the wiring gap.** The gap is
recorded at every level, with its interval, and the selection is a function of the gates alone, so
it cannot read the gap. At eight seeds the gap is descriptive: it can raise a condition B.1c carries,
never a finding.

The gate, the censoring rule, the metric choice, the wiring gap and the drift check are A.2's own
functions, imported from ``operating_point_surface``. A second copy of any of them is how two scripts
come to disagree about a gate.
"""

# pyright: reportPrivateUsage=false
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import operating_point_surface as ops  # noqa: E402  # pyright: ignore[reportMissingImports]
import wiring_premise as wp  # noqa: E402  # pyright: ignore[reportMissingImports]

# ── Seeds ────────────────────────────────────────────────────────────────────────────────────
# Untouched before this pilot: 1-96, 101-108 and 129-160 are burnt, A.2 took 109-112 for its pilot
# and 161-192 for its panels, and the A.1 re-read took 193-224. B.1c's panel starts at 225, so the
# pilot never shares a seed with the contrast it calibrates.
SEEDS_BY_HALF: dict[str, tuple[int, ...]] = {
    "ppo": tuple(range(113, 121)),
    "reading": tuple(range(121, 129)),
}
HALVES = tuple(SEEDS_BY_HALF)
B1C_FIRST_SEED = 225

# ── Arms ─────────────────────────────────────────────────────────────────────────────────────
_PPO = "connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_hard350"
PARENTS: dict[str, dict[str, str]] = {
    "ppo": {
        "wt_learn": f"{_PPO}_fanin",
        "rn_learn": f"{_PPO}_rewired_null_fanin",
        "wt_frozen": f"{_PPO}_frozen_fanin",
        "rn_frozen": f"{_PPO}_rewired_null_frozen_fanin",
    },
    "reading": dict(ops.CENTRE_STEMS["reading"]),
}
ARMS = ops.ARMS

# ── Levels ───────────────────────────────────────────────────────────────────────────────────
# (suffix, weight_prior, measured_weight_scale). The random level is the parents unchanged. The
# multiplier is written only where it is not the default: at 1.0 it would be a no-op key, and under
# the sign-only prior it is refused because nothing reads it.
RANDOM = "random"
LEVELS: tuple[tuple[str, str, float], ...] = (
    ("sign", "measured_signs", 1.0),
    ("m025", "measured", 0.25),
    ("m05", "measured", 0.5),
    ("m1", "measured", 1.0),
    ("m2", "measured", 2.0),
    ("m4", "measured", 4.0),
)
ALL_LEVELS: tuple[str, ...] = (RANDOM, *(s for s, _, _ in LEVELS))
SIGN_LEVEL = "sign"
MULTIPLIER_LEVELS: dict[str, float] = {s: m for s, prior, m in LEVELS if prior == "measured"}
# The level where the covered edges carry the random draw's magnitude, so B.1c compares structure
# rather than size. Chosen whenever it passes.
DEFAULT_LEVEL = "m1"


def level_keys(suffix: str) -> dict[str, Any]:
    """Return the keys a level adds to its parent's brain config: at most two, often one."""
    if suffix == RANDOM:
        return {}
    _, prior, multiplier = next(level for level in LEVELS if level[0] == suffix)
    keys: dict[str, Any] = {"weight_prior": prior}
    if multiplier != 1.0:
        keys["measured_weight_scale"] = multiplier
    return keys


def stem_for(half: str, arm: str, suffix: str) -> str:
    """Build the config stem for one arm at one level: the parent, then the level."""
    parent = PARENTS[half][arm]
    return parent if suffix == RANDOM else f"{parent}_measured_{suffix}"


def _arms_by_stem() -> dict[str, tuple[str, str, str]]:
    """Config stem -> ``(half, arm, level)``, built by a loop and never by a regex."""
    out: dict[str, tuple[str, str, str]] = {}
    for half in HALVES:
        for suffix in ALL_LEVELS:
            for arm in ARMS:
                stem = stem_for(half, arm, suffix)
                if stem in out:
                    msg = f"stem {stem!r} is claimed by {out[stem]} and by {(half, arm, suffix)}"
                    raise ValueError(msg)
                out[stem] = (half, arm, suffix)
    return out


ARM_BY_STEM: dict[str, tuple[str, str, str]] = _arms_by_stem()


class PilotError(ValueError):
    """The panel on disk is not the panel this module scores."""


# ── Manifest ─────────────────────────────────────────────────────────────────────────────────
def build_manifest(campaign_dir: Path, path: Path, half: str, seeds: tuple[int, ...]) -> Path:
    """Write A.2's ``<arm> <level> <seed> <log>`` lines for every run of ``half``.

    The line format is the one ``operating_point_surface``'s gate, drift check and scorer read, with
    the arms named ``wt_learn``, ``rn_learn``, ``wt_frozen`` and ``rn_frozen``.
    """
    logs = campaign_dir / "logs"
    if not logs.is_dir():
        logs = campaign_dir
    seen: set[tuple[str, str, int]] = set()
    lines: list[str] = []
    for log in sorted(logs.glob("*.log")):
        stem, sep, seed_part = log.stem.rpartition("-seed")
        if not sep:
            msg = f"{log.name} has no -seedN suffix, so it cannot be placed in the panel"
            raise PilotError(msg)
        entry = ARM_BY_STEM.get(stem)
        if entry is None:
            msg = f"{log.name} names config {stem!r}, which this pilot does not have"
            raise PilotError(msg)
        log_half, arm, suffix = entry
        seed = int(seed_part)
        if log_half != half or seed not in seeds:
            continue
        key = (arm, suffix, seed)
        if key in seen:
            msg = f"{key} appears twice in {campaign_dir}"
            raise PilotError(msg)
        seen.add(key)
        resolved = log.resolve()
        try:
            out = str(resolved.relative_to(wp.REPO))
        except ValueError:
            out = str(resolved)
        lines.append(f"{arm} {suffix} {seed} {out}")
    path.write_text("\n".join(lines) + "\n")
    return path


def require_complete(manifest: Path, half: str, seeds: tuple[int, ...]) -> None:
    """Refuse a partial panel: all four arms at every level, on every seed."""
    have: dict[tuple[str, str], set[int]] = {}
    for raw in manifest.read_text().splitlines():
        arm, suffix, seed, _ = raw.split()
        have.setdefault((arm, suffix), set()).add(int(seed))
    missing = [
        f"{suffix}/{arm}: {sorted(set(seeds) - have.get((arm, suffix), set()))}"
        for suffix in ALL_LEVELS
        for arm in ARMS
        if set(seeds) - have.get((arm, suffix), set())
    ]
    if missing:
        msg = f"{half} pilot is incomplete — " + "; ".join(missing)
        raise PilotError(msg)


# ── Selection: a function of the gates alone ─────────────────────────────────────────────────
def wild_type_learns(gates: dict[str, Any]) -> bool:
    """Say whether the wild type beats its own floor: can this substrate be learned at all."""
    test = gates["wt"]["vs_floor"] or {}
    return float(test.get("ci_lo", 0.0)) > 0.0


def level_passes(gates: dict[str, Any]) -> bool:
    """Both learning arms beat their floors and the level is not saturated.

    Whether a wiring contrast could be read there, not which arm learns more: a level whose null is
    broken, or whose two arms sit above the instrument's ceiling, cannot carry B.1c's contrast.
    """
    return bool(gates["gate_passes"]) and not bool(gates["saturated"])


def _nearest_default(passing: list[str]) -> str:
    """Return the passing multiplier nearest 1.0 on the log scale, ties to the smaller."""
    return min(
        passing,
        key=lambda s: (abs(math.log(MULTIPLIER_LEVELS[s])), MULTIPLIER_LEVELS[s]),
    )


def select(gates_by_level: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Read the registered branches and choose the multiplier, from the gates and nothing else.

    Takes only the gate records, so the wiring gap cannot enter: that is the registration's rule,
    and a signature that cannot see the gap is how it is kept rather than promised.
    """
    learns = {s: wild_type_learns(g) for s, g in gates_by_level.items()}
    passes = {s: level_passes(g) for s, g in gates_by_level.items()}
    broken = {
        s: [w for w in ("wt", "rn") if not float((g[w]["vs_floor"] or {}).get("ci_lo", 0.0)) > 0.0]
        for s, g in gates_by_level.items()
    }
    passing = [s for s in MULTIPLIER_LEVELS if passes.get(s)]
    out: dict[str, Any] = {
        "wild_type_learns": learns,
        "level_passes": passes,
        "arms_below_floor": {s: arms for s, arms in broken.items() if arms},
        "saturated": {s: bool(g["saturated"]) for s, g in gates_by_level.items()},
        "chosen_level": None,
        "chosen_multiplier": None,
    }
    if not learns[RANDOM]:
        out["branch"] = "uninformative"
        out["why"] = (
            "the wild type does not beat its floor under the random prior, so no failure of a "
            "measured level can be attributed to the measured weights"
        )
        return out
    if passing:
        chosen = DEFAULT_LEVEL if DEFAULT_LEVEL in passing else _nearest_default(passing)
        out.update(
            branch="selected",
            chosen_level=chosen,
            chosen_multiplier=MULTIPLIER_LEVELS[chosen],
            why=(
                "the default multiplier passes"
                if chosen == DEFAULT_LEVEL
                else "the default does not pass; the passing level nearest it on the log scale"
            ),
        )
        return out
    if not any(learns[s] for s in (*MULTIPLIER_LEVELS, SIGN_LEVEL)):
        out["branch"] = "pathway_unlearnable"
        out["why"] = (
            "the wild type learns under the random prior and under no measured level nor the "
            "sign-only prior"
        )
    elif passes[SIGN_LEVEL]:
        out["branch"] = "magnitude_obstacle"
        out["why"] = "the sign-only prior passes and no measured multiplier does"
    else:
        out["branch"] = "no_level_passes"
        out["why"] = (
            "no measured multiplier passes and neither the pathway nor the magnitude branch applies"
        )
    return out


def sign_movement(levels: dict[str, Any]) -> dict[str, Any]:
    """Levels whose gap interval excludes zero on the side opposite to the random level's.

    Descriptive at pilot scale: a condition B.1c carries, never a finding and never an input to the
    selection. The random level's side is its interval's where that excludes zero, else its mean's.
    """

    def gap(suffix: str) -> dict[str, Any]:
        entry = levels[suffix]
        return entry[entry["metric_choice"]["primary_metric"]]["wiring_gap"]

    base = gap(RANDOM)
    test = base["test"]
    if test["ci_lo"] > 0.0:
        side, basis = 1, "interval"
    elif test["ci_hi"] < 0.0:
        side, basis = -1, "interval"
    else:
        side, basis = (1 if (base["gap_mean"] or 0.0) >= 0.0 else -1), "mean"
    moved = [
        s
        for s in ALL_LEVELS
        if s != RANDOM
        and (gap(s)["test"]["ci_hi"] < 0.0 if side > 0 else gap(s)["test"]["ci_lo"] > 0.0)
    ]
    return {"random_side": side, "random_side_basis": basis, "levels_moved": moved}


# ── Scoring ──────────────────────────────────────────────────────────────────────────────────
def score_half(
    campaign_dir: Path,
    half: str,
    out_dir: Path,
    seeds: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Score one learner: gates, metric and gap per level, then the branches and the selection."""
    seeds = seeds if seeds is not None else SEEDS_BY_HALF[half]
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(campaign_dir, out_dir / f"manifest-{half}.txt", half, seeds)
    require_complete(manifest, half, seeds)

    reports = {
        suffix: ops.score_level(manifest, half, suffix, out_dir / f"tmp-{half}-{suffix}")
        for suffix in ALL_LEVELS
    }
    rates = {suffix: ops.censoring_rates(report) for suffix, report in reports.items()}

    levels: dict[str, Any] = {}
    for suffix in ALL_LEVELS:
        # Per level, against the random level: the cells a gap is compared across.
        choice = ops.choose_metric({RANDOM: rates[RANDOM], suffix: rates[suffix]})
        entry: dict[str, Any] = {
            "keys": level_keys(suffix),
            "metric_choice": choice,
            "gates": ops.learning_gates(manifest, half, seeds, suffix, floor_level=suffix),
        }
        for metric in (ops.CENSORED_METRIC, ops.UNCENSORED_METRIC):
            entry[metric] = {"wiring_gap": ops.wiring_gap(reports[suffix], metric)}
        levels[suffix] = entry

    drift = ops.substrate_drift(
        manifest,
        half,
        seeds,
        ALL_LEVELS,
        floor_levels={s: s for s in ALL_LEVELS},
    )
    drift["obligation_applies"] = half == "reading"

    return {
        "half": half,
        "cell": ops.CELL,
        "seeds": list(seeds),
        "levels": levels,
        "selection": select({s: e["gates"] for s, e in levels.items()}),
        "sign_movement": sign_movement(levels),
        "substrate_drift": drift,
        "verdicts": {suffix: report.get("verdict") for suffix, report in reports.items()},
    }


def write_csv(result: dict[str, Any], path: Path) -> Path:
    """One row per (half, level, seed): each arm's plateau, its floor, and the gap on both metrics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        # csv defaults to CRLF, which would make every regeneration read as a whole-file diff.
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "half",
                "level",
                "weight_prior",
                "measured_weight_scale",
                "seed",
                "wt_plateau",
                "wt_floor",
                "rn_plateau",
                "rn_floor",
                f"gap_{ops.CENSORED_METRIC}",
                f"gap_{ops.UNCENSORED_METRIC}",
                "primary_metric",
            ],
        )
        for half, half_result in result["halves"].items():
            for suffix, entry in half_result["levels"].items():
                prior = entry["keys"].get("weight_prior", "random")
                scale = entry["keys"].get("measured_weight_scale", 1.0)
                gates = entry["gates"]
                censored = entry[ops.CENSORED_METRIC]["wiring_gap"]["per_seed"]
                uncensored = entry[ops.UNCENSORED_METRIC]["wiring_gap"]["per_seed"]
                for seed in half_result["seeds"]:
                    wt = gates["wt"]["per_seed"].get(seed, {})
                    rn = gates["rn"]["per_seed"].get(seed, {})
                    writer.writerow(
                        [
                            half,
                            suffix,
                            prior,
                            scale,
                            seed,
                            _fmt(wt.get("learn")),
                            _fmt(wt.get("floor")),
                            _fmt(rn.get("learn")),
                            _fmt(rn.get("floor")),
                            _fmt(censored.get(seed)),
                            _fmt(uncensored.get(seed)),
                            entry["metric_choice"]["primary_metric"],
                        ],
                    )
    return path


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def main(argv: list[str] | None = None) -> int:
    """CLI: score both learners from the campaign directory."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--campaign", type=Path, required=True, help="campaign directory holding logs/")
    ap.add_argument("--out-dir", type=Path, required=True, help="scratch directory for manifests")
    ap.add_argument("--half", choices=HALVES, action="append", help="score only these learners")
    ap.add_argument("--out", type=Path, help="write the analysis JSON here instead of stdout")
    ap.add_argument("--csv", type=Path, help="write the per-seed CSV here")
    args = ap.parse_args(argv)

    halves = tuple(args.half) if args.half else HALVES
    result = {
        "halves": {half: score_half(args.campaign, half, args.out_dir) for half in halves},
    }
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
    else:
        print(payload)
    if args.csv:
        write_csv(result, args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
