#!/usr/bin/env python
"""L.1b: L.1's width x wiring interaction, re-read at the rate-matched 0.0001.

[L.1](../../docs/experiments/logbooks/066-l4-readout-width.md) read `pooling_hid_structure` at
`plasticity_rate` 0.001: interaction **+0.2818** on ``auc_success``, the wild type ahead at the
per-neuron width where the null led at the pooled width.
[L.4/L.5](../../docs/experiments/logbooks/067-l4-feature-ablations.md) then measured the two wide
learning arms at 0.0001 on the same 96 seeds as its rate-matched baseline, and the wiring effect
there is **-0.0977, null ahead**. The sign-flip form of L.1 is already known not to hold at 0.0001.

What is not known is whether L.1's REGISTERED primary, the interaction, holds there. The 2x2 at
0.0001 is missing exactly its two pooled learning cells; this harness reads them beside the two wide
cells L.4 ran and L.1's floors (the rate is inert under ``freeze_updates``)::

                        pooled (8)       per-neuron (78)
    wild type @1e-4     this change      campaigns/feature-ablations
    rewired null @1e-4  this change      campaigns/feature-ablations
    floors              campaigns/readout-width (L.1), shared

    INTERACTION @1e-4   (B - A) - (D - C)    L.1's primary, at one rate
    THREE-WAY           I_1e-3 - I_1e-4      the size of the rate dependence itself

**The minimum effect is a decision rule**: `pool_effect_survives_the_rate` requires the interaction
at 0.0001 to be significantly positive AND at least **half** of L.1's +0.2818 (0.141). A significant
interaction below that is named *shrunk below half* and reads rate-specific. **Eight tests in one
BH-FDR family**: the interaction, both main effects, the three-way and four learning gates. The
priors are L.1's committed tests on the same floors and are not re-run.

Every cell passes through ``connectome_structure_efficiency.analyse``, once per (rate, width) pair,
unmodified; the helpers are imported from L.1's harness rather than copied.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import connectome_structure_efficiency as eff  # noqa: E402  # pyright: ignore[reportMissingImports]
import l4_feature_ablations as fa  # noqa: E402  # pyright: ignore[reportMissingImports]
import l4_mixture_statistic as ms  # noqa: E402  # pyright: ignore[reportMissingImports]
import l4_readout_width as rw  # noqa: E402  # pyright: ignore[reportMissingImports]
from l4_panel import EXPERIMENTS, read_log  # noqa: E402  # pyright: ignore[reportMissingImports]

SEEDS = rw.SEEDS
_STEM = rw._STEM
WIDTHS = rw.WIDTHS
WIRINGS = rw.WIRINGS
RATES = ("1e-4", "1e-3")
MATCHED, COMMITTED = RATES
POOLED_ARMS = ("wt_pooled", "rn_pooled")
WIDE_ARMS = ("wt_wide", "rn_wide")
FLOORS = tuple(f"{a}_frozen" for a in (*POOLED_ARMS, *WIDE_ARMS))
LEARNING_ARMS = (*POOLED_ARMS, *WIDE_ARMS)
CELLS = (*LEARNING_ARMS, *FLOORS)

# The pooled learning arms at 0.0001. The rate tag is the arm's identity here: `_r1e2` is L.0's
# rate-check tag and is refused, and an untagged pooled log is L.1's arm at 0.001, which this scanner
# must never count as a 0.0001 cell.
_LABEL = re.compile(
    rf"^{re.escape(_STEM)}_readout_only_r1e(?P<rate>\d)(?P<rewired>_rewired_null)?-seed(?P<seed>\d+)\.log$",
)
_RATE_TAG = "4"

PRIMARY_METRIC = rw.PRIMARY_METRIC
CENSORED_METRIC = rw.CENSORED_METRIC
METRIC_NOTE = rw.METRIC_NOTE

# L.1's committed interaction and the registered minimum: half of it.
L1_INTERACTION = 0.2818
MIN_RETAIN = 0.141
MIN_SEEDS = 5
_MIN_FOR_SPREAD = 2

REGISTERED = {
    "l1_interaction_at_1e-3": L1_INTERACTION,
    "minimum_retained": MIN_RETAIN,
    "minimum_rationale": (
        "half of L.1's +0.2818: the pool-hiding claim survives the rate only if at least half of it "
        "is present at the matched rate. Below that, a significant interaction is a claim about the "
        "rate"
    ),
    "wide_wiring_effect_at_1e-4_committed": -0.0977,
    # L.0's pooled-width rate check on the wild type, seeds 101-104. Its published table reports the
    # WHOLE-RUN mean foods; every contrast here reads the plateau-tail mean, so both are carried and
    # named. The ordering is the same on either statistic: 0.001 leads at the pooled width.
    "l0_pooled_rate_check_wild_type": {
        "metric": "mean foods, seeds 101-104",
        "whole_run_mean_as_published": {"1e-4": 12.086, "1e-3": 17.417, "1e-2": 12.810},
        "plateau_tail_mean_the_harness_metric": {"1e-4": 14.488, "1e-3": 18.085, "1e-2": 12.842},
    },
    "n_registered": len(SEEDS),
    "power_at_minimum": "~98% at pooled-cell sd 0.22 (se 0.0346); ~87% at sd 0.30 (se 0.0454)",
    "sign_flip_known_absent": (
        "at 0.0001 the wide wiring effect is -0.0977 with the null ahead (Logbook 067, 96 seeds), "
        "so L.1's sign-flip form cannot reproduce here; this reads L.1's registered primary"
    ),
}

READINGS = {
    "pool_effect_survives_the_rate": (
        "the interaction at 0.0001 is significantly positive and retains at least half of L.1's: "
        "widening still favours the wild type relative to the shuffle at the matched rate, while the "
        "wild type leads at neither width there. The pool-hiding claim survives the rate; the "
        "sign-flip form does not"
    ),
    "pool_effect_is_rate_specific": (
        "no interaction retained at the matched rate -- either a FAILURE TO DETECT, size and interval "
        "carried, or a significant interaction shrunk below half of L.1's. L.1's positive is a "
        "property of 0.001"
    ),
    "width_favours_the_shuffle_at_this_rate": (
        "the interaction at 0.0001 is significantly NEGATIVE: widening favours the shuffle at the "
        "matched rate. Reported as the reverse direction and not explained"
    ),
    "no_learning": "a learning gate failed on a 0.0001 arm, so the interaction is uninterpretable",
    "insufficient_seeds": "too few paired seeds survived to score the panel",
}


def scan(campaign_dir: Path, experiments: Path = EXPERIMENTS) -> dict[str, Any]:
    """Read the two pooled 0.0001 learning arms, keyed as L.1's scanner keys the pooled cells."""
    log_dir = campaign_dir / "logs" if (campaign_dir / "logs").is_dir() else campaign_dir
    runs: dict[str, dict[int, Any]] = {name: {} for name in POOLED_ARMS}
    logs: dict[str, dict[int, Path]] = {}
    for log in sorted(log_dir.glob("*.log")):
        match = _LABEL.match(log.name)
        if match is None:
            print(f"  WARN: skipping log with an unrecognised label: {log.name}")
            continue
        if match.group("rate") != _RATE_TAG:
            msg = (
                f"{log.name}: rate tag _r1e{match.group('rate')} is not the registered 0.0001 arm; "
                "this log is not a cell"
            )
            raise ValueError(msg)
        name = "rn_pooled" if match.group("rewired") else "wt_pooled"
        record = read_log(log, experiments)
        if record is None:
            print(f"  WARN: no parseable run lines in {log.name} - dropped")
            continue
        seed = int(match.group("seed"))
        if seed in runs[name]:
            msg = f"two logs for arm {name} seed {seed}: {log.name} duplicates a read run"
            raise ValueError(msg)
        runs[name][seed] = record
        logs.setdefault(name, {})[seed] = log
    return {"runs": runs, "logs": logs}


def assemble(
    pooled_at_rate: dict[str, Any],
    wide_at_rate: dict[str, Any],
    committed: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Build the two 2x2s, keyed by rate, each in L.1's eight-arm shape.

    At the matched rate the learning cells come from this change (pooled) and L.4's rate-matched
    baseline (wide); the floors are L.1's, the rate being inert under ``freeze_updates``. At the
    committed rate the whole panel is L.1's.
    """
    runs = {
        **{n: pooled_at_rate["runs"][n] for n in POOLED_ARMS},
        **{n: wide_at_rate["runs"][n] for n in WIDE_ARMS},
        **{n: committed["runs"][n] for n in FLOORS},
    }
    logs = {
        **{n: pooled_at_rate["logs"].get(n, {}) for n in POOLED_ARMS},
        **{n: wide_at_rate["logs"].get(n, {}) for n in WIDE_ARMS},
        **{n: committed["logs"].get(n, {}) for n in FLOORS},
    }
    return {MATCHED: {"runs": runs, "logs": logs}, COMMITTED: committed}


def require_complete(panels: dict[str, dict[str, Any]], seeds: tuple[int, ...] = SEEDS) -> None:
    """Refuse to score a registered panel missing any cell at any seed, at either rate."""
    missing = [
        f"{rate} {name} seeds {[s for s in seeds if s not in panel['runs'].get(name, {})]}"
        for rate, panel in panels.items()
        for name in CELLS
        if [s for s in seeds if s not in panel["runs"].get(name, {})]
    ]
    if missing:
        msg = "panel is incomplete, so no reading is available: " + "; ".join(missing)
        raise ValueError(msg)


def common_seeds(
    panels: dict[str, dict[str, Any]],
    seeds: tuple[int, ...] = SEEDS,
) -> tuple[int, ...]:
    """Return the seeds present in every cell at both rates."""
    return tuple(
        s
        for s in seeds
        if all(s in panel["runs"].get(name, {}) for panel in panels.values() for name in CELLS)
    )


def efficiency(
    panels: dict[str, dict[str, Any]],
    tmp_dir: Path,
    seeds: tuple[int, ...] = SEEDS,
) -> dict[str, dict[str, Any]]:
    """Score each (rate, width) pair through the committed efficiency harness, unmodified."""
    reports: dict[str, dict[str, Any]] = {}
    for rate, panel in panels.items():
        reports[rate] = {}
        for width in WIDTHS:
            manifest = rw.write_manifest(
                panel,
                width,
                tmp_dir / f"_efficiency_{rate}_{width}.txt",
                seeds,
            )
            reports[rate][width] = eff.analyse(manifest)
    return reports


def contrasts(
    cells: dict[str, dict[str, dict[int, float]]],
    seeds: tuple[int, ...] = SEEDS,
) -> dict[str, Any]:
    """L.1's contrasts at the matched rate, L.1's own at the committed rate, and the three-way."""
    at = rw.contrasts(cells[MATCHED], seeds)
    committed = rw.contrasts(cells[COMMITTED], seeds)
    common = [s for s in at["seeds"] if s in committed["seeds"]]
    three_way = [
        committed["interaction"]["per_seed_deltas"][committed["seeds"].index(s)]
        - at["interaction"]["per_seed_deltas"][at["seeds"].index(s)]
        for s in common
    ]
    at["three_way"] = {
        "role": "secondary",
        "definition": "interaction at 0.001 minus interaction at 0.0001, per seed",
        **rw._two_sided(three_way),
    }
    at["committed_rate_reference"] = {
        "role": "reference",
        "note": "L.1's panel re-scored through the same call; outside the family",
        "interaction": committed["interaction"],
        "cell_means": committed["cell_means"],
        "per_width_wiring_effect": committed["per_width_wiring_effect"],
    }
    at["wide_wiring_effect_at_matched_rate"] = {
        "role": "reference",
        "note": "committed in Logbook 067; outside the family",
        "mean_delta": at["per_width_wiring_effect"].get("wide"),
    }
    return at


def gates(panel: dict[str, Any], seeds: tuple[int, ...] = SEEDS) -> dict[str, Any]:
    """Each 0.0001 learning arm against its own-width floor, one-sided. Read FIRST.

    The priors are not re-run: the floors are L.1's, whose committed prior tests (q = 0.462 at both
    widths) stand.
    """
    runs = panel["runs"]

    def foods(name: str) -> dict[int, float]:
        return {s: r.foods for s, r in runs[name].items() if s in seeds}

    out: dict[str, Any] = {}
    for arm in LEARNING_ARMS:
        out[f"{arm}_gate"] = {
            "arms": [arm, f"{arm}_frozen"],
            "kind": "gate",
            **ms.shift_contrast(foods(arm), foods(f"{arm}_frozen")),
        }
    out["prior_note"] = (
        "the floors are L.1's and its committed priors stand (no pre-update difference detected, "
        "q = 0.462 at both widths); not re-tested here"
    )
    return out


_FAMILY_CONTRASTS = ("interaction", "width_main_effect", "wiring_main_effect", "three_way")


def adjust_family(primary: dict[str, Any], gate_result: dict[str, Any]) -> dict[str, Any]:
    """Correct all eight registered tests under one BH-FDR family, in place."""
    labels = [*_FAMILY_CONTRASTS, *(f"{a}_gate" for a in LEARNING_ARMS)]
    ps = [float(primary[c]["p_two_sided"]) for c in _FAMILY_CONTRASTS] + [
        float(gate_result[f"{a}_gate"]["p_improve"]) for a in LEARNING_ARMS
    ]
    qs = [float(q) for q in ms.bh_fdr(ps)]
    for label, q in zip(labels, qs, strict=True):
        target = primary[label] if label in _FAMILY_CONTRASTS else gate_result[label]
        target["q"] = q
    gate_result["gates_pass"] = bool(
        all(gate_result[f"{a}_gate"]["q"] <= ms.SIG_Q for a in LEARNING_ARMS),
    )
    return {
        "labels": labels,
        "raw_p": ps,
        "q": qs,
        "n_tests": len(labels),
        "note": (
            "eight registered tests under ONE BH-FDR family: the interaction at 0.0001, both main "
            "effects there, the three-way, and four learning gates"
        ),
    }


def reading(
    gate_result: dict[str, Any],
    contrast_result: dict[str, Any],
    *,
    n_common: int,
    min_seeds: int = MIN_SEEDS,
) -> dict[str, Any]:
    """Assign the registered reading: gates first, then significance, then the minimum."""
    interaction = contrast_result["interaction"]
    delta = float(interaction["mean_delta"]) if n_common else 0.0
    q = float(interaction.get("q", interaction.get("p_two_sided", 1.0))) if n_common else 1.0
    clears = abs(delta) >= MIN_RETAIN
    shrunk = False
    if n_common < min_seeds:
        name = "insufficient_seeds"
    elif not gate_result["gates_pass"]:
        name = "no_learning"
    elif q > ms.SIG_Q:
        name = "pool_effect_is_rate_specific"
    elif delta > 0.0 and clears:
        name = "pool_effect_survives_the_rate"
    elif delta > 0.0:
        name = "pool_effect_is_rate_specific"
        shrunk = True
    else:
        name = "width_favours_the_shuffle_at_this_rate"
    return {
        "reading": name,
        "why": READINGS[name],
        "gates_read_first": True,
        "interaction_delta": delta,
        "interaction_q": q,
        "minimum_retained": MIN_RETAIN,
        "clears_minimum": bool(clears),
        "shrunk_below_half": bool(shrunk),
        "fraction_of_l1_interaction_retained": float(delta / L1_INTERACTION),
        "sign_flip_reproduced": False,
        "sign_flip_note": REGISTERED["sign_flip_known_absent"],
    }


def sensitivity(deltas: list[float], n_pairs: int) -> dict[str, Any]:
    """Report the realised spread against the registered expectation, and the power at the minimum."""
    if len(deltas) < _MIN_FOR_SPREAD:
        return {"available": False, "registered": dict(REGISTERED)}
    sd = float(np.std(deltas, ddof=1))
    se = sd / float(np.sqrt(n_pairs))
    return {
        "available": True,
        "n_pairs": n_pairs,
        "observed_sd": sd,
        "standard_error": se,
        "detectable_at_80_percent": float(2.80 * se),
        "power_at_minimum_normal_approx": float(fa._power(MIN_RETAIN, se)),
        "registered": dict(REGISTERED),
    }


def metrics_agree(primary: dict[str, Any], censored: dict[str, Any], *, censored_hib: bool) -> bool:
    """Orient the censored metric before comparing directions."""
    if not primary["n_common"]:
        return True
    sign = 1.0 if censored_hib else -1.0
    return bool(
        primary["interaction"]["mean_delta"] * censored["interaction"]["mean_delta"] * sign >= 0.0,
    )


def analyse(  # noqa: PLR0913 - three campaign directories and a strict flag are distinct inputs
    campaign_dir: Path,
    wide_rate_dir: Path,
    baseline_dir: Path,
    seeds: tuple[int, ...] = SEEDS,
    experiments: Path = EXPERIMENTS,
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """Score the 2x2 at 0.0001 beside L.1's at 0.001: gates first, then the interaction."""
    panels = assemble(
        scan(campaign_dir, experiments),
        fa.scan_rate_matched_wide(wide_rate_dir, experiments),
        rw.scan(baseline_dir, experiments),
    )
    if strict:
        require_complete(panels, seeds)
    seeds = common_seeds(panels, seeds)
    reports = efficiency(panels, campaign_dir, seeds)
    primary_cells = {r: rw.cell_values(reports[r], PRIMARY_METRIC) for r in RATES}
    censored_cells = {r: rw.cell_values(reports[r], CENSORED_METRIC) for r in RATES}
    primary = contrasts(primary_cells, seeds)
    censored = contrasts(censored_cells, seeds)
    gate_result = gates(panels[MATCHED], seeds)
    family = adjust_family(primary, gate_result)
    verdict = reading(gate_result, primary, n_common=primary["n_common"])
    censored_hib = bool(
        reports[MATCHED]["pooled"]["metrics"][CENSORED_METRIC]["higher_is_better"],
    )
    return {
        "seeds": list(seeds),
        "n_seeds_scored": len(seeds),
        "primary_metric": PRIMARY_METRIC,
        "censored_metric": CENSORED_METRIC,
        "censored_metric_higher_is_better": censored_hib,
        "metric_note": METRIC_NOTE,
        "sources": {
            "pooled_at_1e-4": str(campaign_dir),
            "wide_at_1e-4": f"{wide_rate_dir} (L.4's rate-matched baseline)",
            "floors_and_committed_panel": f"{baseline_dir} (L.1)",
        },
        "gates": gate_result,
        "primary": primary,
        "censored_axis": censored,
        "censoring": {r: rw.censoring(reports[r]) for r in RATES},
        "metrics_agree_in_direction": metrics_agree(primary, censored, censored_hib=censored_hib),
        "family": family,
        "sensitivity": sensitivity(primary["interaction"]["per_seed_deltas"], primary["n_common"]),
        "registered": dict(REGISTERED),
        **verdict,
        "efficiency": reports,
    }


def _print(result: dict[str, Any]) -> None:
    """Print the gates, both 2x2s, the interaction, the three-way and the reading."""
    p = result["primary"]
    print("\n" + "=" * 78)
    print("L.1b - L.1's interaction re-read at the rate-matched 0.0001, hard350")
    print("=" * 78)
    print(f"\nseeds scored: {result['n_seeds_scored']}")
    print("\nGATES (each 0.0001 arm against its own-width floor, one-sided):")
    for arm in LEARNING_ARMS:
        g = result["gates"][f"{arm}_gate"]
        print(
            f"  {arm:10s} +{g['effect']:.3f} foods  q={g['q']:.3f}  {g['positive_seeds']}/{g['n']}",
        )
    for rate, means in (
        (MATCHED, p["cell_means"]),
        (COMMITTED, p["committed_rate_reference"]["cell_means"]),
    ):
        print(f"\n2x2 at {rate} on {PRIMARY_METRIC}:")
        for w in WIRINGS:
            print(f"  {w}: pooled {means[f'{w}_pooled']:.4f}  wide {means[f'{w}_wide']:.4f}")
    for label in _FAMILY_CONTRASTS:
        c = p[label]
        print(
            f"  {label:20s} {c['mean_delta']:+.4f}  CI[{c['ci_lo']:+.4f},{c['ci_hi']:+.4f}]  "
            f"q={c['q']:.3f}  +{c['positive_seeds']}/{p['n_common']}",
        )
    ref = p["committed_rate_reference"]["interaction"]
    print(f"  L.1 reference interaction at 0.001: {ref['mean_delta']:+.4f}")
    s = result["sensitivity"]
    if s.get("available"):
        print(
            f"\nsensitivity: sd {s['observed_sd']:.4f}  se {s['standard_error']:.4f}  "
            f"detectable {s['detectable_at_80_percent']:.4f}  power@{MIN_RETAIN} "
            f"{s['power_at_minimum_normal_approx']:.2f}",
        )
    print(
        f"\nREADING: {result['reading'].upper().replace('_', '-')}"
        + ("  (shrunk below half)" if result["shrunk_below_half"] else ""),
    )
    print(f"  {result['why']}")
    if not result["metrics_agree_in_direction"]:
        print("  NOTE: the two metrics disagree in direction; reported, not resolved")


def write_csv(result: dict[str, Any], path: Path) -> None:
    """One row per rate, width, wiring and seed, with the censoring column."""
    metrics = (
        "auc_foods",
        "auc_success",
        "episodes_to_30pct_success",
        "episodes_to_90pct_foods_plateau",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rate", "width", "wiring", "seed", *metrics, "primary_censored"])
        for rate in RATES:
            for width in WIDTHS:
                report = result["efficiency"][rate][width]
                horizon = report["horizon_episodes"]
                for wiring, arm in (("wt", eff._WILD), ("rn", eff._REWIRED)):
                    for seed, row in sorted(
                        report["per_seed"][arm].items(),
                        key=lambda kv: int(kv[0]),
                    ):
                        writer.writerow(
                            [
                                rate,
                                width,
                                wiring,
                                seed,
                                *(f"{row[m]:.6f}" for m in metrics),
                                int(row[CENSORED_METRIC] >= horizon),
                            ],
                        )


def main(argv: list[str] | None = None) -> int:
    """Score the 2x2 at 0.0001 and report the reading."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True, help="the pooled 0.0001 arms")
    parser.add_argument(
        "--wide-rate",
        type=Path,
        default=Path("campaigns/feature-ablations"),
        help="L.4's rate-matched wide learning arms at 0.0001",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("campaigns/readout-width"),
        help="L.1's panel: the four floors and the 0.001 learning arms",
    )
    parser.add_argument("--seeds", type=str, default=f"{SEEDS[0]}-{SEEDS[-1]}")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="score what is present; for a pilot, never for a registered panel",
    )
    args = parser.parse_args(argv)
    low, _, high = args.seeds.partition("-")
    seeds = tuple(range(int(low), int(high or low) + 1))
    result = analyse(
        args.campaign,
        args.wide_rate,
        args.baseline,
        seeds,
        strict=not args.allow_incomplete,
    )
    _print(result)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(result, indent=2, sort_keys=True, default=rw._jsonable) + "\n",
        )
        print(f"\nwrote {args.out}")
    if args.csv:
        write_csv(result, args.csv)
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
