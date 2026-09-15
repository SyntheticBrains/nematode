#!/usr/bin/env python
"""L.0's reading: whether the wiring is legible to a learner that READS it, not writes it.

Phase 7's flagship asked whether the wild-type connectome becomes load-bearing under a rule the
animal could host. For rules that **write** the wiring that is answered, twice over: R.1c returned
`not_reducible` at every perturbation dimension, and R.2 found e-prop reaching competence with the
chemical matrix **frozen**, every arm that writes it doing worse by 3.9 to 15.9 foods.

This asks the form left over. Under ``readout_only`` the chemical matrix is frozen and only the 2x4
motor readout learns, by its own exact gradient -- so the connectome enters as a **fixed feature
map**, and a degree-preserving rewiring changes that map and nothing else. The question is whether
the wild-type edges compute better four-dimensional features for this task than a degree-matched
shuffle of the same edges.

**The committed harnesses are read-only here.** ``connectome_structure_efficiency`` supplies the four
efficiency metrics and their BH-FDR family, unchanged; ``wiring_premise`` is NOT imported at all --
it hard-codes its test family per cell and carries block V's committed verdicts for three of them, so
adding a fourth there would edit a harness whose output is already on the record. The gates and the
prior check live here instead, reusing the same statistics layer.

Three checks travel with the primary, as they did in V.1 and V.3, and two of them can void it:

* **two learning gates** -- each wiring against its OWN frozen floor. A contrast between two arms
  that did not learn is not a wiring result;
* **the untrained prior** -- wild-type frozen against rewired frozen, MEASURED HERE. V.1's -0.17 and
  V.3's -0.01 are reported beside it as context only: those floors were PPO-configured at an action
  std of 1.0 where these run at 0.368, and carrying one regime's figure into another as established
  is the cross-regime comparison this change's own design forbids;
* **credited drift on w_chem, which must read 0.00 for both wirings** -- the check that the substrate
  really was frozen.

The power arithmetic is carried as a field rather than left in the registration, because a null here
closes the phase: at 16 pairs a one-sided sign test needs 12/16 positive, and the comparator's own
per-seed win rate was 21-26 of 32, whose midpoint is 73.4% -- so 16 seeds would have had **57.3%**
power there against **79.2%** at 32 pairs. Sign-test planning figures, not the registered procedure's
power; see ``power``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import connectome_structure_efficiency as eff  # noqa: E402  # pyright: ignore[reportMissingImports]
import l4_mixture_statistic as ms  # noqa: E402  # pyright: ignore[reportMissingImports]
import l4_reduced_perturbation as rp  # noqa: E402  # pyright: ignore[reportMissingImports]
from l4_panel import EXPERIMENTS, read_log  # noqa: E402  # pyright: ignore[reportMissingImports]

SEEDS = tuple(range(1, 33))
_STEM = "connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_hard350_eprop"

# The four arms, and the two names the efficiency script requires of the pair it scores.
ARMS = {
    "wt_learning": {"wiring": "wild type", "learns": True, "efficiency_arm": eff._WILD},
    "rn_learning": {"wiring": "rewired null", "learns": True, "efficiency_arm": eff._REWIRED},
    "wt_frozen": {"wiring": "wild type", "learns": False, "efficiency_arm": None},
    "rn_frozen": {"wiring": "rewired null", "learns": False, "efficiency_arm": None},
}
_LABEL = re.compile(
    rf"^{re.escape(_STEM)}_(?P<arm>readout_only|frozen)(?P<rewired>_rewired_null)?"
    r"-seed(?P<seed>\d+)\.log$",
)

# Block V's registered minimum on time-to-competence, and its primary metric.
MIN_GAIN_FRACTION = 0.20
PRIMARY_METRIC = "episodes_to_30pct_success"
# V.3 on this exact cell, under PPO, over 32 paired seeds. Context for the reading, never a
# quantitative delta: the two run under different learning regimes.
V3_WILD_EPISODES = 892.0
V3_REWIRED_EPISODES = 1165.0
V3_GAIN_FRACTION = 0.235
V3_WIN_RATE_RANGE = (21 / 32, 26 / 32)
# The untrained prior, as block V measured it under PPO-configured floors. Context only.
PRIOR_REFERENCES = {"V.1 thermal": (-0.17, 0.735), "V.3 hard_food": (-0.01, 0.841)}


def scan(campaign_dir: Path, experiments: Path = EXPERIMENTS) -> dict[str, Any]:
    """Read every registered run under a campaign directory, keyed by arm."""
    log_dir = campaign_dir / "logs" if (campaign_dir / "logs").is_dir() else campaign_dir
    runs: dict[str, dict[int, Any]] = {name: {} for name in ARMS}
    logs: dict[str, dict[int, Path]] = {}
    for log in sorted(log_dir.glob("*.log")):
        match = _LABEL.match(log.name)
        if match is None:
            print(f"  WARN: skipping log with an unrecognised label: {log.name}")
            continue
        prefix = "rn" if match.group("rewired") else "wt"
        suffix = "learning" if match.group("arm") == "readout_only" else "frozen"
        name = f"{prefix}_{suffix}"
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


def require_complete(scanned: dict[str, Any], seeds: tuple[int, ...] = SEEDS) -> None:
    """Refuse to score a campaign missing any registered cell.

    The efficiency script fails fast on unpaired seeds for its own reason -- an unpaired run shrinks
    the shared horizon and distorts every metric -- and this catches the same thing earlier, across
    all four arms rather than the two it scores.
    """
    runs = scanned["runs"]
    missing = [
        f"{name} seeds {[s for s in seeds if s not in runs[name]]}"
        for name in ARMS
        if [s for s in seeds if s not in runs[name]]
    ]
    if missing:
        msg = "campaign is incomplete, so no verdict is available: " + "; ".join(missing)
        raise ValueError(msg)


def write_manifest(scanned: dict[str, Any], path: Path, seeds: tuple[int, ...] = SEEDS) -> Path:
    """Write the paired manifest the efficiency script consumes.

    ``arm seed out_path`` lines, with the arm names fixed at ``wild_type`` and ``rewired_null`` --
    the script requires exactly those and fails fast on any unpaired seed. Only the two LEARNING arms
    go in: the floors are the gates' business, not the efficiency contrast's. The paths are the
    campaign's own run logs, which the committed parser reads directly.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for name, meta in ARMS.items():
        arm = meta["efficiency_arm"]
        if arm is None:
            continue
        for seed in seeds:
            log = scanned["logs"].get(name, {}).get(seed)
            if log is None:
                msg = f"no log for {name} seed {seed}; the manifest would be unpaired"
                raise ValueError(msg)
            # The efficiency script resolves each entry as ``REPO / out``, so the manifest
            # carries repo-relative paths. ``log`` is relative when the campaign directory
            # was given relatively, so resolve before relativising.
            lines.append(f"{arm} {seed} {log.resolve().relative_to(rp.ps.hm.REPO)}")
    path.write_text("\n".join(lines) + "\n")
    return path


def gates(scanned: dict[str, Any], seeds: tuple[int, ...] = SEEDS) -> dict[str, Any]:
    """Each wiring's learner against its OWN frozen floor, plus the untrained prior.

    Read BEFORE the contrast. A contrast against a null presupposes that both arms learned, and a
    prior that separates between wirings would mean the rewiring changed the substrate before any
    learning did -- either makes the primary uninterpretable, so both can void it.
    """
    runs = scanned["runs"]

    def foods(name: str) -> dict[int, float]:
        return {s: r.foods for s, r in runs[name].items() if s in seeds}

    tests = {
        "wt_gate": ("wt_learning", "wt_frozen", "gate"),
        "rn_gate": ("rn_learning", "rn_frozen", "gate"),
        "prior": ("wt_frozen", "rn_frozen", "prior"),
    }
    out: dict[str, Any] = {}
    for label, (a, b, kind) in tests.items():
        contrast = ms.shift_contrast(foods(a), foods(b))
        out[label] = {"arms": [a, b], "kind": kind, **contrast}
    # The gates are one-sided -- an arm is asked to beat its own floor. The PRIOR is two-sided:
    # either wiring being ahead before any learning is the same problem. A two-sided p from an exact
    # one-sided pair is 2 * min(one-sided), capped at 1; the earlier `min(...)` alone was not a
    # p-value and doubled the type-I rate on the check that can void the campaign.
    prior_two_sided = min(
        1.0,
        2.0 * min(out["prior"]["p_improve"], out["prior"]["p_degrade"]),
    )
    out["prior"]["p_two_sided"] = float(prior_two_sided)
    family = [out["wt_gate"]["p_improve"], out["rn_gate"]["p_improve"], prior_two_sided]
    qs = ms.bh_fdr(family)
    for label, q in zip(tests, qs, strict=True):
        out[label]["q_improve"] = float(q)
    out["gates_pass"] = bool(
        out["wt_gate"]["q_improve"] <= ms.SIG_Q and out["rn_gate"]["q_improve"] <= ms.SIG_Q,
    )
    # Read off the CORRECTED q of that two-sided p, so the prior is held to the same family as the
    # gates it sits beside rather than to a raw threshold.
    out["prior_separates"] = bool(out["prior"]["q_improve"] <= ms.SIG_Q)
    out["prior_reference"] = dict(PRIOR_REFERENCES)
    out["prior_reference_note"] = (
        "measured here, not inherited: V.1's and V.3's priors were measured on PPO-configured floors "
        "at an action std of 1.0 where these run at 0.368"
    )
    return out


def drift(
    scanned: dict[str, Any],
    seeds: tuple[int, ...] = SEEDS,
    experiments: Path = EXPERIMENTS,
) -> dict[str, Any]:
    """Credited drift on ``w_chem``, which must read 0.00: the substrate is supposed to be frozen."""
    out: dict[str, Any] = {}
    for name in ("wt_learning", "rn_learning"):
        floor = name.replace("_learning", "_frozen")
        values: list[float] = []
        for seed in seeds:
            learning = scanned["logs"].get(name, {}).get(seed)
            frozen = scanned["logs"].get(floor, {}).get(seed)
            if learning is None or frozen is None:
                continue
            a = rp._chemical_weights(learning, experiments)
            b = rp._chemical_weights(frozen, experiments)
            if a is None or b is None or a.shape != b.shape:
                continue
            base = b.float().norm()
            values.append(float((a - b).float().norm() / (base or 1.0)))
        out[name] = {
            "mean_relative": float(np.mean(values)) if values else float("nan"),
            "n_read": len(values),
            "available": bool(values),
        }
    # Complete evidence, not merely available evidence: with one arm's checkpoints missing, `all()`
    # over a short list would report the substrate frozen on the strength of whatever happened to be
    # on disk. Both arms must be read at every requested seed.
    complete = all(out[name]["n_read"] == len(seeds) for name in ("wt_learning", "rn_learning"))
    read = [out[name]["mean_relative"] for name in ("wt_learning", "rn_learning")]
    out["drift_evidence_complete"] = bool(complete)
    out["substrate_frozen"] = bool(complete and all(v < 1e-6 for v in read))
    return out


def power(n_pairs: int) -> dict[str, Any]:
    """Compute the sign-test arithmetic, carried because a null here closes the phase.

    These are **planning figures for a sign test**, not the power of the registered procedure. That
    procedure is a paired rank test corrected across four metrics under BH-FDR, and the two differ in
    both directions -- the rank test uses magnitudes where the sign test uses only signs, while the
    correction across a family costs power that a single test does not pay. Computing the registered
    procedure's power would need an explicit alternative and a distributional assumption for the
    effect, neither of which this design has. So these size the seed count and are quoted as that.
    """
    from math import comb

    def p_at(k: int) -> float:
        return sum(comb(n_pairs, i) for i in range(k, n_pairs + 1)) / 2**n_pairs

    low, high = V3_WIN_RATE_RANGE
    k_needed = next((k for k in range(n_pairs + 1) if p_at(k) <= ms.SIG_Q), None)
    if k_needed is None:
        # Below about five pairs the smallest p an exact one-sided test can return -- 2**-n, even
        # with every seed positive -- is already above the gate. No k reaches it, so the design has
        # no power at any effect size, which is a statement about the design and not an error. A
        # pilot lands here by construction; a registered campaign never should.
        return {
            "n_pairs": n_pairs,
            "k_needed": None,
            "gate_reachable": False,
            "smallest_reachable_p": p_at(n_pairs),
            "power_against_comparator": {},
            "comparator_win_rate_range": [low, high],
            "note": (
                f"no power at any effect size: with {n_pairs} pairs the smallest one-sided p is "
                f"{p_at(n_pairs):.4f}, above the {ms.SIG_Q} gate even with every seed positive"
            ),
        }
    return {
        "n_pairs": n_pairs,
        "k_needed": k_needed,
        "gate_reachable": True,
        "k_needed_fraction": k_needed / n_pairs,
        "power_against_comparator": {
            f"{q:.0%}": sum(
                comb(n_pairs, i) * q**i * (1 - q) ** (n_pairs - i)
                for i in range(k_needed, n_pairs + 1)
            )
            for q in (low, (low + high) / 2, high)
        },
        "comparator_win_rate_range": [low, high],
        "note": (
            "sign-test planning figures, not the registered procedure's power: that is a "
            "paired rank test under BH-FDR across four metrics, which differs in both directions"
        ),
    }


def analyse(
    scanned: dict[str, Any],
    manifest: Path,
    seeds: tuple[int, ...] = SEEDS,
    experiments: Path = EXPERIMENTS,
) -> dict[str, Any]:
    """Score the contrast, read the gates first, and apply the registered reading."""
    checks = gates(scanned, seeds)
    substrate = drift(scanned, seeds, experiments)
    efficiency = eff.analyse(manifest)
    reach = power(len(seeds))
    primary = efficiency["metrics"][PRIMARY_METRIC]
    # Fewer episodes is better, and the script orients its delta so positive means the wild type is
    # faster. The registered bar is a FRACTION off the null's time, which is what block V reported.
    gain = (
        primary["mean_delta"] / primary["rewired_mean"] if primary["rewired_mean"] else float("nan")
    )
    significant = bool(primary["bh_fdr_q"] <= ms.SIG_Q and primary["mean_delta"] > 0)
    clears_bar = bool(significant and gain >= MIN_GAIN_FRACTION)

    if not reach["gate_reachable"]:
        # A pilot, by seed count. Every gate reads as failed because none COULD pass -- the smallest
        # p an exact one-sided test can return at this n is already above the gate -- so reporting
        # `void` on gate grounds would be announcing a gate failure that the design made
        # unavoidable. The per-arm levels are the pilot's job; the reading is not. (R.2's harness
        # had to learn the same lesson, where it printed `does_not_learn -- the programme stops`
        # from four seeds.)
        verdict, reason = (
            "pilot",
            (
                f"a pilot on {len(seeds)} seeds, which is not a registered verdict: {reach['note']}, "
                "so no gate could have passed and no contrast could have reached the bar whatever "
                "the arms did. Read the levels and the frozen-substrate check, not the reading"
            ),
        )
    elif not checks["gates_pass"] or checks["prior_separates"] or not substrate["substrate_frozen"]:
        why = (
            "a learning gate failed"
            if not checks["gates_pass"]
            else (
                "the untrained prior separates between wirings"
                if checks["prior_separates"]
                else (
                    "the frozen-substrate check is incomplete"
                    if not substrate["drift_evidence_complete"]
                    else "the substrate did not stay frozen"
                )
            )
        )
        verdict, reason = (
            "void",
            (
                f"{why}, so the contrast is uninterpretable: the arms did not learn, or the rewiring "
                "changed the substrate before learning did, or the substrate this was supposed to "
                "hold fixed moved -- in which case it is not a fixed-features contrast at all"
            ),
        )
    elif clears_bar:
        verdict, reason = (
            "wiring_is_legible",
            (
                f"the wild type reaches competence {gain:.1%} sooner than its degree-preserving "
                f"rewired null (q = {primary['bh_fdr_q']:.3f}), above the registered "
                f"{MIN_GAIN_FRACTION:.0%} minimum. The wiring supplies better fixed features to a "
                "small learned readout than a degree-matched shuffle does -- a PERFORMANCE claim, "
                "and not D2's primary, which needs the wiring itself to be plastic"
            ),
        )
    elif significant:
        verdict, reason = (
            "below_bar",
            (
                f"a significant advantage of {gain:.1%} (q = {primary['bh_fdr_q']:.3f}) below the "
                f"registered {MIN_GAIN_FRACTION:.0%} minimum: suggestive with the bar unmet"
            ),
        )
    else:
        verdict, reason = (
            "wiring_is_inert_as_features",
            (
                "no significant advantage at the registered bar: 034's degree-statistics verdict "
                "extends to a third learning regime. The wiring is endpoint-inert under gradient "
                "learning, harmful under local rules that write it, and indistinguishable from a "
                "degree-matched shuffle as fixed features"
            ),
        )
    return {
        "verdict": verdict,
        "why": reason,
        "primary_metric": PRIMARY_METRIC,
        "gain_fraction": gain,
        "clears_registered_bar": clears_bar,
        "minimum_gain_fraction": MIN_GAIN_FRACTION,
        "gates": checks,
        "substrate_drift": substrate,
        "efficiency": efficiency,
        "power": reach,
        "comparator": {
            "source": "V.3, the same contrast on this cell under PPO, 32 paired seeds",
            "wild_episodes": V3_WILD_EPISODES,
            "rewired_episodes": V3_REWIRED_EPISODES,
            "gain_fraction": V3_GAIN_FRACTION,
            "note": (
                "context, never a quantitative delta: the two run under different learning regimes "
                "and the project's commensurability rule forbids treating cross-regime deltas "
                "quantitatively"
            ),
        },
        "satisfies_d2_primary": False,
        "d2_note": (
            "D2's primary requires PLASTIC wild-type to beat PLASTIC rewired-null. This learner "
            "leaves the wiring frozen, so no result here can satisfy it or convert Phase 7's SPLIT "
            "into a GO."
        ),
    }


def _print(result: dict[str, Any]) -> None:
    """Print the gates first, then the contrast, then what the reading turns on."""
    checks = result["gates"]
    print("\nL.0 - the wiring as fixed features")
    print("  gates, read before the contrast:")
    for label in ("wt_gate", "rn_gate", "prior"):
        row = checks[label]
        print(
            f"    {label:8} {row['arms'][0]} - {row['arms'][1]}: "
            f"{row.get('effect', float('nan')):+.3f} foods, q = {row['q_improve']:.3f}",
        )
    print(f"    gates pass: {checks['gates_pass']}   prior separates: {checks['prior_separates']}")
    print(f"    prior reference (context only): {checks['prior_reference']}")
    drifts = result["substrate_drift"]
    print(
        f"  substrate frozen: {drifts['substrate_frozen']} "
        f"(w_chem drift wt {drifts['wt_learning']['mean_relative']:.3g}, "
        f"rn {drifts['rn_learning']['mean_relative']:.3g})",
    )
    print("\n  efficiency metrics (positive = wild type better):")
    for name, entry in result["efficiency"]["metrics"].items():
        marker = " <- primary" if name == result["primary_metric"] else ""
        print(
            f"    {name:32} wild {entry['wild_mean']:>9.2f}  rewired {entry['rewired_mean']:>9.2f}  "
            f"delta {entry['mean_delta']:>+9.3f}  q {entry['bh_fdr_q']:.3f}  "
            f"wild-better {entry['wild_better_seeds']}/{result['efficiency']['n_paired_seeds']}{marker}",
        )
    print(
        f"\n  time-to-competence gain: {result['gain_fraction']:.1%} against the registered "
        f"{result['minimum_gain_fraction']:.0%} minimum",
    )
    comparator = result["comparator"]
    print(
        f"  comparator (context, not a delta): V.3 under PPO, {comparator['wild_episodes']:.0f} "
        f"against {comparator['rewired_episodes']:.0f} episodes, {comparator['gain_fraction']:.1%}",
    )
    p = result["power"]
    if p["gate_reachable"]:
        print(
            f"  power: {p['n_pairs']} pairs, {p['k_needed']}/{p['n_pairs']} needed; against the "
            f"comparator's win rate {p['power_against_comparator']}",
        )
    else:
        print(f"  power: {p['note']}")
    print(f"\nVERDICT: {result['verdict']} - {result['why']}")
    print(f"\n{result['d2_note']}")


def write_csv(result: dict[str, Any], path: Path) -> None:
    """One row per arm and seed, so the table can be recomputed from the record.

    The censored entries matter here: ``episodes_to_30pct_success`` is right-censored at the horizon,
    and a reader has to be able to see which rows are censored rather than take the means on trust.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    per_seed = result["efficiency"]["per_seed"]
    horizon = result["efficiency"]["horizon_episodes"]
    metrics = sorted(next(iter(per_seed[eff._WILD].values())))
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["arm", "seed", *metrics, "primary_censored"])
        for arm in (eff._WILD, eff._REWIRED):
            for seed in sorted(per_seed[arm], key=int):
                row = per_seed[arm][seed]
                writer.writerow(
                    [
                        arm,
                        seed,
                        *(f"{row[m]:.6f}" for m in metrics),
                        int(row[PRIMARY_METRIC] >= horizon),
                    ],
                )


def _jsonable(value: object) -> object:
    """Replace not-a-number with null, recursively, so the record is strict JSON."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main(argv: list[str] | None = None) -> int:
    """Score L.0's campaign and print its reading."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--seeds", type=str, default="1-32")
    parser.add_argument("--experiments", type=Path, default=EXPERIMENTS)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="score what is present; for a pilot, never for a registered campaign",
    )
    args = parser.parse_args(argv)

    low, _, high = args.seeds.partition("-")
    seeds = tuple(range(int(low), int(high or low) + 1))
    scanned = scan(args.campaign, args.experiments)
    if not args.allow_incomplete:
        require_complete(scanned, seeds)
    manifest = write_manifest(
        scanned,
        args.manifest or (args.campaign / "efficiency-manifest.txt"),
        seeds,
    )
    result = analyse(scanned, manifest, seeds, args.experiments)
    result["protocol"] = {"seeds": list(seeds), "manifest": str(manifest)}
    _print(result)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(_jsonable(result), indent=2, sort_keys=True, allow_nan=False) + "\n",
        )
    if args.csv:
        write_csv(result, args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
