"""Real-worm behavioural-chemotaxis validation - klinotaxis bias curves vs published *C. elegans*.

Reduces each seed's captured behavioural trajectory (``behaviour_capture.json`` from a
``capture_behaviour`` run) to its two klinotaxis bias statistics - the klinokinesis down/up
turn-rate ratio (Pierce-Shimomura et al. 1999) and the klinotaxis weathervane slope (Iino &
Yoshida 2009) - then grades each across seeds against the behaviour-level literature reference
(§3) with an 80% bootstrap CI as REPRODUCED / PARTIAL / ABSENT.

The reference is behaviour-level (bias direction + a reported magnitude range + citation), NOT a
pixel digitization of the original figures; the weathervane slope is a sign-only reference (see the
reference notes). Both bias curves + the pooled statistics + the verdicts are written to a summary
JSON; the two figures are emitted when ``--figure-dir`` is given.

Usage::

    uv run python scripts/analysis/behavioural_chemotaxis_validation.py \
        --manifest <run-dir>/_manifest.txt --out <run-dir>/behavioural_curves.json \
        [--figure-dir <run-dir>/figures] [--theta-sharp 0.6] [--theta-percentile 85]

The manifest is ``<seed> <behaviour_capture.json>`` per line (``#`` comments / blanks skipped).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from quantumnematode.report.dtypes import BehaviourStep
from quantumnematode.validation.behavioural_agreement import grade_statistic
from quantumnematode.validation.behavioural_curves import (
    BiasCurve,
    curving_rate_vs_bearing,
    kinematics,
    klinokinesis_magnitude_ratio,
    klinokinesis_ratio,
    suggest_theta_sharp,
    turn_rate_vs_dcdt,
    weathervane_slope,
    weathervane_slope_all,
)
from quantumnematode.validation.datasets import load_bias_signatures

REPO = Path(__file__).resolve().parents[2]

# Each bias statistic: (reference key, strategy, kinematics-reducer, family). The thresholded and
# threshold-free reducers for one strategy are cross-checked against each other (§6 decision: the
# |dtheta| distribution saturates at the turn bound, so the thresholded sharp/gradual split has no
# natural cut; the threshold-free companion is theta_sharp-independent).
_STATISTICS = (
    ("klinokinesis", "klinokinesis", klinokinesis_ratio, "thresholded"),
    ("klinokinesis_magnitude", "klinokinesis", klinokinesis_magnitude_ratio, "threshold_free"),
    ("klinotaxis", "klinotaxis", weathervane_slope, "thresholded"),
    ("klinotaxis_all", "klinotaxis", weathervane_slope_all, "threshold_free"),
)
_STRATEGIES = ("klinokinesis", "klinotaxis")


def _steps_from_dicts(step_dicts: list[dict]) -> list[BehaviourStep]:
    """Rebuild ``BehaviourStep`` records from their serialised dicts."""
    return [BehaviourStep(**d) for d in step_dicts]


def load_manifest(manifest: Path) -> dict[int, list[list[BehaviourStep]]]:
    """Return ``{seed: [per-run step series]}`` from a ``<seed> <behaviour_capture.json>`` manifest.

    Blank / ``#``-comment lines are skipped; malformed lines and missing/duplicate seeds are
    reported so analysis issues stay traceable rather than silently dropped.
    """
    seeds: dict[int, list[list[BehaviourStep]]] = {}
    for raw in manifest.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2 or not parts[0].lstrip("-").isdigit():  # expected: `<int seed> <file>`
            print(f"  WARN: skipping malformed manifest line: {raw!r}")
            continue
        seed, capture_path = int(parts[0]), REPO / parts[1]
        if not capture_path.exists():
            print(f"  WARN seed {seed}: {capture_path} not found - dropped")
            continue
        if seed in seeds:
            print(f"  WARN seed {seed}: duplicate manifest entry - overwriting previous")
        data = json.loads(capture_path.read_text())
        runs = [_steps_from_dicts(run["steps"]) for run in data.get("runs", []) if run["steps"]]
        if not runs:
            print(f"  WARN seed {seed}: no captured steps in {capture_path} - dropped")
            continue
        seeds[seed] = runs
    return seeds


def tail_runs(
    seeds: dict[int, list[list[BehaviourStep]]],
    keep: int | None,
) -> dict[int, list[list[BehaviourStep]]]:
    """Keep only each seed's last ``keep`` runs (post-convergence tail); no-op when ``keep`` None."""
    if keep is None:
        return seeds
    return {seed: runs[-keep:] for seed, runs in seeds.items()}


def _resolve_theta_sharp(
    seeds: dict[int, list[list[BehaviourStep]]],
    theta_sharp: float | None,
    theta_percentile: float,
) -> float:
    """Use an explicit ``theta_sharp`` or calibrate it from the pooled |dtheta| distribution."""
    if theta_sharp is not None:
        return theta_sharp
    pooled = [step for runs in seeds.values() for run in runs for step in run]
    return suggest_theta_sharp(pooled, percentile=theta_percentile)


def _per_seed_statistics(
    runs: list[list[BehaviourStep]],
    theta_sharp: float,
) -> dict[str, float | None]:
    """One seed's four bias statistics (keyed by reference key), pooling per-run kinematics."""
    kin = [k for run in runs for k in kinematics(run, theta_sharp)]  # no cross-run transitions
    return {key: reducer(kin) for key, _strategy, reducer, _family in _STATISTICS}


def _combined_verdict(thresholded: str, threshold_free: str) -> str:
    """Reconcile a strategy's thresholded + threshold-free verdicts (§6 robustness cross-check).

    A direction is 'present' unless graded ABSENT. Agreement -> a robust present/absent call;
    disagreement -> equivocal (the point statistic is threshold-sensitive, only the direction is
    trustworthy).
    """
    present_t = thresholded != "ABSENT"
    present_f = threshold_free != "ABSENT"
    if present_t and present_f:
        both_strong = thresholded == "REPRODUCED" and threshold_free == "REPRODUCED"
        return "PRESENT" if both_strong else "PRESENT_PARTIAL"
    if not present_t and not present_f:
        return "ABSENT"
    return "EQUIVOCAL"


def _pooled_curves(
    seeds: dict[int, list[list[BehaviourStep]]],
    theta_sharp: float,
) -> tuple[BiasCurve, BiasCurve]:
    """Build the two bias curves over all seeds' kinematics (for the figures / visual reference)."""
    kin = [k for runs in seeds.values() for run in runs for k in kinematics(run, theta_sharp)]
    return turn_rate_vs_dcdt(kin), curving_rate_vs_bearing(kin)


def analyse(
    seeds: dict[int, list[list[BehaviourStep]]],
    theta_sharp: float,
) -> dict:
    """Grade all four bias statistics across seeds and print the per-strategy agreement table."""
    refs = load_bias_signatures()
    per_seed: dict[str, dict[int, float | None]] = {key: {} for key, *_ in _STATISTICS}
    for seed, runs in sorted(seeds.items()):
        stats = _per_seed_statistics(runs, theta_sharp)
        for key, value in stats.items():
            per_seed[key][seed] = value
            if value is None or not math.isfinite(value):
                print(f"  WARN seed {seed}: {key} not finite ({value}) - dropped from CI")

    graded = {
        key: grade_statistic(
            [v for v in per_seed[key].values() if v is not None and math.isfinite(v)],
            refs[key],
        )
        for key, *_ in _STATISTICS
    }
    family = {key: fam for key, _strategy, _reducer, fam in _STATISTICS}
    strategy_of = {key: strat for key, strat, _reducer, _fam in _STATISTICS}

    print("\n" + "=" * 78)
    print("REAL-WORM BEHAVIOURAL-CHEMOTAXIS VALIDATION - klinotaxis bias curves vs C. elegans")
    print(f"  n_seeds={len(seeds)}  theta_sharp={theta_sharp:.3f} rad")
    print("=" * 78)

    summary: dict = {"theta_sharp": theta_sharp, "n_seeds": len(seeds), "statistics": {}}
    for key, *_ in _STATISTICS:
        res = graded[key]
        usable = [round(v, 3) for v in per_seed[key].values() if v is not None and math.isfinite(v)]
        print(f"\n  {strategy_of[key]} / {res.statistic}  [{family[key]}]")
        print(
            f"    mean={res.mean:+.3f}  80% CI[{res.ci_lo:+.3f}, {res.ci_hi:+.3f}]  "
            f"n={res.n}  null={res.null_value:.1f}  VERDICT: {res.verdict.value}",
        )
        print(f"    per-seed usable={usable}")
        summary["statistics"][key] = {
            **res.to_dict(),
            "family": family[key],
            "strategy": strategy_of[key],
            "per_seed": {str(s): v for s, v in per_seed[key].items()},
        }

    # Per-strategy agreement: reconcile the thresholded + threshold-free verdicts.
    print("\n" + "-" * 78)
    summary["strategy_verdicts"] = {}
    for strategy in _STRATEGIES:
        keys = [key for key, strat, *_ in _STATISTICS if strat == strategy]
        thr = next(k for k in keys if family[k] == "thresholded")
        free = next(k for k in keys if family[k] == "threshold_free")
        combined = _combined_verdict(graded[thr].verdict.value, graded[free].verdict.value)
        summary["strategy_verdicts"][strategy] = {
            "combined": combined,
            "thresholded": graded[thr].verdict.value,
            "threshold_free": graded[free].verdict.value,
            "citation": graded[thr].citation,
        }
        print(
            f"  {strategy}: {combined}  "
            f"(thresholded={graded[thr].verdict.value}, threshold-free={graded[free].verdict.value})",
        )
    return summary


def _write_figures(
    seeds: dict[int, list[list[BehaviourStep]]],
    theta_sharp: float,
    summary: dict,
    figure_dir: Path,
) -> None:
    """Emit the two bias-curve figures (pooled curve + verdict annotation) into ``figure_dir``."""
    from quantumnematode.report.continuous_figures import (
        plot_turn_rate_curve,
        plot_weathervane_curve,
    )
    from quantumnematode.validation.behavioural_agreement import AgreementResult, Verdict

    figure_dir.mkdir(parents=True, exist_ok=True)
    turn_curve, weathervane_curve = _pooled_curves(seeds, theta_sharp)

    def _agreement(key: str) -> AgreementResult:
        d = summary["statistics"][key]
        return AgreementResult(
            statistic=d["statistic"],
            verdict=Verdict(d["verdict"]),
            mean=d["mean"],
            ci_lo=d["ci_lo"],
            ci_hi=d["ci_hi"],
            n=d["n"],
            null_value=d["null_value"],
            sign=d["sign"],
            magnitude_range=tuple(d["magnitude_range"]) if d["magnitude_range"] else None,
            citation=d["citation"],
        )

    plot_turn_rate_curve(
        turn_curve,
        figure_dir / "turn_rate_vs_dcdt.png",
        agreement=_agreement("klinokinesis"),
    )
    plot_weathervane_curve(
        weathervane_curve,
        figure_dir / "curving_rate_vs_bearing.png",
        agreement=_agreement("klinotaxis"),
    )
    print(f"\nwrote figures to {figure_dir}")


def main() -> None:
    """Load the manifest, grade both bias curves across seeds, and write the summary JSON."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="<seed> <behaviour_capture.json> per line",
    )
    ap.add_argument("--out", type=Path, default=None, help="write the summary JSON here")
    ap.add_argument("--figure-dir", type=Path, default=None, help="emit the two bias-curve figures")
    ap.add_argument(
        "--tail-runs",
        type=int,
        default=None,
        help="analyse only each seed's last N runs (the post-convergence tail); default all",
    )
    ap.add_argument(
        "--theta-sharp",
        type=float,
        default=None,
        help="explicit sharp-turn threshold (rad); default calibrates from |dtheta|",
    )
    ap.add_argument(
        "--theta-percentile",
        type=float,
        default=85.0,
        help="percentile of |dtheta| used to calibrate theta_sharp when not given",
    )
    args = ap.parse_args()

    seeds = load_manifest(args.manifest)
    if not seeds:
        print("No usable seeds - nothing to analyse.")
        return
    seeds = tail_runs(seeds, args.tail_runs)
    theta_sharp = _resolve_theta_sharp(seeds, args.theta_sharp, args.theta_percentile)
    summary = analyse(seeds, theta_sharp)

    if args.figure_dir:
        _write_figures(seeds, theta_sharp, summary, args.figure_dir)
    if args.out:
        args.out.write_text(json.dumps(summary, indent=2, default=str))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
