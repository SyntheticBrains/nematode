"""Connectome-structure control - is it the *specific* wiring, or just the degree statistics.

Compares the wild-type *C. elegans* connectome against a **degree-preserving rewired-null** on the
continuous integrated-C3 cell across paired seeds. Reads each run's plateau-tail (final-quarter)
ranked success from the ``run_simulation`` ``.out`` - **reusing the committed ranking metric**
(``t7_continuous_ranking.plateau_tail``), so this control is measured identically to the 029 ranking
- and reports the paired-seed delta (wild-type - rewired) with the committed paired-seed Wilcoxon +
80% bootstrap CI + BH-FDR layer.

Verdict (pre-registered): **SPECIFIC-WIRING** when the wild-type significantly beats the rewired-null
(BH-FDR ``q < 0.05`` and a positive mean delta) - the specific *C. elegans* wiring is genuinely
better-than-degree-matched-random for these behaviours; **DEGREE-STATISTICS** when the two are
indistinguishable (the 80% bootstrap CI spans zero) - it is the connectivity statistics, not the
wiring, that set performance. A significant *negative* delta (the rewired-null beats wild-type) is
reported honestly as its own outcome.

Usage::

    uv run python scripts/analysis/connectome_structure_controls.py \
        --manifest <run-dir>/_manifest.txt --out <run-dir>/controls.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Reuse the committed metric + statistics layers verbatim (identical methodology to the 029 ranking).
from t7_continuous_ranking import plateau_tail
from weight_search_architecture_ranking import bh_fdr, paired_seed_wilcoxon_bootstrap

REPO = Path(__file__).resolve().parents[2]

_WILD = "wild_type"
_REWIRED = "rewired_null"
_MIN_PAIRED_SEEDS = 2  # a paired Wilcoxon needs >= 2 common seeds
# The one-sided signed-rank floor is 1/2^n, so q<0.05 (a `specific_wiring` verdict) is unreachable
# below 5 paired seeds regardless of effect size - the same n>=8 lesson as bit-memory (logbook 030).
_MIN_SEEDS_FOR_SIGNIFICANCE = 5
_SIG_Q = 0.05


def _success(out_path: Path) -> float | None:
    """Plateau-tail full-clear success % for one run (the 029 ranked metric), or None."""
    result = plateau_tail(out_path)
    return None if result is None else result[0]


def load(manifest: Path) -> dict[str, dict[int, float]]:
    """Return ``{arm: {seed: success}}`` from an ``<arm> <seed> <out_path>`` manifest.

    Blank / ``#``-comment lines are skipped silently; any other line that fails the
    ``<arm> <int seed> <out>`` shape, or a duplicate ``(arm, seed)`` overwrite, is reported so
    analysis issues stay traceable rather than silently dropped.
    """
    arms: dict[str, dict[int, float]] = {}
    for raw in manifest.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 3 or not parts[1].isdigit():  # expected: `<arm> <int seed> <out>`
            print(f"  WARN: skipping malformed manifest line: {raw!r}")
            continue
        arm, seed, out_path = parts[0], int(parts[1]), Path(parts[2])
        success = _success(REPO / out_path)
        if success is None:
            print(f"  WARN {arm} seed {seed}: no parseable plateau in {out_path} - dropped")
            continue
        seeds = arms.setdefault(arm, {})
        if seed in seeds:
            print(f"  WARN {arm} seed {seed}: duplicate manifest entry - overwriting previous")
        seeds[seed] = success
    return arms


def analyse(arms: dict[str, dict[int, float]], out: dict) -> None:
    """Print + record the per-arm success, the paired wild-vs-rewired delta, and the verdict."""
    print("\n" + "=" * 72)
    print(
        "CONNECTOME-STRUCTURE CONTROL - plateau-tail C3 ranked success (wild-type vs rewired-null)",
    )
    print("=" * 72)
    out["per_arm"] = {}
    for arm in (_WILD, _REWIRED):
        vals = list(arms.get(arm, {}).values())
        mean = float(np.mean(vals)) if vals else float("nan")
        out["per_arm"][arm] = {"mean": mean, "n": len(vals), "per_seed": arms.get(arm, {})}
        print(f"  {arm:14} {mean:6.2f}   n={len(vals)}   per-seed={[round(v, 1) for v in vals]}")

    common = sorted(set(arms.get(_WILD, {})) & set(arms.get(_REWIRED, {})))
    if len(common) < _MIN_PAIRED_SEEDS:
        out["verdict"] = {"verdict": "insufficient_seeds", "n_common": len(common)}
        print(f"\n  VERDICT: INSUFFICIENT PAIRED SEEDS (only {len(common)} common) - cannot test.")
        return

    underpowered = len(common) < _MIN_SEEDS_FOR_SIGNIFICANCE
    if underpowered:
        print(
            f"  NOTE: n={len(common)} paired seeds - the one-sided signed-rank floor (1/2^n) exceeds "
            f"0.05 below {_MIN_SEEDS_FOR_SIGNIFICANCE}, so a `specific_wiring` verdict is "
            "statistically unreachable at this n (see bit-memory logbook 030); use n>=8.",
        )

    deltas = [arms[_WILD][s] - arms[_REWIRED][s] for s in common]  # +ve = wild-type better
    st = paired_seed_wilcoxon_bootstrap(deltas)
    q = bh_fdr([st["wilcoxon_p"]])[0]
    ci_spans_zero = st["ci_lo"] <= 0.0 <= st["ci_hi"]

    # `paired_seed_wilcoxon_bootstrap` is one-sided (H1: wild > rewired), so a significant
    # rewired-WIN cannot be detected by q - it shows up as the 80% bootstrap CI lying entirely
    # below zero. Use the CI (direction-agnostic) for the rewired-beats branch.
    if q < _SIG_Q and st["mean_delta"] > 0:
        verdict = "specific_wiring"
        headline = "SPECIFIC-WIRING - the wild-type connectome beats its degree-matched rewirings"
    elif st["ci_hi"] < 0.0:
        verdict = "rewired_beats_wildtype"
        headline = (
            "REWIRED-BEATS-WILD-TYPE - the specific wiring is worse than degree-matched random"
        )
    elif ci_spans_zero:
        verdict = "degree_statistics"
        headline = "DEGREE-STATISTICS - wild-type and rewired-null are indistinguishable"
    else:
        verdict = "inconclusive"
        headline = "INCONCLUSIVE - not significant, but the CI does not cleanly span zero"

    out["verdict"] = {
        "verdict": verdict,
        "n_common": len(common),
        "underpowered": underpowered,
        "mean_delta": st["mean_delta"],
        "ci_lo": st["ci_lo"],
        "ci_hi": st["ci_hi"],
        "wilcoxon_p": st["wilcoxon_p"],
        "bh_q": q,
    }
    print(
        f"\n  Paired delta (wild-type - rewired, n={len(common)}): d={st['mean_delta']:+.2f}  "
        f"CI[{st['ci_lo']:+.2f},{st['ci_hi']:+.2f}]  q={q:.3f}",
    )
    print("-" * 72)
    print(f"  VERDICT: {headline}.")


def main() -> None:
    """Load the manifest, compute + print + write the control analysis."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True, help="<arm> <seed> <out_path> per line")
    ap.add_argument("--out", type=Path, default=None, help="write the control summary JSON here")
    args = ap.parse_args()

    arms = load(args.manifest)
    out: dict = {}
    analyse(arms, out)
    if args.out:
        args.out.write_text(json.dumps(out, indent=2, default=str))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
