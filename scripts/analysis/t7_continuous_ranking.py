"""T7 cross-architecture C3 ranking on the continuous-2D substrate.

The Phase-6 Tranche-7 analogue of ``weight_search_architecture_ranking`` (Phase 5,
grid substrate): ranks the 6 MUST arms on their integrated continuous-C3 cells on the
**plateau-tail (final-quarter) full-clear mean**, with the level-agnostic
``post_convergence_success_rate`` (``add-level-agnostic-convergence-metric`` / #250) as
the per-seed convergence verdict. The plateau-tail mean is the robust ranked metric: it
equals ``post_convergence_success_rate`` within sampling noise for *converged* seeds (the
metric's documented cross-check), but for a non-converged seed it avoids the last-10-run
fallback that field degrades to — so a still-trending/wobbling seed is ranked on its
actual plateau level and flagged (the converged fraction is reported alongside as the
seed-stability dimension, NOT chased to 8/8 via tuning — intrinsic fragility is a result).

Reuses the project's committed statistics layer verbatim (paired-seed one-sided
Wilcoxon + 80% bootstrap CI + BH-FDR) by importing the helpers from
``weight_search_architecture_ranking`` — same methodology, only the data source differs.

Metric sources:
- PPO arms (mlpppo / lstmppo / cfcppo / transformerppo / connectomeppo): the per-run
  ``.out`` (plateau-tail success) + ``experiments/<id>/<id>.json`` (``convergence_run``
  verdict + per-behaviour sub-metrics) from ``run_simulation.py --track-experiment``. The
  run manifest (``--manifest``) has one ``<arch> <seed> <experiment_id> <out_path>`` line
  per run; arms may use different episode budgets (the plateau metric normalises across
  budgets — e.g. the dial-able arms at 4000ep, the wobbly ones at 3000ep).
- FeedforwardGA: the precomputed champion full-clear eval (``--ga-results``), the
  GA's analogue of the post-convergence full-clear rate (it trains via
  ``run_evolution.py`` and has no ``--track-experiment`` JSON).

Usage::

    uv run python scripts/analysis/t7_continuous_ranking.py \
        --manifest <run-dir>/_manifest.txt \
        --ga-results <ga-run-dir>/ga_c3_results.json \
        --out <run-dir>/ranking.json
"""

from __future__ import annotations

import argparse
import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np

# Reuse the committed statistics layer verbatim (identical methodology to the T4/Phase-5
# ranking): paired-seed one-sided Wilcoxon + 80% bootstrap CI (1000 resamples, seeded) +
# BH-FDR. Only the data plumbing below is T7-specific.
from weight_search_architecture_ranking import bh_fdr, paired_seed_wilcoxon_bootstrap

REPO = Path(__file__).resolve().parents[2]
EXPERIMENTS = REPO / "experiments"

_FINAL_WINDOW_FRAC = 0.25  # plateau-tail window: the final quarter of a run

PPO_ARCHS = ("mlpppo", "transformerppo", "cfcppo", "connectomeppo", "lstmppo")
ALL_ARCHS = (*PPO_ARCHS, "feedforwardga")


def _plateau_tail(out_path: Path) -> tuple[float, float] | None:
    """(full-clear success %, mean foods) over the final-quarter plateau tail, from the .out.

    The robust primary (success) + foraging sub-metric (foods), read from the per-run
    ``.out`` rather than the JSON's ``post_convergence_*`` / ``avg_*`` fields: the
    post-convergence fields fall back to a noisy last-10-run mean for non-converged runs,
    and the ``avg_*`` fields include the warm-up. The plateau-tail mean is the robust
    plateau level (it equals ``post_convergence_success_rate`` within sampling noise for
    converged seeds — the metric's final-window cross-check).
    """
    if not out_path.exists():
        return None
    succ, foods = [], []
    for ln in out_path.read_text().splitlines():
        m = re.match(r"Run:\s+\d+\s+Status:\s+(\S+).*?Eaten:\s+(\d+)/", ln)
        if m:
            succ.append(1.0 if m.group(1) == "SUCCESS" else 0.0)
            foods.append(int(m.group(2)))
    if not succ:
        return None
    tail = max(1, int(len(succ) * _FINAL_WINDOW_FRAC))
    return 100.0 * float(np.mean(succ[-tail:])), float(np.mean(foods[-tail:]))


def _ppo_metrics(experiment_id: str, out_path: Path) -> dict | None:
    """Per-seed metrics: plateau-tail success + foods (from the .out) + JSON convergence/sub-metrics.

    ``converged`` is the detector's real verdict (``convergence_run`` is non-null), NOT
    "the metric is populated" — the JSON's ``post_convergence_success_rate`` is always
    populated (it falls back to a last-10 mean when the run did not converge), so a
    non-converged run is flagged here and ranked on its plateau-tail mean, never on the
    noisy fallback. ``foods`` likewise uses the plateau tail; ``evasion_rate`` /
    ``temp_comfort`` come from the JSON post-convergence fields (descriptive sub-metrics;
    at a budget where most seeds converge the fallback rarely applies).
    """
    jpath = EXPERIMENTS / experiment_id / f"{experiment_id}.json"
    tail = _plateau_tail(out_path)
    if not jpath.exists() or tail is None:
        return None
    success, foods_tail = tail
    r = json.loads(jpath.read_text()).get("results", {})
    enc = r.get("avg_predator_encounters")
    evas = r.get("avg_successful_evasions")
    return {
        "success": success,
        "converged": r.get("convergence_run") is not None,
        "overall_success": (r.get("success_rate") or 0.0) * 100.0,
        "foods": foods_tail,
        "evasion_rate": (evas / enc * 100.0) if (enc and enc > 0) else None,
        "temp_comfort": r.get("post_convergence_temperature_comfort_score"),
    }


def load_primary(manifest: Path, ga_results: Path | None) -> dict[str, dict[int, dict]]:
    """Return {arch: {seed: metrics}} from the run manifest + the GA champion eval."""
    table: dict[str, dict[int, dict]] = {a: {} for a in PPO_ARCHS}
    for line in manifest.read_text().splitlines():
        parts = line.split()
        # Manifest line: "<arch> <seed> <experiment_id> <out_path>".
        if len(parts) != 4:
            continue
        arch, seed, eid, out_path = parts[0], int(parts[1]), parts[2], Path(parts[3])
        if arch not in table:
            continue
        m = _ppo_metrics(eid, REPO / out_path)
        if m:
            table[arch][seed] = m
    for arch in PPO_ARCHS:
        missing = [s for s in range(1, 9) if s not in table[arch]]
        if missing:
            print(f"  WARN {arch}: {len(missing)} seed(s) missing experiment JSON {missing}")

    if ga_results is not None:
        ga = json.loads(ga_results.read_text()) if ga_results.exists() else {}
        if ga:
            table["feedforwardga"] = {
                int(s): {
                    "success": v["full_clear_rate"],
                    "converged": None,  # n/a — the GA champion eval has no training-plateau notion
                    "foods": v.get("mean_foods"),
                    "evasion_rate": None,
                    "temp_comfort": None,
                }
                for s, v in ga.items()
            }
        else:
            # --ga-results was requested but unusable: do NOT silently drop the GA arm.
            print(f"  WARN feedforwardga: --ga-results {ga_results} missing or empty — GA arm excluded")
    return table


def _per_seed(table_arch: dict[int, dict], key: str) -> dict[int, float]:
    return {s: m[key] for s, m in table_arch.items() if m.get(key) is not None}


def cross_arch_ranking(primary: dict[str, dict[int, dict]], out: dict) -> None:
    """Print + record the ranking + pairwise deltas (Wilcoxon + bootstrap CI + BH-FDR)."""
    archs = [a for a in ALL_ARCHS if primary.get(a)]
    print("\n" + "=" * 78)
    print(
        "T7 CONTINUOUS-C3 RANKING — primary metric: full-clear success (post-conv, level-agnostic)",
    )
    print("=" * 78)
    means = {}
    for a in archs:
        vals = list(_per_seed(primary[a], "success").values())
        has_conv = any(m.get("converged") is not None for m in primary[a].values())
        nconv = sum(1 for m in primary[a].values() if m.get("converged")) if has_conv else None
        means[a] = (float(np.mean(vals)), float(np.std(vals)), vals, nconv)
    ranking = []
    for a in sorted(archs, key=lambda x: -means[x][0]):
        mu, sd, vals, nconv = means[a]
        conv_str = f"{nconv}/{len(vals)}" if nconv is not None else "n/a"
        print(
            f"  {a:16} {mu:5.1f} ± {sd:4.1f}   converged {conv_str:>4}   per-seed={[round(v, 1) for v in vals]}",
        )
        ranking.append(
            {
                "arch": a,
                "mean": mu,
                "std": sd,
                "n": len(vals),
                "converged": nconv,
                "per_seed": vals,
            },
        )
    out["ranking"] = ranking

    print("\n  Pairwise paired-seed deltas (one-sided Wilcoxon a>b, 80% bootstrap CI, BH-FDR):")
    pairs, stats = [], []
    for a, b in combinations(sorted(archs, key=lambda x: -means[x][0]), 2):
        common = sorted(
            set(_per_seed(primary[a], "success")) & set(_per_seed(primary[b], "success")),
        )
        deltas = [primary[a][s]["success"] - primary[b][s]["success"] for s in common]
        pairs.append((a, b))
        stats.append(paired_seed_wilcoxon_bootstrap(deltas))
    qs = bh_fdr([s["wilcoxon_p"] for s in stats])
    out["pairwise"] = []
    for (a, b), st, q in zip(pairs, stats, qs, strict=True):
        sig = "***" if q < 0.05 else ("." if q < 0.10 else "ns")
        print(
            f"    {a:14} vs {b:14}  d={st['mean_delta']:+6.1f}  "
            f"CI[{st['ci_lo']:+5.1f},{st['ci_hi']:+5.1f}]  p={st['wilcoxon_p']:.3f}  q={q:.3f}  {sig}",
        )
        out["pairwise"].append({"arm_a": a, "arm_b": b, **st, "bh_q": q})


def per_behaviour(primary: dict[str, dict[int, dict]], out: dict) -> None:
    """Print + record the per-behaviour sub-metric table (foraging / evasion / thermal)."""
    print("\n" + "=" * 78)
    print("PER-BEHAVIOUR SUB-METRICS (mean over seeds)")
    print("=" * 78)
    print(f"  {'arch':16} {'foraging(foods)':>16} {'predator(evas%)':>16} {'thermal(comfort)':>16}")
    out["per_behaviour"] = {}
    for a in ALL_ARCHS:
        if a not in primary or not primary[a]:
            continue
        foods = list(_per_seed(primary[a], "foods").values())
        evas = list(_per_seed(primary[a], "evasion_rate").values())
        tc = list(_per_seed(primary[a], "temp_comfort").values())
        fs = f"{np.mean(foods):.2f}" if foods else "  -"
        es = f"{np.mean(evas):.1f}" if evas else "  N/A"
        ts = f"{np.mean(tc):.3f}" if tc else "  N/A"
        print(f"  {a:16} {fs:>16} {es:>16} {ts:>16}")
        out["per_behaviour"][a] = {
            "foods": float(np.mean(foods)) if foods else None,
            "evasion_rate": float(np.mean(evas)) if evas else None,
            "temp_comfort": float(np.mean(tc)) if tc else None,
        }


def main() -> None:
    """Load the manifest + GA results, compute + print + write the T7 ranking."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="<arch> <seed> <experiment_id> per line",
    )
    ap.add_argument(
        "--ga-results",
        type=Path,
        default=None,
        help="GA champion full-clear eval JSON",
    )
    ap.add_argument("--out", type=Path, default=None, help="write the ranking summary JSON here")
    args = ap.parse_args()

    primary = load_primary(args.manifest, args.ga_results)
    out: dict = {}
    cross_arch_ranking(primary, out)
    per_behaviour(primary, out)
    if args.out:
        args.out.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
