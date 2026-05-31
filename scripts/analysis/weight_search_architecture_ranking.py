"""Cross-architecture C3 ranking analysis for the weight-search-architecture-ranking change (Phase 5).

Consumes the Phase 4 C3 primary cells (4 architectures x n=8 seeds: 42-49) and the
ablation cells (connectome strict-vs-soft; per-PPO-family reward-shape), and
produces:

- The cross-architecture ranking on the primary metric (full-clear success at
  convergence), with paired-seed one-sided Wilcoxon + 80% bootstrap CIs +
  BH-FDR correction across the pairwise tests.
- Per-behaviour sub-metric tables (foraging, predator-evasion, thermal-comfort).
- The connectome verdict: per-metric wins / ties / losses vs each other family.
- Ablation deltas (strict-vs-soft connectome; reward-shape per PPO family).

Metric sources:
- PPO families (mlpppo / lstmppo / connectomeppo): the per-run
  ``experiments/<id>/<id>.json`` written by ``run_simulation.py --track-experiment``.
  Each run's experiment id is read from the captured ``.out`` (the launcher prints
  ``Experiment ID: <id>``). The primary metric is ``post_convergence_success_rate``
  (the plateau-window full-clear rate).
- FeedforwardGA: a precomputed ``analysis/ga_c3_results.json`` (champion full-clear
  rate + mean foods over a frozen eval) — the GA trains via ``run_evolution.py`` and
  has no ``--track-experiment`` JSON.

The reusable ``paired_seed_wilcoxon_bootstrap`` helper is extracted (genericised to
a plain ``deltas`` list) from ``scripts/campaigns/aggregate_m613_pilot.py``.
"""

from __future__ import annotations

import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]
PHASE4 = REPO / "tmp/evaluations/weight-search-architecture-ranking/phase-4"
EXPERIMENTS = REPO / "experiments"
ANALYSIS = PHASE4 / "analysis"
C3 = PHASE4 / "c3"  # n=4 location — reused byte-unchanged mlpppo/connectome 42-45 only
C3_N8 = PHASE4 / "c3-n8"  # n=8 fresh runs (all lstmppo + cfcppo + ga; mlpppo/connectome 46-49)
SEEDS = (42, 43, 44, 45, 46, 47, 48, 49)
REUSE_SEEDS = (42, 43, 44, 45)  # seeds reused from the n=4 c3/ location for unchanged configs
PPO_ARCHS = ("mlpppo", "lstmppo", "connectomeppo", "cfcppo")
ALL_FRESH_ARCHS = ("lstmppo", "cfcppo")  # always c3-n8 (config changed / new arch — no c3/ reuse)
BOOTSTRAP_RESAMPLES = 1000
CI_LEVEL = 0.80
_EPS = 1e-12


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------


def paired_seed_wilcoxon_bootstrap(deltas: list[float]) -> dict:
    """Paired-seed deltas (a - b) -> mean, one-sided Wilcoxon (a > b), 80% bootstrap CI.

    Genericised from ``aggregate_m613_pilot.compute_cross_arm_delta_stats`` (which
    keys on a 4-tuple survival table) to a plain list of per-seed deltas. Seeded
    bootstrap (rng=42, 1000 resamples) for reproducibility.
    """
    if not deltas:
        return {
            "mean_delta": 0.0,
            "wilcoxon_p": 1.0,
            "ci_lo": 0.0,
            "ci_hi": 0.0,
            "n": 0,
            "per_seed_deltas": [],
        }
    mean_delta = float(np.mean(deltas))
    if all(abs(d) < _EPS for d in deltas):
        p = 1.0
    else:
        p = float(getattr(wilcoxon(deltas, alternative="greater"), "pvalue", 1.0))
    rng = np.random.default_rng(42)
    arr = np.asarray(deltas, dtype=float)
    boots = np.array(
        [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(BOOTSTRAP_RESAMPLES)],
    )
    alpha = 1.0 - CI_LEVEL
    return {
        "mean_delta": mean_delta,
        "wilcoxon_p": p,
        "ci_lo": float(np.quantile(boots, alpha / 2)),
        "ci_hi": float(np.quantile(boots, 1.0 - alpha / 2)),
        "n": len(deltas),
        "per_seed_deltas": deltas,
    }


def bh_fdr(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg q-values (step-up, monotonic) for a list of p-values."""
    n = len(pvals)
    if n == 0:
        return []
    order = np.argsort(pvals)
    q = np.empty(n)
    running = 1.0
    for rank_from_top, idx in enumerate(reversed(order)):
        i = n - rank_from_top  # i = n, n-1, ..., 1
        running = min(running, pvals[idx] * n / i)
        q[idx] = running
    return [float(x) for x in q]


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------


def _experiment_id(out_path: Path) -> str | None:
    if not out_path.exists():
        return None
    m = re.search(r"Experiment ID: (\S+)", out_path.read_text())
    return m.group(1) if m else None


def _ppo_run_metrics(out_path: Path) -> dict | None:
    """Load one PPO run's metrics from its experiment JSON (mapped via the .out id)."""
    eid = _experiment_id(out_path)
    if eid is None:
        return None
    jpath = EXPERIMENTS / eid / f"{eid}.json"
    if not jpath.exists():
        return None
    r = json.loads(jpath.read_text()).get("results", {})
    enc = r.get("avg_predator_encounters")
    evas = r.get("avg_successful_evasions")
    evasion_rate = (evas / enc * 100.0) if (enc and enc > 0) else None
    return {
        "success": (r.get("post_convergence_success_rate") or 0.0) * 100.0,
        "overall_success": (r.get("success_rate") or 0.0) * 100.0,
        "foods": r.get("avg_foods_collected"),
        "evasion_rate": evasion_rate,
        "temp_comfort": r.get("post_convergence_temperature_comfort_score"),
    }


def _c3_out_path(arch: str, seed: int) -> Path:
    """Resolve the .out for a (arch, seed) C3 primary run, honouring the n=8 reuse rule.

    - ``lstmppo`` / ``cfcppo`` (ALL_FRESH_ARCHS): ALWAYS the fresh ``c3-n8/`` run — lstmppo's
      ``c3/`` 42-45 predate the recurrent-stability fixes (stale); cfcppo is a new arch (no c3/).
    - ``mlpppo`` / ``connectomeppo``: their C3 config is byte-unchanged, so 42-45 are
      reused from ``c3/`` (saving the expensive connectome re-runs); 46-49 are fresh.
    """
    if arch in ALL_FRESH_ARCHS:
        return C3_N8 / f"{arch}-c3-s{seed}.out"
    return (C3 if seed in REUSE_SEEDS else C3_N8) / f"{arch}-c3-s{seed}.out"


def load_primary() -> dict[str, dict[int, dict]]:
    """Return {arch: {seed: metrics}} for the C3 primary cells."""
    table: dict[str, dict[int, dict]] = {}
    for arch in PPO_ARCHS:
        table[arch] = {}
        for seed in SEEDS:
            m = _ppo_run_metrics(_c3_out_path(arch, seed))
            if m:
                table[arch][seed] = m
    # GA primary from the precomputed champion eval.
    ga_path = ANALYSIS / "ga_c3_results.json"
    if ga_path.exists():
        ga = json.loads(ga_path.read_text())
        table["feedforwardga"] = {
            int(s): {
                "success": v["full_clear_rate"],
                "foods": v["mean_foods"],
                "evasion_rate": None,
                "temp_comfort": None,
            }
            for s, v in ga.items()
        }
    return table


def load_ablation(tag: str) -> dict[int, dict]:
    """Return {seed: metrics} for an ablation tag (e.g. 'softprior-connectomeppo')."""
    out = {}
    for seed in SEEDS:
        m = _ppo_run_metrics(PHASE4 / "ablations" / f"{tag}-s{seed}.out")
        if m:
            out[seed] = m
    return out


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _per_seed(table_arch: dict[int, dict], key: str) -> dict[int, float]:
    return {s: m[key] for s, m in table_arch.items() if m.get(key) is not None}


def cross_arch_ranking(primary: dict[str, dict[int, dict]]) -> None:
    """Print the cross-architecture ranking + pairwise deltas (Wilcoxon + bootstrap CI + BH-FDR)."""
    archs = [a for a in (*PPO_ARCHS, "feedforwardga") if primary.get(a)]
    print("\n" + "=" * 78)
    print("CROSS-ARCHITECTURE C3 RANKING — primary metric: full-clear success (post-conv)")
    print("=" * 78)
    means = {}
    for a in archs:
        vals = list(_per_seed(primary[a], "success").values())
        means[a] = (float(np.mean(vals)), float(np.std(vals)), vals)
    for a in sorted(archs, key=lambda x: -means[x][0]):
        mu, sd, vals = means[a]
        print(f"  {a:16} {mu:5.1f} ± {sd:4.1f}   per-seed={[round(v, 1) for v in vals]}")

    print("\n  Pairwise paired-seed deltas (one-sided Wilcoxon a>b, 80% bootstrap CI, BH-FDR):")
    pairs, stats = [], []
    for a, b in combinations(sorted(archs, key=lambda x: -means[x][0]), 2):
        common = sorted(
            set(_per_seed(primary[a], "success")) & set(_per_seed(primary[b], "success")),
        )
        deltas = [primary[a][s]["success"] - primary[b][s]["success"] for s in common]
        st = paired_seed_wilcoxon_bootstrap(deltas)
        pairs.append((a, b))
        stats.append(st)
    qs = bh_fdr([s["wilcoxon_p"] for s in stats])
    rows = []
    for (a, b), st, q in zip(pairs, stats, qs, strict=True):
        sig = "***" if q < 0.05 else ("." if q < 0.10 else "ns")
        print(
            f"    {a:14} vs {b:14}  Δ={st['mean_delta']:+6.1f}  "
            f"CI[{st['ci_lo']:+5.1f},{st['ci_hi']:+5.1f}]  p={st['wilcoxon_p']:.3f}  q={q:.3f}  {sig}",
        )
        rows.append({"arm_a": a, "arm_b": b, **st, "bh_q": q})
    (ANALYSIS / "cross_arch_pairwise.json").write_text(json.dumps(rows, indent=2))


def per_behaviour(primary: dict[str, dict[int, dict]]) -> None:
    """Print the per-behaviour sub-metric table (foraging, predator-evasion, thermal-comfort)."""
    print("\n" + "=" * 78)
    print("PER-BEHAVIOUR SUB-METRICS (mean over seeds)")
    print("=" * 78)
    print(f"  {'arch':16} {'foraging(foods)':>16} {'predator(evas%)':>16} {'thermal(comfort)':>16}")
    for a in (*PPO_ARCHS, "feedforwardga"):
        if a not in primary or not primary[a]:
            continue
        foods = list(_per_seed(primary[a], "foods").values())
        evas = list(_per_seed(primary[a], "evasion_rate").values())
        tc = list(_per_seed(primary[a], "temp_comfort").values())
        fs = f"{np.mean(foods):.2f}" if foods else "  -"
        es = f"{np.mean(evas):.1f}" if evas else "  N/A"
        ts = f"{np.mean(tc):.3f}" if tc else "  N/A"
        print(f"  {a:16} {fs:>16} {es:>16} {ts:>16}")


def connectome_verdict(primary: dict[str, dict[int, dict]]) -> None:
    """Print the wild-type-connectome verdict (per-metric win/tie/loss) vs each other family."""
    print("\n" + "=" * 78)
    print("CONNECTOME VERDICT — wild-type connectome vs each family (paired-seed mean Δ)")
    print("=" * 78)
    if "connectomeppo" not in primary:
        print("  (no connectome data)")
        return
    for other in ("mlpppo", "lstmppo", "cfcppo", "feedforwardga"):
        if other not in primary or not primary[other]:
            continue
        for key, label in (
            ("success", "full-clear"),
            ("foods", "foraging"),
            ("evasion_rate", "predator-evasion"),
        ):
            ca, oa = _per_seed(primary["connectomeppo"], key), _per_seed(primary[other], key)
            common = sorted(set(ca) & set(oa))
            if not common:
                continue
            deltas = [ca[s] - oa[s] for s in common]
            st = paired_seed_wilcoxon_bootstrap(deltas)
            d = st["mean_delta"]
            verdict = (
                "WIN"
                if (d > 0 and st["wilcoxon_p"] < 0.10)
                else ("~tie" if abs(d) < 5 else ("ahead" if d > 0 else "behind"))
            )
            print(
                f"  connectome vs {other:14} [{label:16}] Δ={d:+6.1f}  p={st['wilcoxon_p']:.3f}  → {verdict}",
            )


def ablations(primary: dict[str, dict[int, dict]]) -> None:
    """Print the ablation deltas (strict-vs-soft connectome; reward-shape per PPO family)."""
    print("\n" + "=" * 78)
    print("ABLATIONS (paired-seed mean Δ vs canonical C3)")
    print("=" * 78)
    # 4c.1 strict (primary connectome) vs soft-prior
    soft = load_ablation("softprior-connectomeppo")
    if soft and "connectomeppo" in primary:
        common = sorted(
            set(_per_seed(primary["connectomeppo"], "success")) & set(_per_seed(soft, "success")),
        )
        deltas = [primary["connectomeppo"][s]["success"] - soft[s]["success"] for s in common]
        if deltas:
            st = paired_seed_wilcoxon_bootstrap(deltas)
            print(
                f"  connectome strict - soft_prior  Δ={st['mean_delta']:+6.1f}  CI[{st['ci_lo']:+5.1f},{st['ci_hi']:+5.1f}]  p={st['wilcoxon_p']:.3f}",
            )
    else:
        print("  strict-vs-soft: ablation data not yet present")
    # 4c.2 canonical vs gradient_proximity reward (PPO only)
    for arch in PPO_ARCHS:
        grad = load_ablation(f"gradprox-{arch}")
        if grad and arch in primary:
            common = sorted(
                set(_per_seed(primary[arch], "success")) & set(_per_seed(grad, "success")),
            )
            deltas = [primary[arch][s]["success"] - grad[s]["success"] for s in common]
            if deltas:
                st = paired_seed_wilcoxon_bootstrap(deltas)
                print(
                    f"  {arch:14} canonical - gradient_proximity  Δ={st['mean_delta']:+6.1f}  CI[{st['ci_lo']:+5.1f},{st['ci_hi']:+5.1f}]  p={st['wilcoxon_p']:.3f}",
                )
        else:
            print(f"  {arch}: reward-shape ablation data not yet present")


def main() -> None:
    """Run the full cross-architecture analysis and write the CSV/JSON exports."""
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    primary = load_primary()
    cross_arch_ranking(primary)
    per_behaviour(primary)
    connectome_verdict(primary)
    ablations(primary)
    print(f"\nCSV/JSON exports under {ANALYSIS}/")


if __name__ == "__main__":
    main()
