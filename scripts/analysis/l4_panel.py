#!/usr/bin/env python
"""L4 panel: plastic wild-type vs plastic rewired-null, with floors and a matched-rule yardstick.

Seven arms on one cell across paired seeds: the wild-type connectome and its
degree-preserving rewired-null, each frozen, under unmodulated Hebbian learning, and under
the three-factor rule, plus a dense MLP under the same three-factor rule. The per-seed
ranked metric is the committed plateau-tail full-clear success
(``t7_continuous_ranking.plateau_tail``) and the statistics layer is the committed
paired-seed one-sided Wilcoxon + 80% bootstrap CI + BH-FDR
(``weight_search_architecture_ranking``), so the panel is measured exactly as the earlier
rankings were.

What is confirmatory is fixed here, in code, before any panel data exist:

- four one-sided paired tests corrected together as one BH-FDR family -- T1 wild-type
  plastic > rewired plastic (the primary contrast); T2 plastic > frozen and T3 plastic >
  Hebbian on the wild-type wiring (the sanity floors); T4 wild-type learning gain >
  rewired learning gain, where a wiring's gain is its plastic arm minus its own frozen arm;
- one CI-based band test: the 80% interval of wild-type plastic minus MLP plastic contains
  or lies above zero. Containing zero is its *null* outcome, so it can pass on noise; the
  delta's mean and interval width are reported beside it, and a ``recovery`` verdict is
  read as resting on T1;
- an ordered verdict map: ``sanity_floor_fail``, ``rewired_beats_wild_type``, ``recovery``,
  ``structure_only``, ``robustness``, ``inconclusive``.

Everything else -- the remaining pairwise deltas, per-behaviour sub-metrics, converged
fractions, learning curves -- is descriptive and labelled so in the output.

Confirmatory mode accepts panel seeds only (1-8): a pilot log cannot enter a test. The
pilot mode summarises the recipe grid on the pilot seeds (101-102) and applies the
selection and budget rules stated in :func:`analyse_pilot`.

Usage::

    uv run python scripts/analysis/l4_panel.py --campaign-dir campaigns/l4-panel
        --out panel.json --csv per-seed.csv --curves curves.csv

    uv run python scripts/analysis/l4_panel.py --pilot --campaign-dir campaigns/l4-pilot
        --out pilot.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

import numpy as np

# The committed metric and statistics layers, reused verbatim.
from t7_continuous_ranking import plateau_tail
from weight_search_architecture_ranking import bh_fdr, paired_seed_wilcoxon_bootstrap

REPO = Path(__file__).resolve().parents[2]
EXPERIMENTS = REPO / "experiments"
CONFIG_DIR = REPO / "configs" / "scenarios" / "foraging_predator_thermal"

_CONNECTOME_STEM = "connectomeppo_small_continuous2d_combined_klinotaxis_plastic"
_MLP_STEM = "mlpppo_small_continuous2d_combined_klinotaxis_plastic"

# Config stem -> arm key. The registry is the only place a log is tied to an arm; a stem
# that is not here is skipped with a warning, never guessed.
ARMS: dict[str, str] = {
    f"{_CONNECTOME_STEM}_frozen": "wt_frozen",
    f"{_CONNECTOME_STEM}_hebbian": "wt_hebbian",
    _CONNECTOME_STEM: "wt_plastic",
    f"{_CONNECTOME_STEM}_frozen_rewired_null": "rn_frozen",
    f"{_CONNECTOME_STEM}_hebbian_rewired_null": "rn_hebbian",
    f"{_CONNECTOME_STEM}_rewired_null": "rn_plastic",
    _MLP_STEM: "mlp_plastic",
}
ARM_KEYS: tuple[str, ...] = tuple(ARMS.values())
STEM_OF: dict[str, str] = {arm: stem for stem, arm in ARMS.items()}
FROZEN_ARMS: tuple[str, ...] = ("wt_frozen", "rn_frozen")
THREE_FACTOR_ARMS: tuple[str, ...] = ("wt_plastic", "rn_plastic", "mlp_plastic")
RULE_BEARING_ARMS: tuple[str, ...] = tuple(a for a in ARM_KEYS if a not in FROZEN_ARMS)

PANEL_SEEDS: tuple[int, ...] = tuple(range(1, 9))
PILOT_SEEDS: tuple[int, ...] = (101, 102)
# Re-registered after the centred-modulator probe: brackets the strongest learner seen at
# 600 episodes (3e-3, which also starts to saturate) with a rate that learns without
# touching the bound (3e-4); the tie-break sits in the middle.
RATE_GRID: tuple[float, ...] = (3e-4, 1e-3, 3e-3)
DEFAULT_RATE = 1e-3
PILOT_BUDGET = 3000
PILOT_EXTENDED_BUDGET = 6000
BUDGET_HEADROOM = 1.25
BUDGET_STEP = 500
BUDGET_FLOOR = 2000
PANEL_EXTENSION = 1.5
SIG_Q = 0.05
CURVE_WINDOW = 250
MIN_PAIRED_SEEDS = 2  # a paired Wilcoxon needs at least two common seeds
FAMILY: tuple[str, ...] = ("T1", "T2", "T3", "T4")

_RUN_LINE = re.compile(r"Run:\s+(\d+)\s+Status:\s+(\S+).*?Eaten:\s+(\d+)/")
_EXPERIMENT_LINE = re.compile(r"Experiment ID:\s+(\S+)")
# Campaign log labels: `<stem>-seed<N>.log`, or `<stem>__rate_<enc>-seed<N>.log` for a
# pilot config derived at one grid rate.
_LABEL = re.compile(r"^(?P<stem>.+?)(?:__rate_(?P<rate>[0-9p]+))?-seed(?P<seed>\d+)\.log$")


def encode_rate(rate: float) -> str:
    """``0.003`` -> ``0p003``: a rate that survives as a filename fragment."""
    return f"{rate:g}".replace(".", "p")


def decode_rate(text: str) -> float:
    """Inverse of :func:`encode_rate`."""
    return float(text.replace("p", "."))


@dataclass
class SeedRecord:
    """Everything one run contributes."""

    success: float  # plateau-tail full-clear %, the ranked metric
    foods: float  # plateau-tail mean foods
    episodes: int
    converged: bool | None  # None when no experiment record was found
    onset: int | None  # plateau onset (run index) when converged
    evasion_rate: float | None
    temp_comfort: float | None
    curve: list[float]  # full-clear % per CURVE_WINDOW-episode block


def _experiment_json(log_text: str, experiments: Path) -> dict | None:
    match = _EXPERIMENT_LINE.search(log_text)
    if match is None:
        return None
    experiment_id = match.group(1)
    path = experiments / experiment_id / f"{experiment_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def read_log(log: Path, experiments: Path = EXPERIMENTS) -> SeedRecord | None:
    """Score one run from its log (and its experiment record, when it can be found)."""
    tail = plateau_tail(log)
    if tail is None:
        return None
    text = log.read_text()
    outcomes = [
        1.0 if m.group(2) == "SUCCESS" else 0.0
        for m in (_RUN_LINE.match(line) for line in text.splitlines())
        if m
    ]
    curve = [
        100.0 * float(np.mean(outcomes[start : start + CURVE_WINDOW]))
        for start in range(0, len(outcomes), CURVE_WINDOW)
    ]
    record = SeedRecord(
        success=tail[0],
        foods=tail[1],
        episodes=len(outcomes),
        converged=None,
        onset=None,
        evasion_rate=None,
        temp_comfort=None,
        curve=curve,
    )
    experiment = _experiment_json(text, experiments)
    if experiment is not None:
        results = experiment.get("results", {})
        onset = results.get("convergence_run")
        record.converged = onset is not None
        record.onset = int(onset) if onset is not None else None
        encounters = results.get("avg_predator_encounters")
        evasions = results.get("avg_successful_evasions")
        if encounters and encounters > 0 and evasions is not None:
            record.evasion_rate = evasions / encounters * 100.0
        record.temp_comfort = results.get("post_convergence_temperature_comfort_score")
    return record


# One scanned run: arm key, grid rate (None for a panel run), seed, record.
Scanned = list[tuple[str, float | None, int, SeedRecord]]


def scan_campaign(campaign_dir: Path, experiments: Path = EXPERIMENTS) -> Scanned:
    """Read every registered run log under ``<campaign_dir>/logs``."""
    log_dir = campaign_dir / "logs" if (campaign_dir / "logs").is_dir() else campaign_dir
    found: Scanned = []
    for log in sorted(log_dir.glob("*.log")):
        match = _LABEL.match(log.name)
        if match is None:
            print(f"  WARN: skipping log with an unrecognised label: {log.name}")
            continue
        stem = match.group("stem")
        if stem not in ARMS:
            print(f"  WARN: skipping log whose config stem is not a registered arm: {log.name}")
            continue
        record = read_log(log, experiments)
        if record is None:
            print(f"  WARN: no parseable run lines in {log.name} - dropped")
            continue
        rate = decode_rate(match.group("rate")) if match.group("rate") else None
        found.append((ARMS[stem], rate, int(match.group("seed")), record))
    return found


def read_manifest(manifest: Path, experiments: Path = EXPERIMENTS) -> Scanned:
    """``<arm> <seed> <log>`` per line; blank and ``#`` lines skipped."""
    found: Scanned = []
    for raw in manifest.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 3 or not parts[1].isdigit():
            print(f"  WARN: skipping malformed manifest line: {raw!r}")
            continue
        arm, seed, log = parts[0], int(parts[1]), Path(parts[2])
        if arm not in ARM_KEYS:
            print(f"  WARN: skipping manifest line with unknown arm {arm!r}: {raw!r}")
            continue
        record = read_log(REPO / log, experiments)
        if record is None:
            print(f"  WARN {arm} seed {seed}: no parseable plateau in {log} - dropped")
            continue
        found.append((arm, None, seed, record))
    return found


def group_panel(scanned: Scanned) -> dict[str, dict[int, SeedRecord]]:
    """``{arm: {seed: record}}`` for confirmatory use; refuses any seed outside the panel."""
    panel: dict[str, dict[int, SeedRecord]] = {}
    for arm, _rate, seed, record in scanned:
        if seed not in PANEL_SEEDS:
            msg = (
                f"seed {seed} ({arm}) is not a panel seed {PANEL_SEEDS[0]}-{PANEL_SEEDS[-1]}; "
                "pilot runs cannot enter a confirmatory test"
            )
            raise ValueError(msg)
        seeds = panel.setdefault(arm, {})
        if seed in seeds:
            print(f"  WARN {arm} seed {seed}: duplicate run - the later log overwrites the earlier")
        seeds[seed] = record
    return panel


def successes(panel: dict[str, dict[int, SeedRecord]]) -> dict[str, dict[int, float]]:
    """Reduce records to the ranked metric."""
    return {arm: {s: r.success for s, r in seeds.items()} for arm, seeds in panel.items()}


# --- statistics -------------------------------------------------------------------------


def paired(a: dict[int, float], b: dict[int, float]) -> dict:
    """Paired delta a - b over common seeds: mean, one-sided Wilcoxon (a > b), 80% CI."""
    common = sorted(set(a) & set(b))
    deltas = [a[s] - b[s] for s in common]
    stats = paired_seed_wilcoxon_bootstrap(deltas)
    stats["seeds"] = common
    stats["positive_seeds"] = int(sum(1 for d in deltas if d > 0))
    stats["ci_width"] = float(stats["ci_hi"] - stats["ci_lo"])
    return stats


def gains(values: dict[str, dict[int, float]], plastic: str, frozen: str) -> dict[int, float]:
    """Compute a wiring's learning gain per seed: plastic arm minus its own frozen arm."""
    p, f = values.get(plastic, {}), values.get(frozen, {})
    return {s: p[s] - f[s] for s in sorted(set(p) & set(f))}


def family_tests(values: dict[str, dict[int, float]]) -> dict[str, dict]:
    """Compute the four pre-registered tests, corrected together, each with its pass verdict."""
    get = values.get
    raw = {
        "T1": paired(get("wt_plastic", {}), get("rn_plastic", {})),
        "T2": paired(get("wt_plastic", {}), get("wt_frozen", {})),
        "T3": paired(get("wt_plastic", {}), get("wt_hebbian", {})),
        "T4": paired(
            gains(values, "wt_plastic", "wt_frozen"),
            gains(values, "rn_plastic", "rn_frozen"),
        ),
    }
    qs = bh_fdr([raw[t]["wilcoxon_p"] for t in FAMILY])
    for test, q in zip(FAMILY, qs, strict=True):
        stats = raw[test]
        stats["bh_q"] = q
        stats["sufficient"] = stats["n"] >= MIN_PAIRED_SEEDS
        stats["pass"] = bool(stats["sufficient"] and q < SIG_Q and stats["mean_delta"] > 0)
    raw["T1"]["reads"] = "wild-type plastic > rewired-null plastic (primary contrast)"
    raw["T2"]["reads"] = "wild-type plastic > wild-type frozen (something was learned)"
    raw["T3"]["reads"] = "wild-type plastic > wild-type Hebbian (learned from reward)"
    raw["T4"]["reads"] = "wild-type learning gain > rewired-null learning gain"
    return raw


def band_test(values: dict[str, dict[int, float]]) -> dict:
    """Wild-type plastic minus MLP plastic: PASS when the 80% CI contains or exceeds zero."""
    stats = paired(values.get("wt_plastic", {}), values.get("mlp_plastic", {}))
    stats["sufficient"] = stats["n"] >= MIN_PAIRED_SEEDS
    stats["pass"] = bool(stats["sufficient"] and stats["ci_hi"] >= 0.0)
    stats["reads"] = (
        "wild-type plastic reaches the matched-rule MLP band; containing zero is the null "
        "outcome, so read the mean delta and the interval width beside the pass"
    )
    return stats


def verdict(tests: dict[str, dict], band: dict) -> str:
    """Assign the verdict by the ordered map."""
    t1, t2, t3 = tests["T1"], tests["T2"], tests["T3"]
    floors_pass = t2["pass"] and t3["pass"]
    # Both MLP-dependent outcomes read the band; an underpowered band supports neither.
    band_needed_but_underpowered = floors_pass and t1["pass"] and not band["sufficient"]
    if not all(t["sufficient"] for t in (t1, t2, t3)) or band_needed_but_underpowered:
        return "insufficient_seeds"
    if not floors_pass:
        return "sanity_floor_fail"
    if t1["ci_hi"] < 0.0:
        return "rewired_beats_wild_type"
    if t1["pass"]:
        return "recovery" if band["pass"] else "structure_only"
    return "robustness" if t1["ci_lo"] <= 0.0 <= t1["ci_hi"] else "inconclusive"


def descriptive_pairs(values: dict[str, dict[int, float]]) -> list[dict]:
    """Every pair of arms, uncorrected, labelled descriptive."""
    rows = []
    for a, b in combinations(ARM_KEYS, 2):
        if a not in values or b not in values:
            continue
        stats = paired(values[a], values[b])
        stats.update({"a": a, "b": b, "descriptive": True})
        rows.append(stats)
    return rows


def analyse(values: dict[str, dict[int, float]], out: dict) -> dict:
    """Family, band, verdict, ensemble-invariance counts and descriptive pairs, from the ranked metric."""
    tests = family_tests(values)
    band = band_test(values)
    result = verdict(tests, band)
    out["family"] = {t: tests[t] for t in FAMILY}
    out["band"] = band
    out["verdict"] = {
        "verdict": result,
        "gain_agrees_with_primary": bool(tests["T4"]["pass"] == tests["T1"]["pass"]),
        "ensemble_invariance": {
            t: {"positive_seeds": tests[t]["positive_seeds"], "n": tests[t]["n"]}
            for t in ("T1", "T4")
        },
    }
    out["per_arm"] = {
        arm: {
            "mean": float(np.mean(list(seeds.values()))) if seeds else math.nan,
            "n": len(seeds),
            "per_seed": dict(sorted(seeds.items())),
        }
        for arm, seeds in values.items()
    }
    out["descriptive_pairs"] = descriptive_pairs(values)
    return out


def _print_panel(out: dict) -> None:
    print("\n" + "=" * 78)
    print("L4 PANEL - plateau-tail full-clear success, paired seeds")
    print("=" * 78)
    for arm in ARM_KEYS:
        row = out["per_arm"].get(arm)
        if row is None:
            continue
        per_seed = [round(v, 1) for _, v in sorted(row["per_seed"].items())]
        print(f"  {arm:12} {row['mean']:6.2f}   n={row['n']}   per-seed={per_seed}")
    print("\n  Confirmatory family (BH-FDR, alpha 0.05):")
    for test in FAMILY:
        t = out["family"][test]
        flag = "PASS" if t["pass"] else "fail"
        print(
            f"    {test}  d={t['mean_delta']:+6.2f}  CI[{t['ci_lo']:+6.2f},{t['ci_hi']:+6.2f}]  "
            f"p={t['wilcoxon_p']:.3f}  q={t['bh_q']:.3f}  +seeds={t['positive_seeds']}/{t['n']}  "
            f"{flag}   {t['reads']}",
        )
    b = out["band"]
    print(
        f"  Band  d={b['mean_delta']:+6.2f}  CI[{b['ci_lo']:+6.2f},{b['ci_hi']:+6.2f}]  "
        f"width={b['ci_width']:.2f}  {'PASS' if b['pass'] else 'FAIL'}",
    )
    v = out["verdict"]
    print("-" * 78)
    print(
        f"  VERDICT: {v['verdict']}   (gain contrast "
        f"{'agrees' if v['gain_agrees_with_primary'] else 'DISAGREES'} with the primary)",
    )


def write_per_seed_csv(panel: dict[str, dict[int, SeedRecord]], path: Path) -> None:
    """Write one row per arm and seed with the ranked metric and the descriptive sub-metrics."""
    fields = [
        "arm",
        "seed",
        "success",
        "foods",
        "episodes",
        "converged",
        "onset",
        "evasion_rate",
        "temp_comfort",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for arm in ARM_KEYS:
            for seed, record in sorted(panel.get(arm, {}).items()):
                row = {k: v for k, v in asdict(record).items() if k in fields}
                writer.writerow({"arm": arm, "seed": seed, **row})


def write_curves_csv(panel: dict[str, dict[int, SeedRecord]], path: Path) -> None:
    """Write the per-seed learning curves, one row per window."""
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["arm", "seed", "window_end", "success"])
        for arm in ARM_KEYS:
            for seed, record in sorted(panel.get(arm, {}).items()):
                for index, value in enumerate(record.curve, 1):
                    writer.writerow([arm, seed, index * CURVE_WINDOW, f"{value:.2f}"])


# --- pilot ------------------------------------------------------------------------------


def budget_from_onset(latest_onset: int) -> int:
    """Smallest multiple of BUDGET_STEP at or above BUDGET_HEADROOM x onset, never below the floor."""
    raw = BUDGET_HEADROOM * latest_onset
    rounded = math.ceil(raw / BUDGET_STEP) * BUDGET_STEP
    return max(BUDGET_FLOOR, rounded)


def select_rate(pooled: dict[float, float]) -> float | None:
    """Select the rate with the highest pooled mean; ties go to the default."""
    if not pooled:
        return None
    best = max(pooled.values())
    candidates = [r for r, m in pooled.items() if m == best]
    return DEFAULT_RATE if DEFAULT_RATE in candidates else candidates[0]


def _pilot_summary(seeds: dict[int, SeedRecord]) -> dict:
    return {
        "mean": float(np.mean([r.success for r in seeds.values()])),
        "per_seed": {s: r.success for s, r in sorted(seeds.items())},
        "onsets": {s: r.onset for s, r in sorted(seeds.items())},
        "converged": sum(1 for r in seeds.values() if r.converged),
        "unknown_convergence": sum(1 for r in seeds.values() if r.converged is None),
        "episodes": {s: r.episodes for s, r in sorted(seeds.items())},
    }


PilotGrid = dict[float, dict[str, dict[int, SeedRecord]]]


def _group_pilot(scanned: Scanned) -> tuple[PilotGrid, dict[str, dict[int, SeedRecord]]]:
    """Split pilot runs into ``{rate: {arm: {seed: record}}}`` and the rate-free frozen floors."""
    by_rate: PilotGrid = {}
    floors: dict[str, dict[int, SeedRecord]] = {}
    for arm, rate, seed, record in scanned:
        if seed not in PILOT_SEEDS:
            print(f"  WARN: {arm} seed {seed} is not a pilot seed - ignored")
            continue
        if rate is None:
            floors.setdefault(arm, {})[seed] = record
        else:
            by_rate.setdefault(rate, {}).setdefault(arm, {})[seed] = record
    return by_rate, floors


def _pooled_means(by_rate: PilotGrid) -> dict[float, float]:
    """Pool the three three-factor arms' plateau-tail values per rate.

    A rate is eligible only when every three-factor arm has exactly the pilot seeds at it:
    pooling over a missing seed would weight the arms unequally.
    """
    pooled: dict[float, float] = {}
    for rate, arms in by_rate.items():
        if not all(set(arms.get(arm, {})) == set(PILOT_SEEDS) for arm in THREE_FACTOR_ARMS):
            print(
                f"  WARN: rate {rate:g} lacks a three-factor arm or a pilot seed and is ineligible",
            )
            continue
        pooled[rate] = float(
            np.mean([r.success for arm in THREE_FACTOR_ARMS for r in arms[arm].values()]),
        )
    return pooled


def _budget_and_actions(arms_at_rate: dict[str, dict[int, SeedRecord]], selected: float) -> dict:
    """Apply the budget rule at the selected rate and list what is still owed before pinning."""
    # The floors are a descriptive read only: they feed neither the selection nor the budget.
    onsets = [
        r.onset for seeds in arms_at_rate.values() for r in seeds.values() if r.onset is not None
    ]
    unknown = [
        f"{arm} seed {s}"
        for arm, seeds in arms_at_rate.items()
        for s, r in seeds.items()
        if r.converged is None
    ]
    non_converged = [
        arm
        for arm in THREE_FACTOR_ARMS
        if not any(r.converged for r in arms_at_rate.get(arm, {}).values())
    ]
    actions: list[str] = []
    if unknown:
        actions.append(
            "convergence unknown (no experiment record) for: "
            + ", ".join(unknown)
            + " - rerun with experiment tracking before pinning",
        )
    pinned_at_extended = []
    for arm in non_converged:
        longest = max((r.episodes for r in arms_at_rate.get(arm, {}).values()), default=0)
        if longest >= PILOT_EXTENDED_BUDGET:
            pinned_at_extended.append(arm)
        else:
            actions.append(
                f"{arm} has no converged pilot run at rate {selected:g}: extend it once to "
                f"{PILOT_EXTENDED_BUDGET} episodes (a fresh run at the same seeds)",
            )
    latest = max(onsets) if onsets else None
    budget = budget_from_onset(latest) if latest is not None else None
    if pinned_at_extended:
        budget = PILOT_EXTENDED_BUDGET
    if budget is None and not actions:
        actions.append("no converged run at the selected rate - the budget rule has no input")
    return {
        "budget": budget,
        "budget_basis": {
            "latest_onset": latest,
            "headroom": BUDGET_HEADROOM,
            "step": BUDGET_STEP,
            "floor": BUDGET_FLOOR,
            "pinned_at_extended_budget_for": pinned_at_extended,
        },
        "non_converged_three_factor_arms": non_converged,
        "action_required": actions,
    }


def analyse_pilot(scanned: Scanned, out: dict) -> dict:
    """Summarise the grid, select the rate, apply the budget rule, and list what is still owed.

    Selection: the grid rate maximising the pooled mean plateau-tail success of the three
    three-factor arms over the pilot seeds (ties to the default). Budget: from the latest
    plateau onset among converged runs of any arm at the selected rate, rounded up with
    headroom and floored. A three-factor arm with no converged run at the selected rate
    owes one extension to the extended pilot budget; if it has already run that long and
    still has no plateau, the budget is pinned at the extended budget and the arm flagged.
    """
    by_rate, floors = _group_pilot(scanned)
    out["grid"] = {
        encode_rate(rate): {arm: _pilot_summary(seeds) for arm, seeds in arms.items()}
        for rate, arms in sorted(by_rate.items())
    }
    out["floors"] = {arm: _pilot_summary(seeds) for arm, seeds in floors.items()}
    pooled = _pooled_means(by_rate)
    out["pooled"] = {encode_rate(r): m for r, m in sorted(pooled.items())}
    selected = select_rate(pooled)
    out["selected_rate"] = selected
    if selected is None:
        out["budget"] = None
        out["action_required"] = ["no rate has all three three-factor arms - nothing to select"]
        _stamp_unpinned(out)
        return out
    out.update(_budget_and_actions(by_rate[selected], selected))
    _stamp_unpinned(out)
    return out


def _stamp_unpinned(out: dict) -> None:
    """Mark the summary as the rules' output, not a decision: pinning is a separate, recorded act."""
    out["pinned"] = False
    out["pin_note"] = (
        "selected_rate and budget are what the registered rules compute from these logs; "
        "they become the panel recipe only when pinned by dated amendment before launch"
    )


def _print_pilot(out: dict) -> None:
    print("\n" + "=" * 78)
    print("L4 PILOT - recipe grid on the pilot seeds")
    print("=" * 78)
    for rate, arms in out["grid"].items():
        print(f"  rate {rate}:")
        for arm, row in arms.items():
            print(
                f"    {arm:12} mean={row['mean']:6.2f}  per-seed={row['per_seed']}  "
                f"converged={row['converged']}  onsets={row['onsets']}",
            )
    for arm, row in out.get("floors", {}).items():
        print(f"  floor {arm:12} mean={row['mean']:6.2f}  per-seed={row['per_seed']}")
    print(f"\n  pooled three-factor means: {out['pooled']}")
    print(f"  selected rate: {out['selected_rate']}")
    print(f"  budget: {out.get('budget')}   basis: {out.get('budget_basis')}")
    for action in out.get("action_required", []):
        print(f"  ACTION: {action}")


# --- entry point ------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the panel or pilot analysis from the command line; return the exit code."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--campaign-dir", type=Path, help="campaign output dir (reads logs/*.log)")
    source.add_argument("--manifest", type=Path, help="<arm> <seed> <log> per line")
    ap.add_argument("--experiments-dir", type=Path, default=EXPERIMENTS)
    ap.add_argument("--pilot", action="store_true", help="summarise the recipe grid instead")
    ap.add_argument("--out", type=Path, default=None, help="write the summary JSON here")
    ap.add_argument("--csv", type=Path, default=None, help="write the per-seed table here")
    ap.add_argument("--curves", type=Path, default=None, help="write the learning curves here")
    ap.add_argument(
        "--sensitivity",
        action="append",
        default=[],
        metavar="RATE=DIR",
        help="a primary-pair campaign at another rate; T1 is recomputed descriptively",
    )
    args = ap.parse_args(argv)

    scanned = (
        scan_campaign(args.campaign_dir, args.experiments_dir)
        if args.campaign_dir
        else read_manifest(args.manifest, args.experiments_dir)
    )
    out: dict = {}
    if args.pilot:
        analyse_pilot(scanned, out)
        _print_pilot(out)
    else:
        try:
            panel = group_panel(scanned)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        analyse(successes(panel), out)
        out["sensitivity"] = []
        for spec in args.sensitivity:
            rate_text, _, directory = spec.partition("=")
            extra = group_panel(scan_campaign(Path(directory), args.experiments_dir))
            t1 = paired(
                successes(extra).get("wt_plastic", {}),
                successes(extra).get("rn_plastic", {}),
            )
            t1.update({"rate": float(rate_text), "descriptive": True})
            out["sensitivity"].append(t1)
        _print_panel(out)
        if args.csv:
            write_per_seed_csv(panel, args.csv)
        if args.curves:
            write_curves_csv(panel, args.curves)
    if args.out:
        args.out.write_text(json.dumps(out, indent=2, default=str))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
