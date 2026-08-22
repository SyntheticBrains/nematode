"""Helpers shared by the ``scripts/campaigns`` aggregators.

Every function here was **verified byte-identical** (normalised AST, docstrings
excluded) across all its former copies before extraction. Helpers whose copies
had drifted are NOT here unless the drift resolved to a strict superset: they
either take a parameter for the difference, or stay per-script where the
difference is genuine per-milestone reporting rather than accidental divergence.

Nothing here is campaign-specific. If a change needs to branch on which
milestone is calling it, that is a signal the behaviour belongs in the caller.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import re
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

try:
    import numpy as np
    from scipy.stats import wilcoxon  # type: ignore[import-untyped]
except ImportError as exc:  # pragma: no cover
    msg = (
        "scripts.campaigns._common requires scipy + numpy. Install via:\n"
        "  uv sync --extra analysis\n"
        f"(import error: {exc})"
    )
    raise ImportError(msg) from exc

logger = logging.getLogger(__name__)

# Decision-gate thresholds — identical in every former copy.
GATE_F1_RATIO = 0.40
GATE_F2_RATIO = 0.25
GATE_F3_RATIO = 0.15

# Per-arm verdict thresholds — identical in every former copy.
VERDICT_GO_MIN_SEEDS = 2  # >=2 of N -> GO
VERDICT_PIVOT_MIN_SEEDS = 1  # exactly 1 -> PIVOT (below -> STOP)

# Cross-arm bootstrap settings — identical in every former copy.
CROSS_ARM_BOOTSTRAP_RESAMPLES = 1000
CROSS_ARM_BOOTSTRAP_CI_LEVEL = 0.80  # 80% CI => alpha=0.20


__all__ = [
    "aggregate_per_arm_verdict",
    "baseline_success_rates",
    "build_survival_table",
    "compute_cross_arm_delta_stats",
    "evaluate_decision_gate_one_seed",
    "latest_session",
    "load_f0_training_fitness_per_seed",
    "mean",
    "read_f0_training_fitness",
    "read_history",
    "read_per_gen_csv",
    "resolve_session_for",
    "resolve_speed",
    "write_cross_arm_verdict_csv",
]


# Extracted from 3 identical copies: m6, m613, m69
def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# Extracted from 2 identical copies: baldwin_retry, m4
def resolve_speed(g: int | None, fallback_gen: int) -> float:
    """Resolve a per-seed gen-to-target value to a float for averaging.

    Treats never-reached as the fallback (run's max generation + 1) so
    the metric is bounded; conservative for the GO check.
    """
    return float(g) if g is not None else float(fallback_gen)


# Extracted from 3 identical copies: m2, m3, m4
def latest_session(seed_dir: Path) -> Path:
    """Return the most recently modified subdirectory under ``seed_dir``.

    Filtering to directories (rather than relying on lexicographic order over
    ``iterdir()``) avoids stray files (``.DS_Store``, log tails, etc.) being
    mistaken for a session.  Selecting by ``stat().st_mtime`` instead of name
    means we don't depend on a particular session-id format ordering.
    """
    sessions = [p for p in seed_dir.iterdir() if p.is_dir()]
    if not sessions:
        msg = f"No session directory under {seed_dir}"
        raise FileNotFoundError(msg)
    return max(sessions, key=lambda p: p.stat().st_mtime)


# Extracted from 3 identical copies: m6, m613, m69
def read_per_gen_csv(path: Path) -> list[dict]:
    """Read the per-gen choice-index CSV into a list of dict rows."""
    if not path.exists():
        msg = f"per-gen CSV not found: {path}"
        raise FileNotFoundError(msg)
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# Extracted from 4 identical copies: baldwin_retry, m2, m3, m4
def baseline_success_rates(baseline_root: Path) -> dict[int, float]:
    """Extract per-seed success rates from the run_simulation.py log files.

    Note: M2.11's baseline covers seeds 42-45 only (n=4).  M4.5's pilot
    arms run n=8 — the per-seed table will show "—" for the seeds
    without baseline data, and the convergence plot's baseline
    horizontal line is annotated to disclose the n-asymmetry.
    """
    rates: dict[int, float] = {}
    for log in sorted(baseline_root.glob("seed-*.log")):
        seed_match = re.search(r"seed-(\d+)\.log", log.name)
        if not seed_match:
            continue
        seed = int(seed_match.group(1))
        for line in log.read_text().splitlines():
            m = re.match(r"^Success rate:\s+([\d.]+)%", line)
            if m:
                rates[seed] = float(m.group(1)) / 100.0
                break
    return rates


# Extracted from 3 copies (m6, m613, m69). The m613 variant is used: it is a
# strict SUPERSET — it guards the ``generation`` parse with try/except and logs
# a warning instead of raising on a malformed value. On well-formed data all
# three behave identically; on malformed data m6/m69 previously raised
# ValueError and now skip the row with a warning. Declared, not silent.
def read_f0_training_fitness(jsonl_path: Path) -> float | None:  # noqa: C901 - linear JSONL row filter; branches are defensive guards
    """Return the F0 (``generation == 0``) elite's training-time ``fitness`` field, or None.

    Skips rows missing the ``fitness`` key OR with non-finite values
    (NaN, inf, non-numeric string). Mirrors the M6 hardened loader.
    """
    try:
        with jsonl_path.open(encoding="utf-8") as handle:
            for raw in handle:
                stripped = raw.strip()
                if not stripped:
                    continue
                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                try:
                    generation = int(row.get("generation", -1))
                except (TypeError, ValueError):
                    logger.warning(
                        "Skipping row with malformed generation in %s: %s",
                        jsonl_path,
                        row.get("generation"),
                    )
                    continue
                if generation != 0:
                    continue
                if "fitness" not in row:
                    return None
                try:
                    value = float(row["fitness"])
                except (TypeError, ValueError):
                    return None
                if not math.isfinite(value):
                    return None
                return value
    except OSError as exc:
        logger.warning("Failed to read %s: %s", jsonl_path, exc)
    return None


# Extracted from 3 identical copies: m6, m613, m69
def load_f0_training_fitness_per_seed(
    campaign_root: Path,
    *,
    arms: list[str] | None = None,
) -> dict[tuple[str, int], float]:
    """Locate each (arm, seed)'s ``per_gen_elites.jsonl`` and extract F0 training fitness.

    Mirrors M6's loader. Returns ``{(arm, seed): f0_fitness}``. Missing
    files / parse errors are skipped with a warning.
    """
    arms_to_scan = (
        arms if arms is not None else [d.name for d in campaign_root.iterdir() if d.is_dir()]
    )
    out: dict[tuple[str, int], float] = {}
    for arm in arms_to_scan:
        arm_dir = campaign_root / arm
        if not arm_dir.is_dir():
            continue
        for seed_dir in sorted(arm_dir.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed-"):
                continue
            try:
                seed = int(seed_dir.name.split("-", 1)[1])
            except (IndexError, ValueError):
                logger.warning("Skipping non-seed directory: %s", seed_dir)
                continue
            direct = seed_dir / "per_gen_elites.jsonl"
            jsonl_path: Path | None = None
            if direct.exists():
                jsonl_path = direct
            else:
                candidates = [
                    p
                    for p in seed_dir.iterdir()
                    if p.is_dir() and (p / "per_gen_elites.jsonl").is_file()
                ]
                if candidates:
                    jsonl_path = (
                        max(candidates, key=lambda p: p.stat().st_mtime) / "per_gen_elites.jsonl"
                    )
            if jsonl_path is None:
                logger.warning(
                    "No per_gen_elites.jsonl for arm=%s seed=%d under %s; skipping.",
                    arm,
                    seed,
                    seed_dir,
                )
                continue
            f0_fitness = read_f0_training_fitness(jsonl_path)
            if f0_fitness is not None:
                out[(arm, seed)] = f0_fitness
    return out


# Extracted from 3 identical copies: m6, m613, m69
def build_survival_table(rows: list[dict]) -> dict[tuple[str, int, int], float]:
    """Aggregate per-episode rows into ``(arm, seed, generation) -> survival_rate``.

    ``survival_rate = 1 - (n_episodes_ending_in_HEALTH_DEPLETED / n_episodes)``.
    Mirrors ``aggregate_m6_pilot.build_survival_table``. Skips rows
    without ``termination_reason`` (backwards-compat with older CSVs).
    """
    bucket: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for row in rows:
        if "termination_reason" not in row or row["termination_reason"] == "":
            continue
        key = (str(row["arm"]), int(row["seed"]), int(row["generation"]))
        died = 1 if str(row["termination_reason"]).lower() == "health_depleted" else 0
        bucket[key].append(died)
    return {k: 1.0 - mean(v) for k, v in bucket.items()}


# Extracted from 3 identical copies: m6, m613, m69
def evaluate_decision_gate_one_seed(
    *,
    retention: dict[tuple[str, int, int], float],
    arm: str,
    seed: int,
    f0_baseline_override: dict[tuple[str, int], float] | None = None,
) -> dict:
    """Evaluate the per-arm decision gate for one (arm, seed).

    Mirrors the pure-TEI `evaluate_decision_gate_one_seed` exactly.
    The gate is on survival_rate (the primary campaign metric);
    choice_index is not used at the per-arm gate.
    """
    if f0_baseline_override is not None and (arm, seed) in f0_baseline_override:
        f0: float | None = f0_baseline_override[(arm, seed)]
    else:
        f0 = retention.get((arm, seed, 0))
    f1 = retention.get((arm, seed, 1))
    f2 = retention.get((arm, seed, 2))
    f3 = retention.get((arm, seed, 3))
    if any(v is None for v in (f0, f1, f2, f3)):
        return {
            "arm": arm,
            "seed": seed,
            "f0": f0,
            "f1": f1,
            "f2": f2,
            "f3": f3,
            "f1_ratio_pass": False,
            "f2_ratio_pass": False,
            "f3_ratio_pass": False,
            "monotone_pass": False,
            "overall_pass": False,
            "skipped": True,
            "skip_reason": "incomplete-generations",
        }
    f0_v = float(f0)  # type: ignore[arg-type]
    f1_v = float(f1)  # type: ignore[arg-type]
    f2_v = float(f2)  # type: ignore[arg-type]
    f3_v = float(f3)  # type: ignore[arg-type]
    f1_pass = f1_v >= GATE_F1_RATIO * f0_v
    f2_pass = f2_v >= GATE_F2_RATIO * f0_v
    f3_pass = f3_v >= GATE_F3_RATIO * f0_v
    monotone_pass = f0_v >= f1_v >= f2_v >= f3_v
    overall = f1_pass and f2_pass and f3_pass and monotone_pass
    return {
        "arm": arm,
        "seed": seed,
        "f0": f0_v,
        "f1": f1_v,
        "f2": f2_v,
        "f3": f3_v,
        "f1_ratio_pass": f1_pass,
        "f2_ratio_pass": f2_pass,
        "f3_ratio_pass": f3_pass,
        "monotone_pass": monotone_pass,
        "overall_pass": overall,
        "skipped": False,
        "skip_reason": "",
    }


# Extracted from 2 identical copies: m613, m69
def aggregate_per_arm_verdict(seed_evaluations: list[dict]) -> str:
    """Aggregate per-seed evaluations into a per-arm cross-seed verdict.

    GO iff ≥2 seeds pass; PIVOT iff 1; STOP otherwise. Mirrors M6.
    """
    pass_count = sum(1 for s in seed_evaluations if s["overall_pass"])
    if pass_count >= VERDICT_GO_MIN_SEEDS:
        return "GO"
    if pass_count >= VERDICT_PIVOT_MIN_SEEDS:
        return "PIVOT"
    return "STOP"


# Extracted from 2 identical copies: m613, m69
def compute_cross_arm_delta_stats(
    survival_table: dict[tuple[str, int, int], float],
    arm_a: str,
    arm_b: str,
    seeds: list[int],
    *,
    f0_baseline_override: dict[tuple[str, int], float] | None = None,
) -> dict:
    """Compute paired-seed F1+ retention deltas + Wilcoxon + bootstrap CI between two arms.

    For each seed: compute F1+ mean survival_rate per arm (averaged
    across F1, F2, F3 — the post-F0 retention window). Delta is
    ``arm_a - arm_b`` per seed. Reports:

    - Mean delta across seeds.
    - One-sided Wilcoxon signed-rank p (alternative: arm_a > arm_b).
    - 80% bootstrap CI of the mean delta (1000 resamples).

    Returns a dict with ``mean_delta``, ``wilcoxon_p``,
    ``bootstrap_ci_lo``, ``bootstrap_ci_hi``, plus the raw
    ``per_seed_deltas`` list for downstream diagnostics.
    """

    def _f1plus_mean(arm: str, seed: int) -> float | None:
        f1 = survival_table.get((arm, seed, 1))
        f2 = survival_table.get((arm, seed, 2))
        f3 = survival_table.get((arm, seed, 3))
        if any(v is None for v in (f1, f2, f3)):
            return None
        return mean([float(f1), float(f2), float(f3)])  # type: ignore[arg-type]

    per_seed_deltas: list[float] = []
    skipped_seeds: list[int] = []
    for seed in seeds:
        a_mean = _f1plus_mean(arm_a, seed)
        b_mean = _f1plus_mean(arm_b, seed)
        if a_mean is None or b_mean is None:
            skipped_seeds.append(seed)
            continue
        per_seed_deltas.append(a_mean - b_mean)
    if not per_seed_deltas:
        return {
            "arm_a": arm_a,
            "arm_b": arm_b,
            "per_seed_deltas": [],
            "mean_delta": 0.0,
            "wilcoxon_p": 1.0,
            "bootstrap_ci_lo": 0.0,
            "bootstrap_ci_hi": 0.0,
            "skipped_seeds": skipped_seeds,
            "_override_used": f0_baseline_override is not None,
        }
    mean_delta = mean(per_seed_deltas)
    # Wilcoxon signed-rank: one-sided alternative arm_a > arm_b
    # requires at least one non-zero delta. The all-zero short-circuit
    # both guards against scipy's all-zero RuntimeWarning path (which
    # would return p=1.0 anyway) and skips the function-call overhead.
    if all(abs(d) < 1e-12 for d in per_seed_deltas):
        wilcoxon_p = 1.0
    else:
        result = wilcoxon(per_seed_deltas, alternative="greater")
        # scipy returns a NamedTuple-ish ``WilcoxonResult`` with
        # ``.pvalue``; pyright can't statically introspect it, so
        # getattr keeps the runtime path intact + the type-check quiet.
        wilcoxon_p = float(getattr(result, "pvalue", 1.0))
    # Bootstrap CI: resample with replacement N times; compute mean
    # per resample; take alpha/2 and 1-alpha/2 percentiles. Seeded
    # numpy generator for reproducibility.
    rng = np.random.default_rng(42)
    arr = np.asarray(per_seed_deltas, dtype=float)
    boots = np.array(
        [
            rng.choice(arr, size=len(arr), replace=True).mean()
            for _ in range(CROSS_ARM_BOOTSTRAP_RESAMPLES)
        ],
    )
    alpha = 1.0 - CROSS_ARM_BOOTSTRAP_CI_LEVEL
    ci_lo = float(np.quantile(boots, alpha / 2))
    ci_hi = float(np.quantile(boots, 1.0 - alpha / 2))
    return {
        "arm_a": arm_a,
        "arm_b": arm_b,
        "per_seed_deltas": per_seed_deltas,
        "mean_delta": mean_delta,
        "wilcoxon_p": wilcoxon_p,
        "bootstrap_ci_lo": ci_lo,
        "bootstrap_ci_hi": ci_hi,
        "skipped_seeds": skipped_seeds,
        "_override_used": f0_baseline_override is not None,
    }


# Extracted from 2 identical copies: m613, m69
def write_cross_arm_verdict_csv(
    cross_arm_results: list[dict],
    path: Path,
) -> None:
    """Write per-arm-pair Wilcoxon + bootstrap CI stats to CSV."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "arm_a",
                "arm_b",
                "n_seeds",
                "mean_delta",
                "wilcoxon_p",
                "bootstrap_ci_lo",
                "bootstrap_ci_hi",
                "skipped_seeds",
            ],
        )
        for r in cross_arm_results:
            writer.writerow(
                [
                    r["arm_a"],
                    r["arm_b"],
                    len(r["per_seed_deltas"]),
                    f"{r['mean_delta']:.6f}",
                    f"{r['wilcoxon_p']:.6f}",
                    f"{r['bootstrap_ci_lo']:.6f}",
                    f"{r['bootstrap_ci_hi']:.6f}",
                    ";".join(str(s) for s in r["skipped_seeds"]),
                ],
            )


# Extracted from 4 copies (baldwin_retry, m2, m3, m4). The copies differed only
# in which session resolver they called: baldwin_retry used a two-layout
# ``_resolve_session``, the rest used ``latest_session``. That difference is now
# the ``resolve_session`` parameter rather than a fork in the code.
def read_history(
    seed_dir: Path,
    resolve_session: Callable[[Path], Path] = latest_session,
) -> list[dict[str, float]]:
    """Read a seed's ``history.csv`` as a list of float dicts.

    Parameters
    ----------
    seed_dir : Path
        The ``seed-N`` directory.
    resolve_session : Callable[[Path], Path]
        How to find the directory holding ``history.csv``. Defaults to the
        most-recently-modified subdirectory; pass ``resolve_session_for``
        for the layout where the file may sit directly under ``seed_dir``.
    """
    history_path = resolve_session(seed_dir) / "history.csv"
    if not history_path.exists():
        msg = f"history.csv not found at {history_path}"
        raise FileNotFoundError(msg)
    with history_path.open() as handle:
        reader = csv.DictReader(handle)
        return [{k: float(v) for k, v in row.items()} for row in reader]


# Extracted from 2 copies (baldwin_retry, baldwin_f1_postpilot_eval). They were
# structurally identical and differed only in which filename they probed for, so
# that is now a parameter.
def resolve_session_for(seed_dir: Path, filename: str) -> Path:
    """Return the directory containing ``filename`` for this seed.

    Supports both layouts: the file directly under ``seed_dir`` (the layout
    ``run_evolution.py`` has written since logbook 014), or nested one level
    under a per-session subdirectory (older runs).
    """
    if (seed_dir / filename).exists():
        return seed_dir
    sessions = sorted(p for p in seed_dir.iterdir() if p.is_dir())
    for session in sessions:
        if (session / filename).exists():
            return session
    msg = f"No {filename} found under {seed_dir} (direct or nested)"
    raise FileNotFoundError(msg)
