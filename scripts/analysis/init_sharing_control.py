#!/usr/bin/env python
"""A.1: whether block V's wiring advantage survives a shared initialisation.

Block V's learning-speed advantage -- the wild-type connectome reaching competence 23-55% sooner
than its degree-preserving rewired null under PPO -- ships with a standing condition: rewiring and
initialisation vary together. [Dhiman 2026](https://arxiv.org/abs/2604.04033) reports exactly that
advantage dissolving in the fly under shared initialisation plus a degree-preserving null.

Reading the initialisation path narrows what "shared" has to mean. The draw is per-edge over the
``(pre, post)``-sorted edge list, both graphs carry the same edge count, so they already consume the
same standard-normal stream and the nth VALUE matches. What differs is which edge the nth value
lands on. ``weight_draw`` removes that in the two ways that are defensible -- ``dense_mask`` gives
every edge present in both graphs the identical value, ``per_neuron_fanin`` gives every neuron the
identical multiset -- and neither is uniquely "the same initialisation" once the edge set changes,
which is why both run.

**This module is a manifest builder and a branch reporter, and deliberately nothing more.** It
drives ``wiring_premise`` once per draw mode and reads the interaction off its per-seed output. The
instrument is not touched: re-implementing its metrics, its gates or its minimum here would let "the
instrument changed" compete with "the effect is not there", and after the fact those are not
separable. A test asserts both committed harnesses are byte-identical to ``main``.

**The primary is the interaction, not a comparison against block V's published figures.** Those were
produced on a different seed set and a different dependency set. Crossing the draw mode with the
wiring inside one campaign makes the contrast a within-campaign one, and a draw mode that moves both
arms equally is then visibly a fact about the draw rather than about the wiring.

**Which metric carries the interaction is decided by a rule fixed before the rates are known.**
``episodes_to_30pct_success`` is right-censored at the horizon, and the interaction is a difference
of differences across four cells -- the pairing the metric requirement forbids unless censoring is
equal across those cells. So censoring is counted PER CELL, never pooled; where the rates differ the
uncensored ``auc_success`` carries the interaction and the censored metric is reported beside it.
Both are always reported.
"""

# pyright: reportPrivateUsage=false
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import wiring_premise as wp  # noqa: E402  # pyright: ignore[reportMissingImports]

# Fresh in rewiring, task and initialisation together: every campaign in the repository has used
# 1-96, and 101-104 is the programme-wide pilot band. A.1's pilot takes 105-108 so it cannot
# contaminate the registered set.
#
# 32 seeds, not the 16 D15 sets as its floor. The pilot measured no useful across-mode correlation
# (-0.66 to +0.34 on four seeds), so the panel's sensitivity is computed at rho = 0 from V.4's
# committed spread -- and at 16 seeds the minimum detectable interaction on the censored metric is
# 1.25 and 1.14 times the effect it would have to cancel, which cannot detect even a total
# dissolution. At 32 it is 0.88 and 0.81, and the uncensored metric reaches 0.65 and 0.42. A control
# that exists to answer a published critique should not be unable to see the answer.
SEEDS = tuple(range(129, 161))
PILOT_SEEDS = tuple(range(105, 109))
BURNT_SEEDS = frozenset(range(1, 97)) | frozenset(range(101, 105))

# The baseline mode is the committed block-V configuration, unchanged. It is re-run rather than
# reused: the reuse rule forbids partial reuse and requires a parsed-field identity check, and the
# dependency set moved since those runs.
BASELINE_MODE = "edge_order"
SHARING_MODES = ("dense_mask", "per_neuron_fanin")
MODES = (BASELINE_MODE, *SHARING_MODES)

_THERMAL = "connectomeppo_small_continuous2d_thermal_klinotaxis"
_HARD = "connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_hard350"
_SUFFIX_BY_MODE = {"edge_order": "", "dense_mask": "_densemask", "per_neuron_fanin": "_fanin"}


# Config stem -> (cell, arm, mode), stated explicitly rather than derived by pattern. The thermal
# arms carry `_t20` AFTER the arm part and the draw-mode suffix after that, so a regex over suffixes
# is easy to get subtly wrong, and a mis-keyed arm silently drops one side of a paired test.
def _arms_by_stem() -> dict[str, tuple[str, str, str]]:
    """Config stem -> (cell, arm, mode), built once over the three draw modes."""
    out: dict[str, tuple[str, str, str]] = {}
    for mode, sfx in _SUFFIX_BY_MODE.items():
        out |= {
            f"{_THERMAL}_t20{sfx}": ("thermal", "wt_ppo", mode),
            f"{_THERMAL}_rewired_null_t20{sfx}": ("thermal", "rn_ppo", mode),
            f"{_THERMAL}_frozen_t20{sfx}": ("thermal", "wt_frozen", mode),
            f"{_THERMAL}_rewired_null_frozen_t20{sfx}": ("thermal", "rn_frozen", mode),
            f"{_HARD}{sfx}": ("hard_food", "wt_ppo", mode),
            f"{_HARD}_rewired_null{sfx}": ("hard_food", "rn_ppo", mode),
            f"{_HARD}_frozen{sfx}": ("hard_food", "wt_frozen", mode),
            f"{_HARD}_rewired_null_frozen{sfx}": ("hard_food", "rn_frozen", mode),
        }
    return out


ARM_BY_STEM: dict[str, tuple[str, str, str]] = _arms_by_stem()

CELLS = ("thermal", "hard_food")
# The pilot runs ONE cell, as registered: it exists to show the modes run and to measure the
# across-mode correlation, not to score a contrast. The panel runs both. Stating the scope at the
# call site keeps the completeness check strict for the panel while letting the pilot be what it is.
PILOT_CELLS = ("thermal",)

# The metric the interaction is read on, decided by the censoring rule below. Both are always
# reported; this names which one carries the verdict.
CENSORED_METRIC = "episodes_to_30pct_success"
UNCENSORED_METRIC = "auc_success"

# Registered before any rate is known: if the per-cell crossing rates differ by more than this, the
# censored metric cannot carry a difference of differences and the uncensored one takes the primary.
# L.1b is why the threshold exists at all -- a 0.604 censoring spread voided a registered secondary.
CENSORING_TOLERANCE = 0.10


def build_manifest(campaign_dir: Path, path: Path, mode: str, seeds: tuple[int, ...]) -> Path:
    """Write the ``<cell> <arm> <seed> <out>`` manifest the committed harness reads, for one mode.

    Raises on an unrecognised log name rather than skipping it: a mis-keyed arm would silently drop
    one side of a paired test, which is the failure the harness's own ``ManifestError`` exists for.
    """
    log_dir = campaign_dir / "logs" if (campaign_dir / "logs").is_dir() else campaign_dir
    lines: list[str] = []
    seen: set[tuple[str, str, int]] = set()
    for log in sorted(log_dir.glob("*.log")):
        stem, _, seed_part = log.stem.rpartition("-seed")
        if not seed_part.isdigit():
            msg = f"log name has no `-seedN` suffix: {log.name}"
            raise ValueError(msg)
        if stem not in ARM_BY_STEM:
            msg = f"log names a config this panel does not have: {log.name}"
            raise ValueError(msg)
        cell, arm, log_mode = ARM_BY_STEM[stem]
        seed = int(seed_part)
        if log_mode != mode or seed not in seeds:
            continue
        key = (cell, arm, seed)
        if key in seen:
            msg = f"two logs for {cell}/{arm} seed {seed} under {mode}"
            raise ValueError(msg)
        seen.add(key)
        resolved = log.resolve()
        try:
            entry = resolved.relative_to(wp.REPO)
        except ValueError:
            entry = resolved
        lines.append(f"{cell} {arm} {seed} {entry}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def require_complete(
    manifest: Path,
    mode: str,
    seeds: tuple[int, ...],
    cells: tuple[str, ...] = CELLS,
) -> None:
    """Refuse to score a panel missing any registered cell.

    The harness reports ``missing_arms`` and drops unparseable runs with a warning, which surfaces a
    gap without stopping the scoring. A control whose consequence is restating a committed result
    should not be assigned on a partial panel.
    """
    present: dict[tuple[str, str], set[int]] = {}
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        cell, arm, seed, _ = line.split()
        present.setdefault((cell, arm), set()).add(int(seed))
    missing = [
        f"{cell}/{arm} seeds {sorted(set(seeds) - present.get((cell, arm), set()))}"
        for cell in cells
        for arm in wp.TESTED_ARMS
        if set(seeds) - present.get((cell, arm), set())
    ]
    if missing:
        msg = f"panel for {mode} is incomplete, so no branch is available: " + "; ".join(missing)
        raise ValueError(msg)


def _per_seed_gap(report: dict[str, Any], metric: str) -> dict[int, float]:
    """Wild minus rewired, per seed, on one metric -- read off the instrument's own per-seed block.

    Orientation is the instrument's: positive means the wild type is better, whichever direction the
    raw metric improves in.
    """
    per_seed = report["per_seed"]
    wild, rewired = per_seed[wp.efficiency._WILD], per_seed[wp.efficiency._REWIRED]
    # Orientation comes from the instrument's own table, so "positive means the wild type is
    # better" means here exactly what it means in every committed block-V record.
    higher_is_better = wp.efficiency._METRICS[metric]
    out: dict[int, float] = {}
    for seed in sorted(set(wild) & set(rewired)):
        delta = float(wild[seed][metric]) - float(rewired[seed][metric])
        out[int(seed)] = delta if higher_is_better else -delta
    return out


def interaction(
    baseline: dict[str, Any],
    shared: dict[str, Any],
    metric: str,
) -> dict[str, Any]:
    """Measure how much the wiring effect moves when the initialisation is shared -- the primary.

    Paired by seed, so each pair is one seed's wiring gap under a shared draw minus the same seed's
    gap under the committed one. A negative interaction means sharing shrank the advantage.
    """
    base_gap = _per_seed_gap(baseline, metric)
    shared_gap = _per_seed_gap(shared, metric)
    seeds = sorted(set(base_gap) & set(shared_gap))
    deltas = {s: shared_gap[s] - base_gap[s] for s in seeds}
    row = wp.paired_seed_wilcoxon_bootstrap([deltas[s] for s in seeds])
    return {
        "metric": metric,
        "n_pairs": len(seeds),
        "baseline_gap_mean": sum(base_gap[s] for s in seeds) / len(seeds) if seeds else None,
        "shared_gap_mean": sum(shared_gap[s] for s in seeds) / len(seeds) if seeds else None,
        "interaction_mean": sum(deltas.values()) / len(deltas) if deltas else None,
        "per_seed": deltas,
        "test": row,
    }


def censoring_rates(report: dict[str, Any]) -> dict[str, float]:
    """Per-arm crossing rate, through the instrument's own function rather than reimplemented."""
    return {
        arm: wp.crossing_rate(report, arm) for arm in (wp.efficiency._WILD, wp.efficiency._REWIRED)
    }


def choose_metric(rates_by_mode: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Apply the registered censoring rule. Fixed before any rate is known; see the module docstring.

    The censored metric may carry a difference of differences only when censoring is comparable
    across every cell of the design. Where it is not, the uncensored metric carries it and the
    censored one is reported beside it -- which is what the metric requirement asks of a departure.
    """
    observed = [r for rates in rates_by_mode.values() for r in rates.values()]
    spread = max(observed) - min(observed) if observed else 0.0
    equal = spread <= CENSORING_TOLERANCE
    return {
        "primary_metric": CENSORED_METRIC if equal else UNCENSORED_METRIC,
        "reported_beside": UNCENSORED_METRIC if equal else CENSORED_METRIC,
        "crossing_rates_by_mode": rates_by_mode,
        "crossing_rate_spread": spread,
        "tolerance": CENSORING_TOLERANCE,
        "censoring_comparable": equal,
        "why": (
            "censoring is comparable across the design's cells, so the registered censored metric "
            "carries the interaction"
            if equal
            else (
                "censoring differs across the design's cells by more than the registered "
                "tolerance, so a difference of differences on the censored metric is not "
                "interpretable and the uncensored metric carries it"
            )
        ),
    }


def score(
    campaign_dir: Path,
    out_dir: Path,
    seeds: tuple[int, ...],
    cells: tuple[str, ...] = CELLS,
) -> dict[str, Any]:
    """Drive the unmodified instrument once per draw mode and read the interactions off it."""
    reports: dict[str, dict[str, dict[str, Any]]] = {}
    for mode in MODES:
        manifest = build_manifest(campaign_dir, out_dir / f"manifest-{mode}.txt", mode, seeds)
        require_complete(manifest, mode, seeds, cells)
        reports[mode] = {}
        for cell in cells:
            # The instrument writes its own per-cell manifest into this directory and expects the
            # caller to have made it; creating it here keeps the instrument untouched.
            tmp = out_dir / f"tmp-{mode}-{cell}"
            tmp.mkdir(parents=True, exist_ok=True)
            report = wp.efficiency_contrast(manifest, cell, tmp)
            if report is None:
                msg = f"the instrument returned no efficiency report for {cell} under {mode}"
                raise ValueError(msg)
            reports[mode][cell] = report

    out: dict[str, Any] = {"seeds": list(seeds), "modes": list(MODES), "cells": {}}
    for cell in cells:
        rates = {mode: censoring_rates(reports[mode][cell]) for mode in MODES}
        choice = choose_metric(rates)
        out["cells"][cell] = {
            "metric_choice": choice,
            "interactions": {
                mode: {
                    m: interaction(reports[BASELINE_MODE][cell], reports[mode][cell], m)
                    for m in (CENSORED_METRIC, UNCENSORED_METRIC)
                }
                for mode in SHARING_MODES
            },
        }
    return out


def main() -> None:
    """CLI: build the manifests, drive the instrument per mode, report the interactions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true", help="score the pilot band, not the panel")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    seeds = PILOT_SEEDS if args.pilot else SEEDS
    cells = PILOT_CELLS if args.pilot else CELLS
    result = score(args.campaign, args.out_dir, seeds, cells)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    else:
        sys.stdout.write(text + "\n")


if __name__ == "__main__":
    main()
