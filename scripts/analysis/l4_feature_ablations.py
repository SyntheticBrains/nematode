#!/usr/bin/env python
"""L.4 + L.5: which part of the wiring carries the per-neuron effect L.1 found.

[L.1](../../docs/experiments/logbooks/066-l4-readout-width.md) read `pooling_hid_structure`: at the
per-neuron readout width the wild type leads its degree-preserving null by **+0.1852** on
`auc_success`, where at the pooled width the null led. Each ablation here removes one part of the
wiring that learner reads and asks whether the effect survives, as an **interaction against L.1's
wide baseline**::

    I_ablation = (wt_ablated - rn_ablated) - (wt_wide - rn_wide)      per seed, then the paired test

Negative: the feature carried some of the effect. Near zero: it survived. Positive: removing the
feature helped the wild type more, which is not predicted and is reported as itself.

* **L.4, `atlas`** -- `synapse_signs: atlas`: magnitudes, norms and RNG untouched, signs grounded.
  Not a clean removal: it takes the readout pool's inputs from ~50/50 to 275 E / 36 I, which can move
  the tanh operating point, so the atlas frozen floors against the wide frozen floors is a registered
  **diagnostic** that qualifies a `carries` reading as carries-or-saturates.
* **L.5, `nogap`** -- `enable_gap_junctions: false`: every parameter bitwise identical, the forward
  pass and nothing else. 199 gap junctions touch the pool, 47 lie within it.

**The minimum effect is a decision rule**, the gap L.1 recorded: `carries_the_effect` needs
significance **and** at least two-thirds of the +0.1852 removed, ``abs(I) >= 0.123``. The quantity an
ablation can remove is the wide wiring effect, not L.1's +0.2818 interaction, which includes pooled
cells no per-neuron ablation touches.

**The baseline is L.1's committed runs**, reused under a byte-identity check: one seed per reused arm
re-run under the output controls reproduced L.1's logs on every parsed field. Both halves of every
interaction pass through the same ``connectome_structure_efficiency`` call.

Everything reusable is **imported** from L.1's harness rather than copied -- ``_two_sided``,
``censoring``, ``cell_values``, the manifest writer for the baseline -- so the two cannot drift apart.
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
import l4_mixture_statistic as ms  # noqa: E402  # pyright: ignore[reportMissingImports]
import l4_readout_width as rw  # noqa: E402  # pyright: ignore[reportMissingImports]
import l4_reduced_perturbation as rp  # noqa: E402  # pyright: ignore[reportMissingImports]
from l4_panel import EXPERIMENTS, read_log  # noqa: E402  # pyright: ignore[reportMissingImports]

SEEDS = tuple(range(1, 97))
_STEM = rw._STEM
ABLATIONS = ("atlas", "nogap")
WIRINGS = ("wt", "rn")
BASELINE = "baseline"

# The eight new arms. The baseline's four come from L.1's campaign through L.1's own scanner.
ARMS: dict[str, dict[str, Any]] = {}
for _ab in ABLATIONS:
    for _w in WIRINGS:
        ARMS[f"{_w}_{_ab}"] = {"wiring": _w, "ablation": _ab, "learns": True}
        ARMS[f"{_w}_{_ab}_frozen"] = {"wiring": _w, "ablation": _ab, "learns": False}
LEARNING_ARMS = tuple(n for n, m in ARMS.items() if m["learns"])

# An optional rate tag: after the rate check (outcome B) the atlas LEARNING arms run at 0.0001 and
# are named `..._wide_atlas_r1e4{,_rewired_null}`. The tag is recorded, not part of the arm name.
_LABEL = re.compile(
    rf"^{re.escape(_STEM)}_(?P<arm>readout_only|frozen)_wide_(?P<abl>atlas|nogap)"
    r"(?P<rate>_r1e[24])?(?P<rewired>_rewired_null)?-seed(?P<seed>\d+)\.log$",
)
# L.4's rate-matched baseline: L.1's wide LEARNING arms re-run at 0.0001, one key from their parents.
_WIDE_RATE_LABEL = re.compile(
    rf"^{re.escape(_STEM)}_readout_only_wide_r1e4(?P<rewired>_rewired_null)?-seed(?P<seed>\d+)\.log$",
)

PRIMARY_METRIC = rw.PRIMARY_METRIC
CENSORED_METRIC = rw.CENSORED_METRIC
METRIC_NOTE = rw.METRIC_NOTE

# The effect an ablation can remove, and the registered minimum that reads as carrying it.
L1_WIDE_WIRING_EFFECT = 0.1852
MIN_CARRY = 0.123  # two-thirds of L1_WIDE_WIRING_EFFECT: the feature is the majority carrier
L1_INTERACTION_SD = 0.416  # L.1's realised interaction spread at n = 96, rho ~ 0.08
# The reachability floor: at n pairs the smallest one-sided exact p is 2**-n, so below 5 nothing
# can clear q = 0.05 and a verdict would be reading the seed count.
MIN_SEEDS = 5
_MIN_FOR_SPREAD = 2

REGISTERED = {
    "wide_wiring_effect": L1_WIDE_WIRING_EFFECT,
    "minimum_carry": MIN_CARRY,
    "minimum_rationale": (
        "two-thirds of the +0.1852 wide wiring effect: the feature is the majority carrier. Half "
        "(0.093) would have ~59% power at 96 seeds and need ~160 plus a re-run baseline"
    ),
    "expected_interaction_sd": L1_INTERACTION_SD,
    "n_registered": len(SEEDS),
    "power_at_minimum": "~80% (z = 2.89 at se 0.0425)",
    "why_not_l1_interaction": (
        "L.1's +0.2818 interaction includes the pooled cells, where the null led by 0.0966; no "
        "per-neuron ablation touches them, so a feature carrying ALL of the effect gives -0.185"
    ),
}

READINGS = {
    "carries_the_effect": (
        "the interaction is significantly negative and removes at least two-thirds of the wide "
        "wiring effect: this feature is the majority carrier"
    ),
    "survives_without_it": (
        "no significant interaction -- a FAILURE TO DETECT, its size and interval carried -- and the "
        "wiring effect under ablation is itself significant and positive: the effect is still there "
        "without this feature"
    ),
    "amplifies": (
        "the interaction is significantly positive: removing this feature helped the wild type "
        "more than the shuffle. Not predicted; reported, not explained"
    ),
    "inconclusive_at_this_sensitivity": (
        "the panel cannot place this feature's contribution: either the interaction is significant "
        "but below the registered minimum, or neither the interaction nor the ablated wiring effect "
        "is significant"
    ),
    "no_learning": "a learning gate failed, so the interaction is uninterpretable",
    "insufficient_seeds": (
        "too few paired seeds for any verdict: at n pairs the smallest one-sided exact p is 2**-n"
    ),
}


def scan(campaign_dir: Path, experiments: Path = EXPERIMENTS) -> dict[str, Any]:
    """Read every ablation run under a campaign directory, keyed by arm."""
    log_dir = campaign_dir / "logs" if (campaign_dir / "logs").is_dir() else campaign_dir
    runs: dict[str, dict[int, Any]] = {name: {} for name in ARMS}
    logs: dict[str, dict[int, Path]] = {}
    for log in sorted(log_dir.glob("*.log")):
        match = _LABEL.match(log.name)
        if match is None:
            print(f"  WARN: skipping log with an unrecognised label: {log.name}")
            continue
        wiring = "rn" if match.group("rewired") else "wt"
        suffix = "" if match.group("arm") == "readout_only" else "_frozen"
        name = f"{wiring}_{match.group('abl')}{suffix}"
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


def scan_rate_matched_wide(campaign_dir: Path, experiments: Path = EXPERIMENTS) -> dict[str, Any]:
    """Read the rate-matched wide LEARNING arms, keyed as L.1's scanner keys them (`wt_wide`, `rn_wide`)."""
    log_dir = campaign_dir / "logs" if (campaign_dir / "logs").is_dir() else campaign_dir
    runs: dict[str, dict[int, Any]] = {"wt_wide": {}, "rn_wide": {}}
    logs: dict[str, dict[int, Path]] = {}
    for log in sorted(log_dir.glob("*.log")):
        match = _WIDE_RATE_LABEL.match(log.name)
        if match is None:
            print(f"  WARN: skipping log with an unrecognised label: {log.name}")
            continue
        name = "rn_wide" if match.group("rewired") else "wt_wide"
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


def merge_baseline(main: dict[str, Any], learning: dict[str, Any]) -> dict[str, Any]:
    """L.4's baseline under outcome B: learning arms from the rate-matched run, floors from L.1.

    The floors are reused because the rate is inert under ``freeze_updates``; the learning arms are
    not, because an interaction must compare learners at ONE rate.
    """
    runs = {
        "wt_wide": learning["runs"]["wt_wide"],
        "rn_wide": learning["runs"]["rn_wide"],
        "wt_wide_frozen": main["runs"]["wt_wide_frozen"],
        "rn_wide_frozen": main["runs"]["rn_wide_frozen"],
    }
    logs = {
        "wt_wide": learning["logs"].get("wt_wide", {}),
        "rn_wide": learning["logs"].get("rn_wide", {}),
        "wt_wide_frozen": main["logs"].get("wt_wide_frozen", {}),
        "rn_wide_frozen": main["logs"].get("rn_wide_frozen", {}),
    }
    return {"runs": runs, "logs": logs}


def require_complete(scanned: dict[str, Any], seeds: tuple[int, ...] = SEEDS) -> None:
    """Refuse to score a panel missing any registered cell."""
    runs = scanned["runs"]
    missing = [
        f"{name} seeds {[s for s in seeds if s not in runs[name]]}"
        for name in ARMS
        if [s for s in seeds if s not in runs[name]]
    ]
    if missing:
        msg = "campaign is incomplete, so no reading is available: " + "; ".join(missing)
        raise ValueError(msg)


def common_seeds(
    scanned: dict[str, Any],
    baselines: dict[str, dict[str, Any]],
    seeds: tuple[int, ...] = SEEDS,
) -> tuple[int, ...]:
    """Return the seeds present in every ablation cell AND every cell of every baseline."""
    return tuple(
        s
        for s in seeds
        if all(s in scanned["runs"][n] for n in ARMS)
        and all(
            s in b["runs"][n]
            for b in baselines.values()
            for n in ("wt_wide", "rn_wide", "wt_wide_frozen", "rn_wide_frozen")
        )
    )


def write_manifest(
    scanned: dict[str, Any],
    ablation: str,
    path: Path,
    seeds: tuple[int, ...] = SEEDS,
) -> Path:
    """Write the paired ``arm seed out`` manifest for one ablation's two learning arms."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for wiring, arm in (("wt", eff._WILD), ("rn", eff._REWIRED)):
        name = f"{wiring}_{ablation}"
        for seed in seeds:
            log = scanned["logs"].get(name, {}).get(seed)
            if log is None:
                msg = f"no log for {name} seed {seed}; the manifest would be unpaired"
                raise ValueError(msg)
            lines.append(f"{arm} {seed} {log.resolve().relative_to(rp.ps.hm.REPO)}")
    path.write_text("\n".join(lines) + "\n")
    return path


def efficiency(
    scanned: dict[str, Any],
    baselines: dict[str, dict[str, Any]],
    tmp_dir: Path,
    seeds: tuple[int, ...] = SEEDS,
) -> dict[str, Any]:
    """Score each ablation AND its baseline through the committed efficiency harness, unmodified.

    The baseline half of every interaction is scored through the same call as the ablated half, from
    campaign logs rather than JSON, so both halves pass through one code path. One baseline report
    per ablation: under outcome B, L.4's is rate-matched at 0.0001 and L.5's is L.1's at 0.001.
    """
    reports: dict[str, Any] = {}
    for ablation in ABLATIONS:
        manifest = write_manifest(scanned, ablation, tmp_dir / f"_efficiency_{ablation}.txt", seeds)
        reports[ablation] = eff.analyse(manifest)
        manifest = rw.write_manifest(
            baselines[ablation],
            "wide",
            tmp_dir / f"_efficiency_{BASELINE}_{ablation}.txt",
            seeds,
        )
        reports[f"{BASELINE}_{ablation}"] = eff.analyse(manifest)
    return reports


def contrasts(
    cells: dict[str, dict[int, float]],
    ablation: str,
    seeds: tuple[int, ...] = SEEDS,
) -> dict[str, Any]:
    """Return one ablation's interaction and both wiring effects, per seed, on the metric given."""
    keys = (
        f"wt_{ablation}",
        f"rn_{ablation}",
        f"wt_{BASELINE}_{ablation}",
        f"rn_{BASELINE}_{ablation}",
    )
    common = [s for s in seeds if all(s in cells[k] for k in keys)]
    ablated = [cells[f"wt_{ablation}"][s] - cells[f"rn_{ablation}"][s] for s in common]
    base = [
        cells[f"wt_{BASELINE}_{ablation}"][s] - cells[f"rn_{BASELINE}_{ablation}"][s]
        for s in common
    ]
    interaction = [a - b for a, b in zip(ablated, base, strict=True)]
    return {
        "ablation": ablation,
        "n_common": len(common),
        "seeds": common,
        "interaction": {"role": "primary", **rw._two_sided(interaction)},
        "ablated_wiring_effect": {"role": "secondary", **rw._two_sided(ablated)},
        "baseline_wiring_effect": {"role": "reference", **rw._two_sided(base)},
        "cell_means": {k: float(np.mean([cells[k][s] for s in common])) for k in keys}
        if common
        else {},
    }


def gates(
    scanned: dict[str, Any],
    baseline: dict[str, Any],
    ablation: str,
    seeds: tuple[int, ...] = SEEDS,
) -> dict[str, Any]:
    """Each ablated learning arm against its OWN ablated floor, the ablated prior, and the diagnostic.

    The L.4 floors diagnostic -- each atlas floor against the wide floor of the same wiring -- is
    computed for both ablations for symmetry, but only atlas is registered to qualify a reading.
    """
    runs, brs = scanned["runs"], baseline["runs"]

    def foods(src: dict[str, Any], name: str) -> dict[int, float]:
        return {s: r.foods for s, r in src[name].items() if s in seeds}

    out: dict[str, Any] = {}
    for w in WIRINGS:
        out[f"{w}_gate"] = {
            "arms": [f"{w}_{ablation}", f"{w}_{ablation}_frozen"],
            "kind": "gate",
            **ms.shift_contrast(
                foods(runs, f"{w}_{ablation}"),
                foods(runs, f"{w}_{ablation}_frozen"),
            ),
        }
    prior = ms.shift_contrast(
        foods(runs, f"wt_{ablation}_frozen"),
        foods(runs, f"rn_{ablation}_frozen"),
    )
    out["prior"] = {
        "arms": [f"wt_{ablation}_frozen", f"rn_{ablation}_frozen"],
        "kind": "prior",
        **prior,
        "p_two_sided": min(1.0, 2.0 * min(prior["p_improve"], prior["p_degrade"])),
    }
    # The floors diagnostic: did the ablation move the frozen substrate's operating point?
    diag: dict[str, Any] = {}
    for w in WIRINGS:
        c = ms.shift_contrast(foods(runs, f"{w}_{ablation}_frozen"), foods(brs, f"{w}_wide_frozen"))
        diag[w] = {**c, "p_two_sided": min(1.0, 2.0 * min(c["p_improve"], c["p_degrade"]))}
    qs = ms.bh_fdr([diag[w]["p_two_sided"] for w in WIRINGS])
    for w, q in zip(WIRINGS, qs, strict=True):
        diag[w]["q"] = float(q)
    out["floors_diagnostic"] = {
        **diag,
        "fires": bool(any(diag[w]["q"] <= ms.SIG_Q for w in WIRINGS)),
        "registered_for": "atlas",
        "note": (
            "ablated frozen floors against the wide frozen floors, per wiring, two-sided, outside the "
            "family. Arms in which nothing learns, so a difference is the operating point. Fires: a "
            "carries reading on atlas is qualified carries-or-saturates. It qualifies, never rescues"
        ),
    }
    # The GAINS diagnostic, registered after the pilot (2026-09-18) and before the campaign. The
    # floors diagnostic compares floors and was quiet on the pilot -- but the atlas arms' gains OVER
    # those floors were +2.2 and +2.8 foods against the wide arms' +13.7 and +7.8 at the same seeds,
    # bimodally (two seeds below their own floor, two learning). An ablation that removes
    # learnability from BOTH wirings produces a negative interaction because the wild type had more
    # to lose, and reads `carries` for the wrong reason. So: each ablated arm's gain over its floor
    # against the wide arm's gain over its floor, paired per seed, per wiring. Both significantly
    # smaller: a carries on atlas is qualified carries-or-unlearnable. It qualifies, never rescues.
    gains: dict[str, Any] = {}
    for w in WIRINGS:
        abl_gain = {
            s: runs[f"{w}_{ablation}"][s].foods - runs[f"{w}_{ablation}_frozen"][s].foods
            for s in seeds
            if s in runs[f"{w}_{ablation}"] and s in runs[f"{w}_{ablation}_frozen"]
        }
        wide_gain = {
            s: brs[f"{w}_wide"][s].foods - brs[f"{w}_wide_frozen"][s].foods
            for s in seeds
            if s in brs[f"{w}_wide"] and s in brs[f"{w}_wide_frozen"]
        }
        c = ms.shift_contrast(abl_gain, wide_gain)
        gains[w] = {**c, "p_two_sided": min(1.0, 2.0 * min(c["p_improve"], c["p_degrade"]))}
    qs = ms.bh_fdr([gains[w]["p_two_sided"] for w in WIRINGS])
    for w, q in zip(WIRINGS, qs, strict=True):
        gains[w]["q"] = float(q)
    out["gains_diagnostic"] = {
        **gains,
        "fires": bool(
            all(gains[w]["q"] <= ms.SIG_Q and gains[w].get("effect", 0.0) < 0.0 for w in WIRINGS),
        ),
        "registered_for": "atlas",
        "registered_on": "2026-09-18, after the pilot and before the campaign",
        "note": (
            "each ablated arm's gain over its floor against the wide arm's gain over its floor, "
            "paired per seed, per wiring, two-sided, outside the family. Both significantly smaller: "
            "the ablation removed learnability from both wirings, and a carries reading on atlas is "
            "qualified carries-or-unlearnable. It qualifies, never rescues"
        ),
    }
    return out


def adjust_family(per_ablation: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Correct all ten registered tests under one BH-FDR family, in place.

    Per ablation: interaction, ablated wiring effect, two gates, prior. The two interactions share
    the baseline half and are positively dependent, under which BH-FDR holds.
    """
    labels: list[tuple[str, str]] = []
    ps: list[float] = []
    for ab, entry in per_ablation.items():
        for key in ("interaction", "ablated_wiring_effect"):
            labels.append((ab, key))
            ps.append(float(entry["contrasts"][key]["p_two_sided"]))
        for w in WIRINGS:
            labels.append((ab, f"{w}_gate"))
            ps.append(float(entry["gates"][f"{w}_gate"]["p_improve"]))
        labels.append((ab, "prior"))
        ps.append(float(entry["gates"]["prior"]["p_two_sided"]))
    qs = [float(q) for q in ms.bh_fdr(ps)]
    for (ab, key), q in zip(labels, qs, strict=True):
        target = (
            per_ablation[ab]["contrasts"]
            if key in ("interaction", "ablated_wiring_effect")
            else per_ablation[ab]["gates"]
        )
        target[key]["q"] = q
    for entry in per_ablation.values():
        entry["gates"]["gates_pass"] = bool(
            all(entry["gates"][f"{w}_gate"]["q"] <= ms.SIG_Q for w in WIRINGS),
        )
        entry["gates"]["prior_separates"] = bool(entry["gates"]["prior"]["q"] <= ms.SIG_Q)
    return {"n_tests": len(ps), "raw_p": ps, "q": qs, "labels": [f"{a}:{k}" for a, k in labels]}


def reading(ablation: str, entry: dict[str, Any]) -> dict[str, Any]:
    """Assign one ablation's registered reading: gates first, then the interaction, then the minimum."""
    n = entry["contrasts"]["n_common"]
    inter = entry["contrasts"]["interaction"]
    ablated = entry["contrasts"]["ablated_wiring_effect"]
    delta = float(inter["mean_delta"])
    q = float(inter.get("q", inter["p_two_sided"]))
    sig = q <= ms.SIG_Q
    qualified = False
    unlearnable = False
    if n < MIN_SEEDS:
        name = "insufficient_seeds"
    elif not entry["gates"]["gates_pass"]:
        name = "no_learning"
    elif sig and delta < 0.0 and abs(delta) >= MIN_CARRY:
        name = "carries_the_effect"
        qualified = ablation == "atlas" and bool(entry["gates"]["floors_diagnostic"]["fires"])
        unlearnable = ablation == "atlas" and bool(
            entry["gates"].get("gains_diagnostic", {}).get("fires", False),
        )
    elif sig and delta > 0.0:
        name = "amplifies"
    elif sig:
        name = "inconclusive_at_this_sensitivity"  # significant but below the minimum
    elif (
        float(ablated.get("q", ablated["p_two_sided"])) <= ms.SIG_Q
        and float(ablated["mean_delta"]) > 0.0
    ):
        name = "survives_without_it"
    else:
        name = "inconclusive_at_this_sensitivity"
    return {
        "ablation": ablation,
        "reading": name,
        "why": READINGS[name],
        "qualified_carries_or_saturates": qualified,
        "qualified_carries_or_unlearnable": unlearnable,
        "interaction_delta": delta,
        "interaction_ci": [float(inter["ci_lo"]), float(inter["ci_hi"])],
        "interaction_q": q,
        "minimum_carry": MIN_CARRY,
        "clears_minimum": bool(abs(delta) >= MIN_CARRY),
        "fraction_of_wide_effect_removed": float(-delta / L1_WIDE_WIRING_EFFECT),
        "gates_read_first": True,
    }


def sensitivity(deltas: list[float], n_pairs: int) -> dict[str, Any]:
    """Report the realised spread against the registered expectation."""
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
        "power_at_minimum_normal_approx": float(_power(MIN_CARRY, se)),
        "registered": dict(REGISTERED),
    }


def _power(effect: float, se: float) -> float:
    """Two-sided 5% normal-approximation power at ``effect``."""
    from math import erf, sqrt

    z = effect / se - 1.96
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def analyse(
    campaign_dir: Path,
    baseline_dir: Path,
    seeds: tuple[int, ...] = SEEDS,
    experiments: Path = EXPERIMENTS,
    baseline_atlas_dir: Path | None = None,
) -> dict[str, Any]:
    """Score both ablations: gates first, then each interaction, one BH family, read per ablation.

    ``baseline_atlas_dir`` holds L.4's rate-matched wide LEARNING arms (outcome B); its floors are
    L.1's from ``baseline_dir``. Without it, both ablations read against L.1's baseline.
    """
    scanned = scan(campaign_dir, experiments)
    main = rw.scan(baseline_dir, experiments)
    baselines: dict[str, dict[str, Any]] = dict.fromkeys(ABLATIONS, main)
    sources = dict.fromkeys(ABLATIONS, str(baseline_dir))
    if baseline_atlas_dir is not None:
        baselines["atlas"] = merge_baseline(
            main,
            scan_rate_matched_wide(baseline_atlas_dir, experiments),
        )
        sources["atlas"] = (
            f"learning arms {baseline_atlas_dir} (rate-matched, 0.0001); floors {baseline_dir}"
        )
    seeds = common_seeds(scanned, baselines, seeds)
    reports = efficiency(scanned, baselines, campaign_dir, seeds)
    primary_cells = rw.cell_values(reports, PRIMARY_METRIC)
    censored_cells = rw.cell_values(reports, CENSORED_METRIC)
    censored_hib = bool(
        reports[f"{BASELINE}_{ABLATIONS[0]}"]["metrics"][CENSORED_METRIC]["higher_is_better"],
    )
    per: dict[str, dict[str, Any]] = {}
    for ab in ABLATIONS:
        per[ab] = {
            "contrasts": contrasts(primary_cells, ab, seeds),
            "censored_axis": contrasts(censored_cells, ab, seeds),
            "gates": gates(scanned, baselines[ab], ab, seeds),
        }
    family = adjust_family(per)
    for ab in ABLATIONS:
        entry = per[ab]
        entry["reading"] = reading(ab, entry)
        p_d = entry["contrasts"]["interaction"]["mean_delta"]
        c_d = entry["censored_axis"]["interaction"]["mean_delta"]
        entry["metrics_agree_in_direction"] = bool(
            p_d * c_d * (1.0 if censored_hib else -1.0) >= 0.0,
        )
        entry["sensitivity"] = sensitivity(
            entry["contrasts"]["interaction"]["per_seed_deltas"],
            entry["contrasts"]["n_common"],
        )
    return {
        "seeds": list(seeds),
        "n_seeds_scored": len(seeds),
        "primary_metric": PRIMARY_METRIC,
        "censored_metric": CENSORED_METRIC,
        "censored_metric_higher_is_better": censored_hib,
        "metric_note": METRIC_NOTE,
        "baseline": {
            "sources": sources,
            "note": (
                "L.1's committed wide arms, reused under the byte-identity check recorded in "
                "launch.md; under outcome B, L.4's learning half is the rate-matched re-run at "
                "0.0001 with L.1's floors"
            ),
        },
        "family": family,
        "censoring": rw.censoring(reports),
        "ablations": per,
        "readings": {ab: per[ab]["reading"]["reading"] for ab in ABLATIONS},
        "read_per_ablation_never_pooled": True,
        "registered": dict(REGISTERED),
        "efficiency": reports,
    }


def _print(result: dict[str, Any]) -> None:
    """Print each ablation's gates, its 2x2, its interaction and its reading."""
    print("\n" + "=" * 78)
    print("L.4 + L.5 - feature ablations at the per-neuron width, hard350")
    print("=" * 78)
    for ab in ABLATIONS:
        e = result["ablations"][ab]
        g, c, r = e["gates"], e["contrasts"], e["reading"]
        print(f"\n  [{ab.upper()}]  n={c['n_common']} paired")
        for w in WIRINGS:
            x = g[f"{w}_gate"]
            print(f"    gate {w}     d={x.get('effect', float('nan')):+8.3f} foods  q={x['q']:.3f}")
        print(
            f"    prior       d={g['prior'].get('effect', float('nan')):+8.3f} foods  q={g['prior']['q']:.3f}",
        )
        gd = g.get("gains_diagnostic")
        if gd:
            print(
                f"    gains diag  wt d={gd['wt'].get('effect', float('nan')):+7.3f} q={gd['wt']['q']:.3f}"
                f"  rn d={gd['rn'].get('effect', float('nan')):+7.3f} q={gd['rn']['q']:.3f}"
                f"  -> {'FIRES' if gd['fires'] else 'quiet'}",
            )
        d = g["floors_diagnostic"]
        print(
            f"    floors diag wt d={d['wt'].get('effect', float('nan')):+7.3f} q={d['wt']['q']:.3f}"
            f"  rn d={d['rn'].get('effect', float('nan')):+7.3f} q={d['rn']['q']:.3f}"
            f"  -> {'FIRES' if d['fires'] else 'quiet'}",
        )
        m = c["cell_means"]
        print(
            f"    cells: wt_{ab} {m.get(f'wt_{ab}', float('nan')):.4f}  rn_{ab} {m.get(f'rn_{ab}', float('nan')):.4f}"
            f"  | baseline wt {m.get(f'wt_{BASELINE}_{ab}', float('nan')):.4f}"
            f"  rn {m.get(f'rn_{BASELINE}_{ab}', float('nan')):.4f}",
        )
        i = c["interaction"]
        print(
            f"    INTERACTION d={i['mean_delta']:+.4f}  CI[{i['ci_lo']:+.4f},{i['ci_hi']:+.4f}]  q={i['q']:.3f}"
            f"  removes {r['fraction_of_wide_effect_removed']:+.0%} of the wide effect  (min {MIN_CARRY})",
        )
        a = c["ablated_wiring_effect"]
        print(f"    ablated wiring effect d={a['mean_delta']:+.4f}  q={a['q']:.3f}   [secondary]")
        print(f"    metrics agree in direction: {e['metrics_agree_in_direction']}")
        print(
            f"    READING: {r['reading'].upper().replace('_', '-')}"
            + ("  (qualified: carries OR saturates)" if r["qualified_carries_or_saturates"] else "")
            + (
                "  (qualified: carries OR unlearnable)"
                if r["qualified_carries_or_unlearnable"]
                else ""
            ),
        )
        print(f"      {r['why']}")
    print(
        f"\n  {result['family']['n_tests']} tests in one BH-FDR family; read per ablation, never pooled.",
    )


def write_csv(result: dict[str, Any], path: Path) -> None:
    """One row per condition, wiring and seed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    reports = result["efficiency"]
    metrics = sorted(next(iter(next(iter(reports.values()))["per_seed"][eff._WILD].values())))
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition", "wiring", "seed", *metrics, "primary_censored"])
        for cond, report in reports.items():
            horizon = report["horizon_episodes"]
            for wiring, arm in (("wt", eff._WILD), ("rn", eff._REWIRED)):
                rows = report["per_seed"][arm]
                for seed in sorted(rows, key=int):
                    row = rows[seed]
                    writer.writerow(
                        [
                            cond,
                            wiring,
                            seed,
                            *(f"{row[m]:.6f}" for m in metrics),
                            int(row[CENSORED_METRIC] >= horizon),
                        ],
                    )


def main(argv: list[str] | None = None) -> int:
    """Score both ablations against L.1's baseline and report each reading."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=Path("campaigns/readout-width"))
    parser.add_argument(
        "--baseline-atlas",
        type=Path,
        default=None,
        help="L.4's rate-matched wide learning arms (outcome B); floors still come from --baseline",
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
    if not args.allow_incomplete:
        require_complete(scan(args.campaign), seeds)
    result = analyse(args.campaign, args.baseline, seeds, baseline_atlas_dir=args.baseline_atlas)
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
