#!/usr/bin/env python
"""L.1: whether the four-class pooling hid the wiring's features from the learner.

[L.0](../../docs/experiments/logbooks/064-l4-frozen-features.md) returned
`wiring_is_inert_as_features` under `readout_only`, whose readout is a **mean-pool over four motor
classes into an 8-parameter map**: 39 motor neurons in, four numbers out, each neuron carrying
``1/|class|`` of its class's influence. If the wild type's advantage lives in *which neuron* fires
rather than *which class*, that pool destroys it before the learner sees it.

**The question is an INTERACTION, and that is the whole design.** A per-neuron readout has **78
parameters against 8**, so comparing it to L.0's pooled arm and reading a gain would confound "the
wiring's features were there and the pool hid them" with "ten times the parameters learn faster on
any features at all". Crossing width with wiring separates them::

                    pooled (8)      per-neuron (78)
    wild type            A                 B
    rewired null         C                 D

    width main effect   (B + D) - (A + C)   more parameters help -- uninformative alone
    wiring main effect  (A + B) - (C + D)   L.0 measured the A - C half
    INTERACTION         (B - A) - (D - C)   does widening help the WILD TYPE more?

Only the interaction distinguishes the two stories, so it is the primary and everything else here
exists to make it readable.

**The primary metric is ``auc_success``, not block V's ``episodes_to_30pct_success``**, and the
departure is registered with its reason. L.0 met **asymmetric censoring** on this exact cell: five
wild-type seeds of 32 never reached competence within the horizon against one for the null. A
difference of differences cannot be read on a metric whose censoring rate differs across the cells
being differenced -- widening the readout is *expected* to reduce censoring, which would move the
interaction for a reason unrelated to the wiring. ``auc_success`` is defined for every seed at the
same horizon. The censored metric is reported beside it **with its censoring counted per cell**.

The metrics themselves are not computed here: ``connectome_structure_efficiency`` is called **once
per width**, with this change's wild and rewired arms mapped onto its own two, which yields per-seed
values for all four cells with that committed module unmodified. Re-deriving AUC here would
duplicate a committed metric.
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
import l4_reduced_perturbation as rp  # noqa: E402  # pyright: ignore[reportMissingImports]
from l4_panel import EXPERIMENTS, read_log  # noqa: E402  # pyright: ignore[reportMissingImports]
from weight_search_architecture_ranking import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    paired_seed_wilcoxon_bootstrap,
)

SEEDS = tuple(range(1, 97))
_STEM = "connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_hard350_eprop"

WIDTHS = ("pooled", "wide")
WIRINGS = ("wt", "rn")

# The eight arms. `efficiency_arm` is the name the committed efficiency script requires of the pair
# it scores; the floors do not enter it, being the gates' business rather than the contrast's.
ARMS: dict[str, dict[str, Any]] = {
    "wt_pooled": {"wiring": "wt", "width": "pooled", "learns": True, "eff": eff._WILD},
    "rn_pooled": {"wiring": "rn", "width": "pooled", "learns": True, "eff": eff._REWIRED},
    "wt_wide": {"wiring": "wt", "width": "wide", "learns": True, "eff": eff._WILD},
    "rn_wide": {"wiring": "rn", "width": "wide", "learns": True, "eff": eff._REWIRED},
    "wt_pooled_frozen": {"wiring": "wt", "width": "pooled", "learns": False, "eff": None},
    "rn_pooled_frozen": {"wiring": "rn", "width": "pooled", "learns": False, "eff": None},
    "wt_wide_frozen": {"wiring": "wt", "width": "wide", "learns": False, "eff": None},
    "rn_wide_frozen": {"wiring": "rn", "width": "wide", "learns": False, "eff": None},
}
LEARNING_ARMS = tuple(n for n, m in ARMS.items() if m["learns"])

_LABEL = re.compile(
    rf"^{re.escape(_STEM)}_(?P<arm>readout_only|frozen)(?P<wide>_wide)?"
    r"(?P<rewired>_rewired_null)?-seed(?P<seed>\d+)\.log$",
)

# The primary, and the censored metric reported beside it. Registered before the campaign ran.
PRIMARY_METRIC = "auc_success"
CENSORED_METRIC = "episodes_to_30pct_success"
METRIC_NOTE = (
    "auc_success is the primary because the interaction is a difference of differences and L.0 met "
    "ASYMMETRIC censoring on this cell -- five wild-type seeds of 32 never competent against one "
    "for the null. Widening the readout is expected to reduce censoring, so a right-censored "
    "primary would move for a reason unrelated to the wiring. episodes_to_30pct_success is reported "
    "beside it with its censoring counted PER CELL, and a disagreement between them is reported"
)

# L.0's committed figures on the pooled cells of this 2x2, for the record to be read against. Never
# a quantitative delta against this panel's own pooled arms -- those are re-run here.
L0_REFERENCE = {
    "verdict": "wiring_is_inert_as_features",
    "wt_median_episodes": 568.0,
    "rn_median_episodes": 405.0,
    "wt_never_competent": 5,
    "rn_never_competent": 1,
    "note": "L.0 read episodes_to_30pct_success; this panel reads auc_success as its primary",
}

READINGS = {
    "pooling_hid_structure": (
        "widening the readout helps the WILD TYPE more than the shuffle: the four-class pool was "
        "hiding wiring-specific structure, and L.4 and L.5 reopen as their gates registered"
    ),
    "width_favours_the_shuffle": (
        "widening helps the NULL more -- L.0's reverse-direction lean strengthens at width. "
        "Reported as the reverse direction and not explained"
    ),
    "pooling_was_not_the_limit": (
        "no interaction at this panel's sensitivity: more parameters may help, but they do not help "
        "the wild type more. L.0's verdict stands and the width objection is retired at the stated "
        "sensitivity; L.4 and L.5 stay closed-unopened"
    ),
    "no_learning": (
        "a learning gate failed, so an arm carrying the claim did not learn and the interaction is "
        "uninterpretable"
    ),
    "insufficient_seeds": "too few paired seeds survived to score the panel",
}


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
        wiring = "rn" if match.group("rewired") else "wt"
        width = "wide" if match.group("wide") else "pooled"
        suffix = "" if match.group("arm") == "readout_only" else "_frozen"
        name = f"{wiring}_{width}{suffix}"
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

    A 2x2 read as an interaction needs all four cells at every seed: a seed present in three cells
    contributes nothing and a seed silently dropped from one shifts the difference of differences.
    """
    runs = scanned["runs"]
    missing = [
        f"{name} seeds {[s for s in seeds if s not in runs[name]]}"
        for name in ARMS
        if [s for s in seeds if s not in runs[name]]
    ]
    if missing:
        msg = "campaign is incomplete, so no reading is available: " + "; ".join(missing)
        raise ValueError(msg)


def write_manifest(
    scanned: dict[str, Any],
    width: str,
    path: Path,
    seeds: tuple[int, ...] = SEEDS,
) -> Path:
    """Write the paired ``arm seed out`` manifest the efficiency script consumes, for one width.

    One manifest per width, each holding that width's two LEARNING arms under the names the
    committed script requires. Calling it twice is what gives all four cells of the 2x2 their
    per-seed metrics without touching it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for name, meta in ARMS.items():
        if meta["eff"] is None or meta["width"] != width:
            continue
        for seed in seeds:
            log = scanned["logs"].get(name, {}).get(seed)
            if log is None:
                msg = f"no log for {name} seed {seed}; the manifest would be unpaired"
                raise ValueError(msg)
            lines.append(f"{meta['eff']} {seed} {log.resolve().relative_to(rp.ps.hm.REPO)}")
    path.write_text("\n".join(lines) + "\n")
    return path


def efficiency(
    scanned: dict[str, Any],
    tmp_dir: Path,
    seeds: tuple[int, ...] = SEEDS,
) -> dict[str, Any]:
    """Score each width through the committed efficiency harness, unmodified."""
    reports: dict[str, Any] = {}
    for width in WIDTHS:
        manifest = write_manifest(scanned, width, tmp_dir / f"_efficiency_{width}.txt", seeds)
        reports[width] = eff.analyse(manifest)
    return reports


def cell_values(reports: dict[str, Any], metric: str) -> dict[str, dict[int, float]]:
    """Return ``{cell: {seed: value}}`` for the 2x2's four cells, from the harness's own per-seed."""
    out: dict[str, dict[int, float]] = {}
    for width, report in reports.items():
        for wiring, arm in (("wt", eff._WILD), ("rn", eff._REWIRED)):
            rows = report["per_seed"][arm]
            out[f"{wiring}_{width}"] = {int(s): float(r[metric]) for s, r in rows.items()}
    return out


def _two_sided(deltas: list[float]) -> dict[str, Any]:
    """One-sided each way plus a real two-sided p, on a contrast with two registered directions.

    ``min(p_up, p_down)`` is not a p-value and doubles the type-I rate -- the defect PR #375 caught
    in L.0's prior check. The two-sided p of an exact one-sided pair is ``2 * min``, capped at 1.
    """
    up = paired_seed_wilcoxon_bootstrap(deltas)
    down = paired_seed_wilcoxon_bootstrap([-d for d in deltas])
    p_two = min(1.0, 2.0 * min(float(up["wilcoxon_p"]), float(down["wilcoxon_p"])))
    return {
        **up,
        "p_wild_type_gains_more": float(up["wilcoxon_p"]),
        "p_null_gains_more": float(down["wilcoxon_p"]),
        "p_two_sided": float(p_two),
        "positive_seeds": int(sum(1 for d in deltas if d > 0.0)),
    }


def contrasts(
    cells: dict[str, dict[int, float]],
    seeds: tuple[int, ...] = SEEDS,
) -> dict[str, Any]:
    """Return the interaction and both main effects, per seed, on the metric passed in.

    The interaction is ``(wt_wide - wt_pooled) - (rn_wide - rn_pooled)`` at each seed: positive means
    widening bought the wild type more than it bought the degree-matched shuffle. The main effects
    are reported beside it and are **not** evidence about the wiring on their own -- a width main
    effect is a fact about the parameter count.
    """
    common = [s for s in seeds if all(s in cells[c] for c in cells)]
    interaction = [
        (cells["wt_wide"][s] - cells["wt_pooled"][s])
        - (cells["rn_wide"][s] - cells["rn_pooled"][s])
        for s in common
    ]
    width_effect = [
        ((cells["wt_wide"][s] + cells["rn_wide"][s]) / 2.0)
        - ((cells["wt_pooled"][s] + cells["rn_pooled"][s]) / 2.0)
        for s in common
    ]
    wiring_effect = [
        ((cells["wt_wide"][s] + cells["wt_pooled"][s]) / 2.0)
        - ((cells["rn_wide"][s] + cells["rn_pooled"][s]) / 2.0)
        for s in common
    ]
    return {
        "n_common": len(common),
        "seeds": common,
        "interaction": {"role": "primary", **_two_sided(interaction)},
        "width_main_effect": {"role": "secondary", **_two_sided(width_effect)},
        "wiring_main_effect": {"role": "secondary", **_two_sided(wiring_effect)},
        "cell_means": {c: float(np.mean([v[s] for s in common])) for c, v in cells.items()}
        if common
        else {},
        "per_width_wiring_effect": {
            width: float(
                np.mean([cells[f"wt_{width}"][s] - cells[f"rn_{width}"][s] for s in common]),
            )
            for width in WIDTHS
        }
        if common
        else {},
    }


def censoring(reports: dict[str, Any]) -> dict[str, Any]:
    """Count the censored metric's censoring PER CELL, which is why it is not the primary.

    A cell at the horizon has not "taken 3000 episodes"; it has not been measured. Pooling that
    across the 2x2 is what would let a censoring change read as an interaction.
    """
    out: dict[str, Any] = {}
    for width, report in reports.items():
        horizon = report["horizon_episodes"]
        for wiring, arm in (("wt", eff._WILD), ("rn", eff._REWIRED)):
            rows = report["per_seed"][arm]
            censored = [int(s) for s, r in rows.items() if r[CENSORED_METRIC] >= horizon]
            out[f"{wiring}_{width}"] = {
                "n_censored": len(censored),
                "n_seeds": len(rows),
                "censored_seeds": sorted(censored),
                "rate": len(censored) / len(rows) if rows else 0.0,
            }
    rates = [c["rate"] for c in out.values()]
    out["max_rate_difference"] = float(max(rates) - min(rates)) if rates else 0.0
    out["asymmetric"] = bool(out["max_rate_difference"] > 0.0)
    out["note"] = (
        "counted per cell rather than pooled. Any difference between cells is why the censored "
        "metric is not the primary; where the two metrics disagree, the disagreement is reported"
    )
    return out


def gates(scanned: dict[str, Any], seeds: tuple[int, ...] = SEEDS) -> dict[str, Any]:
    """Each learning arm against its OWN frozen floor, plus the untrained prior. Read FIRST.

    Four gates, not two: each width has its own floors, because the widths are the same policy but
    not the same run -- they agree on the action mean to ~1e-8 and the action is sampled around it,
    so a shared floor would be a control the wide arms never shared a trajectory with.
    """
    runs = scanned["runs"]

    def foods(name: str) -> dict[int, float]:
        return {s: r.foods for s, r in runs[name].items() if s in seeds}

    tests: dict[str, tuple[str, str, str]] = {
        f"{arm}_gate": (arm, f"{arm}_frozen", "gate") for arm in LEARNING_ARMS
    }
    out: dict[str, Any] = {}
    for label, (a, b, kind) in tests.items():
        out[label] = {"arms": [a, b], "kind": kind, **ms.shift_contrast(foods(a), foods(b))}
    # The prior is asked at each width, between wirings, before anything learned.
    priors = {f"prior_{width}": (f"wt_{width}_frozen", f"rn_{width}_frozen") for width in WIDTHS}
    for label, (a, b) in priors.items():
        contrast = ms.shift_contrast(foods(a), foods(b))
        two_sided = min(1.0, 2.0 * min(contrast["p_improve"], contrast["p_degrade"]))
        out[label] = {"arms": [a, b], "kind": "prior", **contrast, "p_two_sided": two_sided}

    family_labels = list(tests) + list(priors)
    family_ps = [out[lbl]["p_improve"] for lbl in tests] + [
        out[lbl]["p_two_sided"] for lbl in priors
    ]
    for label, q in zip(family_labels, ms.bh_fdr(family_ps), strict=True):
        out[label]["q"] = float(q)
    out["gates_pass"] = bool(all(out[f"{arm}_gate"]["q"] <= ms.SIG_Q for arm in LEARNING_ARMS))
    out["prior_separates"] = bool(any(out[lbl]["q"] <= ms.SIG_Q for lbl in priors))
    out["prior_note"] = (
        "a non-significant prior is a FAILURE TO DETECT a pre-update difference between the "
        "wirings at this seed count, not a demonstration that none exists"
    )
    return out


# What this panel can resolve, registered before it ran and sized from a pilot rather than assumed.
#
# The MINIMUM INTERESTING interaction is not arbitrary. L.0 found the rewired null AHEAD by 0.1076
# on this metric. For widening to mean the pool was hiding wiring structure -- the `pooling_hid_
# structure` reading, which reopens L.4 and L.5 -- the interaction has to flip that sign, so it must
# exceed ~0.108. An interaction smaller than that does not change L.0's verdict and does not reopen
# anything, so it is the size this panel is built to see.
#
# The first registration bounded the interaction's spread by assuming the two widths' wiring
# differences were INDEPENDENT (sd = sd(D) * sqrt(2)), and expected that to be pessimistic: the two
# widths at a seed share the task draws, the RNG stream and the initial policy, so their wiring
# differences ought to be correlated, and sd = sd(D) * sqrt(2 * (1 - rho)).
#
# **The pilot says otherwise.** At seeds 101-104 the measured correlation was rho = +0.02 and the
# realised interaction sd was 0.3770, against the independence bound of 0.4487 -- learning amplifies
# the divergence enough to wash out the shared start. Four points make that a noisy estimate, but it
# is the only evidence and it points at independence rather than away from it.
#
# So the panel is 96 seeds, not the 32 first registered: 2.80 * 0.3770 / sqrt(96) = 0.1077, which
# MATCHES the 0.1076 sign-flip threshold rather than clearing it -- the two agree to 0.2%, so the
# panel sits at 80% power for exactly the effect that would change the verdict and below it for
# anything smaller. At 32 seeds the same spread resolves only 0.187, which would have found a large
# interaction while leaving the one that matters undetectable.
REGISTERED_SENSITIVITY = {
    "n_registered": 96,
    "minimum_interesting_interaction": 0.1076,
    "minimum_interesting_rationale": (
        "L.0 found the rewired null ahead by 0.1076 on auc_success. An interaction must exceed that "
        "to flip the sign and mean the pool was hiding wiring structure; a smaller one changes "
        "nothing and reopens nothing"
    ),
    "pilot_measured_rho": 0.02,
    "pilot_realised_interaction_sd": 0.3770,
    "independence_bound_sd": 0.4487,
    "detectable_at_80_at_n96": 0.1077,
    "sits_at_threshold_not_below": True,
    "detectable_at_80_at_n32": 0.1866,
    "note": (
        "sized from the pilot's realised spread rather than an assumption. The first registration "
        "expected positive correlation between the widths and would have been over-optimistic; the "
        "pilot measured rho = +0.02 at four seeds, close to independent. Reported so a null states "
        "what it did and did not exclude"
    ),
}

_MIN_FOR_SPREAD = 2
_MIN_FOR_CORRELATION = 3


def sensitivity(
    deltas: list[float],
    n_pairs: int,
    cells: dict[str, dict[int, float]] | None = None,
) -> dict[str, Any]:
    """Report what size of interaction this panel could have seen, from its observed spread.

    Registered because a null closes the width objection, and a null that does not say what it could
    have detected closes nothing.

    The interaction is a difference of two differences, so its spread depends on how correlated
    those halves are: ``sd = sd(D) * sqrt(2 * (1 - rho))`` where ``D`` is the wiring difference at
    one width. The pre-campaign registration bounded ``rho`` at zero, which is pessimistic here --
    the two widths at a seed share the task draws, the RNG stream and the initial policy. So the
    realised figure is computed from the panel's own deltas and the correlation is reported beside
    it, rather than either being assumed.

    A normal-approximation planning figure, not the registered rank test's power.
    """
    if not deltas or n_pairs < _MIN_FOR_SPREAD:
        return {"n_pairs": n_pairs, "available": False, "registered": dict(REGISTERED_SENSITIVITY)}
    sd = float(np.std(deltas, ddof=1))
    se = sd / float(np.sqrt(n_pairs))
    out: dict[str, Any] = {
        "n_pairs": n_pairs,
        "available": True,
        "observed_sd": sd,
        "standard_error": se,
        # 1.96 + 0.84 standard errors: the classic 5% two-sided, 80%-power spacing.
        "detectable_at_80_percent": float(2.80 * se),
        "observed_mean": float(np.mean(deltas)),
        "registered": dict(REGISTERED_SENSITIVITY),
        "note": (
            "a normal-approximation planning figure from THIS panel's observed spread, not the "
            "registered rank test's power. It states what a null here does and does not exclude"
        ),
    }
    if cells:
        common = [s for s in sorted(cells["wt_pooled"]) if all(s in cells[c] for c in cells)]
        if len(common) >= _MIN_FOR_CORRELATION:
            pooled_d = [cells["wt_pooled"][s] - cells["rn_pooled"][s] for s in common]
            wide_d = [cells["wt_wide"][s] - cells["rn_wide"][s] for s in common]
            rho = float(np.corrcoef(pooled_d, wide_d)[0, 1])
            # Undefined when either width's wiring difference has no spread across seeds. Reporting
            # a nan as if it were a correlation would put one in the record's sensitivity table.
            out["width_difference_correlation"] = rho if np.isfinite(rho) else None
            out["correlation_note"] = (
                "the correlation between the wiring difference at each width. The registered bound "
                "assumed zero; a positive value is why the realised sensitivity beats it"
            )
    return out


# The nine registered tests, in the order the record reports them.
_FAMILY_CONTRASTS = ("interaction", "width_main_effect", "wiring_main_effect")


def adjust_family(primary: dict[str, Any], gate_result: dict[str, Any]) -> dict[str, Any]:
    """Correct all nine registered tests under one BH-FDR family, in place.

    The gates and priors already carry a ``q`` from being corrected among themselves; that partial
    correction is replaced here so every test in the registration is held to the same family. Each
    record keeps its raw p beside the adjusted q, so a reader can see both.
    """
    labels = [
        *_FAMILY_CONTRASTS,
        *(f"{a}_gate" for a in LEARNING_ARMS),
        *(f"prior_{w}" for w in WIDTHS),
    ]
    ps = (
        [float(primary[c]["p_two_sided"]) for c in _FAMILY_CONTRASTS]
        + [float(gate_result[f"{a}_gate"]["p_improve"]) for a in LEARNING_ARMS]
        + [float(gate_result[f"prior_{w}"]["p_two_sided"]) for w in WIDTHS]
    )
    qs = [float(q) for q in ms.bh_fdr(ps)]
    for label, q in zip(labels, qs, strict=True):
        target = primary[label] if label in _FAMILY_CONTRASTS else gate_result[label]
        target["q"] = q
    gate_result["gates_pass"] = bool(
        all(gate_result[f"{a}_gate"]["q"] <= ms.SIG_Q for a in LEARNING_ARMS),
    )
    gate_result["prior_separates"] = bool(
        any(gate_result[f"prior_{w}"]["q"] <= ms.SIG_Q for w in WIDTHS),
    )
    gate_result["family_note"] = (
        "all nine registered tests corrected under ONE BH-FDR family: the interaction, both main "
        "effects, four learning gates and two priors"
    )
    return {"labels": labels, "raw_p": ps, "q": qs}


def reading(
    gate_result: dict[str, Any],
    contrast_result: dict[str, Any],
    *,
    n_common: int,
    min_seeds: int = 5,
) -> dict[str, Any]:
    """Assign the registered reading. Gates first, then the interaction, in that order."""
    if n_common < min_seeds:
        name = "insufficient_seeds"
    elif not gate_result["gates_pass"]:
        name = "no_learning"
    else:
        interaction = contrast_result["interaction"]
        # The BH-adjusted q, not the raw p: the interaction sits in a nine-test family.
        if interaction.get("q", interaction["p_two_sided"]) > ms.SIG_Q:
            name = "pooling_was_not_the_limit"
        elif interaction["mean_delta"] > 0.0:
            name = "pooling_hid_structure"
        else:
            name = "width_favours_the_shuffle"
    interaction = contrast_result["interaction"]
    threshold = float(REGISTERED_SENSITIVITY["minimum_interesting_interaction"])
    clears = abs(float(interaction["mean_delta"])) >= threshold
    return {
        "reading": name,
        "why": READINGS[name],
        "reopens_l4_l5": name == "pooling_hid_structure",
        "gates_read_first": True,
        # REPORTED, not a gate on the verdict. L.1 registered 0.1076 as the panel's SENSITIVITY
        # target -- the interaction size that would flip L.0's sign -- and not as a minimum effect
        # beside significance the way block V registers `MIN_EFFICIENCY_GAIN`. Turning it into a
        # decision threshold after seeing the result would be changing the registered rule, so it is
        # carried as a field with the gap named. Registering a minimum effect beside significance is
        # phase-protocol principle 10, and L.1 not having one as a DECISION rule is a design gap to
        # carry forward, not to retro-fit here.
        "clears_registered_sensitivity_threshold": bool(clears),
        "registered_sensitivity_threshold": threshold,
        "threshold_note": (
            "reported, not applied: this was registered as the panel's sensitivity target, not as a "
            "minimum effect gating the verdict. A minimum-effect decision rule was not registered "
            "for L.1 -- a design gap, recorded rather than retro-fitted"
        ),
    }


def common_seeds(scanned: dict[str, Any], seeds: tuple[int, ...] = SEEDS) -> tuple[int, ...]:
    """Return the seeds present in ALL eight cells.

    A partial campaign is only scoreable on seeds every cell has: the interaction is a difference of
    differences, so a seed missing from one cell contributes nothing and cannot be part-counted. Used
    for the `--allow-incomplete` path; a registered panel goes through `require_complete` instead and
    this returns the full set unchanged.
    """
    runs = scanned["runs"]
    return tuple(s for s in seeds if all(s in runs[name] for name in ARMS))


def analyse(
    campaign_dir: Path,
    seeds: tuple[int, ...] = SEEDS,
    experiments: Path = EXPERIMENTS,
) -> dict[str, Any]:
    """Score the 2x2: gates first, then the interaction, with both metrics and the sensitivity."""
    scanned = scan(campaign_dir, experiments)
    seeds = common_seeds(scanned, seeds)
    reports = efficiency(scanned, campaign_dir, seeds)
    primary_cells = cell_values(reports, PRIMARY_METRIC)
    censored_cells = cell_values(reports, CENSORED_METRIC)
    primary = contrasts(primary_cells, seeds)
    censored = contrasts(censored_cells, seeds)
    gate_result = gates(scanned, seeds)
    # ONE BH-FDR family across all nine registered tests, as the proposal registered: the
    # interaction, both main effects, the four learning gates and the two width-specific priors.
    # An earlier draft corrected the gates and priors among themselves and left the interaction and
    # main effects on RAW p, which is not the registered procedure and understates the correction
    # the primary is held to.
    adjust_family(primary, gate_result)
    verdict = reading(gate_result, primary, n_common=primary["n_common"])
    # ORIENT before comparing. `auc_success` is higher-is-better and
    # `episodes_to_30pct_success` is lower-is-better, so agreement between them is OPPOSITE raw
    # signs. Comparing the raw signs reports a disagreement whenever the two actually agree, which
    # would put a caveat in the record that the data does not support.
    censored_orientation = (
        1.0 if reports["pooled"]["metrics"][CENSORED_METRIC]["higher_is_better"] else -1.0
    )
    agree = (
        primary["interaction"]["mean_delta"]
        * censored["interaction"]["mean_delta"]
        * censored_orientation
        >= 0.0
        if primary["n_common"]
        else True
    )
    return {
        "seeds": list(seeds),
        "n_seeds_scored": len(seeds),
        "primary_metric": PRIMARY_METRIC,
        "censored_metric": CENSORED_METRIC,
        "metric_note": METRIC_NOTE,
        "gates": gate_result,
        "primary": primary,
        "censored_axis": censored,
        "censoring": censoring(reports),
        "metrics_agree_in_direction": bool(agree),
        "censored_metric_higher_is_better": bool(censored_orientation > 0),
        "metric_disagreement_note": None
        if agree
        else (
            "the two metrics disagree in DIRECTION. The primary stands, and the disagreement is "
            "reported rather than resolved toward whichever supports a reading"
        ),
        "sensitivity": sensitivity(
            primary["interaction"]["per_seed_deltas"],
            primary["n_common"],
            primary_cells,
        ),
        "l0_reference": dict(L0_REFERENCE),
        **verdict,
        "efficiency": reports,
    }


def _print(result: dict[str, Any]) -> None:
    """Print the gates, then the interaction, then both main effects and the sensitivity."""
    print("\n" + "=" * 78)
    print("L.1 - readout width x wiring, on the hard350 cell")
    print("=" * 78)

    print("\n  GATES (read before the interaction):")
    for arm in LEARNING_ARMS:
        gate = result["gates"][f"{arm}_gate"]
        print(
            f"    {arm:18s} vs its floor  d={gate.get('effect', float('nan')):+8.3f} foods  "
            f"q={gate['q']:.3f}",
        )
    for width in WIDTHS:
        prior = result["gates"][f"prior_{width}"]
        print(
            f"    prior ({width:6s})     wt - rn       "
            f"d={prior.get('effect', float('nan')):+8.3f} foods  q={prior['q']:.3f}",
        )
    print(f"    gates pass: {result['gates']['gates_pass']}")
    print(f"    {result['gates']['prior_note']}")

    cells = result["primary"]["cell_means"]
    print(f"\n  THE 2x2 on {result['primary_metric']} (n={result['primary']['n_common']} paired):")
    print(f"    {'':14s}{'pooled (8)':>14s}{'per-neuron (78)':>18s}")
    for wiring, label in (("wt", "wild type"), ("rn", "rewired null")):
        print(
            f"    {label:14s}{cells.get(f'{wiring}_pooled', float('nan')):14.4f}"
            f"{cells.get(f'{wiring}_wide', float('nan')):18.4f}",
        )

    inter = result["primary"]["interaction"]
    print("\n  INTERACTION (primary) - does widening help the WILD TYPE more?")
    print(
        f"    d={inter['mean_delta']:+.4f}  CI[{inter['ci_lo']:+.4f},{inter['ci_hi']:+.4f}]  "
        f"q={inter.get('q', float('nan')):.3f}  wild-gains-more {inter['positive_seeds']}"
        f"/{result['primary']['n_common']}",
    )
    for name in ("width_main_effect", "wiring_main_effect"):
        effect = result["primary"][name]
        print(
            f"    {name:20s} d={effect['mean_delta']:+.4f}  "
            f"CI[{effect['ci_lo']:+.4f},{effect['ci_hi']:+.4f}]  "
            f"q={effect.get('q', float('nan')):.3f}"
            "   [secondary - not evidence about the wiring on its own]",
        )
    print(f"    {result['gates']['family_note']}")

    censored = result["censored_axis"]["interaction"]
    print(f"\n  {result['censored_metric']} (reported beside, not primary):")
    print(
        f"    interaction d={censored['mean_delta']:+.2f} episodes  "
        f"p={censored['p_two_sided']:.3f}",
    )
    for cell in ("wt_pooled", "rn_pooled", "wt_wide", "rn_wide"):
        entry = result["censoring"][cell]
        print(
            f"      {cell:10s} censored {entry['n_censored']}/{entry['n_seeds']}"
            f"  ({entry['rate']:.0%})",
        )
    if not result["metrics_agree_in_direction"]:
        print(f"    !! {result['metric_disagreement_note']}")

    sens = result["sensitivity"]
    if sens.get("available"):
        print(
            f"\n  SENSITIVITY: sd={sens['observed_sd']:.4f}, se={sens['standard_error']:.4f}, "
            f"detectable at 80% ~ {sens['detectable_at_80_percent']:+.4f}",
        )
        print(f"    {sens['note']}")

    print("-" * 78)
    print(f"  READING: {result['reading'].upper().replace('_', '-')}")
    print(f"    {result['why']}")
    print(f"    reopens L.4/L.5: {result['reopens_l4_l5']}")


def write_csv(result: dict[str, Any], path: Path) -> None:
    """One row per cell and seed, so every table in the record can be recomputed from it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    reports = result["efficiency"]
    metrics = sorted(next(iter(next(iter(reports.values()))["per_seed"][eff._WILD].values())))
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["wiring", "width", "seed", *metrics, "primary_censored"])
        for width, report in reports.items():
            horizon = report["horizon_episodes"]
            for wiring, arm in (("wt", eff._WILD), ("rn", eff._REWIRED)):
                rows = report["per_seed"][arm]
                for seed in sorted(rows, key=int):
                    row = rows[seed]
                    writer.writerow(
                        [
                            wiring,
                            width,
                            seed,
                            *(f"{row[m]:.6f}" for m in metrics),
                            int(row[CENSORED_METRIC] >= horizon),
                        ],
                    )


def _jsonable(value: object) -> object:
    """Coerce numpy scalars the harness layers return into plain JSON types."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    msg = f"not JSON-serialisable: {type(value)!r}"
    raise TypeError(msg)


def main(argv: list[str] | None = None) -> int:
    """Score the 2x2 and report the interaction, both main effects and the sensitivity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    # Derived from SEEDS so the default cannot drift from the registered panel, as it did when the
    # panel moved from 32 seeds to 96 and this default stayed behind.
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
    result = analyse(args.campaign, seeds)
    _print(result)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(result, indent=2, sort_keys=True, default=_jsonable) + "\n",
        )
        print(f"\nwrote {args.out}")
    if args.csv:
        write_csv(result, args.csv)
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
