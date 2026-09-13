"""The perturbation dimension: whether the rule was ever run at a scale it could work at.

Node perturbation forms its eligibility from what each unit's own noise did to a scalar outcome.
With N perturbed units the per-trial estimate of the reward gradient has signal-to-noise ~1/sqrt(N),
so the trials needed for a given amount of progress grow roughly as N (Werfel, Xie & Seung 2005).
That dimension differs by 16x and 38x across the platforms this rule has been measured on, and no
experiment has ever varied it:

* the one-step control it PASSES is ``Linear(K, 8) -> tanh -> Linear(8, 1)``: **8** perturbed units;
* every MLP yardstick arm that FAILED ran two hidden layers of 64 with hidden-only plasticity: **128**;
* the connectome perturbs all **302** neurons, at each of four settling steps -- **1208** draws per
  scored decision.

So "one-step works, multi-step fails" is equally consistent with "8 units works, 128 and 302 do not".

Two sweeps, in this order, because the arithmetic is tested where the rule works before it is read
off where it fails:

**S1, the arithmetic.** The committed one-step control at widths 8 to 128. This platform has no
capacity confound -- the task is solvable by 8 units and every further unit only adds noise -- so the
dimension is isolated. It also yields a RATE, trials-to-criterion, which is the quantity 1/N makes a
claim about and which a pass/fail reading discards.

**S2, the rescue.** The MLP yardstick on the calibrated hard-food cell at the same grid in perturbed
units, each width's learning arm against **its own** frozen control, with a **PPO capability arm** per
width: the prediction runs toward small N, which is exactly where capacity runs out, so without it a
small-width null could not be told from a small-width refutation.

Two things about the reading, both fixed before the run:

* **The frozen arm perturbs.** It carries the same sigma and freezes only the update. The cost the
  perturbation imposes on the policy is therefore present in both arms and cancels, so S2 measures the
  benefit of the update alone -- which is what a claim about estimator quality needs -- and not the
  net effect of switching perturbation on.
* **This sweep cannot separate units from weights.** In a fully-connected layer the weight count is
  proportional to the width, so a slope in N is equally consistent with a per-weight law. What it
  separates is scale from task, which is the question asked.
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

import l4_horizon_multistep as hm  # noqa: E402  # pyright: ignore[reportMissingImports]
import l4_mixture_statistic as ms  # noqa: E402  # pyright: ignore[reportMissingImports]
import l4_rule_positive_control as pc  # noqa: E402  # pyright: ignore[reportMissingImports]
from l4_panel import EXPERIMENTS, read_log  # noqa: E402  # pyright: ignore[reportMissingImports]

SEEDS = tuple(range(1, 9))

# ── S1: the width axis on the one-step control ───────────────────────────────
# On that arrangement -- one plastic layer, frozen readout -- the perturbed-unit count IS the width,
# so this grid is a grid in the perturbation dimension directly.
S1_WIDTHS = (8, 16, 32, 64, 128)
S1_SHAPES = tuple((width, 1) for width in S1_WIDTHS)
# The yardstick's EXACT arrangement: two plastic layers of 64, so 128 perturbed units in the shape
# the failing arms ran rather than in one layer. Added as a dated amendment after the width sweep
# returned a flat dependence and left depth the only difference at a matched unit count.
S1_DEPTH_CONTROL = (64, 2)
S1_ARMS = ("node_perturbation", "analytic")
NODE_NOISE = 0.2  # the scale that passed the control
# The prediction is a slope of +1 in log-log. The bar for "depends on N in the predicted direction at
# all" is set well below it, because the claim under test is the existence of the dependence and not
# its exponent -- and because this platform cannot attribute the exponent to units over weights.
SLOPE_BAR = 0.5
# A slope is only nonzero if it clears floating-point noise. An exactly constant series fits a
# slope of order 1e-16 and, being identical under every resample, a degenerate interval of zero
# width around it -- which would read as a dependence excluding zero.
SLOPE_TOLERANCE = 1e-9
BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20260913  # fixed so a re-analysis of the same runs returns the same interval

# ── S2: the width axis on the hard-food cell ─────────────────────────────────
# Two hidden layers with hidden-only plasticity, so the perturbed-unit count is TWICE the width and
# this grid matches S1's in units: 8, 16, 32, 64, 128.
S2_WIDTHS = (4, 8, 16, 32, 64)
S2_HIDDEN_LAYERS = 2
_S2_STEM = "mlpppo_small_continuous2d_fick_adaptive_klinotaxis_hard350"
# Significance is not sufficient: a paired rank test at eight seeds fires on the consistency of the
# sign, not the size of the shift. Both minima must hold.
#   - 1.0 of the cell's 20 foods: I.3b's 0.5-of-10 bar in proportion.
#   - 10% of that width's own reachable gap, so a width with almost nothing to gain cannot clear the
#     bar on a shift that means nothing.
MIN_FOODS = 1.0
MIN_GAP_FRACTION = 0.10

# ── The connectome's dimension, for the extrapolation ────────────────────────
# Two readings, because the perturbation is drawn per unit at EVERY settling step and nothing in the
# record establishes which the arithmetic tracks. Both are reported.
CONNECTOME_UNITS = 302
CONNECTOME_SETTLING_STEPS = 4
CONNECTOME_DRAWS = CONNECTOME_UNITS * CONNECTOME_SETTLING_STEPS
YARDSTICK_UNITS = 128

_S2_LABEL = re.compile(
    rf"^{re.escape(_S2_STEM)}_(?P<arm>nodepert|ppo)_w(?P<width>\d+)(?P<frozen>_frozen)?"
    r"-seed(?P<seed>\d+)\.log$",
)


# ═════════════════════════════ S1 ═══════════════════════════════════════════


def trials_to_criterion(
    reward_blocks: list[float],
    threshold: float,
    sustain: int = 1,
) -> int | None:
    """Trials until a trailing block mean first reaches ``threshold``, or None if it never does.

    ``sustain`` is the number of CONSECUTIVE blocks required. One -- the registered criterion -- is
    the single trailing 100-trial mean. Two is reported alongside as a robustness column, because a
    lone noisy block can cross early, and at large N an early spurious crossing would FLATTEN the
    slope: the bias from the noise runs against the hypothesis, not for it.
    """
    if sustain < 1:
        msg = f"sustain must be >= 1, got {sustain}"
        raise ValueError(msg)
    run = 0
    for index, value in enumerate(reward_blocks):
        run = run + 1 if value >= threshold else 0
        if run >= sustain:
            return (index + 1) * pc.BLOCK
    return None


def run_s1(
    shapes: tuple[tuple[int, int], ...] = S1_SHAPES,
    seeds: tuple[int, ...] = SEEDS,
    trials: int = pc.TRIALS,
) -> list[dict[str, Any]]:
    """Run the rule and the analytic reference at every shape, at the control's passing settings."""
    task = pc.ContextualAssociation.default()
    runs: list[dict[str, Any]] = []
    for hidden, layers in shapes:
        for seed in seeds:
            for arm in S1_ARMS:
                print(
                    f"  S1 {hidden:>3}x{layers} ({hidden * layers:>3} units)  {arm:<18} seed {seed}",
                    flush=True,
                )
                runs.append(
                    pc.run_arm(
                        arm,
                        seed,
                        task,
                        trials=trials,
                        node_noise=NODE_NOISE if arm in pc.PERTURBING_ARMS else 0.0,
                        hidden=hidden,
                        layers=layers,
                    ),
                )
    return runs


def _scores(runs: list[dict[str, Any]], shape: tuple[int, int], arm: str) -> list[float]:
    hidden, layers = shape
    return [
        r["score"]
        for r in runs
        if r["hidden"] == hidden and r.get("layers", 1) == layers and r["arm"] == arm
    ]


def _gap_fraction(mean: float, floor: float, optimum: float) -> float:
    """Where a mean sits between the closed-form floor and optimum, as a fraction of the gap."""
    span = optimum - floor
    return float("nan") if span == 0 else (mean - floor) / span


def assess_s1_shape(
    runs: list[dict[str, Any]],
    shape: tuple[int, int],
    task: pc.ContextualAssociation,
) -> dict[str, Any]:
    """Score one shape: the pass rule, the reachability control and the criterion times.

    The floor and the optimum are closed-form properties of the TASK, but what a network with a
    frozen random readout can reach is a property of the WIDTH. The analytic reference measures it.
    A width where the reference itself misses the pass bar is void at that width -- the control's own
    void clause, applied per cell -- and the rule's raw fraction there says nothing.
    """
    hidden, layers = shape
    floor, optimum = task.cue_blind_floor(pc.NOISE), task.optimum(pc.NOISE)
    rule_scores = _scores(runs, shape, "node_perturbation")
    reference_scores = _scores(runs, shape, "analytic")
    rule = pc.assess(rule_scores, floor, optimum)
    reference = pc.assess(reference_scores, floor, optimum)
    rule_fraction = _gap_fraction(rule["mean"], floor, optimum)
    reference_fraction = _gap_fraction(reference["mean"], floor, optimum)
    threshold = rule["halfway_threshold"]
    per_seed: dict[str, Any] = {}
    for record in runs:
        if (
            record["hidden"] != hidden
            or record.get("layers", 1) != layers
            or record["arm"] != "node_perturbation"
        ):
            continue
        per_seed[str(record["seed"])] = {
            "score": record["score"],
            "trials_to_criterion": trials_to_criterion(record["reward_blocks"], threshold),
            "trials_to_criterion_sustained": trials_to_criterion(
                record["reward_blocks"],
                threshold,
                sustain=2,
            ),
        }
    crossed = [v["trials_to_criterion"] for v in per_seed.values() if v["trials_to_criterion"]]
    return {
        "width": hidden,
        "layers": layers,
        "perturbed_units": hidden * layers,
        "rule": rule,
        "reference": reference,
        "rule_gap_fraction": rule_fraction,
        "reference_gap_fraction": reference_fraction,
        # What the rule achieved as a share of what THIS width can reach. Only defined where the
        # reference clears the bar; elsewhere the width is void and normalising would manufacture a
        # number out of a broken reference.
        "reachability_normalised": (
            rule_fraction / reference_fraction if reference["passes"] else None
        ),
        "void": not reference["passes"],
        "void_reason": (
            None
            if reference["passes"]
            else "the analytic reference misses the pass bar at this width, so nothing measured here "
            "is interpretable"
        ),
        "per_seed": per_seed,
        "crossed": len(crossed),
        "censored": len(per_seed) - len(crossed),
        "censoring_rate": (1.0 - len(crossed) / len(per_seed)) if per_seed else float("nan"),
        "median_trials_to_criterion": float(np.median(crossed)) if crossed else float("nan"),
    }


def fit_1n(widths: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Fit log2(trials-to-criterion) on log2(N) over per-seed values, with a bootstrap over seeds.

    Censored seeds have no criterion time and are EXCLUDED from the fit and counted in the record. A
    fit over crossers alone with the non-crossers unreported is how a censored metric turns a null
    into a positive; the count travels with the slope so the reader can see what the fit is over.
    """
    per_seed: dict[int, list[tuple[float, float]]] = {}
    for cell in widths.values():
        if cell["void"]:
            continue
        for seed, values in cell["per_seed"].items():
            trials = values["trials_to_criterion"]
            if trials:
                per_seed.setdefault(int(seed), []).append(
                    (math.log2(cell["perturbed_units"]), math.log2(trials)),
                )
    points = [point for pairs in per_seed.values() for point in pairs]
    if len(points) < 3 or len({x for x, _ in points}) < 2:
        return {
            "defined": False,
            "reason": "fewer than three crossing seeds, or crossings at a single width only",
            "n_points": len(points),
            "n_seeds": len(per_seed),
        }
    xs = np.array([x for x, _ in points])
    ys = np.array([y for _, y in points])
    slope, intercept = (float(v) for v in np.polyfit(xs, ys, 1))
    # Resample SEEDS, not points: a seed contributes one value per width and those are paired by
    # construction, so resampling points would break the pairing and narrow the interval.
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    seeds = list(per_seed)
    slopes: list[float] = []
    for _ in range(BOOTSTRAP):
        drawn = [per_seed[seeds[i]] for i in rng.integers(0, len(seeds), len(seeds))]
        sample = [point for pairs in drawn for point in pairs]
        if len({x for x, _ in sample}) < 2:
            continue
        slopes.append(
            float(np.polyfit([x for x, _ in sample], [y for _, y in sample], 1)[0]),
        )
    low, high = (
        (float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5)))
        if slopes
        else (float("nan"), float("nan"))
    )
    return {
        "defined": True,
        "slope": slope,
        "intercept": intercept,
        "ci_low": low,
        "ci_high": high,
        "n_points": len(points),
        "n_seeds": len(per_seed),
        "widths_in_fit": sorted({int(2**x) for x, _ in points}),
        "bootstrap": len(slopes),
        "prediction": 1.0,
        "bar": SLOPE_BAR,
        "meets_bar": bool(slope >= SLOPE_BAR and low > SLOPE_TOLERANCE),
    }


def extrapolate(fit: dict[str, Any], units: int, label: str) -> dict[str, Any]:
    """Read the fit outside its range, as a labelled extrapolation and never as a measurement."""
    if not fit.get("defined"):
        return {"label": label, "units": units, "extrapolation": True, "trials": None}
    trials = 2.0 ** (fit["intercept"] + fit["slope"] * math.log2(units))
    return {
        "label": label,
        "units": units,
        # Read wherever this figure appears: a five-point fit evaluated outside its own range is not
        # a measurement of that platform and is never reported as one.
        "extrapolation": True,
        "outside_fitted_range": units > max(fit["widths_in_fit"]),
        "trials": float(trials),
        # Read off the INTERVAL, not the point estimate: a flat series fits a slope of order 1e-17,
        # whose sign is floating-point noise. Only an interval excluding zero from above says the
        # dimension costs trials at all; anything else is not a budget constraint, and reporting it
        # as one would invert the result -- the dimension would read as costing something when the
        # fit says it does not.
        "is_a_budget_constraint": bool(fit["ci_low"] > SLOPE_TOLERANCE),
        "note": (
            "EXTRAPOLATION from the fitted grid; not a measurement of this platform"
            + (
                ""
                if fit["ci_low"] > SLOPE_TOLERANCE
                else " -- and the fitted interval does not exclude zero from above, so this is not "
                "a budget constraint"
            )
        ),
    }


def _level_trend(widths: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """How the LEVEL the rule reaches moves with N, separately from how long it takes.

    Speed and asymptote are different claims, and 1/N is about the first. A dimension that costs a
    little final performance while costing no time is a real effect and a small one, and pooling the
    two would report whichever is larger as though it were both.
    """
    usable = [w for w in sorted(widths) if not widths[w]["void"]]
    fractions = [float(widths[w]["rule_gap_fraction"]) for w in usable]
    if len(usable) < 3 or float(np.ptp(fractions)) == 0.0:
        return {"defined": False, "reason": "fewer than three usable widths, or a constant level"}
    from scipy import stats

    result: Any = stats.spearmanr(np.array([float(w) for w in usable]), np.array(fractions))
    return {
        "defined": True,
        "widths": usable,
        "gap_fractions": fractions,
        "rho": float(result.statistic),
        "p": float(result.pvalue),
        "span": float(max(fractions) - min(fractions)),
        "descriptive_only": True,
        "note": "five widths: the level's direction, not a tested exponent",
    }


def analyse_s1(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Score every shape, fit the dependence and apply the registered bands."""
    task = pc.ContextualAssociation.default()
    widths = {
        hidden * layers: assess_s1_shape(runs, (hidden, layers), task)
        for hidden, layers in S1_SHAPES
    }
    fit = fit_1n(widths)
    smallest, largest = min(S1_WIDTHS), max(S1_WIDTHS)
    # The single sharpest number in the sweep: the rule passes this control at 8 units. Whether it
    # still passes at the width every failing yardstick arm ran locates the confound or closes it.
    largest_passes = bool(widths[largest]["rule"]["passes"])
    baseline_reproduces = bool(widths[smallest]["rule"]["passes"])
    if not baseline_reproduces:
        verdict = "void"
        reason = (
            f"the rule does not reproduce its pass at {smallest} units, so the platform has drifted "
            "and nothing in the sweep is interpretable until that is found"
        )
    elif fit.get("meets_bar"):
        verdict = "scale_dependent"
        reason = f"slope {fit['slope']:+.2f} at or above the {SLOPE_BAR} bar with a CI excluding 0"
    elif fit.get("defined") and (
        fit["ci_low"] <= SLOPE_TOLERANCE and fit["ci_high"] >= -SLOPE_TOLERANCE
    ):
        verdict = "flat"
        reason = f"slope CI [{fit['ci_low']:+.2f}, {fit['ci_high']:+.2f}] contains 0"
    elif fit.get("defined") and fit["ci_high"] < -SLOPE_TOLERANCE:
        # Not the same as flat, and not a weak version of the prediction: the dependence exists and
        # runs the OTHER WAY. Calling this "below the bar" would read as a small positive effect.
        verdict = "opposite_direction"
        reason = (
            f"slope {fit['slope']:+.2f}, CI [{fit['ci_low']:+.2f}, {fit['ci_high']:+.2f}] entirely "
            "below 0: time-to-criterion FALLS as the dimension grows, opposite to the prediction"
        )
    elif fit.get("defined"):
        verdict = "below_bar"
        reason = f"slope {fit['slope']:+.2f} excludes 0 but is below the {SLOPE_BAR} bar"
    else:
        verdict = "undefined"
        reason = str(fit.get("reason"))
    depth = assess_s1_shape(runs, S1_DEPTH_CONTROL, task)
    matched = widths.get(S1_DEPTH_CONTROL[0] * S1_DEPTH_CONTROL[1])
    return {
        "widths": widths,
        "fit": fit,
        "level_trend": _level_trend(widths),
        # The yardstick's exact arrangement at the same unit count as the widest one-layer cell. With
        # the width sweep flat, depth is the only shape difference left between the platform the rule
        # passes and the platform it fails, so this is what separates shape from task.
        "depth_control": (
            None
            if depth["rule"]["n"] == 0
            else {
                "shape": f"{S1_DEPTH_CONTROL[0]}x{S1_DEPTH_CONTROL[1]}",
                "perturbed_units": depth["perturbed_units"],
                "cell": depth,
                "matched_one_layer_gap_fraction": (
                    None if matched is None else matched["rule_gap_fraction"]
                ),
                "passes": bool(depth["rule"]["passes"]),
            }
        ),
        "baseline_reproduces": baseline_reproduces,
        "largest_width_passes": largest_passes,
        "verdict": verdict,
        "why": reason,
        "derived_budget": [
            extrapolate(fit, YARDSTICK_UNITS, "the MLP yardstick (128 units)"),
            extrapolate(fit, CONNECTOME_UNITS, "the connectome, read as units (302)"),
            extrapolate(
                fit,
                CONNECTOME_DRAWS,
                f"the connectome, read as draws per decision ({CONNECTOME_DRAWS})",
            ),
        ],
    }


# ═════════════════════════════ S2 ═══════════════════════════════════════════


def scan_s2(campaign_dir: Path, experiments: Path = EXPERIMENTS) -> dict[int, dict[str, Any]]:
    """Read every registered run under a campaign directory, keyed by width and arm."""
    log_dir = campaign_dir / "logs" if (campaign_dir / "logs").is_dir() else campaign_dir
    out: dict[int, dict[str, Any]] = {
        w: {"learning": {}, "frozen": {}, "ppo": {}, "logs": {}} for w in S2_WIDTHS
    }
    for log in sorted(log_dir.glob("*.log")):
        match = _S2_LABEL.match(log.name)
        if match is None:
            print(f"  WARN: skipping log with an unrecognised label: {log.name}")
            continue
        width = int(match.group("width"))
        if width not in out:
            print(f"  WARN: skipping log whose width is not registered: {log.name}")
            continue
        if match.group("arm") == "ppo":
            arm = "ppo"
            if match.group("frozen"):
                # A frozen PPO arm is not a registered cell and would score as a capability arm.
                print(f"  WARN: skipping an unregistered frozen PPO arm: {log.name}")
                continue
        else:
            arm = "frozen" if match.group("frozen") else "learning"
        record = read_log(log, experiments)
        if record is None:
            print(f"  WARN: no parseable run lines in {log.name} - dropped")
            continue
        seed = int(match.group("seed"))
        if seed in out[width][arm]:
            # Two logs for one cell: whichever was read second would silently replace the first and
            # the campaign would score as if one run had happened.
            msg = (
                f"two logs for width {width} arm {arm} seed {seed}: {log.name} duplicates an "
                "already-read run"
            )
            raise ValueError(msg)
        out[width][arm][seed] = record
        out[width]["logs"].setdefault(arm, {})[seed] = log
    return out


def require_complete_s2(scanned: dict[int, dict[str, Any]]) -> None:
    """Refuse to score an S2 campaign that is missing any registered cell.

    A dropped or unfinished run would otherwise shrink a width's pairing silently, and the registered
    verdict -- ``not_rescued`` included -- would be assigned on partial evidence while looking exactly
    like a complete result. The capability arm counts: a width whose control did not run cannot be
    called interpretable OR uninterpretable.
    """
    missing: list[str] = []
    for width in S2_WIDTHS:
        for arm in ("learning", "frozen", "ppo"):
            absent = [s for s in SEEDS if s not in scanned[width][arm]]
            if absent:
                missing.append(f"w{width:02d}/{arm} seeds {absent}")
    if missing:
        msg = "campaign is incomplete, so no verdict is available: " + "; ".join(missing)
        raise ValueError(msg)


def capability(width_data: dict[str, Any]) -> dict[str, Any]:
    """Decide whether this width can hold a competent policy at all.

    The prediction runs toward SMALL N, which is exactly where capacity runs out, so a width that
    cannot hold a policy must be reported uninterpretable rather than as evidence against the
    mechanism. Two parts, both reported: the PPO arm beats the do-nothing floor, and it reaches the
    committed competence threshold.

    The comparator is the plastic recipe's frozen arm, which PERTURBS; the PPO arm does not. So this
    is a floor check on the width and not a matched pair, and the record says so rather than reading
    it as a clean PPO-versus-perturbation contrast.
    """
    ppo_foods = {s: r.foods for s, r in width_data["ppo"].items()}
    frozen_foods = {s: r.foods for s, r in width_data["frozen"].items()}
    ppo_clear = [r.success for r in width_data["ppo"].values()]
    contrast = ms.shift_contrast(ppo_foods, frozen_foods)
    mean_clear = float(np.mean(ppo_clear)) if ppo_clear else float("nan")
    beats_floor = bool(contrast.get("defined") and contrast["p_improve"] <= ms.SIG_Q)
    competent = bool(mean_clear >= ms.COMPETENT_THRESHOLD)
    failed = [
        name
        for name, ok in (("beats the do-nothing floor", beats_floor), ("competent", competent))
        if not ok
    ]
    return {
        "ppo_mean_foods": float(np.mean(list(ppo_foods.values()) or [math.nan])),
        "frozen_mean_foods": float(np.mean(list(frozen_foods.values()) or [math.nan])),
        "ppo_mean_full_clear": mean_clear,
        "competence_threshold": ms.COMPETENT_THRESHOLD,
        "beats_floor": beats_floor,
        "p_beats_floor": contrast.get("p_improve"),
        "competent": competent,
        "passes": not failed,
        "why": "passes both parts" if not failed else f"fails: {', '.join(failed)}",
        "reachable_gap_foods": float(
            np.mean(list(ppo_foods.values()) or [math.nan])
            - np.mean(list(frozen_foods.values()) or [math.nan]),
        ),
        "comparator_note": (
            "the frozen comparator perturbs and the PPO arm does not; this is a capability floor on "
            "the width, not a matched pair"
        ),
    }


def compare_width(width_data: dict[str, Any], experiments: Path = EXPERIMENTS) -> dict[str, Any]:
    """Score one width's learning arm against its own frozen control."""
    learning_foods = {s: r.foods for s, r in width_data["learning"].items()}
    frozen_foods = {s: r.foods for s, r in width_data["frozen"].items()}
    return {
        "n_pairs": len(set(learning_foods) & set(frozen_foods)),
        "learning_mean_foods": float(np.mean(list(learning_foods.values()) or [math.nan])),
        "frozen_mean_foods": float(np.mean(list(frozen_foods.values()) or [math.nan])),
        "learning_foods": learning_foods,
        "frozen_foods": frozen_foods,
        "graded": ms.shift_contrast(learning_foods, frozen_foods),
        # Reported so the floor is visible rather than assumed: the plastic arms are expected at it,
        # and where they are the record says so instead of returning a null.
        # I.3b's derivation, reused rather than copied.
        "full_clear": hm._full_clear(
            {s: r.success for s, r in width_data["learning"].items()},
            {s: r.success for s, r in width_data["frozen"].items()},
        ),
        "capability": capability(width_data),
        # Within a width only: weights at different widths have different shapes. Unavailable rather
        # than zero when the campaign's experiment records are missing.
        "drift": hm.drift(width_data["logs"], experiments),
    }


def _minima(cell: dict[str, Any]) -> dict[str, Any]:
    """Apply both registered effect minima and name which one failed."""
    effect = cell["graded"].get("effect", float("nan"))
    reachable = cell["capability"]["reachable_gap_foods"]
    needed = MIN_GAP_FRACTION * reachable if math.isfinite(reachable) else float("nan")
    absolute_ok = bool(effect >= MIN_FOODS)
    relative_ok = bool(effect >= needed) if math.isfinite(needed) else False
    failed = [
        name
        for name, ok in (
            (f"the {MIN_FOODS} foods minimum", absolute_ok),
            (
                f"the {MIN_GAP_FRACTION:.0%}-of-reachable-gap minimum ({needed:.2f} foods)",
                relative_ok,
            ),
        )
        if not ok
    ]
    return {
        "effect": effect,
        "absolute_minimum": MIN_FOODS,
        "relative_minimum": needed,
        "passes": not failed,
        "why": "at or above both minima" if not failed else f"below {' and '.join(failed)}",
    }


def analyse_s2(
    scanned: dict[int, dict[str, Any]],
    experiments: Path = EXPERIMENTS,
) -> dict[str, Any]:
    """Compare every width, correct across them and apply the registered rule."""
    cells = {w: compare_width(scanned[w], experiments) for w in S2_WIDTHS}
    defined = [w for w in S2_WIDTHS if cells[w]["graded"].get("defined")]
    qs = ms.bh_fdr([cells[w]["graded"]["p_improve"] for w in defined])
    for width, q in zip(defined, qs, strict=True):
        cells[width]["graded"]["q_improve"] = float(q)
    for width in S2_WIDTHS:
        cell = cells[width]
        cell["perturbed_units"] = S2_HIDDEN_LAYERS * width
        q = cell["graded"].get("q_improve", float("nan"))
        significant = bool(q <= ms.SIG_Q) if not math.isnan(q) else False
        minima = _minima(cell)
        cell["minima"] = minima
        if not cell["capability"]["passes"]:
            # Never a null: a width that cannot hold a policy has not tested the mechanism.
            cell["verdict"] = "uninterpretable"
            cell["beats_control"] = False
            cell["why"] = f"capability arm {cell['capability']['why']}"
        elif significant and minima["passes"]:
            cell["verdict"] = "beats_control"
            cell["beats_control"] = True
            cell["why"] = "significant and at or above both minima"
        elif significant:
            cell["verdict"] = "below_min_effect"
            cell["beats_control"] = False
            cell["why"] = f"significant but {minima['why']}"
        else:
            cell["verdict"] = "no_improvement"
            cell["beats_control"] = False
            cell["why"] = "not significant"
    interpretable = [w for w in S2_WIDTHS if cells[w]["verdict"] != "uninterpretable"]
    winners = [w for w in interpretable if cells[w]["beats_control"]]
    return {
        "widths": cells,
        "interpretable": interpretable,
        "excluded_uninterpretable": [w for w in S2_WIDTHS if w not in interpretable],
        "winners": winners,
        "trend": _trend(cells, interpretable),
        "verdict": "rescued" if winners else ("void" if not interpretable else "not_rescued"),
    }


def _trend(cells: dict[int, dict[str, Any]], interpretable: list[int]) -> dict[str, Any]:
    """Spearman of the per-width contrast against N. Descriptive: five widths has almost no power."""
    usable = [w for w in interpretable if cells[w]["graded"].get("defined")]
    if len(usable) < 3:
        return {
            "defined": False,
            "reason": f"only {len(usable)} interpretable widths with a defined contrast",
            "descriptive_only": True,
        }
    units = np.array([float(cells[w]["perturbed_units"]) for w in usable])
    effects = np.array([float(cells[w]["graded"]["effect"]) for w in usable])
    if float(np.ptp(effects)) == 0.0:
        # A constant series has no correlation to report. Spearman returns not-a-number here, which
        # in a record reads as a failed test rather than as the flat series it is.
        return {
            "defined": False,
            "reason": "the contrast is identical at every interpretable width",
            "constant_at": float(effects[0]),
            "widths": usable,
            "descriptive_only": True,
        }
    from scipy import stats

    result: Any = stats.spearmanr(units, effects)
    return {
        "defined": True,
        "widths": usable,
        "rho": float(result.statistic),
        "p": float(result.pvalue),
        # The prediction: the contrast should be LARGER at smaller N, so rho should be negative.
        "predicted_sign": "negative",
        "in_predicted_direction": bool(result.statistic < 0),
        "descriptive_only": True,
        "note": "five widths: reported as description, the per-width gates carry the verdict",
    }


# ═════════════════════════ the combined verdict ═════════════════════════════


def combine(s1: dict[str, Any], s2: dict[str, Any] | None) -> dict[str, Any]:
    """Apply the three registered outcomes, and record a mixed reading as mixed."""
    if s1["verdict"] == "void":
        return {"verdict": "void", "why": s1["why"]}
    if s2 is None:
        return {
            "verdict": "s1_only",
            "why": (
                "S1 ran and S2 did not, which the registration allows: S1 does not depend on S2 and "
                "is not weakened by its absence"
            ),
            "s1_verdict": s1["verdict"],
            "largest_width_passes": s1["largest_width_passes"],
        }
    scale_dependent = s1["verdict"] == "scale_dependent"
    largest_fails = not s1["largest_width_passes"]
    rescued = s2["verdict"] == "rescued"
    if scale_dependent and largest_fails and rescued:
        verdict, why = (
            "scale_limited",
            (
                "the dependence holds, the yardstick's own width fails the control, and a width "
                "rescues the cell: the failures are a scale property"
            ),
        )
    elif scale_dependent and not rescued:
        verdict, why = (
            "arithmetic_only",
            (
                "the dependence holds and no width rescues the cell: the multi-step failure is a "
                "second, independent defect"
            ),
        )
    elif s1["verdict"] == "flat" and s1["largest_width_passes"] and not rescued:
        verdict, why = (
            "not_scale_limited",
            (
                "no dependence and the largest width still passes the control: the arithmetic is not "
                "the binding constraint, and every existing negative keeps its reading"
            ),
        )
    else:
        verdict, why = (
            "mixed",
            (
                f"S1 reads {s1['verdict']} with the largest width "
                f"{'passing' if s1['largest_width_passes'] else 'failing'} and S2 reads "
                f"{s2['verdict']}; recorded as mixed with both halves stated rather than resolved "
                "toward the nearer verdict"
            ),
        )
    return {
        "verdict": verdict,
        "why": why,
        "s1_verdict": s1["verdict"],
        "largest_width_passes": s1["largest_width_passes"],
        "s2_verdict": s2["verdict"],
    }


# ═════════════════════════════ reporting ════════════════════════════════════


def _print_s1(s1: dict[str, Any]) -> None:
    print("\nS1 — the arithmetic, on the one-step control")
    print("  units | rule gap | reference | normalised | passes | crossed | median trials")
    for width in S1_WIDTHS:
        cell = s1["widths"][width]
        normalised = cell["reachability_normalised"]
        print(
            f"  {cell['perturbed_units']:>5} | {cell['rule_gap_fraction']:>8.3f} | "
            f"{cell['reference_gap_fraction']:>9.3f} | "
            f"{'    void  ' if normalised is None else f'{normalised:>10.3f}'} | "
            f"{'yes' if cell['rule']['passes'] else 'no ':>6} | "
            f"{cell['crossed']}/{cell['crossed'] + cell['censored']:<5} | "
            f"{cell['median_trials_to_criterion']:>13.0f}",
        )
    level = s1["level_trend"]
    if level.get("defined"):
        print(
            f"  level: rho {level['rho']:+.3f} p {level['p']:.3f} over a span of "
            f"{level['span']:.3f} of the gap (descriptive)",
        )
    depth = s1.get("depth_control")
    if depth is not None:
        matched = depth["matched_one_layer_gap_fraction"]
        print(
            f"  depth control {depth['shape']} ({depth['perturbed_units']} units, the yardstick's "
            f"own shape): gap {depth['cell']['rule_gap_fraction']:.3f} against "
            f"{'n/a' if matched is None else f'{matched:.3f}'} in one layer — "
            f"{'passes' if depth['passes'] else 'FAILS'}",
        )
    fit = s1["fit"]
    if fit.get("defined"):
        print(
            f"  fit: slope {fit['slope']:+.3f} CI [{fit['ci_low']:+.3f}, {fit['ci_high']:+.3f}] "
            f"over {fit['n_points']} points from {fit['n_seeds']} seeds "
            f"(prediction {fit['prediction']:+.1f}, bar {fit['bar']:+.1f})",
        )
    else:
        print(f"  fit undefined: {fit.get('reason')}")
    for row in s1["derived_budget"]:
        trials = row["trials"]
        amount = "undefined" if trials is None else f"{trials:,.0f} trials"
        print(f"  EXTRAPOLATION — {row['label']}: {amount} (not a measurement)")
    print(f"  S1 verdict: {s1['verdict']} — {s1['why']}")


def _print_s2(s2: dict[str, Any]) -> None:
    print("\nS2 — the rescue, on the hard-food cell")
    print("  units | learning | frozen | shift |     q | drift | capability | verdict")
    for width in S2_WIDTHS:
        cell = s2["widths"][width]
        q = cell["graded"].get("q_improve", float("nan"))
        print(
            f"  {cell['perturbed_units']:>5} | {cell['learning_mean_foods']:>8.3f} | "
            f"{cell['frozen_mean_foods']:>6.3f} | {cell['graded'].get('effect', float('nan')):>+5.2f} | "
            f"{q:5.3f} | {cell['drift']['mean_relative']:>5.2f} | "
            f"{'pass' if cell['capability']['passes'] else 'FAIL':>10} | {cell['verdict']}",
        )
        print(f"        {cell['why']}")
    trend = s2["trend"]
    if trend.get("defined"):
        print(
            f"  trend (descriptive): rho {trend['rho']:+.3f} p {trend['p']:.3f}, "
            f"predicted {trend['predicted_sign']}",
        )
    print(f"  S2 verdict: {s2['verdict']}")


def write_s1_csv(runs: list[dict[str, Any]], path: Path) -> None:
    """One row per width, arm and seed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        # LF, matching every committed per-seed record; csv's default dialect would write CRLF and
        # git would normalise it on the way in, leaving the working tree and the commit different.
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ["width", "layers", "perturbed_units", "arm", "seed", "score", "alignment"],
        )
        for run in runs:
            writer.writerow(
                [
                    run["hidden"],
                    run.get("layers", 1),
                    run["perturbed_units"],
                    run["arm"],
                    run["seed"],
                    f"{run['score']:.6f}",
                    f"{run['alignment']:.6f}",
                ],
            )


def main(argv: list[str] | None = None) -> int:
    """Run S1, analyse S2, or both, and write the records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s1", action="store_true", help="run the one-step width sweep")
    parser.add_argument("--trials", type=int, default=pc.TRIALS)
    parser.add_argument("--s2", type=Path, help="analyse an S2 campaign directory")
    parser.add_argument("--experiments", type=Path, default=EXPERIMENTS)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args(argv)
    if not args.s1 and args.s2 is None:
        print("nothing to do: pass --s1, --s2 <campaign-dir>, or both", file=sys.stderr)
        return 2

    s1: dict[str, Any] | None = None
    s2: dict[str, Any] | None = None
    if args.s1:
        # The registered grid, then the yardstick's own shape at a matched unit count.
        runs = run_s1((*S1_SHAPES, S1_DEPTH_CONTROL), trials=args.trials)
        s1 = analyse_s1(runs)
        _print_s1(s1)
        if args.csv:
            write_s1_csv(runs, args.csv)
    if args.s2 is not None:
        scanned = scan_s2(args.s2, args.experiments)
        require_complete_s2(scanned)
        s2 = analyse_s2(scanned, args.experiments)
        _print_s2(s2)

    combined = combine(s1, s2) if s1 is not None else None
    if combined is not None:
        print(f"\nVERDICT: {combined['verdict']} — {combined['why']}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                # The control's own NaN-to-null pass, reused so the record is strict JSON.
                pc._jsonable(
                    {
                        "protocol": {
                            "seeds": list(SEEDS),
                            "s1_widths": list(S1_WIDTHS),
                            "s1_perturbed_units": list(S1_WIDTHS),
                            "s1_depth_control": list(S1_DEPTH_CONTROL),
                            "s2_widths": list(S2_WIDTHS),
                            "s2_perturbed_units": [S2_HIDDEN_LAYERS * w for w in S2_WIDTHS],
                            "node_noise": NODE_NOISE,
                            "slope_bar": SLOPE_BAR,
                            "slope_prediction": 1.0,
                            "min_foods": MIN_FOODS,
                            "min_gap_fraction": MIN_GAP_FRACTION,
                            "connectome_units": CONNECTOME_UNITS,
                            "connectome_draws_per_decision": CONNECTOME_DRAWS,
                        },
                        "s1": s1,
                        "s2": s2,
                        "combined": combined,
                    },
                ),
                indent=2,
            )
            + "\n",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
