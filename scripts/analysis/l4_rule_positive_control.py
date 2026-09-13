#!/usr/bin/env python
"""The rule's positive control: does the rule learn where learning is easiest.

Drives the committed ``ThreeFactorRule`` over the committed ``MLPTopology`` seam on a one-step
contextual association whose optimum and cue-blind floor are both closed-form. No environment,
no runner, no connectome, no action head: everything that could explain a null is removed rather
than controlled for, so a failure here has nothing left to blame.

Three arms, fixed before any run:

``three_factor``  the instrument under test, at the panels' pinned recipe, over a declared rate
                  grid -- **any rate passing counts as a pass**, since the claim is that the rule
                  learns at all and a fail must not be a rate artefact;
``hebbian``       the same rule unmodulated. It never sees reward, so it must NOT solve a task
                  whose answer only reward reveals;
``analytic``      plain gradient descent on the task's own loss through the same topology. It
                  must pass, or the control is VOID and says nothing about the rule.

Pass: the three-factor arm beats the cue-blind floor on at least 7 of 8 seeds AND its mean sits
at least halfway from that floor to the optimum. Void: the reference fails, or the floor arm
passes.

Usage::

    uv run python scripts/analysis/l4_rule_positive_control.py --out control.json --csv per-seed.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from quantumnematode.brain.arch._mlp_topology import MLPTopology
from quantumnematode.brain.arch._node_noise_schedule import NodeNoiseSchedule
from quantumnematode.learning_rules import ScalingOptions, ThreeFactorRule
from quantumnematode.learning_rules.three_factor import ThreeFactorBatch
from quantumnematode.plasticity.positive_control import ContextualAssociation
from torch import nn

# The panels' pins. The instrument under test is the rule the registered results used; a control
# run at some other setting would say nothing about them.
PLASTICITY_RATE = 1e-3
RATE_GRID = (1e-4, 1e-3, 1e-2)
WEIGHT_DECAY = 0.001
WEIGHT_BOUND = 3.0
BASELINE_RATE = 0.01
TRACE_DECAY = 0.9
NOISE = float(np.exp(-1.0))  # the arms' frozen `initial_log_std: -1.0`
HIDDEN = 8

SEEDS = tuple(range(1, 9))
TRIALS = 20_000
BLOCK = 100  # trials per alignment block; a single-step cosine of a stochastic estimator is noise
PASS_SEEDS = 7
PASS_FRACTION = 0.5  # of the floor-to-optimum gap
ANALYTIC_RATE = 0.05

NODE_NOISE_GRID = (0.01, 0.05, 0.2)
# The registered schedule. 0.2 is the scale that passed this control; 0.02 sits an order of
# magnitude below it -- under the grid's 0.05, which still cost 7 of 8 seeds the learning bar,
# and above 0.01, where the estimator was inert. The decay spans the first half of the budget,
# so the score window (the last BLOCK * 10 trials) lies entirely at the floor: a pass means the
# policy the schedule LEFT BEHIND is competent under the perturbation it will actually run with.
ANNEAL_INITIAL = 0.2
ANNEAL_FINAL = 0.02
ANNEAL_FRACTION = 0.5  # of the trial budget

# ── I.3: the settings pinned since A.3 and never examined ───────────────────────────────────
# Homeostasis and the exploration noise act on a one-step trial, so the undelayed control holds
# them. The eligibility horizon does not: this control resets the trace every trial, which is
# what removes the horizon confound from the undelayed question, so it needs a delay to act at
# all. The delays bracket the ten-step scale `trace_decay 0.9` implies -- 0.9^10 = 0.35,
# 0.9^20 = 0.12 -- and run past it; the longer decays ask whether a longer horizon recovers it.
DELAY_GRID = (0, 2, 5, 10, 20)
TRACE_DECAY_GRID = (0.9, 0.99, 0.999)
# The panels' own history: std 1.0 capped every plastic arm near its floor, 0.22 rose and
# collapsed, and a probe selected 0.37. The grid brackets the selected value with the two that
# failed, so an arm that the pinned noise was holding back is visible either side.
ACTION_NOISE_GRID = (0.22, NOISE, 0.61, 1.0)
PERTURBING_ARMS = frozenset({"node_perturbation", "node_perturbation_annealed"})
ARMS = ("three_factor", "node_perturbation", "node_perturbation_annealed", "hebbian", "analytic")


def _actor(n_cues: int, generator: torch.Generator, hidden: int = HIDDEN) -> nn.Sequential:
    """``Linear(K, H) -> tanh -> Linear(H, 1)``: the panels' arrangement, hidden layer plastic.

    ``hidden`` defaults to the pinned ``HIDDEN``, so every value recorded by I.0-I.3b reproduces
    unchanged; it is a parameter because the width is the perturbation dimension and the count of
    perturbed units on this arrangement is exactly ``hidden``.
    """
    if hidden < 1:
        msg = f"hidden must be >= 1, got {hidden}"
        raise ValueError(msg)
    first, readout = nn.Linear(n_cues, hidden), nn.Linear(hidden, 1)
    with torch.no_grad():
        for layer in (first, readout):
            nn.init.orthogonal_(layer.weight, generator=generator)
            layer.bias.zero_()
    return nn.Sequential(first, nn.Tanh(), readout)


def _descend(topology: MLPTopology, gradients: tuple[torch.Tensor | None, ...]) -> None:
    """Take the analytic reference's step: plain gradient descent on the task's own loss."""
    with torch.no_grad():
        for weight, gradient in zip(topology.plastic_weights, gradients, strict=True):
            if gradient is not None:
                weight.add_(-ANALYTIC_RATE * gradient)


def _block_alignment(
    update: list[torch.Tensor],
    gradient: list[torch.Tensor],
) -> float | None:
    """Cosine between a block's accumulated update and its summed gradient-descent direction.

    Measured over a block rather than a step because a single-step cosine of a stochastic
    estimator is noisy by nature and would read near zero even for a correct rule. ``None``
    when either side is identically zero and the angle is undefined.
    """
    flat_update = torch.cat([u.reshape(-1) for u in update])
    flat_gradient = torch.cat([g.reshape(-1) for g in gradient])
    if float(flat_update.norm()) == 0.0 or float(flat_gradient.norm()) == 0.0:
        return None
    return float(
        torch.nn.functional.cosine_similarity(flat_update, flat_gradient, dim=0).item(),
    )


def annealed_schedule(trials: int) -> NodeNoiseSchedule:
    """Build the registered schedule, with its decay scaled to a trial budget."""
    return NodeNoiseSchedule(
        initial=ANNEAL_INITIAL,
        final=ANNEAL_FINAL,
        episodes=max(1, int(trials * ANNEAL_FRACTION)),
    )


def _validate_delay(delay: int) -> None:
    """Refuse a negative delay, which would run undelayed while reporting itself delayed.

    ``range(delay)`` is empty below zero, so the filler steps would simply not happen and the
    record would describe a trial that never ran.
    """
    if delay < 0:
        msg = f"delay must be non-negative, got {delay}"
        raise ValueError(msg)


def _build(  # noqa: PLR0913 — one parameter per pinned dimension of the control
    arm: str,
    seed: int,
    task: ContextualAssociation,
    *,
    rate: float,
    node_noise: float,
    schedule: NodeNoiseSchedule | None,
    trace_decay: float,
    homeostasis: bool,
    hidden: int = HIDDEN,
) -> tuple[MLPTopology, ThreeFactorRule | None]:
    """Build the topology and the rule one arm runs over."""
    generator = torch.Generator().manual_seed(seed)
    perturbing = arm in PERTURBING_ARMS
    topology = MLPTopology(
        _actor(task.n_cues, generator, hidden),
        enable_activity_traces=True,
        trace_decay=trace_decay,
        plastic_layers="hidden",  # frozen readout: a plastic one collapses on its own output
        # The variant's own perturbation, from a dedicated generator; zero for every other arm,
        # which then draws nothing and runs the forward pass unchanged.
        node_noise=node_noise if perturbing else 0.0,
        node_noise_schedule=schedule if perturbing else None,
        perturbation_seed=seed if perturbing else None,
    )
    if arm == "analytic":
        return topology, None
    return topology, ThreeFactorRule(
        topology,
        plasticity_rate=rate,
        weight_decay=WEIGHT_DECAY,
        weight_bound=WEIGHT_BOUND,
        baseline_rate=BASELINE_RATE,
        freeze_updates=False,
        modulated=arm in {"three_factor", *PERTURBING_ARMS},
        eligibility="node_perturbation" if perturbing else "hebbian",
        scaling=ScalingOptions(normalise_modulator=True, normalise_trace=True),
        homeostasis=homeostasis,
        device=torch.device("cpu"),
    )


def run_arm(  # noqa: PLR0913 — one parameter per pinned dimension of the control
    arm: str,
    seed: int,
    task: ContextualAssociation,
    rate: float = PLASTICITY_RATE,
    trials: int = TRIALS,
    noise: float = NOISE,
    node_noise: float = 0.0,
    schedule: NodeNoiseSchedule | None = None,
    delay: int = 0,
    trace_decay: float = TRACE_DECAY,
    *,
    homeostasis: bool = True,
    hidden: int = HIDDEN,
) -> dict[str, Any]:
    """Run one arm at one seed and return its score and diagnosis."""
    _validate_delay(delay)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    perturbing = arm in PERTURBING_ARMS
    topology, rule = _build(
        arm,
        seed,
        task,
        rate=rate,
        node_noise=node_noise,
        schedule=schedule,
        trace_decay=trace_decay,
        homeostasis=homeostasis,
        hidden=hidden,
    )

    rewards: list[float] = []
    modulators: list[float] = []
    traces: list[float] = []
    alignments: list[float] = []
    decay_alignments: list[float] = []
    floor_alignments: list[float] = []
    # Where the decay ends, in trials. Without a schedule everything is "floor": the scale never
    # moved, so there is no decay phase to separate.
    decay_trials = schedule.episodes if schedule is not None else 0
    block_update = [torch.zeros_like(w) for w in topology.plastic_weights]
    block_gradient = [torch.zeros_like(w) for w in topology.plastic_weights]

    for trial in range(trials):
        # Each trial is its own episode: without this the eligibility gating this trial's reward
        # would carry the previous trial's cue -- the horizon confound this control removes.
        topology.reset_traces()
        # This control drives the topology directly and never builds a brain, so nothing else
        # would advance the schedule: the trial IS the schedule's step. A counter that only
        # advanced from a brain would leave an annealed arm at its initial scale throughout and
        # pass a gate the schedule was never tested by. The topology counts trials BEGUN and
        # indexes the running one at that count minus one, so this first trial runs at the
        # initial scale rather than one step into the decay.
        topology.advance_schedule()
        cue = task.sample_cue(rng)
        observation = torch.from_numpy(task.observation(cue))
        mean = topology(observation).squeeze()
        action = float(mean.item() + noise * rng.standard_normal())
        reward = task.reward(cue, action)
        rewards.append(reward)

        # The analytic gradient of the SCORED step's loss, for the alignment and for the reference
        # arm. Taken here and held: under a delay the update lands after the filler steps, where
        # the network's output answers the filler and its loss against the target means nothing.
        loss = (mean - float(task.targets[cue])) ** 2
        gradients = torch.autograd.grad(loss, list(topology.plastic_weights), allow_unused=True)

        # The delay: the reward arrives `delay` steps after the action it scores, so the credited
        # step's share of the eligibility falls as each filler step adds its own term. Dilution,
        # not decay -- a rule that normalises its trace divides a scalar decay straight out.
        if delay:
            filler = torch.from_numpy(task.filler())
            for _ in range(delay):
                topology(filler)

        # Both arms are measured on the same accumulation path, so the reference's alignment is
        # an end-to-end check on the sign convention rather than a unit-level assumption: the
        # arm that IS gradient descent must come out at +1.
        before = [w.detach().clone() for w in topology.plastic_weights]
        if rule is None:
            _descend(topology, gradients)
        else:
            report = rule.step(topology, ThreeFactorBatch(reward=reward))
            modulators.append(float(report.extra["plasticity_modulator"]))
            traces.append(float(report.extra["plasticity_mean_abs_delta"]))
        with torch.no_grad():
            for index, weight in enumerate(topology.plastic_weights):
                block_update[index] += weight.detach() - before[index]
                if gradients[index] is not None:
                    # The gradient-DESCENT direction, so a rule that reduces the loss aligns
                    # positively. The raw gradient would give a correct rule a cosine of -1.
                    block_gradient[index] += -gradients[index]

        # A block is closed on its own boundary and again exactly where the decay ends, so a
        # block spanning that point is not filed whole by whichever phase its last trial fell
        # in. Where the decay length is a multiple of BLOCK -- as the registered budget's is --
        # the second condition never fires on its own and the blocks are unchanged.
        if (trial + 1) % BLOCK == 0 or (0 < decay_trials == trial + 1):
            aligned = _block_alignment(block_update, block_gradient)
            if aligned is not None:
                alignments.append(aligned)
                # At the floor the estimator is nearly silent BY DESIGN, so a low
                # floor-phase alignment is what a good schedule looks like and is not the
                # failure signature. The signature is a decay-phase alignment that does not
                # rise together with a floor-phase score below the bar, which needs the two
                # phases kept apart rather than averaged into one number.
                if trial < decay_trials:
                    decay_alignments.append(aligned)
                else:
                    floor_alignments.append(aligned)
            block_update = [torch.zeros_like(w) for w in topology.plastic_weights]
            block_gradient = [torch.zeros_like(w) for w in topology.plastic_weights]

    return {
        "arm": arm,
        "seed": seed,
        # The perturbation dimension. On this arrangement -- one plastic layer, frozen readout --
        # the count of perturbed units IS the width, which is why this control can vary it without
        # varying anything else about how the estimate is formed.
        "hidden": hidden,
        # The variant runs at the pinned rate too; recording it keeps the row self-describing.
        "rate": rate if arm in {"three_factor", *PERTURBING_ARMS} else None,
        "node_noise": node_noise if perturbing else None,
        # The knobs this run pinned or varied, so a row describes its own settings.
        "delay": delay,
        "trace_decay": trace_decay,
        "homeostasis": homeostasis,
        "action_noise": noise,
        "nominal_credit_ratio": task.nominal_credit_ratio(trace_decay, delay),
        "schedule": (
            {
                "initial": schedule.initial,
                "final": schedule.final,
                "decay_trials": schedule.episodes,
            }
            if schedule is not None
            else None
        ),
        # Trace normalisation is on for every arm here. It matters to an annealed arm
        # specifically: the trace carries the perturbation, so the update's magnitude is
        # linear in the scale, and without this the decay would cut the effective rate as
        # well as the exploration. Recorded so the regime travels with the number.
        "normalise_trace": True,
        # The arm's score: mean reward over the last 1000 trials, so a run is judged
        # on where it ended rather than on the exploration it did getting there.
        "score": float(np.mean(rewards[-BLOCK * 10 :])) if rewards else float("nan"),
        # Per-block mean reward, in order. The score above is where a run ENDED; a rate needs the
        # whole curve, and a trailing-block series is the coarsest form that still carries one.
        # Deliberately absent from the per-seed CSV and from the control's JSON, both of which
        # list their fields explicitly, so no committed record changes shape.
        "reward_blocks": [
            float(np.mean(rewards[start : start + BLOCK]))
            for start in range(0, len(rewards) - len(rewards) % BLOCK, BLOCK)
        ],
        "modulator": float(np.mean(modulators)) if modulators else float("nan"),
        "mean_abs_delta": float(np.mean(traces)) if traces else float("nan"),
        "alignment": float(np.mean(alignments)) if alignments else float("nan"),
        "alignment_decay": (float(np.mean(decay_alignments)) if decay_alignments else float("nan")),
        "alignment_floor": (float(np.mean(floor_alignments)) if floor_alignments else float("nan")),
    }


def assess(scores: list[float], floor: float, optimum: float) -> dict[str, Any]:
    """Apply the registered pass rule to one arm's per-seed scores."""
    above = [s for s in scores if s > floor]
    mean = float(np.mean(scores)) if scores else float("nan")
    threshold = floor + PASS_FRACTION * (optimum - floor)
    return {
        "n": len(scores),
        "mean": mean,
        "seeds_above_floor": len(above),
        "halfway_threshold": threshold,
        "passes": len(scores) == len(SEEDS) and len(above) >= PASS_SEEDS and mean >= threshold,
    }


def analyse(
    runs: list[dict[str, Any]],
    task: ContextualAssociation,
    trials: int = TRIALS,
) -> dict[str, Any]:
    """Score every arm, apply the pass rule and decide pass, fail or void."""
    floor, optimum = task.cue_blind_floor(NOISE), task.optimum(NOISE)
    by_arm: dict[str, Any] = {}
    for arm in ("hebbian", "analytic"):
        scores = [r["score"] for r in runs if r["arm"] == arm]
        by_arm[arm] = assess(scores, floor, optimum)
    rates: dict[str, Any] = {}
    for rate in RATE_GRID:
        scores = [r["score"] for r in runs if r["arm"] == "three_factor" and r["rate"] == rate]
        rates[str(rate)] = assess(scores, floor, optimum)
    noises: dict[str, Any] = {}
    for node_noise in NODE_NOISE_GRID:
        scores = [
            r["score"]
            for r in runs
            if r["arm"] == "node_perturbation" and r["node_noise"] == node_noise
        ]
        noises[str(node_noise)] = assess(scores, floor, optimum)
    by_arm["node_perturbation"] = {
        "by_node_noise": noises,
        # Any perturbation scale passing counts, as any rate does for the three-factor arm:
        # the claim under test is that the variant learns at all.
        "passes": any(v["passes"] for v in noises.values()),
    }
    annealed = [r for r in runs if r["arm"] == "node_perturbation_annealed"]
    by_arm["node_perturbation_annealed"] = assess(
        [r["score"] for r in annealed],
        floor,
        optimum,
    )
    by_arm["node_perturbation_annealed"]["schedule"] = annealed[0]["schedule"] if annealed else None
    # The two phases kept apart. A low floor-phase alignment is expected of a good schedule --
    # the estimator is nearly silent once the scale is small -- so the failure signature is a
    # decay-phase alignment that does not rise together with a floor-phase score below the bar,
    # and averaging the phases would hide exactly that.
    for phase in ("alignment_decay", "alignment_floor"):
        values = [r[phase] for r in annealed if not math.isnan(r[phase])]
        by_arm["node_perturbation_annealed"][phase] = (
            float(np.mean(values)) if values else float("nan")
        )
    by_arm["three_factor"] = {
        "by_rate": rates,
        # Any rate passing counts: the claim is that the rule learns at all.
        "passes": any(v["passes"] for v in rates.values()),
        "pinned_rate": str(PLASTICITY_RATE),
    }

    void_reason = None
    if not by_arm["analytic"]["passes"]:
        void_reason = (
            "the analytic reference did not pass: the task, the topology or the optimiser is at "
            "fault, and the three-factor arm's result carries no information about the rule"
        )
    elif by_arm["hebbian"]["passes"]:
        void_reason = (
            "the unmodulated arm passed: the task leaks its answer without reward, so the "
            "three-factor arm's result carries no information about the rule"
        )
    # `outcome` scores the ORIGINAL rule, which is the question this control was registered to
    # answer. A variant arm carries its own `passes` flag and is read from there, so that adding
    # an arm can never change what the control says about the rule it was built for.
    outcome = "void" if void_reason else ("pass" if by_arm["three_factor"]["passes"] else "fail")

    diagnosis = {arm: _diagnose(runs, arm) for arm in ARMS}
    # Per rate as well as pooled: a rule whose alignment depended on the rate would be a
    # different finding from one whose updates are unaimed at every rate.
    diagnosis["three_factor"]["by_rate"] = {
        str(rate): _diagnose(runs, "three_factor", rate=rate) for rate in RATE_GRID
    }
    diagnosis["node_perturbation"]["by_node_noise"] = {
        str(node_noise): _diagnose(runs, "node_perturbation", node_noise=node_noise)
        for node_noise in NODE_NOISE_GRID
    }
    return {
        "task": {
            "n_cues": task.n_cues,
            "targets": task.targets.tolist(),
            "noise": NOISE,
            "cue_blind_floor": floor,
            "optimum": optimum,
            "gap": task.gap(NOISE),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "trials": trials,
            "rate_grid": list(RATE_GRID),
            "node_noise_grid": list(NODE_NOISE_GRID),
            "pass_seeds": PASS_SEEDS,
            "pass_fraction_of_gap": PASS_FRACTION,
            "note": "any rate passing counts; the claim under test is that the rule learns at all",
        },
        "arms": by_arm,
        "diagnosis": diagnosis,
        "outcome": outcome,
        "void_reason": void_reason,
    }


def _jsonable(value: object) -> object:
    """Replace not-a-number with null, recursively, so the record is strict JSON."""
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _diagnose(
    runs: list[dict[str, Any]],
    arm: str,
    rate: float | None = None,
    node_noise: float | None = None,
) -> dict[str, Any]:
    """Summarise one arm's diagnostic series, pooled or at one rate.

    The alignment carries a median as well as a mean: it is a per-run statistic with a long
    tail (one seed reaching +0.26 while the rest sit near zero), and a mean over eight such
    runs overstates the typical run. Both are recorded so the record and the prose can quote
    the same numbers.
    """
    selected = [
        r
        for r in runs
        if r["arm"] == arm
        and (rate is None or r["rate"] == rate)
        and (node_noise is None or r["node_noise"] == node_noise)
    ]
    out: dict[str, Any] = {}
    for key in ("modulator", "mean_abs_delta", "alignment"):
        values = [r[key] for r in selected if not np.isnan(r[key])]
        out[key] = float(np.mean(values)) if values else float("nan")
        if key == "alignment":
            out["alignment_median"] = float(np.median(values)) if values else float("nan")
    return out


def _print_control(out: dict[str, Any]) -> None:
    """Print the control, floors first."""
    task = out["task"]
    print(
        f"\nRule positive control ({len(out['protocol']['seeds'])} seeds x "
        f"{out['protocol']['trials']} trials)",
    )
    print(
        f"  floor {task['cue_blind_floor']:.4f}   optimum {task['optimum']:.4f}   "
        f"gap {task['gap']:.4f}",
    )
    for arm in ("analytic", "hebbian"):
        row = out["arms"][arm]
        alignment = out["diagnosis"].get(arm, {}).get("alignment", float("nan"))
        print(
            f"  {arm:12} mean {row['mean']:+.4f}  {row['seeds_above_floor']}/{row['n']} above "
            f"floor  alignment {alignment:+.4f}  "
            f"-> {'passes' if row['passes'] else 'does not pass'}",
        )
    print("  three_factor by rate:")
    for rate, row in out["arms"]["three_factor"]["by_rate"].items():
        print(
            f"    rate {rate:>7} mean {row['mean']:+.4f}  {row['seeds_above_floor']}/{row['n']} "
            f"above floor  -> {'passes' if row['passes'] else 'does not pass'}",
        )
    print("  node_perturbation by sigma:")
    for node_noise, row in out["arms"]["node_perturbation"]["by_node_noise"].items():
        alignment = out["diagnosis"]["node_perturbation"]["by_node_noise"][node_noise]["alignment"]
        print(
            f"    sigma {node_noise:>6} mean {row['mean']:+.4f}  "
            f"{row['seeds_above_floor']}/{row['n']} above floor  alignment {alignment:+.4f}"
            f"  -> {'passes' if row['passes'] else 'does not pass'}",
        )
    annealed = out["arms"].get("node_perturbation_annealed")
    if annealed is not None:
        schedule = annealed.get("schedule") or {}
        print(
            f"  node_perturbation_annealed (sigma {schedule.get('initial')} -> "
            f"{schedule.get('final')} over {schedule.get('decay_trials')} trials):",
        )
        print(
            f"    mean {annealed['mean']:+.4f}  "
            f"{annealed['seeds_above_floor']}/{annealed['n']} above floor  "
            f"alignment {annealed['alignment_decay']:+.4f} decay / "
            f"{annealed['alignment_floor']:+.4f} floor"
            f"  -> {'passes' if annealed['passes'] else 'does not pass'}",
        )
    diag = out["diagnosis"]["three_factor"]
    print(
        f"\n  diagnosis (three_factor): modulator {diag['modulator']:+.4f}  "
        f"|dw| {diag['mean_abs_delta']:.2e}  gradient alignment "
        f"{diag['alignment']:+.4f} mean / {diag['alignment_median']:+.4f} median",
    )
    print(f"\n  OUTCOME: {out['outcome']}")
    if out["void_reason"]:
        print(f"  void: {out['void_reason']}")


def write_per_seed_csv(runs: list[dict[str, Any]], path: Path) -> None:
    """One row per arm, rate and seed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "arm",
                "rate",
                "node_noise",
                "seed",
                "score",
                "modulator",
                "mean_abs_delta",
                "alignment",
                "alignment_decay",
                "alignment_floor",
            ],
        )
        for run in runs:
            writer.writerow(
                [
                    run["arm"],
                    run["rate"],
                    run["node_noise"],
                    run["seed"],
                    f"{run['score']:.6f}",
                    f"{run['modulator']:.6f}",
                    f"{run['mean_abs_delta']:.8f}",
                    f"{run['alignment']:.6f}",
                    f"{run['alignment_decay']:.6f}",
                    f"{run['alignment_floor']:.6f}",
                ],
            )


def main(argv: list[str] | None = None) -> int:
    """Run the control and write its records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=TRIALS)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args(argv)

    # The annealed arm decays over the first half of its budget and is scored on the last
    # BLOCK * 10 trials. Below twice that window the score would be taken partly during the
    # decay, so it would measure the exploration the arm did rather than the policy it left.
    minimum = BLOCK * 10 * 2
    if args.trials < minimum:
        print(
            f"--trials must be at least {minimum} to run the annealed arm: at {args.trials} the "
            f"decay (first {int(args.trials * ANNEAL_FRACTION)} trials) reaches into the "
            f"{BLOCK * 10}-trial score window.",
            file=sys.stderr,
        )
        return 2

    task = ContextualAssociation.default()
    runs: list[dict[str, Any]] = []
    for seed in SEEDS:
        runs.append(
            run_arm(
                "node_perturbation_annealed",
                seed,
                task,
                trials=args.trials,
                node_noise=ANNEAL_INITIAL,
                schedule=annealed_schedule(args.trials),
            ),
        )
        runs.extend(run_arm(arm, seed, task, trials=args.trials) for arm in ("analytic", "hebbian"))
        runs.extend(
            run_arm("three_factor", seed, task, rate=rate, trials=args.trials) for rate in RATE_GRID
        )
        runs.extend(
            run_arm("node_perturbation", seed, task, trials=args.trials, node_noise=node_noise)
            for node_noise in NODE_NOISE_GRID
        )
    out = analyse(runs, task, trials=args.trials)
    _print_control(out)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        # `allow_nan=False` refuses bare NaN, which is not JSON and which a strict reader
        # rejects; unavailable diagnostics (the analytic arm has no modulator or eligibility)
        # are written as null instead.
        args.out.write_text(
            json.dumps(_jsonable(out), indent=2, sort_keys=True, allow_nan=False) + "\n",
        )
    if args.csv:
        write_per_seed_csv(runs, args.csv)
    # A `fail` is a valid experimental result and exits zero; a `void` means the control itself
    # did not hold and needs rebuilding, which is an operational failure.
    return 1 if out["outcome"] == "void" else 0


if __name__ == "__main__":
    raise SystemExit(main())
