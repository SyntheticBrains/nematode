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
from pathlib import Path
from typing import Any

import numpy as np
import torch
from quantumnematode.brain.arch._mlp_topology import MLPTopology
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

ARMS = ("three_factor", "hebbian", "analytic")


def _actor(n_cues: int, generator: torch.Generator) -> nn.Sequential:
    """``Linear(K, 8) -> tanh -> Linear(8, 1)``: the panels' arrangement, hidden layer plastic."""
    first, readout = nn.Linear(n_cues, HIDDEN), nn.Linear(HIDDEN, 1)
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
    """Cosine between a block's accumulated update and that block's summed gradient.

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


def run_arm(  # noqa: PLR0913 — one parameter per pinned dimension of the control
    arm: str,
    seed: int,
    task: ContextualAssociation,
    rate: float = PLASTICITY_RATE,
    trials: int = TRIALS,
    noise: float = NOISE,
) -> dict[str, Any]:
    """Run one arm at one seed and return its score and diagnosis."""
    rng = np.random.default_rng(seed)
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    actor = _actor(task.n_cues, generator)
    topology = MLPTopology(
        actor,
        enable_activity_traces=True,
        trace_decay=TRACE_DECAY,
        plastic_layers="hidden",  # frozen readout: a plastic one collapses on its own output
    )
    rule = None
    if arm != "analytic":
        rule = ThreeFactorRule(
            topology,
            plasticity_rate=rate,
            weight_decay=WEIGHT_DECAY,
            weight_bound=WEIGHT_BOUND,
            baseline_rate=BASELINE_RATE,
            freeze_updates=False,
            modulated=(arm == "three_factor"),
            scaling=ScalingOptions(normalise_modulator=True, normalise_trace=True),
            homeostasis=True,
            device=torch.device("cpu"),
        )

    rewards: list[float] = []
    modulators: list[float] = []
    traces: list[float] = []
    alignments: list[float] = []
    block_update = [torch.zeros_like(w) for w in topology.plastic_weights]
    block_gradient = [torch.zeros_like(w) for w in topology.plastic_weights]

    for trial in range(trials):
        # Each trial is its own episode: without this the eligibility gating this trial's reward
        # would carry the previous trial's cue -- the horizon confound this control removes.
        topology.reset_traces()
        cue = task.sample_cue(rng)
        observation = torch.from_numpy(task.observation(cue))
        mean = topology(observation).squeeze()
        action = float(mean.item() + noise * rng.standard_normal())
        reward = task.reward(cue, action)
        rewards.append(reward)

        # The analytic gradient of THIS step's loss, for the alignment and for the reference arm.
        loss = (mean - float(task.targets[cue])) ** 2
        gradients = torch.autograd.grad(loss, list(topology.plastic_weights), allow_unused=True)

        if rule is None:
            _descend(topology, gradients)
            continue

        before = [w.detach().clone() for w in topology.plastic_weights]
        report = rule.step(topology, ThreeFactorBatch(reward=reward))
        with torch.no_grad():
            for index, weight in enumerate(topology.plastic_weights):
                block_update[index] += weight.detach() - before[index]
                if gradients[index] is not None:
                    block_gradient[index] += -gradients[index]
        modulators.append(float(report.extra["plasticity_modulator"]))
        traces.append(float(report.extra["plasticity_mean_abs_delta"]))

        if (trial + 1) % BLOCK == 0:
            aligned = _block_alignment(block_update, block_gradient)
            if aligned is not None:
                alignments.append(aligned)
            block_update = [torch.zeros_like(w) for w in topology.plastic_weights]
            block_gradient = [torch.zeros_like(w) for w in topology.plastic_weights]

    return {
        "arm": arm,
        "seed": seed,
        "rate": rate if arm == "three_factor" else None,
        # The arm's score: mean reward over the last 1000 trials, so a run is judged
        # on where it ended rather than on the exploration it did getting there.
        "score": float(np.mean(rewards[-BLOCK * 10 :])) if rewards else float("nan"),
        "modulator": float(np.mean(modulators)) if modulators else float("nan"),
        "mean_abs_delta": float(np.mean(traces)) if traces else float("nan"),
        "alignment": float(np.mean(alignments)) if alignments else float("nan"),
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
    outcome = "void" if void_reason else ("pass" if by_arm["three_factor"]["passes"] else "fail")

    diagnosis = {
        arm: {
            key: float(np.mean([r[key] for r in runs if r["arm"] == arm and not np.isnan(r[key])]))
            if any(r["arm"] == arm and not np.isnan(r[key]) for r in runs)
            else float("nan")
            for key in ("modulator", "mean_abs_delta", "alignment")
        }
        for arm in ("three_factor", "hebbian")
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
            "pass_seeds": PASS_SEEDS,
            "pass_fraction_of_gap": PASS_FRACTION,
            "note": "any rate passing counts; the claim under test is that the rule learns at all",
        },
        "arms": by_arm,
        "diagnosis": diagnosis,
        "outcome": outcome,
        "void_reason": void_reason,
    }


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
        print(
            f"  {arm:12} mean {row['mean']:+.4f}  {row['seeds_above_floor']}/{row['n']} above "
            f"floor  -> {'passes' if row['passes'] else 'does not pass'}",
        )
    print("  three_factor by rate:")
    for rate, row in out["arms"]["three_factor"]["by_rate"].items():
        print(
            f"    rate {rate:>7} mean {row['mean']:+.4f}  {row['seeds_above_floor']}/{row['n']} "
            f"above floor  -> {'passes' if row['passes'] else 'does not pass'}",
        )
    diag = out["diagnosis"]["three_factor"]
    print(
        f"\n  diagnosis (three_factor): modulator {diag['modulator']:+.4f}  "
        f"|dw| {diag['mean_abs_delta']:.2e}  gradient alignment {diag['alignment']:+.4f}",
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
            ["arm", "rate", "seed", "score", "modulator", "mean_abs_delta", "alignment"],
        )
        for run in runs:
            writer.writerow(
                [
                    run["arm"],
                    run["rate"],
                    run["seed"],
                    f"{run['score']:.6f}",
                    f"{run['modulator']:.6f}",
                    f"{run['mean_abs_delta']:.8f}",
                    f"{run['alignment']:.6f}",
                ],
            )


def main(argv: list[str] | None = None) -> int:
    """Run the control and write its records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=TRIALS)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args(argv)

    task = ContextualAssociation.default()
    runs: list[dict[str, Any]] = []
    for seed in SEEDS:
        runs.extend(run_arm(arm, seed, task, trials=args.trials) for arm in ("analytic", "hebbian"))
        runs.extend(
            run_arm("three_factor", seed, task, rate=rate, trials=args.trials) for rate in RATE_GRID
        )
    out = analyse(runs, task, trials=args.trials)
    _print_control(out)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    if args.csv:
        write_per_seed_csv(runs, args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
