"""Whether the frozen motor readout is what stops the rule learning on the connectome.

R.1c closed the perturbation dimension and eliminated two of the three tensors PPO trains that the rule
cannot write: the sensory projection, which PPO barely rotates, and the action-noise scale, swept by hand
and null. The readout is the one left.

Preparing the arms showed PPO does not refine the anatomical prior but **replaces** it -- norm 7.820
against 1.414, cosine -0.178 -- so the comparison needs both controls it now has:

* ``anatomical`` -- the committed default, R.1c's own `motor` pair. Reused rather than re-run, licensed by
  a load-path equivalence test: loading a prepared anatomical checkpoint leaves every topology tensor, the
  rule's baseline and the homeostatic norm targets bit-identical.
* ``anatomical_scaled`` -- the anatomical **direction** at PPO's **norm**. Isolates scale, which matters
  here because the readout multiplies the motor-class means into the action mean while the action noise is
  fixed, so a 5.5x readout is a change in commitment rather than in what is read.
* ``rotated`` -- a random direction at PPO's norm. Isolates "any direction" from "PPO's direction".
* ``ppo`` -- the harvested readout.

Two things about the reading, both fixed before the run:

* **Beating a floor is not learning the cell.** Whether an arm clears its own frozen control and whether
  it reaches competence are reported separately, because only the second makes block V's
  time-to-competence contrast defined and only the second bears on R.1b. R.1c's verdict condition proved
  weaker than its own stated consequence, and this keeps the two apart from the start.
* **This cannot satisfy the plausibility deliverable whatever it returns.** An arm whose readout was
  trained by backpropagation is not a biologically plausible local learner. A positive result locates a
  handicap; its repair has to come from a better-grounded readout, not from PPO.
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

import l4_mixture_statistic as ms  # noqa: E402  # pyright: ignore[reportMissingImports]
import l4_perturbation_scale as ps  # noqa: E402  # pyright: ignore[reportMissingImports]
import l4_reduced_perturbation as rp  # noqa: E402  # pyright: ignore[reportMissingImports]
from l4_panel import EXPERIMENTS, read_log  # noqa: E402  # pyright: ignore[reportMissingImports]

SEEDS = tuple(range(1, 9))
_STEM = "connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_hard350_nodepert_motor"

# The mask is held fixed at R.1c's best measured operating point; only the readout varies.
MASK = "motor"
# Frobenius norm and cosine to the anatomical default, measured from the prepared checkpoints.
ARMS: dict[str, dict[str, Any]] = {
    "anatomical": {"norm": 1.414, "cosine": 1.000, "baseline": True},
    "anatomical_scaled": {"norm": 7.820, "cosine": 1.000, "baseline": False},
    "rotated": {"norm": 7.820, "cosine": -0.028, "baseline": False},
    "ppo": {"norm": 7.820, "cosine": -0.178, "baseline": False},
}
SUBSTITUTED = tuple(name for name, meta in ARMS.items() if not meta["baseline"])

# PPO on this cell at these eight seeds and at the arms' own action scale. The matched reference: 058's
# 19.31 foods came from 32 seeds at an action std of 1.0, and is reported beside this one.
PPO_MATCHED_FOODS = 18.945
PPO_REFERENCE_058 = 19.31
MIN_FOODS = rp.MIN_FOODS
MIN_GAP_FRACTION = rp.MIN_GAP_FRACTION
COMPETENCE_THRESHOLD = ms.COMPETENT_THRESHOLD

_LABEL = re.compile(
    rf"^{re.escape(_STEM)}(?:_(?P<arm>anatomical_scaled|rotated|ppo))?(?P<frozen>_frozen)?"
    r"-seed(?P<seed>\d+)\.log$",
)


def scan(
    directories: dict[str, Path],
    experiments: Path = EXPERIMENTS,
) -> dict[str, dict[str, Any]]:
    """Read every registered run, keyed by readout and arm.

    Two directories: the anatomical pair comes from R.1c's campaign, the substituted arms from this
    change's. A label with no readout suffix is the anatomical arm, which is why the suffix is optional.
    """
    out: dict[str, dict[str, Any]] = {
        name: {"learning": {}, "frozen": {}, "logs": {}} for name in ARMS
    }
    for label, directory in directories.items():
        log_dir = directory / "logs" if (directory / "logs").is_dir() else directory
        for log in sorted(log_dir.glob("*.log")):
            match = _LABEL.match(log.name)
            if match is None:
                continue
            name = match.group("arm") or "anatomical"
            if name not in out:
                continue
            # R.1c's directory carries every mask; only its `motor` arm is this comparison's baseline,
            # and the label regex already pins the stem to that mask.
            if label == "anatomical" and name != "anatomical":
                continue
            if label != "anatomical" and name == "anatomical":
                continue
            record = read_log(log, experiments)
            if record is None:
                print(f"  WARN: no parseable run lines in {log.name} - dropped")
                continue
            arm = "frozen" if match.group("frozen") else "learning"
            seed = int(match.group("seed"))
            if seed in out[name][arm]:
                msg = f"two logs for readout {name} arm {arm} seed {seed}: {log.name} duplicates a run"
                raise ValueError(msg)
            out[name][arm][seed] = record
            out[name]["logs"].setdefault(arm, {})[seed] = log
    return out


def require_complete(scanned: dict[str, dict[str, Any]], seeds: tuple[int, ...] = SEEDS) -> None:
    """Refuse to score a campaign missing any registered cell."""
    missing: list[str] = []
    for name in ARMS:
        for arm in ("learning", "frozen"):
            absent = [s for s in seeds if s not in scanned[name][arm]]
            if absent:
                missing.append(f"{name}/{arm} seeds {absent}")
    if missing:
        msg = "campaign is incomplete, so no verdict is available: " + "; ".join(missing)
        raise ValueError(msg)


def minima(effect: float, frozen_mean: float) -> dict[str, Any]:
    """Both registered minima, with the reachable gap taken against the matched PPO reference."""
    matched = MIN_GAP_FRACTION * (PPO_MATCHED_FOODS - frozen_mean)
    reported_058 = MIN_GAP_FRACTION * (PPO_REFERENCE_058 - frozen_mean)
    binding = max(matched, MIN_FOODS)
    return {
        "effect": effect,
        "absolute_minimum": MIN_FOODS,
        "matched_minimum": matched,
        "minimum_against_058": reported_058,
        "binding_minimum": binding,
        "passes": bool(effect >= binding),
        "why": (
            "at or above both minima"
            if effect >= binding
            else f"below the binding minimum of {binding:.2f} foods"
        ),
    }


def compare(
    name: str,
    data: dict[str, Any],
    seeds: tuple[int, ...] = SEEDS,
    experiments: Path = EXPERIMENTS,
) -> dict[str, Any]:
    """Score one readout's learning arm against its own frozen control."""
    learning = {s: r.foods for s, r in data["learning"].items() if s in seeds}
    frozen = {s: r.foods for s, r in data["frozen"].items() if s in seeds}
    clear = [r.success for s, r in data["learning"].items() if s in seeds]
    frozen_mean = float(np.mean(list(frozen.values()) or [math.nan]))
    graded = ms.shift_contrast(learning, frozen)
    mean_clear = float(np.mean(clear)) if clear else float("nan")
    return {
        "readout": name,
        "norm": ARMS[name]["norm"],
        "cosine_to_anatomical": ARMS[name]["cosine"],
        "n_pairs": len(set(learning) & set(frozen)),
        "learning_mean_foods": float(np.mean(list(learning.values()) or [math.nan])),
        "frozen_mean_foods": frozen_mean,
        "learning_foods": learning,
        "frozen_foods": frozen,
        "learning_mean_full_clear": mean_clear,
        "reaches_competence": bool(mean_clear >= COMPETENCE_THRESHOLD),
        "competence_threshold": COMPETENCE_THRESHOLD,
        "graded": graded,
        "minima": minima(graded.get("effect", float("nan")), frozen_mean),
        "drift": rp.split_drift(MASK, data["logs"], seeds, experiments),
    }


def _ordering(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Which of scale or direction the ordering supports, per the registered readings."""
    level = {name: cells[name]["learning_mean_foods"] for name in ARMS}
    anatomical, scaled = level["anatomical"], level["anatomical_scaled"]
    rotated, ppo = level["rotated"], level["ppo"]
    span = max(level.values()) - min(level.values())
    scale_gain = scaled - anatomical
    direction_gain = ppo - scaled
    if span < MIN_FOODS:
        reading = "all four alike: the readout is not the handicap"
    elif scale_gain >= MIN_FOODS and abs(direction_gain) < MIN_FOODS:
        reading = (
            "the SCALE is what mattered -- commitment against a fixed action noise, not what is read; "
            "the follow-up is the readout-scale/action-noise interaction, not a better-grounded readout"
        )
    elif direction_gain >= MIN_FOODS and ppo - rotated >= MIN_FOODS:
        reading = "PPO's DIRECTION matters and it found a good one; the follow-up is a better-grounded readout"
    elif max(rotated, ppo) - scaled >= MIN_FOODS:
        reading = (
            "any change of direction helps at this scale: the anatomical direction is actively bad, and "
            "the follow-up is to fix the prior rather than to credit PPO with finding something"
        )
    else:
        reading = "mixed: the ordering matches no registered pattern and is reported as it is"
    return {
        "levels": level,
        "span_foods": span,
        "scale_gain": scale_gain,
        "direction_gain": direction_gain,
        "reading": reading,
    }


def analyse(
    scanned: dict[str, dict[str, Any]],
    seeds: tuple[int, ...] = SEEDS,
    experiments: Path = EXPERIMENTS,
) -> dict[str, Any]:
    """Compare every readout, correct across them and apply the registered rule."""
    cells = {name: compare(name, scanned[name], seeds, experiments) for name in ARMS}
    defined = [n for n in ARMS if cells[n]["graded"].get("defined")]
    qs = ms.bh_fdr([cells[n]["graded"]["p_improve"] for n in defined])
    for name, q in zip(defined, qs, strict=True):
        cells[name]["graded"]["q_improve"] = float(q)
    for name in ARMS:
        cell = cells[name]
        q = cell["graded"].get("q_improve", float("nan"))
        significant = bool(q <= ms.SIG_Q) if not math.isnan(q) else False
        if significant and cell["minima"]["passes"]:
            cell["verdict"] = "beats_floor"
        elif significant:
            cell["verdict"] = "below_min_effect"
        else:
            cell["verdict"] = "no_improvement"
    beat = [n for n in SUBSTITUTED if cells[n]["verdict"] == "beats_floor"]
    competent = [n for n in ARMS if cells[n]["reaches_competence"]]
    if competent:
        verdict, why = (
            "readout_is_the_handicap",
            (
                f"{', '.join(competent)} reaches competence: the handicap is located. The repair must "
                "come from a better-grounded readout, not from PPO -- a rule needing a gradient-trained "
                "tensor is not a plausible local learner"
            ),
        )
    elif beat:
        verdict, why = (
            "readout_helps_but_not_enough",
            (
                f"{', '.join(beat)} beats its own floor by the registered minima without reaching "
                "competence: a result, not a rescue. R.1b stays blocked and R.2 stays live"
            ),
        )
    else:
        verdict, why = (
            "readout_not_the_handicap",
            (
                "no substituted readout beats its own floor by the registered minima: the third of "
                "three frozen tensors is eliminated, and R.2 becomes the live path"
            ),
        )
    return {
        "readouts": cells,
        "substituted_beating_floor": beat,
        "reaching_competence": competent,
        "ordering": _ordering(cells),
        "verdict": verdict,
        "why": why,
        "satisfies_plausibility_deliverable": False,
        "deliverable_note": (
            "This cannot satisfy D1 whatever it returned: the substituted arms take their readout from a "
            "gradient-trained run, so they are not biologically plausible local learners. A positive "
            "result locates a handicap and names a follow-up; it is not the plausible learner working."
        ),
    }


def _print(result: dict[str, Any]) -> None:
    print("\nR.1d — the frozen motor readout")
    print(
        "  readout            |  norm | cos   | learning | frozen | shift |     q | clear % "
        "| drift-cr | verdict",
    )
    for name in ARMS:
        cell = result["readouts"][name]
        q = cell["graded"].get("q_improve", float("nan"))
        print(
            f"  {name:<18} | {cell['norm']:>5.3f} | {cell['cosine_to_anatomical']:>+5.2f} | "
            f"{cell['learning_mean_foods']:>8.3f} | {cell['frozen_mean_foods']:>6.3f} | "
            f"{cell['graded'].get('effect', float('nan')):>+5.2f} | {q:5.3f} | "
            f"{cell['learning_mean_full_clear']:>7.2f} | "
            f"{cell['drift']['credited_mean_relative']:>8.2f} | {cell['verdict']}",
        )
    order = result["ordering"]
    print(f"\n  levels: { ({k: round(v, 3) for k, v in order['levels'].items()}) }")
    print(f"  scale gain {order['scale_gain']:+.3f}, direction gain {order['direction_gain']:+.3f}")
    print(f"  ordering reads: {order['reading']}")
    print(
        f"\n  substituted readouts beating their floor: {result['substituted_beating_floor'] or 'none'}",
    )
    print(f"  readouts reaching competence: {result['reaching_competence'] or 'none'}")
    print(f"\nVERDICT: {result['verdict']} — {result['why']}")
    print(f"\n{result['deliverable_note']}")


def write_csv(result: dict[str, Any], path: Path) -> None:
    """One row per readout and seed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "readout",
                "norm",
                "cosine_to_anatomical",
                "seed",
                "learning_foods",
                "frozen_foods",
                "favours_learning",
            ],
        )
        for name in ARMS:
            cell = result["readouts"][name]
            learning, frozen = cell["learning_foods"], cell["frozen_foods"]
            for seed in sorted(set(learning) & set(frozen)):
                writer.writerow(
                    [
                        name,
                        ARMS[name]["norm"],
                        ARMS[name]["cosine"],
                        seed,
                        f"{learning[seed]:.4f}",
                        f"{frozen[seed]:.4f}",
                        int(learning[seed] > frozen[seed]),
                    ],
                )


def main(argv: list[str] | None = None) -> int:
    """Score the readout arms against R.1c's committed anatomical pair."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anatomical", type=Path, required=True, help="R.1c's campaign directory")
    parser.add_argument("--substituted", type=Path, required=True, help="this change's directory")
    parser.add_argument("--seeds", default="1-8")
    parser.add_argument("--experiments", type=Path, default=EXPERIMENTS)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args(argv)

    low, _, high = args.seeds.partition("-")
    seeds = tuple(range(int(low), int(high or low) + 1))
    scanned = scan(
        {"anatomical": args.anatomical, "substituted": args.substituted},
        args.experiments,
    )
    if not args.allow_incomplete:
        require_complete(scanned, seeds)
    result = analyse(scanned, seeds, args.experiments)
    _print(result)
    if args.csv:
        write_csv(result, args.csv)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                # The control's own NaN-to-null pass, reused so the record is strict JSON.
                ps.pc._jsonable(
                    {
                        "protocol": {
                            "seeds": list(seeds),
                            "mask": MASK,
                            "arms": ARMS,
                            "ppo_matched_foods": PPO_MATCHED_FOODS,
                            "ppo_reference_058": PPO_REFERENCE_058,
                            "min_foods": MIN_FOODS,
                            "min_gap_fraction": MIN_GAP_FRACTION,
                            "competence_threshold": COMPETENCE_THRESHOLD,
                        },
                        "result": result,
                    },
                ),
                indent=2,
            )
            + "\n",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
