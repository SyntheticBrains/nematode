"""The eligibility horizon on a multi-step task, asking whether I.3 carries off the control.

I.3 found the horizon limiting where it could be measured: on the positive control with a delayed
reward, the pinned ``trace_decay 0.9`` takes the rule below the cue-blind floor by twenty steps of
delay, and 0.99 recovers most of the gap there. Every result in this phase ran at that pinned
default -- no panel config sets it -- on episodes of 244 to 2400 steps. This asks whether raising it
moves anything on a real task.

The platform is the MLP yardstick, which runs in minutes where a connectome run takes hours, and
where the rule currently ends *below its own frozen control*. Each horizon is scored against its own
frozen control, because what the perturbation costs a policy is not constant across the setting
being varied, and a control from another horizon would not be the right null.

Two things about the reading, both registered before the run:

* **The full-clear metric cannot serve.** Every yardstick arm ever run sits at its floor, so the
  competence-dependent contrasts are undefined and are reported as such rather than as nulls. The
  graded reading -- plateau-tail mean foods -- is the only measure with room left, and it is not a
  comfortable one: the committed yardstick value is 0.35 of 10 over a seed range of 0.06 to 0.67.
* **Significance is not sufficient.** A paired rank test at eight seeds responds to the consistency
  of the sign, not the size of the shift, so on a floor-adjacent platform a clean-looking but
  meaningless result is reachable. A shift counts only if it is also at least ``MIN_EFFECT_FOODS``.
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
from l4_panel import (  # noqa: E402
    EXPERIMENTS,  # pyright: ignore[reportMissingImports]
    REPO,
    _experiment_json,
    read_log,
)

_STEM = "mlpppo_small_continuous2d_combined_klinotaxis_plastic_nodepert"
HORIZONS: dict[str, float] = {"td09": 0.9, "td099": 0.99, "td0999": 0.999}
SEEDS = tuple(range(1, 9))

# Significance alone does not carry the verdict: the test fires on the consistency of the sign, and
# the arms sit between 0.06 and 0.67 foods of 10. Fixed before the run at roughly the I.3 pilot's
# frozen-vs-learning gap and ~1.5 within-arm standard deviations of the committed table.
MIN_EFFECT_FOODS = 0.5
# The committed yardstick under the ORIGINAL rule (Logbook 040). A descriptive reference, never the
# comparator: it ran under a different eligibility, so a contrast against it would confound the two.
COMMITTED_YARDSTICK_FOODS = 0.348
_LABEL = re.compile(
    rf"^{re.escape(_STEM)}_(?P<horizon>td\d+)(?P<frozen>_frozen)?-seed(?P<seed>\d+)\.log$",
)


def scan(campaign_dir: Path, experiments: Path = EXPERIMENTS) -> dict[str, dict[str, Any]]:
    """Read every registered run under a campaign directory, keyed by horizon and arm."""
    log_dir = campaign_dir / "logs" if (campaign_dir / "logs").is_dir() else campaign_dir
    out: dict[str, dict[str, Any]] = {
        h: {"learning": {}, "frozen": {}, "logs": {}} for h in HORIZONS
    }
    for log in sorted(log_dir.glob("*.log")):
        match = _LABEL.match(log.name)
        if match is None:
            print(f"  WARN: skipping log with an unrecognised label: {log.name}")
            continue
        horizon = match.group("horizon")
        if horizon not in HORIZONS:
            print(f"  WARN: skipping log whose horizon is not registered: {log.name}")
            continue
        record = read_log(log, experiments)
        if record is None:
            print(f"  WARN: no parseable run lines in {log.name} - dropped")
            continue
        arm = "frozen" if match.group("frozen") else "learning"
        seed = int(match.group("seed"))
        if seed in out[horizon][arm]:
            # Two logs for one cell: whichever was read second would silently replace the first,
            # and the campaign would score as if one run had happened.
            msg = (
                f"duplicate run for {horizon}/{arm} seed {seed}: {log.name} and an earlier log "
                "both claim this cell"
            )
            raise ValueError(msg)
        out[horizon][arm][seed] = record
        out[horizon]["logs"].setdefault(arm, {})[seed] = log
    return out


def require_complete(scanned: dict[str, dict[str, Any]]) -> None:
    """Refuse to score a campaign that is missing any registered cell.

    A dropped or unfinished run would otherwise shrink a horizon's pairing silently, and the
    registered verdict -- including ``does_not_transfer`` -- would be assigned on partial evidence
    while looking exactly like a complete result.
    """
    missing: list[str] = []
    for horizon in HORIZONS:
        for arm in ("learning", "frozen"):
            absent = [s for s in SEEDS if s not in scanned[horizon][arm]]
            if absent:
                missing.append(f"{horizon}/{arm} seeds {absent}")
    if missing:
        msg = "campaign is incomplete, so no verdict is available: " + "; ".join(missing)
        raise ValueError(msg)


def _weights(log: Path, experiments: Path = EXPERIMENTS) -> np.ndarray | None:
    """Return the run's final plastic weights, flattened, or None where no export is on disk."""
    import torch

    experiment = _experiment_json(log.read_text(), experiments)
    exports = experiment.get("exports_path") if experiment else None
    if not exports:
        return None
    final = REPO / exports / "weights" / "final.pt"
    if not final.is_file():
        return None
    state = torch.load(final, weights_only=True)
    policy = state.get("policy", state)
    tensors = [v.reshape(-1) for v in policy.values() if hasattr(v, "reshape")]
    return torch.cat(tensors).numpy() if tensors else None


def drift(logs: dict[str, dict[int, Path]], experiments: Path = EXPERIMENTS) -> dict[str, Any]:
    """Measure each seed's learning weights against its own frozen control's.

    A horizon too short leaves the policy near its control, drifting little because almost nothing
    is credited; one too long moves the weights substantially in a direction unrelated to reward.
    Without this the two look the same in the score.
    """
    per_seed: dict[str, float] = {}
    for seed in SEEDS:
        learning = logs.get("learning", {}).get(seed)
        frozen = logs.get("frozen", {}).get(seed)
        if learning is None or frozen is None:
            continue
        a, b = _weights(learning, experiments), _weights(frozen, experiments)
        if a is None or b is None or a.shape != b.shape:
            continue
        per_seed[str(seed)] = float(np.linalg.norm(a - b) / (np.linalg.norm(b) or 1.0))
    values = list(per_seed.values())
    return {
        "per_seed": per_seed,
        "mean_relative": float(np.mean(values)) if values else float("nan"),
        "n_read": len(values),
    }


def _full_clear(
    learning: dict[int, float],
    frozen: dict[int, float],
) -> dict[str, Any]:
    """Report the primary metric and say, from the data, why the level contrast is unavailable.

    The level contrast needs a competent seed in **both** arms. Which arm lacks one is the
    informative part and is derived here rather than asserted, so the record cannot claim there is
    no competent seed while reporting that there is.
    """
    competent = {
        arm: sorted(s for s, v in values.items() if v >= ms.COMPETENT_THRESHOLD)
        for arm, values in (("learning", learning), ("frozen", frozen))
    }
    empty = [arm for arm, seeds in competent.items() if not seeds]
    if not empty:
        note = "both arms have a competent seed, so the level contrast is available"
    elif len(empty) == 2:
        note = (
            "neither arm has a seed at or above the committed competence threshold, so the "
            "level contrast is undefined"
        )
    else:
        note = (
            f"the {empty[0]} arm has no seed at or above the committed competence threshold "
            f"(the {'frozen' if empty[0] == 'learning' else 'learning'} arm has "
            f"{competent['frozen' if empty[0] == 'learning' else 'learning']}), so the level "
            "contrast is undefined"
        )
    return {
        "threshold": ms.COMPETENT_THRESHOLD,
        "learning_mean": float(np.mean(list(learning.values()) or [math.nan])),
        "frozen_mean": float(np.mean(list(frozen.values()) or [math.nan])),
        "competent_seeds": dict(competent),
        "level_contrast_available": not empty,
        "note": note,
    }


def compare(horizon: dict[str, Any], experiments: Path = EXPERIMENTS) -> dict[str, Any]:
    """Score one horizon's learning arm against its own frozen control."""
    learning_foods = {s: r.foods for s, r in horizon["learning"].items()}
    frozen_foods = {s: r.foods for s, r in horizon["frozen"].items()}
    learning_clear = {s: r.success for s, r in horizon["learning"].items()}
    frozen_clear = {s: r.success for s, r in horizon["frozen"].items()}
    graded = ms.shift_contrast(learning_foods, frozen_foods)
    return {
        "n_pairs": len(set(learning_foods) & set(frozen_foods)),
        "learning_foods": learning_foods,
        "frozen_foods": frozen_foods,
        "learning_mean_foods": float(np.mean(list(learning_foods.values()) or [math.nan])),
        "frozen_mean_foods": float(np.mean(list(frozen_foods.values()) or [math.nan])),
        "graded": graded,
        # Reported so the floor is visible rather than assumed; the competence-dependent contrasts
        # are undefined here and say so rather than returning a null.
        "full_clear": _full_clear(learning_clear, frozen_clear),
        "drift": drift(horizon["logs"], experiments),
    }


def analyse(scanned: dict[str, dict[str, Any]], experiments: Path = EXPERIMENTS) -> dict[str, Any]:
    """Compare every horizon, correct across them, and apply the registered rule."""
    cells = {h: compare(scanned[h], experiments) for h in HORIZONS}
    defined = [h for h in HORIZONS if cells[h]["graded"].get("defined")]
    qs = ms.bh_fdr([cells[h]["graded"]["p_improve"] for h in defined])
    for horizon, q in zip(defined, qs, strict=True):
        cells[horizon]["graded"]["q_improve"] = float(q)
    for horizon in HORIZONS:
        cell = cells[horizon]
        q = cell["graded"].get("q_improve", float("nan"))
        effect = cell["graded"]["effect"]
        significant = bool(q <= ms.SIG_Q) if not math.isnan(q) else False
        large_enough = effect >= MIN_EFFECT_FOODS
        cell["beats_control"] = significant and large_enough
        cell["why"] = (
            "significant and at or above the registered minimum"
            if cell["beats_control"]
            else (
                f"significant but below the {MIN_EFFECT_FOODS} foods minimum"
                if significant
                else "not significant"
            )
        )
    return {
        "min_effect_foods": MIN_EFFECT_FOODS,
        "alpha": ms.SIG_Q,
        "committed_yardstick_foods": COMMITTED_YARDSTICK_FOODS,
        "committed_is_reference_only": (
            "Logbook 040 ran the original eligibility; contrasting against it would confound the "
            "eligibility change with the horizon change"
        ),
        "horizons": {str(HORIZONS[h]): cells[h] for h in HORIZONS},
        "verdict": _verdict(cells),
    }


def _verdict(cells: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Apply the registered outcome map."""
    beats = {h for h in HORIZONS if cells[h]["beats_control"]}
    if beats == set():
        return {
            "verdict": "does_not_transfer",
            "licenses": "nothing; I.4 records that the horizon was tested on a multi-step task "
            "and did not carry off the control",
        }
    if "td09" in beats:
        return {
            "verdict": "void",
            "licenses": "nothing: the pinned horizon beats its control here and did not in the "
            "I.3 pilot, so something other than the horizon changed and must be found first",
        }
    return {
        "verdict": "transfers",
        "licenses": "a connectome arm at the raised horizon, and I.4 reading the negative results "
        "as findings about an instrument run at a crippling setting",
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


def _print(out: dict[str, Any]) -> None:
    """Print the comparison, one line per horizon."""
    print("\nThe eligibility horizon on a multi-step task (MLP yardstick, 8 seeds, 3000 episodes)")
    print(
        f"  each horizon against its OWN frozen control, on plateau-tail mean foods; "
        f"a shift counts only if significant AND >= {out['min_effect_foods']} foods\n",
    )
    print("  trace_decay | learning | frozen | shift  |     q | drift | beats control")
    for key, cell in out["horizons"].items():
        graded = cell["graded"]
        q = graded.get("q_improve", float("nan"))
        drift_mean = cell["drift"]["mean_relative"]
        print(
            f"  {float(key):11} | {cell['learning_mean_foods']:8.3f} | "
            f"{cell['frozen_mean_foods']:6.3f} | {graded['effect']:+6.3f} | "
            f"{q:5.3f} | {drift_mean:5.2f} | "
            f"{'YES' if cell['beats_control'] else 'no'} ({cell['why']})",
        )
    first = next(iter(out["horizons"].values()))
    clear = first["full_clear"]
    print(
        f"\n  full-clear metric: learning {clear['learning_mean']:.2f}%, "
        f"frozen {clear['frozen_mean']:.2f}%, competent seeds {clear['competent_seeds']}"
        f"\n    {clear['note']}",
    )
    print(
        f"  committed yardstick reference (original rule): {out['committed_yardstick_foods']:.2f} foods",
    )
    print(f"\n  VERDICT: {out['verdict']['verdict']} -- licenses {out['verdict']['licenses']}")


def main(argv: list[str] | None = None) -> int:
    """Read the campaign, apply the registered rule and write the records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--experiments", type=Path, default=EXPERIMENTS)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args(argv)

    scanned = scan(args.campaign_dir, args.experiments)
    require_complete(scanned)
    out = analyse(scanned, args.experiments)
    _print(out)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(_jsonable(out), indent=2, sort_keys=True, allow_nan=False) + "\n",
        )
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["trace_decay", "seed", "learning_foods", "frozen_foods", "delta"])
            for key, cell in out["horizons"].items():
                for seed in sorted(cell["learning_foods"]):
                    learning = cell["learning_foods"][seed]
                    frozen = cell["frozen_foods"].get(seed)
                    writer.writerow(
                        [
                            key,
                            seed,
                            f"{learning:.4f}",
                            f"{frozen:.4f}" if frozen is not None else "",
                            f"{learning - frozen:.4f}" if frozen is not None else "",
                        ],
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
