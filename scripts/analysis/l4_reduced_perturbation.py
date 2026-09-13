"""The connectome's perturbation dimension, and whether restricting credit lets the rule learn.

The readout mean-pools only the 39 VB/DB/VA/DA motor neurons, and the eligibility is
``pre (x) perturbation``, so a unit's perturbation writes eligibility on every synapse onto it whether
or not it can reach the readout. At settling step ``s`` it can reach it only from within ``depth - s``
hops. On the Cook 2019 graph at depth 4 that leaves 672 of 1208 draws per decision causally connected:
**536 cannot change the action at all**, and every one of them is credited.

Five declared sets, each with its own frozen control at the same sigma and the same mask, so the
perturbation's cost to the policy is matched across every pair and the contrast measures the update
alone.

Three things about the reading, all fixed before the campaign:

* **Beating a floor is not learning the cell.** The registered ``dimension_reducible`` verdict asks
  whether an arm beats its frozen control by the minima; R.1b needs a rule that reaches COMPETENCE, so
  that a time-to-competence contrast is defined at all. Those are reported separately, because the
  first can hold while the second fails and only the second opens 7b's gate.
* **The relative minimum's reference is the arm's own frozen control**, not block V's 3.82-food figure:
  that arm runs at a different action noise, so the gap it implies is not this campaign's. Both are
  reported and the more demanding one binds.
* **Drift is split between excluded and credited synapses.** A restricted set leaves the rule's
  unconditional weight decay as the only update its excluded synapses receive, cancelled radially by
  the homeostatic rescale. The bench measurement says that leaves jitter at a conserved norm; this
  checks it on the real substrate at the real scale.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import l4_mixture_statistic as ms  # noqa: E402  # pyright: ignore[reportMissingImports]
import l4_perturbation_scale as ps  # noqa: E402  # pyright: ignore[reportMissingImports]
from l4_panel import EXPERIMENTS, read_log  # noqa: E402  # pyright: ignore[reportMissingImports]

SEEDS = tuple(range(1, 9))
_STEM = "connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_hard350_nodepert"

# The declared sets, in descending perturbation dimension. The numbers travel with the result because
# a dimension claim whose figures have to be reconstructed afterwards is what R.1 had to do.
ARMS: dict[str, dict[str, int]] = {
    "full": {"units": 302, "adaptable_synapses": 3709, "draws": 1208, "causal_draws": 672},
    "causal": {"units": 277, "adaptable_synapses": 3538, "draws": 672, "causal_draws": 672},
    "hop1": {"units": 109, "adaptable_synapses": 1476, "draws": 436, "causal_draws": 366},
    "motor": {"units": 39, "adaptable_synapses": 323, "draws": 156, "causal_draws": 156},
    "motor_last": {"units": 39, "adaptable_synapses": 323, "draws": 39, "causal_draws": 39},
}

# Block V ran PPO on this exact cell and substrate over 32 seeds, so no PPO arm is re-run.
PPO_REFERENCE_FOODS = 19.31
# That reference's frozen arm does NOT perturb and runs at the default action noise, so it is a
# reference and not the comparator. Kept to report the registered minimum alongside the matched one.
REFERENCE_FROZEN_FOODS = 3.82
TARGET_FOODS = 20.0
MIN_FOODS = 1.0
MIN_GAP_FRACTION = 0.10
# Competence is the committed threshold, on full-clear percentage: what R.1b's time-to-competence
# contrast needs to exist at all.
COMPETENCE_THRESHOLD = ms.COMPETENT_THRESHOLD

_LABEL = re.compile(
    rf"^{re.escape(_STEM)}_(?P<arm>full|causal|hop1|motor_last|motor)(?P<frozen>_frozen)?"
    r"-seed(?P<seed>\d+)\.log$",
)


def scan(campaign_dir: Path, experiments: Path = EXPERIMENTS) -> dict[str, dict[str, Any]]:
    """Read every registered run under a campaign directory, keyed by declared set and arm."""
    log_dir = campaign_dir / "logs" if (campaign_dir / "logs").is_dir() else campaign_dir
    out: dict[str, dict[str, Any]] = {
        name: {"learning": {}, "frozen": {}, "logs": {}} for name in ARMS
    }
    for log in sorted(log_dir.glob("*.log")):
        match = _LABEL.match(log.name)
        if match is None:
            print(f"  WARN: skipping log with an unrecognised label: {log.name}")
            continue
        name = match.group("arm")
        if name not in out:
            print(f"  WARN: skipping log whose set is not registered: {log.name}")
            continue
        record = read_log(log, experiments)
        if record is None:
            print(f"  WARN: no parseable run lines in {log.name} - dropped")
            continue
        arm = "frozen" if match.group("frozen") else "learning"
        seed = int(match.group("seed"))
        if seed in out[name][arm]:
            # Two logs for one cell: whichever was read second would silently replace the first and
            # the campaign would score as if one run had happened.
            msg = f"two logs for set {name} arm {arm} seed {seed}: {log.name} duplicates a read run"
            raise ValueError(msg)
        out[name][arm][seed] = record
        out[name]["logs"].setdefault(arm, {})[seed] = log
    return out


def require_complete(scanned: dict[str, dict[str, Any]], seeds: tuple[int, ...] = SEEDS) -> None:
    """Refuse to score a campaign missing any registered cell.

    A dropped run would otherwise shrink a set's pairing silently and the registered verdict --
    ``not_reducible`` included -- would be assigned on partial evidence while looking complete.
    """
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
    """Apply both registered effect minima and name which one failed.

    The relative minimum is ten per cent of the reachable gap. The gap is taken against this arm's
    OWN frozen mean, because block V's frozen figure runs at a different action noise and implies a
    gap this campaign does not have; the registered figure is reported beside it and the more
    demanding of the two binds.
    """
    matched_gap = PPO_REFERENCE_FOODS - frozen_mean
    registered_gap = PPO_REFERENCE_FOODS - REFERENCE_FROZEN_FOODS
    matched = MIN_GAP_FRACTION * matched_gap
    registered = MIN_GAP_FRACTION * registered_gap
    binding = max(matched, registered)
    absolute_ok = bool(effect >= MIN_FOODS)
    relative_ok = bool(effect >= binding)
    failed = [
        label
        for label, ok in (
            (f"the {MIN_FOODS} foods minimum", absolute_ok),
            (f"the {MIN_GAP_FRACTION:.0%}-of-gap minimum ({binding:.2f} foods)", relative_ok),
        )
        if not ok
    ]
    return {
        "effect": effect,
        "absolute_minimum": MIN_FOODS,
        "matched_gap": matched_gap,
        "matched_minimum": matched,
        "registered_minimum": registered,
        "binding_minimum": binding,
        "passes": not failed,
        "why": "at or above both minima" if not failed else f"below {' and '.join(failed)}",
    }


def _chemical_weights(log: Path, experiments: Path = EXPERIMENTS) -> torch.Tensor | None:
    """Read the run's final chemical synapse matrix, or None where no export is on disk.

    Deliberately NOT the horizon harness's reader: that one takes ``state["policy"]`` and flattens
    every tensor under it, which is the MLP checkpoint's layout. A connectome checkpoint keeps its
    tensors under ``state["topology"]`` and has no ``policy`` key at all, so that reader finds
    nothing and reports drift unavailable for every run -- a silent empty column rather than a wrong
    one, but useless either way.
    """
    import torch
    from l4_panel import _experiment_json

    experiment = _experiment_json(log.read_text(), experiments)
    exports = experiment.get("exports_path") if experiment else None
    if not exports:
        return None
    final = ps.hm.REPO / exports / "weights" / "final.pt"
    if not final.is_file():
        return None
    state = torch.load(final, weights_only=True)
    topology = state.get("topology")
    if not isinstance(topology, dict):
        return None
    return topology.get("w_chem")


def _perturbed_units(name: str) -> torch.Tensor:
    """Return which units the declared set ever perturbs, as a boolean over the neurons."""
    import torch
    from quantumnematode.brain.arch.connectome_ppo import (
        ConnectomeTopology,
    )
    from quantumnematode.connectome.loader import (
        load_cook_2019_hermaphrodite,
    )

    topology = ConnectomeTopology(
        load_cook_2019_hermaphrodite(),
        enable_gap_junctions=True,
        forward_pass_depth=4,
        node_noise=0.1,
        perturbation_set=name,  # pyright: ignore[reportArgumentType]
        n_food_features=3,
        enforce_strict_mask=True,
        enable_predator_projection=False,
        enable_thermotaxis_projection=False,
        device=torch.device("cpu"),
        rng=np.random.default_rng(0),
        continuous=True,
        enable_activity_traces=True,
        trace_decay=0.9,
    )
    return topology._perturbation_mask.any(dim=0)


def split_drift(
    name: str,
    logs: dict[str, dict[int, Path]],
    seeds: tuple[int, ...] = SEEDS,
    experiments: Path = EXPERIMENTS,
) -> dict[str, Any]:
    """Drift from this set's own frozen control, split by whether a synapse is credited.

    ``E[i, j] = h_prev[i] * perturbation[j]``, so a synapse onto an unperturbed unit never receives
    eligibility and the rule's unconditional weight decay is the ONLY update it gets -- cancelled
    radially by the homeostatic rescale. The bench measurement says that leaves jitter at a conserved
    norm; splitting the drift checks it here, on the real substrate at the real scale, which is the
    whole reason it was registered.

    The seed set is a parameter: inheriting it from a module constant is what made R.1's drift read
    nothing on a pilot's disjoint seeds.
    """
    perturbed = _perturbed_units(name)
    credited: list[float] = []
    excluded: list[float] = []
    for seed in seeds:
        learning = logs.get("learning", {}).get(seed)
        frozen = logs.get("frozen", {}).get(seed)
        if learning is None or frozen is None:
            continue
        a = _chemical_weights(learning, experiments)
        b = _chemical_weights(frozen, experiments)
        if a is None or b is None or a.shape != b.shape:
            continue
        for label, columns, sink in (
            ("credited", perturbed, credited),
            ("excluded", ~perturbed, excluded),
        ):
            del label
            if not bool(columns.any()):
                continue
            delta = (a[:, columns] - b[:, columns]).float()
            base = b[:, columns].float().norm()
            sink.append(float(delta.norm() / (base or 1.0)))
    return {
        "credited_mean_relative": float(np.mean(credited)) if credited else float("nan"),
        "excluded_mean_relative": float(np.mean(excluded)) if excluded else float("nan"),
        "n_read": len(credited),
        "available": bool(credited),
        "seeds_expected": list(seeds),
    }


def compare(
    name: str,
    data: dict[str, Any],
    seeds: tuple[int, ...] = SEEDS,
    experiments: Path = EXPERIMENTS,
) -> dict[str, Any]:
    """Score one declared set's learning arm against its own frozen control.

    Restricted to the REQUESTED seeds. ``require_complete`` only checks that those seeds are present,
    so a campaign directory carrying extra runs -- a re-run, a pilot's seeds swept into the same
    folder -- would otherwise have them folded into the contrast, the means, the competence check and
    the verdict, none of which the registration asked for. The completeness guard and the drift
    measurement already key on the requested seeds; this makes the score agree with them.
    """
    learning = {s: r.foods for s, r in data["learning"].items() if s in seeds}
    frozen = {s: r.foods for s, r in data["frozen"].items() if s in seeds}
    learning_clear = [r.success for s, r in data["learning"].items() if s in seeds]
    frozen_mean = float(np.mean(list(frozen.values()) or [math.nan]))
    graded = ms.shift_contrast(learning, frozen)
    mean_clear = float(np.mean(learning_clear)) if learning_clear else float("nan")
    return {
        "n_pairs": len(set(learning) & set(frozen)),
        "learning_mean_foods": float(np.mean(list(learning.values()) or [math.nan])),
        "frozen_mean_foods": frozen_mean,
        "learning_foods": learning,
        "frozen_foods": frozen,
        "learning_mean_full_clear": mean_clear,
        # What R.1b needs, and a different question from beating a floor: a time-to-competence
        # contrast is undefined for an arm that never becomes competent.
        "reaches_competence": bool(mean_clear >= COMPETENCE_THRESHOLD),
        "competence_threshold": COMPETENCE_THRESHOLD,
        "graded": graded,
        "minima": minima(graded.get("effect", float("nan")), frozen_mean),
        "drift": split_drift(name, data["logs"], seeds, experiments),
    }


def analyse(
    scanned: dict[str, dict[str, Any]],
    seeds: tuple[int, ...] = SEEDS,
    experiments: Path = EXPERIMENTS,
) -> dict[str, Any]:
    """Compare every declared set, correct across them and apply the registered rule."""
    cells = {name: compare(name, scanned[name], seeds, experiments) for name in ARMS}
    defined = [n for n in ARMS if cells[n]["graded"].get("defined")]
    qs = ms.bh_fdr([cells[n]["graded"]["p_improve"] for n in defined])
    for name, q in zip(defined, qs, strict=True):
        cells[name]["graded"]["q_improve"] = float(q)
    for name, dimension in ARMS.items():
        cell = cells[name]
        cell["dimension"] = dict(dimension)
        q = cell["graded"].get("q_improve", float("nan"))
        significant = bool(q <= ms.SIG_Q) if not math.isnan(q) else False
        if significant and cell["minima"]["passes"]:
            cell["verdict"] = "beats_floor"
        elif significant:
            cell["verdict"] = "below_min_effect"
        else:
            cell["verdict"] = "no_improvement"
        cell["why"] = {
            "beats_floor": "significant and at or above both minima",
            "below_min_effect": f"significant but {cell['minima']['why']}",
            "no_improvement": "not significant",
        }[cell["verdict"]]
    return {
        "sets": cells,
        "beat_floor": [n for n in ARMS if cells[n]["verdict"] == "beats_floor"],
        "competent": [n for n in ARMS if cells[n]["reaches_competence"]],
        "trend": _trend(cells),
        **_verdict(cells),
    }


def _trend(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Spearman of the contrast against draws per decision. Descriptive: five sets has no power."""
    usable = [n for n in ARMS if cells[n]["graded"].get("defined")]
    if len(usable) < 3:
        return {"defined": False, "reason": f"only {len(usable)} sets with a defined contrast"}
    draws = np.array([float(ARMS[n]["draws"]) for n in usable])
    effects = np.array([float(cells[n]["graded"]["effect"]) for n in usable])
    if float(np.ptp(effects)) == 0.0:
        return {
            "defined": False,
            "reason": "the contrast is identical at every set",
            "constant_at": float(effects[0]),
        }
    from scipy import stats

    result: Any = stats.spearmanr(draws, effects)
    return {
        "defined": True,
        "sets": usable,
        "rho": float(result.statistic),
        "p": float(result.pvalue),
        # Fewer draws should help, so the prediction is a negative correlation with draws.
        "predicted_sign": "negative",
        "in_predicted_direction": bool(result.statistic < 0),
        "descriptive_only": True,
    }


def _verdict(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Apply the three registered outcomes, keeping the floor and competence questions apart."""
    beat = [n for n in ARMS if cells[n]["verdict"] == "beats_floor"]
    competent = [n for n in ARMS if cells[n]["reaches_competence"]]
    reduced_beat = [n for n in beat if n != "full"]
    if "causal" in beat and not [n for n in reduced_beat if n != "causal"]:
        verdict, why = (
            "causal_mask_sufficient",
            (
                "the causal mask alone beats its own frozen control: the failure was exploration "
                "credited against an outcome it could not influence, and nothing had to be given up"
            ),
        )
    elif reduced_beat:
        verdict, why = (
            "dimension_reducible",
            (
                f"a reduced set beats its own frozen control ({', '.join(reduced_beat)}): the "
                "dimension binds on this substrate as it does on the yardstick"
            ),
        )
    else:
        verdict, why = (
            "not_reducible",
            (
                "no set beats its own frozen control by the registered minima: the connectome's "
                "failure is not the perturbation dimension"
            ),
        )
    # Registered separately, because the verdict above is a weaker condition than its own stated
    # consequence: R.1b needs a rule that reaches competence, not one that beats a damaged floor.
    unblocks = bool(competent)
    return {
        "verdict": verdict,
        "why": why,
        "sets_beating_floor": beat,
        "sets_reaching_competence": competent,
        "r1b_unblocked": unblocks,
        "r1b_note": (
            f"R.1b is unblocked at {', '.join(competent)}: a time-to-competence contrast is defined"
            if unblocks
            else "R.1b stays blocked: no set reaches competence, so block V's time-to-competence "
            "contrast is undefined for every arm here, whatever the floor contrasts say"
        ),
    }


def _print(result: dict[str, Any]) -> None:
    print("\nR.1c — the connectome's perturbation dimension")
    print(
        "  set        | units | synapses | draws | learning | frozen | shift |     q "
        "| clear % | drift-cr | drift-ex | verdict",
    )
    for name in ARMS:
        cell = result["sets"][name]
        dim = cell["dimension"]
        q = cell["graded"].get("q_improve", float("nan"))
        print(
            f"  {name:<10} | {dim['units']:>5} | {dim['adaptable_synapses']:>8} | "
            f"{dim['draws']:>5} | {cell['learning_mean_foods']:>8.3f} | "
            f"{cell['frozen_mean_foods']:>6.3f} | {cell['graded'].get('effect', float('nan')):>+5.2f} "
            f"| {q:5.3f} | {cell['learning_mean_full_clear']:>7.2f} | "
            f"{cell['drift']['credited_mean_relative']:>5.2f} | "
            f"{cell['drift']['excluded_mean_relative']:>5.2f} | {cell['verdict']}",
        )
        print(f"        {cell['why']}")
    trend = result["trend"]
    if trend.get("defined"):
        print(f"  trend (descriptive): rho {trend['rho']:+.3f} p {trend['p']:.3f}")
    print(f"\n  sets beating their floor: {result['sets_beating_floor'] or 'none'}")
    print(f"  sets reaching competence: {result['sets_reaching_competence'] or 'none'}")
    print(f"\nVERDICT: {result['verdict']} — {result['why']}")
    print(f"R.1b: {result['r1b_note']}")


def write_csv(result: dict[str, Any], path: Path) -> None:
    """One row per set and seed, both arms side by side."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "set",
                "units",
                "draws_per_decision",
                "seed",
                "learning_foods",
                "frozen_foods",
                "favours_learning",
            ],
        )
        for name in ARMS:
            cell = result["sets"][name]
            learning, frozen = cell["learning_foods"], cell["frozen_foods"]
            for seed in sorted(set(learning) & set(frozen)):
                writer.writerow(
                    [
                        name,
                        ARMS[name]["units"],
                        ARMS[name]["draws"],
                        seed,
                        f"{learning[seed]:.4f}",
                        f"{frozen[seed]:.4f}",
                        int(learning[seed] > frozen[seed]),
                    ],
                )


def main(argv: list[str] | None = None) -> int:
    """Score a campaign directory and write its records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--seeds", type=str, default="1-8")
    parser.add_argument("--experiments", type=Path, default=EXPERIMENTS)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="score without the completeness guard (for a pilot, never for the campaign)",
    )
    args = parser.parse_args(argv)

    low, _, high = args.seeds.partition("-")
    seeds = tuple(range(int(low), int(high or low) + 1))
    scanned = scan(args.campaign, args.experiments)
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
                # The control's NaN-to-null pass, reused so the record is strict JSON.
                ps.pc._jsonable(
                    {
                        "protocol": {
                            "seeds": list(seeds),
                            "arms": ARMS,
                            "ppo_reference_foods": PPO_REFERENCE_FOODS,
                            "reference_frozen_foods": REFERENCE_FROZEN_FOODS,
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
