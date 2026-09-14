#!/usr/bin/env python
"""R.2's reading: does an eligibility built from the dynamics let the rule learn the cell.

Node perturbation is closed. R.1 found the rule solving a multi-step cell at 8 perturbed units and
collapsing at 128; R.1c took the connectome from 1208 draws per decision to 39 and returned
``not_reducible``; R.1d showed the frozen readout is part of the limit without any arm reaching
competence. Across all three, credited-synapse drift stayed at 1.37-1.42x the weight's own norm. The
rule is never starved of signal -- it writes a great deal in a direction that does not help -- so the
remaining suspect is what the eligibility carries.

e-prop replaces the perturbation with the network's own settling derivative and the draw's sign with
a broadcast learning signal. **That signal is the part that is not a drop-in**, and on this substrate
it decides which units can be credited at all: the readout mean-pools 39 of 302 units, and e-prop
drops the multi-hop paths by which the rest reach the action, so a symmetric (true-gradient) signal
reaches exactly the pooled 39 and nothing else. The routing therefore varies the signal's source AND
its reach together, and the campaign crosses them:

===============  =======================  ==========================================
arm              signal source            units it can reach
===============  =======================  ==========================================
``symmetric``    the readout's transpose  the 39-unit pool -- FORCED, not chosen
``random_motor`` a fixed random draw      the same 39, masked to match
``random``       a fixed random draw      all 302
``scalar``       none, ``L_j = 1``        all 302
===============  =======================  ==========================================

The fourth cell of that 2x2 -- true directions reaching every unit -- is one the mechanism forbids,
and this harness reports it as forbidden rather than omitting it.

Three things about the reading, fixed before the campaign:

* **Beating a floor is not learning the cell.** R.1b needs a rule that reaches COMPETENCE, so that a
  time-to-competence contrast is defined at all. Reported separately, because the first can hold
  while the second fails and only the second opens 7b's gate.
* **Stage 1 is a stop clause, not a task.** The one-step control's ``eprop_symmetric`` arm is the
  exact REINFORCE gradient of its plastic layer and MUST pass; its ``eprop_scalar`` arm MUST NOT. If
  either expectation was violated, nothing here is interpretable and the reading is ``void``. This
  harness reads that control's own record rather than taking it on trust.
* **Two matched contrasts carry the interpretation.** ``symmetric - random_motor`` isolates the
  signal's direction at matched reach; ``random - random_motor`` isolates reach at a matched source.
  Neither decides the verdict, and both are reported in every branch.
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
import l4_reduced_perturbation as rp  # noqa: E402  # pyright: ignore[reportMissingImports]
from l4_panel import EXPERIMENTS, read_log  # noqa: E402  # pyright: ignore[reportMissingImports]

# Sixteen, which is what Logbook 059 registered for this gate -- not R.1c's and R.1d's eight.
SEEDS = tuple(range(1, 17))
_STEM = "connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_hard350_eprop"

# The learning arms, in the order the table reads: reach descending, then the ablation.
ARMS: dict[str, dict[str, Any]] = {
    "plastic_readout": {
        "routing": "random",
        "source": "random projection",
        "reach": "all 302",
        "readout": "plastic",
        "chemical": "plastic",
    },
    "readout_only": {
        "routing": "random",
        "source": "random projection",
        "reach": "all 302",
        "readout": "plastic",
        "chemical": "frozen",
    },
    "random": {
        "routing": "random",
        "source": "random projection",
        "reach": "all 302",
        "readout": "frozen",
        "chemical": "plastic",
    },
    "scalar": {
        "routing": "scalar",
        "source": "none (L = 1)",
        "reach": "all 302",
        "readout": "frozen",
        "chemical": "plastic",
    },
    "symmetric": {
        "routing": "symmetric",
        "source": "readout transpose",
        "reach": "the 39-unit pool",
        "readout": "frozen",
        "chemical": "plastic",
    },
    "random_motor": {
        "routing": "random_motor",
        "source": "random projection",
        "reach": "the 39-unit pool",
        "readout": "frozen",
        "chemical": "plastic",
    },
}
# Added after stage 1, before any arm ran, and the reason is measured rather than argued: feedback
# alignment needs the forward path to the output to come into alignment with the feedback matrix, and
# a FROZEN readout cannot. On the one-step control at 20,000 trials the broadcast arm's best is
# -0.7031 against a cue-blind floor of -0.6909 with the readout frozen, and reaches the optimum of
# -0.1353 exactly on every seed with it plastic. So `random` -- the arm the plausibility claim rests
# on -- is structurally unable to work in the four registered arms' configuration, and this is the
# one arm that can. It is also the only arm whose readout eligibility is EXACT: the readout's
# post-synaptic units are the action dimensions, so its signal is the identity and no path is
# dropped.
PLASTIC_READOUT_ARM = "plastic_readout"
# Its control, and the arm every substrate claim is measured against.
READOUT_ONLY_ARM = "readout_only"
# The cell the mechanism forbids, stated rather than left out: under e-prop's truncation a
# true-gradient signal cannot reach a unit the readout does not read.
FORBIDDEN_CELL = {
    "source": "readout transpose",
    "reach": "all 302",
    "why": (
        "the readout reads only the pooled motor classes, and e-prop drops the multi-hop paths by "
        "which any other unit reaches the action, so a true-gradient signal is identically zero "
        "outside the pool"
    ),
}
# The two contrasts that separate what the routing varies. Neither decides the verdict.
MATCHED_CONTRASTS = {
    "direction_at_matched_reach": ("symmetric", "random_motor"),
    "reach_at_matched_source": ("random", "random_motor"),
    # The third, and the one stage 1 predicts will be the largest: the same projection over the same
    # 302 units, differing only in whether the readout may learn.
    "readout_at_matched_signal": ("plastic_readout", "random"),
    # The fourth, and the one a positive plastic-readout result needs: the readout learns in both,
    # and only one of them may also write the substrate. A plastic readout is an 8-parameter linear
    # map over four pooled motor-class means, so "a local rule learns this substrate" and "a small
    # linear readout on frozen recurrent features learns this cell" predict the same success --
    # this difference IS what the substrate's own plasticity contributes.
    "wiring_at_matched_readout": ("plastic_readout", "readout_only"),
}

# Harvested on this exact cell, at these seeds and this action scale, by R.1d.
PPO_MATCHED_FOODS = 18.945
# R.1c's committed node-perturbation pair on this same cell: the mechanism this one replaces, and
# its `motor` set -- R.1c's best arm at +0.601 on 7 of 8 seeds. That set is also `symmetric`'s
# structural comparator: both credit the 39 pooled units and nothing else.
NODEPERT_LEARNING_FOODS = 3.751
NODEPERT_FROZEN_FOODS = 3.150
NODEPERT_SHIFT_FOODS = 0.601
# R.1d's two measured readouts, for the plastic arm to be read against: the committed anatomical
# default, and the PPO harvest that replaced rather than refined it.
ANATOMICAL_READOUT_NORM = 1.414
PPO_READOUT_NORM = 7.820
PPO_READOUT_COSINE = -0.178
TARGET_FOODS = 20.0
MIN_FOODS = 1.0
MIN_GAP_FRACTION = 0.10
COMPETENCE_THRESHOLD = ms.COMPETENT_THRESHOLD
# R.1c measured 1.37-1.38x across every perturbation dimension and R.1d 1.38-1.42x across four
# readouts. A third structural axis holding the same number is the finding; a different one is a
# bigger one.
DRIFT_REFERENCE = (1.37, 1.42)

_LABEL = re.compile(
    rf"^{re.escape(_STEM)}_(?P<arm>plastic_readout|readout_only|symmetric"
    r"|random_motor|random|scalar|frozen)"
    r"-seed(?P<seed>\d+)\.log$",
)


def scan(campaign_dir: Path, experiments: Path = EXPERIMENTS) -> dict[str, dict[str, Any]]:
    """Read every registered run under a campaign directory, keyed by arm.

    One frozen key, not one per arm: with updates frozen no weight moves, so no routing can reach
    the behaviour and a single floor serves every learning arm. That is asserted of the configs by
    test rather than argued here.
    """
    log_dir = campaign_dir / "logs" if (campaign_dir / "logs").is_dir() else campaign_dir
    out: dict[str, dict[int, Any]] = {name: {} for name in ARMS}
    out["frozen"] = {}
    logs: dict[str, dict[int, Path]] = {}
    for log in sorted(log_dir.glob("*.log")):
        match = _LABEL.match(log.name)
        if match is None:
            print(f"  WARN: skipping log with an unrecognised label: {log.name}")
            continue
        name = match.group("arm")
        record = read_log(log, experiments)
        if record is None:
            print(f"  WARN: no parseable run lines in {log.name} - dropped")
            continue
        seed = int(match.group("seed"))
        if seed in out[name]:
            # Two logs for one cell: whichever was read second would silently replace the first and
            # the campaign would score as if one run had happened.
            msg = f"two logs for arm {name} seed {seed}: {log.name} duplicates a read run"
            raise ValueError(msg)
        out[name][seed] = record
        logs.setdefault(name, {})[seed] = log
    return {"runs": out, "logs": logs}


def require_complete(scanned: dict[str, Any], seeds: tuple[int, ...] = SEEDS) -> None:
    """Refuse to score a campaign missing any registered cell.

    A dropped run would shrink an arm's pairing silently, and ``does_not_learn`` -- which stops the
    programme -- would be assigned on partial evidence while looking complete.
    """
    runs = scanned["runs"]
    missing = [
        f"{name} seeds {[s for s in seeds if s not in runs[name]]}"
        for name in (*ARMS, "frozen")
        if [s for s in seeds if s not in runs[name]]
    ]
    if missing:
        msg = "campaign is incomplete, so no verdict is available: " + "; ".join(missing)
        raise ValueError(msg)


def minima(effect: float, frozen_mean: float) -> dict[str, Any]:
    """Apply both registered effect minima and name which one failed.

    The relative minimum is ten per cent of the reachable gap, taken against PPO's MATCHED level --
    harvested on this cell at these seeds and this action scale -- and the absolute one is a food.
    The more demanding binds, as it did in R.1c and R.1d.
    """
    gap = PPO_MATCHED_FOODS - frozen_mean
    relative = MIN_GAP_FRACTION * gap
    binding = max(MIN_FOODS, relative)
    failed = [
        label
        for label, ok in (
            (f"the {MIN_FOODS} foods minimum", bool(effect >= MIN_FOODS)),
            (
                f"the {MIN_GAP_FRACTION:.0%}-of-gap minimum ({relative:.2f} foods)",
                bool(effect >= relative),
            ),
        )
        if not ok
    ]
    return {
        "effect": effect,
        "absolute_minimum": MIN_FOODS,
        "matched_gap": gap,
        "matched_minimum": relative,
        "binding_minimum": binding,
        "passes": not failed,
        "why": "at or above both minima" if not failed else f"below {' and '.join(failed)}",
    }


def _topology(routing: str) -> Any:  # noqa: ANN401 — the substrate's own topology type
    """Build this arm's topology, off the campaign, to read what its signal can reach."""
    import torch
    from quantumnematode.brain.arch.connectome_ppo import ConnectomeTopology
    from quantumnematode.connectome.loader import load_cook_2019_hermaphrodite

    return ConnectomeTopology(
        load_cook_2019_hermaphrodite(),
        enable_gap_junctions=True,
        forward_pass_depth=4,
        node_noise=0.0,
        eligibility="eprop",
        learning_signal=routing,  # pyright: ignore[reportArgumentType]
        learning_signal_seed=0,
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


def reached_units(routing: str) -> torch.Tensor:
    """Which units this routing's learning signal can reach, as a boolean over the neurons.

    The e-prop analogue of R.1c's perturbed set, and the same thing it measured drift against: a
    unit the signal cannot reach contributes a zero factor to every synapse ONTO it, so those
    synapses receive the rule's unconditional weight decay and nothing else.
    """
    import torch

    if routing == "scalar":
        # L_j = 1 everywhere, so every unit is reached.
        return torch.ones(302, dtype=torch.bool)
    projection = _topology(routing).learning_signal_projection()
    return projection.abs().sum(dim=1) != 0


def hop_distances() -> torch.Tensor:
    """Hops from each unit to the nearest readout neuron, over directed chemical edges."""
    topology = _topology("random")
    return topology._readout_hop_distances(topology._motor_flat_indices.tolist())


def split_drift(
    arm: str,
    logs: dict[str, dict[int, Path]],
    seeds: tuple[int, ...] = SEEDS,
    experiments: Path = EXPERIMENTS,
) -> dict[str, Any]:
    """Drift from the shared frozen control, split by whether the signal can reach the unit.

    Reuses R.1c's reader, which takes the connectome's ``state["topology"]`` layout rather than the
    MLP's ``state["policy"]`` -- the repair that made its split measurement possible at all.
    """
    pairs = [(logs.get(arm, {}).get(seed), logs.get("frozen", {}).get(seed)) for seed in seeds]
    usable = [(a, b) for a, b in pairs if a is not None and b is not None]
    if not usable:
        # Nothing on disk to split, so the 302-unit topology this would build to find the reached
        # set is not built: an unavailable column costs no work.
        return {
            "credited_mean_relative": float("nan"),
            "excluded_mean_relative": float("nan"),
            "reached_units": None,
            "n_read": 0,
            "available": False,
            "reference_range": list(DRIFT_REFERENCE),
            "seeds_expected": list(seeds),
        }
    reached = reached_units(ARMS[arm]["routing"])
    credited: list[float] = []
    excluded: list[float] = []
    for learning, frozen in usable:
        a = rp._chemical_weights(learning, experiments)
        b = rp._chemical_weights(frozen, experiments)
        if a is None or b is None or a.shape != b.shape:
            continue
        for columns, sink in ((reached, credited), (~reached, excluded)):
            if not bool(columns.any()):
                continue
            delta = (a[:, columns] - b[:, columns]).float()
            base = b[:, columns].float().norm()
            sink.append(float(delta.norm() / (base or 1.0)))
    return {
        "credited_mean_relative": float(np.mean(credited)) if credited else float("nan"),
        "excluded_mean_relative": float(np.mean(excluded)) if excluded else float("nan"),
        "reached_units": int(reached.sum()),
        "n_read": len(credited),
        "available": bool(credited),
        "reference_range": list(DRIFT_REFERENCE),
        "seeds_expected": list(seeds),
    }


def change_by_hop(
    arm: str,
    logs: dict[str, dict[int, Path]],
    seeds: tuple[int, ...] = SEEDS,
    experiments: Path = EXPERIMENTS,
) -> dict[str, Any]:
    """Where the weight change lands, by the post-synaptic unit's hop distance to the readout pool.

    The prediction e-prop's truncation makes, registered before the campaign: the dropped terms are
    the multi-hop ones, so a learning arm's change should concentrate near the pool. ``random`` is
    where it is testable -- it is the only arm whose signal reaches the far units at all.
    """
    pairs = [(logs.get(arm, {}).get(seed), logs.get("frozen", {}).get(seed)) for seed in seeds]
    usable = [(a, b) for a, b in pairs if a is not None and b is not None]
    if not usable:
        return {"mean_abs_change_by_hop": {}, "units_by_hop": {}, "n_read": 0, "available": False}
    distance = hop_distances()
    per_hop: dict[int, list[float]] = {}
    for learning, frozen in usable:
        a = rp._chemical_weights(learning, experiments)
        b = rp._chemical_weights(frozen, experiments)
        if a is None or b is None or a.shape != b.shape:
            continue
        delta = (a - b).float().abs()
        for hop in sorted({int(d) for d in distance.tolist()}):
            columns = distance == hop
            if not bool(columns.any()):
                continue
            per_hop.setdefault(hop, []).append(float(delta[:, columns].mean()))
    unreachable = 303  # the walk's sentinel: no directed chemical path to the pool
    return {
        "mean_abs_change_by_hop": {
            ("unreachable" if hop >= unreachable else str(hop)): float(np.mean(values))
            for hop, values in sorted(per_hop.items())
        },
        "units_by_hop": {
            ("unreachable" if hop >= unreachable else str(hop)): int((distance == hop).sum())
            for hop in sorted({int(d) for d in distance.tolist()})
        },
        "n_read": len(next(iter(per_hop.values()), [])),
        "available": bool(per_hop),
    }


def readout_change(
    arm: str,
    logs: dict[str, dict[int, Path]],
    seeds: tuple[int, ...] = SEEDS,
    experiments: Path = EXPERIMENTS,
) -> dict[str, Any]:
    """How far the readout moved, for the one arm that may move it.

    Reported against the two readouts already on the record: the anatomical default at norm 1.414,
    and the PPO harvest R.1d substituted at 7.820 with a cosine of -0.178 to it. Where the arm's
    readout ends up between those says whether e-prop rediscovers something like PPO's decoding or
    something else. Empty for every frozen-readout arm, whose readout cannot move.
    """
    import torch

    if ARMS[arm]["readout"] != "plastic":
        return {"available": False, "why": "this arm's readout is frozen"}
    norms: list[float] = []
    cosines: list[float] = []
    for seed in seeds:
        learning = logs.get(arm, {}).get(seed)
        frozen = logs.get("frozen", {}).get(seed)
        if learning is None or frozen is None:
            continue
        after = _readout_tensor(learning, experiments)
        before = _readout_tensor(frozen, experiments)
        if after is None or before is None or after.shape != before.shape:
            continue
        norms.append(float(after.norm()))
        cosines.append(
            float(
                torch.nn.functional.cosine_similarity(
                    after.reshape(-1).float(),
                    before.reshape(-1).float(),
                    dim=0,
                ),
            ),
        )
    return {
        "available": bool(norms),
        "mean_norm": float(np.mean(norms)) if norms else float("nan"),
        "mean_cosine_to_anatomical": float(np.mean(cosines)) if cosines else float("nan"),
        "anatomical_norm": ANATOMICAL_READOUT_NORM,
        "ppo_norm": PPO_READOUT_NORM,
        "ppo_cosine_to_anatomical": PPO_READOUT_COSINE,
        "n_read": len(norms),
    }


def _readout_tensor(log: Path, experiments: Path = EXPERIMENTS) -> torch.Tensor | None:
    """Read the run's final readout, or None where no export is on disk."""
    import torch
    from l4_panel import _experiment_json  # pyright: ignore[reportMissingImports]

    experiment = _experiment_json(log.read_text(), experiments)
    exports = experiment.get("exports_path") if experiment else None
    if not exports:
        return None
    final = rp.ps.hm.REPO / exports / "weights" / "final.pt"
    if not final.is_file():
        return None
    topology = torch.load(final, weights_only=True).get("topology")
    if not isinstance(topology, dict):
        return None
    return topology.get("readout")


def compare(
    arm: str,
    scanned: dict[str, Any],
    seeds: tuple[int, ...] = SEEDS,
    experiments: Path = EXPERIMENTS,
) -> dict[str, Any]:
    """Score one learning arm against the shared frozen control.

    Restricted to the REQUESTED seeds, so a campaign directory carrying a re-run or a pilot's seeds
    does not fold them into the contrast, the means, the competence check or the verdict.
    """
    runs = scanned["runs"]
    learning = {s: r.foods for s, r in runs[arm].items() if s in seeds}
    frozen = {s: r.foods for s, r in runs["frozen"].items() if s in seeds}
    learning_clear = [r.success for s, r in runs[arm].items() if s in seeds]
    frozen_mean = float(np.mean(list(frozen.values()) or [math.nan]))
    graded = ms.shift_contrast(learning, frozen)
    mean_clear = float(np.mean(learning_clear)) if learning_clear else float("nan")
    return {
        "arm": arm,
        **ARMS[arm],
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
        "drift": split_drift(arm, scanned["logs"], seeds, experiments),
        "change_by_hop": change_by_hop(arm, scanned["logs"], seeds, experiments),
        "readout_change": readout_change(arm, scanned["logs"], seeds, experiments),
    }


def matched_contrasts(
    cells: dict[str, dict[str, Any]],
    scanned: dict[str, Any],
    seeds: tuple[int, ...] = SEEDS,
) -> dict[str, Any]:
    """Take the two between-arm contrasts that separate the signal's source from its reach.

    Descriptive and paired by seed. They do NOT decide the verdict: every arm is read against its
    own floor, and these say what a difference between arms is attributable to.
    """
    runs = scanned["runs"]
    out: dict[str, Any] = {}
    for label, (first, second) in MATCHED_CONTRASTS.items():
        a = {s: r.foods for s, r in runs[first].items() if s in seeds}
        b = {s: r.foods for s, r in runs[second].items() if s in seeds}
        out[label] = {
            "arms": [first, second],
            "difference_foods": cells[first]["learning_mean_foods"]
            - cells[second]["learning_mean_foods"],
            "paired": ms.shift_contrast(a, b),
        }
    return out


def stage_one_reading(path: Path | None) -> dict[str, Any]:
    """Read the one-step control's record, which gates this campaign.

    Read rather than assumed: the control's ``eprop_symmetric`` arm is the exact REINFORCE gradient
    of its plastic layer, so a failure there means the implementation is wrong and no connectome
    number is interpretable. Absent a record the reading is ``unknown``, which is refused a verdict
    exactly as an incomplete campaign is.
    """
    if path is None:
        return {"available": False, "reading": "unknown", "why": "no stage-1 record was supplied"}
    record = json.loads(path.read_text())
    eprop = record.get("arms", {}).get("eprop", {})
    reading = eprop.get("reading", "unknown")
    return {
        "available": True,
        "path": str(path),
        "reading": reading,
        "void_reason": eprop.get("void_reason"),
        # Reported whatever the reading: the broadcast arm is the one the plausibility claim rests
        # on, and its position bounds what this campaign could show.
        "broadcast_passes": eprop.get("broadcast_passes"),
        "control_outcome": record.get("outcome"),
    }


def analyse(
    scanned: dict[str, Any],
    seeds: tuple[int, ...] = SEEDS,
    experiments: Path = EXPERIMENTS,
    stage_one: Path | None = None,
    *,
    registered: bool = True,
) -> dict[str, Any]:
    """Compare every arm, correct across them and apply the registered reading.

    ``registered=False`` scores a PILOT: the per-arm table is computed as usual and the verdict is
    withheld. A pilot runs on disjoint seeds and usually on a subset of the arms, and at four seeds
    the exact one-sided paired test cannot reach the significance gate at all -- the smallest p it
    can return is 2^-4 = 0.0625, which BH across five arms pushes above it. So every arm reads as
    `no_improvement` whatever it did, and printing `does_not_learn` from that would be announcing
    that the programme stops on evidence that could not have said otherwise.
    """
    cells = {name: compare(name, scanned, seeds, experiments) for name in ARMS}
    defined = [n for n in ARMS if cells[n]["graded"].get("defined")]
    qs = ms.bh_fdr([cells[n]["graded"]["p_improve"] for n in defined])
    for name, q in zip(defined, qs, strict=True):
        cells[name]["graded"]["q_improve"] = float(q)
    for cell in cells.values():
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

    beat = [n for n in ARMS if cells[n]["verdict"] == "beats_floor"]
    competent = [n for n in ARMS if cells[n]["reaches_competence"]]
    # Only an arm that beats its own floor AND reaches competence has learned the cell: an arm
    # competent without clearing its floor has not been shown to have learned anything, and the
    # floor is what says the update did the work.
    learned = [n for n in beat if cells[n]["reaches_competence"]]
    # And only an arm that learned the cell WHILE WRITING THE SUBSTRATE, by at least the absolute
    # minimum over the readout-only control, is a rule that learns the connectome. The distinction
    # is the whole reason the control exists: a plastic readout is an 8-parameter linear map over
    # four pooled motor-class means, so an arm can learn this cell with the wiring frozen, and
    # "the cell was learned" then says nothing about the wiring. Every substrate rung -- B.5, B.1,
    # B.4, B.4b -- asks its question of a rule that writes the wiring, so each is gated on THIS
    # list and not on `learned`.
    control = cells[READOUT_ONLY_ARM]["learning_mean_foods"]
    learned_with_substrate = [
        n
        for n in learned
        if ARMS[n]["chemical"] == "plastic"
        and cells[n]["learning_mean_foods"] - control >= MIN_FOODS
    ]
    stage = stage_one_reading(stage_one)
    if not registered:
        verdict, why = (
            "pilot",
            (
                f"a pilot on {len(seeds)} seeds, which is not a registered verdict: at this seed "
                f"count the exact paired test cannot reach the significance gate (its smallest "
                f"reachable p is 2^-{len(seeds)}), so no arm could have read as beating its floor "
                "whatever it did. The per-arm levels are the pilot's job; the reading is not"
            ),
        )
    elif stage["reading"] != "valid":
        verdict, why = (
            "void",
            (
                "the one-step control does not license reading this campaign "
                f"({stage['reading']}): {stage.get('void_reason') or stage.get('why')}"
            ),
        )
    elif learned_with_substrate:
        verdict, why = (
            "learns_the_cell",
            (
                f"{', '.join(learned_with_substrate)} beats its own floor, reaches competence AND "
                f"clears the readout-only control by at least {MIN_FOODS} foods: 059's third "
                "outcome, met by a rule that writes the wiring. The wiring contrast becomes "
                "runnable and is registered fresh in its own change; B.5, B.1, B.4 and B.4b "
                "become askable"
            ),
        )
    elif learned:
        verdict, why = (
            "learns_without_the_substrate",
            (
                f"{', '.join(learned)} beats its own floor and reaches competence, but no arm does "
                "so while writing the substrate: the readout-only control, whose chemical matrix is "
                f"frozen, reaches {control:.3f} foods. 059's gate is met in LETTER and not in "
                "substance -- what learned the cell is an 8-parameter linear readout over four "
                "pooled motor-class means, on frozen recurrent features. The substrate rungs B.5, "
                "B.1, B.4 and B.4b stay GATED, since each asks its question of a rule that writes "
                "the wiring. R.1b becomes runnable in a changed form: wild type against its "
                "rewired null as FROZEN FEATURES under the readout-only arm"
            ),
        )
    elif beat:
        verdict, why = (
            "learns_below_competence",
            (
                f"{', '.join(beat)} beats its own floor by the registered minima without reaching "
                "competence: 059's second outcome. R.1b stays blocked and 7b's gate is untouched, "
                "since block V's contrast is on time to competence"
            ),
        )
    else:
        verdict, why = (
            "does_not_learn",
            (
                "no arm beats its own floor by the registered minima: 059's first outcome. The "
                "rule family has now failed with two independent eligibilities, the programme "
                "stops, and 7b proceeds under PPO after the power arithmetic"
            ),
        )
    return {
        "arms": cells,
        "beating_floor": beat,
        "reaching_competence": competent,
        "learned_the_cell": learned,
        "learned_with_substrate": learned_with_substrate,
        "substrate_contribution_foods": {
            n: cells[n]["learning_mean_foods"] - control
            for n in ARMS
            if ARMS[n]["chemical"] == "plastic"
        },
        "forbidden_cell": FORBIDDEN_CELL,
        "matched_contrasts": matched_contrasts(cells, scanned, seeds),
        "stage_one": stage,
        "references": {
            "ppo_matched_foods": PPO_MATCHED_FOODS,
            "nodepert_learning_foods": NODEPERT_LEARNING_FOODS,
            "nodepert_frozen_foods": NODEPERT_FROZEN_FOODS,
            "nodepert_shift_foods": NODEPERT_SHIFT_FOODS,
            "target_foods": TARGET_FOODS,
        },
        "verdict": verdict,
        "why": why,
    }


def _print(result: dict[str, Any]) -> None:
    """Print the per-arm table, then what the reading turns on."""
    print("\nR.2 - e-prop on the hard-food cell")
    print(
        "  arm             | source            | reach     | readout | w_chem  | learning | "
        "frozen | shift |     q | clear % | drift-cr | verdict",
    )
    for name, cell in result["arms"].items():
        graded = cell["graded"]
        print(
            f"  {name:15} | {cell['source']:17} | {cell['reach']:9} | {cell['readout']:7} | "
            f"{cell['chemical']:7} | "
            f"{cell['learning_mean_foods']:8.3f} | {cell['frozen_mean_foods']:6.3f} | "
            f"{graded.get('effect', float('nan')):+5.2f} | "
            f"{graded.get('q_improve', float('nan')):5.3f} | "
            f"{cell['learning_mean_full_clear']:7.2f} | "
            f"{cell['drift']['credited_mean_relative']:8.2f} | {cell['verdict']}",
        )
    plastic = result["arms"][PLASTIC_READOUT_ARM]["readout_change"]
    if plastic.get("available"):
        print(
            f"\n  {PLASTIC_READOUT_ARM} readout: norm {plastic['mean_norm']:.3f} "
            f"(anatomical {plastic['anatomical_norm']}, PPO {plastic['ppo_norm']}), "
            f"cosine to anatomical {plastic['mean_cosine_to_anatomical']:+.3f} "
            f"(PPO's {plastic['ppo_cosine_to_anatomical']:+.3f})",
        )
    print(f"\n  forbidden cell: {FORBIDDEN_CELL['source']} reaching {FORBIDDEN_CELL['reach']}")
    print(f"    {FORBIDDEN_CELL['why']}")
    for label, contrast in result["matched_contrasts"].items():
        first, second = contrast["arms"]
        print(
            f"  {label}: {first} - {second} = {contrast['difference_foods']:+.3f} foods "
            f"(p {contrast['paired'].get('p_improve', float('nan')):.3f})",
        )
    stage = result["stage_one"]
    print(
        f"\n  stage 1: {stage['reading']}   broadcast arm passes: {stage.get('broadcast_passes')}",
    )
    print(f"  arms beating their floor: {result['beating_floor'] or 'none'}")
    print(f"  arms reaching competence: {result['reaching_competence'] or 'none'}")
    print(
        "  arms learning the cell WHILE WRITING THE SUBSTRATE: "
        f"{result['learned_with_substrate'] or 'none'}",
    )
    print(
        "  what the substrate's own plasticity contributes, foods over the readout-only control: "
        + ", ".join(
            f"{name} {value:+.3f}" for name, value in result["substrate_contribution_foods"].items()
        ),
    )
    print(f"\nVERDICT: {result['verdict']} - {result['why']}")


def write_csv(result: dict[str, Any], path: Path) -> None:
    """One row per arm and seed, so the table can be recomputed from the record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["arm", "source", "reach", "seed", "learning_foods", "frozen_foods"])
        for name, cell in result["arms"].items():
            for seed in sorted(cell["learning_foods"]):
                writer.writerow(
                    [
                        name,
                        cell["source"],
                        cell["reach"],
                        seed,
                        f"{cell['learning_foods'][seed]:.6f}",
                        f"{cell['frozen_foods'].get(seed, float('nan')):.6f}",
                    ],
                )


def _jsonable(value: object) -> object:
    """Replace not-a-number with null, recursively, so the record is strict JSON."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main(argv: list[str] | None = None) -> int:
    """Score R.2's campaign and print its reading."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--seeds", type=str, default="1-16")
    parser.add_argument("--experiments", type=Path, default=EXPERIMENTS)
    parser.add_argument(
        "--stage-one",
        type=Path,
        default=None,
        help="the one-step control's JSON record, which gates this campaign",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help=(
            "score what is present and WITHHOLD the verdict; for a pilot, never for a "
            "registered campaign"
        ),
    )
    args = parser.parse_args(argv)

    low, _, high = args.seeds.partition("-")
    seeds = tuple(range(int(low), int(high or low) + 1))
    scanned = scan(args.campaign, args.experiments)
    if not args.allow_incomplete:
        require_complete(scanned, seeds)
    result = analyse(
        scanned,
        seeds,
        args.experiments,
        args.stage_one,
        registered=not args.allow_incomplete,
    )
    result["protocol"] = {
        "seeds": list(seeds),
        "arms": {name: dict(meta) for name, meta in ARMS.items()},
        "min_foods": MIN_FOODS,
        "min_gap_fraction": MIN_GAP_FRACTION,
        "competence_threshold": COMPETENCE_THRESHOLD,
        "drift_reference": list(DRIFT_REFERENCE),
    }
    _print(result)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(_jsonable(result), indent=2, sort_keys=True, allow_nan=False) + "\n",
        )
    if args.csv:
        write_csv(result, args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
