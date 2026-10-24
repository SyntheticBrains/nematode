#!/usr/bin/env python
r"""Read how on-lawn reversal kinematics move the classifier's roaming fraction, as description.

The primitives' reversals were first given published off-food kinematics (2.69 ± 1.75 s at
0.14 mm/s). On a lawn real worms reverse for 2.05 ± 0.90 s at 0.069 mm/s, so an off-food reversal
carries a slow worm nearly three times as far as the animal's. This pilot runs the same
non-learning controllers on two lawn assays under each setting of
``continuous.reversal_kinematics`` and classifies each animal's on-lawn windows with the calibrated
roaming/dwelling model.

**Controllers** (``BatchedConstantHazards``, straight runs at a fixed speed and constant start
hazards; omega and delta at 0.005/s each unless stated):

- 0.01 mm/s, reversal 0.03/s; 0.02 mm/s, reversal 0.05/s; 0.04 mm/s, reversal 0.03/s;
- 0.15 mm/s with hazards (0.02, 0.01, 0.01)/s, the parity suite's constant controller;
- ``BatchedPirouetteHazards`` at the same, beta = 3.

**Assays:** Flavell's uniform lawn and Scheer's small lawn, from their generated configs, read over
each protocol's analysis window. **Seeds** 9001-9064, disjoint from every registered band.

Per controller, assay and setting it reports the mean on-lawn roaming fraction over animals with
on-lawn windows, its SEM, the fraction of steps on a lawn, and sustained exits per animal. Nothing
here is graded.

Usage::

    uv run python scripts/analysis/food_dependent_kinematics_pilot.py \
        --out docs/experiments/logbooks/supporting/108-food-dependent-kinematics/pilot.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lawn_states
import quantumnematode.brain  # noqa: F401  - registers the brains the configs name
from quantumnematode.evaluation.batched import BatchedEvaluator
from quantumnematode.evaluation.controllers import (
    BatchedConstantHazards,
    BatchedPirouetteHazards,
)
from quantumnematode.utils.config_loader import load_simulation_config
from quantumnematode.validation import roaming_dwelling as rd

_REPO = Path(__file__).resolve().parents[2]
_STEM = "mlpppo_small_continuous2d_fick_adaptive_klinotaxis_lawns_internal_s1_primitives_assay_"
ASSAYS = ("uniform_lawn_flavell", "scheer_lawn")
SEEDS = list(range(9001, 9065))
SETTINGS = ("off_food", "food_dependent")
TURNS = 0.005
DESCRIPTION = (
    "How on-lawn reversal kinematics move the classifier's roaming fraction, as description."
)


def controllers() -> dict[str, Any]:
    """Return the pilot's controllers by name."""
    out: dict[str, Any] = {
        f"constant_{speed}mm_rev{rev}": BatchedConstantHazards(speed, (rev, TURNS, TURNS))
        for speed, rev in ((0.01, 0.03), (0.02, 0.05), (0.04, 0.03))
    }
    out["constant_0.15mm_parity"] = BatchedConstantHazards(0.15, (0.02, 0.01, 0.01))
    out["pirouette_0.15mm_parity"] = BatchedPirouetteHazards(0.15, (0.02, 0.01, 0.01), beta=3.0)
    return out


def with_setting(config: Any, setting: str) -> Any:  # noqa: ANN401 - a simulation config
    """Return ``config`` with ``continuous.reversal_kinematics`` set."""
    environment = config.environment
    continuous = environment.continuous.model_copy(update={"reversal_kinematics": setting})
    return config.model_copy(
        update={"environment": environment.model_copy(update={"continuous": continuous})},
    )


def read(config: Any, controller: Any) -> dict[str, float]:  # noqa: ANN401
    """Run the controller on every seed and summarise its on-lawn states."""
    lawns = config.environment.foraging.lawns
    classifier = rd.GaussianClassifier(hmm=rd.load_calibrated_hmm())
    trajectories = BatchedEvaluator(config, controller, SEEDS).run()
    roaming, on_lawn, exits = [], [], []
    for i in range(len(SEEDS)):
        steps = trajectories.to_behaviour_steps(i)
        if trajectories.assay is not None:
            first, stop = trajectories.assay["analysis_steps"]  # type: ignore[misc]
            steps = steps[first:stop]
        episode = lawn_states.episode_windows(
            steps,
            classifier,
            lawns.intake_fraction,
            1.0,
            step_seconds=1.0,
        )
        on = episode.states != rd.OFF_FOOD
        if on.any():
            roaming.append(float(np.mean(episode.states[on] == rd.ROAMING)))
        on_lawn.append(float(np.mean([bool(s.on_lawn) for s in steps])))
        exits.append(len(episode.exits))
    return {
        "animals_on_lawn": len(roaming),
        "roaming": float(np.mean(roaming)) if roaming else math.nan,
        "roaming_sem": float(np.std(roaming, ddof=1) / math.sqrt(len(roaming)))
        if len(roaming) > 1
        else math.nan,
        "on_lawn": float(np.mean(on_lawn)),
        "exits_per_animal": float(np.mean(exits)),
    }


def main() -> int:
    """CLI: run the pilot and write its record."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    results: dict[str, Any] = {}
    for assay in ASSAYS:
        path = _REPO / "configs" / "scenarios" / "foraging" / f"{_STEM}{assay}.yml"
        config = load_simulation_config(str(path))
        for name, controller in controllers().items():
            for setting in SETTINGS:
                key = f"{assay}/{name}/{setting}"
                results[key] = read(with_setting(config, setting), controller)
                print(key, json.dumps(results[key]), flush=True)
    record = {
        "description": DESCRIPTION,
        "seeds": [SEEDS[0], SEEDS[-1]],
        "settings": list(SETTINGS),
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    clean = json.loads(json.dumps(record), parse_constant=lambda _c: None)
    args.out.write_text(json.dumps(clean, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
