#!/usr/bin/env python
"""Emit the standard lawn assays as protocols on the 1-second primitives worm.

Each assay reproduces its reference's arena, lawn, start, food history, duration and analysis window,
on the 1-second primitives arms: the MLP reading internal state, food contact and locomotor state,
the connectome with serotonin and PDF, the data-layer connectome under the compact encoder, and two
following arms: the encoded arm and the MLP arm, each with on-food reversals backing at a speed that
follows the worm's recent run speed (``encoded_following``, C.2's registered body, and
``internal_following``, the body C.1's mechanistic controller is refitted on).

| Protocol | Reference | Arena | Lawn | Phases | Analysis |
|---|---|---|---|---|---|
| ``scheer_lawn`` | Scheer & Bargmann 2023 | 10 mm circle | 1.5 mm radius | 60 min from bare agar | from 20 min |
| ``uniform_lawn_scheer`` | Scheer & Bargmann 2023 | 10 mm circle | the whole well | 60 min on it | from 20 min |
| ``food_removal_lopezcruz`` | López-Cruz et al. 2019 | 100 mm circle | 25 mm radius | 45 min fed, 45 bare | from removal |
| ``food_removal_calhoun`` | Calhoun et al. 2014 | 25.4 mm square | 11 mm radius | 45 min fed, 30 bare | from 2 min |
| ``uniform_lawn_flavell`` | Flavell et al. 2013 | 35 mm circle | 16 mm radius | 90 min on the lawn | all |

Every assay sets three things:

- **A slow reserve.** Satiety empties over 12 h of basal decay, and never ends the episode: a worm
  off food for hours is hungry, not dead.
- **No cost for moving.** Locomotion's metabolic cost has no published rate.
- **A lawn that outlasts the assay.** A stationary worm halves its cell's density in about 2 h, where
  D.1's lawn cell halves it in about 3 min. Satiety per unit eaten is raised so that a fresh cell
  still restores more than ten times the decay.

The per-step values are written for the 5-s reference step, as every config is; ``step_seconds:
1.0`` converts them. Calhoun's worms fed overnight, and the 45-min fed phase approximates that.
López-Cruz's thin, whole-plate lawn is modelled as one 25 mm-radius lawn at full density.

Usage::

    uv run python scripts/campaigns/generate_assay_configs.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_lawn_configs as lawn_gen

FORAGING = lawn_gen.FORAGING
# The encoded arm with ``continuous.on_food_reversal_speed: follows_run_speed``, its only difference.
FOLLOWING_ARM = "encoded_following"
# The MLP arm with the same single difference.
INTERNAL_FOLLOWING_ARM = "internal_following"
# Each following arm and the arm it is derived from.
FOLLOWING_BASES: dict[str, str] = {
    FOLLOWING_ARM: lawn_gen.ENCODED_STEM_ARM,
    INTERNAL_FOLLOWING_ARM: "internal",
}
ARMS: tuple[str, ...] = (
    "internal",
    "modulated",
    lawn_gen.ENCODED_STEM_ARM,
    FOLLOWING_ARM,
    INTERNAL_FOLLOWING_ARM,
)
REFERENCE_STEP_S = 5.0
# Satiety empties over 12 h of basal decay: 300 over 43,200 s, written per 5-s reference step.
INITIAL_SATIETY = 300.0
RESERVE_S = 43_200.0
SATIETY_DECAY = INITIAL_SATIETY / (RESERVE_S / REFERENCE_STEP_S)
# A cell's density halves in about 2 h under a stationary worm: 1e-4 per second, so 5e-4 per 5-s
# reference step; a fresh cell then restores 1e-4 * 300 * 3.0 = 0.09 per second.
ASSAY_INTAKE_FRACTION = 0.0005
ASSAY_SATIETY_PER_INTAKE = 3.0

PROTOCOLS: dict[str, dict[str, Any]] = {
    "scheer_lawn": {
        "reference": "Scheer & Bargmann 2023, eLife 10.7554/eLife.88657",
        "world_size_mm": 10.0,
        "arena_shape": "circle",
        "lawn_radius_mm": 1.5,
        "start_clearance_mm": 0.5,
        "phases": [{"name": "assay", "duration_s": 3600.0, "food": "lawns", "place": "off_lawn"}],
        "analysis_from_s": 1200.0,
    },
    "uniform_lawn_scheer": {
        "reference": "Scheer & Bargmann 2023, eLife 10.7554/eLife.88657 (uniform lawn)",
        "world_size_mm": 10.0,
        "arena_shape": "circle",
        "lawn_radius_mm": 5.0,
        "wall_clearance_mm": 0.0,
        "start_clearance_mm": 0.5,
        "phases": [
            {"name": "assay", "duration_s": 3600.0, "food": "lawns", "place": "lawn_centre"},
        ],
        "analysis_from_s": 1200.0,
    },
    "food_removal_lopezcruz": {
        "reference": "López-Cruz et al. 2019, Neuron 10.1016/j.neuron.2019.01.053",
        "world_size_mm": 100.0,
        "arena_shape": "circle",
        "lawn_radius_mm": 25.0,
        "start_clearance_mm": 2.0,
        "phases": [
            {"name": "fed", "duration_s": 2700.0, "food": "lawns", "place": "lawn_centre"},
            {"name": "assay", "duration_s": 2700.0, "food": "none", "place": "arena_centre"},
        ],
        "analysis_from_s": 0.0,
    },
    "food_removal_calhoun": {
        "reference": "Calhoun, Chalasani & Sharpee 2014, eLife 10.7554/eLife.04220",
        "world_size_mm": 25.4,
        "arena_shape": "square",
        "lawn_radius_mm": 11.0,
        "start_clearance_mm": 2.0,
        "phases": [
            {"name": "fed", "duration_s": 2700.0, "food": "lawns", "place": "lawn_centre"},
            {"name": "assay", "duration_s": 1800.0, "food": "none", "place": "arena_centre"},
        ],
        "analysis_from_s": 120.0,
    },
    "chemotaxis_chen": {
        "reference": "Chen et al. 2025, PLoS Biology 10.1371/journal.pbio.3003005 (naive N2, butanone)",
        "world_size_mm": 100.0,
        "arena_shape": "square",
        "lawn_radius_mm": 25.0,
        "start_clearance_mm": 2.0,
        "phases": [
            {"name": "fed", "duration_s": 2700.0, "food": "lawns", "place": "lawn_centre"},
            {"name": "assay", "duration_s": 1800.0, "food": "none", "place": "arena_centre"},
        ],
        "analysis_from_s": 0.0,
    },
    "uniform_lawn_flavell": {
        "reference": "Flavell et al. 2013, Cell 10.1016/j.cell.2013.08.001",
        "world_size_mm": 35.0,
        "arena_shape": "circle",
        "lawn_radius_mm": 16.0,
        "start_clearance_mm": 2.0,
        "phases": [
            {"name": "assay", "duration_s": 5400.0, "food": "lawns", "place": "lawn_centre"},
        ],
        "analysis_from_s": 0.0,
    },
}


def _base_arm(arm: str) -> str:
    """Return the arm an assay arm's settings come from: itself, or a following arm's base."""
    return FOLLOWING_BASES.get(arm, arm)


def _base_stem(arm: str) -> str:
    """Return the stem of the primitives arm config an assay arm is derived from."""
    if _base_arm(arm) == lawn_gen.ENCODED_STEM_ARM:
        return lawn_gen.encoded_stem()
    return lawn_gen.primitives_stem(_base_arm(arm))


def _arm_stem(arm: str) -> str:
    if arm in FOLLOWING_BASES:
        return f"{_base_stem(arm)}_following"
    return _base_stem(arm)


def stem(arm: str, protocol: str) -> str:
    """Return an assay config's stem: its primitives arm's, then the protocol."""
    return f"{_arm_stem(arm)}_assay_{protocol}"


def derive(arm: str, protocol: str) -> tuple[Path, str]:
    """Build one assay config: the primitives arm with the protocol's arena, lawn and phases."""
    if arm not in ARMS or protocol not in PROTOCOLS:
        msg = f"unknown arm {arm!r} or protocol {protocol!r}"
        raise ValueError(msg)
    spec = PROTOCOLS[protocol]
    base = _base_arm(arm)
    _, text = (
        lawn_gen.derive_encoded()
        if base == lawn_gen.ENCODED_STEM_ARM
        else lawn_gen.derive_primitives(base)
    )
    data: dict[str, Any] = yaml.safe_load(text)
    total_s = sum(phase["duration_s"] for phase in spec["phases"])
    data["max_steps"] = round(total_s / REFERENCE_STEP_S)
    data["satiety"] = {
        **data["satiety"],
        "initial_satiety": INITIAL_SATIETY,
        "satiety_decay_rate": SATIETY_DECAY,
        "starvation_ends_episode": False,
    }
    environment = data["environment"]
    environment["continuous"].update(
        {"world_size_mm": spec["world_size_mm"], "arena_shape": spec["arena_shape"]},
    )
    if arm in FOLLOWING_BASES:
        environment["continuous"]["on_food_reversal_speed"] = "follows_run_speed"
    environment["foraging"]["lawns"].update(
        {
            "count": 1,
            "radius_mm": spec["lawn_radius_mm"],
            "placement": "centre",
            "start_on_lawn": False,
            "start_clearance_mm": spec["start_clearance_mm"],
            **(
                {"wall_clearance_mm": spec["wall_clearance_mm"]}
                if "wall_clearance_mm" in spec
                else {}
            ),
            "intake_fraction": ASSAY_INTAKE_FRACTION,
            "satiety_per_intake": ASSAY_SATIETY_PER_INTAKE,
            "movement_cost_per_mm": 0.0,
        },
    )
    environment["assay"] = {
        "name": protocol,
        "phases": spec["phases"],
        "assay_phase": "assay",
        "analysis_from_s": spec["analysis_from_s"],
    }
    header = (
        f"# Assay `{protocol}` on the 1-second primitives worm, arm `{arm}`.\n#\n"
        f"# Reference: {spec['reference']}.\n"
        f"# Derived from {_base_stem(arm)}.yml: the reference's arena, one centred\n"
        "# lawn, its phases and analysis window; satiety a 12-h reserve that never ends the\n"
        "# episode; no movement cost; a lawn whose cells halve in about 2 h under a still worm.\n"
        + (
            "# On-food reversals back at a speed that follows the worm's recent run speed\n"
            "# (continuous.on_food_reversal_speed: follows_run_speed), the arm's only difference.\n"
            if arm in FOLLOWING_BASES
            else ""
        )
        + (
            "# Calhoun's worms fed overnight; the 45-min fed phase approximates that.\n"
            if protocol == "food_removal_calhoun"
            else ""
        )
        + (
            "# The odour is Chen's measured butanone field, which the batched evaluator reads\n"
            "# off food (quantumnematode.evaluation.odour_field); the assay phase starts each\n"
            "# worm at a real worm's start. The fed phase approximates the naive worms' feeding.\n"
            if protocol == "chemotaxis_chen"
            else ""
        )
        + (
            "# The bacteria cover the well: the lawn's radius is the well's, with no wall\n"
            "# clearance, so the worm is on the lawn throughout.\n"
            if protocol == "uniform_lawn_scheer"
            else ""
        )
        + "#\n# Generated by scripts/campaigns/generate_assay_configs.py; re-checked through the\n"
        "# real config loader by test_assay_configs.py.\n#\n"
    )
    return FORAGING / f"{stem(arm, protocol)}.yml", header + yaml.safe_dump(data, sort_keys=False)


def main() -> int:
    """CLI: write every assay config if missing."""
    written = kept = 0
    for arm in ARMS:
        for protocol in PROTOCOLS:
            path, text = derive(arm, protocol)
            if path.exists():
                kept += 1
            else:
                path.write_text(text)
                written += 1
    print(f"wrote {written} configs; kept {kept}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
