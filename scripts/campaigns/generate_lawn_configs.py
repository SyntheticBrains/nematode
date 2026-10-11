#!/usr/bin/env python
"""Emit D.1's lawn-cell configs: MLP-PPO on patchy lawns, with and without internal state.

The cell is the point worm's MLP-PPO at the hard-food cell's sensing (klinotaxis, the adaptive
fold-change sensor, the Fick field), with point food replaced by lawns:

- **Lawns**: four disc lawns of 2.5 mm radius in a 20 mm arena, 1 mm cells, intake 2% of a cell's
  density per step (a spot lasts minutes, as a real lawn does), no regrowth.
- **Moving costs energy**: 0.013 of intake value per mm, from reward and satiety alike. On a fresh
  cell a worm gains 0.006 of its maximum satiety per step; moving 1 mm costs 0.0039; basal decay is
  0.0027. A spot is worth staying on until it falls to about 0.35 of its density, some 52 steps
  (4.3 minutes), and roaming through a lawn without stopping slowly starves the worm.
- **A lower entropy bonus**, 0.005: at the base's 0.05, PPO kept its speed noise near the boundary
  between the two states (about 0.05 mm/s of jitter alone) and drove the turn's noise until every
  turn was random, since turning cost nothing.
- **The worm starts on a lawn**, as assays place worms on food: the first lawn is centred on its
  start. With movement costed and the worm starting off food, PPO learned to stand still and
  starve before it found a lawn.
- **Both states expressible**: signed speed (zero is the centre of the speed range) and turns of up
  to half a revolution per step, the reorientation a dwelling worm makes.
- **No shaping that favours a state**: the stuck-position, anti-dithering, exploration and distance
  terms are zero, so intake is the only food reward.
- **Satiety matters within an episode**: 720 steps (an hour of worm time), and an idle worm starves
  in about 375 of them.

Arms: ``internal`` reads ``internal_state``; ``blind`` does not; ``internal_frozen`` is the
learner's untrained policy, its floor. The pilot may revise the cell's numbers before registration;
each config is re-checked through the real loader by a test, and an existing config is left alone.

Usage::

    uv run python scripts/campaigns/generate_lawn_configs.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORAGING = PROJECT_ROOT / "configs" / "scenarios" / "foraging"
BASE_STEM = "mlpppo_small_continuous2d_fick_adaptive_klinotaxis_hard350_ppo_w64"
STEM = "mlpppo_small_continuous2d_fick_adaptive_klinotaxis_lawns"
ARMS: tuple[str, ...] = ("internal", "blind", "internal_frozen")
# D.1b's leaving cell: D.1's cell with smaller, more numerous lawns, so one lawn cannot feed an
# episode and leaving pays. Its arms are the learner and its floor.
LEAVING_ARMS: tuple[str, ...] = ("internal", "internal_frozen")
# B.3 stage 0: the connectome on D.1's cell, at two entropy levels, each with its untrained floor.
CONNECTOME_BASE = (
    "connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_hard350_emmons_reversal"
)
CONNECTOME_STEM = "connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_lawns"
CONNECTOME_ENTROPY: dict[str, float] = {"e005": 0.005, "e05": 0.05}
CONNECTOME_ARMS: tuple[str, ...] = tuple(
    f"connectome_{level}{floor}" for level in CONNECTOME_ENTROPY for floor in ("", "_frozen")
)
# B.3: the connectome at stage 0's settings (entropy 0.005) with the serotonin and PDF field. Each
# substrate has the intact learner and its untrained floor; the raised mutants, knocked out from
# the start of training; and the satiety link, NSM's ingestion drive amplified by hunger.
_MODULATED_ROLES: dict[str, dict[str, object]] = {
    "": {},
    "_frozen": {"freeze_updates": True},
    "_tph1": {"modulator_knockout": "tph1"},
    "_pdfr1": {"modulator_knockout": "pdfr1"},
    "_satiety": {"modulator_satiety_link": True},
}
# Two substrates: stage 0's random synapse signs, and signs grounded in the transmitter atlas and
# held under PPO, so that what a receptor cell does downstream carries its biological sign.
MODULATED_SUBSTRATES: dict[str, dict[str, object]] = {
    "modulated": {},
    "modulated_atlas": {"synapse_signs": "atlas", "enforce_synapse_signs": True},
}
MODULATED_ARMS: dict[str, dict[str, object]] = {
    f"{base}{role}": {**signs, **settings}
    for base, signs in MODULATED_SUBSTRATES.items()
    for role, settings in _MODULATED_ROLES.items()
}
# The acute knockouts: the intact learner's trained weights evaluated with one modulator removed.
KNOCKOUTS: tuple[str, ...] = ("tph1", "pdfr1")
ACUTE_ARMS: tuple[str, ...] = tuple(
    f"{base}+{k}" for base in MODULATED_SUBSTRATES for k in KNOCKOUTS
)
CELLS: dict[str, dict[str, object]] = {
    "patch": {},
    "leaving": {"radius_mm": 1.0, "count": 8, "min_separation_mm": 2.0},
    # The second sweep: lawns large enough to graze inside before travelling.
    "leaving15x4": {"radius_mm": 1.5, "count": 4, "min_separation_mm": 2.0},
    "leaving15x6": {"radius_mm": 1.5, "count": 6, "min_separation_mm": 2.0},
}

MAX_STEPS = 720
LAWNS: dict[str, Any] = {
    "count": 4,
    "radius_mm": 2.5,
    "min_separation_mm": 2.0,
    "cell_mm": 1.0,
    "quality": [1.0, 1.0],
    "intake_fraction": 0.02,
    "reward_per_intake": 10.0,
    "satiety_per_intake": 0.3,
    "regrowth_per_step": 0.0,
    "movement_cost_per_mm": 0.013,
    "start_on_lawn": True,
    "start_clearance_mm": 2.0,
    "wall_clearance_mm": 1.0,
}
# A learning setting: at the base's 0.05 the entropy bonus kept the speed noise near the boundary
# between the two states and pushed the turn's noise until every turn was random.
ENTROPY_COEF = 0.005
ZERO_SHAPING = (
    "penalty_stuck_position",
    "penalty_anti_dithering",
    "reward_exploration",
    "reward_distance_scale",
)


def stem(arm: str, cell: str = "patch") -> str:
    """Return the config stem of ``arm`` in ``cell``; connectome arms take the connectome's stem.

    An acute arm, ``<arm>+<knockout>``, runs its trained arm's config, so it has that arm's stem.
    """
    arm = arm.split("+", 1)[0]
    base = CONNECTOME_STEM if arm.startswith(("connectome", "modulated")) else STEM
    return f"{base}_{arm}" if cell == "patch" else f"{base}_{cell}_{arm}"


def derive_connectome(arm: str) -> tuple[Path, str]:
    """Build one connectome arm on D.1's lawn cell from the connectome's point-worm config."""
    if arm not in CONNECTOME_ARMS:
        msg = f"unknown connectome arm {arm!r}; expected one of {CONNECTOME_ARMS}"
        raise ValueError(msg)
    level = arm.removeprefix("connectome_").removesuffix("_frozen")
    data: dict[str, Any] = yaml.safe_load((FORAGING / f"{CONNECTOME_BASE}.yml").read_text())
    lawn: dict[str, Any] = yaml.safe_load((FORAGING / f"{stem('internal')}.yml").read_text())
    data["max_steps"] = MAX_STEPS
    brain = data["brain"]["config"]
    brain["entropy_coef"] = CONNECTOME_ENTROPY[level]
    if arm.endswith("_frozen"):
        brain["freeze_updates"] = True
    data["reward"] = lawn["reward"]
    data["satiety"] = lawn["satiety"]
    data["environment"]["continuous"]["max_turn_rad"] = lawn["environment"]["continuous"][
        "max_turn_rad"
    ]
    data["environment"]["foraging"] = lawn["environment"]["foraging"]
    header = (
        f"# B.3 stage 0, arm `{arm}`: the connectome on D.1's lawn cell.\n#\n"
        f"# Derived from {CONNECTOME_BASE}.yml (Emmons 2024, reversal, settling depth 4):\n"
        f"# max_steps {MAX_STEPS}; entropy_coef {CONNECTOME_ENTROPY[level]};"
        + (" freeze_updates (the floor);" if arm.endswith("_frozen") else "")
        + "\n# reward, satiety, max_turn_rad and foraging (lawns) taken from "
        + f"{stem('internal')}.yml.\n"
        "# No satiety input: it reaches the connectome through B.3's modulators.\n#\n"
        "# Generated by scripts/campaigns/generate_lawn_configs.py; re-checked through the\n"
        "# real config loader by test_lawn_configs.py.\n#\n"
    )
    return FORAGING / f"{stem(arm)}.yml", header + yaml.safe_dump(data, sort_keys=False)


def derive_modulated(arm: str) -> tuple[Path, str]:
    """Build one modulated arm: stage 0's entropy-0.005 connectome with the modulator field on."""
    if arm not in MODULATED_ARMS:
        msg = f"unknown modulated arm {arm!r}; expected one of {tuple(MODULATED_ARMS)}"
        raise ValueError(msg)
    _, text = derive_connectome("connectome_e005")
    data: dict[str, Any] = yaml.safe_load(text)
    data["brain"]["config"].update({"modulators": True, **MODULATED_ARMS[arm]})
    settings = "; ".join(
        f"{key}: {str(value).lower() if isinstance(value, bool) else value}"
        for key, value in MODULATED_ARMS[arm].items()
    )
    header = (
        f"# B.3, arm `{arm}`: serotonin and PDF on the connectome, on D.1's lawn cell.\n#\n"
        "# Derived from stage 0's entropy-0.005 arm,\n"
        f"# {stem('connectome_e005')}.yml:\n"
        f"# modulators: true{'; ' + settings if settings else ''}.\n#\n"
        "# Generated by scripts/campaigns/generate_lawn_configs.py; re-checked through the\n"
        "# real config loader by test_lawn_configs.py.\n#\n"
    )
    return FORAGING / f"{stem(arm)}.yml", header + yaml.safe_dump(data, sort_keys=False)


def derive(arm: str, cell: str = "patch") -> tuple[Path, str]:
    """Build one arm's config in ``cell`` from the point worm's MLP-PPO config."""
    if arm not in ARMS or cell not in CELLS:
        msg = f"unknown arm {arm!r} or cell {cell!r}; expected one of {ARMS} and {tuple(CELLS)}"
        raise ValueError(msg)
    data: dict[str, Any] = yaml.safe_load((FORAGING / f"{BASE_STEM}.yml").read_text())
    data["max_steps"] = MAX_STEPS
    brain = data["brain"]["config"]
    brain["signed_speed"] = True
    brain["entropy_coef"] = ENTROPY_COEF
    modules = [m for m in brain["sensory_modules"] if m != "internal_state"]
    if arm != "blind":
        modules.append("internal_state")
    brain["sensory_modules"] = modules
    if arm == "internal_frozen":
        brain["freeze_updates"] = True
    for key in ZERO_SHAPING:
        data["reward"][key] = 0
    continuous = data["environment"]["continuous"]
    continuous["allow_reversal"] = True
    continuous["max_turn_rad"] = round(math.pi, 6)
    foraging = data["environment"]["foraging"]
    for key in ("foods_on_grid", "target_foods_to_collect", "min_food_distance"):
        foraging.pop(key, None)
    foraging["food_model"] = "lawns"
    foraging["lawns"] = dict(LAWNS) | CELLS[cell]
    title = (
        f"# D.1 lawn cell, arm `{arm}`.\n#\n"
        if cell == "patch"
        else f"# D.1b leaving cell, arm `{arm}`: lawns {CELLS[cell]}.\n#\n"
    )
    header = title + (
        f"# Derived from {BASE_STEM}.yml: max_steps {MAX_STEPS}; signed speed and reversal;\n"
        "# max_turn_rad pi; entropy_coef 0.005; point food replaced by lawns; the stuck-position,\n"
        "# anti-dithering,"
        " exploration and distance rewards zero"
        + {
            "internal": "; reads internal_state.\n",
            "blind": "; does not read internal_state.\n",
            "internal_frozen": "; reads internal_state; freeze_updates (the floor).\n",
        }[arm]
        + "#\n# Generated by scripts/campaigns/generate_lawn_configs.py; re-checked through the\n"
        "# real config loader by test_lawn_configs.py.\n#\n"
    )
    return FORAGING / f"{stem(arm, cell)}.yml", header + yaml.safe_dump(data, sort_keys=False)


def main() -> int:
    """CLI: write every arm's config if missing."""
    written = kept = 0
    for cell, arms in [("patch", ARMS)] + [(c, LEAVING_ARMS) for c in CELLS if c != "patch"]:
        for arm in arms:
            path, text = derive(arm, cell)
            if path.exists():
                kept += 1
            else:
                path.write_text(text)
                written += 1
    derived = [derive_connectome(arm) for arm in CONNECTOME_ARMS]
    derived += [derive_modulated(arm) for arm in MODULATED_ARMS]
    for path, text in derived:
        if path.exists():
            kept += 1
        else:
            path.write_text(text)
            written += 1
    print(f"wrote {written} configs; kept {kept}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
