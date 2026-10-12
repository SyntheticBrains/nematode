#!/usr/bin/env python
"""Emit M.9's configs: the thermal cell through the kinematic body, and the point-worm anchor check.

Each body arm is a committed thermal point-worm config, on Cook 2019, with the body switched on: the
kinematic body model, reversal (the body requires it), and the 25-number body-drive action. The
connectome arms set ``entropy_coef`` 0.004, the point worm's per-dimension entropy pressure on the
body's 25 action numbers, as C.1e's arms did. The MLP arms set 0.05, the setting at which C.1d's body
positive control passed. Every arm is written at target 35 and at target 20, since stage 1 chooses
between them by a rule fixed beforehand.

* Connectome arms: the wild type, the gap-only null and the chemical-only null, each learning and
  frozen.
* MLP arms: the thermal MLP, learning and frozen.
* The point-worm anchor check at target 20: the gap-only null, learning and frozen, beside the
  committed wild-type pair.

A test re-reads every file through the real loader. A config that already exists is left alone.

Usage::

    uv run python scripts/campaigns/generate_thermal_body_configs.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
THERMAL = PROJECT_ROOT / "configs" / "scenarios" / "thermal_foraging"
_CONNECTOME = "connectomeppo_small_continuous2d_thermal_klinotaxis"
_MLP = "mlpppo_small_continuous2d_thermal_klinotaxis"
TARGETS: tuple[int, ...] = (35, 20)
CONNECTOME_ENTROPY = 0.004
MLP_ENTROPY = 0.05
# Wiring tag -> the brain's wiring value.
WIRINGS: dict[str, str] = {
    "wt": "wild_type",
    "gaponly": "rewired_gap_junctions_only",
    "chemnull": "rewired_chemical_only",
}
LEARNERS: tuple[str, ...] = ("ppo", "frozen")


def stem(wiring: str, learner: str, target: int) -> str:
    """Return one body arm's config stem; ``wiring`` is a connectome wiring tag or ``mlp``."""
    base = _MLP if wiring == "mlp" else _CONNECTOME
    tag = "" if wiring == "mlp" else f"_{wiring}"
    return f"{base}_body{tag}_{learner}_t{target}"


def anchor_stem(learner: str) -> str:
    """Return the point-worm gap-only null's stem at target 20, the anchor check."""
    floor = "_frozen" if learner == "frozen" else ""
    return f"{_CONNECTOME}_rewired_gap_only_null{floor}_t20"


def _connectome_parent(learner: str, target: int) -> str:
    return f"{_CONNECTOME}{'_frozen' if learner == 'frozen' else ''}_t{target}"


def _body(data: dict[str, Any], entropy: float) -> list[str]:
    """Switch a thermal point-worm config onto the kinematic body; return the changed keys."""
    continuous = data["environment"].setdefault("continuous", {})
    continuous["body_model"] = "kinematic"
    continuous["allow_reversal"] = True
    brain = data["brain"]["config"]
    brain["action_space"] = "body_drive"
    brain.pop("signed_speed", None)
    brain["entropy_coef"] = entropy
    return [
        "environment.continuous.body_model: kinematic",
        "environment.continuous.allow_reversal: true",
        "brain.config.action_space: body_drive",
        f"brain.config.entropy_coef: {entropy}",
    ]


def derive(wiring: str, learner: str, target: int) -> tuple[Path, str]:
    """Build one body arm from its thermal point-worm parent."""
    if wiring == "mlp":
        parent = f"{_MLP}_t35"
        data: dict[str, Any] = yaml.safe_load((THERMAL / f"{parent}.yml").read_text())
        lines = _body(data, MLP_ENTROPY)
        data["environment"]["foraging"]["target_foods_to_collect"] = target
        lines.append(f"environment.foraging.target_foods_to_collect: {target}")
        if learner == "frozen":
            data["brain"]["config"]["freeze_updates"] = True
            lines.append("brain.config.freeze_updates: true")
    else:
        parent = _connectome_parent(learner, target)
        data = yaml.safe_load((THERMAL / f"{parent}.yml").read_text())
        lines = _body(data, CONNECTOME_ENTROPY)
        data["brain"]["config"]["wiring"] = WIRINGS[wiring]
        lines.append(f"brain.config.wiring: {WIRINGS[wiring]}")
    return _write(stem(wiring, learner, target), parent, data, lines, "the thermal body cell")


def derive_anchor(learner: str) -> tuple[Path, str]:
    """Build the point-worm gap-only null at target 20 from the committed target-20 pair."""
    parent = _connectome_parent(learner, 20)
    data: dict[str, Any] = yaml.safe_load((THERMAL / f"{parent}.yml").read_text())
    data["brain"]["config"]["wiring"] = WIRINGS["gaponly"]
    lines = [f"brain.config.wiring: {WIRINGS['gaponly']}"]
    return _write(anchor_stem(learner), parent, data, lines, "the point-worm anchor at target 20")


def _write(
    child: str,
    parent: str,
    data: dict[str, Any],
    lines: list[str],
    role: str,
) -> tuple[Path, str]:
    header = (
        f"# M.9 arm, {role}: {child}.\n#\n# Delta from {parent}.yml, and nothing else changes:\n"
        + "".join(f"#   {line}\n" for line in lines)
        + "#\n# Generated by scripts/campaigns/generate_thermal_body_configs.py; re-checked through\n"
        "# the real config loader by test_thermal_body.py.\n#\n"
    )
    return THERMAL / f"{child}.yml", header + yaml.safe_dump(data, sort_keys=False)


def every_config() -> list[tuple[Path, str]]:
    """Return every M.9 config: body arms at both targets, then the anchor check."""
    body = [
        derive(wiring, learner, target)
        for target in TARGETS
        for wiring in (*WIRINGS, "mlp")
        for learner in LEARNERS
    ]
    return body + [derive_anchor(learner) for learner in LEARNERS]


def main(argv: list[str] | None = None) -> int:
    """CLI: write every missing config, and report what was already there."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.parse_args(argv)
    configs = every_config()
    written = 0
    for path, text in configs:
        if not path.exists():
            path.write_text(text)
            written += 1
    print(f"wrote {written} configs; kept {len(configs) - written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
