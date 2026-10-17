#!/usr/bin/env python
"""Emit G.0-A's selection-pilot configs: ohmic against log-compressed gap coupling, on both cells.

On the thermal cell at target 35 and on hard350, both on Emmons 2024: for each candidate model (`ohmic`
gap coupling at the default weights; `log_count` weights with additive coupling) the wild type and the
gap-only null, each learning and frozen; the additive-count wild type learning, described beside; and the
MLP-PPO control's frozen floor where none is committed (its learning configs are reused as they stand).

Each config is a committed parent with only the listed keys changed, re-read by a loader test. A config
that already exists is left alone, never overwritten.

Usage::

    uv run python scripts/campaigns/generate_gap_substrate_configs.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = PROJECT_ROOT / "configs" / "scenarios"
THERMAL = SCENARIOS / "thermal_foraging"
FORAGING = SCENARIOS / "foraging"

_T = "connectomeppo_small_continuous2d_thermal_klinotaxis"
_H = "connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_hard350"
_MLP_T = "mlpppo_small_continuous2d_thermal_klinotaxis_t35"
_MLP_H = "mlpppo_small_continuous2d_fick_adaptive_klinotaxis_hard350_ppo_w64"

_EMMONS: dict[str, Any] = {"connectome_source": "emmons_2024_hermaphrodite"}
_GAP_ONLY: dict[str, Any] = {"wiring": "rewired_gap_junctions_only"}
# The ohmic conductance scale, calibrated before any run on untrained brains (the largest scale on the
# probe's grid keeping at least half the gap-off brain's action spread on both cells): at the built gap
# matrix's own scale, ohmic coupling averages the input away.
OHMIC_SCALE = 0.03
MODELS: dict[str, dict[str, Any]] = {
    "ohmic": {"gap_coupling": "ohmic", "gap_weight_scale": OHMIC_SCALE},
    "log": {"gap_weight_mode": "log_count"},
}
WIRINGS: dict[str, dict[str, Any]] = {"wt": {}, "gaponly": _GAP_ONLY}

# Cell -> (directory, learning parent, frozen parent, the stem's suffix).
CELLS: dict[str, tuple[Path, str, str, str]] = {
    "thermal": (THERMAL, f"{_T}_t35", f"{_T}_frozen_t35", "_t35"),
    "hard350": (FORAGING, _H, f"{_H}_frozen", ""),
}


def _children() -> dict[str, tuple[Path, str, dict[str, Any]]]:
    """Child stem -> (directory, parent stem, brain-config changes)."""
    out: dict[str, tuple[Path, str, dict[str, Any]]] = {}
    for directory, learn, frozen, suffix in CELLS.values():
        prefix = learn.removesuffix(suffix)
        for model, model_changes in MODELS.items():
            for wiring, wiring_changes in WIRINGS.items():
                delta = {**_EMMONS, **wiring_changes, **model_changes}
                out[f"{prefix}_g0a_{model}_{wiring}{suffix}"] = (directory, learn, delta)
                out[f"{prefix}_g0a_{model}_{wiring}_frozen{suffix}"] = (directory, frozen, delta)
        out[f"{prefix}_g0a_count_wt{suffix}"] = (directory, learn, dict(_EMMONS))
    out[f"{_MLP_T.removesuffix('_t35')}_g0a_frozen_t35"] = (
        THERMAL,
        _MLP_T,
        {"freeze_updates": True},
    )
    out[f"{_MLP_H}_g0a_frozen"] = (FORAGING, _MLP_H, {"freeze_updates": True})
    return out


CHILDREN: dict[str, tuple[Path, str, dict[str, Any]]] = _children()

# The pilot's arms, by cell: role -> config stem. The MLP's learning arms are its committed configs.
ARMS: dict[str, dict[str, str]] = {
    "thermal": {
        **{
            f"{m}_{w}_{k}": f"{_T}_g0a_{m}_{w}{'_frozen' if k == 'frozen' else ''}_t35"
            for m in MODELS
            for w in WIRINGS
            for k in ("learn", "frozen")
        },
        "count_wt_learn": f"{_T}_g0a_count_wt_t35",
        "mlp_learn": _MLP_T,
        "mlp_frozen": f"{_MLP_T.removesuffix('_t35')}_g0a_frozen_t35",
    },
    "hard350": {
        **{
            f"{m}_{w}_{k}": f"{_H}_g0a_{m}_{w}{'_frozen' if k == 'frozen' else ''}"
            for m in MODELS
            for w in WIRINGS
            for k in ("learn", "frozen")
        },
        "count_wt_learn": f"{_H}_g0a_count_wt",
        "mlp_learn": _MLP_H,
        "mlp_frozen": f"{_MLP_H}_g0a_frozen",
    },
}


def path_of(stem: str) -> Path:
    """Where a pilot config lives: a generated child's directory, or a committed config's."""
    if stem in CHILDREN:
        return CHILDREN[stem][0] / f"{stem}.yml"
    for directory in (THERMAL, FORAGING):
        if (directory / f"{stem}.yml").is_file():
            return directory / f"{stem}.yml"
    msg = f"no config named {stem}"
    raise FileNotFoundError(msg)


def derive(child: str) -> tuple[Path, str]:
    """Build one config: its parent with its listed brain changes."""
    directory, parent, changes = CHILDREN[child]
    data = yaml.safe_load((directory / f"{parent}.yml").read_text())
    data["brain"]["config"].update(changes)
    header = (
        f"# G.0-A selection-pilot arm: {child}.\n#\n# Delta from {parent}.yml, and nothing else "
        "changes:\n"
        + "".join(f"#   brain.config.{k}: {v}\n" for k, v in changes.items())
        + "#\n# Generated by scripts/campaigns/generate_gap_substrate_configs.py; re-checked through\n"
        "# the real config loader by test_gap_substrate_configs.py.\n#\n"
    )
    return directory / f"{child}.yml", header + yaml.safe_dump(data, sort_keys=False)


def main(argv: list[str] | None = None) -> int:
    """CLI: write every missing config, and report what was already there."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.parse_args(argv)
    written = 0
    for child in CHILDREN:
        path, text = derive(child)
        if not path.exists():
            path.write_text(text)
            written += 1
    print(f"wrote {written} configs; kept {len(CHILDREN) - written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
