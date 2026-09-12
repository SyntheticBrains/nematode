"""Tests for the wiring-premise harness.

Covers the verdict order, the registered clauses, and the commensurability claim -- that this
harness and the 034 control read the same contrast the same way.
"""

import sys
from pathlib import Path

import pytest
import yaml

# The analysis scripts import sibling modules, so put that directory on the path first. Locate it
# by walking up to the repo root rather than a hardcoded parent index.
_root = Path(__file__).resolve()
while _root != _root.parent and not (_root / "scripts" / "analysis").is_dir():
    _root = _root.parent
_analysis_dir = _root / "scripts" / "analysis"
if not _analysis_dir.is_dir():  # fail fast rather than an opaque ModuleNotFoundError later
    msg = f"could not locate scripts/analysis walking up from {Path(__file__).resolve()}"
    raise RuntimeError(msg)
sys.path.insert(0, str(_analysis_dir))

import connectome_structure_controls as csc  # noqa: E402  # pyright: ignore[reportMissingImports]
import wiring_premise as wp  # noqa: E402  # pyright: ignore[reportMissingImports]

CONFIG_DIR = _root / "configs" / "scenarios"

# (base config, variant suffix, the keys the variant is allowed to change)
CONFIG_PAIRS = [
    (
        "foraging/connectomeppo_small_continuous2d_fick_adaptive_klinotaxis",
        "_rewired_null",
        {"wiring": "rewired_degree_preserving"},
    ),
    (
        "foraging/connectomeppo_small_continuous2d_fick_adaptive_klinotaxis",
        "_frozen",
        {"freeze_updates": True},
    ),
    (
        "foraging/connectomeppo_small_continuous2d_fick_adaptive_klinotaxis",
        "_rewired_null_frozen",
        {"wiring": "rewired_degree_preserving", "freeze_updates": True},
    ),
    (
        "thermal_foraging/connectomeppo_small_continuous2d_thermal_klinotaxis",
        "_rewired_null",
        {"wiring": "rewired_degree_preserving"},
    ),
    (
        "thermal_foraging/connectomeppo_small_continuous2d_thermal_klinotaxis",
        "_frozen",
        {"freeze_updates": True},
    ),
    (
        "thermal_foraging/connectomeppo_small_continuous2d_thermal_klinotaxis",
        "_rewired_null_frozen",
        {"wiring": "rewired_degree_preserving", "freeze_updates": True},
    ),
]


def _flat(mapping: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in mapping.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flat(value, path + "."))
        else:
            flat[path] = value
    return flat


@pytest.mark.parametrize(("base", "suffix", "expected"), CONFIG_PAIRS)
def test_variant_differs_by_exactly_the_intended_keys(base, suffix, expected):
    """Each arm differs from its base by the registered key(s) and nothing else.

    A pair that diverged on any other key would make its contrast measure two changes at once.
    """
    base_cfg = _flat(yaml.safe_load((CONFIG_DIR / f"{base}.yml").read_text()))
    variant = _flat(yaml.safe_load((CONFIG_DIR / f"{base}{suffix}.yml").read_text()))
    sentinel = object()
    differing = {
        key
        for key in set(base_cfg) | set(variant)
        if base_cfg.get(key, sentinel) != variant.get(key, sentinel)
    }
    assert differing == {f"brain.config.{key}" for key in expected}
    for key, value in expected.items():
        assert variant[f"brain.config.{key}"] == value


@pytest.mark.parametrize(("base", "suffix", "_expected"), CONFIG_PAIRS)
def test_variant_leaves_rewire_seed_unset(base, suffix, _expected):
    """`rewire_seed` stays unset so the rewiring RNG derives from the run seed and arms pair."""
    variant = _flat(yaml.safe_load((CONFIG_DIR / f"{base}{suffix}.yml").read_text()))
    assert "brain.config.rewire_seed" not in variant


def _out_file(directory: Path, name: str, clears: int, total: int, foods: int = 10) -> Path:
    """Write a synthetic run .out whose final quarter has the given full-clear count."""
    statuses = ["FAILED"] * (total - clears) + ["SUCCESS"] * clears
    lines = [
        f"Run: {i}   Status: {s:<7} Reason: foraging Steps: 800    Eaten: "
        f"{foods if s == 'SUCCESS' else 3}/10  "
        for i, s in enumerate(statuses, 1)
    ]
    out = directory / name
    out.write_text("\n".join(lines))
    return out


def _cells(
    wt_ppo: float,
    rn_ppo: float,
    wt_frozen: float,
    rn_frozen: float,
    cell: str = "klinotaxis",
) -> dict:
    """Build the loaded structure directly: {cell: {arm: {seed: (success, foods)}}}."""

    def arm(base: float) -> dict[int, tuple[float, float]]:
        # Small per-seed variation, enough for a one-sided Wilcoxon to reach q < 0.05.
        return {s: (base + 0.5 * ((s % 3) - 1), base / 10.0) for s in range(1, 9)}

    return {
        cell: {
            "wt_ppo": arm(wt_ppo),
            "rn_ppo": arm(rn_ppo),
            "wt_frozen": arm(wt_frozen),
            "rn_frozen": arm(rn_frozen),
        },
    }


def _verdict(cells: dict, cell: str = "klinotaxis") -> dict:
    return wp.verdict(cells, wp.contrasts(cells), cell)


def test_gate_failure_precedes_the_contrast():
    """A cell whose wild type does not beat its own frozen floor yields `no_learning`.

    The contrast is strongly positive here, so this also pins the order: the gate is read first and
    a wiring verdict is not assigned to a cell that did not learn.
    """
    cells = _cells(wt_ppo=60.0, rn_ppo=20.0, wt_frozen=60.0, rn_frozen=20.0)
    result = _verdict(cells)
    assert result["verdict"] == "no_learning"
    assert result["gate_test"] == "V2"


def test_saturation_precedes_the_contrast():
    """Both PPO arms at the ceiling yields `saturated` with the registered remedy, not a verdict."""
    cells = _cells(wt_ppo=97.0, rn_ppo=95.0, wt_frozen=10.0, rn_frozen=10.0)
    result = _verdict(cells)
    assert result["verdict"] == "saturated"
    assert "target_foods_to_collect 20" in result["remedy"]


def test_specific_wiring_requires_the_registered_minimum_effect():
    """A significant contrast below the registered minimum is named, not `specific_wiring`."""
    big = _verdict(_cells(wt_ppo=60.0, rn_ppo=35.0, wt_frozen=10.0, rn_frozen=10.0))
    assert big["verdict"] == "specific_wiring"
    assert big["meets_min_effect"]

    small = _verdict(_cells(wt_ppo=60.0, rn_ppo=58.0, wt_frozen=10.0, rn_frozen=10.0))
    assert small["verdict"] == "below_min_effect"
    assert small["bh_q"] < wp.SIG_Q  # significant, and still licenses nothing
    assert not small["meets_min_effect"]


def test_degree_statistics_when_indistinguishable():
    """Wild type indistinguishable from its null (CI spans zero) -> `degree_statistics`."""
    result = _verdict(_cells(wt_ppo=50.0, rn_ppo=50.0, wt_frozen=10.0, rn_frozen=10.0))
    assert result["verdict"] == "degree_statistics"


def test_rewired_beats_wild_type_is_reported_in_its_own_right():
    """A null clearly ahead is named, not folded into `inconclusive`."""
    result = _verdict(_cells(wt_ppo=35.0, rn_ppo=60.0, wt_frozen=10.0, rn_frozen=10.0))
    assert result["verdict"] == "rewired_beats_wildtype"


def test_incomplete_arms_are_insufficient_seeds():
    """A cell missing an arm cannot be read, whatever the other three say."""
    cells = _cells(wt_ppo=60.0, rn_ppo=35.0, wt_frozen=10.0, rn_frozen=10.0)
    del cells["klinotaxis"]["rn_frozen"]
    result = _verdict(cells)
    assert result["verdict"] == "insufficient_seeds"
    assert result["missing_arms"] == ["rn_frozen"]


def test_thermal_cell_is_scored_on_foods():
    """The thermal cell's scored metric is mean foods -- its satiety recipe is survival-dominant."""
    assert wp.SCORED["thermal"] == "foods"
    assert wp.SCORED["klinotaxis"] == "success"
    cells = _cells(60.0, 35.0, 10.0, 10.0, cell="thermal")
    rows = {r["test"]: r for r in wp.contrasts(cells)}
    # foods are success/10 in the fixture, so the foods-scored delta is a tenth of the success one.
    assert rows["V5"]["metric"] == "foods"
    assert rows["V5"]["mean_delta"] == pytest.approx(2.5, abs=1e-9)


def test_family_is_corrected_together():
    """All eight tests carry a corrected q, computed over the whole family before any is read."""
    cells = _cells(60.0, 35.0, 10.0, 10.0)
    cells.update(_cells(40.0, 40.0, 10.0, 10.0, cell="thermal"))
    rows = wp.contrasts(cells)
    assert len(rows) == len(wp.FAMILY) == 8
    assert all("bh_q" in row for row in rows)


def test_reference_arm_is_never_read_by_a_test():
    """The MLP reference is descriptive: no registered test names it."""
    named = {arm for _, _, a, b, _ in wp.FAMILY for arm in (a, b)}
    assert named.isdisjoint(wp.REFERENCE_ARMS)
    assert named == set(wp.TESTED_ARMS)


def test_manifest_rejects_unknown_cell_or_arm(tmp_path):
    """A typo'd cell or arm raises rather than silently removing an arm from a paired test."""
    out = _out_file(tmp_path, "run.out", clears=2, total=8)
    manifest = tmp_path / "m.txt"

    manifest.write_text(f"klinotaxis wt_ppo 1 {out}")
    assert wp.load(manifest)["klinotaxis"]["wt_ppo"][1][0] == 100.0

    manifest.write_text(f"klinotaxsi wt_ppo 1 {out}")
    with pytest.raises(wp.ManifestError, match="unknown cell"):
        wp.load(manifest)

    manifest.write_text(f"klinotaxis wt_pp0 1 {out}")
    with pytest.raises(wp.ManifestError, match="unknown arm"):
        wp.load(manifest)


def test_manifest_rejects_duplicate_entries(tmp_path):
    """A duplicated (cell, arm, seed) raises rather than overwriting a scored run."""
    out = _out_file(tmp_path, "run.out", clears=2, total=8)
    manifest = tmp_path / "m.txt"
    manifest.write_text(f"klinotaxis wt_ppo 1 {out}\nklinotaxis wt_ppo 1 {out}")
    with pytest.raises(wp.ManifestError, match="duplicate entry"):
        wp.load(manifest)


def test_agrees_with_the_034_control_on_the_same_contrast():
    """The commensurability claim, tested: both harnesses read one contrast the same way.

    The 034 control and this harness call the same metric and statistics layers, so the paired delta
    and the verdict must match on identical per-seed values.
    """
    for wild, rewired, expected in [
        (60.0, 35.0, "specific_wiring"),
        (50.0, 50.0, "degree_statistics"),
        (35.0, 60.0, "rewired_beats_wildtype"),
    ]:
        cells = _cells(wild, rewired, 10.0, 10.0)
        mine = _verdict(cells)

        legacy_out: dict = {}
        csc.analyse(
            {
                "wild_type": {s: v[0] for s, v in cells["klinotaxis"]["wt_ppo"].items()},
                "rewired_null": {s: v[0] for s, v in cells["klinotaxis"]["rn_ppo"].items()},
            },
            legacy_out,
        )

        assert mine["verdict"] == expected
        assert legacy_out["verdict"]["verdict"] == expected
        assert mine["mean_delta"] == pytest.approx(legacy_out["verdict"]["mean_delta"])
