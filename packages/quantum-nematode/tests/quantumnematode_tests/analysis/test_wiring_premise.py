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


# The harder variants -- the registered saturation remedy, and the arms the campaign actually ran.
# Each differs from its committed base by `target_foods_to_collect` plus its own arm key(s).
T20_KEY = "environment.foraging.target_foods_to_collect"
BUDGET_KEYS = {"max_steps", "satiety.satiety_gain_per_food"}
# The budget the V.3 calibration froze. The declared grid was {150, 250, 350}, run on disjoint pilot
# seeds: 150 and 250 censored the primary metric (no arm reached a 30% full-clear rate) and their
# configs are removed so no unregistered budget can be run; their records stay under the logbook's
# `pilot/`. 350 is the only grid point inside the band.
HARD_STEPS = 350
# The V.3 arms at the frozen budget. Each differs from the committed `_t20` base by the budget keys
# plus its own arm key.
CONFIG_PAIRS_HARD = [
    (f"_hard{HARD_STEPS}{suffix}", HARD_STEPS, keys)
    for suffix, keys in (
        ("", {}),
        ("_rewired_null", {"brain.config.wiring": "rewired_degree_preserving"}),
        ("_frozen", {"brain.config.freeze_updates": True}),
        (
            "_rewired_null_frozen",
            {
                "brain.config.wiring": "rewired_degree_preserving",
                "brain.config.freeze_updates": True,
            },
        ),
    )
]
CONFIG_PAIRS_T20 = [
    (base, f"{suffix}_t20", keys)
    for base in (
        "foraging/connectomeppo_small_continuous2d_fick_adaptive_klinotaxis",
        "thermal_foraging/connectomeppo_small_continuous2d_thermal_klinotaxis",
    )
    for suffix, keys in (
        ("", {}),
        ("_rewired_null", {"brain.config.wiring": "rewired_degree_preserving"}),
        ("_frozen", {"brain.config.freeze_updates": True}),
        (
            "_rewired_null_frozen",
            {
                "brain.config.wiring": "rewired_degree_preserving",
                "brain.config.freeze_updates": True,
            },
        ),
    )
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


@pytest.mark.parametrize(("base", "suffix", "expected"), CONFIG_PAIRS_T20)
def test_t20_variant_differs_by_the_target_and_its_arm_keys(base, suffix, expected):
    """Each harder-variant arm differs from its base by the target plus its own key(s).

    These are the arms the registered campaign ran, so a stray key here would change what the
    published result measured.
    """
    base_cfg = _flat(yaml.safe_load((CONFIG_DIR / f"{base}.yml").read_text()))
    variant = _flat(yaml.safe_load((CONFIG_DIR / f"{base}{suffix}.yml").read_text()))
    sentinel = object()
    differing = {
        key
        for key in set(base_cfg) | set(variant)
        if base_cfg.get(key, sentinel) != variant.get(key, sentinel)
    }
    assert differing == set(expected) | {T20_KEY}
    assert variant[T20_KEY] == 20
    assert "brain.config.rewire_seed" not in variant
    for key, value in expected.items():
        assert variant[key] == value


@pytest.mark.parametrize(("suffix", "steps", "expected"), CONFIG_PAIRS_HARD)
def test_hard_cell_arm_differs_by_the_budget_and_its_arm_keys(suffix, steps, expected):
    """Each V.3 arm differs from the committed `_t20` base by the budget keys plus its own key(s).

    The budget is the manipulation, so a stray key here would confound the very comparison the
    change exists to make.
    """
    base_name = "foraging/connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_t20"
    variant_name = base_name.replace("_t20", f"{suffix}")
    base_cfg = _flat(yaml.safe_load((CONFIG_DIR / f"{base_name}.yml").read_text()))
    variant = _flat(yaml.safe_load((CONFIG_DIR / f"{variant_name}.yml").read_text()))
    sentinel = object()
    differing = {
        key
        for key in set(base_cfg) | set(variant)
        if base_cfg.get(key, sentinel) != variant.get(key, sentinel)
    }
    assert differing == set(expected) | BUDGET_KEYS
    assert variant["max_steps"] == steps
    assert variant["satiety.satiety_gain_per_food"] == 0.2
    assert variant[T20_KEY] == 20  # the target is inherited, not part of the manipulation
    assert "brain.config.rewire_seed" not in variant
    for key, value in expected.items():
        assert variant[key] == value


def test_the_hard_cell_is_registered_in_the_family_and_decides_its_own_campaign():
    """The V.3 cell carries four family rows, and its efficiency contrast is its own primary."""
    rows = [row for row in wp.FAMILY if row[1] == "hard_food"]
    assert [row[0] for row in rows] == ["V9", "V10", "V11", "V12"]
    assert [row[4] for row in rows] == ["primary", "gate", "gate_null", "prior"]
    assert "hard_food" in wp.PRIMARY_CELLS
    assert "klinotaxis" not in wp.PRIMARY_CELLS  # it saturates; it never decides a verdict
    assert wp.SCORED["hard_food"] == "success"
    assert wp.MIN_EFFECT["hard_food"] == 5.0


def test_the_family_correction_is_per_campaign():
    """A manifest carrying one cell corrects across its four tests; two cells across eight.

    Pinned with synthetic data rather than a committed manifest: `campaigns/*` is gitignored, so the
    committed manifests' log paths do not exist on a clean checkout.
    """
    one_cell = _cells(60.0, 35.0, 10.0, 10.0, cell="hard_food")
    rows = [row for row in wp.contrasts(one_cell) if "bh_q" in row]
    assert {row["cell"] for row in rows} == {"hard_food"}
    assert len(rows) == 4

    two_cells = dict(one_cell)
    two_cells.update(_cells(60.0, 35.0, 10.0, 10.0, cell="thermal"))
    rows_two = [row for row in wp.contrasts(two_cells) if "bh_q" in row]
    assert {row["cell"] for row in rows_two} == {"hard_food", "thermal"}
    assert len(rows_two) == 8

    # The same contrast is corrected less strictly when it is alone in its family.
    solo = next(r for r in rows if r["test"] == "V9")
    paired = next(r for r in rows_two if r["test"] == "V9")
    assert solo["bh_q"] <= paired["bh_q"]


def test_the_deciding_verdict_is_serialised_for_a_primary_cell():
    """A primary cell's record carries the efficiency verdict, with the peak one beside it.

    This is the case the V.3 campaign actually produced: the hard food-only cell reads
    `below_min_effect` on the peak axis and `specific_wiring_efficiency` on the efficiency axis, and
    a consumer reading the record's verdict must get the one that decides the campaign.
    """
    entry = {"verdict": "below_min_effect", "axis": "peak", "peak_verdict": "below_min_effect"}
    wp.record_deciding_verdict(entry, "hard_food", {"verdict": "specific_wiring_efficiency"})
    assert entry["verdict"] == "specific_wiring_efficiency"
    assert entry["axis"] == "efficiency"
    assert entry["peak_verdict"] == "below_min_effect"
    assert entry["efficiency_verdict"] == "specific_wiring_efficiency"


def test_a_non_primary_cell_keeps_its_peak_verdict_as_the_deciding_one():
    """The klinotaxis cell is not a primary, so its peak verdict stands and both are recorded."""
    entry = {"verdict": "saturated", "axis": "peak", "peak_verdict": "saturated"}
    wp.record_deciding_verdict(entry, "klinotaxis", {"verdict": "degree_statistics"})
    assert entry["verdict"] == "saturated"
    assert entry["axis"] == "peak"
    assert entry["efficiency_verdict"] == "degree_statistics"


def test_analyse_serialises_both_verdicts_when_the_axes_disagree(tmp_path):
    """End to end: the two axes disagree and the record carries both, keyed unambiguously."""
    cells = _cells(60.0, 58.0, 10.0, 10.0, cell="hard_food")  # significant, under the 5-point bar
    manifest = _efficiency_manifest(tmp_path, wild_cross=40, rewired_cross=200)
    manifest.write_text(manifest.read_text().replace("thermal ", "hard_food "))

    out: dict = {}
    wp.analyse(cells, out, manifest)
    entry = out["verdicts"]["hard_food"]
    assert entry["peak_verdict"] == "below_min_effect"
    assert entry["axis"] == "efficiency"
    assert (
        entry["verdict"] == entry["efficiency_verdict"] == out["efficiency"]["hard_food"]["verdict"]
    )
    assert entry["verdict"] != entry["peak_verdict"]


def test_crossing_rate_counts_only_seeds_that_reached_the_threshold():
    """A seed censored at the horizon does not count as having crossed."""
    report = {
        "horizon_episodes": 3000,
        "per_seed": {
            "wild_type": {
                "1": {"episodes_to_30pct_success": 300.0},
                "2": {"episodes_to_30pct_success": 3000.0},
            },
            "rewired_null": {
                "1": {"episodes_to_30pct_success": 500.0},
                "2": {"episodes_to_30pct_success": 900.0},
            },
        },
    }
    assert wp.crossing_rate(report, "wild_type") == pytest.approx(0.5)
    assert wp.crossing_rate(report, "rewired_null") == pytest.approx(1.0)
    assert pytest.approx(0.8) == wp.CROSSING_FLOOR


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
    """Each test in the manifest carries a q computed over that family before any is read."""
    cells = _cells(60.0, 35.0, 10.0, 10.0)
    cells.update(_cells(40.0, 40.0, 10.0, 10.0, cell="thermal"))
    rows = wp.contrasts(cells)
    assert len(rows) == len(wp.FAMILY) == 12  # three cells x four tests
    scored = [row for row in rows if "bh_q" in row]
    assert len(scored) == 8  # the two cells in this manifest; the third is incomplete
    assert {row["cell"] for row in scored} == {"klinotaxis", "thermal"}


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


def _series_out(directory: Path, name: str, cross_at: int, total: int = 400) -> Path:
    """Write a run whose rolling full-clear rate crosses the threshold at `cross_at`."""
    statuses = ["FAILED"] * cross_at + ["SUCCESS"] * (total - cross_at)
    lines = [
        f"Run: {i}   Status: {s:<7} Reason: foraging Steps: 500    Eaten: "
        f"{20 if s == 'SUCCESS' else 2}/20  "
        for i, s in enumerate(statuses, 1)
    ]
    out = directory / name
    out.write_text("\n".join(lines))
    return out


def _efficiency_manifest(tmp_path: Path, wild_cross: int, rewired_cross: int) -> Path:
    """Write a manifest: two paired PPO seeds per arm, plus a frozen arm to be ignored."""
    lines = []
    for seed in (1, 2):
        wild = _series_out(tmp_path, f"wt{seed}.out", wild_cross)
        rewired = _series_out(tmp_path, f"rn{seed}.out", rewired_cross)
        lines.append(f"thermal wt_ppo {seed} {wild}")
        lines.append(f"thermal rn_ppo {seed} {rewired}")
        lines.append(f"thermal wt_frozen {seed} {wild}")
    manifest = tmp_path / "m.txt"
    manifest.write_text("\n".join(lines))
    return manifest


def test_efficiency_contrast_reads_only_the_ppo_arms(tmp_path):
    """The efficiency axis pairs the two PPO arms; the frozen arms are not part of that contrast."""
    manifest = _efficiency_manifest(tmp_path, wild_cross=40, rewired_cross=200)
    report = wp.efficiency_contrast(manifest, "thermal", tmp_path)
    assert report is not None
    assert report["n_paired_seeds"] == 2
    assert (
        report["metrics"]["episodes_to_30pct_success"]["wild_mean"]
        < (report["metrics"]["episodes_to_30pct_success"]["rewired_mean"])
    )


def test_efficiency_contrast_is_none_for_a_cell_with_no_ppo_arms(tmp_path):
    """A cell absent from the manifest yields no efficiency report rather than an error."""
    manifest = _efficiency_manifest(tmp_path, wild_cross=40, rewired_cross=200)
    assert wp.efficiency_contrast(manifest, "klinotaxis", tmp_path) is None


def test_efficiency_gain_is_measured_against_the_registered_minimum(tmp_path):
    """The time-to-competence gain is reported as a fraction of the null's own time."""
    manifest = _efficiency_manifest(tmp_path, wild_cross=100, rewired_cross=200)
    report = wp.efficiency_contrast(manifest, "thermal", tmp_path)
    assert report is not None
    # The rolling window offsets both crossings equally, so the gain is large and positive here.
    assert report["episodes_to_competence_gain"] > wp.MIN_EFFICIENCY_GAIN
    assert report["meets_min_effect"]
    assert report["min_gain"] == wp.MIN_EFFICIENCY_GAIN


def _efficiency_report(verdict: str, wild: float, rewired: float) -> dict:
    """Build a minimal report of the shape `apply_min_effect` consumes."""
    return {
        "verdict": verdict,
        "metrics": {
            "episodes_to_30pct_success": {
                "wild_mean": wild,
                "rewired_mean": rewired,
                "wild_minus_rewired_oriented": rewired - wild,
            },
        },
    }


def test_a_significant_efficiency_result_under_the_minimum_is_downgraded():
    """`specific_wiring_efficiency` under the registered minimum becomes `below_min_effect`.

    Exercises the production branch itself, so the rule cannot pass here while the harness applies
    a different one.
    """
    # 5% faster: significant by the harness's own rule, under the registered minimum.
    small = wp.apply_min_effect(_efficiency_report("specific_wiring_efficiency", 95.0, 100.0))
    assert small["episodes_to_competence_gain"] == pytest.approx(0.05)
    assert not small["meets_min_effect"]
    assert small["verdict"] == "below_min_effect"

    # 40% faster: over the minimum, so the verdict stands.
    large = wp.apply_min_effect(_efficiency_report("specific_wiring_efficiency", 60.0, 100.0))
    assert large["meets_min_effect"]
    assert large["verdict"] == "specific_wiring_efficiency"

    # A non-significant verdict is never promoted by clearing the minimum.
    null = wp.apply_min_effect(_efficiency_report("degree_statistics", 60.0, 100.0))
    assert null["meets_min_effect"]
    assert null["verdict"] == "degree_statistics"


def test_efficiency_contrast_applies_the_minimum_effect_rule(tmp_path):
    """The contrast path reports the gain and the minimum through the same helper."""
    manifest = _efficiency_manifest(tmp_path, wild_cross=40, rewired_cross=200)
    report = wp.efficiency_contrast(manifest, "thermal", tmp_path)
    assert report is not None
    assert report["min_gain"] == wp.MIN_EFFICIENCY_GAIN
    assert report["meets_min_effect"] == (
        report["episodes_to_competence_gain"] >= wp.MIN_EFFICIENCY_GAIN
    )
    assert report["verdict"] in {
        "degree_statistics",
        "specific_wiring_efficiency",
        "below_min_effect",
    }


def test_primary_cells_are_those_the_pilots_showed_are_not_saturated():
    """057's amendment made the thermal cell a primary; V.3 adds the hard food-only cell.

    The klinotaxis cell is never one: both wirings clear 100% of it.
    """
    assert frozenset({"thermal", "hard_food"}) == wp.PRIMARY_CELLS
