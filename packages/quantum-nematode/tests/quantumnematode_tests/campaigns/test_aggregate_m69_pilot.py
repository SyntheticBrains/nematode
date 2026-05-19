"""Tests for the M6.9+ three-arm aggregator.

Covers the spec scenarios in
``openspec/changes/add-transgenerational-memory-redesign/specs/evolution-framework/spec.md``
§ Cross-Arm Statistical Verdict (n=4 Noise-Aware) + § PR-B Trigger
Decision, plus the per-arm gate (mirrors M6) and the F0
training-time override pathway.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

PROJECT_ROOT = Path(__file__).resolve().parents[5]
AGGREGATOR_PATH = PROJECT_ROOT / "scripts/campaigns/aggregate_m69_pilot.py"


def _load_aggregator_module() -> ModuleType:
    """Dynamically load the aggregator script as a module."""
    spec = importlib.util.spec_from_file_location("aggregate_m69_pilot", AGGREGATOR_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Per-arm gate (mirrors M6 — sanity check + three-arm coverage)
# ---------------------------------------------------------------------------


def test_three_arm_aggregator_dispatches_all_three_arms() -> None:
    """``EXPECTED_ARMS`` SHALL include the three M6.9+ arms in spec order."""
    mod = _load_aggregator_module()
    assert mod.EXPECTED_ARMS == ("tei_on", "weights_only", "control")


def test_per_arm_gate_pass_at_perfect_cascade() -> None:
    """A seed with F0=1.0, F1=0.6, F2=0.36, F3=0.216 SHALL pass the per-arm gate."""
    mod = _load_aggregator_module()
    retention = {
        ("tei_on", 42, 0): 1.0,
        ("tei_on", 42, 1): 0.6,
        ("tei_on", 42, 2): 0.36,
        ("tei_on", 42, 3): 0.216,
    }
    result = mod.evaluate_decision_gate_one_seed(retention=retention, arm="tei_on", seed=42)
    assert result["overall_pass"] is True


def test_per_arm_verdict_go_at_two_or_more_passes() -> None:
    """``aggregate_per_arm_verdict`` SHALL emit GO at ≥ 2 passing seeds."""
    mod = _load_aggregator_module()
    evals = [
        {"overall_pass": True},
        {"overall_pass": True},
        {"overall_pass": False},
        {"overall_pass": False},
    ]
    assert mod.aggregate_per_arm_verdict(evals) == "GO"


def test_per_arm_verdict_pivot_at_one_pass() -> None:
    """``aggregate_per_arm_verdict`` SHALL emit PIVOT at exactly 1 passing seed."""
    mod = _load_aggregator_module()
    evals = [{"overall_pass": True}] + [{"overall_pass": False}] * 3
    assert mod.aggregate_per_arm_verdict(evals) == "PIVOT"


def test_per_arm_verdict_stop_at_zero_passes() -> None:
    """``aggregate_per_arm_verdict`` SHALL emit STOP at 0 passing seeds."""
    mod = _load_aggregator_module()
    evals = [{"overall_pass": False}] * 4
    assert mod.aggregate_per_arm_verdict(evals) == "STOP"


# ---------------------------------------------------------------------------
# Cross-arm Wilcoxon + bootstrap CI verdict logic
# ---------------------------------------------------------------------------


def _make_survival_table_for_cross_arm(
    *,
    tei_on_f1_plus: list[float],
    control_f1_plus: list[float],
    seeds: list[int],
) -> dict:
    """Build a survival table where each seed's F1/F2/F3 equal the given F1+ mean.

    Simplifies cross-arm-stats tests: the per-seed F1+ mean (averaged
    across F1, F2, F3) IS the supplied value for each arm.
    """
    table: dict[tuple[str, int, int], float] = {}
    for seed, tei_v, ctrl_v in zip(seeds, tei_on_f1_plus, control_f1_plus, strict=True):
        for gen in (1, 2, 3):
            table[("tei_on", seed, gen)] = tei_v
            table[("control", seed, gen)] = ctrl_v
    return table


def test_cross_arm_stats_computes_mean_delta() -> None:
    """``compute_cross_arm_delta_stats`` SHALL return the mean of per-seed deltas."""
    import pytest

    mod = _load_aggregator_module()
    seeds = [42, 43, 44, 45]
    survival = _make_survival_table_for_cross_arm(
        tei_on_f1_plus=[0.20, 0.22, 0.18, 0.24],  # mean 0.21
        control_f1_plus=[0.10, 0.12, 0.08, 0.14],  # mean 0.11
        seeds=seeds,
    )
    stats = mod.compute_cross_arm_delta_stats(
        survival,
        arm_a="tei_on",
        arm_b="control",
        seeds=seeds,
    )
    # Per-seed deltas are all +0.10; mean delta is +0.10 (10pp).
    assert stats["mean_delta"] == pytest.approx(0.10)
    for d in stats["per_seed_deltas"]:
        assert d == pytest.approx(0.10)


def test_cross_arm_stats_skips_seeds_with_incomplete_data() -> None:
    """When a seed's F1+ is incomplete the stats SHALL omit it from the delta list."""
    mod = _load_aggregator_module()
    survival = _make_survival_table_for_cross_arm(
        tei_on_f1_plus=[0.20, 0.22],
        control_f1_plus=[0.10, 0.12],
        seeds=[42, 43],
    )
    # Add seed 44 but only partial data (missing F3 for control).
    survival[("tei_on", 44, 1)] = 0.18
    survival[("tei_on", 44, 2)] = 0.18
    survival[("tei_on", 44, 3)] = 0.18
    survival[("control", 44, 1)] = 0.08
    survival[("control", 44, 2)] = 0.08
    # ("control", 44, 3) is missing.
    stats = mod.compute_cross_arm_delta_stats(
        survival,
        arm_a="tei_on",
        arm_b="control",
        seeds=[42, 43, 44],
    )
    assert len(stats["per_seed_deltas"]) == 2
    assert 44 in stats["skipped_seeds"]


def test_cross_arm_primary_verdict_go_when_all_four_checks_pass() -> None:
    """Primary verdict SHALL be GO when per-arm gate + Wilcoxon + delta + CI all pass."""
    mod = _load_aggregator_module()
    # tei_on per-arm: 3/4 seeds pass → GO.
    tei_on_evals = [
        {"overall_pass": True},
        {"overall_pass": True},
        {"overall_pass": True},
        {"overall_pass": False},
    ]
    # All 4 paired deltas at +0.10 → strong signal; Wilcoxon highly
    # significant; CI well above zero.
    stats = mod.compute_cross_arm_delta_stats(
        _make_survival_table_for_cross_arm(
            tei_on_f1_plus=[0.20, 0.22, 0.18, 0.24],
            control_f1_plus=[0.10, 0.12, 0.08, 0.14],
            seeds=[42, 43, 44, 45],
        ),
        arm_a="tei_on",
        arm_b="control",
        seeds=[42, 43, 44, 45],
    )
    verdict = mod.compute_cross_arm_primary_verdict(tei_on_evals, stats)
    assert verdict["verdict"] == "GO"
    assert verdict["per_arm_gate_pass"] is True
    assert verdict["wilcoxon_pass"] is True
    assert verdict["delta_pass"] is True
    assert verdict["ci_pass"] is True


def test_cross_arm_primary_verdict_stop_when_delta_below_5pp() -> None:
    """Even if Wilcoxon is significant, delta < 5pp SHALL fail the verdict."""
    mod = _load_aggregator_module()
    tei_on_evals = [{"overall_pass": True}] * 3 + [{"overall_pass": False}]
    # Small but consistent deltas: ~3pp each. Wilcoxon could still
    # fire on consistency, but delta < 5pp SHALL fail.
    stats = mod.compute_cross_arm_delta_stats(
        _make_survival_table_for_cross_arm(
            tei_on_f1_plus=[0.13, 0.13, 0.13, 0.13],
            control_f1_plus=[0.10, 0.10, 0.10, 0.10],
            seeds=[42, 43, 44, 45],
        ),
        arm_a="tei_on",
        arm_b="control",
        seeds=[42, 43, 44, 45],
    )
    verdict = mod.compute_cross_arm_primary_verdict(tei_on_evals, stats)
    assert verdict["delta_pass"] is False
    assert verdict["verdict"] == "STOP"


def test_cross_arm_primary_verdict_stop_when_per_arm_gate_fails() -> None:
    """When tei_on per-arm gate is STOP (0/4 seeds pass), primary verdict SHALL be STOP."""
    mod = _load_aggregator_module()
    # No per-arm passes → tei_on arm verdict is STOP.
    tei_on_evals = [{"overall_pass": False}] * 4
    # Otherwise strong cross-arm signal.
    stats = mod.compute_cross_arm_delta_stats(
        _make_survival_table_for_cross_arm(
            tei_on_f1_plus=[0.20, 0.22, 0.18, 0.24],
            control_f1_plus=[0.10, 0.12, 0.08, 0.14],
            seeds=[42, 43, 44, 45],
        ),
        arm_a="tei_on",
        arm_b="control",
        seeds=[42, 43, 44, 45],
    )
    verdict = mod.compute_cross_arm_primary_verdict(tei_on_evals, stats)
    assert verdict["per_arm_gate_pass"] is False
    assert verdict["verdict"] == "STOP"


def test_cross_arm_primary_verdict_stop_when_ci_overlaps_zero() -> None:
    """CI overlapping zero SHALL fail the verdict even with mean delta ≥ 5pp.

    Construct deltas with strong asymmetry (three large +deltas and
    one even-larger -delta) so the 80% bootstrap CI lo dips
    unambiguously below zero — robust to bootstrap seed/resample
    config changes.
    """
    mod = _load_aggregator_module()
    tei_on_evals = [{"overall_pass": True}] * 3 + [{"overall_pass": False}]
    # Per-seed deltas: [+0.30, +0.30, +0.30, -0.40] ⇒ mean = +0.0625,
    # but the single -40pp outlier guarantees ci_lo < 0 under any
    # plausible bootstrap config (any resample that picks that one
    # outlier ≥ ~40% of the time yields a negative mean — well over
    # the 10% lower tail of the 80% CI).
    stats = mod.compute_cross_arm_delta_stats(
        _make_survival_table_for_cross_arm(
            tei_on_f1_plus=[0.40, 0.40, 0.40, 0.00],
            control_f1_plus=[0.10, 0.10, 0.10, 0.40],
            seeds=[42, 43, 44, 45],
        ),
        arm_a="tei_on",
        arm_b="control",
        seeds=[42, 43, 44, 45],
    )
    verdict = mod.compute_cross_arm_primary_verdict(tei_on_evals, stats)
    # CI lo should be strictly negative — fail the ci_pass check.
    assert verdict["bootstrap_ci_lo"] < 0.0
    assert verdict["ci_pass"] is False
    assert verdict["verdict"] == "STOP"


def test_cross_arm_primary_verdict_indeterminate_when_full_under_powered() -> None:
    """SF10 regression: a FULL-campaign run with n_seeds < 4 SHALL emit INDETERMINATE, NOT STOP.

    Wilcoxon one-sided at n=3 has minimum achievable p = 0.125 ≥ 0.10,
    so the GO threshold is structurally unreachable. A null verdict
    under this regime cannot be distinguished from a real null — the
    operator MUST be told the run is under-powered rather than seeing
    a misleading STOP.
    """
    mod = _load_aggregator_module()
    tei_on_evals = [{"overall_pass": True}] * 2 + [{"overall_pass": False}]
    # 3-seed pair (one seed dropped — simulating a mid-campaign crash).
    stats = mod.compute_cross_arm_delta_stats(
        _make_survival_table_for_cross_arm(
            tei_on_f1_plus=[0.20, 0.22, 0.18],
            control_f1_plus=[0.10, 0.12, 0.08],
            seeds=[42, 43, 44],
        ),
        arm_a="tei_on",
        arm_b="control",
        seeds=[42, 43, 44],
    )
    verdict = mod.compute_cross_arm_primary_verdict(tei_on_evals, stats, mode="full")
    assert verdict["verdict"] == "INDETERMINATE"
    assert verdict["indeterminate_under_powered"] is True
    assert verdict["n_seeds"] == 3


def test_cross_arm_primary_verdict_stop_not_indeterminate_at_full_n_seeds() -> None:
    """SF10 boundary: FULL run, n_seeds = 4, failed verdict SHALL be STOP not INDETERMINATE."""
    mod = _load_aggregator_module()
    tei_on_evals = [{"overall_pass": False}] * 4  # per-arm gate fails
    stats = mod.compute_cross_arm_delta_stats(
        _make_survival_table_for_cross_arm(
            tei_on_f1_plus=[0.20, 0.22, 0.18, 0.24],
            control_f1_plus=[0.10, 0.12, 0.08, 0.14],
            seeds=[42, 43, 44, 45],
        ),
        arm_a="tei_on",
        arm_b="control",
        seeds=[42, 43, 44, 45],
    )
    verdict = mod.compute_cross_arm_primary_verdict(tei_on_evals, stats, mode="full")
    assert verdict["verdict"] == "STOP"
    assert verdict["indeterminate_under_powered"] is False
    assert verdict["n_seeds"] == 4


# ---------------------------------------------------------------------------
# F0 baseline override threading across the three arms
# ---------------------------------------------------------------------------


def test_f0_override_threads_through_per_arm_gate_for_all_three_arms() -> None:
    """``f0_baseline_override`` SHALL replace post-hoc F0 for all three arms.

    The per-arm gate is called once per (arm, seed); the override dict
    is shared across arms so a single (arm, seed) override fires only
    for that pair.
    """
    mod = _load_aggregator_module()
    retention = {
        ("tei_on", 42, 0): 0.08,  # untrained — gets overridden to 0.50
        ("tei_on", 42, 1): 0.30,
        ("tei_on", 42, 2): 0.20,
        ("tei_on", 42, 3): 0.10,
        ("control", 42, 0): 0.08,
        ("control", 42, 1): 0.30,
        ("control", 42, 2): 0.20,
        ("control", 42, 3): 0.10,
    }
    override = {("tei_on", 42): 0.50}
    # tei_on: override applied → F0=0.50.
    tei_on_result = mod.evaluate_decision_gate_one_seed(
        retention=retention,
        arm="tei_on",
        seed=42,
        f0_baseline_override=override,
    )
    assert tei_on_result["f0"] == 0.50
    # control: no override entry → post-hoc F0=0.08.
    control_result = mod.evaluate_decision_gate_one_seed(
        retention=retention,
        arm="control",
        seed=42,
        f0_baseline_override=override,
    )
    assert control_result["f0"] == 0.08


# ---------------------------------------------------------------------------
# Pilot pivot decision + PR-B trigger emission
# ---------------------------------------------------------------------------


def _make_survival_table_with_f0(
    *,
    arm_f0: dict[str, float],
    seeds: list[int],
) -> dict[tuple[str, int, int], float]:
    """Build a survival_table populated with per-arm F0 values for each seed.

    Used by pilot-pivot tests to differentiate the chance-floor (D6
    row 1) pivot from the inert / clean-differentiation pivots.
    """
    out: dict[tuple[str, int, int], float] = {}
    for arm, f0 in arm_f0.items():
        for seed in seeds:
            out[(arm, seed, 0)] = f0
            out[(arm, seed, 1)] = f0 * 0.5
            out[(arm, seed, 2)] = f0 * 0.3
            out[(arm, seed, 3)] = f0 * 0.2
    return out


def test_pilot_pivot_decision_emits_clean_differentiation_pivot(tmp_path: Path) -> None:
    """``_write_pilot_pivot_decision`` SHALL classify a clean signal as D6 row 6 (NO pivot)."""
    mod = _load_aggregator_module()
    out_path = tmp_path / "pilot_pivot_decision.md"
    mod._write_pilot_pivot_decision(
        pilot_observations={
            "tei_on_per_arm_verdict": "GO",
            "primary_verdict_dict": {
                "mean_delta": 0.10,
                "per_arm_gate_pass": True,
                "wilcoxon_pass": True,
            },
            "survival_table": _make_survival_table_with_f0(
                arm_f0={"tei_on": 0.55, "weights_only": 0.40, "control": 0.40},
                seeds=[42],
            ),
            "seeds": [42],
            "cross_arm_results": [
                {"arm_a": "tei_on", "arm_b": "control", "mean_delta": 0.10},
                {"arm_a": "tei_on", "arm_b": "weights_only", "mean_delta": 0.08},
            ],
        },
        path=out_path,
    )
    content = out_path.read_text(encoding="utf-8")
    assert "clean differentiation" in content
    assert "Pivot: NONE" in content
    assert "D6 row 6" in content


def test_pilot_pivot_decision_emits_substrate_inert_pivot(tmp_path: Path) -> None:
    """SHALL classify ``tei_on ≈ control`` at F1+ as substrate-inert (D6 row 2)."""
    mod = _load_aggregator_module()
    out_path = tmp_path / "pilot_pivot_decision.md"
    mod._write_pilot_pivot_decision(
        pilot_observations={
            "tei_on_per_arm_verdict": "STOP",
            "primary_verdict_dict": {
                "mean_delta": 0.005,
                "per_arm_gate_pass": False,
                "wilcoxon_pass": False,
            },
            "survival_table": _make_survival_table_with_f0(
                arm_f0={"tei_on": 0.50, "weights_only": 0.49, "control": 0.50},
                seeds=[42],
            ),
            "seeds": [42],
            "cross_arm_results": [],
        },
        path=out_path,
    )
    content = out_path.read_text(encoding="utf-8")
    assert "substrate likely inert" in content
    assert "hidden_dim" in content  # pivot suggestion
    assert "D6 row 2" in content


def test_pilot_pivot_decision_emits_chance_collapse_pivot(tmp_path: Path) -> None:
    """``_write_pilot_pivot_decision`` SHALL classify all-arms < 0.30 F0 survival as D6 row 1."""
    mod = _load_aggregator_module()
    out_path = tmp_path / "pilot_pivot_decision.md"
    mod._write_pilot_pivot_decision(
        pilot_observations={
            "tei_on_per_arm_verdict": "STOP",
            "primary_verdict_dict": {
                "mean_delta": 0.0,
                "per_arm_gate_pass": False,
                "wilcoxon_pass": False,
            },
            "survival_table": _make_survival_table_with_f0(
                arm_f0={"tei_on": 0.10, "weights_only": 0.12, "control": 0.11},
                seeds=[42],
            ),
            "seeds": [42],
            "cross_arm_results": [],
        },
        path=out_path,
    )
    content = out_path.read_text(encoding="utf-8")
    assert "chance-floor collapse" in content
    assert "D6 row 1" in content
    assert "lawn distribution" in content or "penalty_predator_contact" in content


def test_pilot_pivot_decision_emits_monotone_violated_pivot(tmp_path: Path) -> None:
    """``_write_pilot_pivot_decision`` SHALL classify F1>F0 as monotone-violated (D6 row 4)."""
    mod = _load_aggregator_module()
    out_path = tmp_path / "pilot_pivot_decision.md"
    # Build a survival_table where tei_on F1 > F0 for seed 42.
    survival_table: dict[tuple[str, int, int], float] = {
        ("tei_on", 42, 0): 0.45,
        ("tei_on", 42, 1): 0.55,  # F1 > F0 — monotone violated
        ("tei_on", 42, 2): 0.30,
        ("tei_on", 42, 3): 0.20,
        ("weights_only", 42, 0): 0.50,
        ("control", 42, 0): 0.50,
    }
    mod._write_pilot_pivot_decision(
        pilot_observations={
            "tei_on_per_arm_verdict": "STOP",
            "primary_verdict_dict": {
                "mean_delta": 0.0,
                "per_arm_gate_pass": False,
                "wilcoxon_pass": False,
            },
            "survival_table": survival_table,
            "seeds": [42],
            "cross_arm_results": [],
        },
        path=out_path,
    )
    content = out_path.read_text(encoding="utf-8")
    assert "monotone-decay violated" in content
    assert "D6 row 4" in content


def test_pilot_pivot_decision_emits_matched_by_weights_pivot(tmp_path: Path) -> None:
    """``_write_pilot_pivot_decision`` SHALL classify tei_on≈weights_only as D6 row 5."""
    mod = _load_aggregator_module()
    out_path = tmp_path / "pilot_pivot_decision.md"
    mod._write_pilot_pivot_decision(
        pilot_observations={
            "tei_on_per_arm_verdict": "GO",
            "primary_verdict_dict": {
                "mean_delta": 0.08,
                "per_arm_gate_pass": True,
                "wilcoxon_pass": True,
            },
            "survival_table": _make_survival_table_with_f0(
                arm_f0={"tei_on": 0.50, "weights_only": 0.50, "control": 0.42},
                seeds=[42],
            ),
            "seeds": [42],
            "cross_arm_results": [
                {"arm_a": "tei_on", "arm_b": "control", "mean_delta": 0.08},
                # tei_on - weights_only ≈ 0 → matched.
                {"arm_a": "tei_on", "arm_b": "weights_only", "mean_delta": 0.005},
            ],
        },
        path=out_path,
    )
    content = out_path.read_text(encoding="utf-8")
    assert "matched by weights" in content
    assert "D6 row 5" in content
    assert "PR-B" in content


def test_pr_b_trigger_emitted_on_go(tmp_path: Path) -> None:
    """``_write_pr_b_trigger`` SHALL emit a recommendation for the PR-B scaffold."""
    mod = _load_aggregator_module()
    out_path = tmp_path / "pr_b_trigger.md"
    mod._write_pr_b_trigger(out_path)
    content = out_path.read_text(encoding="utf-8")
    assert "GO" in content
    assert "add-transgenerational-memory-weights" in content
    assert "PR-B" in content


def test_m6_13_punt_note_emitted_on_stop(tmp_path: Path) -> None:
    """``_write_m6_13_punt_note`` SHALL document the null finding + M6.13 deferral."""
    mod = _load_aggregator_module()
    out_path = tmp_path / "m6_13_punt_note.md"
    mod._write_m6_13_punt_note(out_path)
    content = out_path.read_text(encoding="utf-8")
    assert "STOP" in content
    assert "M6.13" in content
    assert "not" in content.lower()  # NOT scaffolded


# ---------------------------------------------------------------------------
# F0 fitness JSONL loader (mirrors M6 — sanity check that the M6.9+
# aggregator's copy of the loader honours the same defensive parser).
# ---------------------------------------------------------------------------


def test_load_f0_training_fitness_per_seed_reads_jsonl(tmp_path: Path) -> None:
    """The loader SHALL extract F0 fitness from each seed's per_gen_elites.jsonl."""
    mod = _load_aggregator_module()
    root = tmp_path / "campaign"
    seed_dir = root / "tei_on" / "seed-42"
    seed_dir.mkdir(parents=True)
    (seed_dir / "per_gen_elites.jsonl").write_text(
        '{"generation": 0, "genome_id": "g0", "params": [], "fitness": 0.4624}\n'
        '{"generation": 1, "genome_id": "g1", "params": [], "fitness": 0.0}\n',
        encoding="utf-8",
    )
    result = mod.load_f0_training_fitness_per_seed(root, arms=["tei_on"])
    assert result == {("tei_on", 42): 0.4624}


def test_load_f0_training_fitness_per_seed_skips_malformed_fitness(tmp_path: Path) -> None:
    """The loader SHALL skip seeds whose F0 row has missing/non-finite ``fitness``."""
    mod = _load_aggregator_module()
    root = tmp_path / "campaign"
    missing = root / "tei_on" / "seed-42"
    missing.mkdir(parents=True)
    (missing / "per_gen_elites.jsonl").write_text(
        '{"generation": 0, "genome_id": "g0", "params": []}\n',
        encoding="utf-8",
    )
    nonnumeric = root / "tei_on" / "seed-43"
    nonnumeric.mkdir(parents=True)
    (nonnumeric / "per_gen_elites.jsonl").write_text(
        '{"generation": 0, "genome_id": "g0", "params": [], "fitness": "not-a-number"}\n',
        encoding="utf-8",
    )
    result = mod.load_f0_training_fitness_per_seed(root, arms=["tei_on"])
    assert result == {}
