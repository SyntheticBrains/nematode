"""Tests for the transgenerational pilot aggregator's decision-gate logic.

Synthetic input fixtures: deterministic-pass case, deterministic-fail
case, monotone-violation case, and the cross-seed verdict aggregator
(GO / PIVOT / STOP boundaries).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
AGGREGATOR_PATH = PROJECT_ROOT / "scripts/campaigns/aggregate_m6_pilot.py"


def _load_aggregator_module():
    """Dynamically load the aggregator script as a module.

    Scripts under ``scripts/campaigns/`` aren't on the package import path
    by default; loading via importlib avoids needing to symlink or repackage.
    """
    spec = importlib.util.spec_from_file_location("aggregate_m6_pilot", AGGREGATOR_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_decision_gate_passes_when_all_four_ratios_satisfied_and_monotone() -> None:
    """A seed with F0=1.0, F1=0.6, F2=0.36, F3=0.216 SHALL pass the gate.

    These values match the planned cascade (decay_factor=0.6: F0=1.0,
    F1=0.6, F2=0.36, F3=0.216) — every ratio test passes by construction
    and the cascade is monotone non-increasing.
    """
    mod = _load_aggregator_module()
    retention = {
        ("tei_on", 42, 0): 1.0,
        ("tei_on", 42, 1): 0.6,
        ("tei_on", 42, 2): 0.36,
        ("tei_on", 42, 3): 0.216,
    }
    result = mod.evaluate_decision_gate_one_seed(retention=retention, arm="tei_on", seed=42)
    assert result["f1_ratio_pass"] is True
    assert result["f2_ratio_pass"] is True
    assert result["f3_ratio_pass"] is True
    assert result["monotone_pass"] is True
    assert result["overall_pass"] is True


def test_decision_gate_fails_when_f1_below_40_percent() -> None:
    """A seed where F1 < 40% x F0 SHALL fail the gate even if F2/F3 pass.

    F1=0.35 < 0.40 x F0=1.0 → fail. Even though the cascade is monotone
    and F2/F3 pass their own ratio thresholds against the same F0, the
    F1 check is independent and binding.
    """
    mod = _load_aggregator_module()
    retention = {
        ("tei_on", 42, 0): 1.0,
        ("tei_on", 42, 1): 0.35,
        ("tei_on", 42, 2): 0.30,  # ≥ 0.25 x F0 → would pass standalone
        ("tei_on", 42, 3): 0.20,  # ≥ 0.15 x F0 → would pass standalone
    }
    result = mod.evaluate_decision_gate_one_seed(retention=retention, arm="tei_on", seed=42)
    assert result["f1_ratio_pass"] is False
    assert result["overall_pass"] is False


def test_decision_gate_fails_monotone_violation() -> None:
    """Monotone non-increasing SHALL be enforced — a rebound at F2 fails the gate.

    F0=1.0, F1=0.6, F2=0.7 (REBOUND), F3=0.2. All three ratio tests pass
    against F0=1.0 individually, but F2 > F1 breaks monotonicity.
    """
    mod = _load_aggregator_module()
    retention = {
        ("tei_on", 42, 0): 1.0,
        ("tei_on", 42, 1): 0.6,
        ("tei_on", 42, 2): 0.7,
        ("tei_on", 42, 3): 0.2,
    }
    result = mod.evaluate_decision_gate_one_seed(retention=retention, arm="tei_on", seed=42)
    assert result["f1_ratio_pass"] is True
    assert result["f2_ratio_pass"] is True
    assert result["f3_ratio_pass"] is True
    assert result["monotone_pass"] is False
    assert result["overall_pass"] is False


def test_decision_gate_skips_when_generation_missing() -> None:
    """When any of F0..F3 is missing the gate SHALL skip the seed.

    A run that early-stopped before completing 4 gens can't be evaluated;
    the aggregator marks ``skipped=True`` and ``overall_pass=False`` so
    the seed counts as a non-pass for the cross-seed verdict.
    """
    mod = _load_aggregator_module()
    retention = {
        ("tei_on", 42, 0): 1.0,
        ("tei_on", 42, 1): 0.6,
        ("tei_on", 42, 2): 0.36,
        # F3 missing
    }
    result = mod.evaluate_decision_gate_one_seed(retention=retention, arm="tei_on", seed=42)
    assert result["skipped"] is True
    assert result["overall_pass"] is False
    assert result["skip_reason"] == "incomplete-generations"


def test_cross_seed_verdict_go_at_two_or_more_passes() -> None:
    """``GO`` iff ≥2 seeds pass the gate."""
    mod = _load_aggregator_module()
    evaluations = [
        {"overall_pass": True},
        {"overall_pass": True},
        {"overall_pass": False},
        {"overall_pass": False},
    ]
    assert mod.aggregate_verdict(evaluations) == "GO"

    evaluations[2]["overall_pass"] = True
    assert mod.aggregate_verdict(evaluations) == "GO"


def test_cross_seed_verdict_pivot_at_exactly_one_pass() -> None:
    """``PIVOT`` iff exactly 1 seed passes."""
    mod = _load_aggregator_module()
    evaluations = [
        {"overall_pass": True},
        {"overall_pass": False},
        {"overall_pass": False},
        {"overall_pass": False},
    ]
    assert mod.aggregate_verdict(evaluations) == "PIVOT"


def test_cross_seed_verdict_stop_at_zero_passes() -> None:
    """``STOP`` iff no seed passes."""
    mod = _load_aggregator_module()
    evaluations = [
        {"overall_pass": False},
        {"overall_pass": False},
        {"overall_pass": False},
        {"overall_pass": False},
    ]
    assert mod.aggregate_verdict(evaluations) == "STOP"


def test_build_retention_table_averages_episodes_per_generation() -> None:
    """``build_retention_table`` SHALL average choice indices across eps per (arm, seed, gen)."""
    mod = _load_aggregator_module()
    rows = [
        # (tei_on, 42, gen 0): two eps at 0.8 and 0.6 → mean 0.7
        {"arm": "tei_on", "seed": "42", "generation": "0", "choice_index": "0.8"},
        {"arm": "tei_on", "seed": "42", "generation": "0", "choice_index": "0.6"},
        # (tei_on, 42, gen 1): single ep at 0.4 → mean 0.4
        {"arm": "tei_on", "seed": "42", "generation": "1", "choice_index": "0.4"},
        # (tei_off, 42, gen 0): single ep at 0.5 → mean 0.5
        {"arm": "tei_off", "seed": "42", "generation": "0", "choice_index": "0.5"},
    ]
    table = mod.build_retention_table(rows)
    assert table[("tei_on", 42, 0)] == 0.7
    assert table[("tei_on", 42, 1)] == 0.4
    assert table[("tei_off", 42, 0)] == 0.5


def test_f0_baseline_override_replaces_post_hoc_f0(tmp_path: Path) -> None:
    """When ``f0_baseline_override`` is provided the gate SHALL compare F1+ against the override.

    Regression for the M6 evaluator quirk: post-hoc F0 measures an
    UNTRAINED brain (since F0 weights are GC'd by substrate extraction),
    not the trained F0 elite the substrate was extracted from. Passing
    the training-time F0 fitness as an override gives the biologically-
    correct retention ratio.
    """
    mod = _load_aggregator_module()
    # Post-hoc retention table: F0 looks like 0.08 (untrained), F1+
    # ~0.10 (substrate-cascade). Without override, monotone fails
    # at F1 > F0. With override (F0 trained at 0.50), monotone passes
    # but ratios fail (F1=0.10 < 0.40 * 0.50 = 0.20).
    retention = {
        ("tei_on", 42, 0): 0.08,
        ("tei_on", 42, 1): 0.10,
        ("tei_on", 42, 2): 0.08,
        ("tei_on", 42, 3): 0.06,
    }
    # Without override: F1=0.10 > F0=0.08 → monotone FAIL → overall FAIL
    no_override = mod.evaluate_decision_gate_one_seed(retention=retention, arm="tei_on", seed=42)
    assert no_override["f0"] == 0.08
    assert no_override["monotone_pass"] is False

    # With override F0=0.50: F0(0.50) >= F1(0.10) >= F2(0.08) >= F3(0.06)
    # → monotone PASS. Ratios: F1=0.10 vs 0.40*0.50=0.20 → FAIL.
    override = {("tei_on", 42): 0.50}
    with_override = mod.evaluate_decision_gate_one_seed(
        retention=retention,
        arm="tei_on",
        seed=42,
        f0_baseline_override=override,
    )
    assert with_override["f0"] == 0.50
    assert with_override["monotone_pass"] is True
    assert with_override["f1_ratio_pass"] is False
    assert with_override["overall_pass"] is False


def test_f0_baseline_override_passes_when_retention_meets_envelope() -> None:
    """A seed whose F1/F2/F3 retain >= thresholds x override F0 SHALL pass under override."""
    mod = _load_aggregator_module()
    retention = {
        ("tei_on", 42, 0): 0.08,  # untrained post-hoc value (gets replaced)
        ("tei_on", 42, 1): 0.45,  # 0.45 vs 0.40 * 1.0 = 0.40 → PASS
        ("tei_on", 42, 2): 0.30,  # 0.30 vs 0.25 * 1.0 = 0.25 → PASS
        ("tei_on", 42, 3): 0.20,  # 0.20 vs 0.15 * 1.0 = 0.15 → PASS
    }
    override = {("tei_on", 42): 1.0}
    result = mod.evaluate_decision_gate_one_seed(
        retention=retention,
        arm="tei_on",
        seed=42,
        f0_baseline_override=override,
    )
    assert result["f0"] == 1.0
    assert result["f1_ratio_pass"] is True
    assert result["f2_ratio_pass"] is True
    assert result["f3_ratio_pass"] is True
    assert result["monotone_pass"] is True
    assert result["overall_pass"] is True


def test_load_f0_training_fitness_per_seed_reads_jsonl(tmp_path: Path) -> None:
    """``load_f0_training_fitness_per_seed`` SHALL extract F0 fitness from each JSONL."""
    mod = _load_aggregator_module()
    # Build a synthetic campaign-root layout:
    # <root>/tei_on/seed-42/per_gen_elites.jsonl (direct)
    # <root>/tei_off/seed-42/<session>/per_gen_elites.jsonl (nested)
    root = tmp_path / "campaign"
    direct = root / "tei_on" / "seed-42"
    direct.mkdir(parents=True)
    (direct / "per_gen_elites.jsonl").write_text(
        '{"generation": 0, "genome_id": "g0", "params": [], "fitness": 0.4624}\n'
        '{"generation": 1, "genome_id": "g1", "params": [], "fitness": 0.0}\n',
        encoding="utf-8",
    )
    nested = root / "tei_off" / "seed-42" / "20260516_133000_abc"
    nested.mkdir(parents=True)
    (nested / "per_gen_elites.jsonl").write_text(
        '{"generation": 0, "genome_id": "g0", "params": [], "fitness": 0.3500}\n',
        encoding="utf-8",
    )

    result = mod.load_f0_training_fitness_per_seed(root)
    assert result == {("tei_on", 42): 0.4624, ("tei_off", 42): 0.35}


def test_load_f0_training_fitness_per_seed_skips_missing(tmp_path: Path) -> None:
    """``load_f0_training_fitness_per_seed`` SHALL skip seeds missing the JSONL artefact."""
    mod = _load_aggregator_module()
    root = tmp_path / "campaign"
    (root / "tei_on" / "seed-42").mkdir(parents=True)
    # No per_gen_elites.jsonl in seed-42

    result = mod.load_f0_training_fitness_per_seed(root)
    assert result == {}
