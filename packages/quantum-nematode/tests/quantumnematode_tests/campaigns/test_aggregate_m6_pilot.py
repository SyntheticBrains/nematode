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
    assert spec is not None and spec.loader is not None
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
    """A seed where F1 < 40% × F0 SHALL fail the gate even if F2/F3 pass.

    F1=0.35 < 0.40 × F0=1.0 → fail. Even though the cascade is monotone
    and F2/F3 pass their own ratio thresholds against the same F0, the
    F1 check is independent and binding.
    """
    mod = _load_aggregator_module()
    retention = {
        ("tei_on", 42, 0): 1.0,
        ("tei_on", 42, 1): 0.35,
        ("tei_on", 42, 2): 0.30,  # ≥ 0.25 × F0 → would pass standalone
        ("tei_on", 42, 3): 0.20,  # ≥ 0.15 × F0 → would pass standalone
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
    """``build_retention_table`` SHALL average choice indices across episodes per (arm, seed, gen)."""
    mod = _load_aggregator_module()
    rows = [
        {"arm": "tei_on", "seed": "42", "generation": "0", "choice_index": "0.8"},
        {"arm": "tei_on", "seed": "42", "generation": "0", "choice_index": "0.6"},
        # mean = 0.7
        {"arm": "tei_on", "seed": "42", "generation": "1", "choice_index": "0.4"},
        # single ep, mean = 0.4
        {"arm": "tei_off", "seed": "42", "generation": "0", "choice_index": "0.5"},
    ]
    table = mod.build_retention_table(rows)
    assert table[("tei_on", 42, 0)] == 0.7
    assert table[("tei_on", 42, 1)] == 0.4
    assert table[("tei_off", 42, 0)] == 0.5
