"""Tests for the composed-mode three-arm aggregator.

Covers the spec scenarios under
``openspec/changes/add-tei-prior-on-m3/specs/evolution-framework/spec.md``:
- Cross-arm primary verdict (reframed) — verdict pair is
  ``tei_weights - weights_only`` (not the prior pure-TEI campaign's
  ``tei_on - control``).
- Pre-declared pilot pivot table — six D6 rows + deterministic
  classification.
- Frequency-prior follow-up trigger — trigger emission on GO;
  null-finding note on STOP.
- Per-arm gate (mirrors the prior pure-TEI campaign; covered for
  three-arm sanity).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

PROJECT_ROOT = Path(__file__).resolve().parents[5]
AGGREGATOR_PATH = PROJECT_ROOT / "scripts/campaigns/aggregate_m613_pilot.py"


def _load_aggregator_module() -> ModuleType:
    """Dynamically load the composed-mode aggregator script as a module."""
    spec = importlib.util.spec_from_file_location("aggregate_m613_pilot", AGGREGATOR_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Arm naming + per-arm gate (sanity)
# ---------------------------------------------------------------------------


def test_three_arm_aggregator_uses_m613_arm_names() -> None:
    """``EXPECTED_ARMS`` SHALL include the three composed-mode arms in spec order."""
    mod = _load_aggregator_module()
    assert mod.EXPECTED_ARMS == ("tei_weights", "weights_only", "control")


def test_per_arm_gate_pass_at_perfect_cascade_for_tei_weights() -> None:
    """``tei_weights`` SHALL pass the per-arm gate at the canonical retention shape."""
    mod = _load_aggregator_module()
    retention = {
        ("tei_weights", 42, 0): 1.0,
        ("tei_weights", 42, 1): 0.6,
        ("tei_weights", 42, 2): 0.36,
        ("tei_weights", 42, 3): 0.216,
    }
    result = mod.evaluate_decision_gate_one_seed(
        retention=retention,
        arm="tei_weights",
        seed=42,
    )
    assert result["overall_pass"] is True


# ---------------------------------------------------------------------------
# Cross-arm stats with the reframed pair (tei_weights - weights_only)
# ---------------------------------------------------------------------------


def _make_survival_table_for_primary_pair(
    *,
    tei_weights_f1_plus: list[float],
    weights_only_f1_plus: list[float],
    seeds: list[int],
) -> dict:
    """Build a survival table for the composed-mode primary pair (tei_weights vs weights_only).

    F0 row is also seeded (same value as F1+) so the per-arm gate
    helper can resolve a non-None F0 baseline. Per-seed F1+ mean is
    the supplied value for each arm.
    """
    table: dict[tuple[str, int, int], float] = {}
    for seed, tw_v, wo_v in zip(
        seeds,
        tei_weights_f1_plus,
        weights_only_f1_plus,
        strict=True,
    ):
        # F0 = F1+ for these synthetic tests (simplifies the cross-arm
        # delta computation; per-arm gate is unused in cross-arm tests).
        for gen in (0, 1, 2, 3):
            table[("tei_weights", seed, gen)] = tw_v
            table[("weights_only", seed, gen)] = wo_v
    return table


def test_cross_arm_stats_primary_pair_uses_tei_weights_minus_weights_only() -> None:
    """The primary pair SHALL compute ``tei_weights - weights_only`` per-seed deltas."""
    mod = _load_aggregator_module()
    seeds = [42, 43, 44, 45]
    survival = _make_survival_table_for_primary_pair(
        tei_weights_f1_plus=[0.70, 0.72, 0.68, 0.74],  # mean 0.71
        weights_only_f1_plus=[0.60, 0.62, 0.58, 0.64],  # mean 0.61
        seeds=seeds,
    )
    stats = mod.compute_cross_arm_delta_stats(
        survival,
        arm_a="tei_weights",
        arm_b="weights_only",
        seeds=seeds,
    )
    # Per-seed deltas all +0.10pp; mean delta +0.10.
    assert stats["mean_delta"] == pytest.approx(0.10)
    for d in stats["per_seed_deltas"]:
        assert d == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# Cross-arm primary verdict — reframed for composed mode
# ---------------------------------------------------------------------------


def _go_verdict_inputs(mod: ModuleType) -> tuple[list[dict], dict]:
    """Build per-arm + cross-arm inputs that satisfy ALL FOUR GO conditions."""
    perfect_eval = {"overall_pass": True}
    seed_evaluations = [perfect_eval, perfect_eval, perfect_eval, perfect_eval]
    cross_arm_stats = {
        "mean_delta": 0.085,  # 8.5pp > 5pp threshold
        "wilcoxon_p": 0.05,  # < 0.10 threshold
        "bootstrap_ci_lo": 0.04,  # > 0
        "bootstrap_ci_hi": 0.13,
        "per_seed_deltas": [0.08, 0.10, 0.07, 0.09],
    }
    return seed_evaluations, cross_arm_stats


def test_primary_verdict_go_when_all_four_checks_pass() -> None:
    """GO iff per-arm gate AND Wilcoxon AND delta AND CI all pass."""
    mod = _load_aggregator_module()
    seed_evals, stats = _go_verdict_inputs(mod)
    verdict = mod.compute_cross_arm_primary_verdict(seed_evals, stats, mode="full")
    assert verdict["verdict"] == "GO"
    assert verdict["per_arm_gate_pass"] is True
    assert verdict["wilcoxon_pass"] is True
    assert verdict["delta_pass"] is True
    assert verdict["ci_pass"] is True


def test_primary_verdict_stop_when_delta_below_5pp_threshold() -> None:
    """Mean delta < 5pp SHALL force STOP even if Wilcoxon and CI pass."""
    mod = _load_aggregator_module()
    seed_evals, stats = _go_verdict_inputs(mod)
    stats["mean_delta"] = 0.03  # 3pp < 5pp
    verdict = mod.compute_cross_arm_primary_verdict(seed_evals, stats, mode="full")
    assert verdict["verdict"] == "STOP"
    assert verdict["delta_pass"] is False


def test_primary_verdict_stop_when_per_arm_gate_fails() -> None:
    """tei_weights per-arm gate failure SHALL force STOP regardless of cross-arm stats."""
    mod = _load_aggregator_module()
    _, stats = _go_verdict_inputs(mod)
    failing_evals = [{"overall_pass": False}] * 4  # tei_weights arm verdict = STOP
    verdict = mod.compute_cross_arm_primary_verdict(failing_evals, stats, mode="full")
    assert verdict["verdict"] == "STOP"
    assert verdict["per_arm_gate_pass"] is False


def test_primary_verdict_indeterminate_when_under_powered_in_full() -> None:
    """Full mode with n<4 seeds on the primary pair SHALL emit INDETERMINATE not STOP."""
    mod = _load_aggregator_module()
    seed_evals, stats = _go_verdict_inputs(mod)
    # Under-powered: 3 seeds is below CROSS_ARM_FULL_N_SEEDS=4. Even
    # if Wilcoxon and CI passed they would be unreliable; the verdict
    # MUST surface this rather than silently STOP.
    stats["mean_delta"] = 0.03  # forces STOP without the under-powered branch
    stats["per_seed_deltas"] = [0.03, 0.03, 0.03]  # n=3
    verdict = mod.compute_cross_arm_primary_verdict(seed_evals, stats, mode="full")
    assert verdict["verdict"] == "INDETERMINATE"
    assert verdict["indeterminate_under_powered"] is True


def test_primary_verdict_stop_not_indeterminate_under_pilot_mode() -> None:
    """Pilot mode with n=1 SHALL emit STOP (NOT INDETERMINATE).

    Pilots run at n=1 by design — INDETERMINATE is reserved for
    under-powered FULL campaigns where the operator must re-run with
    missing seeds. The pilot's primary verdict artefact is
    ``pilot_pivot_decision.md``, not this raw cross-arm verdict;
    surfacing INDETERMINATE here would falsely flag the pilot as
    unfinished when the pilot was never meant to produce a definitive
    cross-arm verdict.
    """
    mod = _load_aggregator_module()
    seed_evals, stats = _go_verdict_inputs(mod)
    stats["mean_delta"] = 0.03  # forces STOP path
    stats["per_seed_deltas"] = [0.03]  # n=1 (pilot)
    verdict = mod.compute_cross_arm_primary_verdict(seed_evals, stats, mode="pilot")
    assert verdict["verdict"] == "STOP"
    assert verdict["indeterminate_under_powered"] is False


# ---------------------------------------------------------------------------
# D6 pivot-table detectors
# ---------------------------------------------------------------------------


def test_pivot_detector_substrate_inert() -> None:
    """Row 1: |Δ| < 2pp SHALL fire the substrate-inert detector."""
    mod = _load_aggregator_module()
    results = [
        {"arm_a": "tei_weights", "arm_b": "weights_only", "mean_delta": 0.01},
    ]
    assert mod._detect_substrate_inert(results) is True
    # Negative side of the band also fires (substrate inert in either direction).
    results[0]["mean_delta"] = -0.005
    assert mod._detect_substrate_inert(results) is True
    # Above the band does NOT fire.
    results[0]["mean_delta"] = 0.03
    assert mod._detect_substrate_inert(results) is False


def test_pivot_detector_substrate_interferes() -> None:
    """Row 4: Δ < -2pp SHALL fire the substrate-interferes detector."""
    mod = _load_aggregator_module()
    results = [
        {"arm_a": "tei_weights", "arm_b": "weights_only", "mean_delta": -0.05},
    ]
    assert mod._detect_substrate_interferes(results) is True
    # Just above the threshold does NOT fire (caught by inert detector instead).
    results[0]["mean_delta"] = -0.015
    assert mod._detect_substrate_interferes(results) is False


def test_pivot_detector_lamarckian_saturation() -> None:
    """Row 5: weights_only F0 ≈ F1 SHALL fire the Lamarckian-saturation detector."""
    mod = _load_aggregator_module()
    saturated_table = {
        ("weights_only", 42, 0): 0.70,
        ("weights_only", 42, 1): 0.71,  # within 2pp of F0 → saturated
    }
    assert mod._detect_lamarckian_saturation(saturated_table, [42]) is True

    headroom_table = {
        ("weights_only", 42, 0): 0.40,
        ("weights_only", 42, 1): 0.60,  # 20pp gain — clear headroom
    }
    assert mod._detect_lamarckian_saturation(headroom_table, [42]) is False


def test_pivot_detector_ppo_destabilised() -> None:
    """Row 6: tei_weights F1 < 5% SHALL fire the PPO-destabilised detector."""
    mod = _load_aggregator_module()
    collapsed_table = {("tei_weights", 42, 1): 0.02}  # F1 collapsed to ~0
    assert mod._detect_ppo_destabilised(collapsed_table, [42]) is True

    healthy_table = {("tei_weights", 42, 1): 0.50}
    assert mod._detect_ppo_destabilised(healthy_table, [42]) is False


# ---------------------------------------------------------------------------
# Pilot pivot decision — classification into the six D6 rows
# ---------------------------------------------------------------------------


def _make_pilot_observations(  # noqa: PLR0913 — orthogonal pivot-test knobs; collapsing to a config dataclass adds boilerplate without clarity
    *,
    mean_delta: float,
    tei_weights_verdict: str = "STOP",
    per_arm_gate_pass: bool = False,
    wilcoxon_pass: bool = False,
    survival_table: dict | None = None,
    seeds: list[int] | None = None,
    cross_arm_results: list[dict] | None = None,
) -> dict:
    """Build a pilot_observations dict matching _write_pilot_pivot_decision's contract."""
    if seeds is None:
        seeds = [42]
    if survival_table is None:
        # Healthy default: weights_only headroom + tei_weights non-zero F1.
        survival_table = {
            ("weights_only", 42, 0): 0.40,
            ("weights_only", 42, 1): 0.60,
            ("tei_weights", 42, 0): 0.40,
            ("tei_weights", 42, 1): 0.60,
        }
    if cross_arm_results is None:
        cross_arm_results = [
            {
                "arm_a": "tei_weights",
                "arm_b": "weights_only",
                "mean_delta": mean_delta,
            },
        ]
    return {
        "tei_weights_per_arm_verdict": tei_weights_verdict,
        "primary_verdict_dict": {
            "mean_delta": mean_delta,
            "per_arm_gate_pass": per_arm_gate_pass,
            "wilcoxon_pass": wilcoxon_pass,
        },
        "survival_table": survival_table,
        "seeds": seeds,
        "cross_arm_results": cross_arm_results,
    }


def test_pilot_pivot_emits_clean_go_at_delta_above_5pp(tmp_path: Path) -> None:
    """Row 2: tei_weights > weights_only by ≥5pp SHALL emit clean-GO pivot."""
    mod = _load_aggregator_module()
    observations = _make_pilot_observations(mean_delta=0.07)
    out = tmp_path / "pilot_pivot_decision.md"
    mod._write_pilot_pivot_decision(observations, out)
    text = out.read_text()
    assert "clean GO at K_test (D6 row 2)" in text
    assert "Proceed to full campaign at K_test only" in text


def test_pilot_pivot_emits_k_sensitivity_at_delta_3pp(tmp_path: Path) -> None:
    """Row 3: tei_weights > weights_only by 2-5pp SHALL emit K-sensitivity pivot."""
    mod = _load_aggregator_module()
    observations = _make_pilot_observations(mean_delta=0.03)
    out = tmp_path / "pilot_pivot_decision.md"
    mod._write_pilot_pivot_decision(observations, out)
    text = out.read_text()
    assert "K-sensitivity pivot (D6 row 3)" in text
    assert "rerun pilot at K=500" in text


def test_pilot_pivot_emits_substrate_inert_at_zero_delta(tmp_path: Path) -> None:
    """Row 1: |Δ| < 2pp SHALL emit substrate-inert pivot."""
    mod = _load_aggregator_module()
    observations = _make_pilot_observations(mean_delta=0.005)
    out = tmp_path / "pilot_pivot_decision.md"
    mod._write_pilot_pivot_decision(observations, out)
    text = out.read_text()
    assert "substrate inert (D6 row 1)" in text
    assert "substrate-accelerates-retraining hypothesis is falsified" in text


def test_pilot_pivot_emits_substrate_interferes_at_negative_delta(tmp_path: Path) -> None:
    """Row 4: tei_weights < weights_only (Δ < -2pp) SHALL emit substrate-interferes pivot."""
    mod = _load_aggregator_module()
    observations = _make_pilot_observations(mean_delta=-0.05)
    out = tmp_path / "pilot_pivot_decision.md"
    mod._write_pilot_pivot_decision(observations, out)
    text = out.read_text()
    assert "substrate INTERFERES with Lamarckian (D6 row 4)" in text
    assert "substrate-policy alignment" in text


def test_pilot_pivot_emits_lamarckian_saturation_pivot(tmp_path: Path) -> None:
    """Row 5: weights_only F0 ≈ F1 SHALL emit Lamarckian-saturation pivot (highest priority)."""
    mod = _load_aggregator_module()
    # Lamarckian saturated AND tei_weights also looks good — row 5 wins by branch order.
    observations = _make_pilot_observations(
        mean_delta=0.07,  # Would otherwise hit row 2
        survival_table={
            ("weights_only", 42, 0): 0.70,
            ("weights_only", 42, 1): 0.71,  # saturated
            ("tei_weights", 42, 0): 0.70,
            ("tei_weights", 42, 1): 0.78,
        },
    )
    out = tmp_path / "pilot_pivot_decision.md"
    mod._write_pilot_pivot_decision(observations, out)
    text = out.read_text()
    assert "Lamarckian saturation (D6 row 5)" in text
    assert "drop K_test to 500" in text


def test_pilot_pivot_emits_ppo_destabilised_pivot(tmp_path: Path) -> None:
    """Row 6: tei_weights F1 collapse SHALL emit PPO-destabilised pivot."""
    mod = _load_aggregator_module()
    # tei_weights F1 collapse AND negative delta — row 6 wins over row 4
    # by branch order (catastrophic failure modes match first).
    observations = _make_pilot_observations(
        mean_delta=-0.30,
        survival_table={
            ("weights_only", 42, 0): 0.40,
            ("weights_only", 42, 1): 0.60,
            ("tei_weights", 42, 0): 0.40,
            ("tei_weights", 42, 1): 0.02,  # collapse below threshold
        },
    )
    out = tmp_path / "pilot_pivot_decision.md"
    mod._write_pilot_pivot_decision(observations, out)
    text = out.read_text()
    assert "PPO destabilised (D6 row 6)" in text
    assert "substrate clamp" in text


# ---------------------------------------------------------------------------
# Frequency-prior trigger + null-finding note emission
# ---------------------------------------------------------------------------


def test_frequency_prior_trigger_emitted_on_go(tmp_path: Path) -> None:
    """``_write_m614_frequency_prior_trigger`` SHALL emit the follow-up scaffold note."""
    mod = _load_aggregator_module()
    out = tmp_path / "m614_frequency_prior_trigger.md"
    mod._write_m614_frequency_prior_trigger(out)
    text = out.read_text()
    assert "Frequency-prior-ablation trigger" in text
    assert "add-frequency-prior-ablation" in text
    # The trigger references the design choice: minimum-viable substrate.
    assert "frequency prior" in text.lower() or "frequency-prior" in text


def test_null_finding_note_emitted_on_stop(tmp_path: Path) -> None:
    """``_write_m613_null_finding_note`` SHALL document the null + close the TEI thread."""
    mod = _load_aggregator_module()
    out = tmp_path / "m613_null_finding_note.md"
    mod._write_m613_null_finding_note(out)
    text = out.read_text()
    assert "TEI-as-prior null finding" in text
    assert "**NOT** scaffolded" in text or "NOT scaffolded" in text
    assert "closes the TEI thread" in text
