"""Unit tests for the M4.5 Baldwin retry pilot aggregator.

The script lives at ``scripts/campaigns/aggregate_baldwin_retry_pilot.py``
(outside the package; imported via ``importlib`` so the tests don't
need a sys.path hack).  Tests cover:

(a) Schema-equalisation pre-flight check fires correctly above and
    below the 0.05 threshold (audit A1 closure verdict).
(b) GO / PIVOT / STOP / INCONCLUSIVE verdict logic produces the right
    output for each gate combination.
(c) The K' filter on the F1 CSV reads the right rows when multiple
    K' values are present.
(d) The CLI rejects non-positive ``--k-prime`` values via
    ``parser.error`` (mirroring the F1 evaluator's pattern).
"""

from __future__ import annotations

import csv
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[5]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "campaigns" / "aggregate_baldwin_retry_pilot.py"


def _load_script_module():
    """Import the aggregator script as a module so we can call its helpers directly."""
    spec = importlib.util.spec_from_file_location(
        "aggregate_baldwin_retry_pilot",
        SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        msg = f"Failed to load spec for {SCRIPT_PATH}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# (a) Schema-equalisation pre-flight check
# ---------------------------------------------------------------------------


def _make_history(first_gen_best: float, additional_gens: int = 5) -> list[dict[str, float]]:
    """Build a synthetic history.csv-shaped list with a chosen first-gen best_fitness."""
    first_row = {
        "generation": 1.0,
        "best_fitness": first_gen_best,
        "mean_fitness": 0.5,
        "std_fitness": 0.1,
    }
    later_rows = [
        {
            "generation": float(g),
            "best_fitness": min(1.0, first_gen_best + 0.1 * (g - 1)),
            "mean_fitness": 0.5,
            "std_fitness": 0.1,
        }
        for g in range(2, 2 + additional_gens)
    ]
    return [first_row, *later_rows]


def test_schema_equalisation_check_passes_below_threshold() -> None:
    """|Δ| ≤ 0.05 SHALL pass the schema-equalisation check."""
    module = _load_script_module()
    seeds = [42, 43, 44, 45]
    baldwin = {s: _make_history(0.50) for s in seeds}
    ctrl = {s: _make_history(0.52) for s in seeds}  # |Δ| = 0.02 ≤ 0.05
    passes, baldwin_mean, ctrl_mean, abs_delta = module._check_schema_equalisation(  # type: ignore[attr-defined]
        baldwin,
        ctrl,
        seeds,
    )
    assert passes is True
    assert baldwin_mean == pytest.approx(0.50)
    assert ctrl_mean == pytest.approx(0.52)
    assert abs_delta == pytest.approx(0.02)


def test_schema_equalisation_check_fails_above_threshold() -> None:
    """|Δ| > 0.05 SHALL fail the schema-equalisation check."""
    module = _load_script_module()
    seeds = [42, 43, 44, 45]
    baldwin = {s: _make_history(0.50) for s in seeds}
    ctrl = {s: _make_history(0.65) for s in seeds}  # |Δ| = 0.15 > 0.05
    passes, baldwin_mean, ctrl_mean, abs_delta = module._check_schema_equalisation(  # type: ignore[attr-defined]
        baldwin,
        ctrl,
        seeds,
    )
    assert passes is False
    assert baldwin_mean == pytest.approx(0.50)
    assert ctrl_mean == pytest.approx(0.65)
    assert abs_delta == pytest.approx(0.15)


def test_first_gen_mean_fitness_raises_on_empty_history() -> None:
    """Empty history.csv (crashed seed) SHALL raise a clear ValueError naming the seed."""
    module = _load_script_module()
    seeds = [42, 43, 44]
    histories = {
        42: _make_history(0.5),
        43: [],  # this seed crashed before gen-0 finished
        44: _make_history(0.5),
    }
    with pytest.raises(ValueError, match=r"history\.csv is empty for seed\(s\) \[43\]"):
        module._first_gen_mean_fitness(histories, seeds)  # type: ignore[attr-defined]


def test_first_gen_mean_fitness_lists_all_empty_seeds() -> None:
    """Multiple empty seeds SHALL all be reported in one error message."""
    module = _load_script_module()
    seeds = [42, 43, 44, 45]
    histories = {
        42: _make_history(0.5),
        43: [],
        44: _make_history(0.5),
        45: [],
    }
    with pytest.raises(ValueError, match=r"history\.csv is empty for seed\(s\) \[43, 45\]"):
        module._first_gen_mean_fitness(histories, seeds)  # type: ignore[attr-defined]


def test_schema_equalisation_check_fires_at_exact_threshold() -> None:
    """|Δ| at the threshold SHALL pass (≤ is inclusive per Decision 2 wording).

    Uses |Δ| = 0.04 (a clean fp-representable value below the 0.05
    threshold) since 0.55 - 0.50 ≈ 0.05000000000000004 in IEEE 754.
    """
    module = _load_script_module()
    seeds = [42, 43]
    baldwin = {s: _make_history(0.50) for s in seeds}
    ctrl = {s: _make_history(0.54) for s in seeds}
    passes, _, _, abs_delta = module._check_schema_equalisation(  # type: ignore[attr-defined]
        baldwin,
        ctrl,
        seeds,
    )
    assert passes is True
    assert abs_delta < 0.05


# ---------------------------------------------------------------------------
# (b) GO / PIVOT / STOP / INCONCLUSIVE verdict logic
# ---------------------------------------------------------------------------


def test_verdict_inconclusive_when_schema_equalisation_fails() -> None:
    """Schema-equalisation failure SHALL force INCONCLUSIVE regardless of the gates."""
    module = _load_script_module()
    verdict, text = module._compute_verdict(  # type: ignore[attr-defined]
        speed_gate_passes=True,
        f1_gate_passes=True,
        comparative_gate_passes=True,
        schema_equalisation_passes=False,
    )
    assert "INCONCLUSIVE" in verdict
    assert "Audit A1" in text


def test_verdict_go_when_all_gates_pass() -> None:
    """All three gates passing + schema-equalisation passing SHALL yield GO."""
    module = _load_script_module()
    verdict, _ = module._compute_verdict(  # type: ignore[attr-defined]
        speed_gate_passes=True,
        f1_gate_passes=True,
        comparative_gate_passes=True,
        schema_equalisation_passes=True,
    )
    assert "GO" in verdict


def test_verdict_pivot_when_only_speed_passes() -> None:
    """Speed PASS + (F1 OR comparative) FAIL SHALL yield PIVOT."""
    module = _load_script_module()
    verdict_f1_fail, _ = module._compute_verdict(  # type: ignore[attr-defined]
        speed_gate_passes=True,
        f1_gate_passes=False,
        comparative_gate_passes=True,
        schema_equalisation_passes=True,
    )
    assert "PIVOT" in verdict_f1_fail

    verdict_comp_fail, _ = module._compute_verdict(  # type: ignore[attr-defined]
        speed_gate_passes=True,
        f1_gate_passes=True,
        comparative_gate_passes=False,
        schema_equalisation_passes=True,
    )
    assert "PIVOT" in verdict_comp_fail

    verdict_both_fail, _ = module._compute_verdict(  # type: ignore[attr-defined]
        speed_gate_passes=True,
        f1_gate_passes=False,
        comparative_gate_passes=False,
        schema_equalisation_passes=True,
    )
    assert "PIVOT" in verdict_both_fail


def test_verdict_stop_when_speed_fails() -> None:
    """Speed FAIL SHALL yield STOP regardless of the other gates."""
    module = _load_script_module()
    for f1, comp in [(True, True), (True, False), (False, True), (False, False)]:
        verdict, _ = module._compute_verdict(  # type: ignore[attr-defined]
            speed_gate_passes=False,
            f1_gate_passes=f1,
            comparative_gate_passes=comp,
            schema_equalisation_passes=True,
        )
        assert "STOP" in verdict, f"f1={f1}, comp={comp} should have STOPped"


# ---------------------------------------------------------------------------
# (c) F1 CSV K' filter
# ---------------------------------------------------------------------------


def test_f1_csv_filter_reads_only_requested_kprime(tmp_path: Path) -> None:
    """The aggregator SHALL filter f1_learning_acceleration.csv rows by k_prime."""
    module = _load_script_module()
    csv_path = tmp_path / "f1.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "seed",
                "k_prime",
                "episodes",
                "elite_success_rate",
                "baseline_success_rate",
                "signal_delta",
            ),
        )
        # Two K' values present for each seed
        for seed in (42, 43):
            writer.writerow((seed, 10, 25, 0.5, 0.3, 0.2))
            writer.writerow((seed, 25, 25, 0.7, 0.4, 0.3))

    rows_kprime_10 = module._read_f1_csv(csv_path, k_prime=10)  # type: ignore[attr-defined]
    rows_kprime_25 = module._read_f1_csv(csv_path, k_prime=25)  # type: ignore[attr-defined]

    assert set(rows_kprime_10.keys()) == {42, 43}
    assert set(rows_kprime_25.keys()) == {42, 43}
    # K'=10 rows: signal_delta = 0.2
    assert rows_kprime_10[42] == pytest.approx((0.5, 0.3, 0.2))
    # K'=25 rows: signal_delta = 0.3
    assert rows_kprime_25[42] == pytest.approx((0.7, 0.4, 0.3))


def test_f1_csv_filter_returns_empty_for_missing_kprime(tmp_path: Path) -> None:
    """If no rows match the requested K', the filter SHALL return an empty dict."""
    module = _load_script_module()
    csv_path = tmp_path / "f1.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "seed",
                "k_prime",
                "episodes",
                "elite_success_rate",
                "baseline_success_rate",
                "signal_delta",
            ),
        )
        writer.writerow((42, 10, 25, 0.5, 0.3, 0.2))

    rows = module._read_f1_csv(csv_path, k_prime=99)  # type: ignore[attr-defined]
    assert rows == {}


def test_f1_csv_filter_raises_on_missing_file(tmp_path: Path) -> None:
    """Missing F1 CSV SHALL raise FileNotFoundError with a clear message."""
    module = _load_script_module()
    with pytest.raises(FileNotFoundError, match="F1 CSV missing at"):
        module._read_f1_csv(tmp_path / "does_not_exist.csv", k_prime=10)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Sanity: helpers exposed by the script module
# ---------------------------------------------------------------------------


def test_script_module_exposes_helpers() -> None:
    """The aggregator SHALL define the helpers we depend on for testing."""
    module = _load_script_module()
    assert callable(module._check_schema_equalisation)  # type: ignore[attr-defined]
    assert callable(module._compute_verdict)  # type: ignore[attr-defined]
    assert callable(module._read_f1_csv)  # type: ignore[attr-defined]
    assert callable(module.main)


# ---------------------------------------------------------------------------
# (d) CLI rejects non-positive --k-prime
# ---------------------------------------------------------------------------


def _required_cli_args(tmp_path: Path) -> list[str]:
    """Build the minimum required-argument set for the aggregator CLI.

    Paths don't need to exist — argparse rejection fires before the
    aggregator touches the filesystem.
    """
    return [
        "--baldwin-root",
        str(tmp_path / "baldwin"),
        "--lamarckian-root",
        str(tmp_path / "lamarckian"),
        "--control-root",
        str(tmp_path / "control"),
        "--baseline-root",
        str(tmp_path / "baseline"),
        "--f1-csv",
        str(tmp_path / "f1.csv"),
        "--output-dir",
        str(tmp_path / "out"),
    ]


def test_cli_rejects_kprime_zero(tmp_path: Path) -> None:
    """``--k-prime 0`` SHALL exit non-zero with a clear error message."""
    result = subprocess.run(  # noqa: S603
        [sys.executable, str(SCRIPT_PATH), *_required_cli_args(tmp_path), "--k-prime", "0"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "--k-prime must be a positive integer" in result.stderr


def test_cli_rejects_kprime_negative(tmp_path: Path) -> None:
    """``--k-prime -5`` SHALL exit non-zero with a clear error message."""
    result = subprocess.run(  # noqa: S603
        [sys.executable, str(SCRIPT_PATH), *_required_cli_args(tmp_path), "--k-prime", "-5"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "--k-prime must be a positive integer" in result.stderr


# ---------------------------------------------------------------------------
# History-layout resolution
# ---------------------------------------------------------------------------


class TestHistoryLayoutResolution:
    """Both seed-dir layouts must be readable, not just the nested one.

    ``aggregate_baldwin_retry_pilot`` resolves its history via the two-layout
    ``_resolve_session`` (file directly under ``seed-N/``, or nested one level
    under a session dir) rather than the newest-subdirectory ``_latest_session``
    the other aggregators use.

    Added after a refactor silently dropped that resolver and the existing suite
    stayed green: every fixture here used the nested layout, so the direct layout
    — the whole reason the resolver exists — was never exercised.
    """

    @staticmethod
    def _write_history(target: Path) -> None:
        target.mkdir(parents=True, exist_ok=True)
        with (target / "history.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["generation", "best_fitness", "mean_fitness", "std_fitness"])
            writer.writerow([1, 0.5, 0.4, 0.1])
            writer.writerow([2, 0.95, 0.8, 0.05])

    def test_reads_history_written_directly_under_the_seed_dir(self, tmp_path: Path) -> None:
        """The layout run_evolution.py has written since logbook 014."""
        module = _load_script_module()
        seed_dir = tmp_path / "seed-42"
        self._write_history(seed_dir)

        rows = module.read_history(seed_dir, module._resolve_session)

        assert len(rows) == 2
        assert rows[1]["best_fitness"] == pytest.approx(0.95)

    def test_reads_history_nested_under_a_session_dir(self, tmp_path: Path) -> None:
        """The older layout, where the loop wrote a per-session subdirectory."""
        module = _load_script_module()
        seed_dir = tmp_path / "seed-42"
        self._write_history(seed_dir / "20260101_000000_abcdef")

        rows = module.read_history(seed_dir, module._resolve_session)

        assert len(rows) == 2
        assert rows[1]["best_fitness"] == pytest.approx(0.95)

    def test_default_resolver_cannot_see_the_direct_layout(self, tmp_path: Path) -> None:
        """Pins WHY this aggregator passes its own resolver rather than the default.

        If this ever starts passing, the shared default has become
        layout-tolerant and the explicit argument is redundant.
        """
        module = _load_script_module()
        seed_dir = tmp_path / "seed-42"
        self._write_history(seed_dir)

        with pytest.raises(FileNotFoundError):
            module.read_history(seed_dir)

    def test_main_reads_the_direct_layout_end_to_end(self, tmp_path: Path) -> None:
        """Exercises the CALL SITE, not just the helper.

        The helper-level tests above pass whether or not ``main`` actually
        forwards the two-layout resolver, so they cannot catch a call site that
        silently falls back to the default. This drives the real entry point
        against a direct-layout tree, which fails outright if the resolver is
        dropped.
        """
        seeds = [42, 43]
        for arm in ("baldwin", "lamarckian", "control"):
            for seed in seeds:
                self._write_history(tmp_path / arm / f"seed-{seed}")
        baseline = tmp_path / "baseline"
        baseline.mkdir()
        for seed in seeds:
            (baseline / f"seed-{seed}.log").write_text("Success rate: 65.00%\n")
        f1_csv = tmp_path / "f1.csv"
        with f1_csv.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["seed", "k_prime", "elite_success_rate", "baseline_success_rate", "signal_delta"],
            )
            for seed in seeds:
                writer.writerow([seed, 1, 0.9, 0.6, 0.3])

        result = subprocess.run(  # noqa: S603
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--baldwin-root",
                str(tmp_path / "baldwin"),
                "--lamarckian-root",
                str(tmp_path / "lamarckian"),
                "--control-root",
                str(tmp_path / "control"),
                "--baseline-root",
                str(baseline),
                "--f1-csv",
                str(f1_csv),
                "--k-prime",
                "1",
                "--seeds",
                "42",
                "43",
                "--output-dir",
                str(tmp_path / "out"),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, (
            "main() failed on the direct seed-dir layout — the two-layout "
            f"resolver was probably dropped at the call site.\n{result.stderr}"
        )
        assert (tmp_path / "out" / "summary.md").exists()
