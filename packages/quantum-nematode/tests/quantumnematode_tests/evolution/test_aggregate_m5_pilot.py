"""Unit tests for the M5 co-evolution aggregator.

The script lives at ``scripts/campaigns/aggregate_m5_pilot.py`` (outside
the package; imported via ``importlib`` so the tests don't need a
sys.path hack — same pattern as ``test_aggregate_baldwin_retry.py``).

Coverage:

- `_block_elite_series` reads `champion_history` records and returns a
  k_block_index-ordered float series.
- `_probe_matrix` reshapes generality_probe.csv rows into a
  (gens x opponents) matrix; NaN-handles deferred-body rows; filters
  by side.
- `_aggregate_verdict` returns GO / PIVOT / STOP / INCONCLUSIVE per
  the design.md D6 rule (≥2 firing → GO, 1 → PIVOT, 0 → STOP, no
  seeds → INCONCLUSIVE).
- `_seed_metrics` returns `gate_fires=True` for synthetic series with
  known cycling or escalation; `False` for flat / short series.
- End-to-end smoke: build a synthetic per-seed dir matching the
  CoevolutionLoop output layout, invoke the aggregator's `main()`,
  verify summary.md + verdict.csv + plots are written.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[5]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "campaigns" / "aggregate_m5_pilot.py"


def _load_script_module() -> Any:
    """Import the aggregator script as a module so we can call its helpers."""
    spec = importlib.util.spec_from_file_location("aggregate_m5_pilot", SCRIPT_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        msg = f"Failed to load spec for {SCRIPT_PATH}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def aggregator() -> Any:
    """Module-scoped fixture loading the aggregator script as a module."""
    return _load_script_module()


# ---------------------------------------------------------------------------
# Series + probe-matrix construction
# ---------------------------------------------------------------------------


class TestBlockEliteSeries:
    """`_block_elite_series` extracts a k_block_index-ordered fitness array."""

    def test_empty_history_yields_empty_array(self, aggregator: Any) -> None:
        """Empty champion_history SHALL produce a 0-length array."""
        out = aggregator._block_elite_series([])
        assert out.shape == (0,)

    def test_records_sorted_by_k_block_index(self, aggregator: Any) -> None:
        """Out-of-order k_block_index records SHALL be sorted ascending."""
        history = [
            {"k_block_index": 2, "fitness": 0.5},
            {"k_block_index": 0, "fitness": 0.1},
            {"k_block_index": 1, "fitness": 0.3},
        ]
        out = aggregator._block_elite_series(history)
        np.testing.assert_array_almost_equal(out, [0.1, 0.3, 0.5])

    def test_missing_fitness_yields_nan(self, aggregator: Any) -> None:
        """Records without a `fitness` key SHALL yield NaN at that slot."""
        history = [
            {"k_block_index": 0, "fitness": 0.5},
            {"k_block_index": 1},  # no fitness key
        ]
        out = aggregator._block_elite_series(history)
        assert out[0] == 0.5
        assert np.isnan(out[1])


class TestProbeMatrix:
    """`_probe_matrix` reshapes CSV rows into a (gens x opponents) matrix."""

    def test_empty_input_returns_empty_matrix(self, aggregator: Any) -> None:
        """Empty probe-rows input SHALL return an empty 2-D array."""
        out = aggregator._probe_matrix([], side_filter="prey")
        assert out.shape == (0, 0)

    def test_filters_by_side(self, aggregator: Any) -> None:
        """`side_filter` SHALL select only rows whose `side` column matches."""
        rows = [
            {"generation": "5", "side": "prey", "opponent_index": "0", "fitness": "0.5"},
            {"generation": "5", "side": "predator", "opponent_index": "0", "fitness": "0.7"},
        ]
        prey = aggregator._probe_matrix(rows, side_filter="prey")
        predator = aggregator._probe_matrix(rows, side_filter="predator")
        assert prey.shape == (1, 1)
        assert prey[0, 0] == 0.5
        assert predator.shape == (1, 1)
        assert predator[0, 0] == 0.7

    def test_reshapes_into_gens_by_opponents(self, aggregator: Any) -> None:
        """Rows SHALL reshape into (gens, opponents) ordered by both keys."""
        rows = [
            {"generation": "10", "side": "prey", "opponent_index": "0", "fitness": "0.4"},
            {"generation": "10", "side": "prey", "opponent_index": "1", "fitness": "0.5"},
            {"generation": "20", "side": "prey", "opponent_index": "0", "fitness": "0.6"},
            {"generation": "20", "side": "prey", "opponent_index": "1", "fitness": "0.7"},
        ]
        out = aggregator._probe_matrix(rows, side_filter="prey")
        assert out.shape == (2, 2)
        np.testing.assert_array_almost_equal(out, [[0.4, 0.5], [0.6, 0.7]])

    def test_nan_fitness_is_preserved(self, aggregator: Any) -> None:
        """Deferred-body probe writes 'nan' string; matrix should preserve as np.nan."""
        rows = [
            {"generation": "5", "side": "prey", "opponent_index": "0", "fitness": "nan"},
            {"generation": "5", "side": "prey", "opponent_index": "1", "fitness": "0.5"},
        ]
        out = aggregator._probe_matrix(rows, side_filter="prey")
        assert np.isnan(out[0, 0])
        assert out[0, 1] == 0.5

    def test_duplicate_row_keeps_last_and_warns(
        self,
        aggregator: Any,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Duplicate (gen, opp_idx) pair SHALL keep the LAST fitness and emit a warning.

        Probe CSV under cadence < K_per_block writes multiple rows at
        the same `side.generation` count. With NaN fitness everywhere
        (deferred-body case) the collision is harmless. Once the probe
        body is wired, real fitness values from the same generation
        would be silently dropped without this warning. The behaviour
        is keep-LAST (matches `setdefault(...)[k] = v` semantic).
        """
        rows = [
            {"generation": "5", "side": "prey", "opponent_index": "0", "fitness": "0.3"},
            {"generation": "5", "side": "prey", "opponent_index": "0", "fitness": "0.7"},
        ]
        with caplog.at_level("WARNING", logger="aggregate_m5_pilot"):
            out = aggregator._probe_matrix(rows, side_filter="prey")
        assert out.shape == (1, 1)
        # Keep-LAST: the second 0.7 wins over the first 0.3.
        assert out[0, 0] == 0.7
        # Warning fired once with the collision count + sample.
        assert any("duplicate" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# Cross-side tie-breaker for verdict.csv slot selection
# ---------------------------------------------------------------------------


class TestPickSide:
    """`_pick_side` selects which side's values populate the verdict.csv slot."""

    def test_only_prey_fires_returns_prey(self, aggregator: Any) -> None:
        """When only prey fires, return prey EVEN IF predator's p-value is smaller."""
        prey = {"cycling_detected": True, "p_value": 0.04, "dominant_period": 8}
        predator = {"cycling_detected": False, "p_value": 0.01, "dominant_period": 99}
        out = aggregator._pick_side(
            prey,
            predator,
            p_key="p_value",
            detected_key="cycling_detected",
        )
        assert out is prey

    def test_only_predator_fires_returns_predator(self, aggregator: Any) -> None:
        """When only predator fires, return predator regardless of prey's p-value."""
        prey = {"cycling_detected": False, "p_value": 0.01, "dominant_period": 99}
        predator = {"cycling_detected": True, "p_value": 0.04, "dominant_period": 8}
        out = aggregator._pick_side(
            prey,
            predator,
            p_key="p_value",
            detected_key="cycling_detected",
        )
        assert out is predator

    def test_both_fire_smaller_pvalue_wins(self, aggregator: Any) -> None:
        """When both fire, prefer the side with the smaller p-value (more decisive)."""
        prey = {"cycling_detected": True, "p_value": 0.04, "dominant_period": 8}
        predator = {"cycling_detected": True, "p_value": 0.01, "dominant_period": 5}
        out = aggregator._pick_side(
            prey,
            predator,
            p_key="p_value",
            detected_key="cycling_detected",
        )
        assert out is predator

    def test_neither_fires_smaller_pvalue_wins_for_cosmetics(self, aggregator: Any) -> None:
        """When neither fires, smaller-p-value side wins (cosmetic; both rows are 0)."""
        prey = {"cycling_detected": False, "p_value": 0.5, "dominant_period": None}
        predator = {"cycling_detected": False, "p_value": 0.3, "dominant_period": None}
        out = aggregator._pick_side(
            prey,
            predator,
            p_key="p_value",
            detected_key="cycling_detected",
        )
        assert out is predator

    def test_tie_on_pvalue_returns_first(self, aggregator: Any) -> None:
        """Equal p-values (both fire) SHALL return the first arg (prey by convention)."""
        prey = {"cycling_detected": True, "p_value": 0.05, "dominant_period": 8}
        predator = {"cycling_detected": True, "p_value": 0.05, "dominant_period": 5}
        out = aggregator._pick_side(
            prey,
            predator,
            p_key="p_value",
            detected_key="cycling_detected",
        )
        assert out is prey


# ---------------------------------------------------------------------------
# Wall-time summary
# ---------------------------------------------------------------------------


class TestWalltimeSummary:
    """`_walltime_summary` reduces walltime CSV rows to per-seed totals."""

    def test_empty_rows_returns_nan_summary(self, aggregator: Any) -> None:
        """No walltime rows SHALL return a NaN-shaped summary (older runs)."""
        out = aggregator._walltime_summary([])
        assert np.isnan(out["mean_eval_wall_seconds"]["prey"])
        assert np.isnan(out["mean_eval_wall_seconds"]["predator"])
        assert np.isnan(out["mean_gen_wall_seconds"]["prey"])
        assert np.isnan(out["mean_gen_wall_seconds"]["predator"])
        assert np.isnan(out["total_run_wall_seconds"])
        assert out["parallel_workers_used"] == 0
        assert out["n_eval_rows"] == 0
        assert out["n_gen_rows"] == 0

    def test_per_side_means_and_total(self, aggregator: Any) -> None:
        """Mean eval/gen walls SHALL be per-side; total sums per-gen rows across both sides."""
        rows = [
            {
                "scope": "evaluation",
                "side": "prey",
                "generation": "0",
                "index": "0",
                "parallel_workers": "1",
                "wall_seconds": "1.0",
            },
            {
                "scope": "evaluation",
                "side": "prey",
                "generation": "0",
                "index": "1",
                "parallel_workers": "1",
                "wall_seconds": "3.0",
            },
            {
                "scope": "generation",
                "side": "prey",
                "generation": "0",
                "index": "2",
                "parallel_workers": "1",
                "wall_seconds": "5.0",
            },
            {
                "scope": "evaluation",
                "side": "predator",
                "generation": "0",
                "index": "0",
                "parallel_workers": "1",
                "wall_seconds": "10.0",
            },
            {
                "scope": "generation",
                "side": "predator",
                "generation": "0",
                "index": "1",
                "parallel_workers": "1",
                "wall_seconds": "12.0",
            },
        ]
        out = aggregator._walltime_summary(rows)
        # Prey eval mean = (1+3)/2 = 2.0; predator eval mean = 10.0.
        assert out["mean_eval_wall_seconds"]["prey"] == 2.0
        assert out["mean_eval_wall_seconds"]["predator"] == 10.0
        # Per-gen: prey has one row (5.0); predator has one row (12.0).
        assert out["mean_gen_wall_seconds"]["prey"] == 5.0
        assert out["mean_gen_wall_seconds"]["predator"] == 12.0
        # Total = sum of all per-gen rows = 5 + 12 = 17.0.
        assert out["total_run_wall_seconds"] == 17.0
        assert out["parallel_workers_used"] == 1
        assert out["n_eval_rows"] == 3  # 2 prey + 1 predator
        assert out["n_gen_rows"] == 2

    def test_modal_parallel_workers_when_split(self, aggregator: Any) -> None:
        """`parallel_workers_used` SHALL be the modal value across rows."""
        rows = [
            {
                "scope": "evaluation",
                "side": "prey",
                "generation": "0",
                "index": str(i),
                "parallel_workers": "4",
                "wall_seconds": "1.0",
            }
            for i in range(3)
        ] + [
            {
                "scope": "evaluation",
                "side": "prey",
                "generation": "1",
                "index": "0",
                "parallel_workers": "1",
                "wall_seconds": "1.0",
            },
        ]
        out = aggregator._walltime_summary(rows)
        # 3 rows at parallel_workers=4 vs 1 at parallel_workers=1 → modal=4.
        assert out["parallel_workers_used"] == 4

    def test_modal_parallel_workers_tie_prefers_1(self, aggregator: Any) -> None:
        """Tied modal values SHALL prefer 1 (conservative sequential interpretation)."""
        rows = [
            {
                "scope": "evaluation",
                "side": "prey",
                "generation": "0",
                "index": "0",
                "parallel_workers": "4",
                "wall_seconds": "1.0",
            },
            {
                "scope": "evaluation",
                "side": "prey",
                "generation": "1",
                "index": "0",
                "parallel_workers": "1",
                "wall_seconds": "1.0",
            },
        ]
        out = aggregator._walltime_summary(rows)
        # Tie (1 vs 4, both freq=1); 1 is among the tied → prefer 1.
        assert out["parallel_workers_used"] == 1

    def test_modal_parallel_workers_tie_without_1_picks_smallest(
        self,
        aggregator: Any,
    ) -> None:
        """Tied modal values without 1 in the tie SHALL pick the smallest deterministically."""
        rows = [
            {
                "scope": "evaluation",
                "side": "prey",
                "generation": "0",
                "index": "0",
                "parallel_workers": "4",
                "wall_seconds": "1.0",
            },
            {
                "scope": "evaluation",
                "side": "prey",
                "generation": "1",
                "index": "0",
                "parallel_workers": "8",
                "wall_seconds": "1.0",
            },
        ]
        out = aggregator._walltime_summary(rows)
        # Tie (4 vs 8, both freq=1); 1 absent → smallest tied wins.
        assert out["parallel_workers_used"] == 4

    def test_main_walltime_summary_emitted(self, aggregator: Any, tmp_path: Path) -> None:
        """Aggregator main() SHALL emit walltime_summary.csv alongside verdict.csv."""
        root = tmp_path / "campaign"
        seed42_session = root / "seed-42" / "session-1"
        _build_synthetic_session(
            seed42_session,
            prey_series=[0.5] * 5,
            predator_series=[0.5] * 5,
            walltime_rows=[
                ("evaluation", "prey", 0, 0, 1, 1.5),
                ("generation", "prey", 0, 1, 1, 1.5),
                ("evaluation", "predator", 0, 0, 1, 2.5),
                ("generation", "predator", 0, 1, 1, 2.5),
            ],
        )
        out = tmp_path / "aggregate"
        argv = [
            "aggregate_m5_pilot.py",
            "--root",
            str(root),
            "--output-dir",
            str(out),
            "--log-level",
            "WARNING",
        ]
        with patch.object(sys, "argv", argv):
            rc = aggregator.main()
        assert rc == 0
        assert (out / "walltime_summary.csv").is_file()
        with (out / "walltime_summary.csv").open() as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 1
        assert rows[0]["seed"] == "42"
        assert rows[0]["mean_eval_wall_seconds_prey"] == "1.500000"
        assert rows[0]["mean_eval_wall_seconds_predator"] == "2.500000"
        assert rows[0]["total_run_wall_seconds"] == "4.000000"
        assert rows[0]["parallel_workers_used"] == "1"
        # Reconciliation table also lands in summary.md.
        summary = (out / "summary.md").read_text()
        assert "## Wall-time reconciliation" in summary
        assert "| 42 | 1 | 1.5000 | 2.5000 |" in summary


# ---------------------------------------------------------------------------
# Verdict gate
# ---------------------------------------------------------------------------


class TestAggregateVerdict:
    """`_aggregate_verdict` per design.md D6: ≥2 → GO, 1 → PIVOT, 0 → STOP."""

    def test_zero_seeds_inconclusive(self, aggregator: Any) -> None:
        """No seeds at all SHALL return INCONCLUSIVE rather than STOP."""
        verdict, fires, total = aggregator._aggregate_verdict([])
        assert verdict == "INCONCLUSIVE"
        assert fires == 0
        assert total == 0

    def test_zero_fires_stop(self, aggregator: Any) -> None:
        """Zero firing seeds out of >0 total SHALL return STOP."""
        verdict, fires, total = aggregator._aggregate_verdict(
            [{"gate_fires": False}, {"gate_fires": False}],
        )
        assert verdict == "STOP"
        assert fires == 0
        assert total == 2

    def test_one_fire_pivot(self, aggregator: Any) -> None:
        """Exactly one firing seed SHALL return PIVOT."""
        verdict, fires, total = aggregator._aggregate_verdict(
            [{"gate_fires": True}, {"gate_fires": False}, {"gate_fires": False}],
        )
        assert verdict == "PIVOT"
        assert fires == 1
        assert total == 3

    def test_two_fires_go(self, aggregator: Any) -> None:
        """Two or more firing seeds SHALL return GO."""
        verdict, fires, total = aggregator._aggregate_verdict(
            [
                {"gate_fires": True},
                {"gate_fires": True},
                {"gate_fires": False},
                {"gate_fires": False},
            ],
        )
        assert verdict == "GO"
        assert fires == 2
        assert total == 4


# ---------------------------------------------------------------------------
# Per-seed metrics on synthetic series
# ---------------------------------------------------------------------------


class TestSeedMetrics:
    """`_seed_metrics` fires the gate on synthetic series with known signals."""

    def _build_metrics(  # noqa: PLR0913 — kw-only knobs map 1:1 to _seed_metrics defaults
        self,
        aggregator: Any,
        prey_series: np.ndarray,
        predator_series: np.ndarray,
        prey_probe: np.ndarray | None = None,
        predator_probe: np.ndarray | None = None,
        cycling_lag_range: tuple[int, int] = (3, 15),
        escalation_gen_window: tuple[int, int] = (5, 30),
        p_threshold: float = 0.05,
    ) -> dict[str, Any]:
        """Thin wrapper around `_seed_metrics` with sensible defaults for tests."""
        return aggregator._seed_metrics(
            prey_series=prey_series,
            predator_series=predator_series,
            prey_probe=prey_probe if prey_probe is not None else np.empty((0, 0), dtype=np.float64),
            predator_probe=predator_probe
            if predator_probe is not None
            else np.empty((0, 0), dtype=np.float64),
            cycling_lag_range=cycling_lag_range,
            escalation_gen_window=escalation_gen_window,
            p_threshold=p_threshold,
        )

    def test_flat_series_no_gate_fire(self, aggregator: Any) -> None:
        """Flat (constant) prey + predator series SHALL NOT fire the gate."""
        flat = np.full(40, 0.5, dtype=np.float64)
        metrics = self._build_metrics(aggregator, flat, flat)
        assert not metrics["gate_fires"]

    def test_monotone_ramp_fires_escalation(self, aggregator: Any) -> None:
        """Linear ramp + small noise (gens 0..34) SHALL fire the escalation gate."""
        rng = np.random.default_rng(seed=42)
        ramp = np.linspace(0.0, 1.0, 35) + rng.normal(0.0, 0.02, size=35)
        flat = np.full(35, 0.5, dtype=np.float64)
        metrics = self._build_metrics(aggregator, ramp, flat)
        assert metrics["escalation_detected"]
        assert metrics["gate_fires"]

    def test_pure_sine_fires_cycling(self, aggregator: Any) -> None:
        """Pure sine at period 8 SHALL fire the cycling gate."""
        t = np.arange(40)
        sine = np.sin(2 * np.pi * t / 8.0)
        flat = np.full(40, 0.5, dtype=np.float64)
        metrics = self._build_metrics(aggregator, sine, flat)
        assert metrics["cycling_detected"]
        assert metrics["gate_fires"]

    def test_short_series_no_fire(self, aggregator: Any) -> None:
        """Series shorter than the lag-range upper bound SHALL NOT fire any gate."""
        # Below the lag-range upper bound and below the escalation
        # window low-end → both metrics return non-detected.
        short = np.array([0.0, 0.1, 0.2], dtype=np.float64)
        metrics = self._build_metrics(aggregator, short, short)
        assert not metrics["gate_fires"]


# ---------------------------------------------------------------------------
# End-to-end smoke: synthetic per-seed dir, invoke main(), check artefacts
# ---------------------------------------------------------------------------


def _build_synthetic_session(
    session_dir: Path,
    *,
    prey_series: list[float],
    predator_series: list[float],
    probe_rows: list[tuple[int, str, int, float]] | None = None,
    walltime_rows: list[tuple[str, str, int, int, int, float]] | None = None,
) -> None:
    """Materialise a synthetic CoevolutionLoop session dir on disk.

    Writes the minimum subset of files the aggregator reads:
    `prey/lineage.csv`, `predator/lineage.csv`,
    `champion_history.json`, `generality_probe.csv`, `walltime.csv`.

    `walltime_rows` schema: `(scope, side, generation, index,
    parallel_workers, wall_seconds)` matching `_record_walltime` output.
    """
    (session_dir / "prey").mkdir(parents=True, exist_ok=True)
    (session_dir / "predator").mkdir(parents=True, exist_ok=True)
    # Lineage CSVs are not strictly required by the aggregator's
    # current code path (it reads champion_history.json for the
    # block-elite series). Write a header so any future code that
    # reads them doesn't crash.
    for side in ("prey", "predator"):
        with (session_dir / side / "lineage.csv").open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["generation", "child_id", "parent_ids", "fitness", "brain_type", "inherited_from"],
            )

    champion = {
        "k_block_index": max(len(prey_series), len(predator_series)),
        "prey": [
            {
                "genome_id": f"prey-{i:03d}",
                "generation": i,
                "k_block_index": i,
                "fitness": float(f),
                "params": [],
            }
            for i, f in enumerate(prey_series)
        ],
        "predator": [
            {
                "genome_id": f"pred-{i:03d}",
                "generation": i,
                "k_block_index": i,
                "fitness": float(f),
                "params": [],
            }
            for i, f in enumerate(predator_series)
        ],
    }
    (session_dir / "champion_history.json").write_text(json.dumps(champion, indent=2))

    with (session_dir / "generality_probe.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["generation", "side", "opponent_index", "fitness"])
        if probe_rows:
            for gen, side, opp_idx, fitness in probe_rows:
                writer.writerow([gen, side, opp_idx, f"{fitness:.6f}"])

    with (session_dir / "walltime.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["scope", "side", "generation", "index", "parallel_workers", "wall_seconds"],
        )
        if walltime_rows:
            for scope, side, gen, index, workers, wall in walltime_rows:
                writer.writerow([scope, side, gen, index, workers, f"{wall:.6f}"])


def test_main_end_to_end_synthetic(aggregator: Any, tmp_path: Path) -> None:
    """Aggregator main() against a synthetic 2-seed root produces all artefacts."""
    root = tmp_path / "campaign"
    out = tmp_path / "aggregate"

    rng = np.random.default_rng(seed=42)
    # Seed 42: monotone ramp → escalation should fire.
    seed42_session = root / "seed-42" / "session-1"
    _build_synthetic_session(
        seed42_session,
        prey_series=list(np.linspace(0.0, 1.0, 35) + rng.normal(0.0, 0.02, size=35)),
        predator_series=[0.5] * 35,
    )
    # Seed 43: flat → no gate.
    seed43_session = root / "seed-43" / "session-1"
    _build_synthetic_session(
        seed43_session,
        prey_series=[0.5] * 35,
        predator_series=[0.5] * 35,
    )

    # Invoke main() with patched argv.
    argv = [
        "aggregate_m5_pilot.py",
        "--root",
        str(root),
        "--output-dir",
        str(out),
        "--log-level",
        "WARNING",
    ]
    with patch.object(sys, "argv", argv):
        rc = aggregator.main()
    assert rc == 0

    # Verify artefacts.
    assert (out / "summary.md").is_file()
    assert (out / "verdict.csv").is_file()
    summary = (out / "summary.md").read_text()
    # Per-seed firing: seed 42 fires (escalation), seed 43 doesn't.
    # Aggregate verdict = PIVOT (1 of 2).
    assert "**Verdict:** PIVOT" in summary
    assert "1 of 2" in summary

    with (out / "verdict.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    seed42_row = next(r for r in rows if r["seed"] == "42")
    seed43_row = next(r for r in rows if r["seed"] == "43")
    assert seed42_row["gate_fires"] == "1"
    assert seed42_row["escalation_detected"] == "1"
    assert seed43_row["gate_fires"] == "0"


def test_main_empty_root_errors(aggregator: Any, tmp_path: Path) -> None:
    """`--root` with no seed dirs and no session-dir fallback SHALL return 1.

    Distinct from the "zero seeds resolved" INCONCLUSIVE case below
    (`test_main_zero_seeds_resolved_yields_inconclusive`) — that case
    has a seed-<N>/ structure but the per-seed dirs are empty/unreadable.
    Here `--root` has nothing the discovery logic recognises at all.
    """
    root = tmp_path / "empty"
    root.mkdir()
    out = tmp_path / "aggregate"
    argv = [
        "aggregate_m5_pilot.py",
        "--root",
        str(root),
        "--output-dir",
        str(out),
        "--log-level",
        "WARNING",
    ]
    with patch.object(sys, "argv", argv):
        rc = aggregator.main()
    assert rc == 1
    assert not out.exists()


def test_main_zero_seeds_resolved_yields_inconclusive(
    aggregator: Any,
    tmp_path: Path,
) -> None:
    """Empty `seed-N/` dirs SHALL yield INCONCLUSIVE summary + empty verdict.csv.

    Per the co-evolution capability's "Verdict Aggregation Across
    Seeds" scenario: zero resolvable seeds produces INCONCLUSIVE
    (distinct from STOP, which is a substantive null over >=1 seeds).
    Aggregator emits summary.md + an empty-body verdict.csv anyway so
    downstream tooling can detect the case without grepping logs.
    """
    root = tmp_path / "campaign"
    # Two seed-<N>/ dirs that are EMPTY (no session subdirs) — discovery
    # finds them (so layout 1 is selected) but `_resolve_session_dir`
    # returns None for each, leaving `per_seed_rows` empty.
    (root / "seed-42").mkdir(parents=True)
    (root / "seed-43").mkdir(parents=True)
    out = tmp_path / "aggregate"
    argv = [
        "aggregate_m5_pilot.py",
        "--root",
        str(root),
        "--output-dir",
        str(out),
        "--log-level",
        "WARNING",
    ]
    with patch.object(sys, "argv", argv):
        rc = aggregator.main()
    assert rc == 0
    assert (out / "summary.md").is_file()
    assert (out / "verdict.csv").is_file()
    summary = (out / "summary.md").read_text()
    assert "**Verdict:** INCONCLUSIVE" in summary
    assert "No seeds resolved" in summary
    # verdict.csv has the header but no per-seed rows.
    with (out / "verdict.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert rows == []


def test_main_single_session_root(aggregator: Any, tmp_path: Path) -> None:
    """Smoke driver writes `<root>/<session_id>/...` directly (no `seed-N/` layer); supported."""
    root = tmp_path / "smoke"
    session_dir = root / "session-abc"
    _build_synthetic_session(
        session_dir,
        prey_series=[0.5] * 10,
        predator_series=[0.5] * 10,
    )
    out = tmp_path / "aggregate"
    argv = [
        "aggregate_m5_pilot.py",
        "--root",
        str(root),
        "--output-dir",
        str(out),
        "--log-level",
        "WARNING",
    ]
    with patch.object(sys, "argv", argv):
        rc = aggregator.main()
    assert rc == 0
    assert (out / "summary.md").is_file()
    summary = (out / "summary.md").read_text()
    # Single-seed fallback labels the seed -1.
    assert "Seed -1" in summary
