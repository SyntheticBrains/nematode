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
) -> None:
    """Materialise a synthetic CoevolutionLoop session dir on disk.

    Writes the minimum subset of files the aggregator reads:
    `prey/lineage.csv`, `predator/lineage.csv`,
    `champion_history.json`, `generality_probe.csv`.
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
    """`--root` with no seed dirs (and no session-dir fallback) returns 1."""
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
