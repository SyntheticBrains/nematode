"""M.1(b)'s placebo-paired reading on synthetic runs: drawdown, the guard, the rule, failed runs.

Covers the remedy selection of `add-collapse-remedies` (design Decision B): each remedy against a
placebo re-roll of the same seeds, paired by seed, on drawdown with a learning-speed guard; a failed
run counts against the remedy; a failed placebo leaves the round unreadable.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_root = Path(__file__).resolve()
while _root != _root.parent and not (_root / "scripts" / "analysis").is_dir():
    _root = _root.parent
sys.path.insert(0, str(_root / "scripts" / "analysis"))
sys.path.insert(0, str(_root / "scripts" / "campaigns"))

import collapse_remedies as cr  # noqa: E402  # pyright: ignore[reportMissingImports]
import generate_collapse_remedy_configs as gen  # noqa: E402  # pyright: ignore[reportMissingImports]


def _series(*segments: tuple[int, float]) -> np.ndarray:
    parts = []
    for length, rate in segments:
        idx = np.arange(length)
        parts.append((np.floor((idx + 1) * rate) > np.floor(idx * rate)).astype(float))
    return np.concatenate(parts)


def _write(where: Path, stem: str, seed: int, series: np.ndarray, *, code: int = 0) -> None:
    lines = [
        f"Run: {i + 1} Status: {'SUCCESS' if s else 'FAILED'} Eaten: {20 if s else 3}/20"
        for i, s in enumerate(series)
    ]
    (where / f"{stem}-seed{seed}.log").write_text("\n".join(lines) + "\n")
    (where / f"{stem}-seed{seed}.exit").write_text(f"{code}\n")


_HOLDS = _series((6000, 0.8))
_FALLS = _series((4500, 0.8), (1500, 0.0))


class TestStatistic:
    def test_a_held_run_has_little_drawdown(self) -> None:
        assert cr.drawdown(_HOLDS) == pytest.approx(0.0, abs=0.02)

    def test_a_collapse_draws_down_to_the_floor(self) -> None:
        assert cr.drawdown(_FALLS) == pytest.approx(0.8, abs=0.02)

    @pytest.mark.parametrize(
        ("mean", "lo", "guard_lo", "ok"),
        [
            (0.2, 0.05, 0.0, True),
            (0.04, 0.01, 0.0, False),
            (0.2, -0.01, 0.0, False),
            (0.2, 0.05, -0.06, False),
        ],
    )
    def test_the_rule(self, mean: float, lo: float, guard_lo: float, ok: bool) -> None:  # noqa: FBT001
        reduction = {"mean": mean, "ci": [lo, 1.0]}
        guard = {"mean": 0.0, "ci": [guard_lo, 0.1]}
        assert cr.passes(reduction, guard) is ok


def _round(
    tmp_path: Path, remedy: np.ndarray, *, failed_placebo: bool = False,
) -> dict[str, object]:
    for parent, seed in cr.RUNS:
        placebo = _FALLS if parent == gen.GAP_ONLY else _HOLDS
        _write(
            tmp_path, gen.stem(parent, "placebo"), seed, placebo, code=1 if failed_placebo else 0,
        )
        _write(tmp_path, gen.stem(parent, "klstop"), seed, remedy)
        _write(tmp_path, parent, seed, placebo)
    return cr.read_round("klstop", [tmp_path], [tmp_path])


class TestRound:
    def test_a_remedy_that_holds_every_run_passes(self, tmp_path: Path) -> None:
        result = _round(tmp_path, _HOLDS)
        assert result["passes"] is True
        assert result["drawdown_reduction"]["mean"] == pytest.approx(0.4, abs=0.02)  # type: ignore[index]

    def test_a_remedy_that_changes_nothing_fails(self, tmp_path: Path) -> None:
        for parent, seed in cr.RUNS:
            run = _FALLS if parent == gen.GAP_ONLY else _HOLDS
            _write(tmp_path, gen.stem(parent, "placebo"), seed, run)
            _write(tmp_path, gen.stem(parent, "klstop"), seed, run)
            _write(tmp_path, parent, seed, run)
        assert cr.read_round("klstop", [tmp_path], [tmp_path])["passes"] is False

    def test_a_remedy_that_slows_learning_fails_the_guard(self, tmp_path: Path) -> None:
        slow = _series((3000, 0.2), (3000, 0.8))
        assert _round(tmp_path, slow)["passes"] is False

    def test_a_failed_remedy_run_counts_against_it(self, tmp_path: Path) -> None:
        parent, seed = cr.RUNS[0]
        _write(tmp_path, gen.stem(parent, "klstop"), seed, _FALLS, code=1)
        reading = cr.run_reading(tmp_path / f"{gen.stem(parent, 'klstop')}-seed{seed}.log")
        assert reading["failed"] is True
        assert reading["early"] == 0.0

    def test_a_failed_placebo_leaves_the_round_unreadable(self, tmp_path: Path) -> None:
        with pytest.raises(cr.RemedyError, match="placebo"):
            _round(tmp_path, _HOLDS, failed_placebo=True)

    def test_only_a_remedy_round_is_read(self, tmp_path: Path) -> None:
        with pytest.raises(cr.RemedyError, match="not a remedy round"):
            cr.read_round("placebo", [tmp_path], [tmp_path])
