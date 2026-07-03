"""Tests for the associative-memory separation harness (metric + split + verdict logic)."""

import sys
from pathlib import Path

import pytest

# The analysis script lives in scripts/analysis/ and imports a sibling helper module, so put
# that directory on the path before importing it.
_ANALYSIS_DIR = Path(__file__).resolve().parents[5] / "scripts" / "analysis"
sys.path.insert(0, str(_ANALYSIS_DIR))

import associative_memory_separation as ams  # noqa: E402  # pyright: ignore[reportMissingImports]


def _out_file(
    tmp_path: Path,
    rewards: list[float],
    splits: list[tuple[float, float]] | None = None,
) -> Path:
    """Write a synthetic .out: a Run line (+ optional AssocMemory split) per episode."""
    lines: list[str] = []
    for i, r in enumerate(rewards, 1):
        if splits is not None:
            rev, non = splits[i - 1]
            lines.append(
                f"AssocMemory: accuracy={r / 20:.4f} reversal={rev:.4f} "
                f"non_reversal={non:.4f} responses=20",
            )
        lines.append(
            f"Run: {i}   Status: FAILED  Reason: associative_memory_completed Steps: 239    "
            f"Reward: {r:.2f}  ",
        )
    out = tmp_path / "run.out"
    out.write_text("\n".join(lines))
    return out


def test_overall_accuracy_takes_final_quarter(tmp_path):
    """Accuracy = reward / num_responses, averaged over the final quarter of episodes."""
    out = _out_file(tmp_path, [2, 4, 8, 12, 14, 16, 18, 18])  # final quarter = last 2 -> 0.9
    assert ams._overall_accuracy(out, num_responses=20) == pytest.approx(0.9)


def test_split_accuracy_parses_the_assocmemory_line(tmp_path):
    """The reversal / non_reversal split is parsed from the printed AssocMemory line."""
    out = _out_file(tmp_path, [16, 20], splits=[(0.6, 0.9), (1.0, 1.0)])  # final quarter = last 1
    rev, non = ams._split_accuracy(out)
    assert rev == pytest.approx(1.0)
    assert non == pytest.approx(1.0)


def test_missing_file_is_none(tmp_path):
    """A missing run .out yields None (the seed is dropped, not counted as zero)."""
    assert ams._overall_accuracy(tmp_path / "nope.out", num_responses=20) is None
    assert ams._split_accuracy(tmp_path / "nope.out") is None


def _arm(base: float) -> dict[int, float]:
    # 8 paired seeds with small variation (enough for a one-sided Wilcoxon to reach q < 0.05).
    return {s: base + 0.01 * ((s % 3) - 1) for s in range(1, 9)}


def test_verdict_separation_when_update_arms_beat_mlp_at_chance():
    """Update-capable arms clearing the threshold + beating an at-chance MLP -> separation."""
    overall = {
        "mlpppo": _arm(0.50),
        "lstmppo": _arm(0.95),
        "transformerppo": _arm(0.93),
        "cfcppo": _arm(0.50),  # a hold-only / non-updating memory arm sits at chance too
    }
    out: dict = {}
    ams.analyse(overall, {}, out)
    assert out["verdict"]["separated"] is True
    assert set(out["verdict"]["separating_arms"]) == {"lstmppo", "transformerppo"}


def test_verdict_null_when_all_at_chance():
    """No arm clears update above the memoryless baseline -> null."""
    overall = {"mlpppo": _arm(0.50), "lstmppo": _arm(0.52), "transformerppo": _arm(0.51)}
    out: dict = {}
    ams.analyse(overall, {}, out)
    assert out["verdict"]["separated"] is False
    assert out["verdict"]["separating_arms"] == []
