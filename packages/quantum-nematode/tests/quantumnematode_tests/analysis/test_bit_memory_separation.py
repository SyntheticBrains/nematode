"""Tests for the bit-memory separation analysis harness (metric + verdict logic)."""

import sys
from pathlib import Path

import pytest

# The analysis script lives in scripts/analysis/ and imports a sibling helper module, so put
# that directory on the path before importing it.
_ANALYSIS_DIR = Path(__file__).resolve().parents[5] / "scripts" / "analysis"
sys.path.insert(0, str(_ANALYSIS_DIR))

import bit_memory_separation as bms  # noqa: E402


def _out_file(tmp_path: Path, rewards: list[float]) -> Path:
    """Write a synthetic run .out with one 'Run:' line per episode reward."""
    lines = [
        f"Run: {i}   Status: FAILED  Reason: bit_memory_completed Steps: 221    Reward: {r:.2f}  "
        for i, r in enumerate(rewards, 1)
    ]
    out = tmp_path / "run.out"
    out.write_text("\n".join(lines))
    return out


def test_plateau_tail_takes_final_quarter(tmp_path):
    """cue-match = reward / num_responses, averaged over the final quarter of episodes."""
    # 8 episodes; final quarter = last 2 (rewards 18, 18 -> 0.9 each).
    out = _out_file(tmp_path, [2, 4, 8, 12, 14, 16, 18, 18])
    assert bms._plateau_tail_cue_match(out, num_responses=20) == pytest.approx(0.9)


def test_plateau_tail_missing_file_is_none(tmp_path):
    """A missing run .out yields None (the seed is dropped, not counted as zero)."""
    assert bms._plateau_tail_cue_match(tmp_path / "nope.out", num_responses=20) is None


def _arm(base: float) -> dict[int, float]:
    # 8 paired seeds with small variation around `base` (enough samples for a one-sided
    # Wilcoxon to reach q < 0.05 when a memory arm clearly beats the MLP).
    return {s: base + 0.01 * ((s % 3) - 1) for s in range(1, 9)}


def test_verdict_separation_when_memory_arms_beat_mlp_at_chance():
    """Memory arms clearing the threshold + significantly beating an at-chance MLP -> separation."""
    table = {
        "mlpppo": _arm(0.50),
        "lstmppo": _arm(0.92),
        "cfcppo": _arm(0.90),
        "transformerppo": _arm(0.91),
        "connectomeppo": _arm(0.50),
    }
    out: dict = {}
    bms.analyse(table, out)
    assert out["verdict"]["separated"] is True
    assert set(out["verdict"]["separating_arms"]) == {"lstmppo", "cfcppo", "transformerppo"}


def test_verdict_null_when_all_at_chance():
    """All arms at chance -> null verdict (the comparison cannot resolve working memory)."""
    table = {
        "mlpppo": _arm(0.50),
        "lstmppo": _arm(0.52),
        "cfcppo": _arm(0.49),
        "transformerppo": _arm(0.51),
        "connectomeppo": _arm(0.50),
    }
    out: dict = {}
    bms.analyse(table, out)
    assert out["verdict"]["separated"] is False
    assert out["verdict"]["separating_arms"] == []
