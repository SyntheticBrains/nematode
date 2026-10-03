"""The wall-exclusion re-read of the chemotaxis validation: manifests, identity and movement.

Covers the realworm-behavioural-validation requirement "Wall-proximal transitions can be excluded"
as it is applied to Logbook 035, and checks the committed readings say what the logbook's note says.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_root = Path(__file__).resolve()
while _root != _root.parent and not (_root / "scripts" / "analysis").is_dir():
    _root = _root.parent
sys.path.insert(0, str(_root / "scripts" / "analysis"))

import wall_exclusion_check as wec  # noqa: E402  # pyright: ignore[reportMissingImports]

COMMITTED = _root / "docs/experiments/logbooks/supporting/035-realworm-chemotaxis-validation"
READINGS = COMMITTED / "wall-exclusion"


def _summary(verdict: str, per_seed: dict[str, float], combined: str = "PRESENT") -> dict:
    stat = {
        "verdict": verdict,
        "mean": 1.0,
        "ci_lo": 0.9,
        "ci_hi": 1.1,
        "n": len(per_seed),
        "per_seed": per_seed,
    }
    return {
        "statistics": {"klinotaxis": stat},
        "strategy_verdicts": {"klinotaxis": {"combined": combined}},
    }


class TestIdentity:
    def test_identical_values_and_verdicts(self) -> None:
        a = _summary("REPRODUCED", {"42": 0.5, "43": 0.25})
        assert wec.identity(a, a) == {"identical": True, "differences": []}

    def test_a_changed_seed_and_a_changed_verdict_are_both_named(self) -> None:
        then = _summary("REPRODUCED", {"42": 0.5})
        now = _summary("PARTIAL", {"42": 0.6})
        got = wec.identity(now, then)
        assert not got["identical"]
        assert got["differences"] == [
            "klinotaxis seed 42: 0.5 then, 0.6 now",
            "klinotaxis verdict: REPRODUCED then, PARTIAL now",
        ]


class TestMoved:
    def test_a_verdict_change_and_a_combined_change_are_reported(self) -> None:
        off = wec.verdicts(_summary("REPRODUCED", {"42": 0.5}, "PRESENT"))
        on = wec.verdicts(_summary("PARTIAL", {"42": 0.4}, "PRESENT_PARTIAL"))
        assert wec.moved(off, on) == [
            "klinotaxis: REPRODUCED -> PARTIAL",
            "klinotaxis (combined): PRESENT -> PRESENT_PARTIAL",
        ]

    def test_nothing_moves_when_verdicts_hold(self) -> None:
        off = wec.verdicts(_summary("REPRODUCED", {"42": 0.5}))
        on = wec.verdicts(_summary("REPRODUCED", {"42": 0.45}))
        assert wec.moved(off, on) == []


class TestManifests:
    def test_a_missing_run_is_refused(self, tmp_path: Path) -> None:
        (tmp_path / "logs").mkdir()
        with pytest.raises(wec.CheckError, match="no finished run"):
            wec.build_manifests(tmp_path, tmp_path / "out")

    def test_the_session_is_read_from_the_log(self, tmp_path: Path) -> None:
        log = tmp_path / "x.log"
        log.write_text("Starting\nSession ID: 20261003_213637_205a3ba8\nmore\n")
        assert wec.session_of(log) == "20261003_213637_205a3ba8"


class TestCommittedReadings:
    """The committed comparison says what Logbook 035's dated note says."""

    def test_the_comparison_matches_the_note(self) -> None:
        result = json.loads((READINGS / "comparison.json").read_text())
        arms = result["arms"]
        assert result["moved_at_primary"] is True
        assert arms["mlp"]["moved"] == {"m1.0": [], "m2.0": []}
        assert arms["connectome"]["moved"] == {"m1.0": [], "m2.0": []}
        for label in ("m1.0", "m2.0"):
            assert arms["control"]["moved"][label] == [
                "klinotaxis: REPRODUCED -> PARTIAL",
                "klinotaxis (combined): PRESENT -> PRESENT_PARTIAL",
            ]
        assert arms["connectome"]["identity_with_committed"]["identical"] is True

    def test_the_comparison_re_derives_from_the_committed_readings(self) -> None:
        assert wec.compare(READINGS, COMMITTED) == json.loads(
            (READINGS / "comparison.json").read_text(),
        )
