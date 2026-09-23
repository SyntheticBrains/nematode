"""The campaign progress reader: what counts as finished, and when a watch stops.

A run is finished when the runner has written its completion marker, never because its log has
bytes in it: stderr is unbuffered, so a warning printed at load time makes a log non-empty while
the run is still going. These tests pin that, the fallback for campaigns that predate markers,
and that a watch on one campaign does not wait on another campaign's workers.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import pytest

_CAMPAIGNS = Path(__file__).resolve().parents[5] / "scripts" / "campaigns"
if str(_CAMPAIGNS) not in sys.path:
    sys.path.insert(0, str(_CAMPAIGNS))

import campaign_progress as cp  # noqa: E402  # pyright: ignore[reportMissingImports]


def _campaign(tmp_path: Path, runs: dict[str, tuple[str, str | None]]) -> Path:
    """Build ``logs/`` from ``{label: (log text, exit code or None)}``."""
    logs = tmp_path / "camp" / "logs"
    logs.mkdir(parents=True)
    for label, (text, code) in runs.items():
        (logs / f"{label}.log").write_text(text)
        if code is not None:
            (logs / f"{label}.exit").write_text(f"{code}\n")
    return tmp_path / "camp"


class TestFinishedMeansTheRunnerSaidSo:
    def test_a_log_with_early_output_and_no_marker_is_still_in_flight(self, tmp_path: Path) -> None:
        camp = _campaign(
            tmp_path,
            {"a-seed1": ("warning at load time\n", None), "a-seed2": ("done\n", "0")},
        )
        s = cp.survey(camp, 2)
        assert s["finished"] == 1
        assert s["in_flight"] == 1

    def test_a_non_zero_marker_is_a_finished_failure(self, tmp_path: Path) -> None:
        camp = _campaign(tmp_path, {"a-seed1": ("", "0"), "a-seed2": ("boom\n", "1")})
        s = cp.survey(camp, 2)
        assert s["finished"] == 2
        assert s["failed"] == 1

    def test_a_campaign_without_markers_falls_back_and_says_so(self, tmp_path: Path) -> None:
        camp = _campaign(tmp_path, {"a-seed1": ("done\n", None), "a-seed2": ("", None)})
        s = cp.survey(camp, 2)
        assert s["finished"] == 1
        assert s["failed"] is None
        assert "log size" in str(s["basis"])


class TestAWatchStopsOnItsOwnCampaign:
    def test_with_a_total_it_stops_when_every_run_is_accounted_for(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Another campaign still running elsewhere must not keep this watch alive.
        monkeypatch.setattr(cp, "any_runner_alive", lambda: True)
        camp = _campaign(tmp_path, {"a-seed1": ("", "0"), "a-seed2": ("", "0")})
        assert cp.campaign_done(camp, 2)

    def test_with_a_total_it_does_not_stop_early(self, tmp_path: Path) -> None:
        camp = _campaign(tmp_path, {"a-seed1": ("", "0")})
        assert not cp.campaign_done(camp, 2)


def test_a_finished_campaigns_clock_stops_at_its_last_run(tmp_path: Path) -> None:
    """Read long after the last run exits, elapsed is still the campaign's own duration."""
    camp = _campaign(tmp_path, {"a-seed1": ("", "0"), "a-seed2": ("", "0")})
    long_ago = time.time() - 3 * 3600
    for marker in (camp / "logs").glob("*.exit"):
        os.utime(marker, (long_ago, long_ago))
    elapsed = cp.survey(camp, 2)["elapsed"]
    assert isinstance(elapsed, timedelta)
    # The directory was made moments ago and the markers are dated before it, so the clock that
    # stops at the markers reads non-positive; one measured to "now" would read a few milliseconds.
    assert elapsed.total_seconds() < 0


def test_a_running_campaigns_clock_runs_to_now(tmp_path: Path) -> None:
    """While a run is in flight, elapsed is measured to the moment of reading."""
    camp = _campaign(tmp_path, {"a-seed1": ("", "0"), "a-seed2": ("started\n", None)})
    for marker in (camp / "logs").glob("*.exit"):
        os.utime(marker, (0, 0))
    elapsed = cp.survey(camp, 2)["elapsed"]
    assert isinstance(elapsed, timedelta)
    assert elapsed.total_seconds() >= 0


def test_a_non_positive_total_is_refused(tmp_path: Path) -> None:
    """A zero total is an argument error, not a division by zero inside the report."""
    camp = _campaign(tmp_path, {"a-seed1": ("", "0")})
    with pytest.raises(SystemExit) as err:
        cp.main(["--campaign", str(camp), "--total", "0"])
    assert err.value.code == 2
