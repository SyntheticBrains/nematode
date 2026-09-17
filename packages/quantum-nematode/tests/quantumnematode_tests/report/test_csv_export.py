"""The incremental detailed-tracking writer, and the switch that turns it off.

The step-level CSVs under ``session/data/detailed/`` are the largest thing a run writes -- about
380 MB per 3000-episode run against a few MB for everything else -- and no analysis script reads
them. A 768-run campaign wrote ~290 GB of them and filled the volume mid-run. ``enabled=False``
has to mean *nothing*: no directory, no files, no handles.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from quantumnematode.brain.arch._brain import BrainHistoryData
from quantumnematode.report.csv_export import IncrementalDetailedTrackingWriter

if TYPE_CHECKING:
    from pathlib import Path


def _history() -> BrainHistoryData:
    return BrainHistoryData(counts=[{"left": 1, "right": 2}, {"left": 0, "right": 3}])


class TestTheEnabledSwitch:
    def test_disabled_writes_nothing_at_all(self, tmp_path: Path) -> None:
        writer = IncrementalDetailedTrackingWriter(tmp_path, enabled=False)
        assert writer.enabled is False
        writer.write_run(1, _history())
        writer.write_run(2, _history())
        writer.close()
        assert not (tmp_path / "detailed").exists(), "a disabled writer created the directory"
        assert list(tmp_path.rglob("*")) == [], "a disabled writer wrote files"

    def test_enabled_is_the_default_and_still_writes(self, tmp_path: Path) -> None:
        # The regression half: turning the switch on must be the pre-existing behaviour.
        writer = IncrementalDetailedTrackingWriter(tmp_path)
        assert writer.enabled is True
        writer.write_run(1, _history())
        writer.close()
        out = tmp_path / "detailed" / "detailed_counts.csv"
        assert out.is_file()
        lines = out.read_text().splitlines()
        assert lines[0] == "run,step,left,right"
        assert lines[1:] == ["1,0,1,2", "1,1,0,3"]

    def test_close_on_a_disabled_writer_is_safe(self, tmp_path: Path) -> None:
        writer = IncrementalDetailedTrackingWriter(tmp_path, enabled=False)
        writer.close()
        writer.close()
