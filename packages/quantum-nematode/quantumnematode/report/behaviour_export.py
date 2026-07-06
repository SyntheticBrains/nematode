"""Persist captured behavioural trajectories for offline klinotaxis-bias validation.

When ``capture_behaviour`` is on, each run carries a ``list[BehaviourStep]`` on its
``SimulationResult``. This writes those series to a single JSON per session (one entry per run,
keyed by seed) that the behavioural-chemotaxis validation harness reads back.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from quantumnematode.report.dtypes import SimulationResult

BEHAVIOUR_CAPTURE_FILENAME = "behaviour_capture.json"


def write_behaviour_capture(
    results: list[SimulationResult],
    data_dir: Path,
) -> Path | None:
    """Write the per-run behavioural series to ``<data_dir>/behaviour_capture.json``.

    Parameters
    ----------
    results : list[SimulationResult]
        The completed runs; only those carrying a captured ``behaviour`` series are written.
    data_dir : Path
        Directory to write ``behaviour_capture.json`` into (created if missing).

    Returns
    -------
    Path | None
        The written file path, or ``None`` when no run captured a behavioural series (capture off).
    """
    runs = [
        {
            "run": r.run,
            "seed": r.seed,
            "steps": [asdict(step) for step in r.behaviour],
        }
        for r in results
        if r.behaviour
    ]
    if not runs:
        return None
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / BEHAVIOUR_CAPTURE_FILENAME
    with output_path.open("w") as f:
        json.dump({"runs": runs}, f)
    return output_path
