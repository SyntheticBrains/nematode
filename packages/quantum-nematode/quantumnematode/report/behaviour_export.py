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

    from quantumnematode.report.dtypes import BehaviourStep, SimulationResult

BEHAVIOUR_CAPTURE_FILENAME = "behaviour_capture.json"
# The step a capture is taken at when it does not say: the point worm's full-speed step at 1 mm.
DEFAULT_CAPTURE_STEP_SECONDS = 5.0


def capture_step_seconds(data: dict[str, object]) -> float:
    """Return the step, in worm-seconds, a loaded capture was taken at; 5 when it does not say."""
    value = data.get("step_seconds", DEFAULT_CAPTURE_STEP_SECONDS)
    if not isinstance(value, int | float):
        msg = f"a capture's step_seconds must be a number, got {value!r}"
        raise TypeError(msg)
    return float(value)


def write_behaviour_capture(
    results: list[SimulationResult],
    data_dir: Path,
    *,
    step_seconds: float | None = None,
    assay: dict[str, object] | None = None,
) -> Path | None:
    """Write the per-run behavioural series to ``<data_dir>/behaviour_capture.json``.

    Parameters
    ----------
    results : list[SimulationResult]
        The completed runs; only those carrying a captured ``behaviour`` series are written.
    data_dir : Path
        Directory to write ``behaviour_capture.json`` into (created if missing).
    step_seconds : float | None
        The runs' step in worm-seconds. Recorded only when it is not the default 5, so a capture at
        the default step is unchanged; a reader takes an absent key as 5.
    assay : dict | None
        The runs' assay protocol in steps (its phases' start steps, the assay phase and the analysis
        window); recorded only under an assay.

    Returns
    -------
    Path | None
        The written file path, or ``None`` when no run captured a behavioural series (capture off).
    """
    runs = [
        {
            "run": r.run,
            "seed": r.seed,
            "steps": [_record(step) for step in r.behaviour],
        }
        for r in results
        if r.behaviour
    ]
    if not runs:
        return None
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / BEHAVIOUR_CAPTURE_FILENAME
    payload: dict[str, object] = {"runs": runs}
    if step_seconds is not None and step_seconds != DEFAULT_CAPTURE_STEP_SECONDS:
        payload["step_seconds"] = step_seconds
    if assay is not None:
        payload["assay"] = assay
    with output_path.open("w") as f:
        json.dump(payload, f)
    return output_path


_OPTIONAL_FIELDS = ("speed_signed", "satiety", "intake", "on_lawn", "primitive", "phase")


def _record(step: BehaviourStep) -> dict[str, object]:
    """Return one step's export record, leaving out each optional field that was not recorded."""
    record = asdict(step)
    for name in _OPTIONAL_FIELDS:
        if record[name] is None:
            del record[name]
    return record
