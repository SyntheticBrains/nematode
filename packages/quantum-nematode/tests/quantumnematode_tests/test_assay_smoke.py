"""The generated assay configs, end to end through ``run_simulation.py``.

Covers the assay-protocols scenario "The assay configs run" and the realworm-behavioural-validation
scenario "A food-removal capture is labelled".
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from quantumnematode.report.dtypes import BehaviourStep
from quantumnematode.validation import roaming_dwelling as rd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "campaigns"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "analysis"))
import generate_assay_configs as gen  # noqa: E402  # pyright: ignore[reportMissingImports]
import lawn_states as ls  # noqa: E402  # pyright: ignore[reportMissingImports]


@pytest.mark.smoke
@pytest.mark.parametrize("arm", gen.ARMS)
@pytest.mark.parametrize("protocol", list(gen.PROTOCOLS))
def test_an_assay_config_runs_and_records_its_protocol(
    tmp_path: Path,
    arm: str,
    protocol: str,
) -> None:
    """One episode completes with its phases in order and labelled, and classifies on food."""
    path, _ = gen.derive(arm, protocol)
    data = yaml.safe_load(path.read_text())
    data["environment"]["sensing"]["capture_behaviour"] = True
    config = tmp_path / path.name
    config.write_text(yaml.safe_dump(data, sort_keys=False))
    result = subprocess.run(  # noqa: S603 — fixed argv, no shell
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_simulation.py"),
            "--config",
            str(config),
            "--runs",
            "1",
            "--seed",
            "5991",
            "--log-level",
            "NONE",
            "--theme",
            "headless",
            "--no-file-log",
        ],
        check=False,
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    (capture,) = (tmp_path / "exports").rglob("behaviour_capture.json")
    payload = json.loads(capture.read_text())
    record = payload["assay"]
    steps = [BehaviourStep(**d) for d in payload["runs"][0]["steps"]]
    phases = data["environment"]["assay"]["phases"]
    assert len(steps) == sum(round(p["duration_s"]) for p in phases)
    for entry in record["phases"]:
        start = entry["start_step"]
        assert steps[start].phase == entry["name"]
        assert all(s.phase == entry["name"] for s in steps[start : start + entry["steps"]])
    assert all(s.satiety is not None and s.on_lawn is not None for s in steps)
    if all(p["food"] == "lawns" for p in phases):
        lawns = data["environment"]["foraging"]["lawns"]
        episode = ls.episode_windows(
            steps,
            rd.GaussianClassifier(hmm=rd.load_calibrated_hmm()),
            1.0 - (1.0 - lawns["intake_fraction"]) ** 0.2,
            1.0,
            step_seconds=1.0,
        )
        assert set(episode.states) <= {rd.ROAMING, rd.DWELLING, rd.OFF_FOOD}
