"""The 1-second lawn worms, end to end through ``run_simulation.py``.

Covers the step-duration scenarios "The 1-second configs run" and "A campaign run records its step",
and the realworm-behavioural-validation scenario "A 1-second capture is labelled".
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
FORAGING = PROJECT_ROOT / "configs" / "scenarios" / "foraging"
CONFIGS = [
    "connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_lawns_modulated_s1",
    "mlpppo_small_continuous2d_fick_adaptive_klinotaxis_lawns_internal_s1",
]
# A short episode keeps the smoke quick: 60 reference steps, 300 one-second steps.
_REFERENCE_STEPS = 60

sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "analysis"))
import lawn_states as ls  # noqa: E402  # pyright: ignore[reportMissingImports]


@pytest.mark.smoke
@pytest.mark.parametrize("stem", CONFIGS)
def test_a_one_second_config_runs_records_and_classifies(tmp_path: Path, stem: str) -> None:
    """It completes, prints its step, labels its capture, and the capture classifies."""
    data = yaml.safe_load((FORAGING / f"{stem}.yml").read_text())
    data["max_steps"] = _REFERENCE_STEPS
    data["environment"]["sensing"]["capture_behaviour"] = True
    config = tmp_path / f"{stem}.yml"
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
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "Step: 1 s (reference 5 s, x5)" in result.stdout
    assert f"  max_steps: {_REFERENCE_STEPS} -> {5 * _REFERENCE_STEPS}" in result.stdout
    (capture,) = (tmp_path / "exports").rglob("behaviour_capture.json")
    payload = json.loads(capture.read_text())
    assert payload["step_seconds"] == 1.0
    steps = [BehaviourStep(**d) for d in payload["runs"][0]["steps"]]
    assert len(steps) > 2 * rd.WINDOW_SECONDS
    lawns = data["environment"]["foraging"]["lawns"]
    episode = ls.episode_windows(
        steps,
        rd.GaussianClassifier(hmm=rd.load_calibrated_hmm()),
        1.0 - (1.0 - lawns["intake_fraction"]) ** 0.2,
        1.0,
        step_seconds=1.0,
    )
    assert len(episode.states) == (len(steps) - 1) // 10
    assert set(episode.states) <= {rd.ROAMING, rd.DWELLING, rd.OFF_FOOD}
