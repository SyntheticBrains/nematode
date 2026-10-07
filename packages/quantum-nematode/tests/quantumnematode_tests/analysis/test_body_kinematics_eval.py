"""The evaluation harness runs trained body-drive weights with posture capture.

Covers the body-kinematics requirement "Kinematic instruments": a trained run's final weights are
evaluated frozen, with the body's posture captured, at the run's own sub-step count or at a doubled
one for the half-step check.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml
from quantumnematode.brain.weights import save_weights

_REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO / "scripts" / "analysis"))

import body_kinematics_eval as harness  # noqa: E402  # pyright: ignore[reportMissingImports]

_BASE = (
    _REPO / "configs/scenarios/foraging/"
    "mlpppo_small_continuous2d_fick_adaptive_klinotaxis_hard350_ppo_w64_reversal.yml"
)


@pytest.fixture
def body_config(tmp_path: Path) -> Path:
    """Write the MLP reversal config with the kinematic body and a 30-step episode."""
    raw = yaml.safe_load(_BASE.read_text())
    raw["max_steps"] = 30
    raw["environment"]["continuous"]["body_model"] = "kinematic"
    raw["brain"]["config"]["action_space"] = "body_drive"
    raw["brain"]["config"].pop("signed_speed", None)
    path = tmp_path / "body.yml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


def test_saved_weights_are_evaluated_with_posture_capture(
    body_config: Path,
    tmp_path: Path,
) -> None:
    """Saved weights round-trip through the harness at 20 and at 40 sub-steps."""
    brain, _, _ = harness.build_brain(body_config, seed=7)
    weights = save_weights(brain, tmp_path / "final.pt")
    assert weights is not None
    k = harness.evaluate(body_config, 7, weights, episodes=2)
    assert k.steps_used + k.steps_near_wall == 60
    assert 0.0 <= k.reversal_fraction <= 1.0
    doubled = harness.evaluate(body_config, 7, weights, episodes=2, substeps=40)
    assert doubled.steps_used + doubled.steps_near_wall == 60


def test_a_point_worm_config_is_refused(tmp_path: Path) -> None:
    """A config without the kinematic body has no posture to read."""
    raw = yaml.safe_load(_BASE.read_text())
    path = tmp_path / "point.yml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    brain, _, _ = harness.build_brain(path, seed=7)
    weights = save_weights(brain, tmp_path / "final.pt")
    assert weights is not None
    with pytest.raises(ValueError, match="does not run the kinematic body"):
        harness.evaluate(path, 7, weights, episodes=1)
