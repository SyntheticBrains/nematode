"""A campaign-launched run computes what a hand-launched run computes.

This is the campaign runner's central guarantee: it changes *when* runs
happen, never *what* they compute, so a campaign's results stay comparable
with every result measured before it existed.

Two real simulations are required to show it, so the test is marked ``slow``
and stays out of pre-commit; it runs before push with the rest of the heavy
integration tests.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

PROJECT_ROOT = Path(__file__).resolve().parents[4].parent
RUNNER_PATH = PROJECT_ROOT / "scripts" / "run_campaign.py"
SIMULATION_PATH = PROJECT_ROOT / "scripts" / "run_simulation.py"
CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "scenarios"
    / "foraging_predator_thermal"
    / "mlpppo_small_continuous2d_combined_klinotaxis.yml"
)

_SEED = 77
_EPISODES = 4

# Summary lines the run script prints; these are downstream of every step of
# the episode, so agreement across all of them is a strong trajectory check.
_METRIC_PATTERN = re.compile(
    r"^(Average reward per run|Average distance efficiency per run|"
    r"Total foods collected|Success rate):.*$",
    re.MULTILINE,
)


@pytest.fixture(scope="module")
def campaign() -> ModuleType:
    """Load ``scripts/run_campaign.py`` by path."""
    spec = importlib.util.spec_from_file_location("run_campaign_equiv", RUNNER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _metrics(text: str) -> list[str]:
    return [match.group(0).strip() for match in _METRIC_PATTERN.finditer(text)]


@pytest.mark.slow
class TestCampaignRunEquivalence:
    """Same config, same seed, two launch paths, one result."""

    def test_campaign_run_matches_hand_launched_run(
        self,
        campaign: ModuleType,
        tmp_path: Path,
    ) -> None:
        output_dir = tmp_path / "campaign"
        exit_code = campaign.main(
            [
                "--config",
                str(CONFIG),
                "--seeds",
                str(_SEED),
                "--runs",
                str(_EPISODES),
                "--workers",
                "1",
                "--output-dir",
                str(output_dir),
                "--",
                "--theme",
                "headless",
                "--log-level",
                "NONE",
            ],
        )
        assert exit_code == 0

        logs = list((output_dir / "logs").glob("*.log"))
        assert len(logs) == 1
        campaign_metrics = _metrics(logs[0].read_text(encoding="utf-8", errors="replace"))

        hand = subprocess.run(  # noqa: S603 — fixed argv, no shell
            [
                sys.executable,
                str(SIMULATION_PATH),
                "--config",
                str(CONFIG),
                "--seed",
                str(_SEED),
                "--runs",
                str(_EPISODES),
                "--theme",
                "headless",
                "--log-level",
                "NONE",
            ],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            check=True,
        )
        hand_metrics = _metrics(hand.stdout)

        # Guard against both sides being empty, which would pass vacuously.
        assert campaign_metrics, "campaign run produced no summary metrics"
        assert campaign_metrics == hand_metrics
