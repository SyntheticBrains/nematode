"""C.1d's configs, its calibration rule, its control gate, its fallback and its half-step check.

Covers the add-body-positive-control change's registered rules: the steering calibration (the best
mean plateau, ties within 5 points to the smaller gain, the diagnosis when nothing learns), the
positive control (the paired floor gate and 30% competence, with the fallback's shortest competent
length), and the half-step agreement of the kinematic instruments.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from quantumnematode.env.body import BodyParams
from quantumnematode.utils.config_loader import load_simulation_config

_REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO / "scripts" / "analysis"))
sys.path.insert(0, str(_REPO / "scripts" / "campaigns"))

import body_control as bc  # noqa: E402  # pyright: ignore[reportMissingImports]
import generate_body_control_configs as gen  # noqa: E402  # pyright: ignore[reportMissingImports]


class TestConfigs:
    @pytest.mark.parametrize("gain", gen.GAINS)
    @pytest.mark.parametrize("arm", ["learn", "frozen"])
    def test_the_committed_pilot_configs_are_the_generators(self, gain: float, arm: str) -> None:
        """Each pilot config on disk is what the generator writes, and it loads."""
        path, text = gen.derive(arm, gain=gain)
        assert path.read_text() == text
        config = load_simulation_config(str(path))
        assert config.environment is not None
        continuous = config.environment.continuous
        assert continuous is not None
        assert continuous.body_model == "kinematic"
        assert continuous.body_steering_gain == gain
        assert config.brain is not None
        brain = config.brain.config
        assert getattr(brain, "action_space", None) == "body_drive"
        assert getattr(brain, "freeze_updates", False) is (arm == "frozen")

    def test_the_control_waits_for_the_frozen_gain(self) -> None:
        """The control is refused at any gain but the body's default."""
        default = BodyParams().steering_gain
        assert len(gen.stage_configs("control", gain=default)) == 2
        with pytest.raises(ValueError, match="freeze the pilot's choice"):
            gen.stage_configs("control", gain=default * 2)

    def test_the_fallback_lengths_are_fixed(self) -> None:
        """Only the registered fallback lengths are written."""
        path, text = gen.stage_configs("fallback", steps=700)[0]
        assert path.name.endswith("_body_steps700.yml")
        assert "max_steps: 700" in text
        with pytest.raises(ValueError, match="fallback lengths"):
            gen.stage_configs("fallback", steps=600)


def _pilot(learn: dict[float, list[float]], frozen: float = 10.0) -> dict:
    seeds = bc.PILOT_SEEDS
    return bc.calibration(
        {g: dict(zip(seeds, v, strict=True)) for g, v in learn.items()},
        {g: dict.fromkeys(seeds, frozen) for g in learn},
    )


class TestCalibration:
    def test_ties_go_to_the_smaller_gain(self) -> None:
        """Gains within 5 points of the best tie, and the smallest of them is chosen."""
        assert bc.select_gain({0.5: 40.0, 1.0: 44.0, 2.0: 46.0, 4.0: 30.0}) == 1.0
        assert bc.select_gain({0.5: 20.0, 1.0: 30.0, 2.0: 46.0, 4.0: 45.0}) == 2.0

    def test_the_chosen_gain_and_its_neighbours(self) -> None:
        """The pilot names its gain and its neighbours' plateaus relative to it."""
        result = _pilot({0.5: [20.0] * 4, 1.0: [30.0] * 4, 2.0: [50.0] * 4, 4.0: [40.0] * 4})
        assert result["verdict"] == "chosen"
        assert result["gain"] == 2.0
        assert result["sensitivity"] == {"1": -20.0, "4": -10.0}

    def test_nothing_learned_is_the_diagnosis(self) -> None:
        """No gain beating its floor at any seed makes the pilot the diagnosis."""
        result = _pilot({g: [5.0] * 4 for g in gen.GAINS})
        assert result["verdict"] == "diagnosis"

    def test_a_missing_run_is_incomplete(self) -> None:
        """A gain short of a seed leaves the pilot unread."""
        learn = {g: dict.fromkeys(bc.PILOT_SEEDS, 50.0) for g in gen.GAINS}
        del learn[1.0][bc.PILOT_SEEDS[0]]
        frozen = {g: dict.fromkeys(bc.PILOT_SEEDS, 10.0) for g in gen.GAINS}
        assert bc.calibration(learn, frozen)["verdict"] == "incomplete"


class TestControl:
    _SEEDS = bc.CONTROL_SEEDS

    def _gate(self, learn: list[float], frozen: float = 10.0) -> str:
        return bc.control_gate(
            dict(zip(self._SEEDS, learn, strict=True)),
            dict.fromkeys(self._SEEDS, frozen),
        )["verdict"]

    def test_floor_and_competence_pass(self) -> None:
        """Every seed above its floor and at least 30% passes."""
        assert self._gate([40.0, 45.0, 50.0, 35.0, 60.0, 42.0, 38.0, 55.0]) == "passes"

    def test_a_seed_below_competence_goes_to_the_fallback(self) -> None:
        """A passing floor with one seed under 30% is the fallback."""
        assert self._gate([40.0, 45.0, 50.0, 25.0, 60.0, 42.0, 38.0, 55.0]) == "fallback"

    def test_no_lift_over_the_floor_fails(self) -> None:
        """No paired lift over the floor fails the control."""
        assert self._gate([10.0] * 8) == "fails"

    def test_a_missing_seed_is_incomplete(self) -> None:
        """Seven of eight seeds is not a reading."""
        learn = dict(zip(self._SEEDS[:7], [40.0] * 7, strict=True))
        assert bc.control_gate(learn, dict.fromkeys(self._SEEDS, 10.0))["verdict"] == "incomplete"

    def test_the_shortest_competent_length_is_chosen(self) -> None:
        """The fallback takes the shortest length every seed reaches competence at."""
        seeds = bc.FALLBACK_PILOT_SEEDS
        by_length = {
            500: dict(zip(seeds, [40.0, 20.0, 50.0, 35.0], strict=True)),
            700: dict.fromkeys(seeds, 35.0),
            1000: dict.fromkeys(seeds, 60.0),
        }
        assert bc.choose_length(by_length) == 700
        assert bc.choose_length({500: dict.fromkeys(seeds, 10.0)}) is None


class TestKinematics:
    def test_the_half_step_tolerance(self) -> None:
        """Means agree within 10%; reversal fraction also within 0.01 absolute."""
        base = {"frequency_hz": 0.30, "wavelength_bl": 0.65, "speed_bl_per_s": 0.10}
        base["reversal_fraction"] = 0.02
        close = {**base, "speed_bl_per_s": 0.091, "reversal_fraction": 0.029}
        assert bc.half_step_agreement(base, close)["agrees"]
        far = {**base, "speed_bl_per_s": 0.089}
        result = bc.half_step_agreement(base, far)
        assert not result["agrees"]
        assert result["instruments"]["speed_bl_per_s"]["agrees"] is False

    def test_final_weights_are_found_through_the_experiment_record(self, tmp_path: Path) -> None:
        """A run's log names its experiment, whose record names the exports holding its weights."""
        exports = tmp_path / "exports" / "session"
        (exports / "weights").mkdir(parents=True)
        (exports / "weights" / "final.pt").write_bytes(b"")
        experiments = tmp_path / "experiments"
        (experiments / "abc").mkdir(parents=True)
        (experiments / "abc" / "abc.json").write_text(json.dumps({"exports_path": str(exports)}))
        log = tmp_path / "run.log"
        log.write_text("...\nExperiment ID: abc\n...")
        assert bc.final_weights(log, experiments) == exports / "weights" / "final.pt"
        log.write_text("no record")
        assert bc.final_weights(log, experiments) is None
