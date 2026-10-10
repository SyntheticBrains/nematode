"""The lawn food model: placement, the area-weighted odour field, intake, and its configuration.

Covers the patchy-lawns spec's scenarios: the default is unchanged; lawns are placed as separate
patches; a grazed region smells weaker; a full lawn smells like one source; quality is not smelled;
eating depletes the cell under the worm; speed does not change intake; off a lawn there is no
intake;
a dwelling penalty and exploration pay are refused; multi-agent lawns are refused.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from quantumnematode.env.continuous_2d import Continuous2DEnvironment, Continuous2DParams
from quantumnematode.env.env import ForagingParams
from quantumnematode.env.lawns import LawnField, LawnParams
from quantumnematode.env.theme import Theme
from quantumnematode.utils.config_loader import SimulationConfig

WORLD_MM = 20.0


def _env(
    lawns: LawnParams | None = None,
    *,
    seed: int = 3,
    **foraging: Any,
) -> Continuous2DEnvironment:
    params = ForagingParams(
        food_model="lawns" if lawns is not None else "points",
        lawns=lawns,
        gradient_field_mode="fick",
        gradient_decay_constant=4.0,
        foods_on_grid=5,
        **foraging,
    )
    return Continuous2DEnvironment(
        continuous=Continuous2DParams(world_size_mm=WORLD_MM),
        foraging=params,
        seed=seed,
        theme=Theme.HEADLESS,
    )


def _single(quality: float = 1.0, centre: tuple[float, float] = (10.0, 10.0)) -> LawnField:
    return LawnField(np.array([centre]), radius_mm=2.5, cell_mm=1.0, qualities=np.array([quality]))


def _fick(distances: np.ndarray) -> np.ndarray:
    return np.exp(-((distances / 4.0) ** 2))


class TestPlacement:
    def test_the_default_is_unchanged(self) -> None:
        """Point food places its sources and carries no lawns."""
        env = _env()
        assert env.lawn_field is None
        assert len(env.foods) == 5

    def test_lawns_are_placed_as_separate_patches(self) -> None:
        """Every lawn lies in the arena, its edge clear of every other's; all cells start full."""
        params = LawnParams(count=3, radius_mm=2.5, min_separation_mm=2.0)
        for seed in range(10):
            env = _env(params, seed=seed)
            field = env.lawn_field
            assert field is not None
            assert env.foods == []
            centres = field.centres
            assert len(centres) == 3
            assert np.all(centres >= params.radius_mm)
            assert np.all(centres <= WORLD_MM - 2.5)
            for i in range(3):
                for j in range(i + 1, 3):
                    gap = math.dist(centres[i], centres[j]) - 2 * params.radius_mm
                    assert gap >= params.min_separation_mm
            assert np.all(field.density == 1.0)

    def test_an_arena_too_small_is_refused(self) -> None:
        """Lawns that cannot fit raise rather than silently placing fewer."""
        with pytest.raises(ValueError, match="too small"):
            _env(LawnParams(count=12, radius_mm=3.0, min_separation_mm=2.0))

    def test_a_pixel_renderer_is_refused(self) -> None:
        """The pixel renderers would draw no lawns."""
        params = ForagingParams(food_model="lawns", lawns=LawnParams())
        with pytest.raises(ValueError, match="does not draw lawns"):
            Continuous2DEnvironment(
                continuous=Continuous2DParams(world_size_mm=WORLD_MM),
                foraging=params,
                seed=1,
                theme=Theme.PIXEL_CONTINUOUS,
            )

    def test_a_copy_is_independent(self) -> None:
        """Grazing a copy leaves the original's lawns full."""
        env = _env(LawnParams())
        copy = env.copy()
        assert copy.lawn_field is not None
        assert env.lawn_field is not None
        copy.lawn_field.eat(copy.lawn_field.centres[0], 0.5)
        assert np.all(env.lawn_field.density == 1.0)


class TestOdour:
    def test_a_full_lawn_smells_like_one_source(self) -> None:
        """Far away, a full lawn's odour matches a unit point source at its centre within 5%."""
        field = _single()
        far = (10.0 + 30.0, 10.0)
        lawn = field.concentration(far, lambda d: np.exp(-((d / 20.0) ** 2)))
        point = math.exp(-((30.0 / 20.0) ** 2))
        assert lawn == pytest.approx(point, rel=0.05)

    def test_a_grazed_region_smells_weaker(self) -> None:
        """Grazing one side lowers the odour there and turns the gradient toward the rest."""
        field = _single()
        left = (8.5, 10.0)
        before = field.concentration(left, _fick)
        for cell in np.flatnonzero(field.cells[:, 0] < 10.0):
            field.density[cell] = 0.0
        assert field.concentration(left, _fick) < before
        gx, _ = field.gradient((10.0, 10.0), _fick)
        assert gx > 0  # toward the uneaten right-hand side

    def test_quality_is_not_smelled(self) -> None:
        """Two lawns that differ only in quality have identical odour fields."""
        poor, rich = _single(quality=0.2), _single(quality=2.0)
        for point in [(10.0, 10.0), (14.0, 9.0), (2.0, 18.0)]:
            assert poor.concentration(point, _fick) == rich.concentration(point, _fick)
            assert poor.gradient(point, _fick) == rich.gradient(point, _fick)

    def test_the_environment_reads_the_lawns(self) -> None:
        """The concentration sensors and the gradient vector both see a lawn, and nothing else."""
        env = _env(LawnParams(count=1))
        assert env.lawn_field is not None
        centre = tuple(env.lawn_field.centres[0])
        corner = (0.5, 0.5)
        assert env.get_food_concentration(centre) > env.get_food_concentration(corner)
        gx, gy = env._compute_food_gradient_vector(corner)
        toward = np.array(centre) - np.array(corner)
        assert np.dot([gx, gy], toward) > 0


class TestIntake:
    def test_eating_depletes_the_cell_under_the_worm(self) -> None:
        """Staying on one cell, intake falls geometrically with the cell's density."""
        field = _single()
        intakes = [field.eat((10.0, 10.0), 0.25).amount for _ in range(4)]
        assert intakes == pytest.approx([0.25 * 0.75**k for k in range(4)])
        assert field.density[field.cell_at((10.0, 10.0))] == pytest.approx(0.75**4)

    def test_speed_does_not_change_intake(self) -> None:
        """Intake depends on position only: a moving and a still worm on equal cells eat alike."""
        still, moving = _single(), _single()
        assert still.eat((10.0, 10.0), 0.1) == moving.eat((11.0, 10.0), 0.1)

    def test_off_a_lawn_there_is_no_intake(self) -> None:
        """Outside every disc, nothing is eaten and nothing is lost."""
        field = _single()
        intake = field.eat((1.0, 1.0), 0.5)
        assert intake.amount == 0.0
        assert intake.lawn is None
        assert np.all(field.density == 1.0)

    def test_quality_scales_the_value_eaten(self) -> None:
        """The same bite is worth more on a richer lawn."""
        intake = _single(quality=2.0).eat((10.0, 10.0), 0.1)
        assert intake.value == pytest.approx(2.0 * intake.amount)

    def test_regrowth_is_capped_at_full(self) -> None:
        """Regrowth refills toward 1 and never past it."""
        field = _single()
        field.eat((10.0, 10.0), 0.5)
        field.regrow(0.3)
        assert field.density.max() == 1.0
        assert field.density.min() == pytest.approx(0.8)


def _config(**reward: float) -> dict[str, Any]:
    zeroed = {
        "penalty_stuck_position": 0.0,
        "penalty_anti_dithering": 0.0,
        "reward_exploration": 0.0,
        "reward_distance_scale": 0.0,
    }
    return {
        "environment": {
            "env_type": "continuous_2d",
            "continuous": {"world_size_mm": 20.0},
            "foraging": {"food_model": "lawns", "lawns": {"count": 3}},
        },
        "reward": zeroed | reward,
    }


class TestConfiguration:
    def test_a_lawn_configuration_loads(self) -> None:
        """A continuous, single-agent lawn configuration with no state shaping loads."""
        config = SimulationConfig.model_validate(_config())
        assert config.environment is not None
        assert config.environment.foraging is not None
        params = config.environment.foraging.to_params()
        assert params.food_model == "lawns"
        assert params.lawns is not None
        assert params.lawns.count == 3

    @pytest.mark.parametrize(
        "key",
        [
            "penalty_stuck_position",
            "penalty_anti_dithering",
            "reward_exploration",
            "reward_distance_scale",
        ],
    )
    def test_state_shaping_is_refused(self, key: str) -> None:
        """A dwelling penalty, exploration pay or approach reward is refused, by name."""
        with pytest.raises(ValueError, match=key):
            SimulationConfig.model_validate(_config(**{key: 0.5}))

    def test_multi_agent_lawns_are_refused(self) -> None:
        """Lawns run one agent."""
        raw = _config() | {"multi_agent": {"enabled": True, "count": 2}}
        with pytest.raises(ValueError, match="food_model 'lawns'"):
            SimulationConfig.model_validate(raw)

    def test_the_lawns_block_matches_the_model(self) -> None:
        """A lawns block without the lawn model, or the reverse, is refused."""
        raw = _config()
        raw["environment"]["foraging"]["food_model"] = "points"
        with pytest.raises(ValueError, match="if and only if"):
            SimulationConfig.model_validate(raw)
