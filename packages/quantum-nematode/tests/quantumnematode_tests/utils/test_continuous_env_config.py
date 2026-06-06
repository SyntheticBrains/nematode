"""Tests for the continuous-2D environment configuration.

Covers the continuous-2D config fields: the `env_type` discriminator defaults to
grid (existing configs unchanged), the continuous fields parse with documented
defaults, explicit overrides load, and the factory selects the right env class.
"""

from __future__ import annotations

from quantumnematode.utils.config_loader import (
    Continuous2DConfig,
    EnvironmentConfig,
    create_env_from_config,
)


class TestContinuous2DConfig:
    """The continuous-2D scale parameters."""

    def test_documented_defaults(self) -> None:
        cfg = Continuous2DConfig()
        assert cfg.world_size_mm == 50.0
        assert cfg.body_length_mm == 1.0
        assert cfg.max_step_mm == 1.0
        assert cfg.capture_radius_mm == 1.0
        assert cfg.sweep_amplitude_mm == 0.5

    def test_overrides_parse(self) -> None:
        cfg = Continuous2DConfig(world_size_mm=30.0, capture_radius_mm=1.5)
        assert cfg.world_size_mm == 30.0
        assert cfg.capture_radius_mm == 1.5


class TestEnvironmentConfigEnvType:
    """The env_type discriminator and continuous sub-config."""

    def test_defaults_to_grid_and_is_backward_compatible(self) -> None:
        # A config with no env_type (existing grid scenarios) is unchanged.
        cfg = EnvironmentConfig(grid_size=20)
        assert cfg.env_type == "grid"
        assert cfg.continuous is None

    def test_continuous_2d_from_dict(self) -> None:
        cfg = EnvironmentConfig.model_validate(
            {
                "env_type": "continuous_2d",
                "continuous": {"world_size_mm": 40.0, "max_step_mm": 0.8},
            },
        )
        assert cfg.env_type == "continuous_2d"
        assert cfg.continuous is not None
        assert cfg.continuous.world_size_mm == 40.0
        assert cfg.continuous.max_step_mm == 0.8

    def test_get_continuous_config_defaults_when_unset(self) -> None:
        # When env_type is continuous_2d but no block is given, the getter
        # supplies documented defaults (mirrors get_foraging_config etc.).
        cfg = EnvironmentConfig(env_type="continuous_2d")
        assert cfg.continuous is None
        resolved = cfg.get_continuous_config()
        assert resolved.world_size_mm == 50.0

    def test_invalid_env_type_rejected(self) -> None:
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            EnvironmentConfig(env_type="hexgrid")  # type: ignore[arg-type]


class TestFactoryDispatch:
    """`create_env_from_config` selects the env class from `env_type`."""

    def test_grid_by_default(self) -> None:
        from quantumnematode.env.env import DynamicForagingEnvironment

        env = create_env_from_config(EnvironmentConfig(grid_size=10))
        assert type(env) is DynamicForagingEnvironment  # exact grid type, not the subclass

    def test_continuous_2d_selected(self) -> None:
        from quantumnematode.env.continuous_2d import Continuous2DEnvironment

        env = create_env_from_config(
            EnvironmentConfig(
                env_type="continuous_2d",
                continuous=Continuous2DConfig(world_size_mm=24.0),
            ),
        )
        assert isinstance(env, Continuous2DEnvironment)
        assert env.continuous.world_size_mm == 24.0
        assert env.grid_size == 24  # extent derived from world_size_mm
