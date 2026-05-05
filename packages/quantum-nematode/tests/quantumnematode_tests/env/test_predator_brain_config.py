"""Tests for predator brain-config plumbing through PredatorParams + YAML.

Covers task 3.7 of add-learning-predators OpenSpec change:
- Default brain_config=None → HeuristicPredatorBrain instance
- Explicit kind: "heuristic" produces same brain type
- predator_id synthesis matches f"predator_{i}" for spawn loop index
- ID stability across env reset() calls within the same env instance
- ID reproducibility across two env instances with the same config + seed
- Unknown kind raises NotImplementedError (forward-compat for M5)
"""

import pytest
import yaml

from quantumnematode.brain.actions import Action
from quantumnematode.env import (
    DynamicForagingEnvironment,
    ForagingParams,
    HeuristicPredatorBrain,
    PredatorBrainConfig,
    PredatorParams,
    PredatorType,
)
from quantumnematode.utils.config_loader import PredatorConfig


def _make_env(brain_config: PredatorBrainConfig | None = None, *, seed: int = 42):
    """Construct a small DynamicForagingEnvironment with 2 predators."""
    return DynamicForagingEnvironment(
        grid_size=20,
        start_pos=(10, 10),
        foraging=ForagingParams(
            foods_on_grid=2,
            target_foods_to_collect=3,
            min_food_distance=2,
            agent_exclusion_radius=2,
            gradient_decay_constant=4.0,
            gradient_strength=1.0,
        ),
        viewport_size=(11, 11),
        max_body_length=2,
        action_set=[Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY],
        predator=PredatorParams(
            enabled=True,
            count=2,
            predator_type=PredatorType.PURSUIT,
            speed=1.0,
            detection_radius=8,
            damage_radius=1,
            brain_config=brain_config,
        ),
        seed=seed,
    )


class TestDefaultBrainIsHeuristic:
    """When brain_config=None, every spawned predator gets HeuristicPredatorBrain."""

    def test_default_none_yields_heuristic(self) -> None:
        env = _make_env(brain_config=None)
        assert len(env.predators) == 2
        for pred in env.predators:
            assert isinstance(pred.brain, HeuristicPredatorBrain)

    def test_explicit_heuristic_kind_yields_heuristic(self) -> None:
        env = _make_env(brain_config=PredatorBrainConfig(kind="heuristic"))
        for pred in env.predators:
            assert isinstance(pred.brain, HeuristicPredatorBrain)

    def test_default_and_explicit_produce_same_brain_type(self) -> None:
        # Same brain type → behavioural equivalence under matching seeds.
        env_default = _make_env(brain_config=None)
        env_explicit = _make_env(brain_config=PredatorBrainConfig(kind="heuristic"))
        for p_default, p_explicit in zip(
            env_default.predators, env_explicit.predators, strict=True
        ):
            assert type(p_default.brain) is type(p_explicit.brain)


class TestPredatorIdSynthesis:
    """predator_id synthesis matches f"predator_{i}" for spawn loop index."""

    def test_id_format_matches_index(self) -> None:
        env = _make_env()
        for i, pred in enumerate(env.predators):
            assert pred.predator_id == f"predator_{i}"

    def test_ids_lexicographically_ordered(self) -> None:
        env = _make_env()
        ids = [p.predator_id for p in env.predators]
        assert ids == sorted(ids)


class TestIdStabilityWithinInstance:
    """Within a single env instance, IDs remain unchanged across update calls."""

    def test_update_predators_preserves_ids(self) -> None:
        # The env doesn't have a top-level reset() that re-spawns predators;
        # _initialize_predators runs ONCE in __init__. The same Predator
        # instances persist across update_predators() calls, so IDs are
        # stable by object identity.
        env = _make_env()
        pre_ids = [p.predator_id for p in env.predators]
        for _ in range(50):
            env.update_predators()
        post_ids = [p.predator_id for p in env.predators]
        assert pre_ids == post_ids


class TestIdReproducibilityAcrossInstances:
    """Same config + seed → same predator IDs (and ordering)."""

    def test_two_envs_same_seed_yield_same_ids(self) -> None:
        env1 = _make_env(seed=42)
        env2 = _make_env(seed=42)
        ids1 = [p.predator_id for p in env1.predators]
        ids2 = [p.predator_id for p in env2.predators]
        assert ids1 == ids2

    def test_two_envs_same_seed_yield_same_positions(self) -> None:
        # Spawn positions depend on the env's RNG; if seeds are equal,
        # positions must be identical too.
        env1 = _make_env(seed=42)
        env2 = _make_env(seed=42)
        positions1 = [p.position for p in env1.predators]
        positions2 = [p.position for p in env2.predators]
        assert positions1 == positions2


class TestUnknownKindRejection:
    """Unknown brain kinds SHALL raise NotImplementedError (M5 forward-compat)."""

    def test_unknown_kind_raises(self) -> None:
        # Bypass Pydantic validation by constructing the dataclass directly
        # (Pydantic schema only allows Literal["heuristic"], so YAML rejects
        # unknown kinds at load time; runtime constructions might still bypass).
        bad_config = PredatorBrainConfig(kind="heuristic")
        # Manually mutate (frozen=True so use object.__setattr__ for the test).
        object.__setattr__(bad_config, "kind", "mlpppo")
        with pytest.raises(NotImplementedError, match="Unknown predator brain kind"):
            _make_env(brain_config=bad_config)


class TestYamlBrainConfigDispatch:
    """YAML PredatorConfig.brain_config translates to runtime PredatorBrainConfig."""

    def test_yaml_no_brain_config_block_yields_none(self) -> None:
        yaml_str = """
        enabled: true
        count: 2
        movement_pattern: pursuit
        speed: 1.0
        detection_radius: 8
        damage_radius: 1
        """
        cfg = PredatorConfig.model_validate(yaml.safe_load(yaml_str))
        params = cfg.to_params()
        assert params.brain_config is None

    def test_yaml_explicit_heuristic_block(self) -> None:
        yaml_str = """
        enabled: true
        count: 2
        movement_pattern: pursuit
        speed: 1.0
        detection_radius: 8
        damage_radius: 1
        brain_config:
            kind: heuristic
        """
        cfg = PredatorConfig.model_validate(yaml.safe_load(yaml_str))
        params = cfg.to_params()
        assert params.brain_config is not None
        assert params.brain_config.kind == "heuristic"

    def test_yaml_unknown_kind_rejected_by_pydantic(self) -> None:
        # Pydantic Literal["heuristic"] should reject "mlpppo" at validation
        # time (M5 will extend the literal type).
        yaml_str = """
        enabled: true
        count: 2
        movement_pattern: pursuit
        brain_config:
            kind: mlpppo
        """
        with pytest.raises(Exception):  # noqa: B017, PT011
            PredatorConfig.model_validate(yaml.safe_load(yaml_str))


class TestCopyEnvironmentPreservesPredators:
    """copy_environment preserves predator_id + brain (via brain.copy)."""

    def test_copy_preserves_ids(self) -> None:
        env = _make_env()
        new_env = env.copy()
        assert [p.predator_id for p in env.predators] == [
            p.predator_id for p in new_env.predators
        ]

    def test_copy_preserves_positions(self) -> None:
        env = _make_env()
        new_env = env.copy()
        assert [p.position for p in env.predators] == [
            p.position for p in new_env.predators
        ]

    def test_copy_yields_independent_brain_instances(self) -> None:
        env = _make_env()
        new_env = env.copy()
        # Brain copies should be functionally equivalent but independent
        # objects (brain.copy() returns a fresh instance).
        for p_orig, p_copy in zip(env.predators, new_env.predators, strict=True):
            assert p_orig.brain is not p_copy.brain
            assert type(p_orig.brain) is type(p_copy.brain)
