"""Tests for predator brain-config plumbing through PredatorParams + YAML.

Covers:

- Default brain_config=None → HeuristicPredatorBrain instance
- Explicit kind: "heuristic" produces same brain type
- predator_id synthesis matches f"predator_{i}" for spawn loop index
- ID stability across env reset() calls within the same env instance
- ID reproducibility across two env instances with the same config + seed
- Unknown kind raises NotImplementedError (forward-compat for future
  learnable brain kinds)
"""

import pytest
import torch
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


def _make_env(
    brain_config: PredatorBrainConfig | None = None,
    *,
    seed: int = 42,
) -> DynamicForagingEnvironment:
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
        """Verify default none yields heuristic."""
        env = _make_env(brain_config=None)
        assert len(env.predators) == 2
        for pred in env.predators:
            assert isinstance(pred.brain, HeuristicPredatorBrain)

    def test_explicit_heuristic_kind_yields_heuristic(self) -> None:
        """Verify explicit heuristic kind yields heuristic."""
        env = _make_env(brain_config=PredatorBrainConfig(kind="heuristic"))
        for pred in env.predators:
            assert isinstance(pred.brain, HeuristicPredatorBrain)

    def test_default_and_explicit_produce_same_brain_type(self) -> None:
        """Verify default and explicit produce same brain type."""
        # Same brain type → behavioural equivalence under matching seeds.
        env_default = _make_env(brain_config=None)
        env_explicit = _make_env(brain_config=PredatorBrainConfig(kind="heuristic"))
        for p_default, p_explicit in zip(
            env_default.predators,
            env_explicit.predators,
            strict=True,
        ):
            assert type(p_default.brain) is type(p_explicit.brain)


class TestPredatorIdSynthesis:
    """predator_id synthesis matches f"predator_{i}" for spawn loop index."""

    def test_id_format_matches_index(self) -> None:
        """Verify id format matches index."""
        env = _make_env()
        for i, pred in enumerate(env.predators):
            assert pred.predator_id == f"predator_{i}"

    def test_ids_lexicographically_ordered(self) -> None:
        """Verify ids lexicographically ordered."""
        env = _make_env()
        ids = [p.predator_id for p in env.predators]
        assert ids == sorted(ids)


class TestIdStabilityWithinInstance:
    """Within a single env instance, IDs remain unchanged across update calls."""

    def test_update_predators_preserves_ids(self) -> None:
        """Verify update predators preserves ids."""
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
        """Verify two envs same seed yield same ids."""
        env1 = _make_env(seed=42)
        env2 = _make_env(seed=42)
        ids1 = [p.predator_id for p in env1.predators]
        ids2 = [p.predator_id for p in env2.predators]
        assert ids1 == ids2

    def test_two_envs_same_seed_yield_same_positions(self) -> None:
        """Verify two envs same seed yield same positions."""
        # Spawn positions depend on the env's RNG; if seeds are equal,
        # positions must be identical too.
        env1 = _make_env(seed=42)
        env2 = _make_env(seed=42)
        positions1 = [p.position for p in env1.predators]
        positions2 = [p.position for p in env2.predators]
        assert positions1 == positions2


class TestUnknownKindRejection:
    """Unknown brain kinds SHALL raise NotImplementedError.

    Forward-compat for future learnable brain kinds — when a new kind
    is added to the dispatcher, this test confirms the dispatcher
    correctly rejects unknown values rather than silently falling back.
    """

    def test_unknown_kind_raises(self) -> None:
        """Verify unknown kind raises."""
        # Bypass Pydantic validation by constructing the dataclass directly
        # (Pydantic schema only allows Literal["heuristic", "mlpppo_predator"],
        # so YAML rejects unknown kinds at load time; runtime constructions
        # might still bypass).
        bad_config = PredatorBrainConfig(kind="heuristic")
        # Manually mutate (frozen=True so use object.__setattr__ for the test).
        # Use "qsnnppo" — clearly a brain name that is NOT a registered
        # predator kind, future-proof against the literal type expanding.
        object.__setattr__(bad_config, "kind", "qsnnppo")
        with pytest.raises(NotImplementedError, match="Unknown predator brain kind"):
            _make_env(brain_config=bad_config)


class TestMLPPPOPredatorDispatch:
    """`kind: "mlpppo_predator"` constructs a `MLPPPOPredatorBrain` instance."""

    def test_explicit_mlpppo_predator_yields_mlpppo_brain(self) -> None:
        """Verify mlpppo_predator dispatch produces MLPPPOPredatorBrain."""
        from quantumnematode.env.mlpppo_predator_brain import MLPPPOPredatorBrain

        env = _make_env(brain_config=PredatorBrainConfig(kind="mlpppo_predator"))
        for pred in env.predators:
            assert isinstance(pred.brain, MLPPPOPredatorBrain)
            # AND it SHALL also be `isinstance(brain, PredatorBrain)`
            # via the @runtime_checkable Protocol.
            from quantumnematode.env import PredatorBrain

            assert isinstance(pred.brain, PredatorBrain)

    def test_mlpppo_predator_with_extra_overrides(self) -> None:
        """Verify `extra` config keys override the default architecture."""
        from quantumnematode.env.mlpppo_predator_brain import MLPPPOPredatorBrain

        env = _make_env(
            brain_config=PredatorBrainConfig(
                kind="mlpppo_predator",
                extra={"actor_hidden_dim": 32, "critic_hidden_dim": 32, "seed": 123},
            ),
        )
        for pred in env.predators:
            assert isinstance(pred.brain, MLPPPOPredatorBrain)
            # Verify the override took effect by checking actor's first
            # Linear layer's out_features.
            first_linear = next(m for m in pred.brain.actor if isinstance(m, torch.nn.Linear))
            assert first_linear.out_features == 32

    def test_mlpppo_predator_extra_dim_floats_coerced_to_int(self) -> None:
        """Verify dim/layer-count keys accept float values (YAML may produce 32.0).

        Without the dispatcher's `int(...)` coercion, passing `32.0` for
        `actor_hidden_dim` would fail mid-construction inside `nn.Linear`
        with `TypeError: 'float' object cannot be interpreted as an
        integer`. This test confirms the coercion holds.
        """
        from quantumnematode.env.mlpppo_predator_brain import MLPPPOPredatorBrain

        env = _make_env(
            brain_config=PredatorBrainConfig(
                kind="mlpppo_predator",
                extra={
                    "actor_hidden_dim": 32.0,
                    "critic_hidden_dim": 32.0,
                    "num_hidden_layers": 2.0,
                    "seed": 123,
                },
            ),
        )
        for pred in env.predators:
            assert isinstance(pred.brain, MLPPPOPredatorBrain)
            first_linear = next(m for m in pred.brain.actor if isinstance(m, torch.nn.Linear))
            assert first_linear.out_features == 32
            # Construction succeeded → coercion held.

    @pytest.mark.parametrize(
        ("sample_value", "expected"),
        [
            # Native bool — direct passthrough.
            (True, True),
            (False, False),
            # Native numeric — 0 is False, non-zero is True.
            (1, True),
            (0, False),
            (1.0, True),
            (0.0, False),
            # Truthy string tokens (case-insensitive, whitespace-tolerant).
            ("true", True),
            ("True", True),
            ("TRUE", True),
            (" true ", True),
            ("1", True),
            ("yes", True),
            ("on", True),
            # Falsy string tokens — including the bug case `"false"` which
            # `bool("false") == True` would have incorrectly flipped.
            ("false", False),
            ("False", False),
            ("FALSE", False),
            ("0", False),
            ("no", False),
            ("off", False),
        ],
    )
    def test_mlpppo_predator_sample_string_normalisation(
        self,
        sample_value: bool | float | str,  # noqa: FBT001 — pytest parametrize requires positional
        expected: bool,  # noqa: FBT001 — pytest parametrize requires positional
    ) -> None:
        """Verify `extra['sample']` accepts bool/int/float/string truthy/falsy tokens.

        Regression test for the bug where `bool("false")` evaluated to
        `True` (any non-empty string is truthy in Python). YAML quoted
        strings (`sample: "false"`) would silently flip semantics under
        naive bool coercion.
        """
        from quantumnematode.env.mlpppo_predator_brain import MLPPPOPredatorBrain

        env = _make_env(
            brain_config=PredatorBrainConfig(
                kind="mlpppo_predator",
                extra={"sample": sample_value, "seed": 42},
            ),
        )
        for pred in env.predators:
            assert isinstance(pred.brain, MLPPPOPredatorBrain)
            assert pred.brain._sample is expected, (
                f"sample={sample_value!r} → expected {expected}, got {pred.brain._sample}"
            )

    def test_mlpppo_predator_sample_unknown_string_raises(self) -> None:
        """Verify unrecognised string tokens raise ValueError instead of silent fallback."""
        with pytest.raises(ValueError, match="not a recognised boolean"):
            _make_env(
                brain_config=PredatorBrainConfig(
                    kind="mlpppo_predator",
                    extra={"sample": "maybe", "seed": 42},
                ),
            )

    def test_mlpppo_predator_sample_wrong_type_raises(self) -> None:
        """Verify non-bool/int/float/str values raise ValueError."""
        with pytest.raises(ValueError, match="must be bool, int, float, or str"):
            _make_env(
                brain_config=PredatorBrainConfig(
                    kind="mlpppo_predator",
                    extra={"sample": ["yes"], "seed": 42},
                ),
            )

    def test_mlpppo_predator_seed_reproducibility(self) -> None:
        """Verify that two envs with the same mlpppo seed produce identical predator weights."""
        env_a = _make_env(
            brain_config=PredatorBrainConfig(
                kind="mlpppo_predator",
                extra={"seed": 42},
            ),
            seed=100,  # env seed; predator seed is from extra.seed
        )
        env_b = _make_env(
            brain_config=PredatorBrainConfig(
                kind="mlpppo_predator",
                extra={"seed": 42},
            ),
            seed=100,
        )
        from quantumnematode.env.mlpppo_predator_brain import MLPPPOPredatorBrain

        for pred_a, pred_b in zip(env_a.predators, env_b.predators, strict=True):
            assert isinstance(pred_a.brain, MLPPPOPredatorBrain)
            assert isinstance(pred_b.brain, MLPPPOPredatorBrain)
            for p_a, p_b in zip(
                pred_a.brain.actor.parameters(),
                pred_b.brain.actor.parameters(),
                strict=True,
            ):
                assert torch.allclose(p_a, p_b)

    def test_mlpppo_predator_seed_derived_from_env_when_extra_omits_seed(self) -> None:
        """Same env seed without extra['seed'] yields identical predator weights.

        Verifies the env-derived seed fallback in `_build_predator_brain`:
        when `extra` lacks a `seed` key, the brain's seed is drawn from
        `self.rng.integers` deterministically. Two envs with the same env
        `seed=` therefore produce bit-identical predator weights even
        without explicit predator-side seed configuration.
        """
        env_a = _make_env(
            brain_config=PredatorBrainConfig(kind="mlpppo_predator"),  # no extra
            seed=2026,
        )
        env_b = _make_env(
            brain_config=PredatorBrainConfig(kind="mlpppo_predator"),
            seed=2026,
        )
        from quantumnematode.env.mlpppo_predator_brain import MLPPPOPredatorBrain

        for pred_a, pred_b in zip(env_a.predators, env_b.predators, strict=True):
            assert isinstance(pred_a.brain, MLPPPOPredatorBrain)
            assert isinstance(pred_b.brain, MLPPPOPredatorBrain)
            for p_a, p_b in zip(
                pred_a.brain.actor.parameters(),
                pred_b.brain.actor.parameters(),
                strict=True,
            ):
                assert torch.allclose(p_a, p_b)

    def test_mlpppo_predator_different_env_seeds_produce_different_weights(self) -> None:
        """Two envs with different env seeds (no extra['seed']) produce different predator weights.

        Confirms the derived seed actually flows from the env RNG —
        without this test, a bug that ignored the env seed entirely
        (e.g. seed=0 hardcoded) would silently pass the
        same-seed-same-weights test above.
        """
        env_a = _make_env(
            brain_config=PredatorBrainConfig(kind="mlpppo_predator"),
            seed=2026,
        )
        env_b = _make_env(
            brain_config=PredatorBrainConfig(kind="mlpppo_predator"),
            seed=9999,
        )
        from quantumnematode.env.mlpppo_predator_brain import MLPPPOPredatorBrain

        # At least one predator's weights SHALL differ between the two envs.
        any_differ = False
        for pred_a, pred_b in zip(env_a.predators, env_b.predators, strict=True):
            assert isinstance(pred_a.brain, MLPPPOPredatorBrain)
            assert isinstance(pred_b.brain, MLPPPOPredatorBrain)
            if any(
                not torch.allclose(p_a, p_b)
                for p_a, p_b in zip(
                    pred_a.brain.actor.parameters(),
                    pred_b.brain.actor.parameters(),
                    strict=True,
                )
            ):
                any_differ = True
                break
        assert any_differ, "different env seeds should produce different predator weights"


class TestYamlMLPPPOPredatorKind:
    """YAML schema accepts `kind: mlpppo_predator`."""

    def test_yaml_mlpppo_predator_kind_accepted(self) -> None:
        """Verify yaml mlpppo_predator kind accepted."""
        yaml_str = """
        enabled: true
        count: 2
        movement_pattern: pursuit
        speed: 1.0
        detection_radius: 8
        damage_radius: 1
        brain_config:
            kind: mlpppo_predator
        """
        cfg = PredatorConfig.model_validate(yaml.safe_load(yaml_str))
        params = cfg.to_params()
        assert params.brain_config is not None
        assert params.brain_config.kind == "mlpppo_predator"

    def test_yaml_mlpppo_predator_with_extra(self) -> None:
        """Verify yaml mlpppo_predator with extra config is accepted."""
        yaml_str = """
        enabled: true
        count: 2
        movement_pattern: pursuit
        speed: 1.0
        detection_radius: 8
        damage_radius: 1
        brain_config:
            kind: mlpppo_predator
            extra:
                actor_hidden_dim: 32
                seed: 7
        """
        cfg = PredatorConfig.model_validate(yaml.safe_load(yaml_str))
        params = cfg.to_params()
        assert params.brain_config is not None
        assert params.brain_config.kind == "mlpppo_predator"
        assert params.brain_config.extra == {"actor_hidden_dim": 32, "seed": 7}


class TestYamlBrainConfigDispatch:
    """YAML PredatorConfig.brain_config translates to runtime PredatorBrainConfig."""

    def test_yaml_no_brain_config_block_yields_none(self) -> None:
        """Verify yaml no brain config block yields none."""
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
        """Verify yaml explicit heuristic block."""
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
        """Verify yaml unknown kind rejected by pydantic."""
        # Pydantic Literal["heuristic"] should reject "mlpppo" at validation
        # time (future learnable brain kinds would extend the literal type).
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
        """Verify copy preserves ids."""
        env = _make_env()
        new_env = env.copy()
        assert [p.predator_id for p in env.predators] == [p.predator_id for p in new_env.predators]

    def test_copy_preserves_positions(self) -> None:
        """Verify copy preserves positions."""
        env = _make_env()
        new_env = env.copy()
        assert [p.position for p in env.predators] == [p.position for p in new_env.predators]

    def test_copy_yields_independent_brain_instances(self) -> None:
        """Verify copy yields independent brain instances."""
        env = _make_env()
        new_env = env.copy()
        # Brain copies should be functionally equivalent but independent
        # objects (brain.copy() returns a fresh instance).
        for p_orig, p_copy in zip(env.predators, new_env.predators, strict=True):
            assert p_orig.brain is not p_copy.brain
            assert type(p_orig.brain) is type(p_copy.brain)
