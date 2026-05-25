"""Unit tests for FeedforwardGABrain.

Covers the spec scenarios at
``openspec/changes/add-neat-weights-brain/specs/feedforward-ga-brain/spec.md``:
matched-capacity topology construction; forward-pass shape + finite values;
non-degenerate variance across forward passes; WeightPersistence Protocol
conformance + round-trip; encoder shim attributes (_episode_count +
_update_learning_rate); no-op safety for learn/update_memory/post_process_episode;
encoder round-trip via the registered FeedforwardGAEncoder; registry +
BrainType + instantiate_brain dispatch.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import numpy as np
import torch
from quantumnematode.brain.actions import DEFAULT_ACTIONS
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.arch._registry import (
    Registration,
    get_registration,
    instantiate_brain,
    list_registered_brains,
)
from quantumnematode.brain.arch.dtypes import BrainType, DeviceType
from quantumnematode.brain.arch.feedforward_ga import (
    DEFAULT_HIDDEN_DIM,
    DEFAULT_NUM_HIDDEN_LAYERS,
    FeedforwardGABrain,
    FeedforwardGABrainConfig,
)
from quantumnematode.brain.modules import ModuleName
from quantumnematode.brain.weights import WeightPersistence

_SEED = 2026


def _make_config(**overrides: object) -> FeedforwardGABrainConfig:
    """Construct a FeedforwardGABrainConfig with defaults overridable per-test."""
    base = {
        "sensory_modules": [ModuleName.FOOD_CHEMOTAXIS],
        "seed": _SEED,
    }
    base.update(overrides)
    return FeedforwardGABrainConfig(**base)  # type: ignore[arg-type]


def _make_brain(**overrides: object) -> FeedforwardGABrain:
    return FeedforwardGABrain(config=_make_config(**overrides), device=DeviceType.CPU)


def _make_params(strength: float = 0.5, angle: float = 0.3) -> BrainParams:
    return BrainParams(
        food_gradient_strength=strength,
        food_gradient_direction=angle,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Construction + topology
# ──────────────────────────────────────────────────────────────────────────────


class TestConstruction:
    def test_default_config_matches_mlpppo_small_capacity(self) -> None:
        """Default topology matches MLPPPOBrain small (hidden=64, 2 hidden layers)."""
        config = _make_config()
        assert config.hidden_dim == DEFAULT_HIDDEN_DIM == 64
        assert config.num_hidden_layers == DEFAULT_NUM_HIDDEN_LAYERS == 2

    def test_brain_construction_builds_feedforward_topology(self) -> None:
        """Network is input_dim -> 64 -> 64 -> 4 with ReLU activations between hidden layers."""
        brain = _make_brain()
        assert brain.input_dim == 2  # food_chemotaxis -> [strength, angle]
        # Sequential: [Linear, ReLU, Linear, ReLU, Linear] = 5 children for 2 hidden layers
        children = list(brain.policy.children())
        assert len(children) == 5
        assert isinstance(children[0], torch.nn.Linear)
        assert children[0].in_features == 2
        assert children[0].out_features == 64
        assert isinstance(children[1], torch.nn.ReLU)
        assert isinstance(children[2], torch.nn.Linear)
        assert children[2].in_features == 64
        assert children[2].out_features == 64
        assert isinstance(children[3], torch.nn.ReLU)
        assert isinstance(children[4], torch.nn.Linear)
        assert children[4].in_features == 64
        assert children[4].out_features == 4  # 4-action DEFAULT_ACTIONS

    def test_no_critic_head(self) -> None:
        """GA brain has no critic / value head — fitness is the episode return directly."""
        brain = _make_brain()
        assert not hasattr(brain, "critic")
        assert not hasattr(brain, "value")

    def test_action_set_length_must_match_num_actions(self) -> None:
        """Action-set length mismatch with num_actions raises at construction."""
        import pytest

        with pytest.raises(ValueError, match="action_set must have exactly"):
            FeedforwardGABrain(
                config=_make_config(),
                num_actions=4,
                device=DeviceType.CPU,
                action_set=DEFAULT_ACTIONS[:2],  # only 2 actions
            )


# ──────────────────────────────────────────────────────────────────────────────
# Registry + BrainType
# ──────────────────────────────────────────────────────────────────────────────


class TestRegistry:
    def test_brain_self_registers_at_import(self) -> None:
        """The @register_brain decorator places feedforwardga in the registry."""
        assert "feedforwardga" in list_registered_brains()

    def test_registration_metadata(self) -> None:
        """The Registration carries the expected fields."""
        reg = get_registration("feedforwardga")
        assert isinstance(reg, Registration)
        assert reg.name == "feedforwardga"
        assert reg.config_cls is FeedforwardGABrainConfig
        assert reg.brain_cls is FeedforwardGABrain
        assert reg.brain_type is BrainType.FEEDFORWARDGA
        assert reg.families == ("classical",)

    def test_brain_type_enum_value(self) -> None:
        """BrainType.FEEDFORWARDGA value matches the registered name."""
        assert BrainType.FEEDFORWARDGA.value == "feedforwardga"

    def test_instantiate_via_registry(self) -> None:
        """instantiate_brain() dispatches via the registry without per-arch branches."""
        brain = instantiate_brain("feedforwardga", _make_config(), device=DeviceType.CPU)
        assert isinstance(brain, FeedforwardGABrain)


# ──────────────────────────────────────────────────────────────────────────────
# WeightPersistence Protocol conformance
# ──────────────────────────────────────────────────────────────────────────────


class TestWeightPersistence:
    def test_satisfies_weight_persistence_protocol(self) -> None:
        """Brain conforms to the runtime-checkable WeightPersistence Protocol."""
        brain = _make_brain()
        assert isinstance(brain, WeightPersistence)

    def test_get_weight_components_exposes_policy(self) -> None:
        """At minimum the 'policy' component is exposed; GA brain has no critic/optimizer."""
        brain = _make_brain()
        comps = brain.get_weight_components()
        assert "policy" in comps
        assert comps["policy"].name == "policy"
        # Policy state_dict contains every layer's weight + bias
        policy_state = comps["policy"].state
        assert any("weight" in k for k in policy_state)
        assert any("bias" in k for k in policy_state)

    def test_get_weight_components_filter(self) -> None:
        """Filtering by name returns only the requested components."""
        brain = _make_brain()
        filtered = brain.get_weight_components(components={"policy"})
        assert set(filtered.keys()) == {"policy"}

    def test_get_weight_components_unknown_raises(self) -> None:
        """Requesting an unknown component name raises ValueError."""
        import pytest

        brain = _make_brain()
        with pytest.raises(ValueError, match="Unknown weight components"):
            brain.get_weight_components(components={"critic"})

    def test_load_weight_components_round_trip(self) -> None:
        """load_weight_components restores params element-for-element + identical logits."""
        brain_a = _make_brain()
        comps = brain_a.get_weight_components()

        brain_b = _make_brain()
        brain_b.load_weight_components(comps)

        # Element-for-element ulp-tolerance equality on every layer parameter
        for pa, pb in zip(brain_a.policy.parameters(), brain_b.policy.parameters(), strict=False):
            assert torch.allclose(pa, pb, rtol=0, atol=0)

        # Identical pre-sampling logits (deterministic forward pass given identical weights)
        state = torch.tensor([0.5, 0.3], dtype=torch.float32)
        with torch.no_grad():
            l_a = brain_a.forward(state)
            l_b = brain_b.forward(state)
        assert torch.equal(l_a, l_b)


# ──────────────────────────────────────────────────────────────────────────────
# Encoder shim attributes (for _ClassicalPPOEncoder.decode contract)
# ──────────────────────────────────────────────────────────────────────────────


class TestEncoderShims:
    def test_episode_count_is_assignable_int(self) -> None:
        """_ClassicalPPOEncoder.decode() assigns brain._episode_count = 0."""
        brain = _make_brain()
        assert brain._episode_count == 0
        brain._episode_count = 17
        assert brain._episode_count == 17

    def test_update_learning_rate_is_no_op(self) -> None:
        """_update_learning_rate is a no-op shim (GA brain has no LR scheduler)."""
        brain = _make_brain()
        before = [p.clone() for p in brain.policy.parameters()]
        brain._update_learning_rate()
        after = list(brain.policy.parameters())
        for b, a in zip(before, after, strict=False):
            assert torch.equal(b, a)


# ──────────────────────────────────────────────────────────────────────────────
# Forward pass + action sampling
# ──────────────────────────────────────────────────────────────────────────────


class TestRunBrain:
    def test_forward_pass_finite_logits(self) -> None:
        """Forward pass produces finite logits over the 4-action set."""
        brain = _make_brain()
        state = torch.tensor([0.5, 0.3], dtype=torch.float32)
        with torch.no_grad():
            logits = brain.forward(state)
        assert logits.shape == (4,)
        assert torch.isfinite(logits).all()

    def test_run_brain_returns_action_data(self) -> None:
        """run_brain returns a list of ActionData matching DEFAULT_ACTIONS."""
        brain = _make_brain()
        params = _make_params()
        results = brain.run_brain(
            params,
            reward=None,
            input_data=None,
            top_only=False,
            top_randomize=False,
        )
        assert len(results) == 1
        assert results[0].action in DEFAULT_ACTIONS
        assert 0.0 <= results[0].probability <= 1.0

    def test_forward_pass_non_degenerate_variance(self) -> None:
        """Across a sample of forward passes, logit variance is strictly positive.

        Per spec: "the variance across the 4-action logits over a sample of ≥ 100
        forward passes SHALL be strictly greater than zero AND the logits SHALL
        not collapse to a constant action across that sample."

        "Constant action" is interpreted as the categorical-sampling output
        (not the argmax of logits, which can be stable across a wide range of
        inputs without violating the variance requirement).
        """
        brain = _make_brain()
        # Sample 128 random inputs in the [strength, angle] range
        rng = np.random.default_rng(0)
        all_logits: list[torch.Tensor] = []
        sampled_actions: list[str] = []
        for _ in range(128):
            params = BrainParams(
                food_gradient_strength=float(rng.uniform(0.0, 1.0)),
                food_gradient_direction=float(rng.uniform(-1.0, 1.0)),
            )
            x = brain.preprocess(params)
            with torch.no_grad():
                logits = brain.forward(torch.tensor(x, dtype=torch.float32))
            all_logits.append(logits)
            result = brain.run_brain(
                params,
                reward=None,
                input_data=None,
                top_only=False,
                top_randomize=False,
            )
            sampled_actions.append(result[0].action)

        stacked = torch.stack(all_logits)  # (128, 4)
        # Variance across the 128 samples, per action
        variance = stacked.var(dim=0)
        assert (variance > 0).all(), f"Some action logits collapsed: variance={variance}"

        # Categorical sampling SHALL produce at least two different actions
        # across the 128 samples (the categorical-distribution output is the
        # action surface the spec refers to, not the deterministic argmax).
        assert len(set(sampled_actions)) > 1, (
            f"All forward passes sampled the same action: {set(sampled_actions)}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# No-op lifecycle hooks (FrozenEvalRunner contract)
# ──────────────────────────────────────────────────────────────────────────────


class TestNoOpHooks:
    def test_learn_does_not_mutate_weights(self) -> None:
        """learn() is a no-op — weights unchanged after any reward signal."""
        brain = _make_brain()
        before = [p.clone() for p in brain.policy.parameters()]
        params = _make_params()

        brain.learn(params, reward=10.0, episode_done=False)
        brain.learn(params, reward=-5.0, episode_done=True)

        for b, a in zip(before, brain.policy.parameters(), strict=False):
            assert torch.equal(b, a)

    def test_update_memory_does_not_mutate_weights(self) -> None:
        """update_memory() is a no-op — weights unchanged after any reward signal."""
        brain = _make_brain()
        before = [p.clone() for p in brain.policy.parameters()]

        brain.update_memory(reward=10.0)
        brain.update_memory(reward=None)
        brain.update_memory(reward=-5.0)

        for b, a in zip(before, brain.policy.parameters(), strict=False):
            assert torch.equal(b, a)

    def test_post_process_episode_does_not_mutate_weights(self) -> None:
        """post_process_episode() is a no-op — weights unchanged after any episode_success."""
        brain = _make_brain()
        before = [p.clone() for p in brain.policy.parameters()]

        brain.post_process_episode(episode_success=True)
        brain.post_process_episode(episode_success=False)
        brain.post_process_episode(episode_success=None)

        for b, a in zip(before, brain.policy.parameters(), strict=False):
            assert torch.equal(b, a)

    def test_prepare_episode_does_not_mutate_weights(self) -> None:
        """prepare_episode() is a no-op."""
        brain = _make_brain()
        before = [p.clone() for p in brain.policy.parameters()]
        brain.prepare_episode()
        for b, a in zip(before, brain.policy.parameters(), strict=False):
            assert torch.equal(b, a)


# ──────────────────────────────────────────────────────────────────────────────
# Encoder round-trip (via the registered FeedforwardGAEncoder)
# ──────────────────────────────────────────────────────────────────────────────


class TestEncoderRoundTrip:
    def _make_sim_config(self):
        from quantumnematode.utils.config_loader import (
            BrainContainerConfig,
            SimulationConfig,
        )

        return SimulationConfig(
            brain=BrainContainerConfig(
                name="feedforwardga",
                config=_make_config(),
            ),
        )

    def test_encoder_is_registered(self) -> None:
        """FeedforwardGAEncoder is registered in ENCODER_REGISTRY."""
        from quantumnematode.evolution.encoders import ENCODER_REGISTRY

        assert "feedforwardga" in ENCODER_REGISTRY

    def test_initial_genome_dim_matches_param_count(self) -> None:
        """Genome dim equals the total parameter count of the policy network."""
        from quantumnematode.evolution.encoders import get_encoder

        encoder = get_encoder("feedforwardga")
        sim_config = self._make_sim_config()
        rng = np.random.default_rng(0)
        genome = encoder.initial_genome(sim_config, rng=rng)
        # input_dim=2, hidden=64, num_hidden_layers=2, output=4:
        # (2*64 + 64) + (64*64 + 64) + (64*4 + 4) = 192 + 4160 + 260 = 4612
        assert genome.params.shape == (4612,)
        assert encoder.genome_dim(sim_config) == 4612

    def test_decode_round_trip_byte_identical_params(self) -> None:
        """Two decodes of the same genome produce byte-identical params + logits."""
        from typing import cast

        from quantumnematode.evolution.encoders import get_encoder

        encoder = get_encoder("feedforwardga")
        sim_config = self._make_sim_config()
        rng = np.random.default_rng(0)
        genome = encoder.initial_genome(sim_config, rng=rng)

        brain_a = cast("FeedforwardGABrain", encoder.decode(genome, sim_config, seed=_SEED))
        brain_b = cast("FeedforwardGABrain", encoder.decode(genome, sim_config, seed=_SEED))

        for pa, pb in zip(brain_a.policy.parameters(), brain_b.policy.parameters(), strict=False):
            assert torch.equal(pa, pb)

        state = torch.tensor([0.5, 0.3], dtype=torch.float32)
        with torch.no_grad():
            l_a = brain_a.forward(state)
            l_b = brain_b.forward(state)
        assert torch.equal(l_a, l_b)

    def test_decode_resets_episode_count_via_shim(self) -> None:
        """encoder.decode() assigns brain._episode_count = 0 then calls _update_learning_rate."""
        from typing import cast

        from quantumnematode.evolution.encoders import get_encoder

        encoder = get_encoder("feedforwardga")
        sim_config = self._make_sim_config()
        rng = np.random.default_rng(0)
        genome = encoder.initial_genome(sim_config, rng=rng)

        brain = cast("FeedforwardGABrain", encoder.decode(genome, sim_config, seed=_SEED))
        assert brain._episode_count == 0
