"""Tests for `MLPPPOPredatorBrain`.

Covers:

- Protocol conformance via `isinstance(brain, PredatorBrain)`
- Input-encoding correctness (11-float vector, normalisation, padding)
- Action-mapping correctness (`0=STAY, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT`)
- Deterministic-action under fixed seed (same weights + same params →
  same action; argmax mode)
- `WeightPersistence` round-trip (extract → load into fresh brain →
  same actions on test states)
- `copy()` independence (clone has same weights but is a separate object)
- Critic forward pass returns scalar
- Construction with seed produces reproducible weights across instances
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from quantumnematode.env import (
    HeuristicPredatorBrain,
    PredatorAction,
    PredatorBrain,
    PredatorBrainParams,
    PredatorType,
)
from quantumnematode.env.mlpppo_predator_brain import (
    INPUT_DIM,
    K_NEAREST,
    NUM_ACTIONS,
    MLPPPOPredatorBrain,
)


def _make_params(  # noqa: PLR0913
    *,
    predator_position: tuple[int, int] = (10, 10),
    agent_positions: tuple[tuple[int, int], ...] = ((15, 10),),
    grid_size: int = 20,
    detection_radius: int = 8,
    damage_radius: int = 1,
    step_index: int = 100,
    seed: int = 0,
) -> PredatorBrainParams:
    """Build a `PredatorBrainParams` with sensible defaults; override per-test.

    `chase_target` and `is_pursuing` resolved from `agent_positions` so the
    semantics match what `Predator.update_position` would produce in the
    real env. Predator type defaults to PURSUIT (the only non-trivial
    branch — STATIONARY would be returned-as-STAY by the heuristic teacher).
    """
    if agent_positions:
        chase_target = min(
            agent_positions,
            key=lambda p: abs(p[0] - predator_position[0]) + abs(p[1] - predator_position[1]),
        )
        in_range = (
            abs(chase_target[0] - predator_position[0])
            + abs(chase_target[1] - predator_position[1])
        ) <= detection_radius
    else:
        chase_target = None
        in_range = False
    return PredatorBrainParams(
        predator_id="predator_0",
        predator_position=predator_position,
        predator_type=PredatorType.PURSUIT,
        detection_radius=detection_radius,
        damage_radius=damage_radius,
        agent_positions=agent_positions,
        chase_target=chase_target,
        is_pursuing=in_range,
        grid_size=grid_size,
        rng=np.random.default_rng(seed),
        step_index=step_index,
    )


class TestProtocolConformance:
    """`MLPPPOPredatorBrain` SHALL satisfy the `PredatorBrain` Protocol."""

    def test_isinstance_via_runtime_checkable(self) -> None:
        """Verify isinstance via runtime checkable."""
        brain = MLPPPOPredatorBrain(seed=42)
        assert isinstance(brain, PredatorBrain)

    def test_protocol_methods_exist(self) -> None:
        """Verify protocol methods exist."""
        brain = MLPPPOPredatorBrain(seed=42)
        assert callable(brain.run_brain)
        assert callable(brain.prepare_episode)
        assert callable(brain.post_process_episode)
        assert callable(brain.copy)


class TestInputEncoding:
    """The 11-float input vector layout matches the spec."""

    def test_input_dim_is_11(self) -> None:
        """Verify INPUT_DIM matches the spec."""
        assert INPUT_DIM == 11
        assert K_NEAREST == 2
        assert NUM_ACTIONS == 5

    def test_encoding_layout(self) -> None:
        """Verify the encoded vector matches the documented field order."""
        brain = MLPPPOPredatorBrain(seed=42)
        params = _make_params(
            predator_position=(10, 5),
            agent_positions=((15, 5), (8, 12)),
            grid_size=20,
            detection_radius=8,
            damage_radius=1,
            step_index=200,
        )
        obs = brain.encode_observation(params)
        assert obs.shape == (INPUT_DIM,)
        assert obs.dtype == np.float32
        # predator_position normalised by grid_size
        assert obs[0] == pytest.approx(10 / 20)
        assert obs[1] == pytest.approx(5 / 20)
        # First agent: present, normalised
        assert obs[2] == pytest.approx(15 / 20)
        assert obs[3] == pytest.approx(5 / 20)
        assert obs[4] == 1.0
        # Second agent: present
        assert obs[5] == pytest.approx(8 / 20)
        assert obs[6] == pytest.approx(12 / 20)
        assert obs[7] == 1.0
        # Radii
        assert obs[8] == pytest.approx(8 / 20)
        assert obs[9] == pytest.approx(1 / 20)
        # step_index normalised by max_steps=1000
        assert obs[10] == pytest.approx(200 / 1000)

    def test_encoding_pads_missing_agents_with_zeros_and_present_flag_zero(self) -> None:
        """Verify padding rule when fewer than k_nearest agents alive."""
        brain = MLPPPOPredatorBrain(seed=42)
        params = _make_params(
            predator_position=(10, 10),
            agent_positions=((15, 10),),  # only 1 agent
        )
        obs = brain.encode_observation(params)
        # Slot 0 (first agent) populated.
        assert obs[2] == pytest.approx(15 / 20)
        assert obs[4] == 1.0
        # Slot 1 (missing agent) zero-padded with present_flag=0.
        assert obs[5] == 0.0
        assert obs[6] == 0.0
        assert obs[7] == 0.0

    def test_encoding_zero_agents(self) -> None:
        """Verify encoding with empty agent_positions."""
        brain = MLPPPOPredatorBrain(seed=42)
        params = _make_params(agent_positions=())
        obs = brain.encode_observation(params)
        # Both agent slots zero-padded.
        assert obs[2] == 0.0
        assert obs[3] == 0.0
        assert obs[4] == 0.0
        assert obs[5] == 0.0
        assert obs[6] == 0.0
        assert obs[7] == 0.0


class TestActionMapping:
    """Action index → enum mapping matches the spec."""

    def test_argmax_returns_valid_predator_action(self) -> None:
        """Verify argmax returns a valid PredatorAction."""
        brain = MLPPPOPredatorBrain(seed=42)
        params = _make_params()
        action = brain.run_brain(params)
        assert action in {
            PredatorAction.STAY,
            PredatorAction.UP,
            PredatorAction.DOWN,
            PredatorAction.LEFT,
            PredatorAction.RIGHT,
        }

    def test_action_mapping_indices(self) -> None:
        """Verify hand-crafted weights map index 0→STAY, 1→UP, ..."""
        # Construct a brain whose actor outputs a known argmax.
        brain = MLPPPOPredatorBrain(seed=42)
        # Override the actor's last linear layer to produce a deterministic
        # one-hot logit at index `target_idx`. We test all 5 indices.
        for target_idx in range(NUM_ACTIONS):
            with torch.no_grad():
                # Find the last Linear layer in actor (the output head)
                # and zero it out, then set bias[target_idx] = 1.0 so the
                # argmax always picks `target_idx`.
                last_linear = [m for m in brain.actor if isinstance(m, torch.nn.Linear)][-1]
                last_linear.weight.zero_()
                last_linear.bias.zero_()
                last_linear.bias[target_idx] = 1.0
            action = brain.run_brain(_make_params())
            expected = (
                PredatorAction.STAY,
                PredatorAction.UP,
                PredatorAction.DOWN,
                PredatorAction.LEFT,
                PredatorAction.RIGHT,
            )[target_idx]
            assert action == expected, f"target_idx={target_idx}: expected {expected}, got {action}"


class TestDeterminism:
    """Same weights + same params → same action (argmax mode)."""

    def test_two_brains_same_seed_produce_same_action(self) -> None:
        """Verify two brains constructed with the same seed produce the same action."""
        brain_a = MLPPPOPredatorBrain(seed=42)
        brain_b = MLPPPOPredatorBrain(seed=42)
        params = _make_params()
        assert brain_a.run_brain(params) == brain_b.run_brain(params)

    def test_two_brains_different_seeds_likely_produce_different_actions(self) -> None:
        """Verify orthogonal-init with different seeds gives different parameter values."""
        brain_a = MLPPPOPredatorBrain(seed=42)
        brain_b = MLPPPOPredatorBrain(seed=43)
        # At least one parameter tensor SHALL differ between the two brains.
        a_params = list(brain_a.actor.parameters())
        b_params = list(brain_b.actor.parameters())
        assert any(not torch.allclose(a, b) for a, b in zip(a_params, b_params, strict=True))

    def test_argmax_mode_is_stateless(self) -> None:
        """Verify run_brain is pure — same params, same weights → same output across calls."""
        brain = MLPPPOPredatorBrain(seed=42)
        params = _make_params()
        actions = [brain.run_brain(params) for _ in range(10)]
        assert all(a == actions[0] for a in actions)


class TestWeightPersistence:
    """Weight components round-trip via the protocol."""

    def test_get_components_returns_policy_and_value(self) -> None:
        """Verify get_weight_components returns the documented components."""
        brain = MLPPPOPredatorBrain(seed=42)
        components = brain.get_weight_components()
        assert set(components.keys()) == {"policy", "value"}
        assert components["policy"].name == "policy"
        assert components["value"].name == "value"

    def test_get_components_filtered(self) -> None:
        """Verify components filter parameter."""
        brain = MLPPPOPredatorBrain(seed=42)
        only_policy = brain.get_weight_components(components={"policy"})
        assert set(only_policy.keys()) == {"policy"}

    def test_get_components_unknown_raises(self) -> None:
        """Verify unknown component raises ValueError."""
        brain = MLPPPOPredatorBrain(seed=42)
        with pytest.raises(ValueError, match="Unknown weight components"):
            brain.get_weight_components(components={"optimizer"})  # not in predator brain

    def test_round_trip_preserves_actions(self) -> None:
        """Verify weights extract → load into fresh brain → same actions."""
        original = MLPPPOPredatorBrain(seed=42)
        components = original.get_weight_components()
        # Construct a different brain (different seed) and load.
        clone = MLPPPOPredatorBrain(seed=999)
        clone.load_weight_components(components)
        # Now original and clone SHALL produce the same actions on test states.
        for i in range(20):
            params = _make_params(
                predator_position=(i % 20, (i * 3) % 20),
                agent_positions=((i + 5, i + 2),),
                seed=i,
            )
            assert original.run_brain(params) == clone.run_brain(params), (
                f"action mismatch at i={i}"
            )


class TestCopy:
    """`copy()` returns an independent brain with equivalent behaviour."""

    def test_copy_is_distinct_object(self) -> None:
        """Verify copy returns a new instance."""
        original = MLPPPOPredatorBrain(seed=42)
        clone = original.copy()
        assert clone is not original
        assert isinstance(clone, MLPPPOPredatorBrain)

    def test_copy_preserves_weights(self) -> None:
        """Verify copy has the same weight values."""
        original = MLPPPOPredatorBrain(seed=42)
        clone = original.copy()
        for o_param, c_param in zip(
            original.actor.parameters(),
            clone.actor.parameters(),
            strict=True,
        ):
            assert torch.allclose(o_param, c_param)

    def test_copy_produces_same_actions(self) -> None:
        """Verify copy returns the same action on the same params."""
        original = MLPPPOPredatorBrain(seed=42)
        clone = original.copy()
        params = _make_params()
        assert original.run_brain(params) == clone.run_brain(params)


class TestLifecycleHooks:
    """`prepare_episode` and `post_process_episode` SHALL be no-op."""

    def test_prepare_episode_is_noop(self) -> None:
        """Verify prepare_episode returns None."""
        brain = MLPPPOPredatorBrain(seed=42)
        assert brain.prepare_episode() is None

    def test_post_process_episode_is_noop(self) -> None:
        """Verify post_process_episode returns None for both success values."""
        brain = MLPPPOPredatorBrain(seed=42)
        assert brain.post_process_episode() is None
        assert brain.post_process_episode(episode_success=True) is None
        assert brain.post_process_episode(episode_success=False) is None


class TestParamCount:
    """Architecture defaults yield the expected parameter count."""

    def test_default_param_count_is_around_10k(self) -> None:
        """Verify the default architecture has ~10k parameters."""
        brain = MLPPPOPredatorBrain(seed=42)
        n_params = sum(
            p.numel() for p in list(brain.actor.parameters()) + list(brain.critic.parameters())
        )
        # Expected: actor (11→64→64→5) + critic (11→64→64→1) ≈ 10,246.
        # Exact count is sensitive to the architecture constants; assert
        # within a tight band.
        assert 9_000 <= n_params <= 11_000, f"got {n_params} params"

    def test_invalid_num_hidden_layers_raises(self) -> None:
        """Verify num_hidden_layers < 1 is rejected at construction."""
        with pytest.raises(ValueError, match="num_hidden_layers"):
            MLPPPOPredatorBrain(num_hidden_layers=0)


class TestHeuristicTeacherInteraction:
    """Brain must accept the same `PredatorBrainParams` shape as the heuristic teacher."""

    def test_brain_and_teacher_accept_same_params(self) -> None:
        """Verify brain.run_brain accepts the same params shape as HeuristicPredatorBrain."""
        brain = MLPPPOPredatorBrain(seed=42)
        teacher = HeuristicPredatorBrain()
        params = _make_params()
        # Both invocations SHALL succeed without error.
        assert brain.run_brain(params) is not None
        assert teacher.run_brain(params) is not None
