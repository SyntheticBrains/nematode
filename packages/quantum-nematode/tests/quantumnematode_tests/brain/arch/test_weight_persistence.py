"""Tests for the unified weight persistence system."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

if TYPE_CHECKING:
    from pathlib import Path

import pytest
import torch
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.brain.arch.mlpppo import MLPPPOBrain, MLPPPOBrainConfig
from quantumnematode.brain.modules import ModuleName
from quantumnematode.brain.weights import (
    WeightPersistence,
    load_weights,
    save_weights,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mlpppo_config() -> MLPPPOBrainConfig:
    """Minimal MLP PPO config for testing."""
    return MLPPPOBrainConfig(
        seed=42,
        actor_hidden_dim=16,
        critic_hidden_dim=16,
        num_hidden_layers=1,
        rollout_buffer_size=32,
        sensory_modules=[ModuleName.FOOD_CHEMOTAXIS],
    )


@pytest.fixture
def mlpppo_brain(mlpppo_config: MLPPPOBrainConfig) -> MLPPPOBrain:
    """Create a small MLP PPO brain for testing."""
    return MLPPPOBrain(
        config=mlpppo_config,
        num_actions=4,
        device=DeviceType.CPU,
    )


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestWeightPersistenceProtocol:
    """Test that brains correctly implement the WeightPersistence protocol."""

    def test_mlpppo_implements_protocol(self, mlpppo_brain: MLPPPOBrain):
        """MLPPPOBrain should satisfy the WeightPersistence protocol."""
        assert isinstance(mlpppo_brain, WeightPersistence)

    def test_non_implementing_brain_not_detected(self):
        """A plain object should not satisfy the protocol."""
        obj = MagicMock(spec=[])
        assert not isinstance(obj, WeightPersistence)


# ---------------------------------------------------------------------------
# MLP PPO save/load round-trip
# ---------------------------------------------------------------------------


class TestMLPPPORoundTrip:
    """Test save and load round-trip for MLP PPO brain."""

    def test_round_trip_state_dicts_match(
        self,
        mlpppo_brain: MLPPPOBrain,
        mlpppo_config: MLPPPOBrainConfig,
        tmp_path: Path,
    ):
        """Saved and loaded brains should have identical state_dicts."""
        # Mutate state so it's not just initial values
        mlpppo_brain._episode_count = 42

        save_path = tmp_path / "weights.pt"
        save_weights(mlpppo_brain, save_path)

        # Create a fresh brain and load
        brain2 = MLPPPOBrain(
            config=mlpppo_config,
            num_actions=4,
            device=DeviceType.CPU,
        )
        load_weights(brain2, save_path)

        # Actor state_dicts should match
        for key in mlpppo_brain.actor.state_dict():
            torch.testing.assert_close(
                mlpppo_brain.actor.state_dict()[key],
                brain2.actor.state_dict()[key],
            )

        # Critic state_dicts should match
        for key in mlpppo_brain.critic.state_dict():
            torch.testing.assert_close(
                mlpppo_brain.critic.state_dict()[key],
                brain2.critic.state_dict()[key],
            )

        # Episode count should match
        assert brain2._episode_count == 42

    def test_action_probability_consistency(
        self,
        mlpppo_brain: MLPPPOBrain,
        mlpppo_config: MLPPPOBrainConfig,
        tmp_path: Path,
    ):
        """Loaded brain should produce identical action probabilities."""
        save_path = tmp_path / "weights.pt"
        save_weights(mlpppo_brain, save_path)

        brain2 = MLPPPOBrain(
            config=mlpppo_config,
            num_actions=4,
            device=DeviceType.CPU,
        )
        load_weights(brain2, save_path)

        # Same input should produce same output
        test_input = torch.randn(1, mlpppo_brain.input_dim)
        mlpppo_brain.actor.eval()
        brain2.actor.eval()

        with torch.no_grad():
            probs1 = torch.softmax(mlpppo_brain.actor(test_input), dim=-1)
            probs2 = torch.softmax(brain2.actor(test_input), dim=-1)

        torch.testing.assert_close(probs1, probs2)

    def test_training_continues_after_load(
        self,
        mlpppo_brain: MLPPPOBrain,
        mlpppo_config: MLPPPOBrainConfig,
        tmp_path: Path,
    ):
        """Weights should update when training after loading."""
        save_path = tmp_path / "weights.pt"
        save_weights(mlpppo_brain, save_path)

        brain2 = MLPPPOBrain(
            config=mlpppo_config,
            num_actions=4,
            device=DeviceType.CPU,
        )
        load_weights(brain2, save_path)

        # Snapshot initial weights
        initial_weights = {k: v.clone() for k, v in brain2.actor.state_dict().items()}

        # Perform a fake gradient step to verify training works
        test_input = torch.randn(1, brain2.input_dim)
        output = brain2.actor(test_input)
        loss = output.sum()
        brain2.optimizer.zero_grad()
        loss.backward()
        brain2.optimizer.step()

        # Weights should have changed
        current_state = brain2.actor.state_dict()
        changed = any(
            not torch.equal(val, current_state[key]) for key, val in initial_weights.items()
        )
        assert changed, "Weights should update after gradient step"


# ---------------------------------------------------------------------------
# Component filtering
# ---------------------------------------------------------------------------


class TestComponentFiltering:
    """Test component filtering on save and load."""

    def test_save_with_component_filter(
        self,
        mlpppo_brain: MLPPPOBrain,
        tmp_path: Path,
    ):
        """Saving with a filter should only include specified components."""
        save_path = tmp_path / "partial.pt"
        save_weights(mlpppo_brain, save_path, components={"policy"})

        checkpoint = torch.load(save_path, weights_only=True)
        assert "policy" in checkpoint
        assert "value" not in checkpoint
        assert "optimizer" not in checkpoint
        assert "_metadata" in checkpoint
        assert checkpoint["_metadata"]["components"] == ["policy"]

    def test_load_subset_leaves_others_unchanged(
        self,
        mlpppo_brain: MLPPPOBrain,
        mlpppo_config: MLPPPOBrainConfig,
        tmp_path: Path,
    ):
        """Loading a subset of components should leave others unchanged."""
        save_path = tmp_path / "weights.pt"
        save_weights(mlpppo_brain, save_path)

        brain2 = MLPPPOBrain(
            config=mlpppo_config,
            num_actions=4,
            device=DeviceType.CPU,
        )

        # Snapshot critic before load
        critic_before = {k: v.clone() for k, v in brain2.critic.state_dict().items()}

        # Load only policy
        load_weights(brain2, save_path, components={"policy"})

        # Critic should be unchanged
        critic_after = brain2.critic.state_dict()
        for key, val in critic_before.items():
            torch.testing.assert_close(val, critic_after[key])

        # Actor should match the saved brain
        for key in mlpppo_brain.actor.state_dict():
            torch.testing.assert_close(
                mlpppo_brain.actor.state_dict()[key],
                brain2.actor.state_dict()[key],
            )

    def test_unknown_component_raises_error(
        self,
        mlpppo_brain: MLPPPOBrain,
    ):
        """Requesting unknown components should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown weight components"):
            mlpppo_brain.get_weight_components(components={"nonexistent"})


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Test error handling in save/load."""

    def test_architecture_mismatch_raises_error(
        self,
        mlpppo_brain: MLPPPOBrain,
        tmp_path: Path,
    ):
        """Loading weights from a different architecture should raise error."""
        save_path = tmp_path / "weights.pt"
        save_weights(mlpppo_brain, save_path)

        # Create brain with different input_dim
        from quantumnematode.brain.modules import ModuleName

        config2 = MLPPPOBrainConfig(
            seed=42,
            actor_hidden_dim=16,
            critic_hidden_dim=16,
            num_hidden_layers=1,
            rollout_buffer_size=32,
            sensory_modules=[
                ModuleName.FOOD_CHEMOTAXIS,
                ModuleName.NOCICEPTION,
                ModuleName.MECHANOSENSATION,
            ],
        )
        brain2 = MLPPPOBrain(
            config=config2,
            num_actions=4,
            device=DeviceType.CPU,
        )

        with pytest.raises(RuntimeError, match="size mismatch"):
            load_weights(brain2, save_path)

    def test_load_file_not_found(
        self,
        mlpppo_brain: MLPPPOBrain,
        tmp_path: Path,
    ):
        """Loading from nonexistent path should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            load_weights(mlpppo_brain, tmp_path / "nonexistent.pt")

    def test_load_non_implementing_brain_raises_type_error(self, tmp_path: Path):
        """Loading on a brain without WeightPersistence should raise TypeError."""
        fake_brain = MagicMock(spec=[])
        with pytest.raises(TypeError, match="does not implement"):
            load_weights(fake_brain, tmp_path / "weights.pt")

    def test_save_non_implementing_brain_noop(self, tmp_path: Path):
        """Saving a non-implementing brain should no-op without error."""
        fake_brain = MagicMock(spec=[])
        result = save_weights(fake_brain, tmp_path / "weights.pt")
        assert result is None
        assert not (tmp_path / "weights.pt").exists()


# ---------------------------------------------------------------------------
# File system
# ---------------------------------------------------------------------------


class TestFileSystem:
    """Test file system behavior."""

    def test_save_creates_parent_directories(
        self,
        mlpppo_brain: MLPPPOBrain,
        tmp_path: Path,
    ):
        """Save should create nested directories that don't exist."""
        deep_path = tmp_path / "a" / "b" / "c" / "weights.pt"
        assert not deep_path.parent.exists()

        save_weights(mlpppo_brain, deep_path)
        assert deep_path.exists()


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestMetadata:
    """Test metadata in saved weight files."""

    def test_metadata_contents(
        self,
        mlpppo_brain: MLPPPOBrain,
        tmp_path: Path,
    ):
        """Saved file should contain expected metadata fields."""
        mlpppo_brain._episode_count = 100
        save_path = tmp_path / "weights.pt"
        save_weights(mlpppo_brain, save_path)

        checkpoint = torch.load(save_path, weights_only=True)
        meta = checkpoint["_metadata"]

        assert meta["brain_type"] == "MLPPPOBrain"
        assert "saved_at" in meta
        assert set(meta["components"]) == {
            "policy",
            "value",
            "optimizer",
            "training_state",
        }
        assert isinstance(meta["shapes"], dict)
        assert len(meta["shapes"]) > 0
        # Verify shapes are lists of ints
        for shape in meta["shapes"].values():
            assert isinstance(shape, list)
            assert all(isinstance(s, int) for s in shape)
        assert meta["episode_count"] == 100

    def test_brain_type_mismatch_warning(
        self,
        mlpppo_brain: MLPPPOBrain,
        mlpppo_config: MLPPPOBrainConfig,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ):
        """Loading weights saved by a different brain type should warn."""
        save_path = tmp_path / "weights.pt"
        save_weights(mlpppo_brain, save_path)

        # Tamper with the brain_type in the saved file
        checkpoint = torch.load(save_path, weights_only=True)
        checkpoint["_metadata"]["brain_type"] = "SomeOtherBrain"
        torch.save(checkpoint, save_path)

        brain2 = MLPPPOBrain(
            config=mlpppo_config,
            num_actions=4,
            device=DeviceType.CPU,
        )
        with caplog.at_level(logging.WARNING):
            load_weights(brain2, save_path)

        assert "mismatch" in caplog.text.lower()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCLI:
    """Test CLI argument parsing."""

    def test_cli_flags_accepted(self):
        """Argparse should accept --load-weights and --save-weights."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--load-weights", type=str, default=None)
        parser.add_argument("--save-weights", type=str, default=None)

        args = parser.parse_args(
            [
                "--load-weights",
                "input.pt",
                "--save-weights",
                "output.pt",
            ],
        )
        assert args.load_weights == "input.pt"
        assert args.save_weights == "output.pt"

    def test_cli_flags_default_none(self):
        """Weight flags should default to None."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--load-weights", type=str, default=None)
        parser.add_argument("--save-weights", type=str, default=None)

        args = parser.parse_args([])
        assert args.load_weights is None
        assert args.save_weights is None


# ---------------------------------------------------------------------------
# PPO buffer reset
# ---------------------------------------------------------------------------


class TestBufferReset:
    """Test that PPO buffer is reset after loading weights."""

    def test_buffer_reset_after_load(
        self,
        mlpppo_brain: MLPPPOBrain,
        mlpppo_config: MLPPPOBrainConfig,
        tmp_path: Path,
    ):
        """Buffer should be empty after loading weights."""
        import numpy as np

        # Add some experience to the buffer
        state = np.array([0.5, 0.3], dtype=np.float32)
        mlpppo_brain.buffer.add(
            state=state,
            action=1,
            log_prob=torch.tensor(-0.5),
            value=torch.tensor([0.8]),
            reward=1.0,
            done=False,
        )
        assert len(mlpppo_brain.buffer) > 0

        # Save and reload
        save_path = tmp_path / "weights.pt"
        save_weights(mlpppo_brain, save_path)

        brain2 = MLPPPOBrain(
            config=mlpppo_config,
            num_actions=4,
            device=DeviceType.CPU,
        )
        # Add experience to brain2's buffer too
        brain2.buffer.add(
            state=state,
            action=0,
            log_prob=torch.tensor(-0.3),
            value=torch.tensor([0.5]),
            reward=0.5,
            done=False,
        )
        assert len(brain2.buffer) > 0

        load_weights(brain2, save_path)
        assert len(brain2.buffer) == 0, "Buffer should be cleared after load"
