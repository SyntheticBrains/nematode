"""Unit tests for the HybridClassicalBrain architecture (QSNN ablation)."""

import numpy as np
import pytest
import torch
from quantumnematode.brain.actions import ActionData
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.arch.dtypes import BrainType, DeviceType
from quantumnematode.brain.arch.hybridclassical import (
    HybridClassicalBrain,
    HybridClassicalBrainConfig,
    _CortexRolloutBuffer,
)
from quantumnematode.brain.modules import ModuleName


class TestHybridClassicalBrainConfig:
    """Test cases for HybridClassicalBrain configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = HybridClassicalBrainConfig()
        assert config.reflex_hidden_dim == 16
        assert config.training_stage == 1
        assert config.cortex_hidden_dim == 64
        assert config.cortex_num_layers == 2
        assert config.num_modes == 3
        assert config.reflex_lr == 0.01
        assert config.ppo_buffer_size == 512
        assert config.reflex_weights_path is None
        assert config.sensory_modules is None

    def test_custom_config(self):
        """Test custom configuration values."""
        config = HybridClassicalBrainConfig(
            reflex_hidden_dim=32,
            training_stage=2,
            cortex_hidden_dim=128,
            ppo_buffer_size=1024,
            reflex_weights_path="exports/stage1/reflex_weights.pt",
        )
        assert config.reflex_hidden_dim == 32
        assert config.training_stage == 2
        assert config.cortex_hidden_dim == 128
        assert config.ppo_buffer_size == 1024
        assert config.reflex_weights_path == "exports/stage1/reflex_weights.pt"

    def test_validation_training_stage_invalid(self):
        """Test validation rejects invalid training stage."""
        with pytest.raises(ValueError, match="training_stage must be 1, 2, or 3"):
            HybridClassicalBrainConfig(training_stage=0)
        with pytest.raises(ValueError, match="training_stage must be 1, 2, or 3"):
            HybridClassicalBrainConfig(training_stage=4)

    def test_validation_num_reinforce_epochs(self):
        """Test validation rejects zero reinforce epochs."""
        with pytest.raises(ValueError, match="num_reinforce_epochs must be >= 1"):
            HybridClassicalBrainConfig(num_reinforce_epochs=0)

    def test_validation_reflex_hidden_dim(self):
        """Test validation rejects zero reflex hidden dim."""
        with pytest.raises(ValueError, match="reflex_hidden_dim must be >= 1"):
            HybridClassicalBrainConfig(reflex_hidden_dim=0)

    def test_config_with_sensory_modules(self):
        """Test config with sensory modules specified."""
        config = HybridClassicalBrainConfig(
            sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
        )
        assert config.sensory_modules is not None
        assert len(config.sensory_modules) == 2


class TestHybridClassicalBrainInit:
    """Test brain instantiation and component dimensions."""

    @pytest.fixture
    def brain_stage1(self):
        """Create a stage 1 brain for testing."""
        config = HybridClassicalBrainConfig(
            training_stage=1,
            seed=42,
        )
        return HybridClassicalBrain(config=config, num_actions=4)

    @pytest.fixture
    def brain_stage2(self):
        """Create a stage 2 brain for testing."""
        config = HybridClassicalBrainConfig(
            training_stage=2,
            seed=42,
        )
        return HybridClassicalBrain(config=config, num_actions=4)

    def test_reflex_mlp_dimensions(self, brain_stage1):
        """Test reflex MLP input/output dimensions."""
        # First layer: Linear(2, 16)
        first_layer = next(iter(brain_stage1.reflex_mlp.children()))
        assert first_layer.in_features == 2
        assert first_layer.out_features == 16
        # Last layer: Linear(16, 4)
        last_layer = list(brain_stage1.reflex_mlp.children())[-1]
        assert last_layer.in_features == 16
        assert last_layer.out_features == 4

    def test_reflex_param_count(self, brain_stage1):
        """Test reflex MLP has ~116 parameters (comparable to QSNN's 92)."""
        total = sum(p.numel() for p in brain_stage1.reflex_mlp.parameters())
        # Linear(2, 16) = 2*16+16 = 48, Linear(16, 4) = 16*4+4 = 68 → 116
        assert total == 116

    def test_reflex_requires_grad_stage1(self, brain_stage1):
        """Test reflex weights require gradient in stage 1."""
        for param in brain_stage1.reflex_mlp.parameters():
            assert param.requires_grad

    def test_reflex_frozen_stage2(self, brain_stage2):
        """Test reflex weights are frozen in stage 2."""
        for param in brain_stage2.reflex_mlp.parameters():
            assert not param.requires_grad

    def test_cortex_actor_output_dim(self, brain_stage1):
        """Test cortex actor output dimension is num_motor + num_modes."""
        # Actor output = num_motor (4) + num_modes (3) = 7
        last_layer = list(brain_stage1.cortex_actor.children())[-1]
        assert last_layer.out_features == 7

    def test_cortex_critic_output_dim(self, brain_stage1):
        """Test cortex critic output dimension is 1."""
        last_layer = list(brain_stage1.cortex_critic.children())[-1]
        assert last_layer.out_features == 1

    def test_cortex_actor_input_dim(self, brain_stage1):
        """Test cortex actor input dimension matches legacy mode."""
        first_layer = next(iter(brain_stage1.cortex_actor.children()))
        assert first_layer.in_features == 2  # legacy mode

    def test_action_set(self, brain_stage1):
        """Test action set has correct length."""
        assert len(brain_stage1.action_set) == 4


class TestReflexForwardPass:
    """Test classical reflex MLP forward pass."""

    @pytest.fixture
    def brain(self):
        """Create a brain for reflex forward pass testing."""
        config = HybridClassicalBrainConfig(
            training_stage=1,
            seed=42,
        )
        return HybridClassicalBrain(config=config, num_actions=4)

    def test_reflex_forward_shape(self, brain):
        """Test reflex forward output shape and range."""
        features = np.array([0.5, 0.3], dtype=np.float32)
        motor_probs = brain._reflex_forward(features)
        assert motor_probs.shape == (4,)
        assert np.all(motor_probs >= 0)
        assert np.all(motor_probs <= 1)

    def test_reflex_forward_differentiable_shape(self, brain):
        """Test differentiable reflex forward output shape and grad tracking."""
        features = np.array([0.5, 0.3], dtype=np.float32)
        motor_probs = brain._reflex_forward_differentiable(features)
        assert motor_probs.shape == (4,)
        assert motor_probs.requires_grad


class TestCortexForwardPass:
    """Test cortex MLP forward pass."""

    @pytest.fixture
    def brain(self):
        """Create a stage 2 brain for cortex forward pass testing."""
        config = HybridClassicalBrainConfig(
            training_stage=2,
            seed=42,
        )
        return HybridClassicalBrain(config=config, num_actions=4)

    def test_cortex_forward_output_split(self, brain):
        """Test cortex forward splits into action biases and mode logits."""
        sensory_t = torch.randn(2)  # legacy 2-feature input
        action_biases, mode_logits = brain._cortex_forward(sensory_t)
        assert action_biases.shape == (4,)
        assert mode_logits.shape == (3,)

    def test_cortex_value_scalar(self, brain):
        """Test cortex value estimate is a scalar."""
        sensory_t = torch.randn(2)
        value = brain._cortex_value(sensory_t)
        assert value.dim() == 0  # scalar


class TestFusionMechanism:
    """Test mode-gated fusion."""

    def test_fusion_math(self):
        """Test fusion with forage mode dominant produces high reflex trust."""
        config = HybridClassicalBrainConfig(
            training_stage=2,
            seed=42,
        )
        brain = HybridClassicalBrain(config=config, num_actions=4)

        reflex_logits = torch.tensor([1.0, -1.0, 0.5, -0.5])
        action_biases = torch.tensor([0.1, 0.2, -0.1, 0.0])
        mode_logits = torch.tensor([10.0, -10.0, -10.0])  # forage mode dominant

        final_logits, reflex_trust, _mode_probs = brain._fuse(
            reflex_logits,
            action_biases,
            mode_logits,
        )

        # With forage mode dominant, reflex_trust should be ~1.0
        assert reflex_trust > 0.99
        # Final logits should be close to reflex_logits + action_biases
        expected = reflex_logits * reflex_trust + action_biases
        torch.testing.assert_close(final_logits, expected)

    def test_fusion_low_trust(self):
        """Test fusion with evade mode dominant produces low reflex trust."""
        config = HybridClassicalBrainConfig(
            training_stage=2,
            seed=42,
        )
        brain = HybridClassicalBrain(config=config, num_actions=4)

        reflex_logits = torch.tensor([1.0, -1.0, 0.5, -0.5])
        action_biases = torch.tensor([0.1, 0.2, -0.1, 0.0])
        mode_logits = torch.tensor([-10.0, 10.0, -10.0])  # evade mode dominant

        final_logits, reflex_trust, _mode_probs = brain._fuse(
            reflex_logits,
            action_biases,
            mode_logits,
        )

        # With evade mode dominant, reflex_trust should be ~0.0
        assert reflex_trust < 0.01
        # Final logits should be close to just action_biases
        torch.testing.assert_close(final_logits, action_biases, atol=0.02, rtol=0.0)

    def test_stage1_bypass(self):
        """Stage 1 should bypass cortex entirely."""
        config = HybridClassicalBrainConfig(
            training_stage=1,
            seed=42,
        )
        brain = HybridClassicalBrain(config=config, num_actions=4)
        params = BrainParams(gradient_strength=0.5, gradient_direction=0.0)
        actions = brain.run_brain(params, top_only=False, top_randomize=False)
        assert len(actions) == 1
        assert isinstance(actions[0], ActionData)
        # Stage 1 should not have fusion diagnostics
        assert len(brain._episode_reflex_trusts) == 0


class TestStageAwareTraining:
    """Test that correct optimizers are active per stage."""

    def test_stage1_reflex_trains(self):
        """Test reflex weights change during stage 1 REINFORCE training."""
        config = HybridClassicalBrainConfig(
            training_stage=1,
            reinforce_window_size=5,
            seed=42,
        )
        brain = HybridClassicalBrain(config=config, num_actions=4)
        params = BrainParams(gradient_strength=0.5, gradient_direction=0.0)

        weights_before = {
            name: p.clone().detach() for name, p in brain.reflex_mlp.named_parameters()
        }

        # Run a few steps
        for i in range(6):
            brain.run_brain(params, top_only=False, top_randomize=False)
            brain.learn(params, reward=1.0, episode_done=(i == 5))

        # Reflex weights should have changed (REINFORCE update)
        changed = False
        for name, p in brain.reflex_mlp.named_parameters():
            if not torch.allclose(weights_before[name], p):
                changed = True
                break
        assert changed

    def test_stage2_reflex_frozen(self):
        """Test reflex weights remain unchanged during stage 2 PPO training."""
        config = HybridClassicalBrainConfig(
            training_stage=2,
            ppo_buffer_size=5,
            ppo_minibatches=1,
            seed=42,
        )
        brain = HybridClassicalBrain(config=config, num_actions=4)
        params = BrainParams(gradient_strength=0.5, gradient_direction=0.0)

        weights_before = {
            name: p.clone().detach() for name, p in brain.reflex_mlp.named_parameters()
        }

        # Run enough steps to trigger PPO update
        for i in range(6):
            brain.run_brain(params, top_only=False, top_randomize=False)
            brain.learn(params, reward=1.0, episode_done=(i == 5))

        # Reflex weights should NOT change in stage 2
        for name, p in brain.reflex_mlp.named_parameters():
            torch.testing.assert_close(weights_before[name], p)


class TestReinforceUpdate:
    """Test reflex REINFORCE update."""

    def test_reinforce_runs_without_error(self):
        """Test REINFORCE update completes without error."""
        config = HybridClassicalBrainConfig(
            training_stage=1,
            reinforce_window_size=3,
            num_reinforce_epochs=1,
            seed=42,
        )
        brain = HybridClassicalBrain(config=config, num_actions=4)
        params = BrainParams(gradient_strength=0.5, gradient_direction=0.0)

        for i in range(4):
            brain.run_brain(params, top_only=False, top_randomize=False)
            brain.learn(params, reward=0.5, episode_done=(i == 3))


class TestPPOBuffer:
    """Test cortex PPO rollout buffer."""

    def test_buffer_fill_and_trigger(self):
        """Test PPO buffer fills and triggers update correctly."""
        config = HybridClassicalBrainConfig(
            training_stage=2,
            ppo_buffer_size=4,
            ppo_minibatches=1,
            ppo_epochs=1,
            seed=42,
        )
        brain = HybridClassicalBrain(config=config, num_actions=4)
        params = BrainParams(gradient_strength=0.5, gradient_direction=0.0)

        for i in range(5):
            brain.run_brain(params, top_only=False, top_randomize=False)
            brain.learn(params, reward=0.5, episode_done=(i == 4))

        # Buffer should have been reset after triggering
        assert len(brain.ppo_buffer) == 0

    def test_rollout_buffer_compute_returns(self):
        """Test rollout buffer computes returns and advantages correctly."""
        device = torch.device("cpu")
        buffer = _CortexRolloutBuffer(buffer_size=4, device=device)

        for i in range(4):
            buffer.add(
                state=np.array([0.5, 0.3]),
                action=i % 4,
                log_prob=torch.tensor(-1.0),
                value=torch.tensor(0.5),
                reward=1.0,
                done=(i == 3),
            )

        returns, advantages = buffer.compute_returns_and_advantages(
            last_value=torch.tensor(0.0),
            gamma=0.99,
            gae_lambda=0.95,
        )
        assert returns.shape == (4,)
        assert advantages.shape == (4,)


class TestEpisodeReset:
    """Test episode boundary handling."""

    def test_state_reset(self):
        """Test state and buffers are cleared after episode."""
        config = HybridClassicalBrainConfig(
            training_stage=1,
            seed=42,
        )
        brain = HybridClassicalBrain(config=config, num_actions=4)
        params = BrainParams(gradient_strength=0.5, gradient_direction=0.0)

        # Run an episode
        for i in range(3):
            brain.run_brain(params, top_only=False, top_randomize=False)
            brain.learn(params, reward=0.5, episode_done=(i == 2))

        # After episode, buffers should be cleared
        assert len(brain.episode_rewards) == 0
        assert len(brain.episode_actions) == 0
        assert brain._step_count == 0


class TestBrainRegistration:
    """Test brain registration in factory and config loader."""

    def test_brain_type_enum(self):
        """Test brain type enum value is correct."""
        assert BrainType.HYBRID_CLASSICAL.value == "hybridclassical"

    def test_brain_type_in_classical_types(self):
        """Test brain type is in classical brain types set."""
        from quantumnematode.brain.arch.dtypes import CLASSICAL_BRAIN_TYPES

        assert BrainType.HYBRID_CLASSICAL in CLASSICAL_BRAIN_TYPES

    def test_config_loader_mapping(self):
        """Test config loader maps 'hybridclassical' to correct config class."""
        from quantumnematode.utils.config_loader import BRAIN_CONFIG_MAP

        assert "hybridclassical" in BRAIN_CONFIG_MAP
        assert BRAIN_CONFIG_MAP["hybridclassical"] is HybridClassicalBrainConfig

    def test_brain_factory(self):
        """Test brain factory creates correct brain type."""
        from quantumnematode.optimizers.gradient_methods import (
            GradientCalculationMethod,
        )
        from quantumnematode.optimizers.learning_rate import ConstantLearningRate
        from quantumnematode.utils.brain_factory import setup_brain_model
        from quantumnematode.utils.config_loader import ParameterInitializerConfig

        config = HybridClassicalBrainConfig(seed=42)
        brain = setup_brain_model(
            brain_type=BrainType.HYBRID_CLASSICAL,
            brain_config=config,
            shots=100,
            qubits=2,
            device=DeviceType.CPU,
            learning_rate=ConstantLearningRate(learning_rate=0.01),
            gradient_method=GradientCalculationMethod.CLIP,
            gradient_max_norm=None,
            parameter_initializer_config=ParameterInitializerConfig(),
        )
        assert isinstance(brain, HybridClassicalBrain)


class TestWeightPersistence:
    """Test reflex weight save/load functionality."""

    def test_save_and_load_round_trip(self, tmp_path, monkeypatch):
        """Test reflex weights survive a save/load round trip."""
        monkeypatch.chdir(tmp_path)
        config1 = HybridClassicalBrainConfig(training_stage=1, seed=42)
        brain1 = HybridClassicalBrain(config=config1, num_actions=4)
        brain1._save_reflex_weights("test_session")

        save_path = tmp_path / "exports" / "test_session" / "reflex_weights.pt"
        assert save_path.exists()

        # Load into new brain
        config2 = HybridClassicalBrainConfig(
            training_stage=2,
            reflex_weights_path=str(save_path),
            seed=43,
        )
        brain2 = HybridClassicalBrain(config=config2, num_actions=4)

        # Weights should match
        for p1, p2 in zip(
            brain1.reflex_mlp.parameters(),
            brain2.reflex_mlp.parameters(),
            strict=False,
        ):
            torch.testing.assert_close(p1, p2)

    def test_load_missing_file_raises(self):
        """Test loading from missing file raises FileNotFoundError."""
        config = HybridClassicalBrainConfig(
            training_stage=2,
            reflex_weights_path="/nonexistent/path/reflex_weights.pt",
        )
        with pytest.raises(FileNotFoundError):
            HybridClassicalBrain(config=config, num_actions=4)

    def test_stage2_without_weights_logs_warning(self, caplog):
        """Test stage 2 without weights path logs a warning."""
        config = HybridClassicalBrainConfig(
            training_stage=2,
            seed=42,
        )
        with caplog.at_level("WARNING"):
            HybridClassicalBrain(config=config, num_actions=4)
        assert "no reflex_weights_path specified" in caplog.text.lower()

    def test_save_reflex_weights(self, tmp_path, monkeypatch):
        """Test saving reflex weights creates correct file."""
        monkeypatch.chdir(tmp_path)
        config = HybridClassicalBrainConfig(training_stage=1, seed=42)
        brain = HybridClassicalBrain(config=config, num_actions=4)
        brain._save_reflex_weights("test_session")

        save_path = tmp_path / "exports" / "test_session" / "reflex_weights.pt"
        assert save_path.exists()

        loaded = torch.load(save_path, weights_only=True)
        # Should be a state_dict with layer keys
        assert "0.weight" in loaded  # First Linear layer weight
        assert "0.bias" in loaded
        assert "2.weight" in loaded  # Second Linear layer weight (after ReLU at index 1)
        assert "2.bias" in loaded

    def test_save_and_load_cortex_round_trip(self, tmp_path, monkeypatch):
        """Test cortex weights survive a save/load round trip."""
        monkeypatch.chdir(tmp_path)
        config = HybridClassicalBrainConfig(training_stage=2, seed=42)
        brain1 = HybridClassicalBrain(config=config, num_actions=4)
        brain1._save_cortex_weights("test_session")

        save_path = tmp_path / "exports" / "test_session" / "cortex_weights.pt"
        assert save_path.exists()

        # Load into new brain
        config2 = HybridClassicalBrainConfig(
            training_stage=3,
            cortex_weights_path=str(save_path),
            seed=43,
        )
        brain2 = HybridClassicalBrain(config=config2, num_actions=4)

        # Cortex actor/critic weights should match
        for p1, p2 in zip(
            brain1.cortex_actor.parameters(),
            brain2.cortex_actor.parameters(),
            strict=False,
        ):
            torch.testing.assert_close(p1, p2)
        for p1, p2 in zip(
            brain1.cortex_critic.parameters(),
            brain2.cortex_critic.parameters(),
            strict=False,
        ):
            torch.testing.assert_close(p1, p2)

    def test_load_cortex_missing_file_raises(self):
        """Test loading cortex from missing file raises FileNotFoundError."""
        config = HybridClassicalBrainConfig(
            training_stage=3,
            cortex_weights_path="/nonexistent/path/cortex_weights.pt",
        )
        with pytest.raises(FileNotFoundError):
            HybridClassicalBrain(config=config, num_actions=4)

    def test_cortex_auto_save_stage2(self, tmp_path, monkeypatch):
        """Test cortex weights are auto-saved during stage 2 training."""
        monkeypatch.chdir(tmp_path)
        config = HybridClassicalBrainConfig(
            training_stage=2,
            ppo_buffer_size=4,
            ppo_minibatches=1,
            ppo_epochs=1,
            seed=42,
        )
        brain = HybridClassicalBrain(config=config, num_actions=4)
        brain.set_session_id("auto_save_test")
        params = BrainParams(gradient_strength=0.5, gradient_direction=0.0)

        for i in range(5):
            brain.run_brain(params, top_only=False, top_randomize=False)
            brain.learn(params, reward=0.5, episode_done=(i == 4))

        # Cortex weights should have been auto-saved
        save_path = tmp_path / "exports" / "auto_save_test" / "cortex_weights.pt"
        assert save_path.exists()


class TestCortexLRScheduling:
    """Test cortex learning rate warmup and decay."""

    def test_lr_warmup(self):
        """Test LR warmup from low to base value."""
        config = HybridClassicalBrainConfig(
            training_stage=2,
            cortex_actor_lr=0.001,
            cortex_lr_warmup_episodes=10,
            cortex_lr_warmup_start=0.0001,
            seed=42,
        )
        brain = HybridClassicalBrain(config=config, num_actions=4)

        # At episode 0, LR should be warmup_start
        assert brain._get_cortex_lr() == pytest.approx(0.0001)

        # Simulate 5 episodes
        brain._episode_count = 5
        expected_lr = 0.0001 + (0.001 - 0.0001) * 0.5
        assert brain._get_cortex_lr() == pytest.approx(expected_lr)

        # At episode 10, warmup done, LR should be base
        brain._episode_count = 10
        assert brain._get_cortex_lr() == pytest.approx(0.001)

    def test_lr_decay(self):
        """Test LR decay after warmup."""
        config = HybridClassicalBrainConfig(
            training_stage=2,
            cortex_actor_lr=0.001,
            cortex_lr_warmup_episodes=10,
            cortex_lr_warmup_start=0.0001,
            cortex_lr_decay_episodes=20,
            cortex_lr_decay_end=0.0001,
            seed=42,
        )
        brain = HybridClassicalBrain(config=config, num_actions=4)

        # At episode 10 (warmup done), LR should be base
        brain._episode_count = 10
        assert brain._get_cortex_lr() == pytest.approx(0.001)

        # At episode 20 (halfway through decay), LR should be midpoint
        brain._episode_count = 20
        expected_lr = 0.001 + (0.0001 - 0.001) * 0.5
        assert brain._get_cortex_lr() == pytest.approx(expected_lr)

        # At episode 30 (decay done), LR should be decay_end
        brain._episode_count = 30
        assert brain._get_cortex_lr() == pytest.approx(0.0001)

        # Beyond decay, LR stays at decay_end
        brain._episode_count = 100
        assert brain._get_cortex_lr() == pytest.approx(0.0001)

    def test_no_lr_scheduling(self):
        """Test that LR is flat when no scheduling configured."""
        config = HybridClassicalBrainConfig(
            training_stage=2,
            cortex_actor_lr=0.001,
            seed=42,
        )
        brain = HybridClassicalBrain(config=config, num_actions=4)

        assert not brain.cortex_lr_scheduling_enabled
        brain._episode_count = 100
        assert brain._get_cortex_lr() == pytest.approx(0.001)

    def test_lr_update_changes_optimizer(self):
        """Test that _update_cortex_learning_rate changes optimizer LR."""
        config = HybridClassicalBrainConfig(
            training_stage=2,
            cortex_actor_lr=0.001,
            cortex_lr_warmup_episodes=10,
            cortex_lr_warmup_start=0.0001,
            seed=42,
        )
        brain = HybridClassicalBrain(config=config, num_actions=4)

        # Initial LR should be warmup_start
        actor_lr = brain.cortex_actor_optimizer.param_groups[0]["lr"]
        assert actor_lr == pytest.approx(0.0001)

        # After 10 episodes, update should set to base LR
        brain._episode_count = 10
        brain._update_cortex_learning_rate()
        actor_lr = brain.cortex_actor_optimizer.param_groups[0]["lr"]
        assert actor_lr == pytest.approx(0.001)


class TestBrainCopy:
    """Test brain copy functionality."""

    def test_copy_preserves_weights(self):
        """Test copied brain has identical weights."""
        config = HybridClassicalBrainConfig(training_stage=1, seed=42)
        brain = HybridClassicalBrain(config=config, num_actions=4)
        copy = brain.copy()

        for p1, p2 in zip(
            brain.reflex_mlp.parameters(),
            copy.reflex_mlp.parameters(),
            strict=False,
        ):
            torch.testing.assert_close(p1, p2)
        assert brain.baseline == copy.baseline
        assert brain._episode_count == copy._episode_count

    def test_copy_is_independent(self):
        """Test copied brain is independent of original."""
        config = HybridClassicalBrainConfig(training_stage=1, seed=42)
        brain = HybridClassicalBrain(config=config, num_actions=4)
        copy = brain.copy()

        # Modify original
        with torch.no_grad():
            for p in brain.reflex_mlp.parameters():
                p.fill_(999.0)

        # Copy should be unaffected
        for p1, p2 in zip(
            brain.reflex_mlp.parameters(),
            copy.reflex_mlp.parameters(),
            strict=False,
        ):
            assert not torch.allclose(p1, p2)
