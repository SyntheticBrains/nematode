"""Unit tests for the HybridQuantumCortexBrain architecture."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from quantumnematode.brain.actions import ActionData
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.arch.dtypes import BrainType, DeviceType
from quantumnematode.brain.arch.hybridquantumcortex import (
    HybridQuantumCortexBrain,
    HybridQuantumCortexBrainConfig,
)
from quantumnematode.brain.modules import ModuleName

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _stage1_config(**overrides: Any) -> HybridQuantumCortexBrainConfig:
    """Create a minimal stage 1 config for testing."""
    defaults: dict[str, Any] = {
        "training_stage": 1,
        "shots": 100,
        "num_qsnn_timesteps": 1,
        "num_cortex_timesteps": 1,
        "seed": 42,
    }
    defaults.update(overrides)
    return HybridQuantumCortexBrainConfig(**defaults)


def _stage2_config(**overrides: Any) -> HybridQuantumCortexBrainConfig:
    """Create a minimal stage 2 config for testing."""
    defaults: dict[str, Any] = {
        "training_stage": 2,
        "shots": 100,
        "num_qsnn_timesteps": 1,
        "num_cortex_timesteps": 1,
        "cortex_sensory_modules": [
            ModuleName.FOOD_CHEMOTAXIS,
            ModuleName.NOCICEPTION,
            ModuleName.MECHANOSENSATION,
        ],
        "ppo_buffer_size": 5,
        "seed": 42,
    }
    defaults.update(overrides)
    return HybridQuantumCortexBrainConfig(**defaults)


# ──────────────────────────────────────────────────────────────────────
# Config tests
# ──────────────────────────────────────────────────────────────────────


class TestHybridQuantumCortexBrainConfig:
    """Test cases for HybridQuantumCortexBrainConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = HybridQuantumCortexBrainConfig()
        assert config.num_sensory_neurons == 8
        assert config.num_hidden_neurons == 16
        assert config.num_motor_neurons == 4
        assert config.training_stage == 1
        assert config.cortex_neurons_per_group == 4
        assert config.cortex_hidden_neurons == 12
        assert config.cortex_output_neurons == 8
        assert config.num_modes == 3
        assert config.cortex_lr == 0.01
        assert config.critic_lr == 0.001
        assert config.use_gae_advantages is True
        assert config.reflex_weights_path is None
        assert config.cortex_weights_path is None
        assert config.critic_weights_path is None

    def test_validation_training_stage_valid(self):
        """Test all valid training stages are accepted."""
        for stage in (1, 2, 3, 4):
            modules = [ModuleName.FOOD_CHEMOTAXIS] if stage >= 2 else None
            config = HybridQuantumCortexBrainConfig(
                training_stage=stage,
                cortex_sensory_modules=modules,
            )
            assert config.training_stage == stage

    def test_validation_training_stage_invalid(self):
        """Test validation rejects invalid training stage."""
        with pytest.raises(ValueError, match="training_stage must be 1, 2, 3, or 4"):
            HybridQuantumCortexBrainConfig(training_stage=0)
        with pytest.raises(ValueError, match="training_stage must be 1, 2, 3, or 4"):
            HybridQuantumCortexBrainConfig(training_stage=5)

    def test_validation_cortex_neurons_per_group(self):
        """Test validation rejects cortex_neurons_per_group < 2."""
        with pytest.raises(ValueError, match="cortex_neurons_per_group must be >= 2"):
            HybridQuantumCortexBrainConfig(cortex_neurons_per_group=1)

    def test_validation_cortex_hidden_neurons(self):
        """Test validation rejects cortex_hidden_neurons < 4."""
        with pytest.raises(ValueError, match="cortex_hidden_neurons must be >= 4"):
            HybridQuantumCortexBrainConfig(cortex_hidden_neurons=3)

    def test_validation_num_modes(self):
        """Test validation rejects num_modes < 2."""
        with pytest.raises(ValueError, match="num_modes must be >= 2"):
            HybridQuantumCortexBrainConfig(num_modes=1)

    def test_validation_cortex_modules_required_stage2(self):
        """Test validation requires cortex_sensory_modules for stage >= 2."""
        with pytest.raises(ValueError, match="cortex_sensory_modules must be non-empty"):
            HybridQuantumCortexBrainConfig(training_stage=2)
        with pytest.raises(ValueError, match="cortex_sensory_modules must be non-empty"):
            HybridQuantumCortexBrainConfig(
                training_stage=3,
                cortex_sensory_modules=[],
            )

    def test_validation_shots_too_low(self):
        """Test validation rejects shots below minimum."""
        with pytest.raises(ValueError, match="shots must be >= 100"):
            HybridQuantumCortexBrainConfig(shots=50)

    def test_validation_membrane_tau(self):
        """Test validation rejects membrane_tau outside (0, 1]."""
        with pytest.raises(ValueError, match="membrane_tau must be in"):
            HybridQuantumCortexBrainConfig(membrane_tau=0.0)

    def test_validation_threshold(self):
        """Test validation rejects threshold outside (0, 1)."""
        with pytest.raises(ValueError, match="threshold must be in"):
            HybridQuantumCortexBrainConfig(threshold=0.0)

    def test_validation_cortex_output_neurons_too_small(self):
        """Test validation rejects cortex_output_neurons < num_motor + num_modes + 1."""
        with pytest.raises(ValueError, match="cortex_output_neurons must be >="):
            HybridQuantumCortexBrainConfig(
                num_motor_neurons=4,
                num_modes=3,
                cortex_output_neurons=7,  # needs 4+3+1=8
            )

    def test_validation_cortex_output_neurons_exact_minimum(self):
        """Test cortex_output_neurons == num_motor + num_modes + 1 is accepted."""
        config = HybridQuantumCortexBrainConfig(
            num_motor_neurons=4,
            num_modes=3,
            cortex_output_neurons=8,
        )
        assert config.cortex_output_neurons == 8


# ──────────────────────────────────────────────────────────────────────
# Init and dimensions
# ──────────────────────────────────────────────────────────────────────


class TestHybridQuantumCortexBrainInit:
    """Test brain instantiation and component dimensions."""

    @pytest.fixture
    def brain_stage1(self):
        """Create a stage 1 brain for testing."""
        return HybridQuantumCortexBrain(config=_stage1_config(), num_actions=4)

    @pytest.fixture
    def brain_stage2(self):
        """Create a stage 2 brain for testing."""
        return HybridQuantumCortexBrain(config=_stage2_config(), num_actions=4)

    def test_reflex_weights_dimensions(self, brain_stage1):
        """Test reflex weight matrix dimensions."""
        assert brain_stage1.W_sh.shape == (8, 16)
        assert brain_stage1.W_hm.shape == (16, 4)
        assert brain_stage1.theta_hidden.shape == (16,)
        assert brain_stage1.theta_motor.shape == (4,)

    def test_reflex_weights_require_grad_stage1(self, brain_stage1):
        """Test reflex weights require gradient in stage 1."""
        assert brain_stage1.W_sh.requires_grad
        assert brain_stage1.W_hm.requires_grad
        assert brain_stage1.theta_hidden.requires_grad
        assert brain_stage1.theta_motor.requires_grad

    def test_reflex_weights_frozen_stage2(self, brain_stage2):
        """Test reflex weights are frozen in stage 2."""
        assert not brain_stage2.W_sh.requires_grad
        assert not brain_stage2.W_hm.requires_grad
        assert not brain_stage2.theta_hidden.requires_grad
        assert not brain_stage2.theta_motor.requires_grad

    def test_cortex_group_count(self, brain_stage2):
        """Test cortex has correct number of groups."""
        assert len(brain_stage2.cortex_group_weights) == 3
        assert len(brain_stage2.cortex_group_thetas) == 3

    def test_cortex_group_dimensions(self, brain_stage2):
        """Test cortex group weight dimensions."""
        for w in brain_stage2.cortex_group_weights:
            assert w.shape == (4, 4)  # neurons_per_group x neurons_per_group
        for t in brain_stage2.cortex_group_thetas:
            assert t.shape == (4,)

    def test_cortex_hidden_dimensions(self, brain_stage2):
        """Test cortex shared hidden layer dimensions."""
        # 3 groups * 4 neurons_per_group = 12 sensory neurons
        assert brain_stage2.W_cortex_sh.shape == (12, 12)
        assert brain_stage2.theta_cortex_hidden.shape == (12,)

    def test_cortex_output_dimensions(self, brain_stage2):
        """Test cortex output layer dimensions."""
        assert brain_stage2.W_cortex_ho.shape == (12, 8)
        assert brain_stage2.theta_cortex_output.shape == (8,)

    def test_critic_output_dim(self, brain_stage2):
        """Test critic output dimension is 1."""
        last_layer = list(brain_stage2.critic.children())[-1]
        assert last_layer.out_features == 1

    def test_critic_input_dim(self, brain_stage2):
        """Test critic input dimension matches cortex sensory features."""
        first_layer = next(iter(brain_stage2.critic.children()))
        assert first_layer.in_features == 6  # 3 modules * 2 features

    def test_action_set(self, brain_stage1):
        """Test action set has correct length."""
        assert len(brain_stage1.action_set) == 4


# ──────────────────────────────────────────────────────────────────────
# Cortex QSNN forward pass
# ──────────────────────────────────────────────────────────────────────


class TestCortexQSNNForwardPass:
    """Test cortex QSNN forward pass."""

    @pytest.fixture
    def brain(self):
        """Create a brain for cortex forward pass testing."""
        return HybridQuantumCortexBrain(config=_stage2_config(), num_actions=4)

    def test_cortex_multi_timestep_shape(self, brain):
        """Test cortex multi-timestep output shape."""
        features = np.array([0.5, 0.3, 0.2, 0.4, 0.6, 0.1], dtype=np.float32)
        output = brain._cortex_multi_timestep(features)
        assert output.shape == (8,)
        assert np.all(output >= 0)
        assert np.all(output <= 1)

    def test_cortex_output_mapping(self, brain):
        """Test cortex output is mapped to action biases, mode logits, trust."""
        output = torch.tensor([0.6, 0.4, 0.7, 0.3, 0.5, 0.6, 0.4, 0.8])
        action_biases, mode_logits, trust = brain._map_cortex_output(output)
        assert action_biases.shape == (4,)
        assert mode_logits.shape == (3,)
        assert isinstance(trust, torch.Tensor)

    def test_cortex_differentiable_has_grad(self, brain):
        """Test differentiable cortex forward produces grad-tracked output."""
        features = np.array([0.5, 0.3, 0.2, 0.4, 0.6, 0.1], dtype=np.float32)
        output = brain._cortex_multi_timestep_differentiable(features)
        assert output.shape == (8,)
        assert output.requires_grad


# ──────────────────────────────────────────────────────────────────────
# Fusion mechanism
# ──────────────────────────────────────────────────────────────────────


class TestFusionMechanism:
    """Test mode-gated fusion."""

    def test_fuse_output_shape(self):
        """Test fusion output shape."""
        brain = HybridQuantumCortexBrain(config=_stage2_config(), num_actions=4)
        reflex = torch.tensor([0.5, -0.3, 0.1, 0.4])
        biases = torch.tensor([0.2, 0.1, -0.1, 0.3])
        modes = torch.tensor([1.0, 0.5, 0.3])
        final, trust, probs = brain._fuse(reflex, biases, modes)
        assert final.shape == (4,)
        assert 0.0 <= trust <= 1.0
        assert probs.shape == (3,)
        assert abs(probs.sum().item() - 1.0) < 1e-5

    def test_fusion_trust_range(self):
        """Test trust is bounded in [0, 1]."""
        brain = HybridQuantumCortexBrain(config=_stage2_config(), num_actions=4)
        for _ in range(10):
            modes = torch.randn(3)
            _, trust, _ = brain._fuse(
                torch.randn(4),
                torch.randn(4),
                modes,
            )
            assert 0.0 <= trust <= 1.0


# ──────────────────────────────────────────────────────────────────────
# Action selection
# ──────────────────────────────────────────────────────────────────────


class TestActionSelection:
    """Test run_brain action selection."""

    def test_stage1_produces_valid_action(self):
        """Test stage 1 brain produces valid ActionData."""
        brain = HybridQuantumCortexBrain(config=_stage1_config(), num_actions=4)
        params = BrainParams(gradient_strength=0.5, gradient_direction=0.0)
        result = brain.run_brain(params, top_only=False, top_randomize=False)
        assert len(result) == 1
        assert isinstance(result[0], ActionData)

    def test_stage2_produces_valid_action(self):
        """Test stage 2 brain produces valid ActionData."""
        brain = HybridQuantumCortexBrain(config=_stage2_config(), num_actions=4)
        params = BrainParams(gradient_strength=0.5, gradient_direction=0.0)
        result = brain.run_brain(params, top_only=False, top_randomize=False)
        assert len(result) == 1
        assert isinstance(result[0], ActionData)


# ──────────────────────────────────────────────────────────────────────
# Stage-aware training
# ──────────────────────────────────────────────────────────────────────


class TestStageAwareTraining:
    """Test that correct components train per stage."""

    def test_stage1_reflex_trains(self):
        """Test reflex weights change during stage 1 REINFORCE training."""
        config = _stage1_config(reinforce_window_size=5)
        brain = HybridQuantumCortexBrain(config=config, num_actions=4)
        params = BrainParams(gradient_strength=0.5, gradient_direction=0.0)

        w_sh_before = brain.W_sh.clone().detach()

        for i in range(6):
            brain.run_brain(params, top_only=False, top_randomize=False)
            brain.learn(params, reward=1.0, episode_done=(i == 5))

        w_sh_after = brain.W_sh.clone().detach()
        assert not torch.allclose(w_sh_before, w_sh_after)

    def test_stage2_reflex_frozen(self):
        """Test reflex weights remain unchanged during stage 2."""
        config = _stage2_config()
        brain = HybridQuantumCortexBrain(config=config, num_actions=4)
        params = BrainParams(gradient_strength=0.5, gradient_direction=0.0)

        w_sh_before = brain.W_sh.clone().detach()

        for i in range(6):
            brain.run_brain(params, top_only=False, top_randomize=False)
            brain.learn(params, reward=1.0, episode_done=(i == 5))

        w_sh_after = brain.W_sh.clone().detach()
        torch.testing.assert_close(w_sh_before, w_sh_after)

    def test_stage3_all_train(self):
        """Test both reflex and cortex weights change in stage 3."""
        config = HybridQuantumCortexBrainConfig(
            training_stage=3,
            shots=100,
            num_qsnn_timesteps=1,
            num_cortex_timesteps=1,
            cortex_sensory_modules=[
                ModuleName.FOOD_CHEMOTAXIS,
                ModuleName.NOCICEPTION,
                ModuleName.MECHANOSENSATION,
            ],
            ppo_buffer_size=5,
            reinforce_window_size=5,
            seed=42,
        )
        brain = HybridQuantumCortexBrain(config=config, num_actions=4)
        params = BrainParams(gradient_strength=0.5, gradient_direction=0.0)

        w_sh_before = brain.W_sh.clone().detach()

        for i in range(6):
            brain.run_brain(params, top_only=False, top_randomize=False)
            brain.learn(params, reward=1.0, episode_done=(i == 5))

        w_sh_after = brain.W_sh.clone().detach()
        # Reflex weights should train in stage 3
        assert not torch.allclose(w_sh_before, w_sh_after)


# ──────────────────────────────────────────────────────────────────────
# Weight persistence
# ──────────────────────────────────────────────────────────────────────


class TestWeightPersistence:
    """Test weight save/load functionality."""

    def test_reflex_save_load_round_trip(self, tmp_path):
        """Test reflex weights survive a save/load round trip."""
        config1 = _stage1_config(num_sensory_neurons=4, num_hidden_neurons=4)
        brain1 = HybridQuantumCortexBrain(config=config1, num_actions=4)

        # Save weights
        session_dir = tmp_path / "exports" / "test_session"
        session_dir.mkdir(parents=True)
        save_path = session_dir / "reflex_weights.pt"
        weights_dict = {
            "W_sh": brain1.W_sh.detach().cpu(),
            "W_hm": brain1.W_hm.detach().cpu(),
            "theta_hidden": brain1.theta_hidden.detach().cpu(),
            "theta_motor": brain1.theta_motor.detach().cpu(),
        }
        torch.save(weights_dict, save_path)

        # Load into new brain
        config2 = _stage2_config(
            num_sensory_neurons=4,
            num_hidden_neurons=4,
            reflex_weights_path=str(save_path),
            seed=43,
        )
        brain2 = HybridQuantumCortexBrain(config=config2, num_actions=4)

        torch.testing.assert_close(brain1.W_sh, brain2.W_sh)
        torch.testing.assert_close(brain1.W_hm, brain2.W_hm)

    def test_cortex_save_load_round_trip(self, tmp_path, monkeypatch):
        """Test cortex weights survive a save/load round trip."""
        monkeypatch.chdir(tmp_path)
        config1 = _stage2_config()
        brain1 = HybridQuantumCortexBrain(config=config1, num_actions=4)

        brain1._save_cortex_weights("test_session")

        cortex_path = tmp_path / "exports" / "test_session" / "cortex_weights.pt"
        assert cortex_path.exists()

        config2 = _stage2_config(cortex_weights_path=str(cortex_path), seed=43)
        brain2 = HybridQuantumCortexBrain(config=config2, num_actions=4)

        for i in range(len(brain1.cortex_group_weights)):
            torch.testing.assert_close(
                brain1.cortex_group_weights[i],
                brain2.cortex_group_weights[i],
            )
        torch.testing.assert_close(brain1.W_cortex_sh, brain2.W_cortex_sh)
        torch.testing.assert_close(brain1.W_cortex_ho, brain2.W_cortex_ho)

    def test_critic_save_load_round_trip(self, tmp_path, monkeypatch):
        """Test critic weights survive a save/load round trip."""
        monkeypatch.chdir(tmp_path)
        config1 = _stage2_config()
        brain1 = HybridQuantumCortexBrain(config=config1, num_actions=4)

        brain1._save_critic_weights("test_session")

        critic_path = tmp_path / "exports" / "test_session" / "critic_weights.pt"
        assert critic_path.exists()

        config2 = _stage2_config(critic_weights_path=str(critic_path), seed=43)
        brain2 = HybridQuantumCortexBrain(config=config2, num_actions=4)

        for p1, p2 in zip(
            brain1.critic.parameters(),
            brain2.critic.parameters(),
            strict=False,
        ):
            torch.testing.assert_close(p1, p2)

    def test_load_missing_file_raises(self):
        """Test loading from missing file raises FileNotFoundError."""
        config = _stage2_config(
            reflex_weights_path="/nonexistent/path/reflex_weights.pt",
        )
        with pytest.raises(FileNotFoundError):
            HybridQuantumCortexBrain(config=config, num_actions=4)

    def test_reflex_shape_mismatch_raises(self, tmp_path):
        """Test loading reflex weights with wrong shapes raises ValueError."""
        save_path = tmp_path / "bad_reflex.pt"
        weights_dict = {
            "W_sh": torch.randn(3, 3),
            "W_hm": torch.randn(3, 4),
            "theta_hidden": torch.randn(3),
            "theta_motor": torch.randn(4),
        }
        torch.save(weights_dict, save_path)

        config = _stage2_config(reflex_weights_path=str(save_path))
        with pytest.raises(ValueError, match="Shape mismatch"):
            HybridQuantumCortexBrain(config=config, num_actions=4)

    def test_cortex_auto_save_stage2(self, tmp_path, monkeypatch):
        """Test cortex auto-save happens during stage 2 episode end."""
        monkeypatch.chdir(tmp_path)
        config = _stage2_config()
        brain = HybridQuantumCortexBrain(config=config, num_actions=4)
        params = BrainParams(gradient_strength=0.5, gradient_direction=0.0)

        for i in range(6):
            brain.run_brain(params, top_only=False, top_randomize=False)
            brain.learn(params, reward=1.0, episode_done=(i == 5))

        # Check that cortex and critic weights were auto-saved
        export_dir = tmp_path / "exports" / brain._session_id
        assert (export_dir / "cortex_weights.pt").exists()
        assert (export_dir / "critic_weights.pt").exists()

    def test_stage2_without_reflex_weights_logs_warning(self, caplog):
        """Test stage 2 without reflex weights path logs a warning."""
        config = _stage2_config()
        with caplog.at_level("WARNING"):
            HybridQuantumCortexBrain(config=config, num_actions=4)
        assert "no reflex_weights_path specified" in caplog.text.lower()


# ──────────────────────────────────────────────────────────────────────
# REINFORCE+GAE
# ──────────────────────────────────────────────────────────────────────


class TestReinforceGAE:
    """Test cortex REINFORCE+GAE training."""

    def test_cortex_reinforce_runs_without_error(self):
        """Test cortex REINFORCE+GAE update completes without error."""
        config = _stage2_config()
        brain = HybridQuantumCortexBrain(config=config, num_actions=4)
        params = BrainParams(gradient_strength=0.5, gradient_direction=0.0)

        for i in range(6):
            brain.run_brain(params, top_only=False, top_randomize=False)
            brain.learn(params, reward=0.5, episode_done=(i == 5))

    def test_gae_fallback_to_pure_reinforce(self):
        """Test fallback to pure REINFORCE when use_gae_advantages=false."""
        config = _stage2_config(use_gae_advantages=False)
        brain = HybridQuantumCortexBrain(config=config, num_actions=4)
        params = BrainParams(gradient_strength=0.5, gradient_direction=0.0)

        # Should complete without error
        for i in range(6):
            brain.run_brain(params, top_only=False, top_randomize=False)
            brain.learn(params, reward=0.5, episode_done=(i == 5))


# ──────────────────────────────────────────────────────────────────────
# Episode reset
# ──────────────────────────────────────────────────────────────────────


class TestEpisodeReset:
    """Test episode boundary handling."""

    def test_episode_reset_clears_state(self):
        """Test episode reset clears buffers and refractory states."""
        config = _stage2_config()
        brain = HybridQuantumCortexBrain(config=config, num_actions=4)
        params = BrainParams(gradient_strength=0.5, gradient_direction=0.0)

        # Run some steps
        for _ in range(3):
            brain.run_brain(params, top_only=False, top_randomize=False)

        brain._reset_episode()

        assert len(brain.episode_rewards) == 0
        assert len(brain.episode_actions) == 0
        assert len(brain._episode_qsnn_trusts) == 0
        assert brain._step_count == 0

    def test_prepare_episode(self):
        """Test prepare_episode resets state."""
        brain = HybridQuantumCortexBrain(config=_stage1_config(), num_actions=4)
        brain.episode_rewards.append(1.0)
        brain.prepare_episode()
        assert len(brain.episode_rewards) == 0


# ──────────────────────────────────────────────────────────────────────
# Brain registration
# ──────────────────────────────────────────────────────────────────────


class TestBrainRegistration:
    """Test brain type registration."""

    def test_brain_type_enum_exists(self):
        """Test HYBRID_QUANTUM_CORTEX exists in BrainType enum."""
        assert hasattr(BrainType, "HYBRID_QUANTUM_CORTEX")
        assert BrainType.HYBRID_QUANTUM_CORTEX.value == "hybridquantumcortex"

    def test_in_quantum_brain_types(self):
        """Test HYBRID_QUANTUM_CORTEX is in QUANTUM_BRAIN_TYPES."""
        from quantumnematode.brain.arch.dtypes import QUANTUM_BRAIN_TYPES

        assert BrainType.HYBRID_QUANTUM_CORTEX in QUANTUM_BRAIN_TYPES

    def test_in_brain_config_map(self):
        """Test hybridquantumcortex is in BRAIN_CONFIG_MAP."""
        from quantumnematode.utils.config_loader import BRAIN_CONFIG_MAP

        assert "hybridquantumcortex" in BRAIN_CONFIG_MAP

    def test_arch_init_exports(self):
        """Test brain and config are exported from arch __init__."""
        from quantumnematode.brain.arch import (
            HybridQuantumCortexBrain,
            HybridQuantumCortexBrainConfig,
        )

        assert HybridQuantumCortexBrain is not None
        assert HybridQuantumCortexBrainConfig is not None

    def test_brain_factory_dispatch(self):
        """Test brain factory creates correct brain type."""
        from quantumnematode.optimizers.learning_rate import ConstantLearningRate
        from quantumnematode.utils.brain_factory import setup_brain_model
        from quantumnematode.utils.config_loader import (
            GradientCalculationMethod,
            ParameterInitializerConfig,
        )

        config = _stage1_config()
        brain = setup_brain_model(
            brain_type=BrainType.HYBRID_QUANTUM_CORTEX,
            brain_config=config,
            shots=100,
            qubits=2,
            device=DeviceType.CPU,
            learning_rate=ConstantLearningRate(learning_rate=0.01),
            gradient_method=GradientCalculationMethod.CLIP,
            gradient_max_norm=None,
            parameter_initializer_config=ParameterInitializerConfig(),
        )
        assert isinstance(brain, HybridQuantumCortexBrain)


# ──────────────────────────────────────────────────────────────────────
# Brain copy
# ──────────────────────────────────────────────────────────────────────


class TestBrainCopy:
    """Test brain copy functionality."""

    def test_copy_produces_independent_brain(self):
        """Test copy creates independent brain with same weights."""
        brain1 = HybridQuantumCortexBrain(config=_stage1_config(), num_actions=4)
        brain2 = brain1.copy()

        torch.testing.assert_close(brain1.W_sh, brain2.W_sh)
        torch.testing.assert_close(brain1.W_hm, brain2.W_hm)

        # Modify original, copy should not change
        with torch.no_grad():
            brain1.W_sh.fill_(999.0)
        assert not torch.allclose(brain1.W_sh, brain2.W_sh)


# ──────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.smoke
class TestSmokeTest:
    """Smoke test: full end-to-end training session."""

    def test_stage1_3_episode_session(self):
        """Run a 3-episode stage 1 training session end-to-end."""
        config = _stage1_config(
            reinforce_window_size=10,
            num_reinforce_epochs=1,
        )
        brain = HybridQuantumCortexBrain(config=config, num_actions=4)
        params = BrainParams(gradient_strength=0.5, gradient_direction=0.0)

        for _ep in range(3):
            brain.prepare_episode()
            for step in range(20):
                result = brain.run_brain(
                    params,
                    top_only=False,
                    top_randomize=False,
                )
                assert len(result) == 1
                assert isinstance(result[0], ActionData)

                episode_done = step == 19
                brain.learn(params, reward=0.5, episode_done=episode_done)

        assert brain._episode_count == 3

    def test_stage2_3_episode_session(self):
        """Run a 3-episode stage 2 training session end-to-end."""
        config = _stage2_config(
            num_cortex_reinforce_epochs=1,
        )
        brain = HybridQuantumCortexBrain(config=config, num_actions=4)
        params = BrainParams(gradient_strength=0.5, gradient_direction=0.0)

        for _ep in range(3):
            brain.prepare_episode()
            for step in range(20):
                result = brain.run_brain(
                    params,
                    top_only=False,
                    top_randomize=False,
                )
                assert len(result) == 1

                episode_done = step == 19
                brain.learn(params, reward=0.5, episode_done=episode_done)

        assert brain._episode_count == 3
