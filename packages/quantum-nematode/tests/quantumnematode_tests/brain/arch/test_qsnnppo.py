"""Unit tests for the QSNN-PPO brain architecture."""

import numpy as np
import pytest
import torch
from quantumnematode.brain.actions import ActionData
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.brain.arch.qsnnppo import (
    QSNNPPOBrain,
    QSNNPPOBrainConfig,
    QSNNPPOCritic,
    QSNNRolloutBuffer,
)
from quantumnematode.brain.modules import ModuleName
from quantumnematode.env import Direction

# ──────────────────────────────────────────────────────────────────────
# Config Tests
# ──────────────────────────────────────────────────────────────────────


class TestQSNNPPOBrainConfig:
    """Test cases for QSNN-PPO brain configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = QSNNPPOBrainConfig()
        assert config.num_sensory_neurons == 8
        assert config.num_hidden_neurons == 16
        assert config.num_motor_neurons == 4
        assert config.membrane_tau == 0.9
        assert config.threshold == 0.5
        assert config.refractory_period == 0
        assert config.shots == 1024
        assert config.gamma == 0.99
        assert config.gae_lambda == 0.95
        assert config.clip_epsilon == 0.2
        assert config.num_epochs == 2
        assert config.num_minibatches == 4
        assert config.rollout_buffer_size == 256

    def test_custom_config(self):
        """Test custom configuration values."""
        config = QSNNPPOBrainConfig(
            num_sensory_neurons=4,
            num_hidden_neurons=8,
            num_motor_neurons=2,
            gamma=0.95,
            clip_epsilon=0.1,
            actor_lr=0.005,
            critic_lr=0.0005,
            rollout_buffer_size=256,
        )
        assert config.num_sensory_neurons == 4
        assert config.num_hidden_neurons == 8
        assert config.num_motor_neurons == 2
        assert config.gamma == 0.95
        assert config.clip_epsilon == 0.1
        assert config.actor_lr == 0.005
        assert config.critic_lr == 0.0005
        assert config.rollout_buffer_size == 256

    def test_invalid_num_sensory_neurons(self):
        """Test validation rejects invalid sensory neuron count."""
        with pytest.raises(ValueError, match="num_sensory_neurons must be >= 1"):
            QSNNPPOBrainConfig(num_sensory_neurons=0)

    def test_invalid_num_motor_neurons(self):
        """Test validation rejects too few motor neurons."""
        with pytest.raises(ValueError, match="num_motor_neurons must be >= 2"):
            QSNNPPOBrainConfig(num_motor_neurons=1)

    def test_invalid_membrane_tau(self):
        """Test validation rejects out-of-range membrane_tau."""
        with pytest.raises(ValueError, match="membrane_tau must be in"):
            QSNNPPOBrainConfig(membrane_tau=0.0)
        with pytest.raises(ValueError, match="membrane_tau must be in"):
            QSNNPPOBrainConfig(membrane_tau=1.5)

    def test_invalid_threshold(self):
        """Test validation rejects out-of-range threshold."""
        with pytest.raises(ValueError, match="threshold must be in"):
            QSNNPPOBrainConfig(threshold=0.0)
        with pytest.raises(ValueError, match="threshold must be in"):
            QSNNPPOBrainConfig(threshold=1.0)

    def test_invalid_shots(self):
        """Test validation rejects too few shots."""
        with pytest.raises(ValueError, match="shots must be >= 100"):
            QSNNPPOBrainConfig(shots=50)

    def test_invalid_num_epochs(self):
        """Test validation rejects zero epochs."""
        with pytest.raises(ValueError, match="num_epochs must be >= 1"):
            QSNNPPOBrainConfig(num_epochs=0)

    def test_sensory_modules_config(self):
        """Test sensory module configuration."""
        config = QSNNPPOBrainConfig(
            sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
        )
        assert config.sensory_modules is not None
        assert len(config.sensory_modules) == 2


# ──────────────────────────────────────────────────────────────────────
# Rollout Buffer Tests
# ──────────────────────────────────────────────────────────────────────


class TestQSNNRolloutBuffer:
    """Test cases for the QSNN rollout buffer."""

    @pytest.fixture
    def buffer(self) -> QSNNRolloutBuffer:
        """Create a test rollout buffer."""
        return QSNNRolloutBuffer(
            buffer_size=10,
            device=torch.device("cpu"),
            rng=np.random.default_rng(42),
        )

    def test_initialization(self, buffer: QSNNRolloutBuffer):
        """Test buffer initialization."""
        assert buffer.buffer_size == 10
        assert len(buffer) == 0
        assert not buffer.is_full()

    def test_add_experience(self, buffer: QSNNRolloutBuffer):
        """Test adding experience to the buffer."""
        buffer.add(
            features=np.array([0.5, 0.3], dtype=np.float32),
            action=1,
            log_prob=-0.5,
            value=0.8,
            reward=1.0,
            done=False,
            hidden_spike_rates=np.array([0.7, 0.2], dtype=np.float32),
        )
        assert len(buffer) == 1
        assert not buffer.is_full()

    def test_buffer_full(self, buffer: QSNNRolloutBuffer):
        """Test buffer fills to capacity."""
        for i in range(10):
            buffer.add(
                features=np.zeros(2, dtype=np.float32),
                action=i % 2,
                log_prob=-0.5,
                value=0.5,
                reward=0.1,
                done=(i == 9),
                hidden_spike_rates=np.zeros(2, dtype=np.float32),
            )
        assert buffer.is_full()
        assert len(buffer) == 10

    def test_reset(self, buffer: QSNNRolloutBuffer):
        """Test buffer reset clears all data."""
        buffer.add(
            features=np.zeros(2, dtype=np.float32),
            action=0,
            log_prob=-0.5,
            value=0.5,
            reward=0.1,
            done=False,
            hidden_spike_rates=np.zeros(2, dtype=np.float32),
        )
        buffer.reset()
        assert len(buffer) == 0
        assert not buffer.is_full()

    def test_compute_returns_and_advantages(self, buffer: QSNNRolloutBuffer):
        """Test GAE computation produces correct shapes and finite values."""
        for i in range(5):
            buffer.add(
                features=np.array([0.1 * i, 0.2 * i], dtype=np.float32),
                action=i % 2,
                log_prob=-0.5,
                value=float(i) / 5,
                reward=1.0 if i == 4 else 0.1,
                done=(i == 4),
                hidden_spike_rates=np.zeros(2, dtype=np.float32),
            )

        returns, advantages = buffer.compute_returns_and_advantages(
            last_value=0.0,
            gamma=0.99,
            gae_lambda=0.95,
        )

        assert returns.shape == (5,)
        assert advantages.shape == (5,)
        assert torch.all(torch.isfinite(returns))
        assert torch.all(torch.isfinite(advantages))

    def test_gae_terminal_last_value_zero(self, buffer: QSNNRolloutBuffer):
        """Test GAE with terminal state uses zero bootstrap."""
        buffer.add(
            features=np.zeros(2, dtype=np.float32),
            action=0,
            log_prob=-0.5,
            value=0.5,
            reward=1.0,
            done=True,
            hidden_spike_rates=np.zeros(2, dtype=np.float32),
        )

        returns, advantages = buffer.compute_returns_and_advantages(
            last_value=0.0,
            gamma=0.99,
            gae_lambda=0.95,
        )

        # With value=0.5, reward=1.0, done=True, last_value=0.0:
        # delta = 1.0 + 0.99 * 0.0 * 0.0 - 0.5 = 0.5
        # advantage equals 0.5
        assert torch.isclose(advantages[0], torch.tensor(0.5))

    def test_get_minibatches(self, buffer: QSNNRolloutBuffer):
        """Test minibatch generation."""
        for i in range(8):
            buffer.add(
                features=np.array([0.1 * i, 0.2 * i], dtype=np.float32),
                action=i % 2,
                log_prob=-0.5,
                value=0.5,
                reward=0.1,
                done=(i == 7),
                hidden_spike_rates=np.zeros(2, dtype=np.float32),
            )

        returns, advantages = buffer.compute_returns_and_advantages(
            last_value=0.0,
            gamma=0.99,
            gae_lambda=0.95,
        )

        minibatches = list(buffer.get_minibatches(2, returns, advantages))
        assert len(minibatches) == 2
        for batch in minibatches:
            assert "indices" in batch
            assert "actions" in batch
            assert "old_log_probs" in batch
            assert "returns" in batch
            assert "advantages" in batch
            assert batch["indices"].shape[0] == 4  # 8 / 2 minibatches

    def test_minibatch_advantages_normalized(self, buffer: QSNNRolloutBuffer):
        """Test advantages are normalized in minibatches."""
        for i in range(8):
            buffer.add(
                features=np.zeros(2, dtype=np.float32),
                action=i % 2,
                log_prob=-0.5,
                value=0.5,
                reward=float(i),
                done=(i == 7),
                hidden_spike_rates=np.zeros(2, dtype=np.float32),
            )

        returns, advantages = buffer.compute_returns_and_advantages(
            last_value=0.0,
            gamma=0.99,
            gae_lambda=0.95,
        )

        # Collect all advantages from minibatches using list comprehension
        all_advs = [batch["advantages"] for batch in buffer.get_minibatches(1, returns, advantages)]
        combined = torch.cat(all_advs)
        # Should be approximately zero-mean, unit-variance
        assert abs(combined.mean().item()) < 0.1
        assert abs(combined.std().item() - 1.0) < 0.1


# ──────────────────────────────────────────────────────────────────────
# Critic Tests
# ──────────────────────────────────────────────────────────────────────


class TestQSNNPPOCritic:
    """Test cases for the QSNN-PPO critic network."""

    @pytest.fixture
    def critic(self) -> QSNNPPOCritic:
        """Create a test critic."""
        return QSNNPPOCritic(input_dim=10, hidden_dim=32, num_layers=2)

    def test_initialization(self, critic: QSNNPPOCritic):
        """Test critic initialization."""
        assert critic.network is not None
        assert isinstance(critic.network, torch.nn.Sequential)

    def test_forward_scalar_output(self, critic: QSNNPPOCritic):
        """Test critic forward pass produces scalar output."""
        test_input = torch.randn(10)
        output = critic(test_input)
        assert output.shape == ()  # Scalar output (squeezed)

    def test_forward_batch_output(self, critic: QSNNPPOCritic):
        """Test critic forward pass with batch input."""
        test_input = torch.randn(5, 10)
        output = critic(test_input)
        assert output.shape == (5,)

    def test_forward_finite_values(self, critic: QSNNPPOCritic):
        """Test critic outputs are finite."""
        test_input = torch.randn(5, 10)
        output = critic(test_input)
        assert torch.all(torch.isfinite(output))

    def test_orthogonal_init(self):
        """Test critic uses orthogonal weight initialization."""
        critic = QSNNPPOCritic(input_dim=4, hidden_dim=8, num_layers=1)
        # Check first linear layer has orthogonal-ish weights
        first_linear = critic.network[0]
        assert isinstance(first_linear, torch.nn.Linear)
        # Orthogonal matrices have singular values all equal to 1
        # Just verify the weights are not all zeros
        assert first_linear.weight.abs().sum() > 0


# ──────────────────────────────────────────────────────────────────────
# Brain Initialization Tests
# ──────────────────────────────────────────────────────────────────────


class TestQSNNPPOBrainInit:
    """Test cases for QSNN-PPO brain initialization."""

    @pytest.fixture
    def config(self) -> QSNNPPOBrainConfig:
        """Create a small test configuration."""
        return QSNNPPOBrainConfig(
            num_sensory_neurons=4,
            num_hidden_neurons=4,
            num_motor_neurons=2,
            shots=100,
            num_epochs=2,
            num_minibatches=2,
            rollout_buffer_size=8,
            seed=42,
        )

    @pytest.fixture
    def brain(self, config: QSNNPPOBrainConfig) -> QSNNPPOBrain:
        """Create a test QSNN-PPO brain."""
        return QSNNPPOBrain(config=config, num_actions=2, device=DeviceType.CPU)

    def test_initialization(self, brain: QSNNPPOBrain, config: QSNNPPOBrainConfig):
        """Test brain initializes with correct attributes."""
        assert brain.num_actions == 2
        assert brain.num_sensory == config.num_sensory_neurons
        assert brain.num_hidden == config.num_hidden_neurons
        assert brain.num_motor == config.num_motor_neurons
        assert brain.critic is not None
        assert brain.actor_optimizer is not None
        assert brain.critic_optimizer is not None

    def test_actor_weight_shapes(self, brain: QSNNPPOBrain):
        """Test actor weight matrix shapes."""
        assert brain.W_sh.shape == (brain.num_sensory, brain.num_hidden)
        assert brain.W_hm.shape == (brain.num_hidden, brain.num_motor)
        assert brain.theta_hidden.shape == (brain.num_hidden,)
        assert brain.theta_motor.shape == (brain.num_motor,)

    def test_actor_weights_require_grad(self, brain: QSNNPPOBrain):
        """Test actor weights require gradients."""
        assert brain.W_sh.requires_grad
        assert brain.W_hm.requires_grad
        assert brain.theta_hidden.requires_grad
        assert brain.theta_motor.requires_grad

    def test_theta_hidden_initial_value(self, brain: QSNNPPOBrain):
        """Test theta_hidden initialized to pi/4."""
        expected = torch.full((brain.num_hidden,), np.pi / 4)
        assert torch.allclose(brain.theta_hidden.detach(), expected, atol=1e-6)

    def test_theta_motor_initial_value(self, brain: QSNNPPOBrain):
        """Test theta_motor initialized to zero."""
        expected = torch.zeros(brain.num_motor)
        assert torch.allclose(brain.theta_motor.detach(), expected, atol=1e-6)

    def test_critic_input_dim(self, brain: QSNNPPOBrain):
        """Test critic input dimension is features + hidden neurons."""
        expected_dim = brain.input_dim + brain.num_hidden
        first_linear = brain.critic.network[0]
        assert first_linear.in_features == expected_dim

    def test_separate_optimizers(self, brain: QSNNPPOBrain):
        """Test actor and critic have separate optimizers."""
        assert brain.actor_optimizer is not brain.critic_optimizer
        # Actor optimizer should have 4 param groups (W_sh, W_hm, theta_h, theta_m)
        actor_params = brain.actor_optimizer.param_groups[0]["params"]
        assert len(actor_params) == 4

    def test_action_set_validation(self):
        """Test action set length must match num_actions."""
        from quantumnematode.brain.actions import Action

        config = QSNNPPOBrainConfig(
            num_sensory_neurons=2,
            num_hidden_neurons=2,
            num_motor_neurons=2,
            shots=100,
        )
        # Provide explicit action_set that doesn't match num_actions
        with pytest.raises(ValueError, match="num_actions.*does not match"):
            QSNNPPOBrain(
                config=config,
                num_actions=3,
                device=DeviceType.CPU,
                action_set=[Action.FORWARD, Action.LEFT],  # length 2 != 3
            )

    def test_lr_scheduler_none_by_default(self, brain: QSNNPPOBrain):
        """Test LR scheduler is None when lr_decay_episodes not set."""
        assert brain.scheduler is None

    def test_lr_scheduler_created(self):
        """Test LR scheduler created when lr_decay_episodes is set."""
        config = QSNNPPOBrainConfig(
            num_sensory_neurons=2,
            num_hidden_neurons=2,
            num_motor_neurons=2,
            shots=100,
            lr_decay_episodes=200,
        )
        brain = QSNNPPOBrain(config=config, num_actions=2, device=DeviceType.CPU)
        assert brain.scheduler is not None


# ──────────────────────────────────────────────────────────────────────
# Preprocessing Tests
# ──────────────────────────────────────────────────────────────────────


class TestQSNNPPOBrainPreprocess:
    """Test cases for feature extraction."""

    @pytest.fixture
    def brain(self) -> QSNNPPOBrain:
        """Create a test brain with legacy preprocessing."""
        config = QSNNPPOBrainConfig(
            num_sensory_neurons=4,
            num_hidden_neurons=4,
            num_motor_neurons=2,
            shots=100,
            seed=42,
        )
        return QSNNPPOBrain(config=config, num_actions=2, device=DeviceType.CPU)

    def test_legacy_preprocess(self, brain: QSNNPPOBrain):
        """Test legacy 2-feature preprocessing."""
        params = BrainParams(
            gradient_strength=0.6,
            gradient_direction=0.3,
            agent_direction=Direction.UP,
        )
        features = brain.preprocess(params)
        assert isinstance(features, np.ndarray)
        assert len(features) == 2
        assert features.dtype == np.float32

    def test_legacy_preprocess_zero_strength(self, brain: QSNNPPOBrain):
        """Test preprocessing with zero gradient strength."""
        params = BrainParams(
            gradient_strength=0.0,
            gradient_direction=0.0,
            agent_direction=Direction.UP,
        )
        features = brain.preprocess(params)
        assert features[0] == 0.0  # gradient_strength

    def test_sensory_module_preprocess(self):
        """Test preprocessing with unified sensory modules."""
        config = QSNNPPOBrainConfig(
            num_sensory_neurons=4,
            num_hidden_neurons=4,
            num_motor_neurons=2,
            shots=100,
            sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
            seed=42,
        )
        brain = QSNNPPOBrain(config=config, num_actions=2, device=DeviceType.CPU)

        params = BrainParams(
            food_gradient_strength=0.7,
            food_gradient_direction=1.0,
            predator_gradient_strength=0.3,
            predator_gradient_direction=-0.5,
            agent_direction=Direction.UP,
        )
        features = brain.preprocess(params)
        assert len(features) == 4  # 2 modules * 2 features

    def test_critic_input_construction(self, brain: QSNNPPOBrain):
        """Test critic input concatenates features and hidden spikes."""
        features = np.array([0.5, 0.3], dtype=np.float32)
        hidden_spikes = (
            np.random.default_rng(42)
            .random(brain.num_hidden)
            .astype(
                np.float32,
            )
        )

        critic_input = brain._get_critic_input(features, hidden_spikes)
        expected_dim = brain.input_dim + brain.num_hidden
        assert critic_input.shape == (expected_dim,)
        # First elements should be features, rest hidden spikes
        assert torch.isclose(
            critic_input[0],
            torch.tensor(0.5, dtype=torch.float32),
        )


# ──────────────────────────────────────────────────────────────────────
# Forward Pass Tests
# ──────────────────────────────────────────────────────────────────────


class TestQSNNPPOBrainForwardPass:
    """Test cases for brain forward pass."""

    @pytest.fixture
    def brain(self) -> QSNNPPOBrain:
        """Create a small test brain."""
        config = QSNNPPOBrainConfig(
            num_sensory_neurons=2,
            num_hidden_neurons=2,
            num_motor_neurons=2,
            shots=100,
            num_integration_steps=2,
            seed=42,
        )
        return QSNNPPOBrain(config=config, num_actions=2, device=DeviceType.CPU)

    def test_run_brain_returns_action(self, brain: QSNNPPOBrain):
        """Test run_brain returns valid action data."""
        params = BrainParams(
            gradient_strength=0.5,
            gradient_direction=1.0,
            agent_direction=Direction.UP,
        )

        actions = brain.run_brain(params, top_only=True, top_randomize=True)

        assert len(actions) == 1
        action_data = actions[0]
        assert isinstance(action_data, ActionData)
        assert 0.0 <= action_data.probability <= 1.0

    def test_run_brain_stores_pending_data(self, brain: QSNNPPOBrain):
        """Test run_brain stores pending data for learn()."""
        params = BrainParams(
            gradient_strength=0.5,
            gradient_direction=1.0,
            agent_direction=Direction.UP,
        )
        brain.run_brain(params, top_only=True, top_randomize=True)

        assert brain._pending_features is not None
        assert isinstance(brain._pending_action, int)
        assert brain._pending_hidden_spikes is not None

    def test_action_probabilities_sum_to_one(self, brain: QSNNPPOBrain):
        """Test action probabilities sum to 1."""
        params = BrainParams(
            gradient_strength=0.5,
            gradient_direction=1.0,
            agent_direction=Direction.UP,
        )
        brain.run_brain(params, top_only=True, top_randomize=True)

        assert brain.current_probabilities is not None
        total = np.sum(brain.current_probabilities)
        assert abs(total - 1.0) < 1e-6

    def test_history_tracking(self, brain: QSNNPPOBrain):
        """Test that actions are tracked in history."""
        params = BrainParams(
            gradient_strength=0.5,
            gradient_direction=1.0,
            agent_direction=Direction.UP,
        )
        brain.run_brain(params, top_only=True, top_randomize=True)

        assert len(brain.history_data.actions) == 1
        assert len(brain.history_data.probabilities) == 1


# ──────────────────────────────────────────────────────────────────────
# Learning Tests
# ──────────────────────────────────────────────────────────────────────


class TestQSNNPPOBrainLearning:
    """Test cases for QSNN-PPO learning."""

    @pytest.fixture
    def brain(self) -> QSNNPPOBrain:
        """Create a test brain for learning tests."""
        config = QSNNPPOBrainConfig(
            num_sensory_neurons=2,
            num_hidden_neurons=2,
            num_motor_neurons=2,
            shots=100,
            num_epochs=2,
            num_minibatches=2,
            rollout_buffer_size=8,
            num_integration_steps=2,
            seed=42,
        )
        return QSNNPPOBrain(config=config, num_actions=2, device=DeviceType.CPU)

    def test_learn_adds_to_buffer(self, brain: QSNNPPOBrain):
        """Test learn() adds experiences to buffer."""
        params = BrainParams(
            gradient_strength=0.5,
            gradient_direction=1.0,
            agent_direction=Direction.UP,
        )

        brain.run_brain(params, top_only=True, top_randomize=True)
        brain.learn(params, reward=1.0, episode_done=False)

        assert len(brain.buffer) == 1

    def test_buffer_triggers_ppo_update(self, brain: QSNNPPOBrain):
        """Test PPO update triggers when buffer is full."""
        params = BrainParams(
            gradient_strength=0.5,
            gradient_direction=1.0,
            agent_direction=Direction.UP,
        )

        for _step in range(brain.config.rollout_buffer_size):
            brain.run_brain(params, top_only=True, top_randomize=True)
            brain.learn(params, reward=0.1, episode_done=False)

        # Buffer should be cleared after PPO update
        assert len(brain.buffer) == 0

    def test_episode_done_does_not_trigger_update(self, brain: QSNNPPOBrain):
        """Test buffer persists across episodes until full."""
        params = BrainParams(
            gradient_strength=0.5,
            gradient_direction=1.0,
            agent_direction=Direction.UP,
        )

        for step in range(5):
            brain.run_brain(params, top_only=True, top_randomize=True)
            brain.learn(params, reward=0.1, episode_done=(step == 4))

        # Buffer should NOT be cleared -- accumulates across episodes
        assert len(brain.buffer) == 5

    def test_ppo_update_changes_actor_weights(self, brain: QSNNPPOBrain):
        """Test PPO update modifies actor weights."""
        params = BrainParams(
            gradient_strength=0.5,
            gradient_direction=1.0,
            agent_direction=Direction.UP,
        )

        initial_w_sh = brain.W_sh.clone().detach()

        # Fill buffer to trigger update
        for _step in range(brain.config.rollout_buffer_size):
            brain.run_brain(params, top_only=True, top_randomize=True)
            brain.learn(params, reward=1.0, episode_done=False)

        # Weights should have changed (PPO update occurred)
        assert not torch.allclose(brain.W_sh, initial_w_sh, atol=1e-6)

    def test_ppo_update_changes_critic_weights(self, brain: QSNNPPOBrain):
        """Test PPO update modifies critic weights."""
        params = BrainParams(
            gradient_strength=0.5,
            gradient_direction=1.0,
            agent_direction=Direction.UP,
        )

        initial_critic_params = [p.clone().detach() for p in brain.critic.parameters()]

        # Fill buffer to trigger update
        for _step in range(brain.config.rollout_buffer_size):
            brain.run_brain(params, top_only=True, top_randomize=True)
            brain.learn(params, reward=1.0, episode_done=False)

        # At least some critic params should have changed
        changed = False
        for old_p, new_p in zip(
            initial_critic_params,
            brain.critic.parameters(),
            strict=True,
        ):
            if not torch.allclose(old_p, new_p, atol=1e-6):
                changed = True
                break
        assert changed

    def test_weight_clamping(self, brain: QSNNPPOBrain):
        """Test weights are clamped after PPO update."""
        # Set extreme weights
        with torch.no_grad():
            brain.W_sh.fill_(100.0)

        brain._clamp_weights()

        assert brain.W_sh.max().item() <= brain.config.weight_clip
        assert brain.W_sh.min().item() >= -brain.config.weight_clip

    def test_theta_motor_norm_clamping(self, brain: QSNNPPOBrain):
        """Test theta_motor L2 norm is clamped."""
        with torch.no_grad():
            brain.theta_motor.fill_(100.0)

        brain._clamp_weights()

        tm_norm = torch.norm(brain.theta_motor).item()
        assert tm_norm <= brain.config.theta_motor_max_norm + 1e-6

    def test_theta_hidden_min_norm_clamping(self, brain: QSNNPPOBrain):
        """Test theta_hidden L2 norm is preserved above min_norm."""
        with torch.no_grad():
            brain.theta_hidden.fill_(0.01)

        brain._clamp_weights()

        th_norm = torch.norm(brain.theta_hidden).item()
        assert th_norm >= brain.config.theta_hidden_min_norm - 1e-6

    def test_rewards_stored_in_history(self, brain: QSNNPPOBrain):
        """Test rewards are recorded in history."""
        params = BrainParams(
            gradient_strength=0.5,
            gradient_direction=1.0,
            agent_direction=Direction.UP,
        )

        brain.run_brain(params, top_only=True, top_randomize=True)
        brain.learn(params, reward=0.5, episode_done=False)

        assert len(brain.history_data.rewards) == 1
        assert brain.history_data.rewards[0] == 0.5


# ──────────────────────────────────────────────────────────────────────
# Episode Lifecycle Tests
# ──────────────────────────────────────────────────────────────────────


class TestQSNNPPOBrainEpisode:
    """Test cases for episode lifecycle."""

    @pytest.fixture
    def brain(self) -> QSNNPPOBrain:
        """Create a test brain."""
        config = QSNNPPOBrainConfig(
            num_sensory_neurons=2,
            num_hidden_neurons=2,
            num_motor_neurons=2,
            shots=100,
            num_epochs=1,
            num_minibatches=2,
            rollout_buffer_size=8,
            num_integration_steps=2,
            seed=42,
        )
        return QSNNPPOBrain(config=config, num_actions=2, device=DeviceType.CPU)

    def test_prepare_episode(self, brain: QSNNPPOBrain):
        """Test episode preparation resets state."""
        brain._step_count = 100
        brain.refractory_hidden.fill(5)

        brain.prepare_episode()

        assert brain._step_count == 0
        assert np.all(brain.refractory_hidden == 0)
        assert np.all(brain.refractory_motor == 0)

    def test_post_process_episode(self, brain: QSNNPPOBrain):
        """Test post-processing increments episode counter."""
        assert brain._episode_count == 0

        brain.post_process_episode()

        assert brain._episode_count == 1

    def test_lr_scheduler_step(self):
        """Test LR scheduler steps on episode end."""
        config = QSNNPPOBrainConfig(
            num_sensory_neurons=2,
            num_hidden_neurons=2,
            num_motor_neurons=2,
            shots=100,
            lr_decay_episodes=10,
            actor_lr=0.01,
            seed=42,
        )
        brain = QSNNPPOBrain(config=config, num_actions=2, device=DeviceType.CPU)

        initial_lr = brain.actor_optimizer.param_groups[0]["lr"]
        for _ in range(5):
            brain.post_process_episode()

        current_lr = brain.actor_optimizer.param_groups[0]["lr"]
        assert current_lr < initial_lr

    def test_multiple_episodes(self, brain: QSNNPPOBrain):
        """Test running multiple complete episodes."""
        params = BrainParams(
            gradient_strength=0.5,
            gradient_direction=1.0,
            agent_direction=Direction.UP,
        )

        for _episode in range(3):
            brain.prepare_episode()
            for step in range(4):
                brain.run_brain(params, top_only=True, top_randomize=True)
                brain.learn(params, reward=0.1, episode_done=(step == 3))
            brain.post_process_episode()

        assert brain._episode_count == 3
        assert len(brain.history_data.rewards) == 12


# ──────────────────────────────────────────────────────────────────────
# Reproducibility Tests
# ──────────────────────────────────────────────────────────────────────


class TestQSNNPPOBrainReproducibility:
    """Test cases for reproducibility with seeding."""

    def test_deterministic_with_seed(self):
        """Test that identical seeds produce identical actions."""
        config = QSNNPPOBrainConfig(
            num_sensory_neurons=2,
            num_hidden_neurons=2,
            num_motor_neurons=2,
            shots=100,
            num_integration_steps=2,
            seed=42,
        )

        brain1 = QSNNPPOBrain(config=config, num_actions=2, device=DeviceType.CPU)
        brain2 = QSNNPPOBrain(
            config=QSNNPPOBrainConfig(**config.model_dump()),
            num_actions=2,
            device=DeviceType.CPU,
        )

        params = BrainParams(
            gradient_strength=0.5,
            gradient_direction=1.0,
            agent_direction=Direction.UP,
        )

        actions1 = brain1.run_brain(params, top_only=True, top_randomize=True)
        actions2 = brain2.run_brain(params, top_only=True, top_randomize=True)

        assert actions1[0].action == actions2[0].action

    def test_different_seeds_produce_different_weights(self):
        """Test that different seeds produce different initial weights."""
        config1 = QSNNPPOBrainConfig(
            num_sensory_neurons=2,
            num_hidden_neurons=2,
            num_motor_neurons=2,
            shots=100,
            seed=42,
        )
        config2 = QSNNPPOBrainConfig(
            num_sensory_neurons=2,
            num_hidden_neurons=2,
            num_motor_neurons=2,
            shots=100,
            seed=123,
        )

        brain1 = QSNNPPOBrain(config=config1, num_actions=2, device=DeviceType.CPU)
        brain2 = QSNNPPOBrain(config=config2, num_actions=2, device=DeviceType.CPU)

        assert not torch.allclose(brain1.W_sh, brain2.W_sh)


# ──────────────────────────────────────────────────────────────────────
# Integration Tests
# ──────────────────────────────────────────────────────────────────────


class TestQSNNPPOBrainIntegration:
    """Integration tests for the QSNN-PPO brain."""

    @pytest.fixture
    def brain(self) -> QSNNPPOBrain:
        """Create a brain for integration tests."""
        config = QSNNPPOBrainConfig(
            num_sensory_neurons=4,
            num_hidden_neurons=4,
            num_motor_neurons=2,
            shots=100,
            num_epochs=2,
            num_minibatches=2,
            rollout_buffer_size=16,
            num_integration_steps=2,
            seed=42,
        )
        return QSNNPPOBrain(config=config, num_actions=2, device=DeviceType.CPU)

    def test_full_training_episode(self, brain: QSNNPPOBrain):
        """Test a complete training episode workflow."""
        params = BrainParams(
            gradient_strength=0.5,
            gradient_direction=1.0,
            agent_direction=Direction.UP,
        )

        brain.prepare_episode()

        for step in range(20):
            actions = brain.run_brain(params, top_only=True, top_randomize=True)
            assert len(actions) == 1
            brain.learn(params, reward=0.1, episode_done=(step == 19))

        brain.post_process_episode()

        # Verify state is consistent
        assert brain._episode_count == 1
        assert len(brain.history_data.rewards) == 20

    def test_multi_episode_training(self, brain: QSNNPPOBrain):
        """Test training across multiple episodes."""
        params_list = [
            BrainParams(
                gradient_strength=0.3,
                gradient_direction=0.0,
                agent_direction=Direction.UP,
            ),
            BrainParams(
                gradient_strength=0.8,
                gradient_direction=1.5,
                agent_direction=Direction.RIGHT,
            ),
        ]

        for episode in range(3):
            brain.prepare_episode()
            params = params_list[episode % len(params_list)]
            for step in range(10):
                brain.run_brain(params, top_only=True, top_randomize=True)
                brain.learn(params, reward=0.1, episode_done=(step == 9))
            brain.post_process_episode()

        assert brain._episode_count == 3

    def test_varying_reward_signals(self, brain: QSNNPPOBrain):
        """Test brain handles varying reward magnitudes."""
        params = BrainParams(
            gradient_strength=0.5,
            gradient_direction=1.0,
            agent_direction=Direction.UP,
        )

        rewards = [0.1, -0.5, 2.0, -10.0, 0.0, 0.5, 1.0, -1.0]

        brain.prepare_episode()
        for step, reward in enumerate(rewards):
            brain.run_brain(params, top_only=True, top_randomize=True)
            brain.learn(
                params,
                reward=reward,
                episode_done=(step == len(rewards) - 1),
            )
        brain.post_process_episode()

        # Should not crash with extreme rewards
        assert brain._episode_count == 1

    def test_sensory_modules_integration(self):
        """Test full workflow with sensory modules."""
        config = QSNNPPOBrainConfig(
            num_sensory_neurons=4,
            num_hidden_neurons=4,
            num_motor_neurons=2,
            shots=100,
            num_epochs=1,
            num_minibatches=2,
            rollout_buffer_size=8,
            num_integration_steps=2,
            sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
            seed=42,
        )
        brain = QSNNPPOBrain(config=config, num_actions=2, device=DeviceType.CPU)

        params = BrainParams(
            food_gradient_strength=0.7,
            food_gradient_direction=1.0,
            predator_gradient_strength=0.3,
            predator_gradient_direction=-0.5,
            agent_direction=Direction.UP,
        )

        brain.prepare_episode()
        for step in range(10):
            brain.run_brain(params, top_only=True, top_randomize=True)
            brain.learn(params, reward=0.1, episode_done=(step == 9))
        brain.post_process_episode()

        assert brain._episode_count == 1


# ──────────────────────────────────────────────────────────────────────
# Error Handling Tests
# ──────────────────────────────────────────────────────────────────────


class TestQSNNPPOBrainErrors:
    """Test cases for error handling."""

    def test_copy_not_implemented(self):
        """Test copy raises NotImplementedError."""
        config = QSNNPPOBrainConfig(
            num_sensory_neurons=2,
            num_hidden_neurons=2,
            num_motor_neurons=2,
            shots=100,
        )
        brain = QSNNPPOBrain(config=config, num_actions=2, device=DeviceType.CPU)

        with pytest.raises(NotImplementedError, match="does not support copying"):
            brain.copy()

    def test_build_brain_not_implemented(self):
        """Test build_brain raises NotImplementedError."""
        config = QSNNPPOBrainConfig(
            num_sensory_neurons=2,
            num_hidden_neurons=2,
            num_motor_neurons=2,
            shots=100,
        )
        brain = QSNNPPOBrain(config=config, num_actions=2, device=DeviceType.CPU)

        with pytest.raises(NotImplementedError, match="does not have a quantum"):
            brain.build_brain()

    def test_action_set_setter_validates_length(self):
        """Test action_set setter validates length."""
        config = QSNNPPOBrainConfig(
            num_sensory_neurons=2,
            num_hidden_neurons=2,
            num_motor_neurons=2,
            shots=100,
        )
        brain = QSNNPPOBrain(config=config, num_actions=2, device=DeviceType.CPU)

        with pytest.raises(ValueError, match="Cannot set action_set"):
            brain.action_set = [brain.action_set[0]]  # Wrong length


# ──────────────────────────────────────────────────────────────────────
# Registration Tests
# ──────────────────────────────────────────────────────────────────────


class TestQSNNPPORegistration:
    """Test cases for brain type registration."""

    def test_brain_type_enum(self):
        """Test QSNN_PPO is in BrainType enum."""
        from quantumnematode.brain.arch.dtypes import BrainType

        assert hasattr(BrainType, "QSNN_PPO")
        assert BrainType.QSNN_PPO.value == "qsnnppo"

    def test_quantum_brain_type(self):
        """Test QSNN_PPO is in QUANTUM_BRAIN_TYPES."""
        from quantumnematode.brain.arch.dtypes import QUANTUM_BRAIN_TYPES, BrainType

        assert BrainType.QSNN_PPO in QUANTUM_BRAIN_TYPES

    def test_brain_config_map(self):
        """Test QSNN-PPO is in config loader's BRAIN_CONFIG_MAP."""
        from quantumnematode.utils.config_loader import BRAIN_CONFIG_MAP

        assert "qsnnppo" in BRAIN_CONFIG_MAP
        assert BRAIN_CONFIG_MAP["qsnnppo"] is QSNNPPOBrainConfig

    def test_init_exports(self):
        """Test QSNN-PPO classes are exported from arch package."""
        from quantumnematode.brain.arch import QSNNPPOBrain, QSNNPPOBrainConfig

        assert QSNNPPOBrain is not None
        assert QSNNPPOBrainConfig is not None
