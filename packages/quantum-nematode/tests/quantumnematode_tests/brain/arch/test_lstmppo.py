"""Unit tests for the LSTM PPO brain architecture."""

import numpy as np
import pytest
import torch
from quantumnematode.brain.actions import ActionData
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.brain.arch.lstmppo import (
    LSTMPPOBrain,
    LSTMPPOBrainConfig,
    LSTMPPORolloutBuffer,
)
from quantumnematode.brain.modules import ModuleName
from quantumnematode.env import Direction

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

SENSORY_MODULES = [ModuleName.FOOD_CHEMOTAXIS, ModuleName.PROPRIOCEPTION]
INPUT_DIM = 4  # 2 modules * 2 features each


def _make_config(**overrides) -> LSTMPPOBrainConfig:
    """Create a config with sensory modules and small defaults for fast tests."""
    defaults = {
        "sensory_modules": SENSORY_MODULES,
        "rollout_buffer_size": 32,
        "bptt_chunk_length": 8,
        "lstm_hidden_dim": 16,
        "actor_hidden_dim": 16,
        "critic_hidden_dim": 16,
        "num_epochs": 2,
        "seed": 42,
    }
    defaults.update(overrides)
    return LSTMPPOBrainConfig(**defaults)


def _make_brain(**overrides) -> LSTMPPOBrain:
    """Create a brain with small defaults for fast tests."""
    config = _make_config(**overrides)
    return LSTMPPOBrain(config=config, num_actions=4, device=DeviceType.CPU)


def _make_params(grad_strength: float = 0.5, direction: Direction = Direction.UP) -> BrainParams:
    """Create a BrainParams for testing."""
    return BrainParams(
        food_gradient_strength=grad_strength,
        food_gradient_direction=np.pi / 2,
        agent_direction=direction,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 3.1 Config Validation Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestLSTMPPOBrainConfig:
    """Test cases for LSTM PPO brain configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LSTMPPOBrainConfig(sensory_modules=SENSORY_MODULES)

        assert config.rnn_type == "lstm"
        assert config.lstm_hidden_dim == 64
        assert config.bptt_chunk_length == 16
        assert config.actor_hidden_dim == 64
        assert config.critic_hidden_dim == 128
        assert config.actor_num_layers == 2
        assert config.critic_num_layers == 2
        assert config.actor_lr == 0.0005
        assert config.critic_lr == 0.0005
        assert config.gamma == 0.99
        assert config.gae_lambda == 0.95
        assert config.clip_epsilon == 0.2
        assert config.value_loss_coef == 0.5
        assert config.num_epochs == 6
        assert config.rollout_buffer_size == 1024
        assert config.max_grad_norm == 0.5
        assert config.entropy_coef == 0.02
        assert config.entropy_coef_end == 0.008
        assert config.entropy_decay_episodes == 500

    def test_custom_config(self):
        """Test custom configuration values."""
        config = LSTMPPOBrainConfig(
            sensory_modules=SENSORY_MODULES,
            rnn_type="gru",
            lstm_hidden_dim=32,
            bptt_chunk_length=8,
            actor_hidden_dim=128,
            actor_lr=0.001,
            rollout_buffer_size=256,
        )

        assert config.rnn_type == "gru"
        assert config.lstm_hidden_dim == 32
        assert config.bptt_chunk_length == 8
        assert config.actor_hidden_dim == 128
        assert config.actor_lr == 0.001
        assert config.rollout_buffer_size == 256

    def test_invalid_lstm_hidden_dim(self):
        """Test that lstm_hidden_dim < 2 is rejected."""
        with pytest.raises(ValueError, match="lstm_hidden_dim must be >= 2"):
            LSTMPPOBrainConfig(sensory_modules=SENSORY_MODULES, lstm_hidden_dim=1)

    def test_invalid_bptt_chunk_length(self):
        """Test that bptt_chunk_length < 4 is rejected."""
        with pytest.raises(ValueError, match="bptt_chunk_length must be >= 4"):
            LSTMPPOBrainConfig(sensory_modules=SENSORY_MODULES, bptt_chunk_length=2)

    def test_invalid_rollout_buffer_size(self):
        """Test that rollout_buffer_size < bptt_chunk_length is rejected."""
        with pytest.raises(ValueError, match="rollout_buffer_size"):
            LSTMPPOBrainConfig(
                sensory_modules=SENSORY_MODULES,
                rollout_buffer_size=8,
                bptt_chunk_length=16,
            )

    def test_sensory_modules_required(self):
        """Test that sensory_modules=None is rejected."""
        with pytest.raises(ValueError, match="sensory_modules is required"):
            LSTMPPOBrainConfig()


# ──────────────────────────────────────────────────────────────────────────────
# 3.2 Rollout Buffer Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestLSTMPPORolloutBuffer:
    """Test cases for the LSTM PPO rollout buffer."""

    @pytest.fixture
    def buffer(self):
        """Create a test rollout buffer."""
        return LSTMPPORolloutBuffer(buffer_size=10, device=torch.device("cpu"))

    def test_buffer_initialization(self, buffer):
        """Test buffer initialization."""
        assert buffer.buffer_size == 10
        assert len(buffer) == 0
        assert not buffer.is_full()

    def test_add_experience(self, buffer):
        """Test adding experience to the buffer."""
        features = np.array([0.5, 0.3, 0.1, 0.2], dtype=np.float32)
        h_state = torch.zeros(16)
        c_state = torch.zeros(16)

        buffer.add(
            features=features,
            action=1,
            log_prob=-0.5,
            value=0.8,
            reward=1.0,
            done=False,
            h_state=h_state,
            c_state=c_state,
        )

        assert len(buffer) == 1
        assert not buffer.is_full()
        np.testing.assert_array_equal(buffer.features[0], features)
        assert buffer.actions[0] == 1
        assert buffer.rewards[0] == 1.0
        assert buffer.dones[0] is False

    def test_buffer_is_full(self, buffer):
        """Test buffer full detection."""
        for i in range(10):
            buffer.add(
                features=np.array([0.1 * i] * 4, dtype=np.float32),
                action=i % 4,
                log_prob=-0.5,
                value=0.5,
                reward=0.1,
                done=(i == 9),
                h_state=torch.zeros(16),
                c_state=torch.zeros(16),
            )
        assert buffer.is_full()
        assert len(buffer) == 10

    def test_buffer_reset(self, buffer):
        """Test buffer reset."""
        for i in range(5):
            buffer.add(
                features=np.array([0.1 * i] * 4, dtype=np.float32),
                action=i % 4,
                log_prob=-0.5,
                value=0.5,
                reward=0.1,
                done=False,
                h_state=torch.zeros(16),
                c_state=torch.zeros(16),
            )
        assert len(buffer) == 5
        buffer.reset()
        assert len(buffer) == 0
        assert not buffer.is_full()

    def test_gae_computation(self, buffer):
        """Test GAE advantage computation."""
        for i in range(5):
            buffer.add(
                features=np.array([0.1] * 4, dtype=np.float32),
                action=0,
                log_prob=-1.0,
                value=float(i) * 0.1,
                reward=1.0,
                done=(i == 4),
                h_state=torch.zeros(16),
                c_state=torch.zeros(16),
            )

        returns, advantages = buffer.compute_returns_and_advantages(
            last_value=0.0,
            gamma=0.99,
            gae_lambda=0.95,
        )
        assert returns.shape == (5,)
        assert advantages.shape == (5,)
        # Returns should be positive (positive rewards)
        assert returns.sum().item() > 0

    def test_sequential_chunks(self, buffer):
        """Test sequential chunk generation with correct h_init/c_init."""
        hidden_dim = 8
        for i in range(8):
            h = torch.randn(hidden_dim)
            c = torch.randn(hidden_dim)
            buffer.add(
                features=np.array([float(i)] * 4, dtype=np.float32),
                action=i % 4,
                log_prob=-0.5,
                value=0.5,
                reward=0.1,
                done=False,
                h_state=h,
                c_state=c,
            )

        returns = torch.ones(8)
        advantages = torch.ones(8)

        chunks = list(buffer.get_sequential_chunks(4, returns, advantages))
        assert len(chunks) == 2  # 8 steps / 4 chunk_length = 2 chunks

        # Each chunk should have correct h_init from buffer
        for chunk in chunks:
            start = chunk["start"]
            torch.testing.assert_close(chunk["h_init"], buffer.h_states[start])
            torch.testing.assert_close(chunk["c_init"], buffer.c_states[start])
            assert chunk["end"] - start == 4

    def test_gru_buffer_none_c_state(self):
        """Test buffer handles None c_state for GRU."""
        buf = LSTMPPORolloutBuffer(buffer_size=5, device=torch.device("cpu"))
        buf.add(
            features=np.array([0.1] * 4, dtype=np.float32),
            action=0,
            log_prob=-0.5,
            value=0.5,
            reward=0.1,
            done=False,
            h_state=torch.zeros(8),
            c_state=None,
        )
        assert buf.c_states[0] is None

    def test_episode_boundary_in_chunks(self):
        """Test that episode boundaries are preserved within chunks."""
        buf = LSTMPPORolloutBuffer(buffer_size=8, device=torch.device("cpu"))
        for i in range(8):
            buf.add(
                features=np.array([float(i)] * 4, dtype=np.float32),
                action=0,
                log_prob=-0.5,
                value=0.5,
                reward=0.1,
                done=(i == 3),  # Episode boundary at step 3
                h_state=torch.zeros(8),
                c_state=torch.zeros(8),
            )

        returns = torch.ones(8)
        advantages = torch.ones(8)
        chunks = list(buf.get_sequential_chunks(8, returns, advantages))
        assert len(chunks) == 1
        # The chunk should contain the done flag
        assert chunks[0]["dones"][3] is True


# ──────────────────────────────────────────────────────────────────────────────
# 3.3 Brain Construction Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestLSTMPPOBrainConstruction:
    """Test cases for brain construction."""

    def test_lstm_construction(self):
        """Test brain construction with LSTM."""
        brain = _make_brain()
        assert brain.input_dim == INPUT_DIM
        assert brain.num_actions == 4
        assert not brain._is_gru
        assert isinstance(brain.rnn, torch.nn.LSTM)
        assert brain.h_t.shape == (1, 1, 16)
        assert brain.c_t is not None
        assert brain.c_t.shape == (1, 1, 16)

    def test_gru_construction(self):
        """Test brain construction with GRU."""
        brain = _make_brain(rnn_type="gru")
        assert brain._is_gru
        assert isinstance(brain.rnn, torch.nn.GRU)
        assert brain.h_t.shape == (1, 1, 16)
        assert brain.c_t is None

    def test_parameter_count(self):
        """Test that parameters are distributed correctly."""
        brain = _make_brain()
        actor_params = set(brain.actor_optimizer.param_groups[0]["params"])
        critic_params = set(brain.critic_optimizer.param_groups[0]["params"])
        # No overlap between actor and critic parameters
        assert len(actor_params & critic_params) == 0

    def test_separate_optimizers(self):
        """Test that actor and critic have separate optimizers."""
        brain = _make_brain()
        assert brain.actor_optimizer is not brain.critic_optimizer
        # Actor optimizer covers RNN + LayerNorm + actor
        actor_param_ids = {id(p) for p in brain.actor_optimizer.param_groups[0]["params"]}
        for p in brain.rnn.parameters():
            assert id(p) in actor_param_ids
        for p in brain.feature_norm.parameters():
            assert id(p) in actor_param_ids
        for p in brain.actor.parameters():
            assert id(p) in actor_param_ids


# ──────────────────────────────────────────────────────────────────────────────
# 3.4 Single-Step Test
# ──────────────────────────────────────────────────────────────────────────────


class TestLSTMPPOBrainSingleStep:
    """Test single-step brain execution."""

    def test_run_brain_returns_action(self):
        """Test that run_brain returns a valid ActionData."""
        brain = _make_brain()
        brain.prepare_episode()
        params = _make_params()

        actions = brain.run_brain(params, top_only=False, top_randomize=False)
        assert len(actions) == 1
        assert isinstance(actions[0], ActionData)
        assert actions[0].action is not None

    def test_hidden_state_updates(self):
        """Test that hidden state changes after a step."""
        brain = _make_brain()
        brain.prepare_episode()

        h_before = brain.h_t.clone()
        params = _make_params()
        brain.run_brain(params, top_only=False, top_randomize=False)
        h_after = brain.h_t.clone()

        # Hidden state should have changed
        assert not torch.equal(h_before, h_after)


# ──────────────────────────────────────────────────────────────────────────────
# 3.5 Multi-Step + Learn Test
# ──────────────────────────────────────────────────────────────────────────────


class TestLSTMPPOBrainMultiStep:
    """Test multi-step execution with learning."""

    def test_buffer_fills_and_ppo_update(self):
        """Test that buffer fills and PPO update runs without error."""
        brain = _make_brain(rollout_buffer_size=16, bptt_chunk_length=4)
        brain.prepare_episode()

        params = _make_params()
        for step in range(20):
            brain.run_brain(params, top_only=False, top_randomize=False)
            done = step == 19
            brain.learn(params, reward=0.1, episode_done=done)

        # After enough steps, a PPO update should have occurred
        assert brain.history_data.losses, "Expected at least one loss recorded"

    def test_loss_is_computed(self):
        """Test that loss is finite after PPO update."""
        brain = _make_brain(rollout_buffer_size=16, bptt_chunk_length=4)
        brain.prepare_episode()

        params = _make_params()
        for step in range(20):
            brain.run_brain(params, top_only=False, top_randomize=False)
            brain.learn(params, reward=0.1, episode_done=(step == 19))

        for loss in brain.history_data.losses:
            assert np.isfinite(loss), f"Non-finite loss: {loss}"


# ──────────────────────────────────────────────────────────────────────────────
# 3.6 Episode Boundary Test
# ──────────────────────────────────────────────────────────────────────────────


class TestLSTMPPOBrainEpisodeBoundary:
    """Test episode boundary handling."""

    def test_prepare_episode_resets_hidden_state(self):
        """Test that prepare_episode resets hidden state to zeros."""
        brain = _make_brain()
        brain.prepare_episode()

        # Run a few steps to change hidden state
        params = _make_params()
        for _ in range(3):
            brain.run_brain(params, top_only=False, top_randomize=False)
            brain.learn(params, reward=0.1, episode_done=False)

        # Hidden state should not be zero
        assert not torch.all(brain.h_t == 0)

        # Reset
        brain.prepare_episode()

        # Hidden state should be zero again
        assert torch.all(brain.h_t == 0)
        if brain.c_t is not None:
            assert torch.all(brain.c_t == 0)

    def test_new_episode_starts_fresh(self):
        """Test that a new episode starts with fresh hidden state."""
        brain = _make_brain()

        # Run episode 1
        brain.prepare_episode()
        params = _make_params()
        for _ in range(5):
            brain.run_brain(params, top_only=False, top_randomize=False)
            brain.learn(params, reward=0.1, episode_done=False)
        brain.post_process_episode()

        h_after_ep1 = brain.h_t.clone()

        # Start episode 2
        brain.prepare_episode()

        # Should be zeros, not carryover from episode 1
        assert torch.all(brain.h_t == 0)
        assert not torch.equal(h_after_ep1, brain.h_t)


# ──────────────────────────────────────────────────────────────────────────────
# 3.7 GRU End-to-End Test
# ──────────────────────────────────────────────────────────────────────────────


class TestLSTMPPOBrainGRU:
    """Test GRU variant end-to-end."""

    def test_gru_end_to_end(self):
        """Test GRU: construct, run steps, learn, verify PPO update completes."""
        brain = _make_brain(rnn_type="gru", rollout_buffer_size=16, bptt_chunk_length=4)
        assert brain._is_gru
        assert brain.c_t is None

        brain.prepare_episode()
        params = _make_params()

        for step in range(20):
            brain.run_brain(params, top_only=False, top_randomize=False)
            brain.learn(params, reward=0.1, episode_done=(step == 19))

        assert brain.history_data.losses, "GRU should produce losses"
        for loss in brain.history_data.losses:
            assert np.isfinite(loss)


# ──────────────────────────────────────────────────────────────────────────────
# 3.8 Weight Persistence Round-Trip Test
# ──────────────────────────────────────────────────────────────────────────────


class TestLSTMPPOBrainWeightPersistence:
    """Test weight persistence save/load round-trip."""

    def test_save_load_roundtrip(self):
        """Test that save and load produces same outputs."""
        brain1 = _make_brain()
        brain1.prepare_episode()

        # Run some steps to modify weights
        params = _make_params()
        for step in range(32):
            brain1.run_brain(params, top_only=False, top_randomize=False)
            brain1.learn(params, reward=0.1, episode_done=(step == 31))
        brain1.post_process_episode()

        # Save weights
        components = brain1.get_weight_components()
        assert "lstm" in components
        assert "layer_norm" in components
        assert "policy" in components
        assert "value" in components
        assert "actor_optimizer" in components
        assert "critic_optimizer" in components
        assert "training_state" in components

        # Create new brain and load weights
        brain2 = _make_brain()
        brain2.load_weight_components(components)

        # Verify episode count was restored
        assert brain2._episode_count == brain1._episode_count

        # Verify network weights match
        for p1, p2 in zip(brain1.rnn.parameters(), brain2.rnn.parameters(), strict=True):
            torch.testing.assert_close(p1, p2)
        for p1, p2 in zip(brain1.actor.parameters(), brain2.actor.parameters(), strict=True):
            torch.testing.assert_close(p1, p2)
        for p1, p2 in zip(brain1.critic.parameters(), brain2.critic.parameters(), strict=True):
            torch.testing.assert_close(p1, p2)

        # Verify same deterministic output on same input
        # Reset both brains to same RNG state and hidden state
        brain1.prepare_episode()
        brain2.prepare_episode()
        brain1.rng = np.random.default_rng(99)
        brain2.rng = np.random.default_rng(99)
        actions1 = brain1.run_brain(params, top_only=False, top_randomize=False)
        actions2 = brain2.run_brain(params, top_only=False, top_randomize=False)
        assert actions1[0].action == actions2[0].action

    def test_selective_components(self):
        """Test loading a subset of weight components."""
        brain = _make_brain()
        components = brain.get_weight_components(components={"policy", "value"})
        assert set(components.keys()) == {"policy", "value"}

    def test_unknown_component_raises(self):
        """Test that unknown component names raise ValueError."""
        brain = _make_brain()
        with pytest.raises(ValueError, match="Unknown weight components"):
            brain.get_weight_components(components={"nonexistent"})


# ──────────────────────────────────────────────────────────────────────────────
# 3.9 Sensory Module Compatibility Test
# ──────────────────────────────────────────────────────────────────────────────


class TestLSTMPPOBrainSensoryModules:
    """Test compatibility with various sensory modules."""

    def test_temporal_derivative_modules(self):
        """Test with temporal/derivative sensory modules + STAM."""
        modules = [
            ModuleName.FOOD_CHEMOTAXIS_TEMPORAL,
            ModuleName.PROPRIOCEPTION,
            ModuleName.STAM,
        ]
        brain = _make_brain(sensory_modules=modules)
        # With only food temporal module, STAM infers 1 channel
        from quantumnematode.agent.stam import compute_memory_dim

        stam_dim = compute_memory_dim(1)  # Only food channel inferred
        expected_dim = 2 + 2 + stam_dim
        assert brain.input_dim == expected_dim

        brain.prepare_episode()
        params = BrainParams(
            food_gradient_strength=0.5,
            food_gradient_direction=np.pi / 2,
            agent_direction=Direction.UP,
            food_concentration=0.7,
            food_dconcentration_dt=0.1,
            stam_state=(0.1,) * stam_dim,
        )
        actions = brain.run_brain(params, top_only=False, top_randomize=False)
        assert len(actions) == 1

    def test_nociception_and_mechanosensation(self):
        """Test with derivative nociception and mechanosensation modules."""
        modules = [
            ModuleName.FOOD_CHEMOTAXIS,
            ModuleName.NOCICEPTION,
            ModuleName.MECHANOSENSATION,
            ModuleName.PROPRIOCEPTION,
        ]
        brain = _make_brain(sensory_modules=modules)
        assert brain.input_dim == 8  # 4 modules * 2 features each

        brain.prepare_episode()
        params = BrainParams(
            food_gradient_strength=0.5,
            food_gradient_direction=np.pi / 2,
            agent_direction=Direction.UP,
            predator_gradient_strength=0.3,
            predator_gradient_direction=np.pi,
            boundary_contact=False,
            predator_contact=False,
        )
        actions = brain.run_brain(params, top_only=False, top_randomize=False)
        assert len(actions) == 1


# ──────────────────────────────────────────────────────────────────────────────
# Entropy Decay and LR Scheduling
# ──────────────────────────────────────────────────────────────────────────────


class TestLSTMPPOBrainScheduling:
    """Test entropy decay and learning rate scheduling."""

    def test_entropy_decays_over_episodes(self):
        """Test that entropy coefficient decreases over episodes."""
        brain = _make_brain(entropy_coef=0.1, entropy_coef_end=0.01, entropy_decay_episodes=100)
        coef_start = brain._get_entropy_coef()
        assert coef_start == pytest.approx(0.1)

        # Simulate 50 episodes
        for _ in range(50):
            brain.post_process_episode()
        coef_mid = brain._get_entropy_coef()
        assert coef_mid < coef_start
        assert coef_mid > 0.01

        # Simulate to end of decay
        for _ in range(50):
            brain.post_process_episode()
        coef_end = brain._get_entropy_coef()
        assert coef_end == pytest.approx(0.01)

        # Past decay episodes, stays at end value
        brain.post_process_episode()
        assert brain._get_entropy_coef() == pytest.approx(0.01)

    def test_lr_warmup_and_decay(self):
        """Test that learning rate follows warmup then decay schedule."""
        brain = _make_brain(
            actor_lr=0.001,
            critic_lr=0.002,
            lr_warmup_episodes=10,
            lr_warmup_start=0.0001,
            lr_decay_episodes=20,
            lr_decay_end=0.0001,
        )
        # At episode 0: should be at warmup start
        lr_0 = brain._get_current_lr()
        assert lr_0 == pytest.approx(0.0001)

        # Warmup to episode 10
        for _ in range(10):
            brain.post_process_episode()
        lr_10 = brain._get_current_lr()
        assert lr_10 == pytest.approx(0.001)

        # Decay: at episode 20 (midpoint of decay)
        for _ in range(10):
            brain.post_process_episode()
        lr_20 = brain._get_current_lr()
        assert lr_20 < 0.001
        assert lr_20 > 0.0001

        # End of decay at episode 30
        for _ in range(10):
            brain.post_process_episode()
        lr_30 = brain._get_current_lr()
        assert lr_30 == pytest.approx(0.0001)

    def test_critic_lr_scales_proportionally(self):
        """Test that critic LR maintains ratio with actor LR."""
        brain = _make_brain(
            actor_lr=0.001,
            critic_lr=0.002,
            lr_warmup_episodes=10,
            lr_warmup_start=0.0001,
        )
        # At start of warmup, critic should be 2x actor
        actor_lr = brain.actor_optimizer.param_groups[0]["lr"]
        critic_lr = brain.critic_optimizer.param_groups[0]["lr"]
        assert critic_lr / actor_lr == pytest.approx(2.0, rel=1e-4)

        # After some warmup, ratio should be preserved
        for _ in range(5):
            brain.post_process_episode()
        actor_lr = brain.actor_optimizer.param_groups[0]["lr"]
        critic_lr = brain.critic_optimizer.param_groups[0]["lr"]
        assert critic_lr / actor_lr == pytest.approx(2.0, rel=1e-4)

    def test_prepare_episode_resets_pending_and_step_count(self):
        """Test that prepare_episode resets pending features and step count."""
        brain = _make_brain()
        brain.prepare_episode()

        params = _make_params()
        brain.run_brain(params, top_only=False, top_randomize=False)
        assert brain._pending_features is not None
        assert brain._step_count == 1

        brain.prepare_episode()
        assert brain._pending_features is None
        assert brain._step_count == 0


# ──────────────────────────────────────────────────────────────────────────────
# Protocol Compliance
# ──────────────────────────────────────────────────────────────────────────────


class TestLSTMPPOBrainProtocol:
    """Test Brain protocol compliance."""

    def test_update_memory_noop(self):
        """Test update_memory is a no-op."""
        brain = _make_brain()
        brain.update_memory(reward=1.0)  # Should not raise

    def test_copy_raises(self):
        """Test copy raises NotImplementedError."""
        brain = _make_brain()
        with pytest.raises(NotImplementedError):
            brain.copy()

    def test_build_brain_raises(self):
        """Test build_brain raises NotImplementedError."""
        brain = _make_brain()
        with pytest.raises(NotImplementedError):
            brain.build_brain()

    def test_action_set_property(self):
        """Test action_set getter and setter."""
        brain = _make_brain()
        original = brain.action_set
        assert len(original) == 4

        new_actions = original[:2]
        brain.action_set = new_actions
        assert brain.action_set == new_actions
