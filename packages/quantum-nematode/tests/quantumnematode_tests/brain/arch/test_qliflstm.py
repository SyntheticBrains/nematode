"""Unit tests for the QLIF-LSTM brain architecture."""

import numpy as np
import pytest
import torch
from quantumnematode.brain.actions import ActionData
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.brain.arch.qliflstm import (
    QLIFLSTMBrain,
    QLIFLSTMBrainConfig,
    QLIFLSTMCell,
    QLIFLSTMCritic,
    QLIFLSTMRolloutBuffer,
)
from quantumnematode.brain.modules import ModuleName
from quantumnematode.env import Direction

# ──────────────────────────────────────────────────────────────────────
# Config Tests
# ──────────────────────────────────────────────────────────────────────


class TestQLIFLSTMBrainConfig:
    """Test cases for QLIF-LSTM brain configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = QLIFLSTMBrainConfig(
            sensory_modules=[ModuleName.FOOD_CHEMOTAXIS],
        )
        assert config.lstm_hidden_dim == 32
        assert config.shots == 1024
        assert config.membrane_tau == 0.9
        assert config.refractory_period == 0
        assert config.gamma == 0.99
        assert config.gae_lambda == 0.95
        assert config.clip_epsilon == 0.2
        assert config.entropy_coef == 0.05
        assert config.entropy_coef_end == 0.005
        assert config.entropy_decay_episodes == 200
        assert config.value_loss_coef == 0.5
        assert config.num_epochs == 2
        assert config.rollout_buffer_size == 256
        assert config.max_grad_norm == 0.5
        assert config.actor_lr == 0.003
        assert config.critic_lr == 0.001
        assert config.critic_hidden_dim == 64
        assert config.critic_num_layers == 2
        assert config.bptt_chunk_length == 16
        assert config.use_quantum_gates is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = QLIFLSTMBrainConfig(
            sensory_modules=[ModuleName.FOOD_CHEMOTAXIS],
            lstm_hidden_dim=16,
            shots=512,
            gamma=0.95,
            bptt_chunk_length=8,
            use_quantum_gates=False,
        )
        assert config.lstm_hidden_dim == 16
        assert config.shots == 512
        assert config.gamma == 0.95
        assert config.bptt_chunk_length == 8
        assert config.use_quantum_gates is False

    def test_invalid_lstm_hidden_dim(self):
        """Test validation rejects too small hidden dim."""
        with pytest.raises(ValueError, match="lstm_hidden_dim must be >= 2"):
            QLIFLSTMBrainConfig(sensory_modules=[ModuleName.FOOD_CHEMOTAXIS], lstm_hidden_dim=1)

    def test_invalid_shots(self):
        """Test validation rejects too few shots."""
        with pytest.raises(ValueError, match="shots must be >= 100"):
            QLIFLSTMBrainConfig(sensory_modules=[ModuleName.FOOD_CHEMOTAXIS], shots=50)

    def test_invalid_membrane_tau(self):
        """Test validation rejects out-of-range membrane_tau."""
        with pytest.raises(ValueError, match="membrane_tau must be in"):
            QLIFLSTMBrainConfig(sensory_modules=[ModuleName.FOOD_CHEMOTAXIS], membrane_tau=0.0)
        with pytest.raises(ValueError, match="membrane_tau must be in"):
            QLIFLSTMBrainConfig(sensory_modules=[ModuleName.FOOD_CHEMOTAXIS], membrane_tau=1.5)

    def test_invalid_num_epochs(self):
        """Test validation rejects zero epochs."""
        with pytest.raises(ValueError, match="num_epochs must be >= 1"):
            QLIFLSTMBrainConfig(sensory_modules=[ModuleName.FOOD_CHEMOTAXIS], num_epochs=0)

    def test_invalid_bptt_chunk_length(self):
        """Test validation rejects too small chunk length."""
        with pytest.raises(ValueError, match="bptt_chunk_length must be >= 4"):
            QLIFLSTMBrainConfig(sensory_modules=[ModuleName.FOOD_CHEMOTAXIS], bptt_chunk_length=2)

    def test_sensory_modules_config(self):
        """Test sensory module configuration."""
        config = QLIFLSTMBrainConfig(
            sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
        )
        assert config.sensory_modules is not None
        assert len(config.sensory_modules) == 2


# ──────────────────────────────────────────────────────────────────────
# QLIF-LSTM Cell Tests
# ──────────────────────────────────────────────────────────────────────


class TestQLIFLSTMCell:
    """Test cases for the QLIF-LSTM cell."""

    @pytest.fixture
    def classical_cell(self) -> QLIFLSTMCell:
        """Create a cell in classical ablation mode."""
        return QLIFLSTMCell(
            input_dim=4,
            hidden_dim=8,
            use_quantum_gates=False,
        )

    def test_cell_forward_shape(self, classical_cell: QLIFLSTMCell):
        """Test cell forward pass produces correct output shapes."""
        x_t = torch.randn(4)
        h_prev = torch.zeros(8)
        c_prev = torch.zeros(8)

        h_t, c_t = classical_cell(x_t, h_prev, c_prev)

        assert h_t.shape == (8,)
        assert c_t.shape == (8,)

    def test_cell_output_finite(self, classical_cell: QLIFLSTMCell):
        """Test cell outputs are finite."""
        x_t = torch.randn(4)
        h_prev = torch.zeros(8)
        c_prev = torch.zeros(8)

        h_t, c_t = classical_cell(x_t, h_prev, c_prev)

        assert torch.all(torch.isfinite(h_t))
        assert torch.all(torch.isfinite(c_t))

    def test_cell_hidden_state_bounded(self, classical_cell: QLIFLSTMCell):
        """Test h_t is bounded by tanh output gate (in [-1, 1])."""
        x_t = torch.randn(4) * 10  # Large input
        h_prev = torch.zeros(8)
        c_prev = torch.zeros(8)

        h_t, _c_t = classical_cell(x_t, h_prev, c_prev)

        # h_t = o_t * tanh(c_t), both bounded in [-1, 1]
        assert torch.all(h_t.abs() <= 1.0 + 1e-6)

    def test_cell_classical_ablation_differentiable(self, classical_cell: QLIFLSTMCell):
        """Test classical ablation mode supports backpropagation."""
        x_t = torch.randn(4, requires_grad=True)
        h_prev = torch.zeros(8)
        c_prev = torch.zeros(8)

        h_t, _c_t = classical_cell(x_t, h_prev, c_prev)
        loss = h_t.sum()
        loss.backward()

        assert x_t.grad is not None
        assert torch.all(torch.isfinite(x_t.grad))

    def test_cell_forget_gate_bias(self, classical_cell: QLIFLSTMCell):
        """Test forget gate bias initialized to 1.0 (remembering bias)."""
        assert classical_cell.W_f.bias is not None
        assert torch.allclose(
            classical_cell.W_f.bias,
            torch.ones_like(classical_cell.W_f.bias),
        )

    def test_cell_sequential_state_update(self, classical_cell: QLIFLSTMCell):
        """Test that sequential calls update state correctly."""
        x1 = torch.randn(4)
        x2 = torch.randn(4)
        h = torch.zeros(8)
        c = torch.zeros(8)

        h1, c1 = classical_cell(x1, h, c)
        h2, _c2 = classical_cell(x2, h1, c1)

        # States should differ after different inputs
        assert not torch.allclose(h1, h2)

    def test_quantum_cell_forward_and_backward(self):
        """Smoke test: quantum gate path runs forward + backward without error."""
        from quantumnematode.brain.arch._quantum_utils import get_qiskit_backend

        backend = get_qiskit_backend(device=DeviceType.CPU)
        cell = QLIFLSTMCell(
            input_dim=2,
            hidden_dim=2,
            use_quantum_gates=True,
            shots=100,
            backend=backend,
        )

        x_t = torch.randn(2)
        h_prev = torch.zeros(2)
        c_prev = torch.zeros(2)

        h_t, c_t = cell(x_t, h_prev, c_prev)

        assert h_t.shape == (2,)
        assert c_t.shape == (2,)
        assert torch.all(torch.isfinite(h_t))
        assert torch.all(torch.isfinite(c_t))

        # Backward pass through surrogate gradients
        loss = h_t.sum()
        loss.backward()

        # Verify gradients flowed through quantum surrogate path
        assert cell.theta_forget.grad is not None
        assert cell.theta_input.grad is not None

    def test_cell_four_linear_projections(self, classical_cell: QLIFLSTMCell):
        """Test cell has 4 separate linear projections."""
        assert hasattr(classical_cell, "W_f")
        assert hasattr(classical_cell, "W_i")
        assert hasattr(classical_cell, "W_c")
        assert hasattr(classical_cell, "W_o")

        expected_in = 4 + 8  # input_dim + hidden_dim
        for layer in [
            classical_cell.W_f,
            classical_cell.W_i,
            classical_cell.W_c,
            classical_cell.W_o,
        ]:
            assert layer.in_features == expected_in
            assert layer.out_features == 8


# ──────────────────────────────────────────────────────────────────────
# Rollout Buffer Tests
# ──────────────────────────────────────────────────────────────────────


class TestQLIFLSTMRolloutBuffer:
    """Test cases for the QLIF-LSTM rollout buffer."""

    _rng = np.random.default_rng(99)

    @pytest.fixture
    def buffer(self) -> QLIFLSTMRolloutBuffer:
        """Create a test rollout buffer."""
        return QLIFLSTMRolloutBuffer(
            buffer_size=16,
            device=torch.device("cpu"),
            rng=np.random.default_rng(42),
        )

    def _add_step(
        self,
        buffer: QLIFLSTMRolloutBuffer,
        *,
        done: bool = False,
    ) -> None:
        """Add a single step to the buffer."""
        buffer.add(
            features=self._rng.standard_normal(4).astype(np.float32),
            action=int(self._rng.integers(0, 4)),
            log_prob=-0.5,
            value=0.5,
            reward=0.1,
            done=done,
            h_state=torch.randn(8),
            c_state=torch.randn(8),
        )

    def test_initialization(self, buffer: QLIFLSTMRolloutBuffer):
        """Test buffer initialization."""
        assert buffer.buffer_size == 16
        assert len(buffer) == 0
        assert not buffer.is_full()

    def test_add_experience(self, buffer: QLIFLSTMRolloutBuffer):
        """Test adding experience to the buffer."""
        self._add_step(buffer)
        assert len(buffer) == 1
        assert not buffer.is_full()
        assert len(buffer.h_states) == 1
        assert len(buffer.c_states) == 1

    def test_buffer_full(self, buffer: QLIFLSTMRolloutBuffer):
        """Test buffer fills to capacity."""
        for _ in range(16):
            self._add_step(buffer)
        assert buffer.is_full()
        assert len(buffer) == 16

    def test_reset(self, buffer: QLIFLSTMRolloutBuffer):
        """Test buffer reset clears all data."""
        self._add_step(buffer)
        buffer.reset()
        assert len(buffer) == 0
        assert len(buffer.h_states) == 0
        assert len(buffer.c_states) == 0

    def test_compute_returns_and_advantages(self, buffer: QLIFLSTMRolloutBuffer):
        """Test GAE computation produces correct shapes and finite values."""
        for i in range(8):
            self._add_step(buffer, done=(i == 7))

        returns, advantages = buffer.compute_returns_and_advantages(
            last_value=0.0,
            gamma=0.99,
            gae_lambda=0.95,
        )

        assert returns.shape == (8,)
        assert advantages.shape == (8,)
        assert torch.all(torch.isfinite(returns))
        assert torch.all(torch.isfinite(advantages))

    def test_sequential_chunks(self, buffer: QLIFLSTMRolloutBuffer):
        """Test chunk generation splits buffer correctly."""
        for i in range(12):
            self._add_step(buffer, done=(i == 11))

        returns, advantages = buffer.compute_returns_and_advantages(
            last_value=0.0,
            gamma=0.99,
            gae_lambda=0.95,
        )

        chunks = list(buffer.get_sequential_chunks(4, returns, advantages))
        # 12 steps / 4 chunk_length = 3 chunks
        assert len(chunks) == 3

        for chunk in chunks:
            assert "start" in chunk
            assert "end" in chunk
            assert "h_init" in chunk
            assert "c_init" in chunk
            assert "features" in chunk
            assert "actions" in chunk
            assert "old_log_probs" in chunk
            assert "returns" in chunk
            assert "advantages" in chunk
            assert "dones" in chunk

    def test_chunk_h_init_matches_stored(self, buffer: QLIFLSTMRolloutBuffer):
        """Test chunk initial hidden states come from stored values."""
        h_states_stored = []
        for _ in range(8):
            h = torch.randn(8)
            buffer.add(
                features=np.zeros(4, dtype=np.float32),
                action=0,
                log_prob=-0.5,
                value=0.5,
                reward=0.1,
                done=False,
                h_state=h,
                c_state=torch.zeros(8),
            )
            h_states_stored.append(h.clone())

        returns, advantages = buffer.compute_returns_and_advantages(
            last_value=0.0,
            gamma=0.99,
            gae_lambda=0.95,
        )

        for chunk in buffer.get_sequential_chunks(4, returns, advantages):
            start = chunk["start"]
            assert torch.allclose(chunk["h_init"], h_states_stored[start])

    def test_chunk_last_partial(self, buffer: QLIFLSTMRolloutBuffer):
        """Test last chunk can be shorter than chunk_length."""
        for _ in range(10):
            self._add_step(buffer)

        returns, advantages = buffer.compute_returns_and_advantages(
            last_value=0.0,
            gamma=0.99,
            gae_lambda=0.95,
        )

        chunks = list(buffer.get_sequential_chunks(4, returns, advantages))
        # 10 steps / 4 = 3 chunks (4, 4, 2)
        assert len(chunks) == 3
        lengths = [chunk["end"] - chunk["start"] for chunk in chunks]
        assert sorted(lengths) == [2, 4, 4]


# ──────────────────────────────────────────────────────────────────────
# Critic Tests
# ──────────────────────────────────────────────────────────────────────


class TestQLIFLSTMCritic:
    """Test cases for the QLIF-LSTM critic network."""

    @pytest.fixture
    def critic(self) -> QLIFLSTMCritic:
        """Create a test critic."""
        return QLIFLSTMCritic(input_dim=12, hidden_dim=32, num_layers=2)

    def test_forward_scalar_output(self, critic: QLIFLSTMCritic):
        """Test critic forward pass produces scalar output."""
        test_input = torch.randn(12)
        output = critic(test_input)
        assert output.shape == ()

    def test_forward_finite_values(self, critic: QLIFLSTMCritic):
        """Test critic outputs are finite."""
        test_input = torch.randn(5, 12)
        output = critic(test_input)
        assert torch.all(torch.isfinite(output))


# ──────────────────────────────────────────────────────────────────────
# Brain Tests (Classical Ablation Mode)
# ──────────────────────────────────────────────────────────────────────


class TestQLIFLSTMBrain:
    """Test cases for the QLIF-LSTM brain in classical ablation mode.

    Uses ``use_quantum_gates=False`` to avoid quantum circuit overhead in
    unit tests. Quantum-mode integration is verified via smoke tests.
    """

    @pytest.fixture
    def config(self) -> QLIFLSTMBrainConfig:
        """Create a small test configuration (classical mode)."""
        return QLIFLSTMBrainConfig(
            sensory_modules=[ModuleName.FOOD_CHEMOTAXIS],
            lstm_hidden_dim=8,
            use_quantum_gates=False,
            rollout_buffer_size=8,
            bptt_chunk_length=4,
            num_epochs=1,
            seed=42,
        )

    @pytest.fixture
    def brain(self, config: QLIFLSTMBrainConfig) -> QLIFLSTMBrain:
        """Create a test QLIF-LSTM brain."""
        return QLIFLSTMBrain(config=config, num_actions=4, device=DeviceType.CPU)

    @pytest.fixture
    def params(self) -> BrainParams:
        """Create test BrainParams."""
        return BrainParams(
            gradient_strength=0.6,
            gradient_direction=0.3,
            agent_direction=Direction.UP,
        )

    def test_initialization(self, brain: QLIFLSTMBrain):
        """Test brain initializes with correct attributes."""
        assert brain.num_actions == 4
        assert brain.config.lstm_hidden_dim == 8
        assert brain.lstm_cell is not None
        assert brain.actor_head is not None
        assert brain.critic is not None
        assert brain.h_t.shape == (8,)
        assert brain.c_t.shape == (8,)

    def test_hidden_state_initially_zero(self, brain: QLIFLSTMBrain):
        """Test LSTM hidden state is initialized to zeros."""
        assert torch.allclose(brain.h_t, torch.zeros(8))
        assert torch.allclose(brain.c_t, torch.zeros(8))

    def test_run_brain(self, brain: QLIFLSTMBrain, params: BrainParams):
        """Test run_brain produces valid action data."""
        result = brain.run_brain(params, top_only=False, top_randomize=False)

        assert len(result) == 1
        assert isinstance(result[0], ActionData)
        assert result[0].probability > 0

    def test_run_brain_updates_hidden_state(self, brain: QLIFLSTMBrain, params: BrainParams):
        """Test run_brain updates LSTM hidden state."""
        h_before = brain.h_t.clone()
        brain.run_brain(params, top_only=False, top_randomize=False)

        # Hidden state should change after processing input
        # (unless by extreme coincidence all gates produce zero)
        assert not torch.allclose(brain.h_t, h_before)

    def test_run_brain_stores_pending(self, brain: QLIFLSTMBrain, params: BrainParams):
        """Test run_brain stores pending data for learn()."""
        brain.run_brain(params, top_only=False, top_randomize=False)

        assert brain._pending_features is not None
        assert isinstance(brain._pending_action, int)
        assert isinstance(brain._pending_log_prob, float)
        assert isinstance(brain._pending_value, float)

    def test_learn_adds_to_buffer(self, brain: QLIFLSTMBrain, params: BrainParams):
        """Test learn() adds experience to buffer."""
        brain.run_brain(params, top_only=False, top_randomize=False)
        brain.learn(params, reward=1.0, episode_done=False)

        assert len(brain.buffer) == 1

    def test_learn_triggers_update_when_full(self, brain: QLIFLSTMBrain, params: BrainParams):
        """Test learn() triggers PPO update when buffer is full."""
        for _ in range(8):
            brain.run_brain(params, top_only=False, top_randomize=False)
            brain.learn(params, reward=0.1, episode_done=False)

        # Buffer should have been cleared after PPO update
        assert len(brain.buffer) == 0

    def test_learn_triggers_update_on_episode_done(
        self,
        brain: QLIFLSTMBrain,
        params: BrainParams,
    ):
        """Test learn() triggers PPO update when episode done and buffer is full."""
        # Fill buffer to exactly bptt_chunk_length (4 steps, less than buffer_size=8)
        for _ in range(brain.config.bptt_chunk_length):
            brain.run_brain(params, top_only=False, top_randomize=False)
            brain.learn(params, reward=0.1, episode_done=False)

        # Buffer should still have data (not full, episode not done)
        assert len(brain.buffer) == brain.config.bptt_chunk_length

        # Now signal episode done — should trigger PPO update and clear buffer
        brain.run_brain(params, top_only=False, top_randomize=False)
        brain.learn(params, reward=0.1, episode_done=True)
        assert len(brain.buffer) == 0

    def test_learn_no_update_on_episode_done_insufficient_data(
        self,
        brain: QLIFLSTMBrain,
        params: BrainParams,
    ):
        """Test learn() does NOT trigger PPO update when episode done but buffer < buffer_size."""
        # Fill buffer with fewer steps than bptt_chunk_length (4).
        # We need total steps strictly < bptt_chunk_length, so use bptt_chunk_length - 2
        # steps in the loop + 1 final step = bptt_chunk_length - 1 total.
        for _ in range(brain.config.bptt_chunk_length - 2):
            brain.run_brain(params, top_only=False, top_randomize=False)
            brain.learn(params, reward=0.1, episode_done=False)

        # Signal episode done with insufficient data — buffer should NOT be cleared
        brain.run_brain(params, top_only=False, top_randomize=False)
        brain.learn(params, reward=0.1, episode_done=True)
        assert len(brain.buffer) == brain.config.bptt_chunk_length - 1

    def test_prepare_episode_resets_hidden_state(self, brain: QLIFLSTMBrain, params: BrainParams):
        """Test prepare_episode() resets LSTM hidden state to zeros."""
        # Run a few steps to change hidden state
        brain.run_brain(params, top_only=False, top_randomize=False)

        # Reset
        brain.prepare_episode()

        assert torch.allclose(brain.h_t, torch.zeros(8))
        assert torch.allclose(brain.c_t, torch.zeros(8))
        assert brain._pending_features is None

    def test_post_process_episode(self, brain: QLIFLSTMBrain):
        """Test post_process_episode increments counter."""
        assert brain._episode_count == 0
        brain.post_process_episode()
        assert brain._episode_count == 1

    def test_copy(self, brain: QLIFLSTMBrain, params: BrainParams):
        """Test copy creates independent brain with fresh hidden state."""
        brain.run_brain(params, top_only=False, top_randomize=False)

        copy = brain.copy()

        # Copy should have zeroed hidden state
        assert torch.allclose(copy.h_t, torch.zeros(8))
        assert torch.allclose(copy.c_t, torch.zeros(8))

        # Original should be unchanged
        assert not torch.allclose(brain.h_t, torch.zeros(8))

    def test_sensory_module_mode(self):
        """Test brain works with unified sensory modules."""
        config = QLIFLSTMBrainConfig(
            lstm_hidden_dim=8,
            use_quantum_gates=False,
            sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
            rollout_buffer_size=8,
            bptt_chunk_length=4,
            seed=42,
        )
        brain = QLIFLSTMBrain(config=config, num_actions=4, device=DeviceType.CPU)
        assert brain.input_dim == 4  # 2 food + 2 nociception features

        params = BrainParams(
            food_gradient_strength=0.7,
            food_gradient_direction=1.0,
            predator_gradient_strength=0.3,
            predator_gradient_direction=-0.5,
        )
        result = brain.run_brain(params, top_only=False, top_randomize=False)
        assert len(result) == 1

    def test_entropy_decay(self, brain: QLIFLSTMBrain):
        """Test entropy coefficient decays linearly."""
        # At episode 0
        coef_start = brain._get_entropy_coef()
        assert coef_start == brain.config.entropy_coef

        # At episode >= decay_episodes
        brain._episode_count = brain.config.entropy_decay_episodes
        coef_end = brain._get_entropy_coef()
        assert coef_end == brain.config.entropy_coef_end

        # At midpoint
        brain._episode_count = brain.config.entropy_decay_episodes // 2
        coef_mid = brain._get_entropy_coef()
        assert coef_start > coef_mid > coef_end

    def test_full_episode_loop(self, brain: QLIFLSTMBrain, params: BrainParams):
        """Test a complete episode loop: prepare -> run/learn -> post_process."""
        brain.prepare_episode()

        for step in range(10):
            brain.run_brain(params, top_only=False, top_randomize=False)
            done = step == 9
            brain.learn(params, reward=0.1, episode_done=done)

        brain.post_process_episode()

        assert brain._episode_count == 1
        assert len(brain.history_data.rewards) == 10
