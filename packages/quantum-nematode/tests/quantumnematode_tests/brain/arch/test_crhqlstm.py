"""Unit tests for the CRH-QLSTM brain architecture."""

import pytest
import torch
from quantumnematode.brain.actions import ActionData
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.arch.crhqlstm import CRHQLSTMBrain, CRHQLSTMBrainConfig
from quantumnematode.brain.arch.dtypes import BrainType, DeviceType
from quantumnematode.brain.modules import ModuleName
from quantumnematode.env import Direction

# ──────────────────────────────────────────────────────────────────────
# Config Tests
# ──────────────────────────────────────────────────────────────────────


class TestCRHQLSTMBrainConfig:
    """Test cases for CRH-QLSTM config."""

    def test_default_config(self):
        """Test default CRH-QLSTM configuration values."""
        config = CRHQLSTMBrainConfig()
        assert config.num_reservoir_neurons == 10
        assert config.reservoir_depth == 3
        assert config.spectral_radius == 0.9
        assert config.input_connectivity == "sparse"
        assert config.feature_channels == ["raw", "cos_sin", "pairwise"]
        assert config.input_encoding == "linear"
        assert config.lstm_hidden_dim == 64
        assert config.bptt_chunk_length == 32
        assert config.shots == 1024
        assert config.use_quantum_gates is True
        assert config.actor_lr == 0.0005
        assert config.critic_lr == 0.0005
        assert config.num_epochs == 6
        assert config.rollout_buffer_size == 1024
        assert config.critic_hidden_dim == 128
        assert config.critic_num_layers == 2

    def test_custom_config(self):
        """Test custom configuration values."""
        config = CRHQLSTMBrainConfig(
            num_reservoir_neurons=6,
            lstm_hidden_dim=32,
            use_quantum_gates=False,
            bptt_chunk_length=16,
            rollout_buffer_size=256,
            spectral_radius=0.95,
        )
        assert config.num_reservoir_neurons == 6
        assert config.lstm_hidden_dim == 32
        assert config.use_quantum_gates is False
        assert config.bptt_chunk_length == 16
        assert config.spectral_radius == 0.95

    def test_invalid_lstm_hidden_dim(self):
        """Test validation rejects too small hidden dim."""
        with pytest.raises(ValueError, match="lstm_hidden_dim must be >= 2"):
            CRHQLSTMBrainConfig(lstm_hidden_dim=1)

    def test_invalid_buffer_vs_chunk(self):
        """Test validation rejects buffer_size < chunk_length."""
        with pytest.raises(ValueError, match="rollout_buffer_size"):
            CRHQLSTMBrainConfig(rollout_buffer_size=8, bptt_chunk_length=16)


# ──────────────────────────────────────────────────────────────────────
# Brain Type Registration Tests
# ──────────────────────────────────────────────────────────────────────


class TestCRHQLSTMBrainTypeRegistration:
    """Test CRH-QLSTM brain type registration in dtypes."""

    def test_crh_qlstm_in_classical_types(self):
        """CRH-QLSTM should be in CLASSICAL_BRAIN_TYPES."""
        from quantumnematode.brain.arch.dtypes import CLASSICAL_BRAIN_TYPES

        assert BrainType.CRH_QLSTM in CLASSICAL_BRAIN_TYPES

    def test_brain_type_value(self):
        """Test brain type enum value."""
        assert BrainType.CRH_QLSTM.value == "crhqlstm"


# ──────────────────────────────────────────────────────────────────────
# CRH-QLSTM Brain Tests
# ──────────────────────────────────────────────────────────────────────


class TestCRHQLSTMBrain:
    """Test cases for CRH-QLSTM brain."""

    @pytest.fixture
    def config(self) -> CRHQLSTMBrainConfig:
        """Create a small test config."""
        return CRHQLSTMBrainConfig(
            num_reservoir_neurons=4,
            lstm_hidden_dim=8,
            use_quantum_gates=False,
            rollout_buffer_size=8,
            bptt_chunk_length=4,
            num_epochs=1,
            seed=42,
        )

    @pytest.fixture
    def brain(self, config: CRHQLSTMBrainConfig) -> CRHQLSTMBrain:
        """Create a test CRH-QLSTM brain."""
        return CRHQLSTMBrain(config=config, num_actions=4, device=DeviceType.CPU)

    @pytest.fixture
    def params(self) -> BrainParams:
        """Create test BrainParams."""
        return BrainParams(
            gradient_strength=0.6,
            gradient_direction=0.3,
            agent_direction=Direction.UP,
        )

    def test_initialization(self, brain: CRHQLSTMBrain):
        """Test CRH-QLSTM brain initializes correctly."""
        assert brain.num_actions == 4
        assert brain.lstm_cell is not None
        assert brain.actor_head is not None
        assert brain.critic is not None
        assert brain.feature_norm is not None
        assert brain.h_t.shape == (8,)
        assert brain.c_t.shape == (8,)
        # 4 neurons, channels [raw, cos_sin, pairwise]: 4 + 8 + 6 = 18
        assert brain.feature_dim == 18

    def test_hidden_state_initially_zero(self, brain: CRHQLSTMBrain):
        """Test LSTM hidden state is initialized to zeros."""
        assert torch.allclose(brain.h_t, torch.zeros(8))
        assert torch.allclose(brain.c_t, torch.zeros(8))

    def test_run_brain(self, brain: CRHQLSTMBrain, params: BrainParams):
        """Test CRH-QLSTM run_brain produces valid action data."""
        result = brain.run_brain(params, top_only=False, top_randomize=False)
        assert len(result) == 1
        assert isinstance(result[0], ActionData)
        assert result[0].probability > 0

    def test_run_brain_updates_hidden_state(self, brain: CRHQLSTMBrain, params: BrainParams):
        """Test run_brain updates LSTM hidden state."""
        h_before = brain.h_t.clone()
        brain.run_brain(params, top_only=False, top_randomize=False)
        assert not torch.allclose(brain.h_t, h_before)

    def test_hidden_state_reset(self, brain: CRHQLSTMBrain, params: BrainParams):
        """Test prepare_episode() zeros h_t/c_t."""
        brain.run_brain(params, top_only=False, top_randomize=False)
        brain.prepare_episode()
        assert torch.allclose(brain.h_t, torch.zeros(8))
        assert torch.allclose(brain.c_t, torch.zeros(8))
        assert brain._pending_features is None

    def test_learn_adds_to_buffer(self, brain: CRHQLSTMBrain, params: BrainParams):
        """Test learn() adds experience to buffer."""
        brain.run_brain(params, top_only=False, top_randomize=False)
        brain.learn(params, reward=1.0, episode_done=False)
        assert len(brain.buffer) == 1

    def test_learn_triggers_update_when_full(self, brain: CRHQLSTMBrain, params: BrainParams):
        """Test learn() triggers PPO update when buffer is full."""
        for _ in range(8):
            brain.run_brain(params, top_only=False, top_randomize=False)
            brain.learn(params, reward=0.1, episode_done=False)
        # Buffer should have been cleared after PPO update
        assert len(brain.buffer) == 0

    def test_post_process_episode(self, brain: CRHQLSTMBrain):
        """Test post_process_episode increments counter."""
        assert brain._episode_count == 0
        brain.post_process_episode()
        assert brain._episode_count == 1

    def test_copy(self, brain: CRHQLSTMBrain, params: BrainParams):
        """Test copy creates independent brain with fresh hidden state."""
        brain.run_brain(params, top_only=False, top_randomize=False)
        copy = brain.copy()
        assert torch.allclose(copy.h_t, torch.zeros(8))
        assert torch.allclose(copy.c_t, torch.zeros(8))
        assert not torch.allclose(brain.h_t, torch.zeros(8))

    def test_full_episode_loop(self, brain: CRHQLSTMBrain, params: BrainParams):
        """Test a complete episode loop for CRH-QLSTM."""
        brain.prepare_episode()
        for step in range(10):
            brain.run_brain(params, top_only=False, top_randomize=False)
            done = step == 9
            brain.learn(params, reward=0.1, episode_done=done)
        brain.post_process_episode()
        assert brain._episode_count == 1
        assert len(brain.history_data.rewards) == 10

    def test_classical_ablation(self, brain: CRHQLSTMBrain, params: BrainParams):
        """Test use_quantum_gates=False produces valid outputs."""
        assert brain.config.use_quantum_gates is False
        result = brain.run_brain(params, top_only=False, top_randomize=False)
        assert len(result) == 1
        assert result[0].probability > 0

    def test_sensory_module_mode(self):
        """Test brain works with unified sensory modules."""
        config = CRHQLSTMBrainConfig(
            num_reservoir_neurons=4,
            lstm_hidden_dim=8,
            use_quantum_gates=False,
            sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
            rollout_buffer_size=8,
            bptt_chunk_length=4,
            num_epochs=1,
            seed=42,
        )
        brain = CRHQLSTMBrain(config=config, num_actions=4, device=DeviceType.CPU)
        params = BrainParams(
            food_gradient_strength=0.7,
            food_gradient_direction=1.0,
            predator_gradient_strength=0.3,
            predator_gradient_direction=-0.5,
        )
        result = brain.run_brain(params, top_only=False, top_randomize=False)
        assert len(result) == 1
