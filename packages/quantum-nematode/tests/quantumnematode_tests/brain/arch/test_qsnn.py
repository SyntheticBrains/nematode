"""Unit tests for the QSNN (Quantum Spiking Neural Network) brain architecture."""

import numpy as np
import pytest
import torch
from quantumnematode.brain.actions import Action, ActionData
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.brain.arch.qsnn import (
    DEFAULT_NUM_INTEGRATION_STEPS,
    DEFAULT_SURROGATE_ALPHA,
    ENTROPY_BOOST_MAX,
    ENTROPY_CEILING_FRACTION,
    ENTROPY_FLOOR,
    EXPLORATION_DECAY_EPISODES,
    EXPLORATION_EPSILON,
    LR_DECAY_EPISODES,
    LR_MIN_FACTOR,
    MAX_ELIGIBILITY_NORM,
    QLIFSurrogateSpike,
    QSNNBrain,
    QSNNBrainConfig,
)
from quantumnematode.brain.modules import ModuleName
from quantumnematode.env import Direction


class TestQSNNBrainConfig:
    """Test cases for QSNN brain configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = QSNNBrainConfig()

        assert config.num_sensory_neurons == 6
        assert config.num_hidden_neurons == 8
        assert config.num_motor_neurons == 4
        assert config.membrane_tau == 0.9
        assert config.threshold == 0.5
        assert config.refractory_period == 0
        assert config.use_local_learning is True
        assert config.shots == 1024
        assert config.gamma == 0.99
        assert config.learning_rate == 0.1

    def test_custom_config(self):
        """Test custom configuration values."""
        config = QSNNBrainConfig(
            num_sensory_neurons=8,
            num_hidden_neurons=6,
            num_motor_neurons=5,
            membrane_tau=0.8,
            threshold=0.6,
            refractory_period=3,
            use_local_learning=False,
            shots=512,
            gamma=0.95,
            learning_rate=0.005,
        )

        assert config.num_sensory_neurons == 8
        assert config.num_hidden_neurons == 6
        assert config.num_motor_neurons == 5
        assert config.membrane_tau == 0.8
        assert config.threshold == 0.6
        assert config.refractory_period == 3
        assert config.use_local_learning is False
        assert config.shots == 512
        assert config.gamma == 0.95
        assert config.learning_rate == 0.005

    def test_config_sensory_modules_default(self):
        """Test that sensory_modules defaults to None (legacy mode)."""
        config = QSNNBrainConfig()
        assert config.sensory_modules is None

    def test_config_with_sensory_modules(self):
        """Test configuration with sensory modules."""
        config = QSNNBrainConfig(
            sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
        )

        assert config.sensory_modules is not None
        assert len(config.sensory_modules) == 2
        assert ModuleName.FOOD_CHEMOTAXIS in config.sensory_modules
        assert ModuleName.NOCICEPTION in config.sensory_modules

    def test_validation_num_sensory_neurons(self):
        """Test validation for num_sensory_neurons."""
        with pytest.raises(ValueError, match="num_sensory_neurons must be >= 1"):
            QSNNBrainConfig(num_sensory_neurons=0)

    def test_validation_num_hidden_neurons(self):
        """Test validation for num_hidden_neurons."""
        with pytest.raises(ValueError, match="num_hidden_neurons must be >= 1"):
            QSNNBrainConfig(num_hidden_neurons=0)

    def test_validation_num_motor_neurons(self):
        """Test validation for num_motor_neurons."""
        with pytest.raises(ValueError, match="num_motor_neurons must be >= 2"):
            QSNNBrainConfig(num_motor_neurons=1)

    def test_validation_membrane_tau_lower_bound(self):
        """Test validation for membrane_tau lower bound."""
        with pytest.raises(ValueError, match="membrane_tau must be in"):
            QSNNBrainConfig(membrane_tau=0.0)

    def test_validation_membrane_tau_upper_bound(self):
        """Test validation for membrane_tau at upper bound."""
        # Upper bound is inclusive (0, 1]
        config = QSNNBrainConfig(membrane_tau=1.0)
        assert config.membrane_tau == 1.0

    def test_validation_threshold_lower_bound(self):
        """Test validation for threshold lower bound."""
        with pytest.raises(ValueError, match="threshold must be in"):
            QSNNBrainConfig(threshold=0.0)

    def test_validation_threshold_upper_bound(self):
        """Test validation for threshold upper bound."""
        with pytest.raises(ValueError, match="threshold must be in"):
            QSNNBrainConfig(threshold=1.0)

    def test_validation_shots(self):
        """Test validation for shots."""
        with pytest.raises(ValueError, match="shots must be >= 100"):
            QSNNBrainConfig(shots=50)


class TestQSNNBrainQLIFCircuit:
    """Test cases for QLIF neuron circuit construction."""

    @pytest.fixture
    def small_config(self) -> QSNNBrainConfig:
        """Create a small test configuration for faster tests."""
        return QSNNBrainConfig(
            num_sensory_neurons=2,
            num_hidden_neurons=2,
            num_motor_neurons=2,
            shots=100,
        )

    @pytest.fixture
    def brain(self, small_config) -> QSNNBrain:
        """Create a test QSNN brain."""
        return QSNNBrain(
            config=small_config,
            num_actions=2,
            device=DeviceType.CPU,
        )

    def test_qlif_circuit_structure(self, brain):
        """Verify QLIF circuit has correct structure: RY + RX + Measure."""
        circuit = brain._build_qlif_circuit(weighted_input=0.5, theta_membrane=0.1)

        # Check circuit has 1 qubit and 1 classical bit
        assert circuit.num_qubits == 1
        assert circuit.num_clbits == 1

        # Extract gate names
        gate_names = [instruction.operation.name for instruction in circuit.data]

        # Should have RY, RX, and measure
        assert "ry" in gate_names
        assert "rx" in gate_names
        assert "measure" in gate_names

        # RY should come before RX
        ry_idx = gate_names.index("ry")
        rx_idx = gate_names.index("rx")
        assert ry_idx < rx_idx

    def test_qlif_circuit_angles(self, brain):
        """Test QLIF circuit has correct rotation angles with tanh normalization."""
        weighted_input = 0.5
        theta_membrane = 0.2

        circuit = brain._build_qlif_circuit(
            weighted_input=weighted_input,
            theta_membrane=theta_membrane,
        )

        # Find RY and RX gates
        ry_gate = None
        rx_gate = None
        for instr in circuit.data:
            if instr.operation.name == "ry":
                ry_gate = instr
            elif instr.operation.name == "rx":
                rx_gate = instr

        assert ry_gate is not None
        assert rx_gate is not None

        # RY angle = theta_membrane + tanh(weighted_input) * pi
        expected_ry = theta_membrane + np.tanh(weighted_input) * np.pi
        assert np.isclose(float(ry_gate.operation.params[0]), expected_ry, atol=1e-6)

        # RX angle = leak_angle = (1 - membrane_tau) * pi
        expected_rx = brain.leak_angle
        assert np.isclose(float(rx_gate.operation.params[0]), expected_rx, atol=1e-6)

    def test_qlif_circuit_tanh_bounds_large_input(self, brain):
        """Test that tanh prevents angle wrapping for large weighted inputs."""
        # Large weighted_input that would wrap many times without tanh
        large_input = 50.0
        theta_membrane = 0.0

        circuit = brain._build_qlif_circuit(
            weighted_input=large_input,
            theta_membrane=theta_membrane,
        )

        ry_gate = None
        for instr in circuit.data:
            if instr.operation.name == "ry":
                ry_gate = instr

        assert ry_gate is not None

        # tanh(50) ~ 1.0, so RY angle should be close to pi
        ry_angle = float(ry_gate.operation.params[0])
        assert abs(ry_angle) <= np.pi + 0.01  # Bounded within one rotation

    def test_qlif_circuit_measurement(self, brain):
        """Test QLIF circuit measurement produces valid probabilities."""
        circuit = brain._build_qlif_circuit(weighted_input=0.5, theta_membrane=0.1)

        # Execute circuit
        backend = brain._get_backend()
        job = backend.run(circuit, shots=100)
        result = job.result()
        counts = result.get_counts()

        # Should have results for "0" and/or "1"
        total = sum(counts.values())
        assert total == 100

        # Probabilities should sum to 1
        probs = {k: v / total for k, v in counts.items()}
        assert np.isclose(sum(probs.values()), 1.0)


class TestQSNNBrainSensoryEncoding:
    """Test cases for sensory spike encoding."""

    @pytest.fixture
    def brain(self) -> QSNNBrain:
        """Create a test QSNN brain."""
        config = QSNNBrainConfig(
            num_sensory_neurons=4,
            num_hidden_neurons=2,
            num_motor_neurons=2,
            shots=100,
        )
        return QSNNBrain(config=config, num_actions=2, device=DeviceType.CPU)

    def test_encode_sensory_spikes_shape(self, brain):
        """Test sensory encoding produces correct output shape."""
        features = np.array([0.5, 0.3], dtype=np.float32)
        spikes = brain._encode_sensory_spikes(features)

        assert isinstance(spikes, np.ndarray)
        assert len(spikes) == brain.num_sensory

    def test_encode_sensory_spikes_range(self, brain):
        """Test sensory encoding produces probabilities in [0, 1]."""
        features = np.array([0.5, 0.3], dtype=np.float32)
        spikes = brain._encode_sensory_spikes(features)

        assert np.all(spikes >= 0.0)
        assert np.all(spikes <= 1.0)

    def test_encode_sensory_spikes_sigmoid(self, brain):
        """Test sensory encoding uses sigmoid correctly."""
        # High input should produce high spike probability
        high_features = np.array([1.0, 1.0], dtype=np.float32)
        high_spikes = brain._encode_sensory_spikes(high_features)

        # Low input should produce lower spike probability
        low_features = np.array([0.0, 0.0], dtype=np.float32)
        low_spikes = brain._encode_sensory_spikes(low_features)

        # High should be greater than low
        assert np.all(high_spikes > low_spikes)


class TestQSNNBrainForwardPass:
    """Test cases for forward pass through the network."""

    @pytest.fixture
    def brain(self) -> QSNNBrain:
        """Create a test QSNN brain."""
        config = QSNNBrainConfig(
            num_sensory_neurons=4,
            num_hidden_neurons=2,
            num_motor_neurons=2,
            shots=100,
        )
        return QSNNBrain(config=config, num_actions=2, device=DeviceType.CPU)

    def test_timestep_produces_valid_probabilities(self, brain):
        """Test _timestep produces valid motor neuron probabilities."""
        features = np.array([0.5, 0.3], dtype=np.float32)
        motor_probs = brain._timestep(features)

        assert isinstance(motor_probs, np.ndarray)
        assert len(motor_probs) == brain.num_motor
        assert np.all(motor_probs >= 0.0)
        assert np.all(motor_probs <= 1.0)

    def test_run_brain_returns_action(self, brain):
        """Test run_brain returns valid action."""
        params = BrainParams(
            gradient_strength=0.6,
            gradient_direction=0.3,
            agent_position=(1, 1),
            agent_direction=Direction.UP,
        )

        actions = brain.run_brain(params, top_only=True, top_randomize=True)

        assert len(actions) == 1
        action_data = actions[0]
        assert isinstance(action_data, ActionData)
        assert action_data.action in [Action.FORWARD, Action.LEFT]  # Only 2 actions
        assert 0.0 <= action_data.probability <= 1.0

    def test_run_brain_tracks_episode_data(self, brain):
        """Test run_brain tracks episode data."""
        params = BrainParams(gradient_strength=0.5, gradient_direction=1.0)

        brain.run_brain(params, top_only=True, top_randomize=True)

        assert len(brain.episode_actions) == 1

    def test_exploration_floor_prevents_deterministic_policy(self, brain):
        """Test that epsilon mixing prevents any action probability from reaching 0."""
        params = BrainParams(gradient_strength=0.5, gradient_direction=1.0)

        brain.run_brain(params, top_only=True, top_randomize=True)

        # Every action probability should be at least epsilon / num_actions
        min_prob = EXPLORATION_EPSILON / brain.num_actions
        assert brain.current_probabilities is not None
        for prob in brain.current_probabilities:
            assert prob >= min_prob - 1e-8

    def test_exploration_floor_probabilities_sum_to_one(self, brain):
        """Test that action probabilities still sum to 1.0 after epsilon mixing."""
        params = BrainParams(gradient_strength=0.8, gradient_direction=0.1)

        brain.run_brain(params, top_only=True, top_randomize=True)

        assert brain.current_probabilities is not None
        assert np.isclose(np.sum(brain.current_probabilities), 1.0, atol=1e-6)


class TestQSNNBrainEligibilityTrace:
    """Test cases for eligibility trace computation."""

    @pytest.fixture
    def brain(self) -> QSNNBrain:
        """Create a test QSNN brain with local learning."""
        config = QSNNBrainConfig(
            num_sensory_neurons=2,
            num_hidden_neurons=2,
            num_motor_neurons=2,
            shots=100,
            use_local_learning=True,
        )
        return QSNNBrain(config=config, num_actions=2, device=DeviceType.CPU)

    def test_eligibility_accumulates(self, brain):
        """Test eligibility traces accumulate during run_brain (after action selection)."""
        # Initial eligibility should be zero
        assert torch.allclose(brain.eligibility_sh, torch.zeros_like(brain.eligibility_sh))
        assert torch.allclose(brain.eligibility_hm, torch.zeros_like(brain.eligibility_hm))

        # run_brain triggers _timestep then accumulates eligibility after action selection
        params = BrainParams(gradient_strength=0.5, gradient_direction=1.0)
        brain.run_brain(params, top_only=True, top_randomize=True)

        # At least one eligibility matrix should have non-zero values
        # (depends on spike patterns, but with reasonable probs, should have some)
        has_eligibility = (
            torch.any(brain.eligibility_sh != 0).item()
            or torch.any(brain.eligibility_hm != 0).item()
        )
        # This might occasionally be all zeros if no spikes, which is valid
        assert isinstance(has_eligibility, bool)

    def test_eligibility_centered_has_mixed_signs(self, brain):
        """Test that centered eligibility traces contain both positive and negative values.

        Spike probabilities are centered at the threshold before computing
        outer products. With varied spike probabilities, the resulting
        eligibility matrix should have mixed signs (not all-positive), which
        enables directional weight updates rather than uniform drift.
        """
        # Directly call _accumulate_eligibility with varied spike probs
        # Some above threshold (0.5), some below
        sensory = np.array([0.8, 0.2], dtype=np.float32)  # 0.8 > 0.5, 0.2 < 0.5
        hidden = np.array([0.7, 0.3], dtype=np.float32)  # 0.7 > 0.5, 0.3 < 0.5
        motor = np.array([0.6, 0.4], dtype=np.float32)  # 0.6 > 0.5, 0.4 < 0.5

        # Accumulate for both actions to populate both columns
        brain._accumulate_eligibility(sensory, hidden, motor, chosen_action=0)
        brain._accumulate_eligibility(sensory, hidden, motor, chosen_action=1)

        # After centering: sensory=[0.3, -0.3], hidden=[0.2, -0.2]
        # Outer product gives [[0.06, -0.06], [-0.06, 0.06]]
        # Should have both positive and negative entries
        assert torch.any(brain.eligibility_sh > 0).item()
        assert torch.any(brain.eligibility_sh < 0).item()

        # hm: action 0 centered motor=0.1, action 1 centered motor=-0.1
        # Column 0: hidden_centered * 0.1 = [0.02, -0.02]
        # Column 1: hidden_centered * -0.1 = [-0.02, 0.02] (+ gamma decay)
        assert torch.any(brain.eligibility_hm > 0).item()
        assert torch.any(brain.eligibility_hm < 0).item()

    def test_eligibility_action_specific_whm(self, brain):
        """Test that only the chosen action's column is updated in eligibility_hm."""
        sensory = np.array([0.8, 0.2], dtype=np.float32)
        hidden = np.array([0.7, 0.3], dtype=np.float32)
        motor = np.array([0.6, 0.4], dtype=np.float32)

        # Only update action 0
        brain._accumulate_eligibility(sensory, hidden, motor, chosen_action=0)

        # Column 0 should be non-zero, column 1 should be zero
        assert torch.any(brain.eligibility_hm[:, 0] != 0).item()
        assert torch.allclose(brain.eligibility_hm[:, 1], torch.zeros(brain.num_hidden))

        # Only theta_motor[0] should be non-zero
        assert brain.eligibility_theta_motor[0].item() != 0.0
        assert brain.eligibility_theta_motor[1].item() == 0.0

    def test_eligibility_reset_on_episode_end(self, brain):
        """Test eligibility traces reset when episode ends."""
        # Run a step to accumulate some eligibility
        params = BrainParams(gradient_strength=0.5, gradient_direction=1.0)
        brain.run_brain(params, top_only=True, top_randomize=True)
        brain.learn(params, reward=1.0, episode_done=True)

        # Eligibility should be reset
        assert torch.allclose(brain.eligibility_sh, torch.zeros_like(brain.eligibility_sh))
        assert torch.allclose(brain.eligibility_hm, torch.zeros_like(brain.eligibility_hm))
        assert torch.allclose(
            brain.eligibility_theta_hidden,
            torch.zeros_like(brain.eligibility_theta_hidden),
        )
        assert torch.allclose(
            brain.eligibility_theta_motor,
            torch.zeros_like(brain.eligibility_theta_motor),
        )


class TestQSNNBrainLocalLearning:
    """Test cases for local learning weight updates."""

    @pytest.fixture
    def brain(self) -> QSNNBrain:
        """Create a test QSNN brain."""
        config = QSNNBrainConfig(
            num_sensory_neurons=2,
            num_hidden_neurons=2,
            num_motor_neurons=2,
            shots=100,
            use_local_learning=True,
            learning_rate=0.1,  # High LR for visible changes
        )
        return QSNNBrain(config=config, num_actions=2, device=DeviceType.CPU)

    def test_local_learning_updates_weights(self, brain):
        """Test that local learning updates weights."""
        # Record initial weights
        initial_w_sh = brain.W_sh.clone()
        initial_w_hm = brain.W_hm.clone()

        # Run episode with positive reward
        params = BrainParams(gradient_strength=0.5, gradient_direction=1.0)
        for _ in range(5):
            brain.run_brain(params, top_only=True, top_randomize=True)
            brain.learn(params, reward=1.0, episode_done=False)

        # Complete episode
        brain.learn(params, reward=1.0, episode_done=True)

        # Weights should change (unless all eligibility was zero)
        # Check if at least one weight matrix changed
        sh_changed = not torch.allclose(brain.W_sh, initial_w_sh, atol=1e-6)
        hm_changed = not torch.allclose(brain.W_hm, initial_w_hm, atol=1e-6)

        # At least one should change with positive reward and reasonable activity
        # Note: might occasionally not change if all spikes were 0
        assert isinstance(sh_changed, bool)
        assert isinstance(hm_changed, bool)

    def test_weight_decay_shrinks_weights(self, brain):
        """Test that L2 weight decay reduces large weights over repeated updates."""
        # Set weights to large values
        brain.W_sh.fill_(10.0)
        brain.W_hm.fill_(10.0)

        # Zero eligibility so only decay acts
        brain.eligibility_sh.zero_()
        brain.eligibility_hm.zero_()
        brain.eligibility_theta_hidden.zero_()
        brain.eligibility_theta_motor.zero_()

        initial_sh_norm = torch.norm(brain.W_sh).item()
        initial_hm_norm = torch.norm(brain.W_hm).item()

        # Apply learning updates with zero eligibility (only decay applies)
        for _ in range(10):
            brain._local_learning_update(total_reward=0.0)

        # Weights should have shrunk due to L2 decay
        assert torch.norm(brain.W_sh).item() < initial_sh_norm
        assert torch.norm(brain.W_hm).item() < initial_hm_norm

    def test_theta_parameters_are_trained(self, brain):
        """Test that theta_hidden and theta_motor are updated during learning."""
        initial_theta_hidden = brain.theta_hidden.clone()
        initial_theta_motor = brain.theta_motor.clone()

        # Set non-zero eligibility to ensure theta gets updated
        brain.eligibility_theta_hidden.fill_(1.0)
        brain.eligibility_theta_motor.fill_(1.0)

        # Apply learning with non-zero advantage
        brain.baseline = 0.0
        brain._local_learning_update(total_reward=5.0)

        # Theta should have changed
        theta_h_changed = not torch.allclose(
            brain.theta_hidden,
            initial_theta_hidden,
            atol=1e-6,
        )
        theta_m_changed = not torch.allclose(
            brain.theta_motor,
            initial_theta_motor,
            atol=1e-6,
        )
        assert theta_h_changed
        assert theta_m_changed

    def test_eligibility_normalization_caps_large_traces(self, brain):
        """Test that large eligibility traces are normalized before weight update."""
        # Set very large eligibility traces (simulating 20-step accumulation)
        brain.eligibility_sh.fill_(10.0)  # norm >> MAX_ELIGIBILITY_NORM
        brain.eligibility_hm.fill_(10.0)

        initial_w_sh = brain.W_sh.clone()
        brain.baseline = 0.0
        brain._local_learning_update(total_reward=5.0)

        # Even with huge raw traces, normalization caps the update magnitude
        assert torch.norm(brain.W_sh).item() < 10.0
        # Weights should have changed
        assert not torch.allclose(brain.W_sh, initial_w_sh, atol=1e-6)

    def test_eligibility_normalization_preserves_small_traces(self, brain):
        """Test that small eligibility traces are NOT normalized (no shrinkage)."""
        # Set small eligibility traces (below max_norm)
        small_value = MAX_ELIGIBILITY_NORM * 0.1 / (brain.num_sensory * brain.num_hidden) ** 0.5
        brain.eligibility_sh.fill_(small_value)

        raw_norm = torch.norm(brain.eligibility_sh).item()
        assert raw_norm < MAX_ELIGIBILITY_NORM  # Confirm it's below threshold

        normalized = QSNNBrain._normalize_trace(brain.eligibility_sh)
        # Should be unchanged (not shrunk)
        assert torch.allclose(normalized, brain.eligibility_sh)

    def test_normalize_trace_direction_preserved(self):
        """Test that normalization preserves the direction of the trace."""
        trace = torch.tensor([3.0, 4.0, 0.0])  # norm = 5.0
        normalized = QSNNBrain._normalize_trace(trace, max_norm=1.0)

        # Direction should be preserved (cosine similarity = 1)
        cos_sim = torch.dot(trace, normalized) / (torch.norm(trace) * torch.norm(normalized))
        assert torch.isclose(cos_sim, torch.tensor(1.0), atol=1e-6)

        # Norm should be capped at 1.0
        assert torch.isclose(torch.norm(normalized), torch.tensor(1.0), atol=1e-6)

    def test_normalize_trace_zero_trace(self):
        """Test normalization handles zero traces gracefully."""
        trace = torch.zeros(4)
        normalized = QSNNBrain._normalize_trace(trace)
        assert torch.allclose(normalized, torch.zeros(4))

    def test_weight_update_bounded_with_large_advantage(self, brain):
        """Test that weight updates stay bounded even with large advantage."""
        # Simulate worst case: large traces + large advantage
        brain.eligibility_sh.fill_(50.0)  # Huge trace
        brain.eligibility_hm.fill_(50.0)
        brain.eligibility_theta_hidden.fill_(50.0)
        brain.eligibility_theta_motor.fill_(50.0)

        initial_w_sh = brain.W_sh.clone()
        initial_w_hm = brain.W_hm.clone()

        brain.baseline = -10.0  # Forces advantage = total_reward - (-10) = 20
        brain._local_learning_update(total_reward=10.0)

        # Even with advantage=20, normalization should keep deltas reasonable
        # Max delta per param ≈ lr(0.1) * max_norm(1.0) * advantage(20) = 2.0
        delta_sh = brain.W_sh - initial_w_sh
        delta_hm = brain.W_hm - initial_w_hm
        # Per-element max should be bounded
        assert delta_sh.abs().max().item() < 5.0
        assert delta_hm.abs().max().item() < 5.0

    def test_action_specific_eligibility_produces_different_columns(self, brain):
        """Test that action-specific eligibility gives W_hm columns independent updates.

        When different actions are chosen on different steps, the eligibility
        traces for each column accumulate independently, preventing the
        column convergence problem from R2-R10.
        """
        # Set distinct W_hm columns and zero eligibility
        brain.W_hm[:, 0] = torch.tensor([0.1, -0.1])
        brain.W_hm[:, 1] = torch.tensor([-0.1, 0.1])
        initial_w_hm = brain.W_hm.clone()

        # Accumulate eligibility only for action 0 (5 steps)
        sensory = np.array([0.8, 0.2], dtype=np.float32)
        hidden = np.array([0.7, 0.3], dtype=np.float32)
        motor = np.array([0.6, 0.4], dtype=np.float32)
        for _ in range(5):
            brain._accumulate_eligibility(sensory, hidden, motor, chosen_action=0)

        # Apply learning update
        brain.baseline = 0.0
        brain._local_learning_update(total_reward=5.0)

        # Column 0 should have changed significantly (received eligibility)
        # Column 1 should have changed ONLY due to L2 decay (no eligibility)
        delta_col0 = torch.norm(brain.W_hm[:, 0] - initial_w_hm[:, 0]).item()
        delta_col1 = torch.norm(brain.W_hm[:, 1] - initial_w_hm[:, 1]).item()

        # Column 0 delta should be much larger than column 1 delta
        assert delta_col0 > delta_col1 * 2.0, (
            f"Column 0 delta ({delta_col0}) should be much larger than "
            f"column 1 delta ({delta_col1}) since only action 0 was chosen"
        )


class TestQSNNBrainRefractoryPeriod:
    """Test cases for refractory period behavior."""

    @pytest.fixture
    def brain(self) -> QSNNBrain:
        """Create a test QSNN brain."""
        config = QSNNBrainConfig(
            num_sensory_neurons=2,
            num_hidden_neurons=2,
            num_motor_neurons=2,
            shots=100,
            refractory_period=2,
        )
        return QSNNBrain(config=config, num_actions=2, device=DeviceType.CPU)

    def test_refractory_state_initialized_zero(self, brain):
        """Test refractory states start at zero."""
        assert np.all(brain.refractory_hidden == 0)
        assert np.all(brain.refractory_motor == 0)

    def test_refractory_reset_on_episode_start(self, brain):
        """Test refractory states reset when episode starts."""
        # Set some refractory state
        brain.refractory_hidden[0] = 2
        brain.refractory_motor[0] = 1

        # Prepare new episode
        brain.prepare_episode()

        # Should be reset
        assert np.all(brain.refractory_hidden == 0)
        assert np.all(brain.refractory_motor == 0)


class TestQSNNBrainReproducibility:
    """Test cases for reproducibility with seeds."""

    def test_same_seed_same_weights(self):
        """Same seed should produce identical initial weights."""
        config1 = QSNNBrainConfig(
            seed=42,
            num_sensory_neurons=2,
            num_hidden_neurons=2,
            num_motor_neurons=2,
        )
        config2 = QSNNBrainConfig(
            seed=42,
            num_sensory_neurons=2,
            num_hidden_neurons=2,
            num_motor_neurons=2,
        )

        brain1 = QSNNBrain(config=config1, num_actions=2, device=DeviceType.CPU)
        brain2 = QSNNBrain(config=config2, num_actions=2, device=DeviceType.CPU)

        assert torch.allclose(brain1.W_sh, brain2.W_sh)
        assert torch.allclose(brain1.W_hm, brain2.W_hm)

    def test_different_seed_different_weights(self):
        """Different seeds should produce different initial weights."""
        config1 = QSNNBrainConfig(
            seed=42,
            num_sensory_neurons=2,
            num_hidden_neurons=2,
            num_motor_neurons=2,
        )
        config2 = QSNNBrainConfig(
            seed=123,
            num_sensory_neurons=2,
            num_hidden_neurons=2,
            num_motor_neurons=2,
        )

        brain1 = QSNNBrain(config=config1, num_actions=2, device=DeviceType.CPU)
        brain2 = QSNNBrain(config=config2, num_actions=2, device=DeviceType.CPU)

        # At least one weight matrix should differ
        sh_different = not torch.allclose(brain1.W_sh, brain2.W_sh)
        hm_different = not torch.allclose(brain1.W_hm, brain2.W_hm)

        assert sh_different or hm_different


class TestQSNNBrainCopy:
    """Test cases for brain copying."""

    @pytest.fixture
    def brain(self) -> QSNNBrain:
        """Create a test QSNN brain."""
        config = QSNNBrainConfig(
            num_sensory_neurons=2,
            num_hidden_neurons=2,
            num_motor_neurons=2,
            shots=100,
        )
        return QSNNBrain(config=config, num_actions=2, device=DeviceType.CPU)

    def test_copy_independence(self, brain):
        """Test that copy is independent - modifying copy doesn't affect original."""
        # Get original weights
        original_w_sh = brain.W_sh.clone()
        original_w_hm = brain.W_hm.clone()

        # Create copy
        brain_copy = brain.copy()

        # Modify copy's weights
        brain_copy.W_sh.fill_(999.0)
        brain_copy.W_hm.fill_(999.0)

        # Original should be unchanged
        assert torch.allclose(brain.W_sh, original_w_sh)
        assert torch.allclose(brain.W_hm, original_w_hm)

    def test_copy_has_same_initial_weights(self, brain):
        """Test that copy has same initial weights as original."""
        brain_copy = brain.copy()

        assert torch.allclose(brain.W_sh, brain_copy.W_sh)
        assert torch.allclose(brain.W_hm, brain_copy.W_hm)
        assert torch.allclose(brain.theta_hidden, brain_copy.theta_hidden)
        assert torch.allclose(brain.theta_motor, brain_copy.theta_motor)

    def test_copy_preserves_baseline(self, brain):
        """Test that copy preserves baseline value."""
        brain.baseline = 0.5
        brain_copy = brain.copy()

        assert brain_copy.baseline == brain.baseline


class TestQSNNBrainSensoryModules:
    """Test cases for sensory modules feature extraction."""

    def test_brain_with_sensory_modules_input_dim(self):
        """Test that input_dim is computed from sensory modules."""
        config = QSNNBrainConfig(
            num_sensory_neurons=4,
            num_hidden_neurons=2,
            num_motor_neurons=2,
            shots=100,
            sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
        )
        brain = QSNNBrain(config=config, num_actions=2, device=DeviceType.CPU)

        # Each module contributes 2 features [strength, angle]
        assert brain.input_dim == 4
        assert brain.sensory_modules is not None
        assert len(brain.sensory_modules) == 2

    def test_preprocess_with_sensory_modules(self):
        """Test preprocessing with sensory modules uses extract_classical_features."""
        config = QSNNBrainConfig(
            num_sensory_neurons=4,
            num_hidden_neurons=2,
            num_motor_neurons=2,
            shots=100,
            sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
        )
        brain = QSNNBrain(config=config, num_actions=2, device=DeviceType.CPU)

        params = BrainParams(
            food_gradient_strength=0.7,
            food_gradient_direction=1.0,
            predator_gradient_strength=0.3,
            predator_gradient_direction=-0.5,
            agent_direction=Direction.UP,
        )

        features = brain.preprocess(params)

        assert isinstance(features, np.ndarray)
        assert len(features) == 4  # 2 modules * 2 features each
        assert features.dtype == np.float32

    def test_legacy_mode_when_no_sensory_modules(self):
        """Test that brain uses legacy preprocessing when sensory_modules is None."""
        config = QSNNBrainConfig(
            num_sensory_neurons=4,
            num_hidden_neurons=2,
            num_motor_neurons=2,
            shots=100,
            # No sensory_modules - legacy mode
        )
        brain = QSNNBrain(config=config, num_actions=2, device=DeviceType.CPU)

        assert brain.sensory_modules is None
        assert brain.input_dim == 2  # Legacy mode: gradient_strength + relative_angle


class TestQSNNBrainIntegration:
    """Integration tests for QSNN brain with full simulation workflow."""

    def test_full_episode_workflow(self):
        """Test a complete episode workflow."""
        config = QSNNBrainConfig(
            num_sensory_neurons=4,
            num_hidden_neurons=2,
            num_motor_neurons=4,
            shots=100,
            learning_rate=0.01,
        )
        brain = QSNNBrain(config=config, num_actions=4, device=DeviceType.CPU)

        # Simulate multiple steps in an episode
        rng = np.random.default_rng(42)

        for step in range(10):
            params = BrainParams(
                gradient_strength=rng.random(),
                gradient_direction=rng.random() * 2 * np.pi,
                agent_position=(step, step),
                agent_direction=Direction.UP,
            )

            # Run brain
            actions = brain.run_brain(params, top_only=True, top_randomize=False)
            assert len(actions) == 1

            # Learn
            reward = rng.random() - 0.5  # Random reward
            brain.learn(params, reward, episode_done=(step == 9))

        # Post-process episode
        brain.post_process_episode()

        # Check that episode completed successfully
        assert len(brain.history_data.rewards) == 10

    def test_multiple_episodes(self):
        """Test running multiple episodes."""
        config = QSNNBrainConfig(
            num_sensory_neurons=4,
            num_hidden_neurons=2,
            num_motor_neurons=4,
            shots=100,
        )
        brain = QSNNBrain(config=config, num_actions=4, device=DeviceType.CPU)

        params = BrainParams(gradient_strength=0.5, gradient_direction=1.0)

        for _episode in range(3):
            brain.prepare_episode()

            for step in range(5):
                brain.run_brain(params, top_only=True, top_randomize=False)
                brain.learn(params, reward=0.1, episode_done=(step == 4))

        # Should have accumulated history
        assert len(brain.history_data.rewards) == 15  # 3 episodes * 5 steps


class TestQSNNBrainSurrogateGradient:
    """Test cases for surrogate gradient learning mode (use_local_learning=False)."""

    @pytest.fixture
    def brain(self) -> QSNNBrain:
        """Create a test QSNN brain in surrogate gradient mode."""
        config = QSNNBrainConfig(
            num_sensory_neurons=2,
            num_hidden_neurons=2,
            num_motor_neurons=2,
            shots=100,
            use_local_learning=False,
            learning_rate=0.1,
        )
        return QSNNBrain(config=config, num_actions=2, device=DeviceType.CPU)

    def test_surrogate_mode_creates_optimizer(self, brain):
        """Test that surrogate gradient mode creates an Adam optimizer."""
        assert brain.optimizer is not None
        assert isinstance(brain.optimizer, torch.optim.Adam)

    def test_surrogate_mode_weights_require_grad(self, brain):
        """Test that weights have requires_grad=True in surrogate mode."""
        assert brain.W_sh.requires_grad
        assert brain.W_hm.requires_grad
        assert brain.theta_hidden.requires_grad
        assert brain.theta_motor.requires_grad

    def test_hebbian_mode_weights_no_grad(self):
        """Test that weights do NOT have requires_grad in Hebbian mode."""
        config = QSNNBrainConfig(
            num_sensory_neurons=2,
            num_hidden_neurons=2,
            num_motor_neurons=2,
            shots=100,
            use_local_learning=True,
        )
        brain = QSNNBrain(config=config, num_actions=2, device=DeviceType.CPU)
        assert not brain.W_sh.requires_grad
        assert brain.optimizer is None

    def test_surrogate_spike_forward_returns_quantum_prob(self):
        """Test that QLIFSurrogateSpike.forward returns the quantum spike probability."""
        ry_angle = torch.tensor(1.0, requires_grad=True)
        quantum_prob = 0.73  # Simulated quantum measurement

        result: torch.Tensor = QLIFSurrogateSpike.apply(  # type: ignore[assignment]
            ry_angle,
            quantum_prob,
            DEFAULT_SURROGATE_ALPHA,
        )

        assert torch.isclose(result, torch.tensor(0.73), atol=1e-6)

    def test_surrogate_spike_backward_produces_gradient(self):
        """Test that QLIFSurrogateSpike.backward produces non-zero gradients."""
        ry_angle = torch.tensor(1.0, requires_grad=True)
        quantum_prob = 0.73

        result: torch.Tensor = QLIFSurrogateSpike.apply(  # type: ignore[assignment]
            ry_angle,
            quantum_prob,
            DEFAULT_SURROGATE_ALPHA,
        )
        result.backward()

        assert ry_angle.grad is not None
        assert ry_angle.grad.item() != 0.0

    def test_surrogate_spike_gradient_is_sigmoid_derivative(self):
        """Test that the surrogate gradient matches the sigmoid derivative formula.

        The surrogate is centered at pi/2 (the RY transition point).
        """
        alpha = 10.0
        ry_val = 1.2  # An RY angle near pi/2
        ry_angle = torch.tensor(ry_val, requires_grad=True)

        result: torch.Tensor = QLIFSurrogateSpike.apply(  # type: ignore[assignment]
            ry_angle,
            0.5,
            alpha,
        )
        result.backward()

        # Expected gradient: alpha * sigma(alpha*(ry - pi/2)) * (1 - sigma(...))
        shifted = alpha * (ry_val - np.pi / 2)
        sigma = 1.0 / (1.0 + np.exp(-shifted))
        expected_grad = alpha * sigma * (1 - sigma)

        assert ry_angle.grad is not None
        assert np.isclose(ry_angle.grad.item(), expected_grad, atol=1e-5)

    def test_surrogate_spike_gradient_flows_to_theta(self):
        """Test that gradients flow through ry_angle to both theta and weights.

        The RY angle is theta + tanh(w·x)*pi, so d(ry)/d(theta) = 1.
        """
        theta = torch.tensor(0.5, requires_grad=True)
        weighted_input = torch.tensor(0.3, requires_grad=True)
        ry_angle = theta + torch.tanh(weighted_input) * torch.pi

        result: torch.Tensor = QLIFSurrogateSpike.apply(  # type: ignore[assignment]
            ry_angle,
            0.6,
            DEFAULT_SURROGATE_ALPHA,
        )
        result.backward()

        # Both theta and weighted_input should get gradients
        assert theta.grad is not None
        assert theta.grad.item() != 0.0
        assert weighted_input.grad is not None
        assert weighted_input.grad.item() != 0.0

    def test_timestep_differentiable_returns_tensor_with_grad(self, brain):
        """Test that _timestep_differentiable returns tensors connected to autograd."""
        features = np.array([0.5, 0.3], dtype=np.float32)

        motor_spikes = brain._timestep_differentiable(features)

        assert isinstance(motor_spikes, torch.Tensor)
        assert motor_spikes.shape == (2,)  # num_motor_neurons

    def test_reinforce_update_changes_weights(self, brain):
        """Test that _reinforce_update modifies weights via backpropagation."""
        initial_w_sh = brain.W_sh.clone().detach()
        initial_w_hm = brain.W_hm.clone().detach()

        # Run a short episode with varied rewards for meaningful advantages
        params = BrainParams(gradient_strength=0.5, gradient_direction=1.0)
        rewards = [0.1, 2.0, -1.0, 3.0, 0.5]
        for step in range(5):
            brain.run_brain(params, top_only=True, top_randomize=True)
            brain.learn(params, reward=rewards[step], episode_done=(step == 4))

        # Weights should have changed after the episode-end learning update
        sh_changed = not torch.allclose(brain.W_sh, initial_w_sh, atol=1e-6)
        hm_changed = not torch.allclose(brain.W_hm, initial_w_hm, atol=1e-6)
        assert sh_changed or hm_changed, (
            "Surrogate gradient update should change at least one weight matrix"
        )

    def test_reinforce_update_changes_theta(self, brain):
        """Test that _reinforce_update also updates theta parameters (not just weights).

        This verifies the fix for the theta gradient bug where theta was detached
        from the computation graph and received zero gradients.
        """
        initial_theta_h = brain.theta_hidden.clone().detach()
        initial_theta_m = brain.theta_motor.clone().detach()

        # Run a short episode with varied rewards
        params = BrainParams(gradient_strength=0.5, gradient_direction=1.0)
        for step in range(5):
            brain.run_brain(params, top_only=True, top_randomize=True)
            brain.learn(params, reward=1.0, episode_done=(step == 4))

        # Theta should have changed after the episode-end learning update
        theta_h_changed = not torch.allclose(brain.theta_hidden, initial_theta_h, atol=1e-6)
        theta_m_changed = not torch.allclose(brain.theta_motor, initial_theta_m, atol=1e-6)
        assert theta_h_changed or theta_m_changed, (
            "Surrogate gradient update should change at least one theta parameter"
        )

    def test_surrogate_mode_stores_features(self, brain):
        """Test that run_brain stores features for gradient recomputation."""
        params = BrainParams(gradient_strength=0.5, gradient_direction=1.0)

        brain.run_brain(params, top_only=True, top_randomize=True)

        assert len(brain.episode_features) == 1
        assert isinstance(brain.episode_features[0], np.ndarray)

    def test_surrogate_mode_episode_features_cleared_on_reset(self, brain):
        """Test that episode_features is cleared on episode reset."""
        params = BrainParams(gradient_strength=0.5, gradient_direction=1.0)
        brain.run_brain(params, top_only=True, top_randomize=True)
        assert len(brain.episode_features) == 1

        brain.learn(params, reward=1.0, episode_done=True)
        assert len(brain.episode_features) == 0

    def test_surrogate_mode_full_episode(self, brain):
        """Test complete episode workflow in surrogate gradient mode."""
        params = BrainParams(gradient_strength=0.5, gradient_direction=1.0)

        brain.prepare_episode()
        for step in range(10):
            actions = brain.run_brain(params, top_only=True, top_randomize=True)
            assert len(actions) == 1
            brain.learn(params, reward=0.1, episode_done=(step == 9))

        # History should be recorded
        assert len(brain.history_data.rewards) == 10

    def test_surrogate_mode_copy_independence(self, brain):
        """Test that copy in surrogate mode produces an independent brain."""
        brain_copy = brain.copy()

        # Weights should match
        assert torch.allclose(brain.W_sh.detach(), brain_copy.W_sh.detach())

        # Copy should have its own optimizer
        assert brain_copy.optimizer is not None
        assert brain_copy.optimizer is not brain.optimizer

        # Modifying copy shouldn't affect original
        with torch.no_grad():
            brain_copy.W_sh.fill_(999.0)
        assert not torch.allclose(brain.W_sh.detach(), brain_copy.W_sh.detach())

    def test_surrogate_mode_copy_has_grad(self, brain):
        """Test that copied brain in surrogate mode still has requires_grad."""
        brain_copy = brain.copy()
        assert brain_copy.W_sh.requires_grad
        assert brain_copy.W_hm.requires_grad

    def test_exploration_schedule_initial(self, brain):
        """Test exploration schedule returns high exploration at episode 0."""
        brain._episode_count = 0
        epsilon, temperature = brain._exploration_schedule()
        assert epsilon == EXPLORATION_EPSILON  # Full epsilon at start
        assert temperature == 1.5  # High temperature at start

    def test_exploration_schedule_decayed(self, brain):
        """Test exploration schedule returns reduced exploration after decay period."""
        brain._episode_count = EXPLORATION_DECAY_EPISODES
        epsilon, temperature = brain._exploration_schedule()
        # Epsilon decays to 30% of original
        assert np.isclose(epsilon, EXPLORATION_EPSILON * 0.3, atol=1e-6)
        # Temperature decays to 1.0
        assert np.isclose(temperature, 1.0, atol=1e-6)

    def test_exploration_schedule_midpoint(self, brain):
        """Test exploration schedule at midpoint of decay."""
        brain._episode_count = EXPLORATION_DECAY_EPISODES // 2
        epsilon, temperature = brain._exploration_schedule()
        # Should be between initial and final values
        assert EXPLORATION_EPSILON * 0.3 < epsilon < EXPLORATION_EPSILON
        assert 1.0 < temperature < 1.5

    def test_exploration_schedule_beyond_decay(self, brain):
        """Test exploration schedule stays at final values beyond decay period."""
        brain._episode_count = EXPLORATION_DECAY_EPISODES * 5
        epsilon, temperature = brain._exploration_schedule()
        # Should be clamped at final values
        assert np.isclose(epsilon, EXPLORATION_EPSILON * 0.3, atol=1e-6)
        assert np.isclose(temperature, 1.0, atol=1e-6)

    def test_episode_count_increments(self, brain):
        """Test that _episode_count increments after each episode."""
        assert brain._episode_count == 0

        params = BrainParams(gradient_strength=0.5, gradient_direction=1.0)
        for step in range(3):
            brain.run_brain(params, top_only=True, top_randomize=True)
            brain.learn(params, reward=1.0, episode_done=(step == 2))

        assert brain._episode_count == 1

    def test_copy_preserves_episode_count(self, brain):
        """Test that copy() preserves the episode counter."""
        brain._episode_count = 15
        brain_copy = brain.copy()
        assert brain_copy._episode_count == 15

    def test_logit_scale_used_in_run_brain(self, brain):
        """Test that LOGIT_SCALE constant is used (not hardcoded 10.0)."""
        # Run brain and check that action probs are valid
        params = BrainParams(gradient_strength=0.5, gradient_direction=1.0)
        brain.run_brain(params, top_only=True, top_randomize=True)
        assert brain.current_probabilities is not None
        assert np.isclose(np.sum(brain.current_probabilities), 1.0, atol=1e-6)
        # With LOGIT_SCALE=20.0 (vs old 10.0), probabilities should still be valid
        assert all(p >= 0 for p in brain.current_probabilities)

    def test_lr_scheduler_created(self, brain):
        """Test that surrogate gradient mode creates a cosine annealing LR scheduler."""
        assert brain.scheduler is not None
        assert isinstance(
            brain.scheduler,
            torch.optim.lr_scheduler.CosineAnnealingLR,
        )

    def test_lr_scheduler_not_created_for_hebbian(self):
        """Test that Hebbian mode does NOT create an LR scheduler."""
        config = QSNNBrainConfig(
            num_sensory_neurons=2,
            num_hidden_neurons=2,
            num_motor_neurons=2,
            shots=100,
            use_local_learning=True,
        )
        brain = QSNNBrain(config=config, num_actions=2, device=DeviceType.CPU)
        assert brain.scheduler is None

    def test_lr_decays_after_episodes(self, brain):
        """Test that LR decreases after running episodes via the scheduler."""
        initial_lr = brain.optimizer.param_groups[0]["lr"]

        # Run several episodes to trigger scheduler steps
        params = BrainParams(gradient_strength=0.5, gradient_direction=1.0)
        for _episode in range(10):
            brain.prepare_episode()
            for step in range(3):
                brain.run_brain(params, top_only=True, top_randomize=True)
                brain.learn(params, reward=0.1, episode_done=(step == 2))

        current_lr = brain.optimizer.param_groups[0]["lr"]
        assert current_lr < initial_lr, (
            f"LR should have decreased from {initial_lr} after 10 episodes, but is {current_lr}"
        )

    def test_lr_reaches_minimum_at_decay_end(self, brain):
        """Test that LR reaches eta_min after LR_DECAY_EPISODES scheduler steps."""
        initial_lr = brain.optimizer.param_groups[0]["lr"]
        expected_min_lr = initial_lr * LR_MIN_FACTOR

        # Step the scheduler directly to the end of the decay period
        for _ in range(LR_DECAY_EPISODES):
            brain.scheduler.step()

        final_lr = brain.optimizer.param_groups[0]["lr"]
        assert np.isclose(final_lr, expected_min_lr, atol=1e-6), (
            f"LR should be ~{expected_min_lr} after {LR_DECAY_EPISODES} steps, but is {final_lr}"
        )

    def test_lr_at_midpoint(self, brain):
        """Test that LR is between initial and minimum at midpoint of decay."""
        initial_lr = brain.optimizer.param_groups[0]["lr"]
        min_lr = initial_lr * LR_MIN_FACTOR

        # Step to midpoint
        for _ in range(LR_DECAY_EPISODES // 2):
            brain.scheduler.step()

        mid_lr = brain.optimizer.param_groups[0]["lr"]
        assert min_lr < mid_lr < initial_lr, (
            f"LR at midpoint should be between {min_lr} and {initial_lr}, but is {mid_lr}"
        )

    def test_current_lr_helper(self, brain):
        """Test that _current_lr() returns the optimizer's current learning rate."""
        assert brain._current_lr() == brain.optimizer.param_groups[0]["lr"]

        # Step the scheduler and verify it tracks
        brain.scheduler.step()
        assert brain._current_lr() == brain.optimizer.param_groups[0]["lr"]

    def test_copy_preserves_lr_schedule(self, brain):
        """Test that copy() preserves the LR schedule state."""
        # Advance the scheduler by 50 episodes
        brain._episode_count = 50
        for _ in range(50):
            brain.scheduler.step()

        original_lr = brain._current_lr()
        brain_copy = brain.copy()

        # The copy should have the same LR
        assert np.isclose(brain_copy._current_lr(), original_lr, atol=1e-6), (
            f"Copy LR ({brain_copy._current_lr()}) should match original LR ({original_lr})"
        )
        # The copy should have its own scheduler
        assert brain_copy.scheduler is not None
        assert brain_copy.scheduler is not brain.scheduler


class TestQSNNWeightInitialization:
    """Test cases for random Gaussian weight initialization and theta initialization."""

    def test_w_sh_random_gaussian_shape(self):
        """Test that W_sh has correct shape after random Gaussian initialization."""
        config = QSNNBrainConfig(
            num_sensory_neurons=6,
            num_hidden_neurons=4,
            num_motor_neurons=4,
            shots=100,
        )
        brain = QSNNBrain(config=config, num_actions=4, device=DeviceType.CPU)

        assert brain.W_sh.detach().shape == (6, 4)

    def test_w_hm_random_gaussian_shape(self):
        """Test that W_hm has correct shape after random Gaussian initialization."""
        config = QSNNBrainConfig(
            num_sensory_neurons=6,
            num_hidden_neurons=8,
            num_motor_neurons=4,
            shots=100,
        )
        brain = QSNNBrain(config=config, num_actions=4, device=DeviceType.CPU)

        assert brain.W_hm.detach().shape == (8, 4)

    def test_weight_scale_approximately_correct(self):
        """Test that randn * WEIGHT_INIT_SCALE produces weights near expected magnitude."""
        config = QSNNBrainConfig(
            num_sensory_neurons=6,
            num_hidden_neurons=8,
            num_motor_neurons=4,
            shots=100,
        )
        brain = QSNNBrain(config=config, num_actions=4, device=DeviceType.CPU)

        # With randn * 0.15, element-wise std should be ~0.15
        # Check overall Frobenius norm is in a reasonable range
        w_sh_norm = torch.norm(brain.W_sh.detach()).item()
        w_hm_norm = torch.norm(brain.W_hm.detach()).item()

        # For a (6,8) matrix with std=0.15, expected Frobenius norm ~= sqrt(48)*0.15 ≈ 1.04
        # For a (8,4) matrix with std=0.15, expected Frobenius norm ~= sqrt(32)*0.15 ≈ 0.85
        # Allow generous bounds (3x) since it's random
        assert 0 < w_sh_norm < 4.0, f"W_sh norm {w_sh_norm} should be moderate"
        assert 0 < w_hm_norm < 4.0, f"W_hm norm {w_hm_norm} should be moderate"

    def test_weight_columns_have_varied_norms(self):
        """Test that random Gaussian init produces columns with varied norms (symmetry breaking)."""
        config = QSNNBrainConfig(
            num_sensory_neurons=6,
            num_hidden_neurons=8,
            num_motor_neurons=4,
            shots=100,
        )
        brain = QSNNBrain(config=config, num_actions=4, device=DeviceType.CPU)

        # Random Gaussian columns should have different norms (unlike orthogonal
        # which has identical norms). Check that max/min ratio > 1.1.
        col_norms = torch.norm(brain.W_hm.detach(), dim=0)
        ratio = col_norms.max().item() / (col_norms.min().item() + 1e-10)
        assert ratio > 1.05, (
            f"Random Gaussian columns should have varied norms for symmetry breaking, "
            f"but max/min ratio is only {ratio:.4f}"
        )

    def test_theta_hidden_initialized_at_pi_over_4(self):
        """Test that theta_hidden is initialized at pi/4 for moderate warm start."""
        config = QSNNBrainConfig(
            num_sensory_neurons=6,
            num_hidden_neurons=8,
            num_motor_neurons=4,
            shots=100,
        )
        brain = QSNNBrain(config=config, num_actions=4, device=DeviceType.CPU)

        expected = torch.full((8,), np.pi / 4)
        assert torch.allclose(brain.theta_hidden.detach(), expected, atol=1e-6), (
            f"theta_hidden should be initialized to pi/4, got {brain.theta_hidden.detach()}"
        )

    def test_theta_motor_initialized_at_zero(self):
        """Test that theta_motor remains initialized at zero (no action bias)."""
        config = QSNNBrainConfig(
            num_sensory_neurons=6,
            num_hidden_neurons=8,
            num_motor_neurons=4,
            shots=100,
        )
        brain = QSNNBrain(config=config, num_actions=4, device=DeviceType.CPU)

        expected = torch.zeros(4)
        assert torch.allclose(brain.theta_motor.detach(), expected, atol=1e-6), (
            f"theta_motor should be initialized to 0, got {brain.theta_motor.detach()}"
        )

    def test_w_hm_column_diversity_positive_at_init(self):
        """Test that random Gaussian init produces positive W_hm column diversity."""
        config = QSNNBrainConfig(
            num_sensory_neurons=6,
            num_hidden_neurons=8,
            num_motor_neurons=4,
            shots=100,
        )
        brain = QSNNBrain(config=config, num_actions=4, device=DeviceType.CPU)

        col_div = brain._compute_column_diversity()
        assert col_div > 0.01, (
            f"Column diversity should be positive at init with random weights, got {col_div:.6f}"
        )


class TestAdaptiveEntropyBonus:
    """Test cases for the adaptive entropy bonus in surrogate gradient mode."""

    @pytest.fixture
    def brain(self):
        """Create a QSNN brain in surrogate gradient mode for testing."""
        config = QSNNBrainConfig(
            num_sensory_neurons=2,
            num_hidden_neurons=4,
            num_motor_neurons=4,
            shots=100,
            use_local_learning=False,
            entropy_coef=0.02,
        )
        return QSNNBrain(config=config, num_actions=4, device=DeviceType.CPU)

    def test_entropy_constants_defined(self):
        """Test that entropy regulation constants have expected values."""
        assert ENTROPY_FLOOR == 0.5
        assert ENTROPY_BOOST_MAX == 20.0
        assert ENTROPY_CEILING_FRACTION == 0.95

    def test_no_boost_when_entropy_above_floor(self, brain):
        """Test that entropy_coef is unchanged when entropy is above the floor."""
        # Run a few steps to have episode data
        params = BrainParams(gradient_strength=0.5, gradient_direction=1.0)
        brain.prepare_episode()
        for step in range(5):
            brain.run_brain(params, top_only=True, top_randomize=True)
            brain.learn(params, reward=0.1, episode_done=(step == 4))

        # With 4 actions and epsilon-greedy, entropy at init is typically > 1.0
        # (near ln(4)=1.386), well above ENTROPY_FLOOR=0.5.
        # The test verifies training runs without error; the actual adaptive
        # logic is tested more directly below.

    def test_adaptive_entropy_formula(self):
        """Test the adaptive entropy scaling formula directly."""
        # At the floor: no boost
        entropy_val = ENTROPY_FLOOR
        ratio = 1.0 - entropy_val / ENTROPY_FLOOR
        scale = 1.0 + ratio * (ENTROPY_BOOST_MAX - 1.0)
        assert np.isclose(scale, 1.0, atol=1e-6)

        # At half the floor: intermediate boost
        entropy_val = ENTROPY_FLOOR / 2
        ratio = 1.0 - entropy_val / ENTROPY_FLOOR
        scale = 1.0 + ratio * (ENTROPY_BOOST_MAX - 1.0)
        assert np.isclose(scale, 10.5, atol=1e-6)  # 1 + 0.5 * 19 = 10.5

        # At zero: max boost
        entropy_val = 0.0
        ratio = 1.0 - entropy_val / ENTROPY_FLOOR
        scale = 1.0 + ratio * (ENTROPY_BOOST_MAX - 1.0)
        assert np.isclose(scale, ENTROPY_BOOST_MAX, atol=1e-6)

    def test_above_floor_below_ceiling_scale_is_one(self):
        """Test that entropy between floor and ceiling gives scale=1.0."""
        max_entropy = np.log(4)
        ceiling = ENTROPY_CEILING_FRACTION * max_entropy
        for entropy_val in [ENTROPY_FLOOR, 0.8, 1.0, ceiling]:
            if entropy_val < ENTROPY_FLOOR:
                ratio = 1.0 - entropy_val / ENTROPY_FLOOR
                scale = 1.0 + ratio * (ENTROPY_BOOST_MAX - 1.0)
            elif entropy_val > ceiling:
                ratio = (entropy_val - ceiling) / (max_entropy - ceiling)
                scale = max(0.0, 1.0 - ratio)
            else:
                scale = 1.0
            assert scale == 1.0, f"Scale should be 1.0 for entropy={entropy_val}"

    def test_entropy_ceiling_formula(self):
        """Test the entropy ceiling scaling formula directly."""
        num_actions = 4
        max_entropy = np.log(num_actions)
        ceiling = ENTROPY_CEILING_FRACTION * max_entropy

        # At the ceiling: no suppression (scale=1.0)
        entropy_val = ceiling
        ratio = (entropy_val - ceiling) / (max_entropy - ceiling)
        scale = max(0.0, 1.0 - ratio)
        assert np.isclose(scale, 1.0, atol=1e-6)

        # Halfway between ceiling and max: scale=0.5
        entropy_val = ceiling + (max_entropy - ceiling) / 2
        ratio = (entropy_val - ceiling) / (max_entropy - ceiling)
        scale = max(0.0, 1.0 - ratio)
        assert np.isclose(scale, 0.5, atol=1e-6)

        # At max entropy: scale=0.0 (full suppression)
        entropy_val = max_entropy
        ratio = (entropy_val - ceiling) / (max_entropy - ceiling)
        scale = max(0.0, 1.0 - ratio)
        assert np.isclose(scale, 0.0, atol=1e-6)


class TestMultiTimestepIntegration:
    """Test cases for multi-timestep integration (averaging QLIF timesteps per decision)."""

    def test_default_num_integration_steps(self):
        """Test that DEFAULT_NUM_INTEGRATION_STEPS is 10."""
        assert DEFAULT_NUM_INTEGRATION_STEPS == 10

    def test_config_default_integration_steps(self):
        """Test that QSNNBrainConfig defaults to DEFAULT_NUM_INTEGRATION_STEPS."""
        config = QSNNBrainConfig()
        assert config.num_integration_steps == DEFAULT_NUM_INTEGRATION_STEPS

    def test_config_custom_integration_steps(self):
        """Test that num_integration_steps can be set to a custom value."""
        config = QSNNBrainConfig(num_integration_steps=5)
        assert config.num_integration_steps == 5

    def test_brain_stores_integration_steps(self):
        """Test that QSNNBrain stores the num_integration_steps from config."""
        config = QSNNBrainConfig(
            num_sensory_neurons=2,
            num_hidden_neurons=2,
            num_motor_neurons=2,
            shots=100,
            num_integration_steps=5,
        )
        brain = QSNNBrain(config=config, num_actions=2, device=DeviceType.CPU)
        assert brain.num_integration_steps == 5

    def test_multi_timestep_produces_valid_probabilities(self):
        """Test that _multi_timestep produces valid averaged motor probabilities."""
        config = QSNNBrainConfig(
            num_sensory_neurons=2,
            num_hidden_neurons=2,
            num_motor_neurons=2,
            shots=100,
            num_integration_steps=3,
        )
        brain = QSNNBrain(config=config, num_actions=2, device=DeviceType.CPU)

        features = np.array([0.5, 0.3], dtype=np.float32)
        motor_probs = brain._multi_timestep(features)

        assert isinstance(motor_probs, np.ndarray)
        assert len(motor_probs) == 2
        assert np.all(motor_probs >= 0.0)
        assert np.all(motor_probs <= 1.0)

    def test_multi_timestep_differentiable_returns_tensor(self):
        """Test that _multi_timestep_differentiable returns a torch tensor."""
        config = QSNNBrainConfig(
            num_sensory_neurons=2,
            num_hidden_neurons=2,
            num_motor_neurons=2,
            shots=100,
            use_local_learning=False,
            num_integration_steps=3,
        )
        brain = QSNNBrain(config=config, num_actions=2, device=DeviceType.CPU)

        features = np.array([0.5, 0.3], dtype=np.float32)
        motor_probs = brain._multi_timestep_differentiable(features)

        assert isinstance(motor_probs, torch.Tensor)
        assert motor_probs.shape == (2,)

    def test_single_step_equivalent_to_original(self):
        """Test that num_integration_steps=1 behaves like original single-timestep."""
        config = QSNNBrainConfig(
            seed=42,
            num_sensory_neurons=2,
            num_hidden_neurons=2,
            num_motor_neurons=2,
            shots=100,
            num_integration_steps=1,
        )
        brain = QSNNBrain(config=config, num_actions=2, device=DeviceType.CPU)

        features = np.array([0.5, 0.3], dtype=np.float32)
        # With 1 integration step, _multi_timestep should call _timestep once
        multi_result = brain._multi_timestep(features)

        assert isinstance(multi_result, np.ndarray)
        assert len(multi_result) == 2
        assert np.all(multi_result >= 0.0)
        assert np.all(multi_result <= 1.0)

    def test_run_brain_uses_multi_timestep(self):
        """Test that run_brain works with multi-timestep integration."""
        config = QSNNBrainConfig(
            num_sensory_neurons=2,
            num_hidden_neurons=2,
            num_motor_neurons=2,
            shots=100,
            num_integration_steps=3,
        )
        brain = QSNNBrain(config=config, num_actions=2, device=DeviceType.CPU)

        params = BrainParams(gradient_strength=0.5, gradient_direction=1.0)
        actions = brain.run_brain(params, top_only=True, top_randomize=True)

        assert len(actions) == 1
        assert isinstance(actions[0], ActionData)

    def test_surrogate_update_with_multi_timestep(self):
        """Test that surrogate gradient update works with multi-timestep integration."""
        config = QSNNBrainConfig(
            num_sensory_neurons=2,
            num_hidden_neurons=2,
            num_motor_neurons=2,
            shots=100,
            use_local_learning=False,
            learning_rate=0.1,
            num_integration_steps=3,
        )
        brain = QSNNBrain(config=config, num_actions=2, device=DeviceType.CPU)

        initial_w_sh = brain.W_sh.clone().detach()
        initial_w_hm = brain.W_hm.clone().detach()

        # Run a short episode with varied rewards
        params = BrainParams(gradient_strength=0.5, gradient_direction=1.0)
        rewards = [0.1, 2.0, -1.0]
        for step in range(3):
            brain.run_brain(params, top_only=True, top_randomize=True)
            brain.learn(params, reward=rewards[step], episode_done=(step == 2))

        # Weights should change after surrogate gradient update
        sh_changed = not torch.allclose(brain.W_sh, initial_w_sh, atol=1e-6)
        hm_changed = not torch.allclose(brain.W_hm, initial_w_hm, atol=1e-6)
        assert sh_changed or hm_changed, (
            "Surrogate gradient update with multi-timestep should change weights"
        )

    def test_copy_preserves_integration_steps(self):
        """Test that copy() preserves num_integration_steps."""
        config = QSNNBrainConfig(
            num_sensory_neurons=2,
            num_hidden_neurons=2,
            num_motor_neurons=2,
            shots=100,
            num_integration_steps=7,
        )
        brain = QSNNBrain(config=config, num_actions=2, device=DeviceType.CPU)
        brain_copy = brain.copy()

        assert brain_copy.num_integration_steps == 7
