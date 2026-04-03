"""Tests for Short-Term Associative Memory (STAM) buffer."""

from __future__ import annotations

import numpy as np
import pytest
from quantumnematode.agent.stam import STAMBuffer


class TestSTAMBufferBasics:
    """Basic buffer operations."""

    def test_initial_state(self) -> None:
        """Test that a new buffer is empty with correct memory dimension."""
        buf = STAMBuffer(buffer_size=10, decay_rate=0.1)
        assert len(buf) == 0
        assert buf.memory_dimension == 11

    def test_record_single_entry(self) -> None:
        """Test that recording a single entry increments the buffer length."""
        buf = STAMBuffer(buffer_size=10, decay_rate=0.1)
        buf.record(np.array([0.5, 0.3, 0.1, 0.0]), (1.0, 0.0), 0)
        assert len(buf) == 1

    def test_record_multiple_entries(self) -> None:
        """Test that recording multiple entries tracks buffer length correctly."""
        buf = STAMBuffer(buffer_size=10, decay_rate=0.1)
        for i in range(5):
            buf.record(np.array([float(i), 0.0, 0.0, 0.0]), (1.0, 0.0), 0)
        assert len(buf) == 5

    def test_buffer_size_limit(self) -> None:
        """Test that the buffer does not exceed its configured size limit."""
        buf = STAMBuffer(buffer_size=5, decay_rate=0.1)
        for i in range(10):
            buf.record(np.array([float(i), 0.0, 0.0, 0.0]), (1.0, 0.0), 0)
        assert len(buf) == 5

    def test_reset_clears_buffer(self) -> None:
        """Test that reset empties the buffer and zeroes the memory state."""
        buf = STAMBuffer(buffer_size=10, decay_rate=0.1)
        for i in range(5):
            buf.record(np.array([float(i), 0.0, 0.0, 0.0]), (1.0, 0.0), 0)
        buf.reset()
        assert len(buf) == 0
        state = buf.get_memory_state()
        np.testing.assert_array_equal(state, np.zeros(11, dtype=np.float32))


class TestDecayWeights:
    """Exponential decay weight computation."""

    def test_weights_decrease_with_age(self) -> None:
        """Test that decay weights decrease monotonically with age."""
        buf = STAMBuffer(buffer_size=10, decay_rate=0.1)
        for i in range(1, len(buf._weights)):
            assert buf._weights[i] < buf._weights[i - 1]

    def test_most_recent_weight_is_one(self) -> None:
        """Test that the most recent entry has a weight of 1.0."""
        buf = STAMBuffer(buffer_size=10, decay_rate=0.1)
        assert buf._weights[0] == pytest.approx(1.0)

    def test_weight_formula(self) -> None:
        """Test that weights follow the exponential decay formula."""
        decay_rate = 0.2
        buf = STAMBuffer(buffer_size=5, decay_rate=decay_rate)
        for i in range(5):
            expected = np.exp(-decay_rate * i)
            assert buf._weights[i] == pytest.approx(expected)

    def test_weighted_mean_emphasizes_recent(self) -> None:
        """Test that the weighted mean is biased toward recent entries."""
        buf = STAMBuffer(buffer_size=10, decay_rate=0.5)
        # Record: old value 0.0, then recent value 1.0
        buf.record(np.array([0.0, 0.0, 0.0, 0.0]), (0.0, 0.0), 0)
        buf.record(np.array([1.0, 0.0, 0.0, 0.0]), (0.0, 0.0), 0)
        state = buf.get_memory_state()
        # Weighted mean of food channel should be closer to 1.0 than 0.5
        assert state[0] > 0.5


class TestTemporalDerivative:
    """Temporal derivative computation via weighted finite difference."""

    def test_insufficient_history_returns_zero(self) -> None:
        """Test that temporal derivative returns zero with insufficient history."""
        buf = STAMBuffer(buffer_size=10, decay_rate=0.1)
        assert buf.compute_temporal_derivative(0) == 0.0

        buf.record(np.array([0.5, 0.0, 0.0, 0.0]), (0.0, 0.0), 0)
        assert buf.compute_temporal_derivative(0) == 0.0

    def test_positive_derivative_increasing_concentration(self) -> None:
        """Test that increasing concentration produces a positive derivative."""
        buf = STAMBuffer(buffer_size=10, decay_rate=0.1)
        # Older: 0.2, newer: 0.8 → concentration increased → positive derivative
        buf.record(np.array([0.2, 0.0, 0.0, 0.0]), (0.0, 0.0), 0)
        buf.record(np.array([0.8, 0.0, 0.0, 0.0]), (0.0, 0.0), 0)
        deriv = buf.compute_temporal_derivative(0)
        assert deriv > 0.0

    def test_negative_derivative_decreasing_concentration(self) -> None:
        """Test that decreasing concentration produces a negative derivative."""
        buf = STAMBuffer(buffer_size=10, decay_rate=0.1)
        # Older: 0.8, newer: 0.2 → concentration decreased → negative derivative
        buf.record(np.array([0.8, 0.0, 0.0, 0.0]), (0.0, 0.0), 0)
        buf.record(np.array([0.2, 0.0, 0.0, 0.0]), (0.0, 0.0), 0)
        deriv = buf.compute_temporal_derivative(0)
        assert deriv < 0.0

    def test_zero_derivative_stationary(self) -> None:
        """Test that stationary concentration produces a zero derivative."""
        buf = STAMBuffer(buffer_size=10, decay_rate=0.1)
        for _ in range(5):
            buf.record(np.array([0.5, 0.0, 0.0, 0.0]), (0.0, 0.0), 0)
        deriv = buf.compute_temporal_derivative(0)
        assert deriv == pytest.approx(0.0)

    def test_derivative_per_channel(self) -> None:
        """Test that derivatives are computed independently per channel."""
        buf = STAMBuffer(buffer_size=10, decay_rate=0.1)
        # Food increasing, temp decreasing, predator constant
        buf.record(np.array([0.2, 0.8, 0.5, 0.0]), (0.0, 0.0), 0)
        buf.record(np.array([0.8, 0.2, 0.5, 0.0]), (0.0, 0.0), 0)
        assert buf.compute_temporal_derivative(0) > 0.0  # food increasing
        assert buf.compute_temporal_derivative(1) < 0.0  # temp decreasing
        assert buf.compute_temporal_derivative(2) == pytest.approx(0.0)  # predator flat

    def test_derivative_exact_value_two_entries(self) -> None:
        """Test that the derivative matches the expected exact value with two entries."""
        buf = STAMBuffer(buffer_size=10, decay_rate=0.1)
        buf.record(np.array([0.3, 0.0, 0.0, 0.0]), (0.0, 0.0), 0)
        buf.record(np.array([0.7, 0.0, 0.0, 0.0]), (0.0, 0.0), 0)
        # With 2 entries, only one diff: C[0] - C[1] = 0.7 - 0.3 = 0.4
        # weight[0] = 1.0, sum(weights) = 1.0
        deriv = buf.compute_temporal_derivative(0)
        assert deriv == pytest.approx(0.4)


class TestMemoryState:
    """Fixed-size memory state vector output."""

    def test_empty_buffer_returns_zeros(self) -> None:
        """Test that an empty buffer returns a zero-filled memory state."""
        buf = STAMBuffer(buffer_size=10, decay_rate=0.1)
        state = buf.get_memory_state()
        assert state.shape == (11,)
        assert state.dtype == np.float32
        np.testing.assert_array_equal(state, np.zeros(11, dtype=np.float32))

    def test_shape_always_11(self) -> None:
        """Test that memory state shape is always 11 regardless of buffer contents."""
        buf = STAMBuffer(buffer_size=10, decay_rate=0.1)
        for i in range(7):
            buf.record(np.array([float(i), 0.0, 0.0, 0.0]), (1.0, 0.0), i % 4)
            state = buf.get_memory_state()
            assert state.shape == (11,)

    def test_weighted_means_in_first_four(self) -> None:
        """Test that the first four state elements contain weighted means of channels."""
        buf = STAMBuffer(buffer_size=10, decay_rate=0.0)  # No decay = equal weights
        buf.record(np.array([0.2, 0.4, 0.6, 0.8]), (0.0, 0.0), 0)
        buf.record(np.array([0.8, 0.6, 0.4, 0.2]), (0.0, 0.0), 0)
        state = buf.get_memory_state()
        # With decay_rate=0, equal weights → simple mean
        assert state[0] == pytest.approx(0.5)  # food mean
        assert state[1] == pytest.approx(0.5)  # temp mean
        assert state[2] == pytest.approx(0.5)  # predator mean
        assert state[3] == pytest.approx(0.5)  # oxygen mean

    def test_derivatives_in_indices_4_to_7(self) -> None:
        """Test that state indices 4-7 contain per-channel temporal derivatives."""
        buf = STAMBuffer(buffer_size=10, decay_rate=0.1)
        buf.record(np.array([0.2, 0.8, 0.5, 0.0]), (0.0, 0.0), 0)
        buf.record(np.array([0.8, 0.2, 0.5, 0.0]), (0.0, 0.0), 0)
        state = buf.get_memory_state()
        assert state[4] > 0.0  # food derivative positive
        assert state[5] < 0.0  # temp derivative negative
        assert state[6] == pytest.approx(0.0)  # predator derivative zero
        assert state[7] == pytest.approx(0.0)  # oxygen derivative zero

    def test_partially_filled_buffer(self) -> None:
        """Test that a partially filled buffer produces correct weighted means."""
        buf = STAMBuffer(buffer_size=30, decay_rate=0.1)
        buf.record(np.array([0.5, 0.3, 0.1, 0.2]), (1.0, 0.0), 0)
        state = buf.get_memory_state()
        assert state.shape == (11,)
        # With one entry, weighted mean == that entry's values
        assert state[0] == pytest.approx(0.5)
        assert state[1] == pytest.approx(0.3)
        assert state[2] == pytest.approx(0.1)
        assert state[3] == pytest.approx(0.2)


class TestPositionDeltas:
    """Position delta computation uses step-to-step differences."""

    def test_single_entry_zero_deviation(self) -> None:
        """Test that a single entry produces zero position deviation."""
        buf = STAMBuffer(buffer_size=10, decay_rate=0.1)
        buf.record(np.array([0.0, 0.0, 0.0, 0.0]), (1.0, 0.0), 0)
        state = buf.get_memory_state()
        # Only one entry → deviation from mean is 0
        assert state[8] == pytest.approx(0.0)
        assert state[9] == pytest.approx(0.0)

    def test_consistent_movement_zero_deviation(self) -> None:
        """Test that consistent movement produces zero deviation from the mean."""
        buf = STAMBuffer(buffer_size=10, decay_rate=0.0)  # Equal weights
        for _ in range(5):
            buf.record(np.array([0.0, 0.0, 0.0, 0.0]), (1.0, 0.0), 0)
        state = buf.get_memory_state()
        # All movements identical → latest matches mean → zero deviation
        assert state[8] == pytest.approx(0.0)
        assert state[9] == pytest.approx(0.0)

    def test_direction_change_nonzero_deviation(self) -> None:
        """Test that a sudden direction change produces nonzero deviation."""
        buf = STAMBuffer(buffer_size=10, decay_rate=0.0)  # Equal weights
        # Consistent rightward movement
        for _ in range(4):
            buf.record(np.array([0.0, 0.0, 0.0, 0.0]), (1.0, 0.0), 0)
        # Then sudden upward movement
        buf.record(np.array([0.0, 0.0, 0.0, 0.0]), (0.0, 1.0), 0)
        state = buf.get_memory_state()
        # Latest (0, 1) deviates from mean which is biased toward (1, 0)
        assert state[8] < 0.0  # moved less right than average
        assert state[9] > 0.0  # moved more up than average

    def test_uses_deltas_not_absolute_positions(self) -> None:
        """Verify we store step-to-step movement, not absolute coords."""
        buf = STAMBuffer(buffer_size=10, decay_rate=0.1)
        # These are (dx, dy) per step, not positions
        buf.record(np.array([0.0, 0.0, 0.0, 0.0]), (1.0, 0.0), 0)  # Moved right
        buf.record(np.array([0.0, 0.0, 0.0, 0.0]), (-1.0, 0.0), 0)  # Moved left
        state = buf.get_memory_state()
        # The position deltas should reflect movement changes,
        # not grow with absolute position
        assert abs(state[8]) <= 2.0
        assert abs(state[9]) <= 2.0


class TestActionEntropy:
    """Action variety metric computation."""

    def test_single_action_zero_entropy(self) -> None:
        """Test that repeating a single action produces zero entropy."""
        buf = STAMBuffer(buffer_size=10, decay_rate=0.1)
        for _ in range(5):
            buf.record(np.array([0.0, 0.0, 0.0, 0.0]), (0.0, 0.0), 0)
        state = buf.get_memory_state()
        assert state[10] == pytest.approx(0.0)

    def test_diverse_actions_positive_entropy(self) -> None:
        """Test that diverse actions produce positive entropy."""
        buf = STAMBuffer(buffer_size=10, decay_rate=0.1)
        for i in range(8):
            buf.record(np.array([0.0, 0.0, 0.0, 0.0]), (0.0, 0.0), i % 4)
        state = buf.get_memory_state()
        assert state[10] > 0.0

    def test_maximum_diversity(self) -> None:
        """Test that equally distributed actions produce maximum entropy."""
        buf = STAMBuffer(buffer_size=10, decay_rate=0.1)
        # Equal distribution across 4 actions
        for i in range(8):
            buf.record(np.array([0.0, 0.0, 0.0, 0.0]), (0.0, 0.0), i % 4)
        state = buf.get_memory_state()
        # Should be close to 1.0 (max entropy with 4 actions)
        assert state[10] == pytest.approx(1.0)


class TestEpisodeReset:
    """STAM resets properly between episodes."""

    def test_reset_produces_zero_state(self) -> None:
        """Test that reset returns the memory state to all zeros."""
        buf = STAMBuffer(buffer_size=10, decay_rate=0.1)
        for i in range(5):
            buf.record(np.array([float(i), float(i), float(i), float(i)]), (1.0, 1.0), i)
        buf.reset()
        state = buf.get_memory_state()
        np.testing.assert_array_equal(state, np.zeros(11, dtype=np.float32))

    def test_reset_allows_fresh_recording(self) -> None:
        """Test that recording after reset is not influenced by pre-reset data."""
        buf = STAMBuffer(buffer_size=10, decay_rate=0.1)
        buf.record(np.array([1.0, 0.0, 0.0, 0.0]), (0.0, 0.0), 0)
        buf.reset()
        buf.record(np.array([0.5, 0.0, 0.0, 0.0]), (0.0, 0.0), 0)
        state = buf.get_memory_state()
        assert state[0] == pytest.approx(0.5)  # Fresh value, not influenced by pre-reset
        assert len(buf) == 1
