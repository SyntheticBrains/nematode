"""Short-Term Associative Memory (STAM) buffer for temporal sensing.

Provides exponential-decay memory buffers that store recent sensory readings,
step-to-step position changes, and actions. Enables agents to infer gradient
direction from temporal comparisons of their own movement and sensory experience.

Biological basis:
    STAM in C. elegans involves cAMP-dependent protein kinase (PKA) and
    calcium/calmodulin signaling pathways. Memory formation is immediate
    (no protein synthesis required) with exponential decay over minutes
    to ~30 minutes.

Timescale mapping:
    With default parameters (buffer_size=30, decay_rate=0.1), the half-life
    is ~7 steps (ln(2)/0.1 ≈ 6.9). If each simulation step represents
    ~1-2 seconds of biological time, this maps to ~10-15 seconds of salient
    memory — appropriate for the short-term chemotaxis comparisons that
    ASE sensory neurons perform during head sweeps.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class STAMBuffer:
    """Short-Term Associative Memory with exponential decay.

    Stores recent sensory readings with biological decay kinetics.
    Produces a fixed-size memory state vector suitable for neural network input.

    Parameters
    ----------
    buffer_size : int
        Maximum number of timesteps retained (default 30).
    decay_rate : float
        Exponential decay lambda per step (default 0.1).
        Weight for entry i steps ago: w[i] = exp(-decay_rate * i).
    num_channels : int
        Number of scalar sensory channels (default 3: food, temperature, predator).

    Attributes
    ----------
    MEMORY_DIM : int
        Fixed dimension of the memory state vector (9).
    """

    MEMORY_DIM = 9

    # Named indices into the 9-dim memory state vector
    IDX_WEIGHTED_FOOD = 0
    IDX_WEIGHTED_TEMP = 1
    IDX_WEIGHTED_PRED = 2
    IDX_DERIV_FOOD = 3
    IDX_DERIV_TEMP = 4
    IDX_DERIV_PRED = 5
    IDX_POS_DELTA_X = 6
    IDX_POS_DELTA_Y = 7
    IDX_ACTION_ENTROPY = 8

    def __init__(
        self,
        buffer_size: int = 30,
        decay_rate: float = 0.1,
        num_channels: int = 3,
    ) -> None:
        if num_channels != 3:  # noqa: PLR2004
            msg = (
                f"STAMBuffer requires num_channels=3 (food, predator, temperature) "
                f"to produce the fixed {self.MEMORY_DIM}-dim memory state. "
                f"Got num_channels={num_channels}."
            )
            raise ValueError(msg)
        self._buffer_size = buffer_size
        self._decay_rate = decay_rate
        self._num_channels = num_channels

        # Precompute exponential decay weights for efficiency
        self._weights = np.array(
            [np.exp(-decay_rate * i) for i in range(buffer_size)],
            dtype=np.float64,
        )

        # Circular buffers for each data type
        self._scalar_history: deque[np.ndarray] = deque(maxlen=buffer_size)
        self._position_delta_history: deque[tuple[float, float]] = deque(
            maxlen=buffer_size,
        )
        self._action_history: deque[int] = deque(maxlen=buffer_size)

    def record(
        self,
        scalars: np.ndarray,
        position_delta: tuple[float, float],
        action: int,
    ) -> None:
        """Record one timestep of sensory data.

        Parameters
        ----------
        scalars : np.ndarray
            Scalar readings for each channel, shape (num_channels,).
            Order: [food_concentration, temperature, predator_concentration].
            Disabled channels should pass 0.0.
        position_delta : tuple[float, float]
            Step-to-step position change (dx, dy) — proprioceptive movement
            signal. NOT absolute grid coordinates.
        action : int
            Action taken this step (integer action index).
        """
        scalars_arr = np.asarray(scalars, dtype=np.float64)
        if scalars_arr.ndim != 1 or scalars_arr.shape[0] != self._num_channels:
            msg = (
                f"scalars must be a 1-D array of length {self._num_channels}, "
                f"got shape {scalars_arr.shape}"
            )
            raise ValueError(msg)
        self._scalar_history.appendleft(scalars_arr)
        self._position_delta_history.appendleft(position_delta)
        self._action_history.appendleft(action)

    def reset(self) -> None:
        """Reset all buffers at episode start.

        No STAM state carries across episodes. Cross-episode learning
        belongs to the brain's trained weights, not short-term memory.
        """
        self._scalar_history.clear()
        self._position_delta_history.clear()
        self._action_history.clear()

    def compute_temporal_derivative(self, channel: int) -> float:
        """Compute dC/dt for a given channel using weighted finite difference.

        Uses the formula: dC/dt = Σ(w[i] * (C[i] - C[i+1])) / Σ(w[i])
        where i=0 is most recent and w[i] = exp(-decay_rate * i).

        Parameters
        ----------
        channel : int
            Channel index (0=food, 1=temperature, 2=predator).

        Returns
        -------
        float
            Weighted average rate of change. Positive means increasing,
            negative means decreasing. Returns 0.0 if fewer than 2 entries.
        """
        n = len(self._scalar_history)
        if n < 2:  # noqa: PLR2004
            return 0.0

        # Compute weighted finite differences
        weights = self._weights[: n - 1]
        diffs = np.array(
            [
                self._scalar_history[i][channel] - self._scalar_history[i + 1][channel]
                for i in range(n - 1)
            ],
            dtype=np.float64,
        )

        weight_sum = np.sum(weights)
        if weight_sum == 0.0:
            return 0.0

        return float(np.sum(weights * diffs) / weight_sum)

    def get_memory_state(self) -> np.ndarray:
        """Return fixed-size memory state vector for neural network input.

        Returns
        -------
        np.ndarray
            Array of shape (9,) containing:
            - [0:3] Weighted scalar means (food, temperature, predator)
            - [3:6] Temporal derivatives (dC/dt per channel)
            - [6:8] Position deltas (deviation of most recent step-to-step
                     movement from weighted mean of recent movements —
                     captures change in movement pattern)
            - [8]   Action variety metric (entropy of recent actions —
                     computational convenience for learning, not biologically
                     motivated; helps brains distinguish "stuck in a loop"
                     from "actively exploring")
        """
        state = np.zeros(self.MEMORY_DIM, dtype=np.float64)
        n = len(self._scalar_history)

        if n == 0:
            return state.astype(np.float32)

        # --- Weighted scalar means (indices 0-2) ---
        weights = self._weights[:n]
        weight_sum = np.sum(weights)
        if weight_sum > 0.0:
            scalars_array = np.array(list(self._scalar_history), dtype=np.float64)
            for ch in range(self._num_channels):
                state[ch] = np.sum(weights * scalars_array[:, ch]) / weight_sum

        # --- Temporal derivatives (indices 3-5) ---
        for ch in range(self._num_channels):
            state[3 + ch] = self.compute_temporal_derivative(ch)

        # --- Position deltas (indices 6-7) ---
        if n >= 1:
            deltas = np.array(list(self._position_delta_history), dtype=np.float64)
            if n >= 2:  # noqa: PLR2004
                # Weighted mean of recent movements
                pos_weights = self._weights[:n]
                pos_weight_sum = np.sum(pos_weights)
                if pos_weight_sum > 0.0:
                    mean_dx = np.sum(pos_weights * deltas[:, 0]) / pos_weight_sum
                    mean_dy = np.sum(pos_weights * deltas[:, 1]) / pos_weight_sum
                    # Deviation of most recent movement from weighted mean
                    state[6] = deltas[0, 0] - mean_dx
                    state[7] = deltas[0, 1] - mean_dy
            # If only 1 entry, deviation from mean is 0 (no trend yet)

        # --- Action variety / entropy (index 8) ---
        if n >= 1:
            state[8] = self._compute_action_entropy()

        return state.astype(np.float32)

    def _compute_action_entropy(self) -> float:
        """Compute entropy of recent actions as a diversity metric.

        This is a computational convenience for learning, not biologically
        motivated. There is no direct C. elegans neural correlate for
        "action diversity awareness."

        Returns
        -------
        float
            Shannon entropy of action distribution, normalized to [0, 1].
            0 = all same action, 1 = maximum diversity.
        """
        if len(self._action_history) == 0:
            return 0.0

        actions = np.array(list(self._action_history))
        unique, counts = np.unique(actions, return_counts=True)

        if len(unique) <= 1:
            return 0.0

        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log2(probs))

        # Normalize by max possible entropy (log2 of number of action types)
        # Use 4 as default action count (FORWARD, LEFT, RIGHT, STAY)
        max_entropy = np.log2(max(len(unique), 4))
        if max_entropy > 0:
            return float(entropy / max_entropy)
        return 0.0

    @property
    def memory_dimension(self) -> int:
        """Dimension of the memory state vector."""
        return self.MEMORY_DIM

    def __len__(self) -> int:
        """Return number of entries currently in the buffer."""
        return len(self._scalar_history)
