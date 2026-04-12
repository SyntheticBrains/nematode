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
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from quantumnematode.env import DynamicForagingEnvironment


@dataclass(frozen=True)
class STAMChannelDef:
    """Definition of a single STAM sensory channel.

    Attributes
    ----------
    name : str
        Unique channel identifier (e.g., "food", "pheromone_alarm").
    derivative_key : str or None
        Key in the temporal data result dict for dC/dt (e.g., "food_dconcentration_dt").
    sensing_mode_attr : str or None
        Attribute name on SensingConfig controlling this channel's mode
        (e.g., "chemotaxis_mode"). Used to determine if derivative should be computed.
    """

    name: str
    derivative_key: str | None = None
    sensing_mode_attr: str | None = None


# ── Channel Registry ────────────────────────────────────────────────────────
# All possible STAM channels. Active channels are resolved at runtime
# from environment config via resolve_active_channels().

CHANNEL_REGISTRY: dict[str, STAMChannelDef] = {
    "food": STAMChannelDef(
        name="food",
        derivative_key="food_dconcentration_dt",
        sensing_mode_attr="chemotaxis_mode",
    ),
    "temperature": STAMChannelDef(
        name="temperature",
        derivative_key="temperature_ddt",
        sensing_mode_attr="thermotaxis_mode",
    ),
    "predator": STAMChannelDef(
        name="predator",
        derivative_key="predator_dconcentration_dt",
        sensing_mode_attr="nociception_mode",
    ),
    "oxygen": STAMChannelDef(
        name="oxygen",
        derivative_key="oxygen_dconcentration_dt",
        sensing_mode_attr="aerotaxis_mode",
    ),
    "pheromone_food": STAMChannelDef(
        name="pheromone_food",
        derivative_key="pheromone_food_dconcentration_dt",
        sensing_mode_attr="pheromone_food_mode",
    ),
    "pheromone_alarm": STAMChannelDef(
        name="pheromone_alarm",
        derivative_key="pheromone_alarm_dconcentration_dt",
        sensing_mode_attr="pheromone_alarm_mode",
    ),
    "pheromone_aggregation": STAMChannelDef(
        name="pheromone_aggregation",
        derivative_key="pheromone_aggregation_dconcentration_dt",
        sensing_mode_attr="pheromone_aggregation_mode",
    ),
}


def resolve_active_channels(
    env: DynamicForagingEnvironment | None,
) -> tuple[STAMChannelDef, ...]:
    """Determine active STAM channels from environment configuration.

    Channels are included only when their corresponding environment feature
    is enabled. A foraging-only config gets 1 channel (food); a full config
    with thermotaxis, predators, aerotaxis, and all pheromones gets 7.

    Parameters
    ----------
    env : DynamicForagingEnvironment or None
        Environment instance. None returns food-only (1 channel).

    Returns
    -------
    tuple[STAMChannelDef, ...]
        Active channel definitions in recording order.
    """
    channels: list[STAMChannelDef] = [CHANNEL_REGISTRY["food"]]  # Always present

    if env is not None:
        if env.thermotaxis.enabled:
            channels.append(CHANNEL_REGISTRY["temperature"])
        if env.predator.enabled:
            channels.append(CHANNEL_REGISTRY["predator"])
        if env.aerotaxis.enabled:
            channels.append(CHANNEL_REGISTRY["oxygen"])
        if env.pheromones.enabled:
            channels.append(CHANNEL_REGISTRY["pheromone_food"])
            channels.append(CHANNEL_REGISTRY["pheromone_alarm"])
            if env.pheromones.aggregation is not None:
                channels.append(CHANNEL_REGISTRY["pheromone_aggregation"])

    return tuple(channels)


def stam_dim_from_env(env: DynamicForagingEnvironment | None) -> int:
    """Compute STAM memory dimension for a given environment.

    Convenience function combining resolve_active_channels + compute_memory_dim.

    Parameters
    ----------
    env : DynamicForagingEnvironment or None
        Environment instance.

    Returns
    -------
    int
        STAM memory state dimension.
    """
    return compute_memory_dim(len(resolve_active_channels(env)))


def compute_memory_dim(num_channels: int) -> int:
    """Compute STAM memory state dimension from channel count.

    Layout: weighted_means(N) + derivatives(N) + pos_deltas(2) + action_entropy(1)
    = 2*N + 3

    Parameters
    ----------
    num_channels : int
        Number of scalar sensory channels.

    Returns
    -------
    int
        Total memory state dimension.
    """
    return num_channels * 2 + 3


# Minimum channel count (food is always present)
MIN_CHANNELS = 1


class STAMBuffer:
    """Short-Term Associative Memory with exponential decay.

    Stores recent sensory readings with biological decay kinetics.
    Produces a dynamic-size memory state vector suitable for neural network input.

    Parameters
    ----------
    buffer_size : int
        Maximum number of timesteps retained (default 30).
    decay_rate : float
        Exponential decay lambda per step (default 0.1).
        Weight for entry i steps ago: w[i] = exp(-decay_rate * i).
    num_channels : int
        Number of scalar sensory channels. Determined dynamically from
        environment config via resolve_active_channels(). Minimum 1 (food only).

    Attributes
    ----------
    MEMORY_DIM : int
        Dimension of the memory state vector (2*num_channels + 3).
    """

    def __init__(
        self,
        buffer_size: int = 30,
        decay_rate: float = 0.1,
        num_channels: int = MIN_CHANNELS,
    ) -> None:
        if num_channels < MIN_CHANNELS:
            msg = f"STAMBuffer requires at least {MIN_CHANNELS} channel. Got {num_channels}."
            raise ValueError(msg)

        self._buffer_size = buffer_size
        self._decay_rate = decay_rate
        self._num_channels = num_channels
        self.MEMORY_DIM = compute_memory_dim(num_channels)

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
            Channel order is determined at runtime by resolve_active_channels().
            Only active channels are included — no zero-padding for disabled features.
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
            Channel index into the active channels list (determined at
            runtime by resolve_active_channels). Index 0 is always food.

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
        """Return dynamic-size memory state vector for neural network input.

        Returns
        -------
        np.ndarray
            Array of shape (MEMORY_DIM,) containing:
            - [0:N] Weighted scalar means (N = num_channels)
            - [N:2N] Temporal derivatives (dC/dt per channel)
            - [2N:2N+2] Position deltas (deviation from weighted mean)
            - [2N+2] Action variety metric (Shannon entropy)

            For 4 channels: shape (11,). For 6 channels: shape (15,).
        """
        state = np.zeros(self.MEMORY_DIM, dtype=np.float64)
        n = len(self._scalar_history)

        if n == 0:
            return state.astype(np.float32)

        # --- Weighted scalar means (indices 0:num_channels) ---
        weights = self._weights[:n]
        weight_sum = np.sum(weights)
        if weight_sum > 0.0:
            scalars_array = np.array(list(self._scalar_history), dtype=np.float64)
            for ch in range(self._num_channels):
                state[ch] = np.sum(weights * scalars_array[:, ch]) / weight_sum

        # --- Temporal derivatives (indices num_channels:2*num_channels) ---
        for ch in range(self._num_channels):
            state[self._num_channels + ch] = self.compute_temporal_derivative(ch)

        # --- Position deltas (indices 2*num_channels:2*num_channels+2) ---
        pos_start = 2 * self._num_channels
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
                    state[pos_start] = deltas[0, 0] - mean_dx
                    state[pos_start + 1] = deltas[0, 1] - mean_dy
            # If only 1 entry, deviation from mean is 0 (no trend yet)

        # --- Action variety / entropy (last index) ---
        if n >= 1:
            state[pos_start + 2] = self._compute_action_entropy()

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
    def num_channels(self) -> int:
        """Number of sensory channels."""
        return self._num_channels

    @property
    def memory_dimension(self) -> int:
        """Dimension of the memory state vector."""
        return self.MEMORY_DIM

    def __len__(self) -> int:
        """Return number of entries currently in the buffer."""
        return len(self._scalar_history)
