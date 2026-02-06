"""
Quantum Reservoir Computing (QRC) Brain Architecture.

This architecture implements a quantum reservoir computing approach using a fixed random
quantum reservoir circuit with a trainable classical readout network. QRC inherently
avoids barren plateau issues that affect parameterized quantum circuits by keeping the
quantum reservoir fixed and only training the classical readout layer.

Key Features:
- **Fixed Quantum Reservoir**: Random quantum circuit (Hadamard + random rotations + CZ
  entanglement) that generates rich feature representations without trainable parameters
- **Classical Readout Network**: MLP or linear layer trained via REINFORCE policy gradients
- **Reservoir Reproducibility**: Deterministic seed-based reservoir construction
- **Probability Distribution Output**: Full 2^n dimensional state vector from measurements
- **Episode-Level Learning**: REINFORCE algorithm matching MLPReinforceBrain

Architecture:
- Input: Sensory features (gradient strength, relative angle) encoded as RY rotations
- Reservoir: Fixed random quantum circuit with configurable depth and qubit count
- Output: Action probabilities via classical readout network

The QRC brain learns by:
1. Encoding sensory inputs as RY rotations on reservoir qubits
2. Executing the fixed reservoir circuit to generate quantum state
3. Measuring the quantum state to get probability distribution
4. Passing probabilities through classical readout network
5. Updating readout weights via REINFORCE policy gradients

References
----------
- QRC-Lab: An Educational Toolbox for Quantum Reservoir Computing (arXiv:2602.03522)
"""

from copy import deepcopy

import numpy as np
import torch
from pydantic import Field, field_validator
from qiskit import QuantumCircuit
from torch import nn, optim

from quantumnematode.brain.actions import DEFAULT_ACTIONS, Action, ActionData
from quantumnematode.brain.arch import BrainData, BrainParams, ClassicalBrain
from quantumnematode.brain.arch._brain import BrainHistoryData
from quantumnematode.brain.arch.dtypes import BrainConfig, DeviceType
from quantumnematode.brain.modules import (
    ModuleName,
    extract_classical_features,
    get_classical_feature_dimension,
)
from quantumnematode.env import Direction
from quantumnematode.errors import ERROR_MISSING_IMPORT_QISKIT_AER
from quantumnematode.logging_config import logger
from quantumnematode.utils.seeding import ensure_seed, get_rng, set_global_seed

# Defaults
DEFAULT_NUM_RESERVOIR_QUBITS = 8
DEFAULT_RESERVOIR_DEPTH = 3
DEFAULT_RESERVOIR_SEED = 42
DEFAULT_READOUT_HIDDEN = 32
DEFAULT_READOUT_TYPE = "mlp"
DEFAULT_SHOTS = 1024
DEFAULT_GAMMA = 0.99
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_BASELINE_ALPHA = 0.05
DEFAULT_ENTROPY_COEF = 0.01

# Validation constants
MIN_RESERVOIR_QUBITS = 2
MIN_RESERVOIR_DEPTH = 1
MIN_READOUT_HIDDEN = 1
MIN_SHOTS = 100


class QRCBrainConfig(BrainConfig):
    """Configuration for the QRCBrain architecture.

    Supports two modes for input feature extraction:

    1. **Legacy mode** (default): Uses 2 features (gradient_strength, relative_angle)
       - Set `sensory_modules=None` (default)

    2. **Unified sensory mode**: Uses modular feature extraction from brain/modules.py
       - Set `sensory_modules` to a list of ModuleName values
       - Uses extract_classical_features() which outputs semantic-preserving ranges
       - Each module contributes 2 features [strength, angle] in [0,1] and [-1,1]
       - Required for multi-sensory scenarios (predator evasion + foraging)

    Example unified mode config:
        >>> config = QRCBrainConfig(
        ...     sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
        ... )
        >>> # input features: [food_strength, food_angle, predator_strength, predator_angle]

    Attributes
    ----------
    num_reservoir_qubits : int
        Number of qubits in the quantum reservoir (default 8).
    reservoir_depth : int
        Number of entangling layers in the reservoir circuit (default 3).
    reservoir_seed : int
        Seed for deterministic reservoir construction (default 42).
    readout_hidden : int
        Number of hidden units in MLP readout (default 32).
    readout_type : str
        Type of readout network: "mlp" or "linear" (default "mlp").
    shots : int
        Number of measurement shots for probability estimation (default 1024).
    gamma : float
        Discount factor for REINFORCE returns (default 0.99).
    learning_rate : float
        Learning rate for readout network optimizer (default 0.001).
    baseline_alpha : float
        Smoothing factor for baseline running average (default 0.05).
    entropy_coef : float
        Entropy regularization coefficient for exploration (default 0.01).
    sensory_modules : list[ModuleName] | None
        List of sensory modules for feature extraction (default None = legacy mode).
    """

    num_reservoir_qubits: int = Field(
        default=DEFAULT_NUM_RESERVOIR_QUBITS,
        description="Number of qubits in the quantum reservoir.",
    )
    reservoir_depth: int = Field(
        default=DEFAULT_RESERVOIR_DEPTH,
        description="Number of entangling layers in the reservoir circuit.",
    )
    reservoir_seed: int = Field(
        default=DEFAULT_RESERVOIR_SEED,
        description="Seed for deterministic reservoir construction.",
    )
    readout_hidden: int = Field(
        default=DEFAULT_READOUT_HIDDEN,
        description="Number of hidden units in MLP readout.",
    )
    readout_type: str = Field(
        default=DEFAULT_READOUT_TYPE,
        description="Type of readout network: 'mlp' or 'linear'.",
    )
    shots: int = Field(
        default=DEFAULT_SHOTS,
        description="Number of measurement shots for probability estimation.",
    )
    gamma: float = Field(
        default=DEFAULT_GAMMA,
        description="Discount factor for REINFORCE returns.",
    )
    learning_rate: float = Field(
        default=DEFAULT_LEARNING_RATE,
        description="Learning rate for readout network optimizer.",
    )
    baseline_alpha: float = Field(
        default=DEFAULT_BASELINE_ALPHA,
        description="Smoothing factor for baseline running average.",
    )
    entropy_coef: float = Field(
        default=DEFAULT_ENTROPY_COEF,
        description="Entropy regularization coefficient for exploration.",
    )

    # Unified sensory feature extraction (optional)
    # When set, uses extract_classical_features() which outputs:
    # - strength: [0, 1] where 0 = no signal (matches legacy semantics)
    # - angle: [-1, 1] where 0 = aligned with agent heading
    sensory_modules: list[ModuleName] | None = Field(
        default=None,
        description="List of sensory modules for feature extraction (None = legacy mode).",
    )

    @field_validator("num_reservoir_qubits")
    @classmethod
    def validate_num_reservoir_qubits(cls, v: int) -> int:
        """Validate num_reservoir_qubits >= 2."""
        if v < MIN_RESERVOIR_QUBITS:
            msg = f"num_reservoir_qubits must be >= {MIN_RESERVOIR_QUBITS}, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("reservoir_depth")
    @classmethod
    def validate_reservoir_depth(cls, v: int) -> int:
        """Validate reservoir_depth >= 1."""
        if v < MIN_RESERVOIR_DEPTH:
            msg = f"reservoir_depth must be >= {MIN_RESERVOIR_DEPTH}, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("readout_hidden")
    @classmethod
    def validate_readout_hidden(cls, v: int) -> int:
        """Validate readout_hidden >= 1."""
        if v < MIN_READOUT_HIDDEN:
            msg = f"readout_hidden must be >= {MIN_READOUT_HIDDEN}, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("shots")
    @classmethod
    def validate_shots(cls, v: int) -> int:
        """Validate shots >= 100."""
        if v < MIN_SHOTS:
            msg = f"shots must be >= {MIN_SHOTS}, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("readout_type")
    @classmethod
    def validate_readout_type(cls, v: str) -> str:
        """Validate readout_type is 'mlp' or 'linear'."""
        valid_types = ("mlp", "linear")
        if v not in valid_types:
            msg = f"readout_type must be one of {valid_types}, got '{v}'"
            raise ValueError(msg)
        return v


class QRCBrain(ClassicalBrain):
    """
    Quantum Reservoir Computing brain architecture.

    Uses a fixed quantum reservoir circuit to generate feature representations,
    with only a classical readout network trained via REINFORCE policy gradients.
    """

    def __init__(
        self,
        config: QRCBrainConfig,
        num_actions: int = 5,
        device: DeviceType = DeviceType.CPU,
        action_set: list[Action] | None = None,
    ) -> None:
        """Initialize the QRCBrain.

        Parameters
        ----------
        config : QRCBrainConfig
            Configuration for the QRC brain architecture.
        num_actions : int
            Number of available actions (default 5: forward, left, right, stay, backward).
        device : DeviceType
            Device for PyTorch operations (default CPU).
        action_set : list[Action] | None
            Custom action set (default is DEFAULT_ACTIONS).
        """
        super().__init__()

        self.config = config
        self.num_actions = num_actions
        self.device = torch.device(device.value)
        self._action_set = action_set if action_set is not None else DEFAULT_ACTIONS

        # Initialize seeding for reproducibility
        self.seed = ensure_seed(config.seed)
        self.rng = get_rng(self.seed)
        set_global_seed(self.seed)
        logger.info(f"QRCBrain using seed: {self.seed}")

        # Store sensory modules for feature extraction
        self.sensory_modules = config.sensory_modules

        # Determine input dimension based on sensory modules
        if config.sensory_modules is not None:
            # Unified sensory mode: use classical feature extraction
            self.input_dim = get_classical_feature_dimension(config.sensory_modules)
            logger.info(
                f"Using unified sensory modules: "
                f"{[m.value for m in config.sensory_modules]} "
                f"(input_dim={self.input_dim})",
            )
        else:
            # Legacy mode: 2 features (gradient_strength, relative_angle)
            self.input_dim = 2
            logger.info("Using legacy 2-feature preprocessing (gradient_strength, rel_angle)")

        # Initialize data tracking
        self.history_data = BrainHistoryData()
        self.latest_data = BrainData()

        # Reservoir configuration
        self.num_qubits = config.num_reservoir_qubits
        self.reservoir_depth = config.reservoir_depth
        self.reservoir_seed = config.reservoir_seed
        self.shots = config.shots

        # Note: Reservoir circuit is now built dynamically with data re-uploading
        # in _encode_inputs(), so we don't pre-build it here.
        self._reservoir_circuit = None  # Kept for API compatibility
        self._backend = None

        # Build readout network
        self.reservoir_output_dim = 2**self.num_qubits
        self.readout = self._build_readout_network(
            config.readout_type,
            config.readout_hidden,
        ).to(self.device)

        # Optimizer for readout network
        self.optimizer = optim.Adam(self.readout.parameters(), lr=config.learning_rate)

        # REINFORCE parameters
        self.gamma = config.gamma
        self.baseline = 0.0
        self.baseline_alpha = config.baseline_alpha
        self.entropy_coef = config.entropy_coef

        # Episode buffers for REINFORCE
        self.episode_states: list[np.ndarray] = []
        self.episode_actions: list[int] = []
        self.episode_rewards: list[float] = []
        self.episode_log_probs: list[torch.Tensor] = []
        self.episode_probs: list[torch.Tensor] = []  # For entropy calculation

        # Current action probabilities
        self.current_probabilities: np.ndarray | None = None
        self.training = True

        logger.info(
            f"QRCBrain initialized: {self.num_qubits} qubits, {self.reservoir_depth} layers, "
            f"{config.readout_type} readout ({self.reservoir_output_dim} -> {num_actions})",
        )

    def _build_readout_network(
        self,
        readout_type: str,
        hidden_dim: int,
    ) -> nn.Module:
        """Build the classical readout network.

        Parameters
        ----------
        readout_type : str
            Type of readout: "mlp" or "linear".
        hidden_dim : int
            Number of hidden units for MLP readout.

        Returns
        -------
        nn.Module
            The readout network.
        """
        if readout_type == "mlp":
            network = nn.Sequential(
                nn.Linear(self.reservoir_output_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.num_actions),
            )
        else:
            # Linear readout
            network = nn.Linear(self.reservoir_output_dim, self.num_actions)

        # Initialize weights for better gradient flow
        for module in network.modules():
            if isinstance(module, nn.Linear):
                # Orthogonal initialization for better learning dynamics
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        return network

    def _get_backend(self):  # noqa: ANN202
        """Get or create the Qiskit Aer backend for circuit execution."""
        if self._backend is None:
            try:
                from qiskit_aer import AerSimulator
            except ImportError as err:
                error_message = ERROR_MISSING_IMPORT_QISKIT_AER
                logger.error(error_message)
                raise ImportError(error_message) from err

            self._backend = AerSimulator(
                device="CPU",
                seed_simulator=self.seed,
            )
        return self._backend

    def _encode_inputs(self, features: np.ndarray) -> QuantumCircuit:
        """Encode sensory features using data re-uploading pattern.

        Uses **data re-uploading**: the input features are encoded before EACH
        reservoir layer, not just once at the beginning. This creates a richer
        representation because the input signal is interleaved with the reservoir
        dynamics rather than being scrambled once.

        Structure: H → [Input_enc → Reservoir_layer] x depth → Measure

        Parameters
        ----------
        features : np.ndarray
            Input features to encode (e.g., [gradient_strength, relative_angle]).

        Returns
        -------
        QuantumCircuit
            Complete circuit with interleaved input encoding and reservoir layers.
        """
        qc = QuantumCircuit(self.num_qubits, self.num_qubits)

        # Use seeded RNG for reproducible random rotations
        reservoir_rng = np.random.default_rng(self.reservoir_seed)

        # Initial Hadamard layer
        for q in range(self.num_qubits):
            qc.h(q)

        # Data re-uploading: encode inputs before each reservoir layer
        for _layer in range(self.reservoir_depth):
            # Input encoding: dense encoding across all qubits
            for i, feature in enumerate(features):
                angle = float(feature) * np.pi
                for q in range(self.num_qubits):
                    if i % 2 == 0:
                        # Even-indexed features (e.g., gradient_strength): RY (amplitude)
                        qc.ry(angle, q)
                    else:
                        # Odd-indexed features (e.g., relative_angle): RZ (phase)
                        qc.rz(angle, q)

            # Reservoir layer: structured rotations (fractional angles)
            # Use fixed angles based on qubit index and layer for reproducibility
            # This creates structure that preserves input signal better than random
            for q in range(self.num_qubits):
                # Use seeded random but bounded angles for some variation
                rx_angle = reservoir_rng.uniform(0, np.pi / 2)  # Max π/2 instead of 2π
                ry_angle = reservoir_rng.uniform(0, np.pi / 2)
                rz_angle = reservoir_rng.uniform(0, np.pi / 2)
                qc.rx(rx_angle, q)
                qc.ry(ry_angle, q)
                qc.rz(rz_angle, q)

            # Entangling layer: CZ in circular topology
            for q in range(self.num_qubits):
                qc.cz(q, (q + 1) % self.num_qubits)

        # Add measurements
        qc.measure(range(self.num_qubits), range(self.num_qubits))

        return qc

    def _extract_reservoir_state(self, features: np.ndarray) -> np.ndarray:
        """Execute the reservoir circuit and extract probability distribution.

        Parameters
        ----------
        features : np.ndarray
            Input features to encode.

        Returns
        -------
        np.ndarray
            Probability distribution over all 2^n bitstrings, shape (2^num_qubits,).
        """
        # Build circuit with input encoding
        qc = self._encode_inputs(features)

        # Execute on backend
        backend = self._get_backend()
        job = backend.run(qc, shots=self.shots)
        result = job.result()
        counts = result.get_counts()

        # Convert counts to probability distribution
        num_states = 2**self.num_qubits
        probs = np.zeros(num_states)

        for bitstring, count in counts.items():
            # Convert bitstring to index (Qiskit uses little-endian)
            idx = int(bitstring, 2)
            probs[idx] = count / self.shots

        # Debug: Track reservoir output statistics (sample every 50 steps)
        if len(self.episode_states) % 50 == 0:
            logger.debug(
                f"QRC reservoir: features={features}, "
                f"probs_entropy={-np.sum(probs * np.log(probs + 1e-8)):.4f}, "
                f"probs_max={np.max(probs):.4f}, "
                f"probs_nonzero={np.sum(probs > 0.01)}",  # noqa: PLR2004
            )

        return probs

    def preprocess(self, params: BrainParams) -> np.ndarray:
        """Preprocess BrainParams to extract features for the reservoir.

        Two modes:
        1. **Unified sensory mode** (when sensory_modules is set):
           Uses extract_classical_features() which outputs semantic-preserving ranges:
           - strength: [0, 1] where 0 = no signal
           - angle: [-1, 1] where 0 = aligned with agent heading

        2. **Legacy mode** (default):
           Computes gradient strength normalized to [0, 1] and relative angle
           normalized to [-1, 1].

        Parameters
        ----------
        params : BrainParams
            Brain parameters containing sensory information.

        Returns
        -------
        np.ndarray
            Preprocessed features.
        """
        # Unified sensory mode: use classical feature extraction
        if self.sensory_modules is not None:
            return extract_classical_features(params, self.sensory_modules)

        # Legacy mode: 2-feature preprocessing
        # Use gradient_strength as-is (assumed [0, 1])
        grad_strength = float(params.gradient_strength or 0.0)

        # Compute relative angle to goal ([-pi, pi])
        grad_direction = float(params.gradient_direction or 0.0)
        direction_map = {
            Direction.UP: np.pi / 2,
            Direction.DOWN: -np.pi / 2,
            Direction.LEFT: np.pi,
            Direction.RIGHT: 0.0,
        }
        agent_facing_angle = direction_map.get(params.agent_direction or Direction.UP, np.pi / 2)
        relative_angle = (grad_direction - agent_facing_angle + np.pi) % (2 * np.pi) - np.pi
        # Normalize relative angle to [-1, 1]
        rel_angle_norm = relative_angle / np.pi

        return np.array([grad_strength, rel_angle_norm], dtype=np.float32)

    def forward(self, reservoir_state: np.ndarray) -> torch.Tensor:
        """Forward pass through the readout network.

        Parameters
        ----------
        reservoir_state : np.ndarray
            Probability distribution from reservoir measurement.

        Returns
        -------
        torch.Tensor
            Action logits.
        """
        x = torch.from_numpy(reservoir_state).float().to(self.device)
        return self.readout(x)

    def run_brain(
        self,
        params: BrainParams,
        reward: float | None = None,  # noqa: ARG002
        input_data: list[float] | None = None,  # noqa: ARG002
        *,
        top_only: bool,  # noqa: ARG002
        top_randomize: bool,  # noqa: ARG002
    ) -> list[ActionData]:
        """Run the QRC brain and select an action.

        Parameters
        ----------
        params : BrainParams
            Brain parameters containing sensory information.
        reward : float | None
            Current reward (unused, kept for protocol compatibility).
        input_data : list[float] | None
            Optional input data (unused).
        top_only : bool
            Whether to return only top action (unused, always returns one).
        top_randomize : bool
            Whether to randomize selection (unused, always samples).

        Returns
        -------
        list[ActionData]
            List containing the selected action.
        """
        # Preprocess inputs
        features = self.preprocess(params)

        # Get reservoir state (probability distribution)
        reservoir_state = self._extract_reservoir_state(features)

        # Forward through readout network
        logits = self.forward(reservoir_state)

        # Apply softmax to get action probabilities
        probs = torch.softmax(logits, dim=-1)
        probs_np = probs.detach().cpu().numpy()

        # Debug: Log action probabilities (sample every 50 steps)
        if len(self.episode_states) % 50 == 0:
            logger.debug(
                f"QRC action probs: {probs_np}, "
                f"logits_range=[{logits.min().item():.3f}, {logits.max().item():.3f}]",
            )

        # Sample action from categorical distribution
        action_idx = self.rng.choice(self.num_actions, p=probs_np)
        action_name = self.action_set[action_idx]

        # Store log probability for learning
        log_prob = torch.log(probs[action_idx] + 1e-8)

        # Update tracking data
        self.latest_data.action = ActionData(
            state=action_name,
            action=action_name,
            probability=probs_np[action_idx],
        )

        # Store for episode-based learning
        self.episode_states.append(reservoir_state)
        self.episode_actions.append(action_idx)
        self.episode_log_probs.append(log_prob)
        self.episode_probs.append(probs)  # Store full probs for entropy

        self.current_probabilities = probs_np
        self.latest_data.probability = float(probs_np[action_idx])
        self.history_data.actions.append(self.latest_data.action)
        self.history_data.probabilities.append(float(probs_np[action_idx]))

        return [self.latest_data.action]

    def learn(
        self,
        params: BrainParams,  # noqa: ARG002
        reward: float,
        *,
        episode_done: bool = False,
    ) -> None:
        """Perform REINFORCE learning on the readout network.

        Parameters
        ----------
        params : BrainParams
            Brain parameters (unused).
        reward : float
            Reward for the current step.
        episode_done : bool
            Whether the episode has ended.
        """
        # Store the reward
        self.episode_rewards.append(reward)
        self.history_data.rewards.append(reward)

        logger.debug(
            f"QRC learn: episode_done={episode_done}, "
            f"rewards_len={len(self.episode_rewards)}, "
            f"states_len={len(self.episode_states)}, "
            f"actions_len={len(self.episode_actions)}, "
            f"log_probs_len={len(self.episode_log_probs)}",
        )

        # Only update at episode end
        if episode_done and len(self.episode_rewards) > 0:
            logger.info(
                f"QRC performing policy update: {len(self.episode_rewards)} rewards, "
                f"{len(self.episode_states)} states",
            )
            self._perform_policy_update()
            self._reset_episode_buffer()

    def _perform_policy_update(self) -> None:  # noqa: C901
        """Perform the REINFORCE policy gradient update."""
        if len(self.episode_states) == 0:
            logger.warning("QRC _perform_policy_update: episode_states is empty, skipping update")
            return

        # Ensure all lists have the same length
        min_length = min(
            len(self.episode_states),
            len(self.episode_actions),
            len(self.episode_rewards),
            len(self.episode_log_probs),
            len(self.episode_probs),
        )

        if min_length == 0:
            logger.warning("QRC _perform_policy_update: min_length is 0, skipping update")
            return

        logger.debug(f"QRC _perform_policy_update: min_length={min_length}")

        # Truncate to same length
        rewards = self.episode_rewards[:min_length]
        log_probs = self.episode_log_probs[:min_length]
        probs_list = self.episode_probs[:min_length]

        # Compute discounted returns: G_t = r_t + gamma * G_{t+1}
        returns = []
        discounted_return = 0.0
        for reward in reversed(rewards):
            discounted_return = reward + self.gamma * discounted_return
            returns.insert(0, discounted_return)

        # Convert to tensor and normalize for variance reduction
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
        raw_mean = returns_tensor.mean().item()
        raw_std = returns_tensor.std().item()

        logger.debug(
            f"QRC rewards: sum={sum(rewards):.2f}, "
            f"min={min(rewards):.2f}, max={max(rewards):.2f}, "
            f"raw_return_mean={raw_mean:.4f}, raw_return_std={raw_std:.4f}",
        )

        if len(returns_tensor) > 1:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (
                returns_tensor.std() + 1e-8
            )

        # Update baseline (moving average of returns)
        mean_return = returns_tensor.mean().item()
        self.baseline = (
            1 - self.baseline_alpha
        ) * self.baseline + self.baseline_alpha * mean_return

        # Compute advantages
        advantages = returns_tensor - self.baseline

        # Compute policy loss: L = -Σ log_prob(a_t) · G_t
        policy_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        entropy_sum = torch.tensor(0.0, device=self.device, requires_grad=True)
        for t in range(len(log_probs)):
            policy_loss = policy_loss - log_probs[t] * advantages[t]
            # Entropy: H(π) = -Σ π(a) log π(a)
            entropy_sum = entropy_sum - torch.sum(
                probs_list[t] * torch.log(probs_list[t] + 1e-8),
            )

        # Average over episode length
        avg_policy_loss = policy_loss / len(log_probs)
        avg_entropy = entropy_sum / len(log_probs)

        # Total loss: policy loss - entropy bonus (entropy bonus encourages exploration)
        total_loss = avg_policy_loss - self.entropy_coef * avg_entropy

        # Backpropagate through readout only
        self.optimizer.zero_grad()
        total_loss.backward()

        # Debug: Check gradients before clipping
        grad_norm_before = 0.0
        for param in self.readout.parameters():
            if param.grad is not None:
                grad_norm_before += param.grad.data.norm(2).item() ** 2
        grad_norm_before = grad_norm_before**0.5

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.readout.parameters(), max_norm=1.0)

        # Debug: Check gradients after clipping
        grad_norm_after = 0.0
        for param in self.readout.parameters():
            if param.grad is not None:
                grad_norm_after += param.grad.data.norm(2).item() ** 2
        grad_norm_after = grad_norm_after**0.5

        self.optimizer.step()

        # Store loss for tracking
        self.latest_data.loss = total_loss.item()
        if self.latest_data.loss is not None:
            self.history_data.losses.append(self.latest_data.loss)

        # Debug logging
        logger.info(
            f"QRC policy update complete: "
            f"policy_loss={avg_policy_loss.item():.4f}, "
            f"entropy={avg_entropy.item():.4f}, "
            f"total_loss={total_loss.item():.4f}, "
            f"grad_norm_before={grad_norm_before:.4f}, "
            f"grad_norm_after={grad_norm_after:.4f}, "
            f"mean_return={mean_return:.4f}, "
            f"baseline={self.baseline:.4f}",
        )

    def _reset_episode_buffer(self) -> None:
        """Reset episode buffers for next episode."""
        self.episode_states.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()
        self.episode_log_probs.clear()
        self.episode_probs.clear()

    def update_memory(self, reward: float | None = None) -> None:
        """Update internal memory (no-op for QRCBrain)."""

    def prepare_episode(self) -> None:
        """Prepare for a new episode."""
        self._reset_episode_buffer()

    def post_process_episode(self, *, episode_success: bool | None = None) -> None:
        """Post-process the episode (no-op for QRCBrain)."""

    def copy(self) -> "QRCBrain":
        """Create an independent copy of the QRCBrain.

        The copy produces an identical reservoir circuit (same seed ensures reproducibility) but
        has independent readout network weights.

        Returns
        -------
        QRCBrain
            Independent copy of this brain.
        """
        # Create new config with resolved seed
        config_copy = QRCBrainConfig(
            **{**self.config.model_dump(), "seed": self.seed},
        )

        new_brain = QRCBrain(
            config=config_copy,
            num_actions=self.num_actions,
            device=DeviceType(self.device.type),
            action_set=self._action_set,
        )

        # Copy readout weights (independent copy)
        new_brain.readout.load_state_dict(deepcopy(self.readout.state_dict()))

        # Copy optimizer state
        new_brain.optimizer.load_state_dict(deepcopy(self.optimizer.state_dict()))

        # Copy REINFORCE state
        new_brain.baseline = self.baseline

        # The reservoir circuit is the same (references the same fixed circuit)
        # This is intentional - the reservoir is fixed and never changes

        return new_brain

    @property
    def action_set(self) -> list[Action]:
        """Get the list of available actions."""
        return self._action_set

    @action_set.setter
    def action_set(self, actions: list[Action]) -> None:
        """Set the list of available actions."""
        self._action_set = actions
