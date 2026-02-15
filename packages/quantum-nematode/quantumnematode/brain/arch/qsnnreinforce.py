"""
Quantum Spiking Neural Network (QSNN) Brain Architecture.

A hybrid quantum-classical spiking neural network that executes quantum circuits
(QLIF neurons) in the forward pass and uses classical surrogate gradients in the
backward pass for REINFORCE policy gradient learning. This sidesteps both the
parameter-shift rule's computational cost and the barren plateau problem inherent
in deep variational quantum circuits.

Key Features:
- **QLIF Neurons**: 2-gate circuit per Brand & Petruccione (2024):
  |0> -> RY(theta + tanh(w·x)·pi) -> RX(theta_leak) -> Measure
- **Surrogate Gradient Learning**: QLIFSurrogateSpike autograd function — forward
  pass returns quantum-measured spike probability, backward pass uses sigmoid
  surrogate centered at pi/2 (the RY gate's quantum transition point)
- **REINFORCE Policy Gradient**: Advantage-normalized returns with clipping,
  entropy regularization, and exploration decay schedule
- **Layered Feedforward SNN**: sensory -> hidden -> motor (e.g. 6->8->4, ~60 params)
- **Dual Learning Modes**: Surrogate gradient (default) or 3-factor Hebbian (legacy)

Architecture::

    Sensors -> Sigmoid -> QLIF(sensory->hidden) -> QLIF(hidden->motor) -> Softmax -> Action
                          [quantum circuits]       [quantum circuits]

The hybrid training approach works as follows:
1. Forward: Encode features as spike probabilities, propagate through QLIF quantum
   circuits layer by layer, measure spike probabilities from quantum state
2. Action: Convert motor spike probabilities to action logits, apply temperature
   scaling and epsilon-greedy exploration, sample action
3. Backward: Recompute forward pass with QLIFSurrogateSpike wrapping each quantum
   measurement — sigmoid surrogate gradient flows through RY angles to both
   theta (membrane bias) and weight parameters via autograd
4. Update: REINFORCE loss with advantage clipping and entropy bonus, Adam optimizer
   with gradient clipping, weight clamping for stability

References
----------
- Brand & Petruccione (2024). "A quantum leaky integrate-and-fire spiking neuron
  and network." npj Quantum Information, 10(1), 16.
- Neftci et al. (2019). "Surrogate gradient learning in spiking neural networks."
  IEEE Signal Processing Magazine, 36(6), 51-63.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from pydantic import Field, field_validator
from torch import nn

from quantumnematode.brain.actions import DEFAULT_ACTIONS, Action, ActionData
from quantumnematode.brain.arch import BrainData, BrainParams, ClassicalBrain
from quantumnematode.brain.arch._brain import BrainHistoryData
from quantumnematode.brain.arch._qlif_layers import (
    DEFAULT_SURROGATE_ALPHA,  # noqa: F401  # re-exported
    LOGIT_SCALE,
    WEIGHT_INIT_SCALE,
    QLIFSurrogateSpike,  # noqa: F401  # re-exported
    build_qlif_circuit,  # noqa: F401  # re-exported
    encode_sensory_spikes,
    execute_qlif_layer,
    execute_qlif_layer_differentiable,
    execute_qlif_layer_differentiable_cached,
    get_qiskit_backend,
)
from quantumnematode.brain.arch.dtypes import BrainConfig, DeviceType
from quantumnematode.brain.modules import (
    ModuleName,
    extract_classical_features,
    get_classical_feature_dimension,
)
from quantumnematode.env import Direction
from quantumnematode.logging_config import logger
from quantumnematode.utils.seeding import ensure_seed, get_rng, set_global_seed

# Defaults
DEFAULT_NUM_SENSORY_NEURONS = 6
DEFAULT_NUM_HIDDEN_NEURONS = 8
DEFAULT_NUM_MOTOR_NEURONS = 4
DEFAULT_MEMBRANE_TAU = 0.9
DEFAULT_THRESHOLD = 0.5
DEFAULT_REFRACTORY_PERIOD = 0
DEFAULT_USE_LOCAL_LEARNING = True
DEFAULT_SHOTS = 1024
DEFAULT_GAMMA = 0.99
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_ENTROPY_COEF = 0.01
DEFAULT_WEIGHT_CLIP = 3.0
DEFAULT_UPDATE_INTERVAL = 20
DEFAULT_CLIP_EPSILON = 0.2

# Theta motor norm clamping: max L2 norm for theta_motor vector.
# Prevents "theta_m death spiral" where motor threshold parameters absorb
# disproportionate gradient (94%+ of total) and saturate entropy to near-uniform.
# With 4 motor neurons and max_norm=1.0, individual theta values stay ≤ 0.5,
# keeping motor neurons in the responsive region of the QLIF circuit.
DEFAULT_THETA_MOTOR_MAX_NORM = 1.0

# Running reward normalization: EMA smoothing factor for reward mean/variance.
# Normalizes rewards to zero-mean unit-variance before storing, making REINFORCE
# learning invariant to reward magnitude. This prevents negative reward dominance
# where large penalties drown out food collection signals.
# Alpha=0.01 gives ~100-step effective window, smoothing across intra-episode
# REINFORCE windows while adapting to changing reward distributions.
DEFAULT_REWARD_NORM_ALPHA = 0.01

# Minimum batch size for REINFORCE updates. With only 1 step, returns_std=0
# and advantages reduce to a single clamped value, producing uninformative
# gradient spikes that destabilise learning.
_MIN_REINFORCE_BATCH_SIZE = 2


@dataclass
class _ReinforceUpdateStats:
    """Bundled statistics for REINFORCE optimizer step logging."""

    raw_mean: float
    returns_tensor: torch.Tensor
    epoch: int
    num_epochs: int


# Multi-epoch REINFORCE: number of gradient passes per update window.
# Epoch 0 runs quantum circuits and caches spike probabilities;
# epochs 1+ reuse cached spike probs but recompute ry_angles from updated
# weights, so surrogate gradients reflect the latest parameters.
# PPO clipping constrains policy ratio across epochs, preventing divergence.
DEFAULT_NUM_REINFORCE_EPOCHS = 1

# Classical critic defaults for QSNN-AC (actor-critic) mode.
# A small MLP critic provides V(s) estimates for GAE advantages,
# replacing vanilla REINFORCE's (returns - mean) / std normalization.
# The critic is a separate classical network with its own optimizer.
DEFAULT_CRITIC_HIDDEN_DIM = 64
DEFAULT_CRITIC_NUM_LAYERS = 2
DEFAULT_CRITIC_LR = 0.003
DEFAULT_GAE_LAMBDA = 0.95
DEFAULT_VALUE_LOSS_COEF = 0.5

# Validation constants
MIN_SENSORY_NEURONS = 1
MIN_HIDDEN_NEURONS = 1
MIN_MOTOR_NEURONS = 2
MIN_SHOTS = 100

# L2 weight decay — gentle regularization now that eligibility normalization
# bounds update magnitude. Keeps weights from drifting unboundedly over many
# episodes without fighting against the learning signal.
WEIGHT_DECAY_FACTOR = 0.01

# Eligibility trace normalization — caps the Frobenius norm of eligibility traces
# before multiplying by lr * advantage. This bounds the maximum weight delta per
# update to approximately lr * MAX_ELIGIBILITY_NORM * |advantage|.
# With lr=0.1, max_norm=1.0, and typical advantage ~1-5, max delta per update
# is ~0.1-0.5, keeping weights in tanh's useful range [-2, 2].
MAX_ELIGIBILITY_NORM = 1.0

# Exploration floor — mixes softmax action probs with uniform distribution,
# guaranteeing every action is sampled with at least epsilon/num_actions
# probability and preventing policy collapse where one action reaches >99%.
# With epsilon=0.1 and 4 actions, minimum per-action probability = 2.5%.
EXPLORATION_EPSILON = 0.1

# Gradient clipping max norm for surrogate gradient mode.
# Prevents large gradient spikes from destabilizing training.
SURROGATE_GRAD_CLIP = 1.0

# Advantage clipping for surrogate gradient REINFORCE.
# Prevents outlier returns from producing catastrophically large policy updates.
# SpikingReinforceBrain uses 2.0; this caps normalized advantages to [-2, +2].
ADVANTAGE_CLIP = 2.0

# Exploration decay: default number of episodes over which exploration decreases.
# Epsilon and temperature decay linearly from initial to final values over
# this many episodes, allowing more exploitation as the policy matures.
# Configurable per-task via QSNNReinforceBrainConfig.exploration_decay_episodes.
EXPLORATION_DECAY_EPISODES = 80

# Learning rate decay: number of episodes over which LR decays via cosine
# annealing. After this many episodes, LR reaches its minimum value.
# 200 matches the typical training length for foraging experiments.
LR_DECAY_EPISODES = 200

# Minimum LR as a fraction of the initial LR. With lr=0.01 and factor=0.1,
# the LR decays from 0.01 to 0.001 over LR_DECAY_EPISODES episodes.
# This prevents late-episode weight perturbation that causes catastrophic
# forgetting in converged policies. Note: 0.05 (min LR=0.0005) is too low —
# cosine annealing reaches sub-0.001 LR by episode ~100, starving gradient
# signal during mid-training refinement. 0.1 is the minimum viable floor.
LR_MIN_FACTOR = 0.1

# Multi-timestep integration: number of QLIF timesteps per decision.
# SpikingReinforceBrain uses 100 classical timesteps; QSNN uses quantum circuits
# (much more expensive per timestep), so we use a smaller default. Averaging
# spike probabilities across multiple timesteps reduces quantum shot noise
# variance, giving REINFORCE cleaner gradient signal. With shots=1024 and
# 10 timesteps, effective samples per decision = 10240.
DEFAULT_NUM_INTEGRATION_STEPS = 10

# Adaptive entropy bonus: when mean episode entropy drops below this threshold,
# entropy_coef is scaled up to push the policy back toward exploration.
# 0.5 nats ≈ 36% of max entropy for 4 actions (ln(4)≈1.386).
ENTROPY_FLOOR = 0.5

# Maximum multiplier for entropy_coef when entropy is critically low.
# When entropy → 0, effective entropy_coef = base * ENTROPY_BOOST_MAX.
# With base entropy_coef=0.02, max effective = 0.02 * 20.0 = 0.40,
# which is competitive with the REINFORCE policy gradient force (~1.0)
# and sufficient to counter premature policy commitment.
ENTROPY_BOOST_MAX = 20.0

# Entropy ceiling: when entropy exceeds this fraction of max entropy,
# suppress entropy bonus to let the policy gradient sharpen the policy.
# With 4 actions, max entropy = ln(4) ≈ 1.386, so ceiling at 0.95 * 1.386
# ≈ 1.317 nats. This prevents entropy explosion where the policy drifts
# to uniform random and never recovers.
ENTROPY_CEILING_FRACTION = 0.95


class CriticMLP(nn.Module):
    """Classical MLP for state value estimation in QSNN-AC mode."""

    def __init__(self, input_dim: int, hidden_dim: int = 32, num_layers: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute state value estimate from input features."""
        return self.network(x).squeeze(-1)


class QSNNReinforceBrainConfig(BrainConfig):
    """Configuration for the QSNNReinforceBrain architecture.

    Supports two modes for input feature extraction:

    1. **Legacy mode** (default): Uses 2 features (gradient_strength, relative_angle)
       - Set `sensory_modules=None` (default)

    2. **Unified sensory mode**: Uses modular feature extraction from brain/modules.py
       - Set `sensory_modules` to a list of ModuleName values
       - Uses extract_classical_features() which outputs semantic-preserving ranges
       - Each module contributes 2 features [strength, angle] in [0,1] and [-1,1]

    Attributes
    ----------
    num_sensory_neurons : int
        Number of sensory layer neurons (default 6).
    num_hidden_neurons : int
        Number of hidden interneuron layer neurons (default 4).
    num_motor_neurons : int
        Number of motor layer neurons matching action space (default 4).
    membrane_tau : float
        Leak time constant for QLIF neurons in (0, 1] (default 0.9).
    threshold : float
        Firing threshold in (0, 1) (default 0.5).
    refractory_period : int
        Timesteps to suppress activity after firing (default 2).
    use_local_learning : bool
        Use 3-factor local learning vs REINFORCE fallback (default True).
    shots : int
        Number of quantum measurement shots (default 1024).
    gamma : float
        Discount factor for returns (default 0.99).
    learning_rate : float
        Learning rate for weight updates (default 0.01).
    entropy_coef : float
        Entropy regularization coefficient (default 0.01).
    weight_clip : float
        Maximum absolute weight value for stability (default 5.0).
    sensory_modules : list[ModuleName] | None
        List of sensory modules for feature extraction (None = legacy mode).
    """

    num_sensory_neurons: int = Field(
        default=DEFAULT_NUM_SENSORY_NEURONS,
        description="Number of sensory layer neurons.",
    )
    num_hidden_neurons: int = Field(
        default=DEFAULT_NUM_HIDDEN_NEURONS,
        description="Number of hidden interneuron layer neurons.",
    )
    num_motor_neurons: int = Field(
        default=DEFAULT_NUM_MOTOR_NEURONS,
        description="Number of motor layer neurons.",
    )
    membrane_tau: float = Field(
        default=DEFAULT_MEMBRANE_TAU,
        description="Leak time constant for QLIF neurons in (0, 1].",
    )
    threshold: float = Field(
        default=DEFAULT_THRESHOLD,
        description="Firing threshold in (0, 1).",
    )
    refractory_period: int = Field(
        default=DEFAULT_REFRACTORY_PERIOD,
        description="Timesteps to suppress activity after firing.",
    )
    use_local_learning: bool = Field(
        default=DEFAULT_USE_LOCAL_LEARNING,
        description="Use 3-factor local learning vs REINFORCE fallback.",
    )
    shots: int = Field(
        default=DEFAULT_SHOTS,
        description="Number of quantum measurement shots.",
    )
    gamma: float = Field(
        default=DEFAULT_GAMMA,
        description="Discount factor for returns.",
    )
    learning_rate: float = Field(
        default=DEFAULT_LEARNING_RATE,
        description="Learning rate for weight updates.",
    )
    entropy_coef: float = Field(
        default=DEFAULT_ENTROPY_COEF,
        description="Entropy regularization coefficient.",
    )
    weight_clip: float = Field(
        default=DEFAULT_WEIGHT_CLIP,
        description="Maximum absolute weight value for stability.",
    )
    update_interval: int = Field(
        default=DEFAULT_UPDATE_INTERVAL,
        description="Apply weight updates every N steps within an episode (0 = episode-end only).",
    )
    num_integration_steps: int = Field(
        default=DEFAULT_NUM_INTEGRATION_STEPS,
        description="Number of QLIF timesteps per decision (averages out quantum shot noise).",
    )
    num_reinforce_epochs: int = Field(
        default=DEFAULT_NUM_REINFORCE_EPOCHS,
        description="Number of gradient passes per REINFORCE update window. "
        "Epoch 0 runs quantum circuits; epochs 1+ reuse cached spike probabilities.",
    )

    # Training schedule constants (configurable per-task, defaults match foraging baseline)
    exploration_decay_episodes: int = Field(
        default=80,
        description="Episodes over which exploration epsilon/temperature decay. "
        "Longer values keep exploration high for harder tasks.",
    )
    exploration_epsilon: float = Field(
        default=EXPLORATION_EPSILON,
        description="Initial exploration epsilon for epsilon-greedy action selection.",
    )
    lr_decay_episodes: int = Field(
        default=LR_DECAY_EPISODES,
        description="Episodes over which learning rate decays via cosine annealing.",
    )
    lr_min_factor: float = Field(
        default=LR_MIN_FACTOR,
        description="Minimum LR as fraction of initial LR after decay.",
    )
    logit_scale: float = Field(
        default=LOGIT_SCALE,
        description="Scaling factor for converting spike probabilities to action logits.",
    )
    advantage_clip: float = Field(
        default=ADVANTAGE_CLIP,
        description="Clamp normalized advantages to [-clip, +clip].",
    )
    clip_epsilon: float = Field(
        default=DEFAULT_CLIP_EPSILON,
        description="PPO-style clipping epsilon for policy ratio. "
        "Constrains ratio to [1-eps, 1+eps] to prevent catastrophic policy shifts.",
    )
    theta_motor_max_norm: float = Field(
        default=DEFAULT_THETA_MOTOR_MAX_NORM,
        description="Max L2 norm for theta_motor vector. "
        "Prevents theta_m death spiral where motor thresholds absorb excessive gradient.",
    )
    reward_norm_alpha: float = Field(
        default=DEFAULT_REWARD_NORM_ALPHA,
        description="EMA smoothing factor for running reward normalization. "
        "Normalizes rewards to zero-mean unit-variance before REINFORCE updates.",
    )
    use_reward_normalization: bool = Field(
        default=True,
        description="Enable running reward normalization. "
        "Makes learning invariant to reward magnitude, preventing penalty dominance.",
    )
    use_critic: bool = Field(
        default=False,
        description="Enable classical critic for actor-critic mode (GAE advantages).",
    )
    critic_hidden_dim: int = Field(
        default=DEFAULT_CRITIC_HIDDEN_DIM,
        description="Hidden layer dimension for the critic MLP.",
    )
    critic_num_layers: int = Field(
        default=DEFAULT_CRITIC_NUM_LAYERS,
        description="Number of hidden layers in the critic MLP.",
    )
    critic_lr: float = Field(
        default=DEFAULT_CRITIC_LR,
        description="Learning rate for the critic optimizer (no cosine decay).",
    )
    gae_lambda: float = Field(
        default=DEFAULT_GAE_LAMBDA,
        description="Lambda for GAE advantage estimation (higher = less biased).",
    )
    value_loss_coef: float = Field(
        default=DEFAULT_VALUE_LOSS_COEF,
        description="Coefficient for critic value loss.",
    )
    entropy_floor: float = Field(
        default=ENTROPY_FLOOR,
        description="Entropy threshold below which adaptive entropy boost activates.",
    )
    entropy_boost_max: float = Field(
        default=ENTROPY_BOOST_MAX,
        description="Maximum multiplier for entropy_coef when entropy is critically low.",
    )
    entropy_ceiling_fraction: float = Field(
        default=ENTROPY_CEILING_FRACTION,
        description="Fraction of max entropy above which entropy bonus is suppressed.",
    )

    # Unified sensory feature extraction (optional)
    sensory_modules: list[ModuleName] | None = Field(
        default=None,
        description="List of sensory modules for feature extraction (None = legacy mode).",
    )

    @field_validator("num_sensory_neurons")
    @classmethod
    def validate_num_sensory_neurons(cls, v: int) -> int:
        """Validate num_sensory_neurons >= 1."""
        if v < MIN_SENSORY_NEURONS:
            msg = f"num_sensory_neurons must be >= {MIN_SENSORY_NEURONS}, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("num_hidden_neurons")
    @classmethod
    def validate_num_hidden_neurons(cls, v: int) -> int:
        """Validate num_hidden_neurons >= 1."""
        if v < MIN_HIDDEN_NEURONS:
            msg = f"num_hidden_neurons must be >= {MIN_HIDDEN_NEURONS}, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("num_motor_neurons")
    @classmethod
    def validate_num_motor_neurons(cls, v: int) -> int:
        """Validate num_motor_neurons >= 2."""
        if v < MIN_MOTOR_NEURONS:
            msg = f"num_motor_neurons must be >= {MIN_MOTOR_NEURONS}, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("membrane_tau")
    @classmethod
    def validate_membrane_tau(cls, v: float) -> float:
        """Validate membrane_tau in (0, 1]."""
        if not (0 < v <= 1):
            msg = f"membrane_tau must be in (0, 1], got {v}"
            raise ValueError(msg)
        return v

    @field_validator("threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        """Validate threshold in (0, 1)."""
        if not (0 < v < 1):
            msg = f"threshold must be in (0, 1), got {v}"
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

    @field_validator("num_reinforce_epochs")
    @classmethod
    def validate_num_reinforce_epochs(cls, v: int) -> int:
        """Validate num_reinforce_epochs >= 1."""
        if v < 1:
            msg = f"num_reinforce_epochs must be >= 1, got {v}"
            raise ValueError(msg)
        return v


class QSNNReinforceBrain(ClassicalBrain):
    """
    Quantum Spiking Neural Network brain architecture.

    Uses QLIF neurons with trainable synaptic weights and local learning rules
    to avoid both QRC's representation problem and QVarCircuit's barren plateaus.
    """

    def __init__(
        self,
        config: QSNNReinforceBrainConfig,
        num_actions: int = 4,
        device: DeviceType = DeviceType.CPU,
        action_set: list[Action] | None = None,
    ) -> None:
        """Initialize the QSNNReinforceBrain.

        Parameters
        ----------
        config : QSNNReinforceBrainConfig
            Configuration for the QSNN brain architecture.
        num_actions : int
            Number of available actions (default 4: forward, left, right, backward).
        device : DeviceType
            Device for PyTorch operations (default CPU).
        action_set : list[Action] | None
            Custom action set (default is DEFAULT_ACTIONS).
        """
        super().__init__()

        self.config = config
        self.num_actions = num_actions
        self.device = torch.device(device.value)
        self._action_set = action_set if action_set is not None else DEFAULT_ACTIONS[:num_actions]

        # Validate action_set length matches num_actions
        if self.num_actions != len(self._action_set):
            msg = (
                f"num_actions ({self.num_actions}) does not match "
                f"action_set length ({len(self._action_set)})"
            )
            raise ValueError(msg)

        # Initialize seeding for reproducibility
        self.seed = ensure_seed(config.seed)
        self.rng = get_rng(self.seed)
        set_global_seed(self.seed)
        logger.info(f"QSNNReinforceBrain using seed: {self.seed}")

        # Store sensory modules for feature extraction
        self.sensory_modules = config.sensory_modules

        # Determine input dimension based on sensory modules
        if config.sensory_modules is not None:
            self.input_dim = get_classical_feature_dimension(config.sensory_modules)
            logger.info(
                f"Using unified sensory modules: "
                f"{[m.value for m in config.sensory_modules]} "
                f"(input_dim={self.input_dim})",
            )
        else:
            self.input_dim = 2
            logger.info("Using legacy 2-feature preprocessing (gradient_strength, rel_angle)")

        # Initialize data tracking
        self.history_data = BrainHistoryData()
        self.latest_data = BrainData()

        # Network configuration from config
        self.num_sensory = config.num_sensory_neurons
        self.num_hidden = config.num_hidden_neurons
        self.num_motor = config.num_motor_neurons
        self.membrane_tau = config.membrane_tau
        self.threshold = config.threshold
        self.refractory_period = config.refractory_period
        self.shots = config.shots
        self.use_local_learning = config.use_local_learning
        self.gamma = config.gamma
        self.learning_rate = config.learning_rate
        self.entropy_coef = config.entropy_coef
        self.weight_clip = config.weight_clip
        self.update_interval = config.update_interval
        self.num_integration_steps = config.num_integration_steps

        # Classical critic for actor-critic mode (GAE advantages)
        self.use_critic = config.use_critic
        self.critic: CriticMLP | None = None
        self.critic_optimizer: torch.optim.Adam | None = None

        if self.use_critic:
            self.critic = CriticMLP(
                input_dim=self.input_dim,
                hidden_dim=config.critic_hidden_dim,
                num_layers=config.critic_num_layers,
            ).to(self.device)
            self.critic_optimizer = torch.optim.Adam(
                self.critic.parameters(),
                lr=config.critic_lr,
            )
            if self.use_local_learning:
                logger.warning("use_critic=True has no effect with use_local_learning=True")

        # Step counter for intra-episode updates
        self._step_count = 0

        # Episode counter for exploration decay
        self._episode_count = 0

        # Compute leak angle: theta_leak = (1 - membrane_tau) * pi
        self.leak_angle = (1 - self.membrane_tau) * np.pi

        # Initialize network components
        self._init_network_weights()
        self._init_episode_state()

        logger.info(
            f"QSNNReinforceBrain initialized: "
            f"{self.num_sensory}->{self.num_hidden}->{self.num_motor} "
            f"neurons, membrane_tau={self.membrane_tau}, threshold={self.threshold}, "
            f"use_local_learning={self.use_local_learning}, "
            f"num_integration_steps={self.num_integration_steps}",
        )

    def _init_network_weights(self) -> None:
        """Initialize trainable weight matrices and neuron parameters."""
        # Random Gaussian init with moderate scale. Combined with theta_hidden=π/4,
        # this gives spike probs 0.05-0.35 — enough for REINFORCE to differentiate
        # actions while keeping gradient noise bounded.
        self.W_sh = (
            torch.randn(
                self.num_sensory,
                self.num_hidden,
                device=self.device,
            )
            * WEIGHT_INIT_SCALE
        )

        self.W_hm = (
            torch.randn(
                self.num_hidden,
                self.num_motor,
                device=self.device,
            )
            * WEIGHT_INIT_SCALE
        )

        # Trainable membrane potential parameters per neuron.
        # Theta=π/4 provides a moderate "warm start": hidden neurons begin at
        # P(spike) ≈ sin²(π/8) ≈ 0.15, with surrogate gradient at ~60% of peak.
        # This gives meaningful gradient sensitivity from step 1 without causing
        # entropy collapse (which happens at θ=π/2) or flat landscapes (at θ=0).
        # Motor thetas stay at zero to avoid biasing initial action preferences.
        self.theta_hidden = torch.full(
            (self.num_hidden,),
            np.pi / 4,
            device=self.device,
        )
        self.theta_motor = torch.zeros(self.num_motor, device=self.device)

        # Surrogate gradient mode: enable autograd and create optimizer
        if not self.use_local_learning:
            self.W_sh.requires_grad_(True)  # noqa: FBT003
            self.W_hm.requires_grad_(True)  # noqa: FBT003
            self.theta_hidden.requires_grad_(True)  # noqa: FBT003
            self.theta_motor.requires_grad_(True)  # noqa: FBT003
            self.optimizer = torch.optim.Adam(
                [self.W_sh, self.W_hm, self.theta_hidden, self.theta_motor],
                lr=self.learning_rate,
            )
            # Cosine annealing LR decay: lr decays from initial to initial*lr_min_factor
            # over lr_decay_episodes episodes. Prevents late-episode weight perturbation
            # that causes catastrophic forgetting in converged policies.
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.lr_decay_episodes,
                eta_min=self.learning_rate * self.config.lr_min_factor,
            )
        else:
            self.optimizer = None
            self.scheduler = None

        # Eligibility traces for local learning (accumulated per episode)
        self.eligibility_sh = torch.zeros_like(self.W_sh)
        self.eligibility_hm = torch.zeros_like(self.W_hm)

        # Eligibility traces for theta (membrane) parameters
        self.eligibility_theta_hidden = torch.zeros_like(self.theta_hidden)
        self.eligibility_theta_motor = torch.zeros_like(self.theta_motor)

        # Refractory state tracking
        self.refractory_hidden = np.zeros(self.num_hidden, dtype=np.int32)
        self.refractory_motor = np.zeros(self.num_motor, dtype=np.int32)

        # Last timestep spike data for deferred eligibility accumulation
        self._last_sensory_spikes = np.zeros(self.num_sensory)
        self._last_hidden_spikes = np.zeros(self.num_hidden)
        self._last_motor_spikes = np.zeros(self.num_motor)

        # Qiskit backend for circuit execution
        self._backend = None

    def _init_episode_state(self) -> None:
        """Initialize episode tracking state."""
        self.episode_rewards: list[float] = []
        self.episode_actions: list[int] = []
        self.episode_features: list[np.ndarray] = []
        self.episode_old_log_probs: list[float] = []
        self.episode_values: list[float] = []

        # Deferred update flag for correct bootstrap computation in AC mode.
        # When a window boundary is hit in learn(), the update is deferred until
        # the next run_brain() call, which has access to V(s_{N+1}) — the value
        # of the state AFTER the window, needed as GAE bootstrap.
        self._pending_critic_update: bool = False

        # Cached quantum spike probabilities for multi-epoch REINFORCE.
        # Structure: [timestep][integration_step] -> {'hidden': [...], 'motor': [...]}
        # Epoch 0 fills this cache; epochs 1+ reuse it to avoid re-running quantum circuits.
        self._cached_spike_probs: list[list[dict[str, list[float]]]] = []

        # REINFORCE baseline
        self.baseline = 0.0
        self.baseline_alpha = 0.05

        # Running reward normalization stats (persist across episodes)
        # Initialized here but NOT reset between episodes — the running
        # statistics accumulate to track the reward distribution over time.
        self.reward_running_mean: float = 0.0
        self.reward_running_var: float = 1.0

        # Current action probabilities
        self.current_probabilities: np.ndarray | None = None
        self.training = True

    def _get_backend(self):  # noqa: ANN202
        """Get or create the Qiskit Aer backend for circuit execution."""
        if self._backend is None:
            self._backend = get_qiskit_backend(
                DeviceType(self.device.type) if hasattr(self.device, "type") else DeviceType.CPU,
                seed=self.seed,
            )
        return self._backend

    def _execute_qlif_layer(
        self,
        pre_spikes: np.ndarray,
        weights: torch.Tensor,
        theta_membrane: torch.Tensor,
        refractory_state: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Execute a layer of QLIF neurons.

        Delegates to :func:`_qlif_layers.execute_qlif_layer`.
        """
        return execute_qlif_layer(
            pre_spikes,
            weights,
            theta_membrane,
            refractory_state,
            backend=self._get_backend(),
            shots=self.shots,
            threshold=self.threshold,
            refractory_period=self.refractory_period,
            leak_angle=self.leak_angle,
        )

    def _execute_qlif_layer_differentiable(
        self,
        pre_spikes: torch.Tensor,
        weights: torch.Tensor,
        theta_membrane: torch.Tensor,
        refractory_state: np.ndarray,
    ) -> tuple[torch.Tensor, np.ndarray]:
        """Execute a QLIF layer with surrogate gradient support.

        Delegates to :func:`_qlif_layers.execute_qlif_layer_differentiable`.
        """
        return execute_qlif_layer_differentiable(
            pre_spikes,
            weights,
            theta_membrane,
            refractory_state,
            backend=self._get_backend(),
            shots=self.shots,
            threshold=self.threshold,
            refractory_period=self.refractory_period,
            leak_angle=self.leak_angle,
            device=self.device,
        )

    def _execute_qlif_layer_differentiable_cached(
        self,
        pre_spikes: torch.Tensor,
        weights: torch.Tensor,
        theta_membrane: torch.Tensor,
        refractory_state: np.ndarray,
        cached_spike_probs: list[float],
    ) -> tuple[torch.Tensor, np.ndarray]:
        """Execute a QLIF layer using cached spike probabilities.

        Delegates to :func:`_qlif_layers.execute_qlif_layer_differentiable_cached`.
        """
        return execute_qlif_layer_differentiable_cached(
            pre_spikes,
            weights,
            theta_membrane,
            refractory_state,
            cached_spike_probs=cached_spike_probs,
            threshold=self.threshold,
            refractory_period=self.refractory_period,
            device=self.device,
        )

    def _encode_sensory_spikes(self, features: np.ndarray) -> np.ndarray:
        """Encode sensory features as spike probabilities.

        Delegates to :func:`_qlif_layers.encode_sensory_spikes`.
        """
        return encode_sensory_spikes(features, self.num_sensory)

    def _timestep(self, features: np.ndarray) -> np.ndarray:
        """Execute one timestep of QSNN dynamics.

        Propagates spikes through sensory→hidden→motor layers.
        Stores spike data for deferred eligibility accumulation
        (called after action selection in run_brain).

        Parameters
        ----------
        features : np.ndarray
            Input features from preprocessing.

        Returns
        -------
        np.ndarray
            Motor neuron firing probabilities.
        """
        # Encode sensory inputs
        sensory_spikes = self._encode_sensory_spikes(features)

        # Sensory → Hidden layer
        hidden_spikes, self.refractory_hidden = self._execute_qlif_layer(
            sensory_spikes,
            self.W_sh,
            self.theta_hidden,
            self.refractory_hidden,
        )

        # Hidden → Motor layer
        motor_spikes, self.refractory_motor = self._execute_qlif_layer(
            hidden_spikes,
            self.W_hm,
            self.theta_motor,
            self.refractory_motor,
        )

        # Store spike data for deferred eligibility accumulation.
        # Eligibility is accumulated after action selection so we can
        # apply action-specific credit assignment to W_hm.
        self._last_sensory_spikes = sensory_spikes
        self._last_hidden_spikes = hidden_spikes
        self._last_motor_spikes = motor_spikes

        return motor_spikes

    def _timestep_differentiable(self, features: np.ndarray) -> torch.Tensor:
        """Execute one timestep with gradient tracking for surrogate gradient mode.

        Same QLIF dynamics as _timestep(), but uses _execute_qlif_layer_differentiable()
        so that motor spike probabilities are torch tensors with grad_fn, enabling
        backpropagation through the weight matrices via surrogate gradients.

        Parameters
        ----------
        features : np.ndarray
            Input features from preprocessing.

        Returns
        -------
        torch.Tensor
            Motor neuron firing probabilities (with grad_fn for backprop).
        """
        sensory_spikes = self._encode_sensory_spikes(features)

        # Convert to tensor for differentiable computation
        sensory_tensor = torch.tensor(
            sensory_spikes,
            dtype=torch.float32,
            device=self.device,
        )

        # Sensory → Hidden (differentiable)
        hidden_spikes, self.refractory_hidden = self._execute_qlif_layer_differentiable(
            sensory_tensor,
            self.W_sh,
            self.theta_hidden,
            self.refractory_hidden,
        )

        # Hidden → Motor (differentiable)
        motor_spikes, self.refractory_motor = self._execute_qlif_layer_differentiable(
            hidden_spikes,
            self.W_hm,
            self.theta_motor,
            self.refractory_motor,
        )

        return motor_spikes

    def _timestep_differentiable_cached(
        self,
        features: np.ndarray,
        cached_step: dict[str, list[float]],
    ) -> torch.Tensor:
        """Execute one timestep using cached spike probabilities.

        Same as _timestep_differentiable() but uses pre-cached quantum spike
        probabilities, recomputing ry_angles from current weights for correct
        surrogate gradients.

        Parameters
        ----------
        features : np.ndarray
            Input features from preprocessing.
        cached_step : dict[str, list[float]]
            Cached spike probs for this integration step: {'hidden': [...], 'motor': [...]}.

        Returns
        -------
        torch.Tensor
            Motor neuron firing probabilities (with grad_fn for backprop).
        """
        sensory_spikes = self._encode_sensory_spikes(features)
        sensory_tensor = torch.tensor(
            sensory_spikes,
            dtype=torch.float32,
            device=self.device,
        )

        hidden_spikes, self.refractory_hidden = self._execute_qlif_layer_differentiable_cached(
            sensory_tensor,
            self.W_sh,
            self.theta_hidden,
            self.refractory_hidden,
            cached_step["hidden"],
        )

        motor_spikes, self.refractory_motor = self._execute_qlif_layer_differentiable_cached(
            hidden_spikes,
            self.W_hm,
            self.theta_motor,
            self.refractory_motor,
            cached_step["motor"],
        )

        return motor_spikes

    def _multi_timestep(self, features: np.ndarray) -> np.ndarray:
        """Execute multiple QLIF timesteps and average motor spike probabilities.

        Runs num_integration_steps timesteps with the same input, averaging
        motor spike probabilities across steps. This reduces quantum shot noise
        variance by a factor of num_integration_steps, giving REINFORCE cleaner
        gradient signal.

        Refractory state accumulates across timesteps within a single decision,
        providing natural temporal dynamics.

        Parameters
        ----------
        features : np.ndarray
            Input features from preprocessing.

        Returns
        -------
        np.ndarray
            Averaged motor neuron firing probabilities.
        """
        motor_accumulator = np.zeros(self.num_motor)
        for _ in range(self.num_integration_steps):
            motor_spikes = self._timestep(features)
            motor_accumulator += motor_spikes
        return motor_accumulator / self.num_integration_steps

    def _multi_timestep_differentiable(self, features: np.ndarray) -> torch.Tensor:
        """Execute multiple QLIF timesteps with gradient tracking and average.

        Same as _multi_timestep() but uses _timestep_differentiable() so that
        gradients flow through the averaged motor spike probabilities back to
        weights and theta parameters.

        Parameters
        ----------
        features : np.ndarray
            Input features from preprocessing.

        Returns
        -------
        torch.Tensor
            Averaged motor neuron firing probabilities (with grad_fn).
        """
        motor_accumulator = torch.zeros(self.num_motor, device=self.device)
        for _ in range(self.num_integration_steps):
            motor_spikes = self._timestep_differentiable(features)
            motor_accumulator = motor_accumulator + motor_spikes
        return motor_accumulator / self.num_integration_steps

    def _timestep_differentiable_caching(
        self,
        features: np.ndarray,
        cache_out: dict[str, list[float]],
    ) -> torch.Tensor:
        """Execute one timestep with gradient tracking, caching spike probs.

        Same as _timestep_differentiable() but captures the quantum spike
        probabilities from each layer for reuse in subsequent epochs.

        Parameters
        ----------
        features : np.ndarray
            Input features from preprocessing.
        cache_out : dict[str, list[float]]
            Mutable dict to store spike probs: {'hidden': [...], 'motor': [...]}.

        Returns
        -------
        torch.Tensor
            Motor neuron firing probabilities (with grad_fn for backprop).
        """
        sensory_spikes = self._encode_sensory_spikes(features)
        sensory_tensor = torch.tensor(
            sensory_spikes,
            dtype=torch.float32,
            device=self.device,
        )

        hidden_spikes, self.refractory_hidden = self._execute_qlif_layer_differentiable(
            sensory_tensor,
            self.W_sh,
            self.theta_hidden,
            self.refractory_hidden,
        )
        cache_out["hidden"] = hidden_spikes.detach().cpu().tolist()

        motor_spikes, self.refractory_motor = self._execute_qlif_layer_differentiable(
            hidden_spikes,
            self.W_hm,
            self.theta_motor,
            self.refractory_motor,
        )
        cache_out["motor"] = motor_spikes.detach().cpu().tolist()

        return motor_spikes

    def _multi_timestep_differentiable_caching(
        self,
        features: np.ndarray,
        cache_out: list[dict[str, list[float]]],
    ) -> torch.Tensor:
        """Execute multiple QLIF timesteps with gradient tracking, caching spike probs.

        Same as _multi_timestep_differentiable() but captures quantum spike
        probabilities at each integration step for reuse in subsequent epochs.

        Parameters
        ----------
        features : np.ndarray
            Input features from preprocessing.
        cache_out : list[dict[str, list[float]]]
            Mutable list to append per-integration-step cache entries.
            Each entry: {'hidden': [float, ...], 'motor': [float, ...]}.

        Returns
        -------
        torch.Tensor
            Averaged motor neuron firing probabilities (with grad_fn).
        """
        motor_accumulator = torch.zeros(self.num_motor, device=self.device)
        for _ in range(self.num_integration_steps):
            step_cache: dict[str, list[float]] = {}
            motor_spikes = self._timestep_differentiable_caching(features, step_cache)
            motor_accumulator = motor_accumulator + motor_spikes
            cache_out.append(step_cache)

        return motor_accumulator / self.num_integration_steps

    def _multi_timestep_differentiable_cached(
        self,
        features: np.ndarray,
        cached_timestep: list[dict[str, list[float]]],
    ) -> torch.Tensor:
        """Execute multiple QLIF timesteps using cached spike probabilities.

        Same as _multi_timestep_differentiable() but uses pre-cached quantum
        spike probabilities from epoch 0, recomputing ry_angles from current
        weights for correct surrogate gradients.

        Parameters
        ----------
        features : np.ndarray
            Input features from preprocessing.
        cached_timestep : list[dict[str, list[float]]]
            Per-integration-step cache entries from epoch 0.

        Returns
        -------
        torch.Tensor
            Averaged motor neuron firing probabilities (with grad_fn).
        """
        motor_accumulator = torch.zeros(self.num_motor, device=self.device)
        for step_idx in range(self.num_integration_steps):
            motor_spikes = self._timestep_differentiable_cached(
                features,
                cached_timestep[step_idx],
            )
            motor_accumulator = motor_accumulator + motor_spikes
        return motor_accumulator / self.num_integration_steps

    def _accumulate_eligibility(
        self,
        sensory_spikes: np.ndarray,
        hidden_spikes: np.ndarray,
        motor_spikes: np.ndarray,
        chosen_action: int,
    ) -> None:
        """Accumulate eligibility traces for local learning.

        Each step, existing traces decay by gamma before new correlations
        are added. Spike probabilities are centered at the firing threshold
        before computing outer products, producing mixed-sign eligibility
        traces that enable directional weight updates (neurons firing above
        threshold get positive eligibility, below get negative).

        **Action-specific credit assignment**: Only the chosen action's
        column in eligibility_hm is updated each step. This prevents all
        columns from receiving identical correlated updates (which caused
        column direction collapse in R2-R10). Each motor neuron column
        accumulates eligibility only when its action is selected, giving
        each column independent training signal over time.

        Parameters
        ----------
        sensory_spikes : np.ndarray
            Sensory layer spike probabilities.
        hidden_spikes : np.ndarray
            Hidden layer spike probabilities.
        motor_spikes : np.ndarray
            Motor layer spike probabilities.
        chosen_action : int
            Index of the action selected this step.
        """
        # Decay existing traces so recent steps matter more
        self.eligibility_sh *= self.gamma
        self.eligibility_hm *= self.gamma
        self.eligibility_theta_hidden *= self.gamma
        self.eligibility_theta_motor *= self.gamma

        # Center spike probabilities at firing threshold before computing
        # outer products. Raw spikes are in [0, 1] and always positive,
        # which produces all-positive eligibility traces — meaning every
        # weight gets pushed in the same direction. Centering at threshold
        # gives values in [-0.5, +0.5], so neurons firing above threshold
        # contribute positive eligibility and below contribute negative.
        # This enables the learning rule to develop directional structure
        # in W_sh (e.g., strengthen connections from active sensory neurons
        # to responsive hidden neurons while weakening others).
        pre_sh = torch.tensor(
            sensory_spikes - self.threshold,
            dtype=torch.float32,
            device=self.device,
        )
        post_sh = torch.tensor(
            hidden_spikes - self.threshold,
            dtype=torch.float32,
            device=self.device,
        )
        self.eligibility_sh += torch.outer(pre_sh, post_sh)

        # Action-specific eligibility for W_hm: only update the chosen
        # action's column. This is the key difference from standard Hebbian
        # learning — in REINFORCE, only the chosen action's parameters are
        # updated. By restricting eligibility to the chosen column, each
        # motor neuron develops independent weight structure based on the
        # outcomes when IT was selected, preventing column convergence.
        pre_hm = torch.tensor(
            hidden_spikes - self.threshold,
            dtype=torch.float32,
            device=self.device,
        )
        motor_centered = float(motor_spikes[chosen_action] - self.threshold)
        self.eligibility_hm[:, chosen_action] += pre_hm * motor_centered

        # Theta eligibility: W_sh uses all hidden neurons, W_hm only chosen action
        self.eligibility_theta_hidden += post_sh
        self.eligibility_theta_motor[chosen_action] += motor_centered

    @staticmethod
    def _normalize_trace(
        trace: torch.Tensor,
        max_norm: float = MAX_ELIGIBILITY_NORM,
    ) -> torch.Tensor:
        """Normalize an eligibility trace to cap its Frobenius/L2 norm.

        Preserves direction while bounding magnitude so that weight deltas
        stay proportional to lr * max_norm * advantage regardless of how
        large the raw trace accumulated during the update interval.

        Parameters
        ----------
        trace : torch.Tensor
            Raw eligibility trace (matrix or vector).
        max_norm : float
            Maximum allowed norm.

        Returns
        -------
        torch.Tensor
            Normalized trace (same shape).
        """
        norm = torch.norm(trace)
        if norm > max_norm:
            return trace * (max_norm / norm)
        return trace

    def _local_learning_update(self, total_reward: float) -> None:
        """Apply local learning weight update with baseline advantage.

        dw = lr * normalize(eligibility) * advantage - weight_decay * w

        Eligibility traces are normalized before computing deltas to bound
        the maximum weight update magnitude. This prevents weights from
        blowing past tanh's useful range regardless of trace accumulation.

        Parameters
        ----------
        total_reward : float
            Total discounted reward for the episode.
        """
        # Compute advantage relative to baseline
        advantage = total_reward - self.baseline

        # Update baseline (exponential moving average)
        self.baseline = (
            1 - self.baseline_alpha
        ) * self.baseline + self.baseline_alpha * total_reward

        # Log raw eligibility norms before normalization
        raw_sh_norm = torch.norm(self.eligibility_sh).item()
        raw_hm_norm = torch.norm(self.eligibility_hm).item()

        # Normalize eligibility traces — caps update magnitude while preserving direction
        norm_elig_sh = self._normalize_trace(self.eligibility_sh)
        norm_elig_hm = self._normalize_trace(self.eligibility_hm)
        norm_elig_theta_h = self._normalize_trace(self.eligibility_theta_hidden)
        norm_elig_theta_m = self._normalize_trace(self.eligibility_theta_motor)

        # Compute weight updates using normalized traces
        delta_sh = self.learning_rate * norm_elig_sh * advantage
        delta_hm = self.learning_rate * norm_elig_hm * advantage

        # Compute theta updates
        delta_theta_hidden = self.learning_rate * norm_elig_theta_h * advantage
        delta_theta_motor = self.learning_rate * norm_elig_theta_m * advantage

        # Apply updates with L2 weight decay for soft stability
        self.W_sh += delta_sh - WEIGHT_DECAY_FACTOR * self.W_sh
        self.W_hm += delta_hm - WEIGHT_DECAY_FACTOR * self.W_hm

        # Update theta parameters (membrane potential biases)
        self.theta_hidden += delta_theta_hidden - WEIGHT_DECAY_FACTOR * self.theta_hidden
        self.theta_motor += delta_theta_motor - WEIGHT_DECAY_FACTOR * self.theta_motor

        # Clamp theta_motor L2 norm to prevent death spiral
        tm_norm = torch.norm(self.theta_motor)
        if tm_norm > self.config.theta_motor_max_norm:
            self.theta_motor.mul_(self.config.theta_motor_max_norm / tm_norm)

        # Log update statistics
        col_diffs = self._compute_column_diversity()
        logger.debug(
            f"QSNN local learning: reward={total_reward:.4f}, "
            f"advantage={advantage:.4f}, baseline={self.baseline:.4f}, "
            f"raw_elig_sh_norm={raw_sh_norm:.4f}, "
            f"raw_elig_hm_norm={raw_hm_norm:.4f}, "
            f"norm_elig_sh_norm={torch.norm(norm_elig_sh).item():.4f}, "
            f"norm_elig_hm_norm={torch.norm(norm_elig_hm).item():.4f}, "
            f"delta_sh_norm={torch.norm(delta_sh).item():.4f}, "
            f"delta_hm_norm={torch.norm(delta_hm).item():.4f}, "
            f"theta_hidden_norm={torch.norm(self.theta_hidden).item():.4f}, "
            f"theta_motor_norm={torch.norm(self.theta_motor).item():.4f}, "
            f"W_hm_col_diversity={col_diffs:.4f}",
        )

    def _last_entropy(self) -> float:
        """Compute entropy of the most recent action probability distribution.

        Returns ln(num_actions) if no probabilities are stored (maximum entropy).
        """
        if self.current_probabilities is None:
            return float(np.log(self.num_actions))
        probs = np.clip(self.current_probabilities, 1e-10, 1.0)
        return float(-np.sum(probs * np.log(probs)))

    def _compute_column_diversity(self) -> float:
        """Compute mean pairwise L2 distance between W_hm columns.

        Returns 0.0 when all columns are identical (collapsed state).
        """
        cols = self.W_hm.T  # shape: (num_motor, num_hidden)
        n = cols.shape[0]
        total_dist = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_dist += torch.norm(cols[i] - cols[j]).item()
                count += 1
        return total_dist / max(count, 1)

    def _exploration_schedule(self) -> tuple[float, float]:
        """Compute current exploration parameters based on episode count.

        Linearly decays epsilon and temperature over EXPLORATION_DECAY_EPISODES
        episodes, allowing more exploitation as the policy matures.

        Returns
        -------
        tuple[float, float]
            (current_epsilon, current_temperature)
        """
        progress = min(1.0, self._episode_count / max(self.config.exploration_decay_episodes, 1))
        # Epsilon decays from initial to 30% of initial (still ensures min 0.75% per action)
        current_epsilon = self.config.exploration_epsilon * (1.0 - progress * 0.7)
        # Temperature decays from 1.5 to 1.0 (sharper softmax over time)
        current_temperature = 1.5 - 0.5 * progress
        return current_epsilon, current_temperature

    def _current_lr(self) -> float:
        """Get the current learning rate from the optimizer (or config default)."""
        if self.optimizer is not None:
            return float(self.optimizer.param_groups[0]["lr"])
        return self.learning_rate

    def _normalize_reward(self, reward: float) -> float:
        """Normalize a reward using running mean and variance.

        Updates running statistics with EMA and returns the normalized reward.
        This makes REINFORCE learning invariant to reward magnitude, preventing
        negative reward dominance where large penalties drown food signals.

        Parameters
        ----------
        reward : float
            Raw reward from the environment.

        Returns
        -------
        float
            Normalized reward: (reward - running_mean) / (running_std + 1e-8).
        """
        alpha = self.config.reward_norm_alpha
        # Update running mean
        self.reward_running_mean = (1 - alpha) * self.reward_running_mean + alpha * reward
        # Update running variance
        diff = reward - self.reward_running_mean
        self.reward_running_var = (1 - alpha) * self.reward_running_var + alpha * diff * diff
        # Normalize
        running_std = np.sqrt(self.reward_running_var)
        return (reward - self.reward_running_mean) / (running_std + 1e-8)

    def _compute_gae(self, bootstrap_value: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns from critic value estimates.

        Parameters
        ----------
        bootstrap_value : float
            V(s) estimate for the state after the last step in the window.
            0.0 for terminal states, critic estimate for intra-episode boundaries.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (advantages, returns) tensors of shape (num_steps,).
        """
        num_steps = len(self.episode_rewards)
        advantages = torch.zeros(num_steps, device=self.device)
        last_gae = 0.0

        for t in reversed(range(num_steps)):
            next_value = bootstrap_value if t == num_steps - 1 else self.episode_values[t + 1]
            delta = self.episode_rewards[t] + self.gamma * next_value - self.episode_values[t]
            advantages[t] = last_gae = delta + self.gamma * self.config.gae_lambda * last_gae

        values_t = torch.tensor(self.episode_values, dtype=torch.float32, device=self.device)
        returns = advantages + values_t
        return advantages, returns

    def _update_critic(self, target_returns: torch.Tensor) -> None:
        """Update critic MLP via MSE loss against target returns.

        Parameters
        ----------
        target_returns : torch.Tensor
            GAE returns to regress critic predictions toward.
        """
        if self.critic is None or self.critic_optimizer is None:
            return

        num_steps = len(self.episode_features)
        features_batch = torch.tensor(
            np.array(self.episode_features[:num_steps]),
            dtype=torch.float32,
            device=self.device,
        )
        predicted = self.critic(features_batch)
        # Huber loss (smooth L1) is robust to extreme target returns from death penalties.
        # MSE amplifies outliers quadratically; Huber is linear for |error| > 1.0,
        # capping gradient magnitude and preventing critic destabilization.
        value_loss = torch.nn.functional.smooth_l1_loss(predicted, target_returns.detach())

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=SURROGATE_GRAD_CLIP)
        self.critic_optimizer.step()

        logger.debug(
            f"QSNN critic: value_loss={value_loss.item():.4f}, "
            f"pred_mean={predicted.mean().item():.4f}, "
            f"target_mean={target_returns.mean().item():.4f}",
        )

    def _compute_reinforce_advantages(
        self,
        num_steps: int,
        bootstrap_value: float,
    ) -> tuple[torch.Tensor, float, torch.Tensor, torch.Tensor | None]:
        """Compute advantages for REINFORCE update.

        Returns (advantages, raw_mean, returns_tensor, returns_for_critic).
        """
        returns_for_critic: torch.Tensor | None = None
        if self.use_critic and self.critic is not None and len(self.episode_values) == num_steps:
            advantages, returns_for_critic = self._compute_gae(bootstrap_value)
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            advantages = torch.clamp(
                advantages,
                -self.config.advantage_clip,
                self.config.advantage_clip,
            )
            raw_mean = returns_for_critic.mean().item()
            returns_tensor = returns_for_critic
        else:
            returns: list[float] = []
            discounted_return = 0.0
            for reward in reversed(self.episode_rewards):
                discounted_return = reward + self.gamma * discounted_return
                returns.insert(0, discounted_return)

            returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
            raw_mean = returns_tensor.mean().item()

            self.baseline = (
                1 - self.baseline_alpha
            ) * self.baseline + self.baseline_alpha * raw_mean

            if len(returns_tensor) > 1:
                advantages = (returns_tensor - returns_tensor.mean()) / (
                    returns_tensor.std() + 1e-8
                )
            else:
                advantages = returns_tensor - self.baseline

            advantages = torch.clamp(
                advantages,
                -self.config.advantage_clip,
                self.config.advantage_clip,
            )

        return advantages, raw_mean, returns_tensor, returns_for_critic

    def _run_reinforce_epoch(
        self,
        num_steps: int,
        advantages: torch.Tensor,
        old_log_probs_t: torch.Tensor,
        stats: _ReinforceUpdateStats,
    ) -> None:
        """Run a single REINFORCE gradient epoch with PPO clipping."""
        self.refractory_hidden.fill(0)
        self.refractory_motor.fill(0)

        log_probs_list: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []

        for t in range(num_steps):
            features = self.episode_features[t]
            action_idx = self.episode_actions[t]

            if stats.num_epochs > 1 and stats.epoch == 0:
                timestep_cache: list[dict[str, list[float]]] = []
                motor_spikes = self._multi_timestep_differentiable_caching(
                    features,
                    timestep_cache,
                )
                self._cached_spike_probs.append(timestep_cache)
            elif stats.epoch > 0:
                motor_spikes = self._multi_timestep_differentiable_cached(
                    features,
                    self._cached_spike_probs[t],
                )
            else:
                motor_spikes = self._multi_timestep_differentiable(features)

            motor_clipped = torch.clamp(motor_spikes, 1e-8, 1.0 - 1e-8)
            logits = (motor_clipped - 0.5) * self.config.logit_scale

            epsilon, temperature = self._exploration_schedule()
            softmax_probs = torch.softmax(logits / temperature, dim=-1)
            uniform = torch.ones_like(softmax_probs) / self.num_actions
            action_probs = (1 - epsilon) * softmax_probs + epsilon * uniform

            log_prob = torch.log(action_probs[action_idx] + 1e-8)
            log_probs_list.append(log_prob)

            entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-10))
            entropies.append(entropy)

        log_probs = torch.stack(log_probs_list)
        mean_entropy = torch.stack(entropies).mean()
        effective_entropy_coef = self._adaptive_entropy_coef(mean_entropy.item())

        ratio = torch.exp(log_probs - old_log_probs_t)
        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(
                ratio,
                1.0 - self.config.clip_epsilon,
                1.0 + self.config.clip_epsilon,
            )
            * advantages
        )
        policy_loss = -torch.min(surr1, surr2).mean() - effective_entropy_coef * mean_entropy

        if self.optimizer is not None:
            self.optimizer.zero_grad()
        policy_loss.backward()

        self._step_optimizer_and_log(policy_loss, mean_entropy, advantages, stats)

    def _reinforce_update(self, bootstrap_value: float = 0.0) -> None:
        """REINFORCE policy gradient update with surrogate gradients.

        Recomputes forward passes with gradient tracking enabled, computes
        policy loss from log-probabilities and advantages, and backpropagates
        through the QLIFSurrogateSpike function to update weights.

        This requires episode_features to be stored during run_brain().
        """
        if len(self.episode_rewards) == 0 or self.optimizer is None:
            return

        num_steps = len(self.episode_rewards)

        if num_steps < _MIN_REINFORCE_BATCH_SIZE:
            logger.debug(
                "QSNN: skipping degenerate single-step update "
                f"(reward={self.episode_rewards[0]:.4f})",
            )
            self.episode_rewards.clear()
            self.episode_features.clear()
            self.episode_actions.clear()
            self.episode_old_log_probs.clear()
            if self.use_critic:
                self.episode_values.clear()
            self._cached_spike_probs = []
            return

        advantages, raw_mean, returns_tensor, returns_for_critic = (
            self._compute_reinforce_advantages(num_steps, bootstrap_value)
        )

        num_epochs = self.config.num_reinforce_epochs
        self._cached_spike_probs = []

        old_log_probs_t = torch.tensor(
            self.episode_old_log_probs[:num_steps],
            dtype=torch.float32,
            device=self.device,
        )

        for epoch in range(num_epochs):
            self._run_reinforce_epoch(
                num_steps,
                advantages,
                old_log_probs_t,
                _ReinforceUpdateStats(
                    raw_mean=raw_mean,
                    returns_tensor=returns_tensor,
                    epoch=epoch,
                    num_epochs=num_epochs,
                ),
            )

        self._clamp_weights()

        # Update critic if enabled (once after all actor epochs)
        if returns_for_critic is not None and self.use_critic and self.critic is not None:
            self._update_critic(returns_for_critic)

    def _adaptive_entropy_coef(self, entropy_val: float) -> float:
        """Compute effective entropy coefficient with two-sided regulation.

        Two-sided entropy regulation:
        - Floor: when entropy drops below ENTROPY_FLOOR, scale entropy_coef up
          to push the policy back toward exploration (prevents collapse).
        - Ceiling: when entropy exceeds ENTROPY_CEILING_FRACTION of max,
          suppress entropy bonus so policy gradient can sharpen the policy
          (prevents drift to uniform random).
        """
        max_entropy = np.log(self.num_actions)
        entropy_ceiling = self.config.entropy_ceiling_fraction * max_entropy
        if entropy_val < self.config.entropy_floor:
            ratio = 1.0 - entropy_val / self.config.entropy_floor
            entropy_scale = 1.0 + ratio * (self.config.entropy_boost_max - 1.0)
        elif entropy_val > entropy_ceiling:
            ratio = (entropy_val - entropy_ceiling) / (max_entropy - entropy_ceiling)
            entropy_scale = max(0.0, 1.0 - ratio)
        else:
            entropy_scale = 1.0
        effective_coef = self.entropy_coef * entropy_scale
        if entropy_scale != 1.0:
            side = "FLOOR" if entropy_scale > 1.0 else "CEILING"
            threshold = self.config.entropy_floor if entropy_scale > 1.0 else entropy_ceiling
            logger.debug(
                f"QSNN adaptive entropy {side}: entropy={entropy_val:.4f}, "
                f"threshold={threshold:.4f}, scale={entropy_scale:.2f}x, "
                f"effective_coef={effective_coef:.4f}",
            )
        return effective_coef

    def _step_optimizer_and_log(
        self,
        policy_loss: torch.Tensor,
        mean_entropy: torch.Tensor,
        advantages: torch.Tensor,
        stats: _ReinforceUpdateStats,
    ) -> None:
        """Clip gradients, step optimizer, and log diagnostics (per-epoch).

        Weight clamping is NOT done here — it must be applied once after all
        epochs via _clamp_weights() to avoid discarding gradient information
        through repeated clamping at boundaries.
        """
        if self.optimizer is None:
            return
        params_list = [self.W_sh, self.W_hm, self.theta_hidden, self.theta_motor]
        grad_norm = torch.nn.utils.clip_grad_norm_(params_list, max_norm=SURROGATE_GRAD_CLIP)

        # Per-parameter gradient norms (after clipping)
        grad_norms = [p.grad.norm().item() if p.grad is not None else 0.0 for p in params_list]

        w_sh_pre = torch.norm(self.W_sh).item()
        w_hm_pre = torch.norm(self.W_hm).item()

        self.optimizer.step()

        w_sh_post = torch.norm(self.W_sh).item()
        w_hm_post = torch.norm(self.W_hm).item()
        col_div = self._compute_column_diversity()
        epsilon, temperature = self._exploration_schedule()

        epoch_str = f"epoch={stats.epoch}/{stats.num_epochs}, " if stats.num_epochs > 1 else ""
        returns_std = stats.returns_tensor.std().item() if len(stats.returns_tensor) > 1 else 0.0
        adv_std = advantages.std().item() if len(advantages) > 1 else 0.0
        logger.debug(
            f"QSNN surrogate gradient update: {epoch_str}loss={policy_loss.item():.4f}, "
            f"mean_entropy={mean_entropy.item():.4f}, "
            f"W_sh_norm={w_sh_post:.4f}, W_hm_norm={w_hm_post:.4f}, "
            f"W_sh_delta={w_sh_post - w_sh_pre:.6f}, "
            f"W_hm_delta={w_hm_post - w_hm_pre:.6f}, "
            f"W_hm_col_div={col_div:.4f}, "
            f"theta_h_norm={torch.norm(self.theta_hidden).item():.4f}, "
            f"theta_m_norm={torch.norm(self.theta_motor).item():.4f}, "
            f"grad_total={grad_norm.item():.4f}, "
            f"grad_sh={grad_norms[0]:.4f}, grad_hm={grad_norms[1]:.4f}, "
            f"grad_th={grad_norms[2]:.4f}, grad_tm={grad_norms[3]:.4f}, "
            f"returns_mean={stats.raw_mean:.4f}, "
            f"returns_std={returns_std:.4f}, "
            f"adv_mean={advantages.mean().item():.4f}, "
            f"adv_std={adv_std:.4f}, "
            f"epsilon={epsilon:.4f}, temperature={temperature:.4f}, "
            f"lr={self.optimizer.param_groups[0]['lr']:.6f}, "
            f"episode={self._episode_count}",
        )

    def _clamp_weights(self) -> None:
        """Clamp all weights to [-weight_clip, weight_clip] and theta_motor L2 norm.

        Called once after all REINFORCE epochs to bound weights without
        discarding gradient information through repeated per-epoch clamping.
        """
        params_list = [self.W_sh, self.W_hm, self.theta_hidden, self.theta_motor]
        with torch.no_grad():
            for p in params_list:
                p.clamp_(-self.weight_clip, self.weight_clip)

            # Clamp theta_motor L2 norm to prevent death spiral
            tm_norm = torch.norm(self.theta_motor)
            if tm_norm > self.config.theta_motor_max_norm:
                self.theta_motor.mul_(self.config.theta_motor_max_norm / tm_norm)

    def preprocess(self, params: BrainParams) -> np.ndarray:
        """Preprocess BrainParams to extract features.

        Two modes:
        1. **Unified sensory mode** (when sensory_modules is set)
        2. **Legacy mode** (default): gradient strength + relative angle

        Parameters
        ----------
        params : BrainParams
            Brain parameters containing sensory information.

        Returns
        -------
        np.ndarray
            Preprocessed features.
        """
        # Unified sensory mode
        if self.sensory_modules is not None:
            return extract_classical_features(params, self.sensory_modules)

        # Legacy mode: 2-feature preprocessing
        grad_strength = float(params.gradient_strength or 0.0)

        grad_direction = float(params.gradient_direction or 0.0)
        direction_map = {
            Direction.UP: np.pi / 2,
            Direction.DOWN: -np.pi / 2,
            Direction.LEFT: np.pi,
            Direction.RIGHT: 0.0,
        }
        agent_facing_angle = direction_map.get(params.agent_direction or Direction.UP, np.pi / 2)
        relative_angle = (grad_direction - agent_facing_angle + np.pi) % (2 * np.pi) - np.pi
        rel_angle_norm = relative_angle / np.pi

        return np.array([grad_strength, rel_angle_norm], dtype=np.float32)

    def _log_motor_dynamics(
        self,
        features: np.ndarray,
        motor_probs: np.ndarray,
        logits: np.ndarray,
        action_probs: np.ndarray,
        action_entropy: float,
    ) -> None:
        """Log motor layer dynamics for debugging (called periodically)."""
        col_div = self._compute_column_diversity()
        sensory = self._encode_sensory_spikes(features)
        w_sh_np = self.W_sh.detach().cpu().numpy()
        w_hm_np = self.W_hm.detach().cpu().numpy()
        wi_sh = sensory @ w_sh_np
        wi_hm_max = float(np.abs(w_hm_np).sum(axis=0).max())
        logger.debug(
            f"QSNN step {self._step_count}: "
            f"features={features}, "
            f"motor_spikes={np.array2string(motor_probs, precision=3)}, "
            f"logits={np.array2string(logits, precision=2)}, "
            f"action_probs={np.array2string(action_probs, precision=3)}, "
            f"entropy={action_entropy:.3f}, "
            f"W_sh_norm={torch.norm(self.W_sh).item():.3f}, "
            f"W_hm_norm={torch.norm(self.W_hm).item():.3f}, "
            f"W_hm_col_div={col_div:.3f}, "
            f"wi_sh_range=[{wi_sh.min():.2f},{wi_sh.max():.2f}], "
            f"wi_hm_max_possible={wi_hm_max:.2f}, "
            f"theta_h_norm={torch.norm(self.theta_hidden).item():.3f}, "
            f"theta_m_norm={torch.norm(self.theta_motor).item():.3f}",
        )

    def run_brain(
        self,
        params: BrainParams,
        reward: float | None = None,  # noqa: ARG002
        input_data: list[float] | None = None,  # noqa: ARG002
        *,
        top_only: bool,  # noqa: ARG002
        top_randomize: bool,  # noqa: ARG002
    ) -> list[ActionData]:
        """Run the QSNN brain and select an action.

        Parameters
        ----------
        params : BrainParams
            Brain parameters containing sensory information.
        reward : float | None
            Current reward (unused).
        input_data : list[float] | None
            Optional input data (unused).
        top_only : bool
            Whether to return only top action (unused).
        top_randomize : bool
            Whether to randomize selection (unused).

        Returns
        -------
        list[ActionData]
            List containing the selected action.
        """
        # Preprocess inputs
        features = self.preprocess(params)

        # Handle deferred critic update: at intra-episode window boundaries,
        # the update was deferred until this run_brain() call so we can compute
        # V(s_{N+1}) — the value of the current (new) state as GAE bootstrap.
        if self._pending_critic_update and self.use_critic and self.critic is not None:
            with torch.no_grad():
                bootstrap_features = torch.tensor(
                    features,
                    dtype=torch.float32,
                    device=self.device,
                )
                bootstrap_value = self.critic(bootstrap_features).item()
            self._reinforce_update(bootstrap_value=bootstrap_value)
            self.episode_rewards.clear()
            self.episode_actions.clear()
            self.episode_features.clear()
            self.episode_old_log_probs.clear()
            self.episode_values.clear()
            self._cached_spike_probs = []
            self._pending_critic_update = False

        # Execute QSNN timestep(s) to get motor firing probabilities.
        # Multi-timestep integration averages out quantum shot noise.
        motor_probs = self._multi_timestep(features)

        # Convert spike probabilities [0,1] to action logits
        # Shift from [0,1] to centered range, then scale to create spread
        motor_probs = np.clip(motor_probs, 1e-8, 1.0 - 1e-8)
        logits = (motor_probs - 0.5) * self.config.logit_scale

        # Exploration schedule: epsilon and temperature decay over episodes
        epsilon, temperature = self._exploration_schedule()

        # Softmax with temperature scaling
        scaled_logits = logits / temperature
        exp_probs = np.exp(scaled_logits - np.max(scaled_logits))
        softmax_probs = exp_probs / np.sum(exp_probs)

        # Exploration floor: mix with uniform to prevent policy collapse.
        # Guarantees every action has at least epsilon/num_actions probability,
        # breaking the positive feedback loop where one dominant action
        # accumulates all eligibility and locks the policy.
        uniform = np.ones(self.num_actions) / self.num_actions
        action_probs = (1 - epsilon) * softmax_probs + epsilon * uniform

        # Compute action entropy for collapse detection
        action_entropy = -np.sum(action_probs * np.log(action_probs + 1e-10))

        # Log motor dynamics periodically for debugging
        if self._step_count % 50 == 0:
            self._log_motor_dynamics(
                features,
                motor_probs,
                logits,
                action_probs,
                action_entropy,
            )

        # Sample action from categorical distribution
        action_idx = self.rng.choice(self.num_actions, p=action_probs)
        action_name = self.action_set[action_idx]

        # Update tracking data
        self.latest_data.action = ActionData(
            state=action_name,
            action=action_name,
            probability=action_probs[action_idx],
        )

        # Accumulate eligibility traces with action-specific credit assignment.
        # Must happen after action selection so we know which W_hm column to update.
        if self.use_local_learning and self.training:
            self._accumulate_eligibility(
                self._last_sensory_spikes,
                self._last_hidden_spikes,
                self._last_motor_spikes,
                action_idx,
            )

        # Store features and old log-probs for surrogate gradient recomputation
        if not self.use_local_learning:
            self.episode_features.append(features)
            old_log_prob = float(np.log(action_probs[action_idx] + 1e-8))
            self.episode_old_log_probs.append(old_log_prob)

        # Store critic value estimate for GAE computation
        if self.use_critic and self.critic is not None:
            with torch.no_grad():
                features_t = torch.tensor(features, dtype=torch.float32, device=self.device)
                value = self.critic(features_t).item()
            self.episode_values.append(value)

        # Store for learning
        self.episode_actions.append(action_idx)

        self.current_probabilities = action_probs
        self.latest_data.probability = float(action_probs[action_idx])
        self.history_data.actions.append(self.latest_data.action)
        self.history_data.probabilities.append(float(action_probs[action_idx]))

        return [self.latest_data.action]

    def learn(
        self,
        params: BrainParams,  # noqa: ARG002
        reward: float,
        *,
        episode_done: bool = False,
    ) -> None:
        """Accumulate rewards and trigger learning updates.

        For both local learning and surrogate gradient modes, intra-episode
        updates are applied every ``update_interval`` steps. At episode end,
        a final update is applied to any remaining steps.

        Parameters
        ----------
        params : BrainParams
            Brain parameters (unused).
        reward : float
            Reward for the current step.
        episode_done : bool
            Whether the episode has ended.
        """
        # Normalize reward if enabled, making REINFORCE invariant to penalty magnitude.
        # Raw reward is still logged in history_data for diagnostics.
        # When use_critic=True, skip normalization: the critic needs stationary raw rewards
        # to learn a stable V(s). Running EMA normalization makes targets non-stationary,
        # preventing critic convergence. GAE + advantage normalization handles variance
        # reduction in actor-critic mode instead.
        if self.config.use_reward_normalization and not self.use_critic:
            normalized_reward = self._normalize_reward(reward)
            self.episode_rewards.append(normalized_reward)
        else:
            self.episode_rewards.append(reward)
        self.history_data.rewards.append(reward)
        self._step_count += 1

        # Intra-episode updates: apply learning every update_interval steps
        if (
            self.use_local_learning
            and self.update_interval > 0
            and self._step_count % self.update_interval == 0
            and not episode_done
        ):
            recent_reward = sum(self.episode_rewards[-self.update_interval :])
            self._local_learning_update(recent_reward)
            # Reset eligibility after applying update (fresh accumulation)
            self.eligibility_sh.zero_()
            self.eligibility_hm.zero_()
            self.eligibility_theta_hidden.zero_()
            self.eligibility_theta_motor.zero_()
        elif (
            not self.use_local_learning
            and self.update_interval > 0
            and self._step_count % self.update_interval == 0
            and not episode_done
        ):
            # Intra-episode REINFORCE: update on the recent window of steps.
            # This gives much stronger credit assignment for recent events
            # (e.g. death signal within 20 steps: 0.99^19 * -3.0 = -2.45
            # vs episode-end: 0.99^199 * -3.0 = -0.40).
            if self.use_critic and self.critic is not None:
                # Defer update to next run_brain() where we can compute V(s_{N+1}).
                # At this point we only have V(s_N) (the last state in the window),
                # but GAE needs V(s_{N+1}) as bootstrap. The next run_brain() call
                # will have access to the new state's features for correct bootstrap.
                self._pending_critic_update = True
            else:
                # Vanilla REINFORCE: no bootstrap needed, update immediately
                self._reinforce_update(bootstrap_value=0.0)
                # Clear buffers so next window starts fresh
                self.episode_rewards.clear()
                self.episode_actions.clear()
                self.episode_features.clear()
                self.episode_old_log_probs.clear()
                self.episode_values.clear()
                self._cached_spike_probs = []

        if episode_done and len(self.episode_rewards) > 0:
            # Compute total discounted reward for final update
            total_reward = 0.0
            discount = 1.0
            for r in self.episode_rewards:
                total_reward += discount * r
                discount *= self.gamma

            epsilon, temperature = self._exploration_schedule()
            col_div = self._compute_column_diversity()
            logger.info(
                f"QSNN episode complete: episode={self._episode_count}, "
                f"steps={len(self.episode_rewards)}, "
                f"total_reward={total_reward:.4f}, "
                f"W_sh_norm={torch.norm(self.W_sh).item():.4f}, "
                f"W_hm_norm={torch.norm(self.W_hm).item():.4f}, "
                f"W_hm_col_div={col_div:.4f}, "
                f"theta_h_norm={torch.norm(self.theta_hidden).item():.4f}, "
                f"theta_m_norm={torch.norm(self.theta_motor).item():.4f}, "
                f"entropy={self._last_entropy():.4f}, "
                f"epsilon={epsilon:.4f}, temperature={temperature:.4f}, "
                f"lr={self._current_lr():.6f}, "
                f"reward_norm_mean={self.reward_running_mean:.4f}, "
                f"reward_norm_std={np.sqrt(self.reward_running_var):.4f}",
            )

            # Apply final learning update
            if self.use_local_learning:
                self._local_learning_update(total_reward)
            else:
                self._reinforce_update(bootstrap_value=0.0)  # terminal state

            # Increment episode counter (for exploration decay)
            self._episode_count += 1

            # Step LR scheduler (cosine annealing decay)
            if self.scheduler is not None:
                self.scheduler.step()

            # Reset episode state
            self._reset_episode()

    def update_memory(self, reward: float | None = None) -> None:
        """Update internal memory (no-op for QSNNReinforceBrain)."""

    def prepare_episode(self) -> None:
        """Prepare for a new episode."""
        self._reset_episode()

    def post_process_episode(self, *, episode_success: bool | None = None) -> None:
        """Post-process the episode (no-op, learning happens in learn())."""

    def _reset_episode(self) -> None:
        """Reset episode state."""
        self.episode_rewards.clear()
        self.episode_actions.clear()
        self.episode_features.clear()
        self.episode_old_log_probs.clear()
        self.episode_values.clear()
        self._pending_critic_update = False
        self._cached_spike_probs = []

        # Reset eligibility traces
        self.eligibility_sh.zero_()
        self.eligibility_hm.zero_()
        self.eligibility_theta_hidden.zero_()
        self.eligibility_theta_motor.zero_()

        # Reset refractory states
        self.refractory_hidden.fill(0)
        self.refractory_motor.fill(0)

        # Reset step counter
        self._step_count = 0

    def copy(self) -> QSNNReinforceBrain:
        """Create an independent copy of the QSNNReinforceBrain.

        Returns
        -------
        QSNNReinforceBrain
            Independent copy of this brain.
        """
        config_copy = QSNNReinforceBrainConfig(
            **{**self.config.model_dump(), "seed": self.seed},
        )

        new_brain = QSNNReinforceBrain(
            config=config_copy,
            num_actions=self.num_actions,
            device=DeviceType(self.device.type),
            action_set=self._action_set,
        )

        # Copy weights (independent copy, preserving requires_grad)
        new_brain.W_sh = self.W_sh.clone().detach().requires_grad_(self.W_sh.requires_grad)
        new_brain.W_hm = self.W_hm.clone().detach().requires_grad_(self.W_hm.requires_grad)
        new_brain.theta_hidden = (
            self.theta_hidden.clone().detach().requires_grad_(self.theta_hidden.requires_grad)
        )
        new_brain.theta_motor = (
            self.theta_motor.clone().detach().requires_grad_(self.theta_motor.requires_grad)
        )

        # Recreate optimizer and scheduler for surrogate gradient mode with new parameter refs
        if not self.use_local_learning:
            new_brain.optimizer = torch.optim.Adam(
                [new_brain.W_sh, new_brain.W_hm, new_brain.theta_hidden, new_brain.theta_motor],
                lr=self.learning_rate,
            )
            new_brain.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                new_brain.optimizer,
                T_max=self.config.lr_decay_episodes,
                eta_min=self.learning_rate * self.config.lr_min_factor,
            )
            # Advance scheduler to match current episode count
            for _ in range(self._episode_count):
                new_brain.scheduler.step()

        # Copy learning state
        new_brain.baseline = self.baseline
        new_brain._episode_count = self._episode_count
        new_brain.reward_running_mean = self.reward_running_mean
        new_brain.reward_running_var = self.reward_running_var
        new_brain._pending_critic_update = self._pending_critic_update

        # Copy critic if enabled
        if self.use_critic and self.critic is not None:
            new_brain.critic = CriticMLP(
                input_dim=self.input_dim,
                hidden_dim=self.config.critic_hidden_dim,
                num_layers=self.config.critic_num_layers,
            ).to(self.device)
            new_brain.critic.load_state_dict(self.critic.state_dict())
            new_brain.critic_optimizer = torch.optim.Adam(
                new_brain.critic.parameters(),
                lr=self.config.critic_lr,
            )

        return new_brain

    @property
    def action_set(self) -> list[Action]:
        """Get the list of available actions."""
        return self._action_set

    @action_set.setter
    def action_set(self, actions: list[Action]) -> None:
        """Set the list of available actions."""
        if len(actions) != self.num_actions:
            msg = (
                f"Cannot set action_set of length {len(actions)}: "
                f"network expects {self.num_actions} actions"
            )
            raise ValueError(msg)
        self._action_set = actions
