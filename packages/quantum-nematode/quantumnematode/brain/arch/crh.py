"""
Classical Reservoir Hybrid (CRH) Brain Architecture.

This architecture implements a fixed classical Echo State Network (ESN) reservoir
with configurable feature channels and a PPO-trained classical actor-critic readout.
It serves as both a quantum ablation control for QRH (matching feature dimension and
architecture) and a standalone benchmark architecture filling the "classical fixed
reservoir + PPO readout" niche.

Key Features:
- **Echo State Network Reservoir**: Fixed random weight matrices (W_in, W_res) with
  tanh nonlinearity and configurable spectral radius
- **Configurable Feature Channels**: raw, cos_sin, squared, pairwise — enabling both
  ablation-matched (75 features matching QRH) and standalone-optimized modes
- **Sparse/Dense Input Connectivity**: Sparse mode routes input only to sensory
  neurons (matching QRH's sensory qubit pattern); dense routes to all neurons
- **PPO Actor-Critic Readout**: Inherited from ReservoirHybridBase — identical
  training infrastructure as QRH
- **Stateless**: No hidden state carried between run_brain() calls (matches QRH)

Architecture:
    Sensory Input -> W_in Projection -> ESN Reservoir (FIXED)
    -> Feature Channels -> PPO Readout

The reservoir avoids gradient issues because no reservoir parameters are trained.
Only the classical actor-critic readout is optimized via PPO.

References
----------
- Jaeger (2001) "The echo state approach to analysing and training recurrent
  neural networks"
- Lukoševičius & Jaeger (2009) "Reservoir computing approaches to recurrent
  neural network training"
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import Field, field_validator, model_validator

from quantumnematode.brain.arch._reservoir_hybrid_base import (
    MIN_READOUT_HIDDEN_DIM,
    ReservoirHybridBase,
    ReservoirHybridBaseConfig,
)
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.logging_config import logger

if TYPE_CHECKING:
    from quantumnematode.brain.actions import Action

# =============================================================================
# Default Hyperparameters (ESN-specific)
# =============================================================================

DEFAULT_NUM_RESERVOIR_NEURONS = 10
DEFAULT_RESERVOIR_DEPTH = 3
DEFAULT_RESERVOIR_SEED = 42
DEFAULT_SPECTRAL_RADIUS = 0.9
DEFAULT_INPUT_CONNECTIVITY = "sparse"
DEFAULT_INPUT_SCALE = 1.0
DEFAULT_FEATURE_CHANNELS: list[str] = ["raw", "cos_sin", "pairwise"]

# Validation constants
MIN_RESERVOIR_NEURONS = 2
MIN_RESERVOIR_DEPTH = 1
SPECTRAL_RADIUS_EPSILON = 1e-10  # Guard for degenerate W_res matrices

# =============================================================================
# Type Aliases
# =============================================================================

FeatureChannel = Literal["raw", "cos_sin", "squared", "pairwise"]


# =============================================================================
# Configuration
# =============================================================================


class CRHBrainConfig(ReservoirHybridBaseConfig):
    """Configuration for the CRHBrain architecture.

    Inherits PPO readout, LR scheduling, and sensory module config from
    ReservoirHybridBaseConfig. Adds ESN-specific fields.
    """

    # Reservoir parameters
    num_reservoir_neurons: int = Field(
        default=DEFAULT_NUM_RESERVOIR_NEURONS,
        description="Number of neurons in the ESN reservoir.",
    )
    reservoir_depth: int = Field(
        default=DEFAULT_RESERVOIR_DEPTH,
        description="Number of reservoir layers (data re-uploading depth).",
    )
    reservoir_seed: int = Field(
        default=DEFAULT_RESERVOIR_SEED,
        description="Seed for deterministic reservoir construction.",
    )
    spectral_radius: float = Field(
        default=DEFAULT_SPECTRAL_RADIUS,
        description="Spectral radius for W_res eigenvalue scaling.",
    )
    input_connectivity: str = Field(
        default=DEFAULT_INPUT_CONNECTIVITY,
        description="Input connectivity mode: 'sparse' or 'dense'.",
    )
    input_scale: float = Field(
        default=DEFAULT_INPUT_SCALE,
        description="Scaling factor for W_in entries.",
    )
    feature_channels: list[FeatureChannel] = Field(
        default_factory=lambda: ["raw", "cos_sin", "pairwise"],
        description="Feature channels to extract from ESN activations.",
    )
    num_sensory_neurons: int | None = Field(
        default=None,
        description=(
            "Number of sensory neurons for sparse connectivity. "
            "None = auto-compute as min(input_dim, num_reservoir_neurons)."
        ),
    )

    @field_validator("num_reservoir_neurons")
    @classmethod
    def validate_num_reservoir_neurons(cls, v: int) -> int:
        """Validate num_reservoir_neurons >= 2."""
        if v < MIN_RESERVOIR_NEURONS:
            msg = f"num_reservoir_neurons must be >= {MIN_RESERVOIR_NEURONS}, got {v}"
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

    @field_validator("spectral_radius")
    @classmethod
    def validate_spectral_radius(cls, v: float) -> float:
        """Validate spectral_radius > 0."""
        if v <= 0:
            msg = f"spectral_radius must be > 0, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("input_connectivity")
    @classmethod
    def validate_input_connectivity(cls, v: str) -> str:
        """Validate input_connectivity is 'sparse' or 'dense'."""
        if v not in {"sparse", "dense"}:
            msg = f"input_connectivity must be 'sparse' or 'dense', got '{v}'"
            raise ValueError(msg)
        return v

    @field_validator("feature_channels")
    @classmethod
    def validate_feature_channels(cls, v: list[FeatureChannel]) -> list[FeatureChannel]:
        """Validate feature_channels is non-empty."""
        if not v:
            msg = "feature_channels must be non-empty"
            raise ValueError(msg)
        return v

    @field_validator("readout_hidden_dim")
    @classmethod
    def validate_readout_hidden_dim(cls, v: int) -> int:
        """Validate readout_hidden_dim >= 1."""
        if v < MIN_READOUT_HIDDEN_DIM:
            msg = f"readout_hidden_dim must be >= {MIN_READOUT_HIDDEN_DIM}, got {v}"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def validate_sensory_neuron_count(self) -> CRHBrainConfig:
        """Validate num_sensory_neurons bounds when explicitly set."""
        if self.num_sensory_neurons is not None:
            if self.num_sensory_neurons < 1:
                msg = f"num_sensory_neurons must be >= 1, got {self.num_sensory_neurons}"
                raise ValueError(msg)
            if self.num_sensory_neurons > self.num_reservoir_neurons:
                msg = (
                    f"num_sensory_neurons ({self.num_sensory_neurons}) must be "
                    f"<= num_reservoir_neurons ({self.num_reservoir_neurons})"
                )
                raise ValueError(msg)
        return self


# =============================================================================
# CRH Brain
# =============================================================================


class CRHBrain(ReservoirHybridBase):
    """Classical Reservoir Hybrid brain architecture.

    Uses a fixed Echo State Network reservoir to generate feature representations
    via configurable channels, with a PPO-trained classical actor-critic readout.
    """

    _brain_name = "CRH"

    def __init__(
        self,
        config: CRHBrainConfig,
        num_actions: int = 4,
        device: DeviceType = DeviceType.CPU,
        action_set: list[Action] | None = None,
    ) -> None:
        # Store ESN config (must be set before super().__init__ uses feature_dim)
        self.num_neurons = config.num_reservoir_neurons
        self.reservoir_depth = config.reservoir_depth
        self.reservoir_seed = config.reservoir_seed
        self.spectral_radius = config.spectral_radius
        self.input_connectivity = config.input_connectivity
        self.input_scale = config.input_scale
        self.feature_channels = config.feature_channels

        # Compute feature dimension before calling base init (Decision 7)
        feature_dim = self._compute_feature_dim()

        # Base class handles: seeding, sensory modules, input_dim, actor/critic,
        # LayerNorm, optimizer, LR scheduling, PPO params, buffer, state tracking
        super().__init__(config, feature_dim, num_actions, device, action_set)

        # Compute sensory neuron count for input routing
        if config.num_sensory_neurons is not None:
            self.num_sensory = config.num_sensory_neurons
        else:
            self.num_sensory = min(self.input_dim, self.num_neurons)

        # Build reservoir matrices (deterministic from seed)
        self.W_in, self.W_res = self._build_reservoir_matrices()

        logger.info(
            f"CRHBrain initialized: {self.num_neurons} neurons "
            f"({self.num_sensory} sensory), "
            f"{self.reservoir_depth} layers, "
            f"spectral_radius={self.spectral_radius}, "
            f"connectivity={self.input_connectivity}, "
            f"channels={self.feature_channels}, "
            f"feature_dim={self.feature_dim}, input_dim={self.input_dim}, "
            f"actor/critic ({config.readout_hidden_dim}x{config.readout_num_layers})",
        )

    # =========================================================================
    # Abstract method implementations
    # =========================================================================

    def _compute_feature_dim(self) -> int:
        """Compute feature dimension from configured channels."""
        dim = 0
        for ch in self.feature_channels:
            if ch == "raw":
                dim += self.num_neurons
            elif ch == "cos_sin":
                dim += 2 * self.num_neurons
            elif ch == "squared":
                dim += self.num_neurons
            elif ch == "pairwise":
                dim += self.num_neurons * (self.num_neurons - 1) // 2
        return dim

    def _get_reservoir_features(self, sensory_features: np.ndarray) -> np.ndarray:
        """Run sensory features through ESN reservoir and extract features."""
        # ESN forward pass: h_0 = tanh(W_in @ x)
        x = sensory_features.astype(np.float64)
        h = np.tanh(self.W_in @ x)

        # Data re-uploading: h_l = tanh(W_res @ h_{l-1} + W_in @ x)
        for _ in range(1, self.reservoir_depth):
            h = np.tanh(self.W_res @ h + self.W_in @ x)

        return self._extract_features(h)

    def _create_copy_instance(
        self,
        config: ReservoirHybridBaseConfig,
    ) -> CRHBrain:
        """Construct a fresh CRHBrain for the copy() method."""
        return CRHBrain(
            config=config,  # type: ignore[arg-type]
            num_actions=self.num_actions,
            device=self._device_type,
            action_set=self._action_set,
        )

    # =========================================================================
    # Reservoir Construction
    # =========================================================================

    def _build_reservoir_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """Build W_in and W_res matrices deterministically from seed.

        W_in: Input weight matrix (num_neurons, input_dim).
            - Dense: all entries drawn from Uniform[-input_scale, input_scale].
            - Sparse: same, but rows beyond num_sensory are zeroed out.

        W_res: Reservoir weight matrix (num_neurons, num_neurons).
            Random normal, then scaled so largest eigenvalue magnitude equals
            spectral_radius. If max eigenvalue < epsilon, skip scaling.
        """
        rng = np.random.default_rng(self.reservoir_seed)

        # W_in: input weight matrix
        w_in = rng.uniform(
            -self.input_scale,
            self.input_scale,
            size=(self.num_neurons, self.input_dim),
        )

        # Sparse mode: zero out rows for non-sensory neurons
        if self.input_connectivity == "sparse":
            w_in[self.num_sensory :] = 0.0

        # W_res: reservoir weight matrix
        w_res = rng.standard_normal(size=(self.num_neurons, self.num_neurons))

        # Scale to target spectral radius
        eigenvalues = np.linalg.eigvals(w_res)
        max_eigenvalue = np.max(np.abs(eigenvalues))

        if max_eigenvalue > SPECTRAL_RADIUS_EPSILON:
            w_res = w_res * (self.spectral_radius / max_eigenvalue)
        else:
            logger.warning(
                f"CRH: W_res max eigenvalue magnitude ({max_eigenvalue:.2e}) < "
                f"{SPECTRAL_RADIUS_EPSILON:.2e}, skipping spectral radius scaling. "
                f"This is extremely unlikely with random normal initialization.",
            )

        return w_in.astype(np.float64), w_res.astype(np.float64)

    # =========================================================================
    # Feature Extraction
    # =========================================================================

    def _extract_features(self, activations: np.ndarray) -> np.ndarray:
        """Extract features from ESN activations using configured channels.

        Parameters
        ----------
        activations : np.ndarray
            Final-layer ESN activations, shape (num_neurons,), values in [-1, 1].

        Returns
        -------
        np.ndarray
            Concatenated feature vector (float32).
        """
        features: list[np.ndarray] = []

        for ch in self.feature_channels:
            if ch == "raw":
                features.append(activations)
            elif ch == "cos_sin":
                features.append(np.cos(np.pi * activations))
                features.append(np.sin(np.pi * activations))
            elif ch == "squared":
                features.append(activations**2)
            elif ch == "pairwise":
                n = len(activations)
                pairs = [activations[i] * activations[j] for i in range(n) for j in range(i + 1, n)]
                features.append(np.array(pairs))

        return np.concatenate(features).astype(np.float32)
