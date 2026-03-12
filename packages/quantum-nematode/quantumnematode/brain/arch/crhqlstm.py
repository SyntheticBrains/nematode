"""
CRH-QLSTM Brain: Classical Reservoir Hybrid + QLIF-LSTM Temporal Readout.

Uses a fixed CRH Echo State Network as a feature extractor with a trainable
QLIF-LSTM readout. Serves as the classical ablation companion for QRH-QLSTM.

See ``_reservoir_lstm_base.py`` for the shared base class and architecture
documentation.
"""

from __future__ import annotations

from pydantic import Field, field_validator

from quantumnematode.brain.arch._reservoir_lstm_base import (
    ReservoirLSTMBase,
    ReservoirLSTMBaseConfig,
)
from quantumnematode.brain.arch.crh import CRHBrain, CRHBrainConfig, FeatureChannel, InputEncoding


class CRHQLSTMBrainConfig(ReservoirLSTMBaseConfig):
    """Configuration for CRH-QLSTM: classical reservoir + QLIF-LSTM readout.

    Extends ReservoirLSTMBaseConfig with CRH-specific reservoir params.
    """

    # Reservoir parameters (CRH)
    num_reservoir_neurons: int = Field(default=10, description="ESN reservoir neurons.")
    reservoir_depth: int = Field(default=3, description="ESN reservoir layers.")
    reservoir_seed: int = Field(default=42, description="Seed for deterministic reservoir.")
    spectral_radius: float = Field(default=0.9, description="ESN spectral radius.")
    input_connectivity: str = Field(default="sparse", description="Input connectivity mode.")
    input_scale: float = Field(default=1.0, description="W_in scaling factor.")
    feature_channels: list[FeatureChannel] = Field(
        default_factory=lambda: ["raw", "cos_sin", "pairwise"],
        description="Feature channels from ESN activations.",
    )
    input_encoding: InputEncoding = Field(default="linear", description="Input encoding mode.")

    @field_validator("feature_channels")
    @classmethod
    def validate_feature_channels(cls, v: list[FeatureChannel]) -> list[FeatureChannel]:
        """Validate feature_channels is non-empty."""
        if not v:
            msg = "feature_channels must be non-empty"
            raise ValueError(msg)
        return v


class CRHQLSTMBrain(ReservoirLSTMBase):
    """CRH classical reservoir + QLIF-LSTM temporal readout.

    Uses a fixed CRH Echo State Network as a feature extractor with a
    trainable QLIF-LSTM readout. Serves as the classical ablation
    companion for QRH-QLSTM.
    """

    _brain_name = "CRH-QLSTM"

    def _create_reservoir(
        self,
        config: ReservoirLSTMBaseConfig,
    ) -> CRHBrain:
        """Create a CRH brain instance as feature extractor."""
        if not isinstance(config, CRHQLSTMBrainConfig):
            msg = f"CRHQLSTMBrain requires CRHQLSTMBrainConfig, got {type(config).__name__}"
            raise TypeError(msg)

        crh_config = CRHBrainConfig(
            num_reservoir_neurons=config.num_reservoir_neurons,
            reservoir_depth=config.reservoir_depth,
            reservoir_seed=config.reservoir_seed,
            spectral_radius=config.spectral_radius,
            input_connectivity=config.input_connectivity,
            input_scale=config.input_scale,
            feature_channels=config.feature_channels,
            input_encoding=config.input_encoding,
            sensory_modules=config.sensory_modules,
            seed=config.seed,
        )
        return CRHBrain(
            config=crh_config,
            num_actions=self.num_actions,
            device=self._device_type,
        )

    def _compute_reservoir_feature_dim(
        self,
        config: ReservoirLSTMBaseConfig,
    ) -> int:
        """Compute CRH feature dimension from configured channels.

        Computed purely from config fields (not from reservoir instance)
        so this method is safe to call independently of init ordering.
        """
        if not isinstance(config, CRHQLSTMBrainConfig):
            msg = f"CRHQLSTMBrain requires CRHQLSTMBrainConfig, got {type(config).__name__}"
            raise TypeError(msg)
        n = config.num_reservoir_neurons
        dim = 0
        for ch in config.feature_channels:
            if ch == "raw":
                dim += n
            elif ch == "cos_sin":
                dim += 2 * n
            elif ch == "squared":
                dim += n
            elif ch == "pairwise":
                dim += n * (n - 1) // 2
        return dim
