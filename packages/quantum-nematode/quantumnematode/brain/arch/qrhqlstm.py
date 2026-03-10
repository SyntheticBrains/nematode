"""
QRH-QLSTM Brain: Quantum Reservoir Hybrid + QLIF-LSTM Temporal Readout.

Uses a fixed QRH quantum reservoir as a feature extractor (X/Y/Z + ZZ
features, 3N + N(N-1)/2 dims) with a trainable QLIF-LSTM readout.

See ``_reservoir_lstm_base.py`` for the shared base class and architecture
documentation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field

from quantumnematode.brain.arch._reservoir_lstm_base import (
    ReservoirLSTMBase,
    ReservoirLSTMBaseConfig,
)
from quantumnematode.brain.arch.qrh import QRHBrain, QRHBrainConfig

if TYPE_CHECKING:
    from typing import Any

    from quantumnematode.brain.modules import ModuleName


class QRHQLSTMBrainConfig(ReservoirLSTMBaseConfig):
    """Configuration for QRH-QLSTM: quantum reservoir + QLIF-LSTM readout.

    Extends ReservoirLSTMBaseConfig with QRH-specific reservoir params.
    """

    # Reservoir parameters (QRH)
    num_reservoir_qubits: int = Field(default=8, description="Number of qubits in QRH reservoir.")
    reservoir_depth: int = Field(default=3, description="Entanglement layers in reservoir circuit.")
    reservoir_seed: int = Field(default=42, description="Seed for deterministic reservoir.")
    use_random_topology: bool = Field(
        default=False,
        description="Use random vs structured topology.",
    )
    num_sensory_qubits: int | None = Field(
        default=None,
        description="Number of sensory qubits (None = auto-compute).",
    )

    # Sensory modules
    sensory_modules: list[ModuleName] | None = Field(
        default=None,
        description="Sensory modules for feature extraction (None = legacy mode).",
    )


class QRHQLSTMBrain(ReservoirLSTMBase):
    """QRH quantum reservoir + QLIF-LSTM temporal readout.

    Uses a fixed QRH quantum reservoir as a feature extractor (X/Y/Z + ZZ
    features, 3N + N(N-1)/2 dims) with a trainable QLIF-LSTM readout.
    """

    _brain_name = "QRH-QLSTM"

    def _create_reservoir(
        self,
        config: ReservoirLSTMBaseConfig,
    ) -> Any:  # noqa: ANN401
        """Create a QRH brain instance as feature extractor."""
        if not isinstance(config, QRHQLSTMBrainConfig):
            msg = f"QRHQLSTMBrain requires QRHQLSTMBrainConfig, got {type(config).__name__}"
            raise TypeError(msg)

        qrh_config = QRHBrainConfig(
            num_reservoir_qubits=config.num_reservoir_qubits,
            reservoir_depth=config.reservoir_depth,
            reservoir_seed=config.reservoir_seed,
            use_random_topology=config.use_random_topology,
            num_sensory_qubits=config.num_sensory_qubits,
            sensory_modules=config.sensory_modules,
            seed=config.seed,
        )
        return QRHBrain(
            config=qrh_config,
            num_actions=self.num_actions,
            device=self._device_type,
        )

    def _compute_reservoir_feature_dim(
        self,
        config: ReservoirLSTMBaseConfig,
    ) -> int:
        """Compute QRH feature dimension: 3N + N(N-1)/2."""
        if not isinstance(config, QRHQLSTMBrainConfig):
            msg = f"QRHQLSTMBrain requires QRHQLSTMBrainConfig, got {type(config).__name__}"
            raise TypeError(msg)
        n = config.num_reservoir_qubits
        return 3 * n + n * (n - 1) // 2
