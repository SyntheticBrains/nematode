"""
Shared Quantum Utilities.

Common functions for quantum brain architectures that use Qiskit.
Extracted from _qlif_layers.py to avoid QLIF-specific coupling.

Functions
---------
- get_qiskit_backend: Lazy Qiskit AerSimulator initialization
- run_circuit_shots: Execute a circuit and return normalized probability distribution
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from quantumnematode.errors import ERROR_MISSING_IMPORT_QISKIT_AER
from quantumnematode.logging_config import logger

if TYPE_CHECKING:
    from qiskit import QuantumCircuit

    from quantumnematode.brain.arch.dtypes import DeviceType


def get_qiskit_backend(
    device: DeviceType,  # noqa: ARG001  # reserved for future QPU routing
    seed: int | None = None,
) -> Any:  # noqa: ANN401
    """Get or create a Qiskit Aer simulator backend.

    Parameters
    ----------
    device : DeviceType
        Device selection (currently always uses CPU simulator).
    seed : int or None
        Seed for the simulator for reproducibility.

    Returns
    -------
    AerSimulator
        Configured Qiskit Aer backend.
    """
    try:
        from qiskit_aer import AerSimulator
    except ImportError as err:
        error_message = ERROR_MISSING_IMPORT_QISKIT_AER
        logger.error(error_message)
        raise ImportError(error_message) from err

    return AerSimulator(
        device="CPU",
        seed_simulator=seed,
    )


def run_circuit_shots(
    backend: Any,  # noqa: ANN401
    circuit: QuantumCircuit,
    shots: int,
    num_qubits: int,
) -> np.ndarray:
    """Execute a quantum circuit and return the normalized probability distribution.

    Parameters
    ----------
    backend : AerSimulator
        Qiskit Aer backend for circuit execution.
    circuit : QuantumCircuit
        Quantum circuit with measurement gates.
    shots : int
        Number of measurement shots.
    num_qubits : int
        Number of qubits in the circuit.

    Returns
    -------
    np.ndarray
        Probability distribution over all 2^n bitstrings, shape (2^num_qubits,).
    """
    job = backend.run(circuit, shots=shots)
    result = job.result()
    counts = result.get_counts()

    num_states = 2**num_qubits
    probs = np.zeros(num_states)

    for bitstring, count in counts.items():
        idx = int(bitstring, 2)
        probs[idx] = count / shots

    return probs
