"""Unit tests for shared quantum utilities (_quantum_utils.py)."""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from quantumnematode.brain.arch._quantum_utils import get_qiskit_backend, run_circuit_shots
from quantumnematode.brain.arch.dtypes import DeviceType


class TestGetQiskitBackend:
    """Test cases for get_qiskit_backend()."""

    def test_returns_aer_simulator(self):
        """Backend should be an AerSimulator instance."""
        from qiskit_aer import AerSimulator

        backend = get_qiskit_backend(DeviceType.CPU)
        assert isinstance(backend, AerSimulator)

    def test_seeded_backend(self):
        """Seeded backends should produce reproducible results."""
        backend1 = get_qiskit_backend(DeviceType.CPU, seed=42)
        backend2 = get_qiskit_backend(DeviceType.CPU, seed=42)

        # Both should be valid backends
        assert backend1 is not None
        assert backend2 is not None

        # Run a simple circuit on both and compare results
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])

        result1 = backend1.run(qc, shots=100).result().get_counts()
        result2 = backend2.run(qc, shots=100).result().get_counts()

        assert result1 == result2

    def test_different_seeds_different_results(self):
        """Different seeds should generally produce different shot results."""
        backend1 = get_qiskit_backend(DeviceType.CPU, seed=42)
        backend2 = get_qiskit_backend(DeviceType.CPU, seed=999)

        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])

        result1 = backend1.run(qc, shots=1000).result().get_counts()
        result2 = backend2.run(qc, shots=1000).result().get_counts()

        # Results should differ (extremely unlikely to match with different seeds)
        assert result1 != result2


class TestRunCircuitShots:
    """Test cases for run_circuit_shots()."""

    @pytest.fixture
    def backend(self):
        """Create a seeded AerSimulator backend."""
        return get_qiskit_backend(DeviceType.CPU, seed=42)

    def test_returns_probability_distribution(self, backend):
        """Output should be a valid probability distribution."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])

        probs = run_circuit_shots(backend, qc, shots=1000, num_qubits=2)

        assert isinstance(probs, np.ndarray)
        assert probs.shape == (4,)  # 2^2 = 4 states
        assert np.isclose(probs.sum(), 1.0)
        assert np.all(probs >= 0.0)

    def test_correct_dimension(self, backend):
        """Output dimension should be 2^num_qubits."""
        for num_qubits in [1, 2, 3]:
            qc = QuantumCircuit(num_qubits, num_qubits)
            qc.h(0)
            qc.measure(range(num_qubits), range(num_qubits))

            probs = run_circuit_shots(backend, qc, shots=100, num_qubits=num_qubits)
            assert probs.shape == (2**num_qubits,)

    def test_deterministic_with_seed(self, backend):
        """Same backend and circuit should produce identical results."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])

        backend1 = get_qiskit_backend(DeviceType.CPU, seed=42)
        backend2 = get_qiskit_backend(DeviceType.CPU, seed=42)

        probs1 = run_circuit_shots(backend1, qc, shots=500, num_qubits=2)
        probs2 = run_circuit_shots(backend2, qc, shots=500, num_qubits=2)

        np.testing.assert_array_equal(probs1, probs2)

    def test_bell_state_distribution(self, backend):
        """Bell state should produce roughly 50/50 on |00⟩ and |11⟩."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])

        probs = run_circuit_shots(backend, qc, shots=10000, num_qubits=2)

        # Bell state: ~50% |00⟩ and ~50% |11⟩
        assert probs[0b00] > 0.4  # |00⟩
        assert probs[0b11] > 0.4  # |11⟩
        assert probs[0b01] < 0.05  # Should be ~0
        assert probs[0b10] < 0.05  # Should be ~0
