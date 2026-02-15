"""
Quantum Leaky Integrate-and-Fire (QLIF) Layers.

Shared components for QLIF-based brain architectures (QSNNReinforce, QSNN-PPO).
Implements quantum circuit execution and surrogate gradient functions following
Brand & Petruccione (2024).

Key Components
--------------
- QLIFSurrogateSpike: Differentiable spike function using sigmoid surrogate
  centered at the RY gate's transition point (pi/2)
- build_qlif_circuit: Constructs minimal 2-gate QLIF neuron circuit
  (RY for membrane + input, RX for leak)
- execute_qlif_layer: Non-differentiable layer execution (inference mode)
- execute_qlif_layer_differentiable: Layer execution with surrogate gradients
- execute_qlif_layer_differentiable_cached: Cached execution for multi-epoch
  training (reuses quantum measurements, recomputes ry_angles)
- encode_sensory_spikes: Sigmoid encoding of continuous features to spike probs
- get_qiskit_backend: Lazy Qiskit Aer backend initialization

References
----------
- Brand & Petruccione (2024). "A quantum leaky integrate-and-fire spiking neuron
  and network." npj Quantum Information, 10(1), 16.
- Neftci et al. (2019). "Surrogate gradient learning in spiking neural networks."
  IEEE Signal Processing Magazine, 36(6), 51-63.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from qiskit import QuantumCircuit

from quantumnematode.errors import ERROR_MISSING_IMPORT_QISKIT_AER
from quantumnematode.logging_config import logger

if TYPE_CHECKING:
    from quantumnematode.brain.arch.dtypes import DeviceType

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

# Surrogate gradient sharpness parameter (alpha).
# Controls how closely the sigmoid surrogate approximates the Heaviside step.
# Higher = sharper but noisier gradients. 1.0 matches SpikingReinforceBrain's
# proven setting; smoother gradients reduce noise from quantum shot variance.
DEFAULT_SURROGATE_ALPHA = 1.0

# Weight initialization scale. With 6 sensory neurons and scale=0.15, typical
# weighted_input ~ N(0, 6*0.0225) = N(0, 0.135), std≈0.37. This produces
# tanh(0.37)*pi ≈ 1.10 rad peak RY angles, giving spike probs 0.05-0.35 —
# enough differentiation for REINFORCE without gradient amplification
# instability.
WEIGHT_INIT_SCALE = 0.15

# Logit scaling factor for converting spike probabilities to action logits.
# Maps spike probs in [0,1] to logits via (prob - 0.5) * scale.
# Higher values create sharper action differentiation from small spike diffs.
LOGIT_SCALE = 5.0


# ──────────────────────────────────────────────────────────────────────
# Autograd Function
# ──────────────────────────────────────────────────────────────────────


class QLIFSurrogateSpike(torch.autograd.Function):
    """Custom autograd function for QLIF neuron surrogate gradients.

    Forward pass uses the actual quantum-measured spike probability (from QLIF
    circuit execution). Backward pass uses a sigmoid surrogate gradient, which
    approximates how the spike probability changes with the RY angle.

    The QLIF circuit computes: |0> -> RY(ry_angle) -> RX(leak) -> Measure
    where ry_angle = theta + tanh(w·x / sqrt(fan_in)) * pi. The fan-in
    scaling keeps tanh in its responsive regime regardless of layer width.
    The surrogate approximates d(spike_prob)/d(ry_angle), and autograd
    chains through to both theta and weights via d(ry_angle)/d(theta) = 1
    and d(ry_angle)/d(w) = pi * sech²(w·x / sqrt(fan_in)) * x / sqrt(fan_in).
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,  # noqa: ANN401
        ry_angle: torch.Tensor,
        quantum_spike_prob: float,
        alpha: float = DEFAULT_SURROGATE_ALPHA,
    ) -> torch.Tensor:
        """Forward pass: return quantum spike probability as a differentiable tensor.

        Parameters
        ----------
        ctx : torch.autograd.Function context
            Autograd context for saving tensors.
        ry_angle : torch.Tensor
            The RY rotation angle (theta + tanh(w·x)*pi), scalar tensor with grad.
            This is the primary differentiable input connecting both theta and weights.
        quantum_spike_prob : float
            The spike probability measured from the quantum circuit.
        alpha : float
            Surrogate gradient sharpness parameter.
        """
        ctx.save_for_backward(
            ry_angle,
            torch.tensor(alpha, device=ry_angle.device),
        )
        return torch.tensor(
            quantum_spike_prob,
            dtype=torch.float32,
            device=ry_angle.device,
        )

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any,  # noqa: ANN401
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None]:
        """Backward pass: sigmoid surrogate gradient.

        Approximates d(spike_prob)/d(ry_angle) using a sigmoid derivative.
        The sigmoid is centered at pi/2 (where spike probability transitions
        from low to high for the RY gate).
        """
        ry_angle, alpha = ctx.saved_tensors
        # Sigmoid surrogate centered at pi/2 (the RY transition point)
        # For RY(angle): P(1) ~ sin²(angle/2), which transitions at angle=pi/2
        shifted = alpha * (ry_angle - torch.pi / 2)
        sigma = torch.sigmoid(shifted)
        grad_surrogate = alpha * sigma * (1 - sigma)
        return grad_output * grad_surrogate, None, None


# ──────────────────────────────────────────────────────────────────────
# Backend Helper
# ──────────────────────────────────────────────────────────────────────


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


# ──────────────────────────────────────────────────────────────────────
# Circuit Construction
# ──────────────────────────────────────────────────────────────────────


def build_qlif_circuit(
    weighted_input: float,
    theta_membrane: float,
    leak_angle: float,
) -> QuantumCircuit:
    """Build a QLIF neuron circuit.

    The minimal 2-gate QLIF circuit from Brand & Petruccione (2024):
    |0> -> RY(theta_membrane + weighted_input) -> RX(theta_leak) -> Measure

    Parameters
    ----------
    weighted_input : float
        Pre-scaled weighted input (sum(w_ij * spike_j) / sqrt(fan_in)).
        Fan-in scaling is applied by callers to keep tanh in its
        responsive regime regardless of layer width.
    theta_membrane : float
        Trainable membrane potential parameter.
    leak_angle : float
        Leak angle for the RX gate, derived from membrane_tau.

    Returns
    -------
    QuantumCircuit
        The QLIF neuron circuit.
    """
    qc = QuantumCircuit(1, 1)

    # RY encodes membrane potential + weighted input
    # tanh bounds the weighted input to [-1, 1] before scaling by pi,
    # preventing angle wrapping through multiple cycles (hash function behavior)
    normalized_input = float(np.tanh(weighted_input))
    ry_angle = float(theta_membrane + normalized_input * np.pi)
    qc.ry(ry_angle, 0)

    # RX implements leak
    qc.rx(leak_angle, 0)

    # Measure
    qc.measure(0, 0)

    return qc


# ──────────────────────────────────────────────────────────────────────
# Layer Execution
# ──────────────────────────────────────────────────────────────────────


def execute_qlif_layer(  # noqa: PLR0913
    pre_spikes: np.ndarray,
    weights: torch.Tensor,
    theta_membrane: torch.Tensor,
    refractory_state: np.ndarray,
    backend: Any,  # noqa: ANN401
    shots: int,
    threshold: float,
    refractory_period: int,
    leak_angle: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Execute a layer of QLIF neurons (non-differentiable).

    Parameters
    ----------
    pre_spikes : np.ndarray
        Spike probabilities from presynaptic layer, shape (num_pre,).
    weights : torch.Tensor
        Weight matrix, shape (num_pre, num_post).
    theta_membrane : torch.Tensor
        Membrane potential parameters, shape (num_post,).
    refractory_state : np.ndarray
        Refractory countdown per neuron, shape (num_post,).
    backend : AerSimulator
        Qiskit backend for circuit execution.
    shots : int
        Number of measurement shots per circuit.
    threshold : float
        Firing threshold for refractory period triggering.
    refractory_period : int
        Number of timesteps to remain refractory after firing.
    leak_angle : float
        Leak angle for the RX gate.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (spike_probs, updated_refractory_state) for the layer.
    """
    num_post = weights.shape[1]
    spike_probs = np.zeros(num_post)

    # Convert weights to numpy for computation
    weights_np = weights.detach().cpu().numpy()
    theta_np = theta_membrane.detach().cpu().numpy()

    for j in range(num_post):
        # Check refractory period
        if refractory_state[j] > 0:
            refractory_state[j] -= 1
            spike_probs[j] = 0.0
            continue

        # Compute weighted input: sum(w_ij * spike_i)
        weighted_input = np.dot(pre_spikes, weights_np[:, j])

        # Scale by 1/sqrt(fan_in) to keep tanh in responsive regime
        # regardless of layer width. Without this, tanh saturates when
        # fan_in * avg_spike * |w| > ~2, killing weight gradients.
        fan_in_scale = np.sqrt(weights_np.shape[0])
        scaled_input = weighted_input / fan_in_scale

        # Build and execute QLIF circuit
        qc = build_qlif_circuit(scaled_input, theta_np[j], leak_angle)
        job = backend.run(qc, shots=shots)
        result = job.result()
        counts = result.get_counts()

        # Firing probability = P(measure |1>)
        spike_prob = counts.get("1", 0) / shots
        spike_probs[j] = spike_prob

        # Update refractory state if fired
        if spike_prob > threshold:
            refractory_state[j] = refractory_period

    return spike_probs, refractory_state


def execute_qlif_layer_differentiable(  # noqa: PLR0913
    pre_spikes: torch.Tensor,
    weights: torch.Tensor,
    theta_membrane: torch.Tensor,
    refractory_state: np.ndarray,
    backend: Any,  # noqa: ANN401
    shots: int,
    threshold: float,
    refractory_period: int,
    leak_angle: float,
    device: torch.device,
    alpha: float = DEFAULT_SURROGATE_ALPHA,
) -> tuple[torch.Tensor, np.ndarray]:
    """Execute a QLIF layer with surrogate gradient support.

    Same quantum circuit execution as execute_qlif_layer(), but wraps
    the result in QLIFSurrogateSpike so that gradients flow back through
    the weight matrix via the sigmoid surrogate.

    Parameters
    ----------
    pre_spikes : torch.Tensor
        Spike probabilities from presynaptic layer, shape (num_pre,).
    weights : torch.Tensor
        Weight matrix with requires_grad=True, shape (num_pre, num_post).
    theta_membrane : torch.Tensor
        Membrane potential parameters with requires_grad=True, shape (num_post,).
    refractory_state : np.ndarray
        Refractory countdown per neuron, shape (num_post,).
    backend : AerSimulator
        Qiskit backend for circuit execution.
    shots : int
        Number of measurement shots per circuit.
    threshold : float
        Firing threshold for refractory period triggering.
    refractory_period : int
        Number of timesteps to remain refractory after firing.
    leak_angle : float
        Leak angle for the RX gate.
    device : torch.device
        Torch device for tensor creation.
    alpha : float
        Surrogate gradient sharpness parameter.

    Returns
    -------
    tuple[torch.Tensor, np.ndarray]
        (spike_probs_tensor, updated_refractory_state) — spike probs
        are torch tensors with grad_fn for backprop.
    """
    num_post = weights.shape[1]
    spike_probs_list: list[torch.Tensor] = []

    for j in range(num_post):
        if refractory_state[j] > 0:
            refractory_state[j] -= 1
            # Zero spike with gradient connection through weights
            spike_probs_list.append(
                torch.zeros(1, device=device, dtype=torch.float32).squeeze(),
            )
            continue

        # Differentiable weighted input (in autograd graph)
        weighted_input = torch.dot(pre_spikes, weights[:, j])

        # Fan-in-aware scaling: divide by sqrt(fan_in) to keep tanh in
        # responsive regime regardless of layer width.
        fan_in_scale = (weights.shape[0]) ** 0.5
        scaled_input = weighted_input / fan_in_scale

        # Differentiable RY angle: theta + tanh(w·x / sqrt(fan_in)) * pi
        ry_angle = theta_membrane[j] + torch.tanh(scaled_input) * torch.pi

        # Execute quantum circuit for forward spike probability (detached)
        wi_np = float(scaled_input.detach().cpu())
        qc = build_qlif_circuit(
            wi_np,
            float(theta_membrane[j].detach().cpu()),
            leak_angle,
        )
        job = backend.run(qc, shots=shots)
        result = job.result()
        counts = result.get_counts()
        quantum_spike_prob = counts.get("1", 0) / shots

        # Wrap in surrogate gradient function
        spike_prob: torch.Tensor = QLIFSurrogateSpike.apply(  # type: ignore[assignment]
            ry_angle,
            quantum_spike_prob,
            alpha,
        )
        spike_probs_list.append(spike_prob)

        # Update refractory state
        if quantum_spike_prob > threshold:
            refractory_state[j] = refractory_period

    spike_probs = torch.stack(spike_probs_list)
    return spike_probs, refractory_state


def execute_qlif_layer_differentiable_cached(  # noqa: PLR0913
    pre_spikes: torch.Tensor,
    weights: torch.Tensor,
    theta_membrane: torch.Tensor,
    refractory_state: np.ndarray,
    cached_spike_probs: list[float],
    threshold: float,
    refractory_period: int,
    device: torch.device,
    alpha: float = DEFAULT_SURROGATE_ALPHA,
) -> tuple[torch.Tensor, np.ndarray]:
    """Execute a QLIF layer using cached spike probabilities.

    Same as execute_qlif_layer_differentiable() but skips quantum circuit
    execution, using pre-cached spike probabilities from epoch 0 instead.
    The ry_angle is recomputed from current (updated) weights so surrogate
    gradients reflect the latest parameters.

    Parameters
    ----------
    pre_spikes : torch.Tensor
        Spike probabilities from presynaptic layer, shape (num_pre,).
    weights : torch.Tensor
        Weight matrix with requires_grad=True, shape (num_pre, num_post).
    theta_membrane : torch.Tensor
        Membrane potential parameters with requires_grad=True, shape (num_post,).
    refractory_state : np.ndarray
        Refractory countdown per neuron, shape (num_post,).
    cached_spike_probs : list[float]
        Pre-cached quantum spike probabilities for each neuron in this layer.
    threshold : float
        Firing threshold for refractory period triggering.
    refractory_period : int
        Number of timesteps to remain refractory after firing.
    device : torch.device
        Torch device for tensor creation.
    alpha : float
        Surrogate gradient sharpness parameter.

    Returns
    -------
    tuple[torch.Tensor, np.ndarray]
        (spike_probs_tensor, updated_refractory_state).
    """
    num_post = weights.shape[1]
    spike_probs_list: list[torch.Tensor] = []

    for j in range(num_post):
        if refractory_state[j] > 0:
            refractory_state[j] -= 1
            spike_probs_list.append(
                torch.zeros(1, device=device, dtype=torch.float32).squeeze(),
            )
            continue

        # Recompute ry_angle from current weights (differentiable)
        weighted_input = torch.dot(pre_spikes, weights[:, j])
        fan_in_scale = (weights.shape[0]) ** 0.5
        scaled_input = weighted_input / fan_in_scale
        ry_angle = theta_membrane[j] + torch.tanh(scaled_input) * torch.pi

        # Use cached quantum spike probability instead of running circuit
        quantum_spike_prob = cached_spike_probs[j]

        spike_prob: torch.Tensor = QLIFSurrogateSpike.apply(  # type: ignore[assignment]
            ry_angle,
            quantum_spike_prob,
            alpha,
        )
        spike_probs_list.append(spike_prob)

        if quantum_spike_prob > threshold:
            refractory_state[j] = refractory_period

    spike_probs = torch.stack(spike_probs_list)
    return spike_probs, refractory_state


# ──────────────────────────────────────────────────────────────────────
# Sensory Encoding
# ──────────────────────────────────────────────────────────────────────


def encode_sensory_spikes(
    features: np.ndarray,
    num_sensory: int,
) -> np.ndarray:
    """Encode sensory features as spike probabilities.

    Uses sigmoid function with scaling to convert continuous features
    to spike probabilities in [0, 1].

    Parameters
    ----------
    features : np.ndarray
        Input features from preprocessing.
    num_sensory : int
        Number of sensory neurons.

    Returns
    -------
    np.ndarray
        Spike probabilities for sensory neurons.
    """
    num_features = len(features)
    sensory_spikes = np.zeros(num_sensory)

    for i in range(num_sensory):
        # Cycle through features if we have more neurons than features
        feature_idx = i % num_features
        feature_val = features[feature_idx]

        # Sigmoid with scaling: sigmoid(feature * 5.0)
        # This maps [0,1] inputs to roughly [0.5, 0.99] probabilities
        sensory_spikes[i] = 1.0 / (1.0 + np.exp(-feature_val * 5.0))

    return sensory_spikes
