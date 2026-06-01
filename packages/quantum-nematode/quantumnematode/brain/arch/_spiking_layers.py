"""
Spiking Neural Network Layers with Surrogate Gradients.

This module implements biologically-inspired spiking neural network components
using surrogate gradient descent for gradient-based learning. The approach enables
backpropagation through discrete spike events while maintaining biological plausibility.

Key Components
--------------
- SurrogateGradientSpike: Differentiable spike function using sigmoid surrogate
- LIFLayer: Leaky Integrate-and-Fire neuron layer with stateful dynamics
- SpikingPolicyNetwork: Multi-layer spiking network for reinforcement learning

References
----------
- Neftci et al. (2019). "Surrogate Gradient Learning in Spiking Neural Networks"
- Zenke & Ganguli (2018). "SuperSpike: Supervised Learning in Multilayer SNNs"
- SpikingJelly: https://github.com/fangwei123456/spikingjelly
"""

import math
from typing import Any, Literal, NamedTuple

import torch
from torch import nn

# Output mode for temporal spike aggregation
OutputMode = Literal["accumulator", "final", "membrane"]


def _inverse_sigmoid(p: float) -> float:
    """Return ``logit(p) = ln(p / (1 - p))`` for a decay/probability in ``(0, 1)``.

    Used to initialize a learnable decay stored in logit space so the
    effective ``sigmoid(raw)`` value reproduces ``p`` at init while remaining
    confined to ``(0, 1)`` under training.
    """
    eps = 1e-6
    p_clamped = min(max(p, eps), 1.0 - eps)
    return math.log(p_clamped / (1.0 - p_clamped))


class PopulationEncoder(nn.Module):
    """
    Population coding encoder for scalar inputs.

    Encodes each scalar input value across multiple neurons using Gaussian
    tuning curves with different preferred values. This creates distinct
    activation patterns for different input values, enabling better
    discrimination by downstream spiking layers.

    Parameters
    ----------
    input_dim : int
        Number of input features to encode
    neurons_per_feature : int
        Number of encoding neurons per input feature (default: 8)
    sigma : float
        Width of Gaussian tuning curves (default: 0.2)
    min_val : float
        Minimum expected input value (default: -1.0)
    max_val : float
        Maximum expected input value (default: 1.0)

    Example
    -------
    For input_dim=2 and neurons_per_feature=8:
    - Input: [0.5, -0.3] (2 values)
    - Output: [g1_1, g1_2, ..., g1_8, g2_1, g2_2, ..., g2_8] (16 values)
    where g_i_j is the Gaussian response for feature i at preferred value j
    """

    # Type annotation for buffer
    preferred_values: torch.Tensor

    def __init__(
        self,
        input_dim: int,
        neurons_per_feature: int = 8,
        sigma: float = 0.2,
        min_val: float = -1.0,
        max_val: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.neurons_per_feature = neurons_per_feature
        self.sigma = sigma
        self.output_dim = input_dim * neurons_per_feature

        # Create preferred values (centers of Gaussian tuning curves)
        # Evenly spaced across the input range
        preferred_values = torch.linspace(min_val, max_val, neurons_per_feature)
        # Expand to all input features: [input_dim, neurons_per_feature]
        self.register_buffer(
            "preferred_values",
            preferred_values.unsqueeze(0).expand(input_dim, -1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode inputs using population coding.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor [batch_size, input_dim]

        Returns
        -------
        torch.Tensor
            Population-coded tensor [batch_size, input_dim * neurons_per_feature]
        """
        batch_size = x.shape[0]

        # Expand input for broadcasting: [batch_size, input_dim, 1]
        x_expanded = x.unsqueeze(-1)

        # Compute Gaussian responses for each input feature by broadcasting
        # x_expanded against preferred_values to get difference tensor
        diff = x_expanded - self.preferred_values.unsqueeze(0)

        # Gaussian tuning curve: exp(-diff^2 / (2 * sigma^2))
        responses = torch.exp(-(diff**2) / (2 * self.sigma**2))

        # Flatten to [batch_size, input_dim * neurons_per_feature]
        return responses.view(batch_size, -1)


class SurrogateGradientSpike(torch.autograd.Function):
    """
    Differentiable spike function using surrogate gradient approximation.

    Forward pass computes discrete spikes using Heaviside step function.
    Backward pass uses smooth sigmoid derivative for gradient flow.

    The surrogate gradient enables backpropagation through the non-differentiable
    spike threshold while maintaining biologically realistic spike generation.

    Parameters
    ----------
    alpha : float
        Controls smoothness of surrogate gradient. Higher values create
        sharper approximations (default: 10.0).

    Notes
    -----
    The surrogate gradient is:
        d_spike/d_v ≈ alpha * sigmoid(alpha * (v - v_th)) * (1 - sigmoid(alpha * (v - v_th)))

    where sigmoid is the sigmoid function. This approximation is only used during
    backward pass; forward pass uses true Heaviside step function.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,  # noqa: ANN401
        v: torch.Tensor,
        v_threshold: float,
        alpha: float = 10.0,
    ) -> torch.Tensor:
        """
        Generate spikes when membrane potential exceeds threshold.

        Parameters
        ----------
        ctx : torch.autograd.Function context
            Context for saving tensors for backward pass
        v : torch.Tensor
            Membrane potentials [batch_size, num_neurons]
        v_threshold : float
            Spike threshold voltage
        alpha : float
            Surrogate gradient smoothness parameter

        Returns
        -------
        torch.Tensor
            Binary spike tensor [batch_size, num_neurons]
            1.0 where v >= v_threshold, 0.0 otherwise
        """
        spike = (v >= v_threshold).float()
        ctx.save_for_backward(
            v,
            torch.tensor(v_threshold, device=v.device),
            torch.tensor(alpha, device=v.device),
        )
        return spike

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any,  # noqa: ANN401
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None]:
        """
        Compute surrogate gradient for backpropagation.

        Parameters
        ----------
        ctx : torch.autograd.Function context
            Context containing saved tensors from forward pass
        grad_output : torch.Tensor
            Gradient from downstream layers

        Returns
        -------
        tuple[torch.Tensor, None, None]
            Gradient with respect to v (using surrogate), None for v_threshold and alpha
        """
        v, v_threshold, alpha = ctx.saved_tensors

        # Sigmoid surrogate gradient (derivative of sigmoid function)
        shifted = alpha * (v - v_threshold)
        sigma = torch.sigmoid(shifted)
        grad_surrogate = alpha * sigma * (1 - sigma)

        grad_input = grad_output * grad_surrogate
        return grad_input, None, None


class LIFLayer(nn.Module):
    """
    Leaky Integrate-and-Fire neuron layer with surrogate gradients.

    Implements biologically-inspired neural dynamics with membrane potential
    leakage, spike generation, and reset. Enables gradient-based learning
    through differentiable spike approximation.

    Parameters
    ----------
    input_dim : int
        Dimension of input features
    output_dim : int
        Number of neurons in this layer
    tau_m : float
        Membrane time constant in milliseconds (default: 20.0)
    v_threshold : float
        Spike threshold voltage (default: 1.0)
    v_reset : float
        Reset potential after spike (default: 0.0)
    v_rest : float
        Resting membrane potential (default: 0.0)
    surrogate_alpha : float
        Surrogate gradient smoothness (default: 10.0)

    Attributes
    ----------
    fc : nn.Linear
        Linear transformation for input currents
    spike_fn : function
        Surrogate gradient spike function

    Notes
    -----
    Membrane potential dynamics follow:
        dv/dt = (v_rest - v + I_input) / tau_m

    When v >= v_threshold, neuron spikes and v → v_reset.
    """

    def __init__(  # noqa: PLR0913
        self,
        input_dim: int,
        output_dim: int,
        tau_m: float = 20.0,
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        v_rest: float = 0.0,
        surrogate_alpha: float = 10.0,
    ) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.tau_m: float = tau_m
        self.v_threshold: float = v_threshold
        self.v_reset: float = v_reset
        self.v_rest: float = v_rest
        self.surrogate_alpha: float = surrogate_alpha
        # Type annotation for spike function
        self.spike_fn: type[SurrogateGradientSpike] = SurrogateGradientSpike

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for one simulation timestep.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor [batch_size, input_dim]
        state : tuple[torch.Tensor, torch.Tensor] | None
            Previous (membrane_potential, spikes) or None to initialize

        Returns
        -------
        tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]
            (spikes, new_state) where:
            - spikes: [batch_size, output_dim] binary spike tensor
            - new_state: (membrane_potential, spikes) for next timestep
        """
        batch_size = x.shape[0]

        # Initialize membrane potential if no state provided
        if state is None:
            v_membrane = torch.full(
                (batch_size, self.fc.out_features),
                self.v_rest,
                device=x.device,
                dtype=x.dtype,
            )
        else:
            v_membrane, _ = state

        # Compute input current
        i_input = self.fc(x)

        # Update membrane potential using Euler integration
        # dv/dt = (v_rest - v + I) / tau_m
        decay = 1.0 - 1.0 / self.tau_m
        v_membrane = decay * v_membrane + (1.0 - decay) * self.v_rest + i_input

        # Generate spikes with surrogate gradient
        spikes = self.spike_fn.apply(v_membrane, self.v_threshold, self.surrogate_alpha)
        assert isinstance(spikes, torch.Tensor)  # noqa: S101

        # Reset membrane potential for neurons that spiked
        v_membrane = v_membrane * (1 - spikes) + self.v_reset * spikes

        new_state: tuple[torch.Tensor, torch.Tensor] = (v_membrane, spikes)
        return spikes, new_state


class RecurrentAdaptiveLIFState(NamedTuple):
    """Carried state of a :class:`RecurrentAdaptiveLIFCell`.

    Attributes
    ----------
    v : torch.Tensor
        Membrane potential ``[batch, num_neurons]``.
    a : torch.Tensor
        Adaptation variable ``[batch, num_neurons]`` driving the adaptive
        threshold.
    s : torch.Tensor
        Last emitted spikes ``[batch, num_neurons]`` (fed back through the
        recurrent connection on the next tick).
    """

    v: torch.Tensor
    a: torch.Tensor
    s: torch.Tensor


class RecurrentAdaptiveLIFCell(nn.Module):
    r"""Recurrent adaptive leaky-integrate-and-fire cell (one tick per call).

    A spiking recurrent core that carries ``(v, a, s)`` across calls. Per tick,
    for ``num_neurons`` units:

    - input current ``I = W_in . x + W_rec . s_prev`` — the encoded input
      current plus a learnable recurrent **spike-feedback** current;
    - membrane ``v <- beta * v + (1 - beta) * I`` with a learnable per-neuron
      decay ``beta = sigmoid(raw_beta)``;
    - adaptive threshold ``theta = v_threshold + adapt_scale * a``;
    - spike ``s = SurrogateGradientSpike(v - theta, slope)`` (compared against
      zero after subtracting the threshold so the threshold can adapt);
    - soft reset ``v <- v - s * theta`` (subtractive, gradient-friendly);
    - adaptation ``a <- rho * a + s`` with decay ``rho``.

    The surrogate ``slope`` is a runtime forward argument so it can be
    scheduled (shallow while exploring, sharper while fine-tuning).

    Parameters
    ----------
    input_dim : int
        Dimensionality of the encoded input current ``x``.
    num_neurons : int
        Number of recurrent LIF units.
    v_threshold : float
        Base spike threshold before adaptation (default: 1.0).
    membrane_decay_init : float
        Initial per-neuron membrane decay ``beta`` (default: 0.9). Stored as
        ``raw_beta = logit(membrane_decay_init)`` so the trained ``beta`` stays
        in ``(0, 1)`` via a sigmoid.
    adaptation_decay : float
        Decay ``rho`` of the adaptation variable ``a`` (default: 0.9). Fixed
        (not learned) — the adaptive threshold's gain is learned via
        ``adapt_scale`` instead.
    adapt_scale_init : float
        Initial per-neuron adaptive-threshold gain ``adapt_scale`` (default:
        0.1). Learnable.

    Notes
    -----
    ``adapt_scale`` and the recurrent weights are learnable; ``rho`` is a fixed
    decay. The membrane decay ``beta`` is learnable per neuron.
    """

    def __init__(  # noqa: PLR0913
        self,
        input_dim: int,
        num_neurons: int,
        v_threshold: float = 1.0,
        membrane_decay_init: float = 0.9,
        adaptation_decay: float = 0.9,
        adapt_scale_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_neurons = num_neurons
        self.v_threshold: float = v_threshold
        self.adaptation_decay: float = adaptation_decay
        self.spike_fn: type[SurrogateGradientSpike] = SurrogateGradientSpike

        # Recurrent spike-feedback weights (the spiking-RNN connection). No
        # bias: the encoded input current already carries the encoder bias.
        self.recurrent = nn.Linear(num_neurons, num_neurons, bias=False)

        # Learnable per-neuron membrane decay, stored as a logit so the
        # effective ``beta = sigmoid(raw_beta)`` is confined to ``(0, 1)``.
        raw_beta = _inverse_sigmoid(membrane_decay_init)
        self.raw_membrane_decay = nn.Parameter(torch.full((num_neurons,), raw_beta))

        # Learnable per-neuron adaptive-threshold gain.
        self.adapt_scale = nn.Parameter(torch.full((num_neurons,), float(adapt_scale_init)))

    @property
    def membrane_decay(self) -> torch.Tensor:
        """Effective per-neuron membrane decay ``beta = sigmoid(raw_beta)`` in ``(0, 1)``."""
        return torch.sigmoid(self.raw_membrane_decay)

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> RecurrentAdaptiveLIFState:
        """Return a zero-initialized carried state ``(v, a, s)``."""
        zeros = torch.zeros(batch_size, self.num_neurons, device=device, dtype=dtype)
        return RecurrentAdaptiveLIFState(v=zeros.clone(), a=zeros.clone(), s=zeros.clone())

    def forward(
        self,
        input_current: torch.Tensor,
        state: RecurrentAdaptiveLIFState | None = None,
        slope: float = 2.0,
    ) -> tuple[torch.Tensor, RecurrentAdaptiveLIFState]:
        """Advance the recurrent adaptive-LIF cell by one tick.

        Parameters
        ----------
        input_current : torch.Tensor
            Encoded input current ``W_in . x``, shape ``[batch, num_neurons]``.
        state : RecurrentAdaptiveLIFState | None
            Previous ``(v, a, s)`` or ``None`` to zero-initialize.
        slope : float
            Surrogate-gradient slope ``alpha`` for this tick's spike (passed to
            the surrogate backward; runtime-schedulable).

        Returns
        -------
        tuple[torch.Tensor, RecurrentAdaptiveLIFState]
            ``(spikes, new_state)`` where ``spikes`` has shape
            ``[batch, num_neurons]``.
        """
        batch_size = input_current.shape[0]
        if state is None:
            state = self.init_state(batch_size, input_current.device, input_current.dtype)
        v_prev, a_prev, s_prev = state

        # Total input current: encoded input + recurrent spike feedback.
        total_current = input_current + self.recurrent(s_prev)

        # Membrane integration with the learnable per-neuron decay.
        beta = self.membrane_decay
        v = beta * v_prev + (1.0 - beta) * total_current

        # Adaptive threshold and spike generation. Subtract the threshold and
        # compare against zero so the surrogate sees ``v - theta``.
        theta = self.v_threshold + self.adapt_scale * a_prev
        spikes = self.spike_fn.apply(v - theta, 0.0, slope)
        assert isinstance(spikes, torch.Tensor)  # noqa: S101

        # Subtractive (soft) reset and adaptation update.
        v = v - spikes * theta
        a = self.adaptation_decay * a_prev + spikes

        return spikes, RecurrentAdaptiveLIFState(v=v, a=a, s=spikes)


class LeakyIntegratorReadout(nn.Module):
    r"""Non-spiking leaky-integrator readout cell (one tick per call).

    Integrates incoming spikes into a carried output membrane and returns that
    membrane as a smooth, continuous output (better policy gradients than
    binary spikes):

        ``m <- beta_out * m + W_out . s``,  output ``= m``.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the incoming spike vector ``s``.
    output_dim : int
        Number of readout units (e.g. action count).
    readout_decay_init : float
        Initial output-membrane decay ``beta_out`` (default: 0.9). Stored as a
        logit so the trained decay stays in ``(0, 1)``.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        readout_decay_init: float = 0.9,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc = nn.Linear(input_dim, output_dim)
        raw_decay = _inverse_sigmoid(readout_decay_init)
        self.raw_readout_decay = nn.Parameter(torch.tensor(raw_decay))

    @property
    def readout_decay(self) -> torch.Tensor:
        """Effective output-membrane decay ``beta_out = sigmoid(raw_beta_out)`` in ``(0, 1)``."""
        return torch.sigmoid(self.raw_readout_decay)

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return a zero-initialized output membrane ``m``."""
        return torch.zeros(batch_size, self.output_dim, device=device, dtype=dtype)

    def forward(
        self,
        spikes: torch.Tensor,
        membrane: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance the readout membrane by one tick.

        Parameters
        ----------
        spikes : torch.Tensor
            Incoming spikes ``[batch, input_dim]``.
        membrane : torch.Tensor | None
            Previous output membrane ``m`` or ``None`` to zero-initialize.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(output, new_membrane)``; ``output`` IS the new membrane.
        """
        batch_size = spikes.shape[0]
        if membrane is None:
            membrane = self.init_state(batch_size, spikes.device, spikes.dtype)
        new_membrane = self.readout_decay * membrane + self.fc(spikes)
        return new_membrane, new_membrane


class SpikingPolicyNetwork(nn.Module):
    """
    Multi-layer spiking neural network for policy learning.

    Implements a deep spiking network with constant current input encoding,
    multiple LIF hidden layers with temporal dynamics, and configurable output
    modes for action selection. Designed for reinforcement learning tasks.

    Parameters
    ----------
    input_dim : int
        Dimension of state features
    hidden_dim : int
        Number of neurons in each hidden layer
    output_dim : int
        Number of output actions
    num_timesteps : int
        Simulation timesteps per forward pass
    num_hidden_layers : int
        Number of LIF hidden layers (default: 2)
    tau_m : float
        Membrane time constant (default: 20.0)
    v_threshold : float
        Spike threshold (default: 1.0)
    v_reset : float
        Reset potential (default: 0.0)
    v_rest : float
        Resting potential (default: 0.0)
    surrogate_alpha : float
        Surrogate gradient smoothness (default: 10.0)
    output_mode : OutputMode
        How to compute output from temporal spike dynamics:
        - "accumulator": Sum spikes over all timesteps (default, original behavior)
        - "final": Use only the final timestep's spike pattern (preserves temporal
          dynamics without accumulator smoothing)
        - "membrane": Use final membrane potentials instead of spikes (continuous
          values preserve more input-dependent variation)
    temporal_modulation : bool
        If True, apply sinusoidal modulation to input current over timesteps.
        This prevents LIF neurons from reaching steady state, creating richer
        temporal dynamics that vary with input magnitude (default: False).
    modulation_amplitude : float
        Amplitude of sinusoidal modulation as fraction of input (default: 0.3).
        Input is scaled by (1 + amplitude * sin(phase)).
    modulation_period : int
        Period of sinusoidal modulation in timesteps (default: 20).
        With 100 timesteps, this creates ~5 oscillation cycles.
    population_coding : bool
        If True, encode inputs using population coding with Gaussian tuning
        curves before the input layer. This creates distinct activation patterns
        for different input values (default: False).
    neurons_per_feature : int
        Number of encoding neurons per input feature when population_coding
        is enabled (default: 8).
    population_sigma : float
        Width of Gaussian tuning curves for population coding (default: 0.25).

    Architecture
    ------------
    Input features → [PopulationEncoder] → Linear → constant current
                              ↓ (repeated num_timesteps)
                            LIF Layer 1
                              ↓
                            LIF Layer 2
                              ↓
                            ... (num_hidden_layers)
                              ↓
                            [output_mode determines aggregation]
                              ↓
                            Linear → action logits
    """

    def __init__(  # noqa: PLR0913
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_timesteps: int,
        num_hidden_layers: int = 2,
        tau_m: float = 20.0,
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        v_rest: float = 0.0,
        surrogate_alpha: float = 10.0,
        output_mode: OutputMode = "accumulator",
        modulation_amplitude: float = 0.3,
        modulation_period: int = 20,
        neurons_per_feature: int = 8,
        population_sigma: float = 0.25,
        *,
        population_coding: bool = False,
        temporal_modulation: bool = False,
    ) -> None:
        super().__init__()
        self.num_timesteps = num_timesteps
        self.hidden_dim = hidden_dim
        self.output_mode = output_mode
        self.temporal_modulation = temporal_modulation
        self.modulation_amplitude = modulation_amplitude
        self.modulation_period = modulation_period
        self.population_coding = population_coding

        # Population coding encoder (optional)
        # Expands input_dim to input_dim * neurons_per_feature
        if population_coding:
            self.population_encoder = PopulationEncoder(
                input_dim=input_dim,
                neurons_per_feature=neurons_per_feature,
                sigma=population_sigma,
                min_val=-1.0,  # Inputs normalized to [-1, 1]
                max_val=1.0,
            )
            effective_input_dim = input_dim * neurons_per_feature
        else:
            self.population_encoder = None
            effective_input_dim = input_dim

        # Input encoding: maps (population-coded) features to constant input currents
        self.input_layer = nn.Linear(effective_input_dim, hidden_dim)

        # Spiking hidden layers with recurrent dynamics
        self.hidden_layers = nn.ModuleList(
            [
                LIFLayer(
                    hidden_dim,
                    hidden_dim,
                    tau_m=tau_m,
                    v_threshold=v_threshold,
                    v_reset=v_reset,
                    v_rest=v_rest,
                    surrogate_alpha=surrogate_alpha,
                )
                for _ in range(num_hidden_layers)
            ],
        )

        # Output layer: maps spikes/membrane to action logits
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through spiking network.

        Simulates the network for num_timesteps and produces action logits
        using the configured output_mode.

        Parameters
        ----------
        x : torch.Tensor
            State features [batch_size, input_dim]

        Returns
        -------
        torch.Tensor
            Action logits [batch_size, output_dim]
        """
        batch_size = x.shape[0]

        # Apply population coding if enabled
        if self.population_coding and self.population_encoder is not None:
            x = self.population_encoder(x)

        # Encode input as constant current (applied at each timestep)
        input_current = self.input_layer(x)  # [batch_size, hidden_dim]

        # Initialize hidden states for each layer
        hidden_states: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(
            self.hidden_layers,
        )

        # Simulate for num_timesteps
        spike_accumulator = torch.zeros(
            batch_size,
            self.hidden_dim,
            device=x.device,
            dtype=x.dtype,
        )
        final_spikes = torch.zeros(
            batch_size,
            self.hidden_dim,
            device=x.device,
            dtype=x.dtype,
        )

        for t in range(self.num_timesteps):
            # Apply temporal modulation if enabled
            # This prevents LIF neurons from reaching steady state by varying input over time
            if self.temporal_modulation:
                # Sinusoidal modulation: input * (1 + amplitude * sin(2π * t / period))
                phase = 2.0 * math.pi * t / self.modulation_period
                modulation_factor = 1.0 + self.modulation_amplitude * math.sin(phase)
                h = input_current * modulation_factor
            else:
                h = input_current  # Feed same input at each timestep

            # Pass through spiking hidden layers
            for i, layer in enumerate(self.hidden_layers):
                h, hidden_states[i] = layer(h, hidden_states[i])

            # Accumulate spikes over time (for accumulator mode)
            spike_accumulator += h

            # Store final timestep spikes (for final mode)
            if t == self.num_timesteps - 1:
                final_spikes = h

        # Store spike rate statistics for monitoring (accessible via last_spike_rates)
        # Detach to prevent memory leak from computation graph retention
        self.last_spike_rates = (spike_accumulator / self.num_timesteps).detach()

        # Select output based on configured mode
        if self.output_mode == "final":
            # Use only final timestep's spike pattern
            # This preserves temporal dynamics without accumulator smoothing
            # Scale up to compensate for single timestep (vs accumulated counts)
            output_features = final_spikes * self.num_timesteps
        elif self.output_mode == "membrane":
            # Use final membrane potentials (continuous values)
            # This preserves more input-dependent variation than binary spikes
            # Get membrane potentials from last hidden layer's state
            if hidden_states[-1] is not None:
                final_membrane, _ = hidden_states[-1]
                output_features = final_membrane
            else:
                output_features = spike_accumulator  # Fallback
        else:  # "accumulator" (default)
            # Sum spikes over all timesteps (original behavior)
            # Spike counts range from 0 to num_timesteps (e.g., 0-100)
            output_features = spike_accumulator

        # Convert to action logits
        return self.output_layer(output_features)
