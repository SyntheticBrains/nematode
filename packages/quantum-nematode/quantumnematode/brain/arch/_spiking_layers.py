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

import torch
from torch import nn


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
        ∂spike/∂v ≈ α·σ(α(v - v_th))·(1 - σ(α(v - v_th)))

    where σ is the sigmoid function. This approximation is only used during
    backward pass; forward pass uses true Heaviside step function.
    """

    @staticmethod
    def forward(ctx, v: torch.Tensor, v_threshold: float, alpha: float = 10.0) -> torch.Tensor:
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
        ctx.save_for_backward(v, torch.tensor(v_threshold, device=v.device), torch.tensor(alpha, device=v.device))
        return spike

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, *args) -> tuple[torch.Tensor, None, None]:  # type: ignore[override]
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

        # Sigmoid surrogate gradient: α·σ(α(v - v_th))·(1 - σ(α(v - v_th)))
        # This is equivalent to the derivative of the sigmoid function
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

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        tau_m: float = 20.0,
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        v_rest: float = 0.0,
        surrogate_alpha: float = 10.0,
    ):
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
        self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor] | None = None,
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
        assert isinstance(spikes, torch.Tensor)  # Type hint for pyright

        # Reset membrane potential for neurons that spiked
        v_membrane = v_membrane * (1 - spikes) + self.v_reset * spikes

        new_state: tuple[torch.Tensor, torch.Tensor] = (v_membrane, spikes)
        return spikes, new_state


class SpikingPolicyNetwork(nn.Module):
    """
    Multi-layer spiking neural network for policy learning.

    Implements a deep spiking network with constant current input encoding,
    multiple LIF hidden layers with temporal dynamics, and spike accumulation
    for action selection. Designed for reinforcement learning tasks.

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

    Architecture
    ------------
    Input features → Linear → constant current
                              ↓ (repeated num_timesteps)
                            LIF Layer 1
                              ↓
                            LIF Layer 2
                              ↓
                            ... (num_hidden_layers)
                              ↓
                            Sum spikes over time
                              ↓
                            Linear → action logits
    """

    def __init__(
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
    ):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.hidden_dim = hidden_dim

        # Input encoding: maps state features to constant input currents
        self.input_layer = nn.Linear(input_dim, hidden_dim)

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

        # Output layer: maps accumulated spikes to action logits
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through spiking network.

        Simulates the network for num_timesteps, accumulating spikes across
        time to produce action logits.

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

        # Encode input as constant current (applied at each timestep)
        input_current = self.input_layer(x)  # [batch_size, hidden_dim]

        # Initialize hidden states for each layer
        hidden_states = [None] * len(self.hidden_layers)

        # Simulate for num_timesteps
        spike_accumulator = torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)

        for _ in range(self.num_timesteps):
            h = input_current  # Feed same input at each timestep

            # Pass through spiking hidden layers
            for i, layer in enumerate(self.hidden_layers):
                h, hidden_states[i] = layer(h, hidden_states[i])

            # Accumulate spikes over time
            spike_accumulator += h

        # Convert total spike counts to action logits
        action_logits = self.output_layer(spike_accumulator)

        return action_logits
