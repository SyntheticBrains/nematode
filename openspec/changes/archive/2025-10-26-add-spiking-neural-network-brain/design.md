# Technical Design: Spiking Neural Network Brain Architecture

## Architecture Overview

The SpikingBrain implementation follows the existing ClassicalBrain protocol while introducing temporal neural dynamics through spiking mechanisms. The design emphasizes biological plausibility while maintaining compatibility with the existing reinforcement learning framework.

## Core Components

### 1. Neuron Models

#### Leaky Integrate-and-Fire (LIF) Neuron
```python
class LIFNeuron:
    def __init__(self, tau_m: float = 20.0, v_threshold: float = 1.0, v_reset: float = 0.0)
    def step(self, input_current: float, dt: float) -> bool  # Returns True if spike occurs
```

**Parameters:**
- `tau_m`: Membrane time constant (ms)
- `v_threshold`: Spike threshold voltage
- `v_reset`: Reset voltage after spike
- `v_rest`: Resting potential (default 0.0)

**Dynamics:**
```
dV/dt = (v_rest - V + R*I) / tau_m
```

### 2. Network Architecture

#### Topology
- **Input Layer**: 2 neurons (gradient_strength, gradient_direction)
- **Hidden Layer**: Configurable size (default 32 neurons)
- **Output Layer**: 4 neurons (forward, left, right, stay actions)

#### Connectivity
- Fully connected between layers
- Recurrent connections within hidden layer (optional)
- Excitatory and inhibitory synapses

### 3. Encoding and Decoding

#### Input Encoding (Rate Coding)
```python
def encode_state(self, state: List[float], duration: float) -> List[List[float]]:
    # Convert continuous state values to spike rates
    # Generate Poisson spike trains for each input neuron
```

#### Output Decoding
```python
def decode_actions(self, spike_counts: List[int], duration: float) -> List[ActionData]:
    # Convert output spike counts to action probabilities
    # Apply softmax to spike rates for action selection
```

### 4. Learning Algorithm

#### Spike-Timing Dependent Plasticity (STDP)
```python
class STDPRule:
    def __init__(self, tau_plus: float = 20.0, tau_minus: float = 20.0, 
                 A_plus: float = 0.01, A_minus: float = 0.01)
    def update_weight(self, w: float, delta_t: float, reward_signal: float) -> float
```

**STDP Window:**
- Pre-before-post: Potentiation (strengthen synapse)
- Post-before-pre: Depression (weaken synapse)
- Reward modulation: Scale updates by reward signal

#### Learning Process
1. **Forward Pass**: Simulate network for episode duration
2. **Spike Collection**: Record all spike times and neuron activities
3. **Reward Assignment**: Apply reward signal to recent spike patterns
4. **Weight Updates**: Update synaptic weights using reward-modulated STDP
5. **Homeostasis**: Apply weight normalization and stability mechanisms

## Implementation Details

### 5. SpikingBrain Class Structure

```python
class SpikingBrain(ClassicalBrain):
    def __init__(self, config: SpikingBrainConfig):
        # Initialize network topology
        # Create neuron populations
        # Initialize synaptic weights
        # Setup STDP learning rule
        
    def run_brain(self, params: BrainParams, reward: float | None, 
                  input_data: List[float] | None, *, top_only: bool, 
                  top_randomize: bool) -> List[ActionData]:
        # Encode input state to spike trains
        # Simulate network dynamics for episode duration
        # Decode output spikes to action probabilities
        # Return selected actions
        
    def learn(self, params: BrainParams, reward: float, *, episode_done: bool) -> None:
        # Apply reward-modulated STDP to recent spike patterns
        # Update synaptic weights
        # Perform weight normalization
        
    def update_memory(self, reward: float | None) -> None:
        # Store episode experience for batch learning
        # Maintain spike pattern history
        
    def post_process_episode(self) -> None:
        # Batch weight updates if configured
        # Apply homeostatic mechanisms
        # Clear episode-specific data
```

### 6. Configuration Schema

```yaml
# Example spiking brain configuration
brain:
  type: spiking
  hidden_size: 32
  simulation_duration: 100.0  # ms per decision
  time_step: 1.0  # ms
  
  neuron:
    model: lif
    tau_m: 20.0
    v_threshold: 1.0
    v_reset: 0.0
    
  encoding:
    method: rate_coding
    max_rate: 100.0  # Hz
    min_rate: 0.0
    
  plasticity:
    rule: stdp
    tau_plus: 20.0
    tau_minus: 20.0
    learning_rate: 0.001
    reward_scaling: 1.0
    
  initialization:
    weight_mean: 0.1
    weight_std: 0.05
    weight_clip: 1.0
```

## Performance Considerations

### 7. Computational Efficiency

- **Sparse Computation**: Only process neurons that receive spikes
- **Event-driven Simulation**: Skip time steps with no activity
- **Batch Processing**: Group similar operations for efficiency
- **Memory Management**: Reuse spike buffers and temporary arrays

### 8. Numerical Stability

- **Weight Clipping**: Prevent runaway synaptic weights
- **Spike Rate Limiting**: Cap maximum firing rates
- **Gradient Clipping**: Stabilize learning updates
- **Regularization**: Add weight decay to prevent overfitting

## Integration Points

### 9. Existing Framework Integration

- **BrainData**: Extend to include spike timing information
- **Configuration**: Add spiking-specific parameters to YAML schema
- **CLI**: Support `--brain spiking` option in run_simulation.py
- **Testing**: Integrate with existing test infrastructure

### 10. Monitoring and Debugging

- **Spike Monitoring**: Track spike rates and patterns
- **Weight Visualization**: Monitor synaptic weight evolution
- **Network Activity**: Visualize population dynamics
- **Learning Metrics**: Track STDP updates and convergence

## Future Extensions

### 11. Advanced Features

- **Multiple Neuron Types**: Excitatory/inhibitory populations
- **Complex Topologies**: Hierarchical and modular structures
- **Temporal Coding**: Beyond simple rate coding schemes
- **Homeostatic Plasticity**: Activity-dependent scaling
- **Neuromorphic Hardware**: Preparation for hardware acceleration

### 12. Research Capabilities

- **Comparative Analysis**: Performance vs quantum/classical brains
- **Biological Validation**: Comparison with C. elegans neural data
- **Learning Dynamics**: Analysis of temporal learning patterns
- **Robustness Testing**: Noise tolerance and fault resilience
