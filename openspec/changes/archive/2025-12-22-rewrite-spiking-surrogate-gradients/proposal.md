# Change: Rewrite Spiking Neural Network with Surrogate Gradient Descent

## Why

The current SpikingBrain implementation using Spike-Timing Dependent Plasticity (STDP) has **critical implementation bugs** that prevent learning entirely, including:
- Wrong input preprocessing (uses absolute angles instead of relative angles to goal)
- Broken temporal credit assignment (uses episode average reward instead of discounted returns)
- Missing recurrent weight updates
- Incorrect STDP depression formula

Beyond bugs, STDP has **fundamental architectural limitations** for reinforcement learning tasks:
- Local learning rules cannot implement global credit assignment
- Sparse spike coincidence requirements create very sparse learning signals
- Stochastic Poisson encoding adds unnecessary noise
- Cannot leverage proven gradient-based optimization techniques

After 400 episodes across 4 experimental runs (100 episodes each), the spiking agent showed **zero learning** with 0% success rate, while MLP and quantum modular agents achieve consistent learning and high success rates on identical tasks.

**State-of-the-art neuromorphic research** uses surrogate gradient descent, which combines biological plausibility (spiking neurons) with effective gradient-based learning. This is the approach used by leading spiking neural network libraries (SpikingJelly, snnTorch, Norse).

## What Changes

This is a **complete architectural rewrite** of the spiking neural network implementation:

1. **Replace STDP with Surrogate Gradient Descent**
   - Remove: Manual spike-timing analysis and weight updates
   - Add: Differentiable spike function approximation
   - Add: Backpropagation through time (BPTT)
   - Add: PyTorch autograd integration

2. **Implement Policy Gradient Learning (REINFORCE)**
   - Use same proven algorithm as MLPBrain
   - Discounted returns with baseline subtraction
   - Gradient clipping and Adam optimization
   - Episode-level batch updates

3. **Fix Input Preprocessing**
   - Compute relative angles (agent-facing direction → goal direction)
   - Match MLPBrain preprocessing exactly
   - Remove stochastic Poisson encoding
   - Use constant current input over simulation timesteps

4. **New Layer Architecture**
   - Create `_spiking_layers.py` module with:
     - `SurrogateGradientSpike`: Custom PyTorch autograd function
     - `LIFLayer`: Stateful LIF neuron layer with gradient flow
     - `SpikingPolicyNetwork`: Multi-layer spiking network
   - Replace manual neuron simulation with PyTorch modules
   - Maintain biological plausibility (LIF dynamics, spike-based computation)

5. **Update Configuration Schema**
   - Remove STDP parameters: `tau_plus, tau_minus, a_plus, a_minus, reward_scaling`
   - Remove Poisson encoding parameters: `simulation_duration, time_step, max_rate, min_rate`
   - Add policy gradient parameters: `gamma, baseline_alpha, entropy_beta`
   - Add network structure parameters: `num_timesteps, num_hidden_layers, surrogate_alpha`

6. **Maintain Compatibility**
   - Keep `ClassicalBrain` protocol interface
   - Keep existing config file structure (update parameter values only)
   - Keep brain factory registration ("spiking" brain type)
   - Keep CLI arguments unchanged

## Impact

**Affected Specs:**
- `brain-architecture`: Complete rewrite of spiking requirements (MODIFIED: 6 requirements)

**Affected Code:**
- `packages/quantum-nematode/quantumnematode/brain/arch/spiking.py` (complete rewrite, ~700 lines)
- `packages/quantum-nematode/quantumnematode/brain/arch/_spiking_layers.py` (new file, ~150 lines)
- `configs/examples/spiking_foraging_small.yml` (update parameters)
- `configs/examples/spiking_foraging_medium.yml` (update parameters)
- `configs/examples/spiking_foraging_large.yml` (update parameters)
- `configs/examples/spiking_static_medium.yml` (update parameters)

**Breaking Changes:**
- **BREAKING**: Existing spiking brain checkpoints/saved models incompatible
- **BREAKING**: STDP-specific config parameters removed
- **BREAKING**: Different hyperparameters required for new algorithm

**Benefits:**
- Proven learning algorithm (same as successful MLPBrain)
- Dense gradient signals (every step contributes to learning)
- Deterministic forward pass (reduced variance)
- Compatible with standard RL techniques
- Biological plausibility maintained (LIF neurons, spike-based computation)
- Comparable to state-of-the-art neuromorphic implementations

**Risks:**
- May not match MLP performance (more complex dynamics)
- More hyperparameters to tune
- Computational cost of BPTT through spiking timesteps

**Migration:**
- Users must update configuration files to new parameter schema
- Existing spiking brain results cannot be directly compared (different algorithm)
- Documentation must clearly indicate this is a new implementation
