# Implementation Tasks

## 1. Create Surrogate Gradient Infrastructure
- [ ] 1.1 Create `packages/quantum-nematode/quantumnematode/brain/arch/_spiking_layers.py`
- [ ] 1.2 Implement `SurrogateGradientSpike` custom autograd function with sigmoid surrogate
- [ ] 1.3 Implement `LIFLayer` module with forward/backward pass and stateful membrane potential
- [ ] 1.4 Implement `SpikingPolicyNetwork` multi-layer architecture with timestep simulation
- [ ] 1.5 Add unit tests for gradient flow (finite difference check on LIFLayer)

## 2. Rewrite SpikingBrain Class
- [ ] 2.1 Update `SpikingBrainConfig` schema (remove STDP params, add policy gradient params)
- [ ] 2.2 Rewrite `__init__` to instantiate `SpikingPolicyNetwork` and Adam optimizer
- [ ] 2.3 Rewrite `preprocess` to compute relative angles (match MLPBrain exactly)
- [ ] 2.4 Rewrite `run_brain` to forward through policy network and sample actions
- [ ] 2.5 Rewrite `learn` to implement REINFORCE with discounted returns and baseline
- [ ] 2.6 Remove old classes: `LIFNeuron`, `STDPRule`, encoding/simulation methods
- [ ] 2.7 Keep protocol compliance: `copy()`, `prepare_episode()`, `post_process_episode()`
- [ ] 2.8 Add integration test: 10 episodes with learning enabled, check loss decreases

## 3. Update Configuration Files
- [ ] 3.1 Update `configs/examples/spiking_foraging_small.yml` with new parameter schema
- [ ] 3.2 Update `configs/examples/spiking_foraging_medium.yml` with new parameter schema
- [ ] 3.3 Update `configs/examples/spiking_foraging_large.yml` with new parameter schema
- [ ] 3.4 Update `configs/examples/spiking_static_medium.yml` with new parameter schema
- [ ] 3.5 Add smoke tests: Run all 4 configs for 5 episodes each, verify no crashes

## 4. Validation & Testing
- [ ] 4.1 Run type checking (`uv run pyright`) - ensure no new type errors
- [ ] 4.2 Run linting (`uv run ruff check`) - ensure no new lint errors
- [ ] 4.3 Run unit tests (`uv run pytest tests/`) - ensure all pass
- [ ] 4.4 Manual test: `python scripts/run_simulation.py --brain spiking --config configs/examples/spiking_foraging_small.yml --num-runs 10`
- [ ] 4.5 Verify learning: Plot rewards vs episode, check positive trend

## 5. Benchmarking
- [ ] 5.1 Run 100 episodes on `spiking_foraging_small.yml`, record results
- [ ] 5.2 Compare learning curve to MLP baseline (same config, same seed)
- [ ] 5.3 If successful (>50% success rate by episode 100), run medium environment
- [ ] 5.4 Document performance metrics: success rate, average reward, convergence speed
- [ ] 5.5 Document hyperparameter sensitivity: `num_timesteps`, `hidden_dim`, `learning_rate`

## 6. Documentation & Cleanup
- [ ] 6.1 Update docstrings in `spiking.py` to reflect new implementation
- [ ] 6.2 Add comments explaining surrogate gradient method
- [ ] 6.3 Update configuration example comments with new parameters
- [ ] 6.4 (Optional) Archive old STDP implementation in `spiking_stdp_legacy.py` for reference
- [ ] 6.5 Update project documentation noting breaking changes from STDP version

## 7. OpenSpec Archiving
- [ ] 7.1 Verify all tasks completed
- [ ] 7.2 Run `openspec validate rewrite-spiking-surrogate-gradients --strict`
- [ ] 7.3 Run `openspec archive rewrite-spiking-surrogate-gradients --yes` after deployment
- [ ] 7.4 Verify specs updated correctly with `openspec show brain-architecture --type spec`
