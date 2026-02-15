## Why

QSNNReinforce achieves 67% foraging success but fails to learn predator evasion due to high REINFORCE gradient variance — single-sample returns provide noisy learning signal that quantum shot noise compounds further. QSNN-PPO was built to address this but its importance sampling ratio is always 1.0 (the `QLIFSurrogateSpike` forward pass returns a parameter-independent constant from quantum measurement, making `pi_new == pi_old` always). Rather than stripping quantum character from the forward pass to fix PPO, adding a classical critic to QSNNReinforce provides variance reduction while preserving quantum circuit participation in both forward and backward passes.

## What Changes

- Add a classical MLP critic network to QSNNReinforce that estimates state values `V(s)` from sensory features and hidden spike rates
- Replace raw REINFORCE discounted returns with advantage estimates `A_t = R_t - V(s_t)` for variance reduction
- Add critic loss (MSE or Huber on value predictions vs actual returns) to the training loop
- Add critic-specific configuration parameters (hidden dim, num layers, learning rate, value loss coefficient)
- Add optional GAE (Generalized Advantage Estimation) as an alternative to simple advantage baseline
- Reuse `CriticMLP` infrastructure already proven in QSNN-PPO (adapted for REINFORCE update pattern)
- Add new config files for A2C predator experiments

## Capabilities

### New Capabilities

_(none — this extends an existing capability)_

### Modified Capabilities

- `qsnn-reinforce-brain`: Add actor-critic (A2C) variance reduction mode with classical MLP critic. New configuration parameters for critic architecture and advantage estimation. Existing REINFORCE-only mode remains as default when critic is disabled.

## Impact

- **Code**: `qsnnreinforce.py` — add CriticMLP, advantage computation, critic training loop, new config fields
- **Code**: `_qlif_layers.py` — no changes (QLIF forward/backward passes unchanged)
- **Code**: `config_loader.py` — new config fields for critic parameters
- **Tests**: `test_qsnnreinforce.py` — new tests for critic, advantage computation, A2C training loop
- **Configs**: New `qsnnreinforce_a2c_*.yml` experiment configs
- **Dependencies**: None (uses existing PyTorch nn.Module infrastructure)
- **Breaking changes**: None — critic is opt-in via configuration, existing configs work unchanged
