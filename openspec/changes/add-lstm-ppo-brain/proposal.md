## Why

MLP PPO plateaus at ~60% of oracle performance on pursuit predators with derivative temporal sensing. After 6 rounds of experiments (reward tuning, curriculum learning, hyperparameter sweeps, LSTM bolt-on, 8000 episodes), the ~L500≈5.9 ceiling is confirmed as structural. The agent has excellent evasion skills (92% evasion rate) but navigates 3.2x slower than oracle, creating fatal predator exposure. The bottleneck is memoryless per-step processing — the MLP cannot learn sequential temporal patterns needed for efficient navigation with temporal sensing.

## What Changes

- **New brain architecture**: `LSTMPPOBrain` — a classical LSTM/GRU + PPO brain with proper chunk-based truncated BPTT training, designed specifically for temporal sensing tasks
- **New rollout buffer**: `LSTMPPORolloutBuffer` — stores per-step LSTM hidden states for proper recurrent PPO training with sequential chunk processing
- **Architecture**: Features → LayerNorm → LSTM → Actor MLP / Critic MLP (detached hidden state)
- **Training**: Chunk-based truncated BPTT (not shuffled minibatches), separate actor/critic optimizers, entropy decay, LR warmup/decay
- **GRU option**: `rnn_type: "lstm" | "gru"` for ablation experiments
- **Sensory module support**: Full compatibility with temporal/derivative sensing modules and STAM
- **Weight persistence**: `get_weight_components()` / `load_weight_components()` for curriculum learning
- **Example configs**: Derivative foraging and pursuit predator configurations

## Capabilities

### New Capabilities

- `lstm-ppo-brain`: LSTM/GRU-augmented PPO brain with chunk-based truncated BPTT, designed for temporal sensing. Replaces memoryless MLP processing with sequential temporal memory.

### Modified Capabilities

- `configuration-system`: New `lstmppo` brain type registered in BRAIN_CONFIG_MAP with `LSTMPPOBrainConfig`.

## Impact

- **New file**: `brain/arch/lstmppo.py` — LSTMPPOBrainConfig, LSTMPPORolloutBuffer, LSTMPPOBrain
- **Modified**: `brain/arch/__init__.py` — exports
- **Modified**: `brain/arch/dtypes.py` — BrainType enum, CLASSICAL_BRAIN_TYPES
- **Modified**: `utils/config_loader.py` — BRAIN_CONFIG_MAP
- **Modified**: `utils/brain_factory.py` — instantiation handler
- **New configs**: lstmppo derivative foraging and pursuit predator configs
- **Backward compatibility**: No changes to existing brains. MLP PPO unchanged.
- **No new dependencies**: Uses only PyTorch nn.LSTM/nn.GRU (already available).
