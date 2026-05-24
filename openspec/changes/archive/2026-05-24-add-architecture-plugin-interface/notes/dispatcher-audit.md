# Dispatcher Audit

Audit of `packages/quantum-nematode/quantumnematode/utils/brain_factory.py` `setup_brain_model()` prior to the registry refactor. Sizing + branch enumeration ground the refactor scope.

## Sizing

- File LOC: 459
- `setup_brain_model()` body: lines 51-459
- Branch count: 19 dispatched architectures (1 `if` + 18 `elif`)
- Fallback: `ValueError("Unknown brain type: {brain_type}")` at the end

## Branch enumeration

| Line | BrainType member | Config class | Brain module | Family tags (proposed) |
|------|------------------|--------------|--------------|------------------------|
| 106 | QVARCIRCUIT | QVarCircuitBrainConfig | qvarcircuit | (quantum,) |
| 138 | QQLEARNING | QQLearningBrainConfig | qqlearning | (quantum,) |
| 167 | MLP_REINFORCE | MLPReinforceBrainConfig | mlpreinforce | (classical,) |
| 188 | MLP_PPO | MLPPPOBrainConfig | mlpppo | (classical,) |
| 207 | MLP_DQN | MLPDQNBrainConfig | mlpdqn | (classical,) |
| 227 | SPIKING_REINFORCE | SpikingReinforceBrainConfig | spikingreinforce | (spiking,) |
| 244 | QRC | QRCBrainConfig | qrc | (quantum, classical) |
| 260 | QRH | QRHBrainConfig | qrh | (quantum,) |
| 276 | QEF | QEFBrainConfig | qef | (quantum,) |
| 292 | CRH | CRHBrainConfig | crh | (classical,) |
| 308 | QSNN_REINFORCE | QSNNReinforceBrainConfig | qsnnreinforce | (quantum, spiking) |
| 324 | HYBRID_QUANTUM | HybridQuantumBrainConfig | hybridquantum | (quantum,) |
| 340 | HYBRID_QUANTUM_CORTEX | HybridQuantumCortexBrainConfig | hybridquantumcortex | (quantum,) |
| 357 | HYBRID_CLASSICAL | HybridClassicalBrainConfig | hybridclassical | (classical,) |
| 374 | QSNN_PPO | QSNNPPOBrainConfig | qsnnppo | (quantum, spiking) |
| 390 | QLIF_LSTM | QLIFLSTMBrainConfig | qliflstm | (quantum,) |
| 406 | QRH_QLSTM | QRHQLSTMBrainConfig | qrhqlstm | (quantum,) |
| 422 | CRH_QLSTM | CRHQLSTMBrainConfig | crhqlstm | (classical,) |
| 438 | LSTM_PPO | LSTMPPOBrainConfig | lstmppo | (classical,) |

## Files touched today when adding a new architecture

1. `brain/arch/dtypes.py` — add `BrainType` enum member; add to `BRAIN_TYPES` Literal; add to one of `QUANTUM_BRAIN_TYPES` / `CLASSICAL_BRAIN_TYPES` / `SPIKING_BRAIN_TYPES`
2. `brain/arch/<name>.py` — new file with `<Name>BrainConfig` + `<Name>Brain` classes
3. `brain/arch/__init__.py` — re-export the new brain + config
4. `utils/brain_factory.py` — add an `elif` branch + import the config + import the brain class lazily
5. `utils/config_loader.py` — add the YAML name → config-class mapping entry in `BRAIN_CONFIG_MAP`
6. `tests/.../brain/arch/test_<name>.py` — unit tests

## Branch repetition pattern

Every branch repeats this shape (sample from `MLP_PPO`):

```python
elif brain_type == BrainType.MLP_PPO:
    if not isinstance(brain_config, MLPPPOBrainConfig):
        error_message = (
            "The 'mlpppo' brain architecture requires a MLPPPOBrainConfig. "
            f"Provided brain config type: {type(brain_config)}."
        )
        logger.error(error_message)
        raise ValueError(error_message)
    from quantumnematode.brain.arch.mlpppo import MLPPPOBrain
    brain_model = MLPPPOBrain(...)
```

The registry collapses these branches to a single `instantiate_brain(name, config, **kwargs)` call where the config-class type check + lazy brain-class import are factored into the registry plumbing.

## Cross-reference

Per the L1 refactor plan in `design.md`: the registry pattern handles dispatch via a single decorator-registration table; topology + rule factoring is internal-only for the migrated architectures (the external `Brain` Protocol surface at `packages/quantum-nematode/quantumnematode/brain/arch/_brain.py:346-368` is unchanged).
