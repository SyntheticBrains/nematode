# Standardization Decisions

**Last Updated**: February 2026

This document records our evaluation of external frameworks and the rationale for our choices.

______________________________________________________________________

## Environment: Keep Custom (No Gymnasium)

**Decision**: Keep `DynamicForagingEnvironment` as-is. Add a Gymnasium wrapper only if external collaboration requires it.

**Rationale**:

- Our environment uses domain-specific multi-modal Pydantic params (`BrainParams` with quantum rotation fields RX/RY/RZ, thermotaxis, mechanosensation) that don't map cleanly to Gymnasium's `Box`/`Dict` observation spaces.
- Biological fidelity features (AFD neuron simulation, satiety system, health/HP model) are tightly integrated with custom state management.
- We don't currently use any Gymnasium-dependent libraries (Stable-Baselines3, RLlib, CleanRL). Our brain architectures have their own training loops.
- Migration cost is high (~19 files in `env/` module plus extensive tests) with no immediate benefit.

**Reconsider if**:

- We want to integrate with Stable-Baselines3 or similar for additional SOTA baselines.
- External collaborators need a standard interface.
- We adopt a framework that requires Gymnasium (e.g., RLlib for distributed training).

**Optional future work**: A thin `GymnasiumWrapper` that flattens observations could be added with minimal effort if needed.

______________________________________________________________________

## Configuration: Keep Pydantic + YAML (No Hydra)

**Decision**: Keep the current Pydantic BaseModel + YAML configuration system.

**Rationale**:

- The current `config_loader.py` is mature and handles all our needs: brain-specific config validation, environment params, reward tuning, learning rate scheduling.
- Pydantic provides runtime type validation with clear error messages, which is stronger than Hydra's `DictConfig`.
- We don't need Hydra's key features: multirun sweeps (we use CMA-ES for hyperparameter optimization), config composition (our configs are self-contained YAML files), or CLI overrides (we use config files directly).
- Migration would require rewriting 47+ config files and the config loader.

**Reconsider if**:

- We need systematic hyperparameter sweeps beyond what CMA-ES provides.
- Config composition becomes necessary (e.g., mixing brain configs with different environment configs dynamically).
- The config file count grows significantly and we need better organization.

______________________________________________________________________

## Benchmarking: Enhance NematodeBench (No External Framework)

**Decision**: Continue developing NematodeBench as our custom benchmarking system.

**Rationale**:

- No standard RL benchmarking framework fits our domain-specific needs: chemotaxis indices, thermotaxis precision, biological validation metrics, multi-objective survival scores.
- NematodeBench already has: submission validation, leaderboard generation, convergence detection, session aggregation across 10+ runs.
- Standard options evaluated:
  - **OpenAI Gym Monitor**: Too basic (just episode rewards/lengths).
  - **RLBench**: Focused on robotics manipulation tasks.
  - **Weights & Biases**: Good for experiment tracking but doesn't replace our domain-specific metrics or leaderboard system.

**Planned enhancements** (Phase 2):

- Hierarchical benchmark categories (from OpenSpec remaining tasks)
- Statistical testing framework (confidence intervals, significance tests, effect sizes)
- Improved export formats and visualizations
- Spiking as a separate brain class alongside quantum/classical

______________________________________________________________________

## Summary

| Area | Decision | Framework | Reason |
|------|----------|-----------|--------|
| Environment | Keep custom | — | Domain-specific multi-modal sensing, quantum params |
| Configuration | Keep custom | Pydantic + YAML | Type-safe, mature, no sweep needs |
| Benchmarking | Enhance custom | NematodeBench | Domain-specific metrics, no standard fits |
