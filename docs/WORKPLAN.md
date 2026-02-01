# Work Plan: Current Development Focus

**Last Updated**: February 2026

This file tracks tactical, near-term work. The [roadmap](roadmap.md) covers strategic direction.

______________________________________________________________________

## Active Work Streams

### Stream 1: Brain Architecture Renaming

Rename all brain architectures to use paradigm prefix + algorithm naming.

| Old Name | New Name | Config key |
|----------|----------|------------|
| ModularBrain | QVarCircuitBrain | qvarcircuit |
| QModularBrain | QQLearningBrain | qqlearning |
| MLPBrain | MLPReinforceBrain | mlpreinforce |
| PPOBrain | MLPPPOBrain | mlpppo |
| QMLPBrain | MLPDQNBrain | mlpdqn |
| SpikingBrain | SpikingReinforceBrain | spikingreinforce |

- [x] Finalize naming scheme
- [x] Update BrainType enum (`brain/arch/dtypes.py`)
- [x] Add new class names with old as deprecated aliases (6 brain files)
- [x] Update `brain/arch/__init__.py` exports
- [x] Update config_loader.py match cases (accept both old and new names)
- [x] Migrate ~47 config files in `configs/examples/`
- [x] Update `benchmark/categorization.py`
- [x] Update AGENTS.md, README.md references
- [x] Rename config files from old brain names to new brain names
- [x] Rename brain .py files to canonical names
- [x] Rename actual class definitions to canonical names (old names as deprecated aliases)
- [x] Update OpenSpec `add-ablation-toolkit` brain name references

### Stream 2: Roadmap Phase Restructuring

- [x] Move ablation toolkit from Phase 1 → Phase 2
- [x] Move oxygen sensing from Phase 1 → Phase 3
- [x] Expand Phase 2 to include standardization decisions + brain naming
- [x] Update brain names throughout roadmap
- [x] Bump roadmap version to 2.0

### Stream 3: Standardization Decisions

- [x] Document Gymnasium decision (keep custom, see [STANDARDIZATION.md](STANDARDIZATION.md))
- [x] Document Hydra decision (keep Pydantic + YAML)
- [x] Document benchmarking decision (enhance NematodeBench)

### Stream 4: OpenSpec Cleanup

- [ ] Complete `add-multi-sensory-environment` remaining tasks (hierarchical benchmark naming + docs)
- [ ] Complete `add-thermotaxis-system` remaining tasks (hierarchical benchmarks, biological validation, docs)
- [x] Update `add-ablation-toolkit` proposal with new brain names + Phase 2 placement
- [ ] Archive `add-multi-sensory-environment`
- [ ] Archive `add-thermotaxis-system`

______________________________________________________________________

## Backlog

- Hierarchical benchmark categories (from OpenSpec remaining tasks)
- Quantum thermotaxis (get thermotaxis working with QVarCircuit or other quantum brain)
- Run new benchmarks for all important scenarios with new brain names
- QSNN (quantum spiking neural network) brain architecture
- Pixel visualization theme

______________________________________________________________________

## Recently Completed

- Thermotaxis system core implementation
- Multi-sensory environment foundation (health, predator types, mechanosensation)
- PPO brain architecture
- Spiking brain rewrite (STDP → surrogate gradients)
- CMA-ES evolutionary optimization for quantum circuits
