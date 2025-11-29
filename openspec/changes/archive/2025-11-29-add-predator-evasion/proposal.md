# Change: Add Predator Evasion Behavior

## Why

C. elegans nematodes in nature must balance foraging for food with avoiding predators to survive. Currently, the simulation only models food-seeking behavior, providing an incomplete picture of realistic nematode decision-making. Adding predator evasion enables research into multi-objective reinforcement learning, approach-avoidance conflict resolution, and survival-foraging trade-offs—core challenges in both biological systems and AI agent design.

This change introduces biologically realistic predator avoidance based on C. elegans research showing chemosensory detection, gradient-based repulsion, and escape responses.

## What Changes

- Add configurable predator entities to dynamic foraging environments
- Implement unified gradient system combining food attraction and predator repulsion
- Extend observation space to include predator threat gradients
- Add predator-specific metrics (encounters, evasions, deaths)
- Introduce proximity penalties for being near predators
- Add predator-enabled benchmark categories
- Support instant death termination on predator collision
- Restructure configuration to nest foraging settings under dedicated subsection
- Add themed visualization for predators (spider emoji 🕷️ for emoji mode, # for ASCII)
- Include detection radius visualization (configurable, default enabled)
- Display agent danger status during simulation runs

## Impact

- **Affected specs**: environment-simulation, configuration-system, benchmark-management, experiment-tracking
- **Affected code**:
  - `packages/quantum-nematode/quantumnematode/env.py` - Add Predator class and predator support to DynamicForagingEnvironment
  - `packages/quantum-nematode/quantumnematode/config_loader.py` - Extend configuration schema with predator settings and restructured foraging config
  - `packages/quantum-nematode/quantumnematode/agent/metrics.py` - Add predator-specific metrics
  - `packages/quantum-nematode/quantumnematode/agent/reward_calculator.py` - Add proximity penalty calculation
  - `packages/quantum-nematode/quantumnematode/report/plots.py` - Add predator-specific plots
  - `packages/quantum-nematode/quantumnematode/report/csv_export.py` - Include predator metrics in exports
  - `packages/quantum-nematode/quantumnematode/benchmark/categorization.py` - Add predator benchmark categories
  - `configs/examples/` - Add example configurations with predators enabled
- **Backward compatibility**: All predator features default to disabled; existing configurations work unchanged
- **Breaking changes**: None - all changes are additive with sensible defaults

## Future Enhancements

The proposal includes documentation notes for future capabilities:

1. **Health system**: Replace instant death with HP-based damage model allowing multiple encounters
2. **Additional predator types**:
   - Nematode-trapping fungi (stationary traps)
   - Predatory nematodes (active pursuit behavior)
   - Toxic bacteria patches (stationary danger zones)
3. **Advanced predator behaviors**:
   - Patrol patterns (fixed routes)
   - Pursuit behavior (active tracking of nematode)
   - Group hunting (coordinated multi-predator attacks)
