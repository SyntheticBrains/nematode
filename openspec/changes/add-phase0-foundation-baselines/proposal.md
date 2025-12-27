# Change: Add Phase 0 Foundation and Baselines

## Why

Phase 0 of the Quantum Nematode roadmap requires establishing rigorous baselines, integrating biological validation data, and creating public benchmark infrastructure before proceeding to more complex research phases. The roadmap defines specific exit criteria that must be met:

1. **Missing SOTA RL Baseline**: Current classical baselines use REINFORCE (circa 2015). Modern algorithms like PPO are required to credibly benchmark quantum vs classical approaches. The roadmap requires "Classical SOTA baseline (PPO) achieves >85% success on foraging tasks."

2. **No Biological Validation**: Simulation agents are not validated against real C. elegans behavioral data. The roadmap requires "At least 1 real C. elegans behavioral dataset integrated for validation."

3. **Optimization Method Clarity**: The December 2025 breakthrough showed CMA-ES achieves 88% vs 22% for gradient methods on quantum circuits, but this finding is not documented for users. The roadmap requires "Clear documentation of which optimization methods work for which architectures."

4. **Public Benchmark Adoption**: The benchmark system exists but lacks comprehensive public documentation for external researchers to submit and reproduce results.

## What Changes

### 1. PPO Brain Implementation (SOTA Classical Baseline)

Add Proximal Policy Optimization (PPO) as a new brain architecture following the existing ClassicalBrain pattern:

- **New brain type**: `ppo` with actor-critic architecture
- **Clipped surrogate objective**: Prevents large policy updates (epsilon=0.2)
- **Generalized Advantage Estimation**: Reduces variance (lambda=0.95)
- **Rollout buffer**: Collects experience before batch updates
- **Integration**: Same protocol as MLPBrain, registered in brain factory

### 2. Chemotaxis Validation System

Add biological validation capability using real C. elegans behavioral data:

- **Chemotaxis index calculation**: CI = (N_attractant - N_control) / N_total
- **Literature dataset**: Published CI values from peer-reviewed papers (Bargmann & Horvitz 1991, Saeki et al. 2020)
- **Validation benchmark**: Compare agent behavior to biological ranges
- **Integration**: New `--validate-chemotaxis` flag in simulation scripts

### 3. NematodeBench Public Documentation

Create comprehensive public documentation for benchmark submissions:

- **Submission guide**: Step-by-step instructions for external researchers
- **Evaluation methodology**: Clear explanation of composite scoring
- **Reproducibility requirements**: Standards for valid submissions
- **Evaluation script**: Automated validation of submissions

### 4. Optimization Method Documentation

Document validated optimization approaches for each architecture:

- **Summary table**: Architecture to optimization method mapping
- **Detailed findings**: Why CMA-ES works better than gradients for quantum
- **Configuration examples**: Recommended settings for each approach
- **Selection guidance**: When to use which method

## Impact

**Affected Specs:**
- `brain-architecture`: ADDED - PPO brain requirements (6 requirements)
- `benchmark-management`: MODIFIED - NematodeBench public documentation requirements (2 requirements)

**New Specs:**
- `validation-system`: NEW capability for biological validation (4 requirements)

**Affected Code:**
- `packages/quantum-nematode/quantumnematode/brain/arch/ppo.py` (new, ~400 lines)
- `packages/quantum-nematode/quantumnematode/brain/arch/__init__.py` (add export)
- `packages/quantum-nematode/quantumnematode/validation/` (new module)
- `data/chemotaxis/` (new data directory)
- `docs/nematodebench/` (new documentation)
- `docs/OPTIMIZATION_METHODS.md` (new documentation)
- `scripts/evaluate_submission.py` (new script)
- `configs/examples/ppo_*.yml` (new configs)

**Breaking Changes:**
- None. All changes are additive.

**Benefits:**
- Credible classical baseline for quantum advantage claims
- Biological validation of agent behavior
- Clear guidance for architecture/optimization selection
- Lower barrier for external researchers to contribute

**Risks:**
- PPO may not reach 85% success target (mitigation: tune hyperparameters, accept if within 5% of MLPBrain)
- Literature CI interpretation (mitigation: use conservative ranges, document assumptions)

**Migration:**
- No migration required. Existing code unchanged.
