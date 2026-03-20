## Why

After 290+ experiment sessions across 11+ quantum architectures (QA-1 through QA-5), no quantum architecture has demonstrated statistically significant performance advantage over classical equivalents on our current environments. The one remaining testable hypothesis at current environment complexity is that PQC unitarity prevents catastrophic forgetting during sequential multi-objective training (arXiv:2511.17228, Chen & Zhang 2025). This is low-effort (1-2 weeks), uses existing architectures, and could reframe quantum advantage as an optimisation landscape property rather than single-task speedup — a genuinely publishable result either way.

QA-7 is the final quantum experiment before pivoting to environment enrichment. If PQC anti-forgetting is confirmed, it provides a concrete quantum value proposition for the project going forward. If not, the quantum architecture campaign is conclusively complete at current complexity.

## What Changes

- **New evaluation protocol script** (`scripts/run_plasticity_test.py`): Implements sequential multi-objective training — train on Objective A (foraging) → B (pursuit predators) → C (thermotaxis) → return to A, measuring forgetting at each transition
- **New plasticity-specific configs** (`configs/studies/plasticity/`): Consistent 200-episode-per-objective configs for each test architecture (QRH, CRH, HybridQuantum, HybridClassical, MLP PPO)
- **Plasticity metrics**: Forward transfer (does prior learning help new task?), backward forgetting (how much does task A degrade after B and C?), plasticity retention (does learning rate on task C match task A?)
- **Statistical validation**: 4-8 seeds per architecture, t-test between quantum and classical forgetting rates
- **Results integration**: Metrics exported to CSV, summary added to logbook 008 and quantum-architectures.md

## Capabilities

### New Capabilities

- `plasticity-evaluation`: Sequential multi-objective training evaluation protocol. Manages objective switching (loading different environment configs mid-training while preserving brain weights), captures per-transition metrics (success rate, loss, entropy before/after each switch), computes forgetting/transfer metrics, and runs multi-seed statistical comparisons between quantum and classical architectures.

### Modified Capabilities

- `experiment-tracking`: Extended to support mid-session environment switching and per-objective metric segmentation within a single evaluation run.

## Impact

- **New script**: `scripts/run_plasticity_test.py` — orchestrates the sequential training protocol
- **New configs**: `configs/studies/plasticity/` — 5 architecture configs × consistent parameters
- **Existing brains**: No changes — QRH, CRH, HybridQuantum, HybridClassical, MLP PPO used as-is
- **Existing environment**: No changes — foraging, pursuit predator, and thermotaxis configs already exist
- **Experiment tracking**: Minor extension to support objective-switch checkpointing
- **Dependencies**: No new dependencies
- **Key threshold**: PQC must show ≤50% of classical network's backward forgetting to confirm the quantum plasticity hypothesis
