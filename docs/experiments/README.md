# Experiment Logbooks

Human-written analysis and documentation of experiment series.

## Relationship to Other Systems

This project has complementary systems for tracking experiments:

| System | Location | Git Tracked | Purpose |
|--------|----------|-------------|---------|
| **Auto-tracking** | `experiments/*.json` | No | Raw metadata from every simulation run |
| **Evolution results** | `evolution_results/` | No | All evolution run outputs |
| **Artifacts** | `artifacts/` | Yes | Curated outputs referenced in logbooks |
| **Logbooks** (this) | `docs/experiments/logbooks/` | Yes | Human analysis, insights, narrative |
| **Benchmarks** | `benchmarks/` | Yes | Curated best results for leaderboards |

### Workflow

```markdown
1. Run simulations/evolution
   └── Auto-saved to experiments/*.json or evolution_results/

2. Query results
   └── python scripts/experiment_query.py list
   └── python scripts/experiment_query.py show <id>

3. Preserve notable results
   └── Copy to artifacts/ for git tracking
   └── e.g., cp -r evolution_results/20251209_205950 artifacts/evolutions/

4. Document findings
   └── Write logbook in docs/experiments/logbooks/
   └── Reference artifacts: "See artifacts/evolutions/20251209_205950/"

5. Promote best results
   └── python scripts/benchmark_submit.py submit <id>
   └── Saved to benchmarks/ for leaderboards
```

## Active Experiments

| # | Title | Status | Summary |
|---|-------|--------|---------|
| 001 | [Quantum Predator Optimization](logbooks/001-quantum-predator-optimization.md) | completed | Gradient-based learning approaches |
| 002 | [Evolutionary Parameter Search](logbooks/002-evolutionary-parameter-search.md) | completed | CMA-ES and GA optimization |
| 003 | [Spiking Brain Optimization](logbooks/003-spiking-brain-optimization.md) | completed | Surrogate gradients, LIF neurons, decay schedules |
| 004 | [PPO Brain Implementation](logbooks/004-ppo-brain-implementation.md) | completed | SOTA RL baseline, actor-critic, 98.5% foraging, 93% predators |
| 005 | [Health System Predator Scaling](logbooks/005-health-system-predator-scaling.md) | completed | Health system enables better evasion learning via survival + richer rewards |
| 006 | [Unified Sensory Modules](logbooks/006-unified-sensory-modules.md) | completed | Biologically-inspired 4-feature architecture matches legacy 2-feature performance |
| 007 | [PPO Thermotaxis Baselines](logbooks/007-ppo-thermotaxis-baselines.md) | completed | PPO baselines for 9 thermotaxis configs (3 sizes × 3 tasks), 84-98% post-conv |
| 008 | [Quantum Brain Evaluation](logbooks/008-quantum-brain-evaluation.md) | completed | 300+ sessions, 11+ quantum architectures (QA-1 to QA-7). QRH genuine quantum advantage on pursuit (+9.4pp). HybridQuantum 96.9% but classical ablation matches. QEF competitive, not advantageous. QA-7 plasticity: classical shows zero forgetting — hypothesis untestable at current complexity. Pivot to environment enrichment. |
| 009 | [Temporal Sensing Evaluation](logbooks/009-temporal-sensing-evaluation.md) | completed | Phase 3: Temporal/derivative sensing replacing oracle gradients. New GRU PPO brain (19th architecture). Temporal Mode A achieves L100=95% on hardest environment (matching oracle L100=95%; L500=94% vs oracle 97%, within 3pp). GRU outperforms LSTM. Chunk length is critical hyperparameter. |
| 010 | [Aerotaxis Baselines](logbooks/010-aerotaxis-baselines.md) | completed | Oracle + derivative + temporal evaluation across 7 scenarios + 3 controls. Derivative exceeds oracle on all 6 scenarios (+4pp to +28pp). Temporal: 99% L100 single-modality, 89% L100 triple-modality (12k episodes, all seeds converge). Training time scales with modality count. |
| 011 | [Multi-Agent Evaluation](logbooks/011-multi-agent-evaluation.md) | in-progress | 7 pre-klinotaxis campaigns (A-H) plus 7 Klinotaxis Era campaigns (K1-K7, 104 sessions). Key positive findings: food-marking pheromones +77pp on single-cluster (K1); +47.8% social feeding under scarcity (K6); phenotype frequency-dependent fitness +40-45pp (K3) — meets Phase 4 emergent-behaviour exit criterion. Key negative findings: alarm pheromones inert across 5 conditions including biologically faithful no-nociception baseline; aggregation pheromone informationally inert; single-cluster benefit collapses to +2.7pp on multi-cluster. Three bugs (#112, #115, STAM heterogeneous-dim) found and fixed. |

## How to Use Logbooks

### Reading

Each logbook follows a consistent structure:

- **Objective**: What we're trying to achieve
- **Hypothesis**: What we expected
- **Results**: What actually happened
- **Analysis**: Why it happened
- **Next Steps**: Where to go from here

### Creating New Logbooks

1. Copy `templates/experiment.md` to `logbooks/NNN-descriptive-name.md`
2. Use the next sequential number
3. Update the index table above
4. Reference session IDs from `experiments/*.json` for reproducibility

### Linking to Auto-Tracked Data

Reference specific experiments by session ID:

```markdown
- Session: `20251209_205950` (80% success with CMA-ES)
- Query: `python scripts/experiment_query.py show 20251209_205950`
```

## Key Findings Summary

### Experiment 001: Quantum Circuit Limitations

- 2-qubit circuits max ~31% success with gradient learning
- Learning actively degrades good initializations
- Combined gradient (chemotaxis) works; separated gradients fail

### Experiment 002: Evolutionary Approach

- CMA-ES achieved 80% success on foraging-only
- GA achieved 70% with more stable convergence
- Evolution bypasses gradient noise problem

## Directory Structure

```markdown
docs/experiments/
├── README.md                    # This file
├── templates/
│   └── experiment.md            # Template for new logbooks
└── logbooks/
    ├── 001-quantum-predator-optimization.md
    ├── 002-evolutionary-parameter-search.md
    └── ...

experiments/                     # Auto-generated (gitignored)
├── 20251207_123456.json
└── ...

benchmarks/                      # Curated results (git tracked)
├── foraging_small/classical/
└── ...
```
