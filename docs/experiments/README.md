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
| 008 | [Quantum Brain Evaluation](logbooks/008-quantum-brain-evaluation.md) | in_progress | QRC failed (0% success), evaluating QSNN and other quantum architectures |

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
