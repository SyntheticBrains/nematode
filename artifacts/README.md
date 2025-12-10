# Artifacts

Curated experiment outputs referenced in logbooks and documentation.

## Purpose

This directory stores **selectively preserved** experiment outputs that are:
- Referenced in experiment logbooks (`docs/experiments/logbooks/`)
- Important for reproducibility of documented findings
- Worth keeping for future reference or comparison

## Directory Structure

```
artifacts/
├── README.md           # This file
├── evolution/          # Evolution run results
│   └── <timestamp>/    # e.g., 20251209_205950
│       ├── best_params.json
│       ├── history.csv
│       └── checkpoint_gen30.pkl (optional)
├── experiments/        # Simulation experiment snapshots
│   └── <session_id>/   # e.g., 20251207_035803
│       └── metadata.json
└── models/             # Trained model weights (future)
    └── ...
```

## Relationship to Other Systems

| System | Location | Git Tracked | Purpose |
|--------|----------|-------------|---------|
| Auto-tracking | `experiments/` | No | All simulation run metadata |
| Evolution results | `evolution_results/` | No | All evolution run outputs |
| **Artifacts** | `artifacts/` | **Yes** | Curated outputs worth keeping |
| Benchmarks | `benchmarks/` | Yes | Top-performing submissions |
| Logbooks | `docs/experiments/logbooks/` | Yes | Human analysis narratives |

## Workflow

### Preserving Evolution Results

```bash
# 1. Run evolution (outputs to evolution_results/)
uv run python scripts/run_evolution.py --config configs/examples/evolution_foraging_only.yml ...

# 2. Copy notable results to artifacts/
cp -r evolution_results/20251209_205950 artifacts/evolution/

# 3. Reference in logbook
# "Best parameters from artifacts/evolution/20251209_205950/"
```

### Preserving Experiment Snapshots

```bash
# 1. Run simulation with tracking (outputs to experiments/)
uv run scripts/run_simulation.py --track-experiment ...

# 2. Copy notable experiments to artifacts/
cp experiments/20251207_035803.json artifacts/experiments/20251207_035803/metadata.json

# 3. Reference in logbook
# "See artifacts/experiments/20251207_035803/"
```

## What to Store Here

**Do store:**
- Results explicitly referenced in logbooks
- Parameters that achieved notable performance
- Checkpoints needed to reproduce documented experiments
- Model weights for significant milestones (future)

**Don't store:**
- Every experiment run (use `experiments/` for that)
- Large checkpoint files unless specifically needed
- Duplicate data already in `benchmarks/`

## Naming Conventions

- Use original timestamps/session IDs for traceability
- Keep directory names matching source system IDs
- Add descriptive suffixes if needed: `20251209_205950_cmaes_80pct/`

## File Size Considerations

- JSON/CSV files: Always acceptable
- Checkpoint files (`.pkl`): Include only if needed for reproduction
- Large model files: Consider Git LFS for files > 10MB
