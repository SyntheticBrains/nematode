# Artifacts

Curated experiment outputs referenced in logbooks and documentation.

> Stored in Git LFS and **not fetched by default**: the committed `.lfsconfig` limits a fresh clone to `data/**`, so everything under `artifacts/` arrives as LFS pointers. Materialise what you need with `git lfs pull --include='artifacts/**'` (or a narrower path such as `artifacts/logbooks/029-*/**`). CI fetches every object.

## Purpose

This directory stores **selectively preserved** experiment outputs that are:

- Referenced in experiment logbooks (`docs/experiments/logbooks/`)
- Important for reproducibility of documented findings
- Worth keeping for future reference or comparison

## Directory Structure

```markdown
artifacts/
├── README.md           # This file
├── evolutions/         # Evolution run results
│   └── <timestamp>/    # e.g., 20251209_205950
│       ├── best_params.json
│       ├── history.csv
│       └── checkpoint.pkl (periodic, for --resume)
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
| Logbooks | `docs/experiments/logbooks/` | Yes | Human analysis narratives |

## Workflow

### Preserving Evolution Results

```bash
# 1. Run evolution (outputs to evolution_results/)
uv run python scripts/run_evolution.py --config configs/evolution/feedforwardga_foraging_small.yml --algorithm ga ...

# 2. Copy notable results to artifacts/
cp -r evolution_results/20251209_205950 artifacts/evolutions/

# 3. Reference in logbook
# "Best parameters from artifacts/evolutions/20251209_205950/"
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

## Naming Conventions

- Use original timestamps/session IDs for traceability
- Keep directory names matching source system IDs
- Add descriptive suffixes if needed: `20251209_205950_cmaes_80pct/`

## File Size Considerations

- JSON/CSV files: Always acceptable
- Checkpoint files (`.pkl`) and model weights (`.pt`): tracked via Git LFS automatically
- Training logs (`.log`) and all JSON files in `artifacts/`: tracked via Git LFS automatically
- For other binary formats > 500 KB: add a matching rule to `.gitattributes` before committing
