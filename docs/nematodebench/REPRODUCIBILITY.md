# Reproducibility Requirements

This document outlines the requirements for reproducible NematodeBench submissions.

## Core Requirements

### 1. Automatic Seed Management

The system automatically generates and tracks seeds:

```bash
# Run with automatic seed generation
uv run scripts/run_simulation.py \
    --config configs/examples/ppo_foraging_small.yml \
    --track-experiment \
    --runs 50
```

Or specify an explicit seed:

```bash
# Run with explicit seed (for reproduction)
uv run scripts/run_simulation.py \
    --config configs/examples/ppo_foraging_small.yml \
    --track-experiment \
    --seed 12345 \
    --runs 50
```

### 2. Per-Run Seed Tracking

Every run has its own seed, automatically recorded in the experiment JSON:

```python
# Seeds are derived from base seed for each run
# Run 0: seed, Run 1: seed + 1, Run 2: seed + 2, etc.
```

The experiment JSON includes `per_run_results` array with seed and metrics for each run.

### 3. Seeding Infrastructure

The system uses a centralized seeding module (`quantumnematode/utils/seeding.py`):

```python
from quantumnematode.utils.seeding import generate_seed, set_global_seed, get_rng

# Auto-generate cryptographically random seed
seed = generate_seed()  # Uses secrets.randbelow(2**32)

# Set global numpy/torch seeds
set_global_seed(seed)

# Get seeded RNG for local operations
rng = get_rng(seed)
```

### 4. Version Control

All submissions must include:

- **Git commit hash** - Exact code version used for experiments
- **Configuration file** - Automatically copied to experiment folder
- **Config preservation** - Original config stored with each experiment

### 5. Environment Documentation

Record the execution environment:

```json
{
  "environment": {
    "python_version": "3.11.0",
    "torch_version": "2.1.0",
    "numpy_version": "1.24.0",
    "platform": "Linux-5.15.0-x86_64",
    "cuda_version": "11.8",
    "gpu_model": "NVIDIA RTX 4090"
  }
}
```

### 6. Determinism

For fully reproducible results:

```python
# PyTorch determinism
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Note: Some operations may be slower with determinism enabled
```

## Submission Checklist

- [ ] **10+ independent sessions** completed
- [ ] **50+ runs per session** for statistical significance
- [ ] **ALL seeds unique** across all runs in all sessions
- [ ] **Consistent configuration** across sessions (brain type, environment)
- [ ] **Git state clean** - no uncommitted changes
- [ ] **evaluate_submission.py passes** validation
- [ ] **Config files copied** to experiment folders automatically

## Verification Process

### Automated Checks

The `evaluate_submission.py` script verifies:

1. JSON structure validity
2. Minimum run count (≥50)
3. Minimum session count (≥10)
4. Config file exists at specified path
5. Git commit hash recorded
6. Required metrics present
7. No NaN/Inf values
8. All seeds unique

### Manual Review

Maintainers may additionally:

1. Spot-check random runs for reproducibility
2. Verify configuration matches results
3. Check for anomalous statistics

### Optional Reproduction

Submitters can request full reproduction verification:

```bash
uv run scripts/evaluate_submission.py \
    --submission my_submission.json \
    --reproduce \
    --reproduce-runs 5
```

This will:

1. Re-run 5 experiments with recorded seeds
2. Compare results to submission
3. Flag significant discrepancies

## Known Sources of Non-Determinism

### Expected Variance

Some variance is expected even with seeds:

| Source | Typical Variance | Notes |
|--------|-----------------|-------|
| Environment initialization | ~1% | Food/predator positions |
| Network initialization | ~2% | Weight sampling |
| Training dynamics | ~5% | Exploration noise |

### Uncontrollable Factors

These may cause larger variance:

- **GPU non-determinism** - Atomic operations, parallel reduction
- **Library updates** - Different NumPy/PyTorch versions
- **Hardware differences** - CPU vs GPU vs QPU, different GPU models

### Acceptable Variance

Submissions are considered reproducible if:

- Re-run results fall within reported confidence intervals
- No systematic bias (all re-runs consistently higher/lower)
- Variance matches reported standard deviation

## Best Practices

### 1. Lock Dependencies

Use exact versions:

```bash
# Create reproducible environment
uv lock
uv sync --frozen
```

### 2. Log Everything

```python
# Log configuration and seeds
logger.info(f"Config: {config_path}")
logger.info(f"Config hash: {config_hash}")
logger.info(f"Random seed: {seed}")
logger.info(f"Git commit: {git_hash}")
```

### 3. Save Checkpoints

```python
# Save model at key points
torch.save({
    'model_state': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'episode': episode,
    'seed': seed,
}, f'checkpoints/run_{run_id}_ep_{episode}.pt')
```

### 4. Use Tracking

Enable experiment tracking:

```bash
uv run scripts/run_simulation.py \
    --config my_config.yml \
    --track-experiment \
    --runs 50
```

## Handling Non-Reproducibility

If results can't be exactly reproduced:

1. **Document known variance sources** in submission
2. **Provide error bars** with sufficient runs (100+)
3. **Note hardware differences** if using specialized equipment
4. **Consider simpler configuration** that's more deterministic

## Questions?

For reproducibility issues, open a GitHub issue with:

- Submission ID
- Specific run that failed to reproduce
- Error logs or discrepancy details
- Environment comparison
