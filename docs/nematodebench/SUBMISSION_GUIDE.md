# NematodeBench Submission Guide

This guide walks through the process of submitting benchmark results to NematodeBench.

## Requirements

NematodeBench requires multiple independent training sessions to ensure scientific reproducibility:

| Requirement | Minimum | Description |
|-------------|---------|-------------|
| Sessions | 10+ | Independent training sessions |
| Runs per session | 50+ | Episodes per session |
| Seed uniqueness | All | No duplicate seeds across all runs in all sessions |
| Config consistency | Required | Same brain type and environment across sessions |

## Prerequisites

Before submitting a NematodeBench benchmark, ensure you have:

1. **Minimum 10 independent sessions** - Each with 50+ runs for statistical significance
2. **Unique seeds across all runs** - No duplicate seeds across any sessions
3. **Consistent configuration** - Same brain type and environment across sessions
4. **Clean git state** - All code changes committed to your fork

See [Optimization Methods](../OPTIMIZATION_METHODS.md) for guidance on which optimization strategy works best for each brain architecture.

## Step 1: Set Up Your Environment

```bash
# Clone and install
git clone https://github.com/YOUR_FORK/quantum-nematode.git
cd quantum-nematode
uv sync

# Verify installation
uv run python -c "from quantumnematode import brain; print('OK')"
```

## Step 2: Prepare Your Configuration

Create or modify a configuration file in `configs/examples/`:

```yaml
# configs/examples/my_brain_foraging_medium.yml
max_steps: 1000
brain:
  name: ppo  # or mlp, modular, spiking, etc.
  config:
    hidden_dim: 128
    # ... your hyperparameters

environment:
  type: dynamic
  dynamic:
    grid_size: 50
    foraging:
      foods_on_grid: 20
      target_foods_to_collect: 30
```

## Step 3: Run Multiple Training Sessions

Run **at least 10 independent training sessions**:

```bash
# Run 10+ independent sessions (each will generate unique seeds automatically)
for session in {1..10}; do
    uv run scripts/run_simulation.py \
        --config configs/examples/my_brain_foraging_medium.yml \
        --track-experiment \
        --runs 50
done
```

Each session will be saved to `experiments/<timestamp>/` with:

- The experiment JSON with per-run seeds
- A copy of the configuration file

## Step 4: Submit to NematodeBench

After running 10+ sessions, submit them as a NematodeBench benchmark:

```bash
uv run scripts/benchmark_submit.py \
    --experiments experiments/20251228_A experiments/20251228_B experiments/20251228_C ... \
    --category foraging_small/classical \
    --contributor "Your Name" \
    --github "your-username" \
    --notes "Brief description of your approach"
```

This will:

1. Validate all experiments meet requirements
2. Check seed uniqueness across all runs in all sessions
3. Copy experiment JSONs to `artifacts/benchmarks/<submission_id>/`
4. Copy a single config.yml from the first session
5. Aggregate metrics using StatValue (mean/std/min/max across sessions)
6. Save the benchmark metadata to `benchmarks/<category>/<submission_id>.json`

## Step 5: Regenerate Leaderboards

After submitting, regenerate the leaderboard documentation:

```bash
uv run scripts/benchmark_submit.py regenerate
```

This automatically updates:

- `docs/nematodebench/LEADERBOARD.md` - Full leaderboard tables
- `README.md` - Current Leaders section

## Step 6: Validate Your Submission

Validate before creating a pull request:

```bash
uv run scripts/evaluate_submission.py \
    --submission benchmarks/foraging_small/classical/20251228_123456.json
```

Expected output:

```text
✓ Submission is VALID.

Summary:
  brain_type: ppo
  category: foraging_small/classical
  total_sessions: 10
  total_runs: 500
  success_rate: 92.3% ± 4.1%
  composite_score: 0.870 ± 0.030
  contributor: Your Name
  all_seeds_unique: True
```

## Step 7: Submit Pull Request

1. Fork the main repository (if not already done)

2. Stage the benchmark, leaderboard, and artifact files:

   ```bash
   git add benchmarks/<category>/<submission_id>.json
   git add artifacts/benchmarks/<submission_id>/
   git add README.md docs/nematodebench/LEADERBOARD.md
   ```

3. Create a pull request with:

   - Title: `[NematodeBench] <Brain> on <Task> - <Score>`
   - Description including your approach and findings

### PR Template

```markdown
## NematodeBench Submission

**Brain Architecture:** PPO
**Category:** foraging_small/classical
**Composite Score:** 0.870 ± 0.030
**Success Rate:** 92.3% ± 4.1%

### Configuration
- Hidden dim: 128
- Learning rate: 0.001
- Optimization: PPO with clipped surrogate

### Approach
Brief description of your approach, hyperparameter choices, and any novel techniques.

### Reproducibility
- [ ] 10+ independent sessions completed
- [ ] 50+ runs per session
- [ ] All seeds unique across all runs
- [ ] evaluate_submission.py passes
- [ ] Artifacts in artifacts/benchmarks/<submission_id>/
- [ ] Leaderboards regenerated

### Files Changed
- `benchmarks/<category>/<submission_id>.json`
- `artifacts/benchmarks/<submission_id>/` (all session JSONs + config.yml)
- `README.md` (Current Leaders section updated)
- `docs/nematodebench/LEADERBOARD.md` (Full leaderboard updated)
```

## Submission JSON Format

```json
{
  "submission_id": "20251228_123456",
  "timestamp": "2025-12-28T12:34:56Z",
  "brain_type": "ppo",
  "brain_config": {
    "type": "ppo",
    "hidden_dim": 128,
    "learning_rate": 0.0003
  },
  "environment": {
    "type": "dynamic",
    "grid_size": 20
  },
  "category": "foraging_small/classical",
  "sessions": [
    {
      "experiment_id": "20251228_A",
      "file_path": "artifacts/benchmarks/20251228_123456/20251228_A.json",
      "session_seed": 12345,
      "num_runs": 50
    }
  ],
  "total_sessions": 10,
  "total_runs": 500,
  "metrics": {
    "success_rate": {"mean": 0.923, "std": 0.041, "min": 0.82, "max": 0.98},
    "composite_score": {"mean": 0.87, "std": 0.03, "min": 0.80, "max": 0.92},
    "learning_speed": {"mean": 0.75, "std": 0.10, "min": 0.60, "max": 0.85},
    "stability": {"mean": 0.92, "std": 0.04, "min": 0.85, "max": 0.98}
  },
  "all_seeds_unique": true,
  "contributor": "Your Name",
  "github_username": "your-username",
  "notes": "Brief description of approach"
}
```

## Troubleshooting

### "Insufficient sessions"

NematodeBench requires minimum 10 independent sessions. Run more training sessions.

### "Insufficient runs"

Each session requires minimum 50 runs. Increase the `--runs` parameter.

### "Duplicate seeds found"

All seeds must be unique across ALL runs in ALL sessions. This typically indicates:

- Reusing the same experiment in multiple submissions
- Setting explicit seeds that overlap across sessions
- Let the system auto-generate seeds for each run

### "Config consistency error"

All sessions must use the same brain type, environment type, and grid size.
Minor differences (like seeds) are allowed.

### "Session reference not found"

The experiment JSON files must exist. Check that:

- Artifacts haven't been moved or deleted
- Paths point to `artifacts/benchmarks/<submission_id>/<experiment_id>.json`

### "Invalid JSON structure"

Check your submission against the schema above. Common issues:

- Missing required fields
- Incorrect data types
- NaN or Inf values

## Questions?

Open an issue on GitHub with the `benchmark` label.
