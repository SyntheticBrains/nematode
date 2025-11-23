# Quantum Nematode Benchmarks

This document contains the complete benchmark leaderboards for the Quantum Nematode project. Benchmarks are organized by environment type and brain architecture class.

## Table of Contents

- [About Benchmarks](#about-benchmarks)
- [Benchmark Categories](#benchmark-categories)
- [How to Submit](#how-to-submit)
- [Leaderboards](#leaderboards)
  - [Static Maze](#static-maze)
  - [Dynamic Small (≤20x20)](#dynamic-small-20x20)
  - [Dynamic Medium (≤50x50)](#dynamic-medium-50x50)
  - [Dynamic Large (>50x50)](#dynamic-large-50x50)

## About Benchmarks

Benchmarks represent verified, high-quality simulation results that demonstrate the performance of different brain architectures and optimization strategies. Each benchmark includes:

- **Complete metadata**: Configuration, git commit, system info, dependency versions
- **Performance metrics**: Success rate, average steps, food collected, distance efficiency
- **Convergence analysis**: Learning speed, stability, and post-convergence performance
- **Composite score**: Weighted benchmark score combining multiple performance aspects
- **Reproducibility info**: Exact commit hash, clean git state, configuration file
- **Contributor attribution**: Name and optional GitHub username

### Convergence-Based Evaluation

**Key Metrics**:

1. **Composite Score** (0.0 to 1.0): Weighted combination of:
   - **Success Rate** (40%): Post-convergence success percentage
   - **Distance Efficiency** (30%): Navigation efficiency (optimal path / actual path)
   - **Learning Speed** (20%): How quickly the strategy converges
   - **Stability** (10%): Consistency of performance after convergence

2. **Convergence Detection**: Adaptive algorithm that identifies when success rate variance drops below 5% for 10 consecutive runs, indicating the strategy has stabilized.

3. **Fallback Strategy**: If convergence is not detected within the session, metrics are calculated from the last 10 runs with a warning indicator (⚠).

**Why Convergence-Based?**

- **Accurate Assessment**: Evaluates the strategy's final learned behavior, not the learning process
- **Learning Speed Matters**: Rewards strategies that converge quickly
- **Stability Bonus**: Penalizes strategies with inconsistent performance
- **Fair Comparison**: Separates exploration noise from converged performance

### Quality Standards

To ensure benchmark quality, submissions must meet these criteria:

- **Minimum runs**: At least 50 simulation runs
- **Reproducibility**: Clean git state (no uncommitted changes)
- **Complete metadata**: All required fields populated
- **Convergence**: Strategies that converge are preferred (indicated by ✓)
- **Verification**: Manual review by maintainers before merging

## Benchmark Categories

Benchmarks are organized into categories based on:

1. **Environment Type**:
   - **Static Maze**: Traditional maze navigation
   - **Dynamic Small**: Foraging environment with grid size ≤20x20
   - **Dynamic Medium**: Foraging environment with grid size ≤50x50
   - **Dynamic Large**: Foraging environment with grid size >50x50

2. **Brain Architecture Class**:
   - **Quantum**: Modular quantum brain architectures
   - **Classical**: MLP and other classical architectures

This results in 8 total categories:
- `static_maze_quantum`, `static_maze_classical`
- `dynamic_small_quantum`, `dynamic_small_classical`
- `dynamic_medium_quantum`, `dynamic_medium_classical`
- `dynamic_large_quantum`, `dynamic_large_classical`

## How to Submit

### Step 1: Run Your Experiment

Run a simulation with experiment tracking enabled:

```bash
uv run scripts/run_simulation.py \
  --config configs/your_config.yml \
  --track-experiment
```

This saves experiment metadata to `experiments/` for your review.

### Step 2: Review Your Results

Check your experiment results:

```bash
# List recent experiments
uv run scripts/experiment_query.py list --limit 5

# View detailed results
uv run scripts/experiment_query.py show <experiment-id>
```

### Step 3: Submit as Benchmark

When you're satisfied with the results, submit as a benchmark:

```bash
uv run scripts/benchmark_submit.py submit <experiment-id> \
  --contributor "Your Name" \
  --github "your-username" \
  --notes "Brief description of your optimization approach"
```

Or submit directly from a simulation run:

```bash
uv run scripts/run_simulation.py \
  --config configs/your_config.yml \
  --save-benchmark \
  --benchmark-notes "Your optimization approach"
```

This creates a benchmark file in `benchmarks/<category>/<experiment-id>.json`.

### Step 4: Create a Pull Request

1. Review the generated benchmark file
2. Add the file to git: `git add benchmarks/<category>/<experiment-id>.json`
3. Commit: `git commit -m "Add benchmark: <brief description>"`
4. Push your branch and create a pull request
5. Maintainers will verify and merge

### Submission Guidelines

**Do**:
- Run at least 50+ simulation runs for convergence detection and statistical significance
- Use a clean git state (commit your changes first)
- Include meaningful optimization notes
- Test with standard configurations when possible
- Document any novel approaches or techniques
- Aim for convergence (✓) - strategies that stabilize are ranked higher

**Don't**:
- Submit preliminary or exploratory results
- Use uncommitted code changes
- Submit duplicate benchmarks for the same approach
- Include sensitive information in notes
- Submit sessions with fewer than 50 runs (required for convergence analysis)

## Leaderboards

### Static Maze

#### Quantum Architectures

| Brain | Success Rate | Avg Steps | Contributor | Date |
|---|---|---|---|---|
| qmodular | 100% | 42 | @chrisjz | 2025-11-09 |
| modular | 100% | 35 | @chrisjz | 2025-11-09 |

#### Classical Architectures

| Brain | Success Rate | Avg Steps | Contributor | Date |
|---|---|---|---|---|
| qmlp | 100% | 49 | @chrisjz | 2025-11-09 |
| mlp | 100% | 66 | @chrisjz | 2025-11-09 |
| spiking | 75% | 189 | @chrisjz | 2025-11-09 |
| spiking | 65% | 173 | @chrisjz | 2025-11-09 |

### Dynamic Small (≤20x20)

#### Quantum Architectures

| Brain | Success Rate | Avg Steps | Foods/Run | Dist Eff | Contributor | Date |
|---|---|---|---|---|---|---|
| modular | 30% | 430 | 7.8 | 0.27 | @chrisjz | 2025-11-09 |

#### Classical Architectures

| Brain | Success Rate | Avg Steps | Foods/Run | Dist Eff | Contributor | Date |
|---|---|---|---|---|---|---|
| mlp | 60% | 351 | 8.0 | 0.25 | @chrisjz | 2025-11-09 |

### Dynamic Medium (≤50x50)

#### Quantum Architectures

_No benchmarks submitted yet._

#### Classical Architectures

| Brain | Success Rate | Avg Steps | Foods/Run | Dist Eff | Contributor | Date |
|---|---|---|---|---|---|---|
| mlp | 80% | 804 | 27.2 | 0.52 | @chrisjz | 2025-11-09 |

### Dynamic Large (>50x50)

#### Quantum Architectures

_No benchmarks submitted yet._

#### Classical Architectures

_No benchmarks submitted yet._

---

## Updating Leaderboards

Leaderboards are automatically generated from benchmark submissions. To regenerate:

```bash
uv run scripts/benchmark_submit.py regenerate
```

This generates both the README summary section and the full leaderboard tables for this document.

## Technical Details: Convergence Algorithm

For developers and researchers interested in the convergence detection algorithm:

### Detection Algorithm

The convergence detection algorithm uses an adaptive variance-based approach:

1. **Minimum Runs**: Requires at least 50 runs before convergence can be declared
2. **Variance Threshold**: Success rate variance must drop below 0.05 (5%)
3. **Stability Window**: Low variance must be sustained for 10 consecutive runs
4. **Detection Point**: The first run where a stable 10-run window begins

**Algorithm**:
```python
for each run i from min_runs to (total_runs - stability_window):
    window = runs[i : i + stability_window]
    if variance(window.success_rate) < 0.05:
        return i  # Convergence detected at run i
return None  # Never converged
```

### Composite Score Calculation

The composite score combines four normalized components:

```python
composite_score = (
    0.40 * success_rate +           # Post-convergence success %
    0.30 * distance_efficiency +    # Optimal path / actual path
    0.20 * learning_speed +         # Inverse of runs_to_convergence
    0.10 * stability                # Inverse of variance
)
```

**Normalization**:
- Success rate: Already in [0, 1]
- Distance efficiency: Already in [0, 1] where 1.0 = perfect optimal navigation
- Learning speed: `1.0 - (runs_to_convergence / total_runs)`
- Stability: `1.0 - (variance / 0.2)` where 0.2 is max expected variance

**Static Environments**: For environments without distance tracking, efficiency component uses success rate as a proxy.

### Post-Convergence Metrics

After detecting convergence at run N:

- **Success Rate**: Percentage of successful runs from N onwards
- **Avg Steps**: Mean steps in successful runs from N onwards
- **Avg Foods**: Mean foods collected in runs from N onwards
- **Variance**: Variance of success indicators from N onwards
- **Distance Efficiency**: Average distance efficiency in successful runs from N onwards

**Fallback Strategy**: If convergence is not detected, metrics are calculated from the last 10 runs with `converged=False` flag.

### Implementation

Core convergence analysis is implemented in:
- [`quantumnematode/benchmark/convergence.py`](packages/quantum-nematode/quantumnematode/benchmark/convergence.py)

Key functions:
- `detect_convergence()`: Convergence detection algorithm
- `calculate_post_convergence_metrics()`: Post-convergence performance calculation
- `calculate_composite_score()`: Weighted composite score calculation
- `analyze_convergence()`: Main entry point for full analysis

## Questions or Issues?

- Check the [Contributing Guide](CONTRIBUTING.md) for detailed guidelines
- Open an issue on GitHub for questions about the benchmark system
- Contact maintainers for verification or benchmark disputes
