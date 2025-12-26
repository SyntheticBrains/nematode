# Quantum Nematode Benchmarks

This document contains the complete benchmark leaderboards for the Quantum Nematode project. Benchmarks are organized by environment type and brain architecture class.

## Table of Contents

- [About Benchmarks](#about-benchmarks)
- [Benchmark Categories](#benchmark-categories)
- [How to Submit](#how-to-submit)
- [Leaderboards](#leaderboards)

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

This results in 14 total categories:
- `static_maze_quantum`, `static_maze_classical`
- `dynamic_small_quantum`, `dynamic_small_classical`
- `dynamic_medium_quantum`, `dynamic_medium_classical`
- `dynamic_large_quantum`, `dynamic_large_classical`
- `dynamic_predator_small_quantum`, `dynamic_predator_small_classical`
- `dynamic_predator_medium_quantum`, `dynamic_predator_medium_classical`
- `dynamic_predator_large_quantum`, `dynamic_predator_large_classical`

### Predator Environment Characteristics

Predator-enabled environments exhibit significantly different convergence profiles compared to foraging-only environments:

- **Slower Convergence**: Agents typically require 3-4x more runs to converge (e.g., ~70 runs vs ~20 runs) due to the added complexity of learning both foraging and threat avoidance behaviors simultaneously.

- **Higher Variance**: Post-convergence stability variance is inherently higher (0.10-0.15 vs 0.00-0.05) because optimal strategies must balance exploration with risk avoidance, leading to more variable outcomes based on predator positioning.

- **Lower Composite Scores**: Predator tasks typically yield lower benchmark scores (0.55-0.65 vs 0.75-0.80) due to the multi-objective nature of the task and increased episode termination from predator collisions.

**These characteristics are expected and reflect the increased challenge of multi-objective learning.** A benchmark with Converge@Run of 70 and Stability of 0.12 in a predator environment represents successful learning, not poor performance.

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

| Brain | Score | Success% | Steps | Converge@Run | Stability | Contributor | Date |
|---|---|---|---|---|---|---|---|
| ✓ modular | 0.980 | 100% | 34 | 20 | 0.000 | @chrisjz | 2025-11-29 |
| ✓ modular | 0.960 | 100% | 32 | 20 | 0.000 | @chrisjz | 2025-11-23 |

#### Classical Architectures

| Brain | Score | Success% | Steps | Converge@Run | Stability | Contributor | Date |
|---|---|---|---|---|---|---|---|
| ✓ mlp | 0.960 | 100% | 24 | 20 | 0.000 | @chrisjz | 2025-11-23 |
| ✓ spiking | 0.932 | 100% | 67 | 34 | 0.000 | @chrisjz | 2025-12-21 |
| ✓ spiking | 0.896 | 100% | 79 | 52 | 0.000 | @chrisjz | 2025-12-19 |

### Dynamic Small (≤20x20)

#### Quantum Architectures

| Brain | Score | Success% | Steps | Converge@Run | Stability | Contributor | Date |
|---|---|---|---|---|---|---|---|
| ✓ modular | 0.762 | 100% | 207 | 20 | 0.000 | @chrisjz | 2025-12-14 |
| ✓ modular | 0.633 | 84% | 350 | 27 | 0.136 | @chrisjz | 2025-11-30 |
| ✓ modular | 0.600 | 81% | 380 | 38 | 0.151 | @chrisjz | 2025-11-30 |
| ✓ modular | 0.598 | 80% | 317 | 43 | 0.162 | @chrisjz | 2025-11-28 |
| ✓ modular | 0.503 | 64% | 370 | 43 | 0.231 | @chrisjz | 2025-11-27 |

#### Classical Architectures

| Brain | Score | Success% | Steps | Converge@Run | Stability | Contributor | Date |
|---|---|---|---|---|---|---|---|
| ✓ ppo | 0.832 | 100% | 178 | 20 | 0.000 | @chrisjz | 2025-12-26 |
| ✓ mlp | 0.822 | 100% | 181 | 20 | 0.000 | @chrisjz | 2025-11-27 |
| ✓ mlp | 0.776 | 100% | 240 | 20 | 0.000 | @chrisjz | 2025-11-23 |
| ✓ spiking | 0.733 | 100% | 267 | 22 | 0.000 | @chrisjz | 2025-12-21 |

### Dynamic Medium (≤50x50)

#### Quantum Architectures

_No benchmarks submitted yet._

#### Classical Architectures

_No benchmarks submitted yet._

### Dynamic Large (>50x50)

#### Quantum Architectures

_No benchmarks submitted yet._

#### Classical Architectures

_No benchmarks submitted yet._

### Dynamic Predator Small (≤20x20)

#### Quantum Architectures

| Brain | Score | Success% | Steps | Converge@Run | Stability | Contributor | Date |
|---|---|---|---|---|---|---|---|
| ✓ modular | 0.675 | 95% | 224 | 29 | 0.045 | @chrisjz | 2025-12-13 |
| ✓ modular | 0.402 | 32% | 320 | 24 | 0.217 | @chrisjz | 2025-11-29 |
| ✓ modular | 0.395 | 31% | 344 | 27 | 0.215 | @chrisjz | 2025-11-27 |

#### Classical Architectures

| Brain | Score | Success% | Steps | Converge@Run | Stability | Contributor | Date |
|---|---|---|---|---|---|---|---|
| ✓ mlp | 0.740 | 92% | 199 | 30 | 0.076 | @chrisjz | 2025-11-27 |
| ✓ mlp | 0.618 | 82% | 195 | 78 | 0.148 | @chrisjz | 2025-11-23 |
| ✓ mlp | 0.587 | 87% | 192 | 70 | 0.116 | @chrisjz | 2025-11-23 |
| ✓ spiking | 0.556 | 63% | 247 | 20 | 0.234 | @chrisjz | 2025-12-22 |
| ✓ spiking | 0.390 | 32% | 326 | 20 | 0.219 | @chrisjz | 2025-12-21 |

### Dynamic Predator Medium (≤50x50)

#### Quantum Architectures

_No benchmarks submitted yet._

#### Classical Architectures

_No benchmarks submitted yet._

### Dynamic Predator Large (>50x50)

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

> **Note**: While the algorithm can begin checking for convergence after 20 runs (plus the 10-run stability window), benchmark submissions require a minimum of 50 total runs to ensure adequate data collection across all learning phases.

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
