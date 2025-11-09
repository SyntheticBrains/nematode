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
- **Reproducibility info**: Exact commit hash, clean git state, configuration file
- **Contributor attribution**: Name and optional GitHub username

### Quality Standards

To ensure benchmark quality, submissions must meet these criteria:

- **Minimum runs**: At least 20 simulation runs
- **Reproducibility**: Clean git state (no uncommitted changes)
- **Complete metadata**: All required fields populated
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
- Run at least 20+ simulation runs for statistical significance
- Use a clean git state (commit your changes first)
- Include meaningful optimization notes
- Test with standard configurations when possible
- Document any novel approaches or techniques

**Don't**:
- Submit preliminary or exploratory results
- Use uncommitted code changes
- Submit duplicate benchmarks for the same approach
- Include sensitive information in notes

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

## Questions or Issues?

- Check the [Contributing Guide](CONTRIBUTING.md) for detailed guidelines
- Open an issue on GitHub for questions about the benchmark system
- Contact maintainers for verification or benchmark disputes
