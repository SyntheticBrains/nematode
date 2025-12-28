# Quantum Nematode Benchmarks

Official benchmarks for the Quantum Nematode project, powered by [NematodeBench](docs/nematodebench/).

## Quick Links

| Resource | Description |
|----------|-------------|
| [**Leaderboard**](docs/nematodebench/LEADERBOARD.md) | Current benchmark rankings |
| [**Submission Guide**](docs/nematodebench/SUBMISSION_GUIDE.md) | How to submit your results |
| [**Evaluation Methodology**](docs/nematodebench/EVALUATION.md) | Scoring and metrics |
| [**Reproducibility**](docs/nematodebench/REPRODUCIBILITY.md) | Requirements for valid submissions |
| [**Optimization Methods**](docs/OPTIMIZATION_METHODS.md) | Which optimization works for each architecture |

## Overview

NematodeBench is a standardized framework for evaluating neural architectures in biologically-inspired navigation tasks. Benchmarks are organized by environment type (static maze, dynamic foraging, predator evasion) and brain architecture class (quantum, classical).

### Requirements

| Requirement | Minimum | Description |
|-------------|---------|-------------|
| Sessions | 10+ | Independent training sessions |
| Runs per session | 50+ | Episodes per session |
| Seed uniqueness | All | No duplicate seeds across all runs |
| Config consistency | Required | Same brain type and environment |

### Composite Score

Benchmarks are ranked by a weighted composite score:

- **Success Rate** (40%): Post-convergence success percentage
- **Distance Efficiency** (30%): Navigation efficiency
- **Learning Speed** (20%): How quickly the strategy converges
- **Stability** (10%): Consistency after convergence

## Quick Start

```bash
# Run 10+ independent training sessions
for session in {1..10}; do
    uv run scripts/run_simulation.py \
        --config configs/your_config.yml \
        --track-experiment \
        --runs 50
done

# Submit all sessions together
uv run scripts/benchmark_submit.py \
    --experiments experiments/* \
    --category foraging_small/classical \
    --contributor "Your Name"

# Regenerate leaderboards
uv run scripts/benchmark_submit.py regenerate
```

See the [Submission Guide](docs/nematodebench/SUBMISSION_GUIDE.md) for detailed instructions.

## Categories

Benchmarks are organized into 14 categories:

| Environment | Quantum | Classical |
|-------------|---------|-----------|
| Static Maze | `static_maze/quantum` | `static_maze/classical` |
| Foraging Small (≤20x20) | `foraging_small/quantum` | `foraging_small/classical` |
| Foraging Medium (≤50x50) | `foraging_medium/quantum` | `foraging_medium/classical` |
| Foraging Large (>50x50) | `foraging_large/quantum` | `foraging_large/classical` |
| Predator Small (≤20x20) | `predator_small/quantum` | `predator_small/classical` |
| Predator Medium (≤50x50) | `predator_medium/quantum` | `predator_medium/classical` |
| Predator Large (>50x50) | `predator_large/quantum` | `predator_large/classical` |

## External Submissions

We welcome benchmark submissions from external researchers.

1. Fork this repository
2. Follow the [Submission Guide](docs/nematodebench/SUBMISSION_GUIDE.md)
3. Create a pull request with your benchmark JSON

## Questions?

- Check the [Contributing Guide](CONTRIBUTING.md)
- Open an issue on GitHub with the `benchmark` label
