# NematodeBench

NematodeBench is the project's internal benchmark system. It provides a standardized framework for evaluating neural architectures in biologically-inspired navigation tasks, and acts as the reproducibility scaffolding (evaluation scripts, leaderboard, submission guidelines, Docker images) used by the architecture-comparison protocol described in [docs/roadmap.md](../roadmap.md). A public-facing community launch — external submissions, public leaderboard — is in Future Directions; the infrastructure documented here is the internal tooling that persists through Phase 6 and Phase 7.

## Documentation

| Document | Description |
|----------|-------------|
| [**LEADERBOARD.md**](LEADERBOARD.md) | Current benchmark rankings (auto-generated) |
| [**SUBMISSION_GUIDE.md**](SUBMISSION_GUIDE.md) | Step-by-step submission process |
| [**EVALUATION.md**](EVALUATION.md) | Scoring methodology and metrics |
| [**REPRODUCIBILITY.md**](REPRODUCIBILITY.md) | Requirements for valid submissions |

## Related Documentation

| Document | Description |
|----------|-------------|
| [**BENCHMARKS.md**](../../BENCHMARKS.md) | Overview and quick start guide |
| [**OPTIMIZATION_METHODS.md**](../OPTIMIZATION_METHODS.md) | Which optimization works for each architecture |

## Quick Start

```bash
# 1. Run 10+ independent training sessions
for session in {1..10}; do
    uv run scripts/run_simulation.py \
        --config configs/your_config.yml \
        --track-experiment \
        --runs 50
done

# 2. Submit all sessions together
uv run scripts/benchmark_submit.py \
    --experiments experiments/* \
    --category foraging_small/classical \
    --contributor "Your Name"

# 3. Validate your submission
uv run scripts/evaluate_submission.py \
    --submission benchmarks/<category>/<timestamp>.json
```

## Regenerating Leaderboards

To update the leaderboard after new submissions:

```bash
uv run scripts/benchmark_submit.py regenerate
```

This updates:

- `docs/nematodebench/LEADERBOARD.md` - Full leaderboard tables
- `README.md` - Current Leaders section
