# Experiment Tracking and Benchmarking System Design

## Architecture Overview

### Two-Tier System Design

The system implements two distinct but related workflows:

1. **Experiment Tracking** (Lightweight, Auto-tracking)
   - Captures metadata for every simulation run when `--track-experiment` flag is used
   - Stores in `experiments/{timestamp}.json`
   - Focused on reproducibility and historical comparison
   - Low overhead, minimal user interaction

2. **Benchmark Management** (Curated, Quality-controlled)
   - Subset of experiments explicitly marked as benchmarks
   - Requires `--save-benchmark` flag and additional metadata (contributor name)
   - Stores in `benchmarks/{category}/{timestamp}.json`
   - Subject to validation rules (minimum runs, success criteria)
   - Displayed in README.md and BENCHMARKS.md

### Data Model

#### ExperimentMetadata Structure
```python
@dataclass
class ExperimentMetadata:
    # Identity
    experiment_id: str          # timestamp-based ID
    timestamp: datetime

    # Configuration
    config_file: str            # relative path to config file
    config_hash: str            # SHA256 of config content

    # Git Context
    git_commit: str | None      # current commit hash
    git_branch: str | None      # current branch
    git_dirty: bool             # uncommitted changes?

    # Environment Setup
    environment: EnvironmentMetadata
    brain: BrainMetadata

    # Results
    results: ResultsMetadata

    # System Info
    system: SystemMetadata

    # Optional: Benchmark-specific
    benchmark: BenchmarkMetadata | None
```

#### Metadata Components
```python
@dataclass
class EnvironmentMetadata:
    type: str                   # "static" | "dynamic"
    grid_size: int
    # Dynamic-specific
    num_foods: int | None
    initial_satiety: float | None
    # ... other env params

@dataclass
class BrainMetadata:
    type: str                   # "modular" | "mlp" | "qmodular" | etc.
    qubits: int | None          # quantum brains only
    shots: int | None           # quantum brains only
    hidden_dim: int | None      # classical brains only
    learning_rate: float
    # ... other brain params

@dataclass
class ResultsMetadata:
    total_runs: int
    success_rate: float
    avg_steps: float
    avg_reward: float
    # Foraging-specific
    avg_foods_collected: float | None
    avg_distance_efficiency: float | None
    # Termination breakdown
    completed_all_food: int
    starved: int
    max_steps_reached: int
    goal_reached: int

@dataclass
class SystemMetadata:
    python_version: str
    qiskit_version: str
    torch_version: str | None
    device_type: str            # "cpu" | "gpu" | "qpu"
    qpu_backend: str | None

@dataclass
class BenchmarkMetadata:
    contributor: str            # display name
    github_username: str | None
    category: str               # e.g., "dynamic_medium_quantum"
    notes: str | None           # optimization details
    verified: bool              # admin verification flag
```

### Storage Strategy

#### File Organization
```text
nematode/
├── experiments/               # Auto-tracked experiments (gitignored)
│   └── {timestamp}.json       # One file per experiment
│
├── benchmarks/                # Curated benchmarks (git-tracked)
│   ├── static_maze/
│   │   ├── quantum/
│   │   │   └── {timestamp}.json
│   │   └── classical/
│   │       └── {timestamp}.json
│   │
│   ├── dynamic_small/
│   │   ├── quantum/
│   │   └── classical/
│   │
│   ├── dynamic_medium/
│   │   ├── quantum/
│   │   └── classical/
│   │
│   └── dynamic_large/
│       ├── quantum/
│       └── classical/
│
└── exports/                   # Existing detailed data exports (gitignored)
    └── {timestamp}/
        └── ...
```

#### JSON Format Example
```json
{
  "experiment_id": "20251109_143022",
  "timestamp": "2025-11-09T14:30:22.456Z",
  "config_file": "configs/examples/modular_dynamic_medium.yml",
  "config_hash": "a3f5c9d8...",

  "git_commit": "abc123def",
  "git_branch": "main",
  "git_dirty": false,

  "environment": {
    "type": "dynamic",
    "grid_size": 50,
    "num_foods": 20,
    "max_active_foods": 30,
    "initial_satiety": 500.0,
    "satiety_decay_rate": 1.0,
    "viewport_size": [11, 11]
  },

  "brain": {
    "type": "modular",
    "qubits": 2,
    "shots": 1500,
    "num_layers": 2,
    "learning_rate": 1.0,
    "modules": {"chemotaxis": [0, 1]}
  },

  "results": {
    "total_runs": 20,
    "success_rate": 0.85,
    "avg_steps": 245.3,
    "avg_reward": 15.24,
    "avg_foods_collected": 18.5,
    "avg_distance_efficiency": 0.823,
    "completed_all_food": 17,
    "starved": 2,
    "max_steps_reached": 1,
    "goal_reached": 0
  },

  "system": {
    "python_version": "3.12.0",
    "qiskit_version": "1.3.1",
    "torch_version": null,
    "device_type": "cpu",
    "qpu_backend": null
  },

  "benchmark": {
    "contributor": "John Doe",
    "github_username": "johndoe",
    "category": "dynamic_medium_quantum",
    "notes": "Optimized learning rate schedule with inverse time decay",
    "verified": false
  }
}
```

### Query System Design

#### Query Operations
```python
# List experiments with filters
experiments = query_experiments(
    environment_type="dynamic",
    brain_type="modular",
    min_success_rate=0.8,
    limit=10
)

# Compare two experiments
comparison = compare_experiments(exp_id_1, exp_id_2)

# Get best performing experiment for a category
best = get_best_experiment(
    category="dynamic_medium_quantum",
    metric="avg_foods_collected"
)

# List benchmarks by category
benchmarks = list_benchmarks(category="dynamic_medium")
```

#### Query CLI Examples
```bash
# List all tracked experiments
uv run scripts/experiment_query.py list

# Filter by environment type
uv run scripts/experiment_query.py list --env-type dynamic --brain-type modular

# Compare two experiments
uv run scripts/experiment_query.py compare 20251109_143022 20251108_120000

# Show best performers
uv run scripts/experiment_query.py leaderboard --category dynamic_medium
```

### Benchmark Submission Workflow

#### Submission Process
1. User runs simulation with `--save-benchmark` flag
2. System performs pre-submission validation:
   - Minimum 20 runs completed
   - Config file exists and is tracked in git
   - Git repository is clean (no uncommitted changes)
   - Results meet quality threshold (configurable)
3. User prompted for additional metadata:
   - Contributor name (required)
   - GitHub username (optional)
   - Notes about optimization approach (optional)
4. Benchmark JSON generated in appropriate category directory
5. User creates PR with benchmark file
6. Maintainers review and verify results
7. Merge updates BENCHMARKS.md and README.md via CI/automation

#### Validation Rules
```python
@dataclass
class BenchmarkValidationRules:
    min_runs: int = 20
    min_success_rate: float | None = None  # environment-specific
    require_clean_git: bool = True
    require_config_in_repo: bool = True
    require_contributor_name: bool = True
```

### Leaderboard Generation

#### README.md Format
```markdown
## 🏆 Top Benchmarks

### Dynamic Foraging - Medium (50×50, 20 foods)

#### Quantum Brains
| Brain | Success Rate | Avg Steps | Foods/Run | Dist Eff | Contributor | Date |
|-------|--------------|-----------|-----------|----------|-------------|------|
| Modular | 95% | 245 | 19.8 | 0.85 | @johndoe | 2025-11-08 |
| QModular | 92% | 268 | 19.2 | 0.81 | @janedoe | 2025-11-07 |

#### Classical Brains
| Brain | Success Rate | Avg Steps | Foods/Run | Dist Eff | Contributor | Date |
|-------|--------------|-----------|-----------|----------|-------------|------|
| MLP | 90% | 280 | 18.9 | 0.79 | @mlpfan | 2025-11-06 |

See [BENCHMARKS.md](BENCHMARKS.md) for detailed results and reproduction instructions.
```

#### BENCHMARKS.md Format
- Full table for each category
- Reproduction instructions
- Notes from contributors
- Links to config files and commit hashes
- Verification status

### Integration Points

#### run_simulation.py Integration
```python
# Add CLI flags
parser.add_argument("--track-experiment", action="store_true")
parser.add_argument("--save-benchmark", action="store_true")
parser.add_argument("--benchmark-notes", type=str)

# After simulation completes
if args.track_experiment or args.save_benchmark:
    metadata = capture_experiment_metadata(
        config=config,
        results=all_results,
        metrics=metrics,
        timestamp=timestamp
    )

    if args.save_benchmark:
        save_benchmark(metadata, args.benchmark_notes)
    else:
        save_experiment(metadata)
```

### Technical Decisions

#### Why JSON over Database?
- **Simplicity**: No external dependencies, works everywhere
- **Git-friendly**: Easy to diff, review in PRs
- **Portable**: Easy to share, archive, analyze with any tool
- **Sufficient scale**: Expect <1000 experiments, <100 benchmarks
- **Transparent**: Users can inspect/edit files directly

#### Why Two-Tier System?
- **Flexibility**: Users choose level of formality
- **Low friction**: Auto-tracking doesn't require user input
- **Quality control**: Benchmarks maintain high standards
- **Clear separation**: Experiments vs. curated results

#### Why Hierarchical Categories?
- **Natural organization**: Environment type is primary differentiator
- **Fair comparison**: Quantum vs classical within same task
- **Scalability**: Easy to add new categories as project grows

### Future Extensions (Not in Initial Scope)

- Web-based leaderboard viewer
- Automatic PR generation for benchmarks
- Experiment diff visualization tool
- Integration with CI for automated benchmarking
- Export to standard formats (MLflow, wandb)
- Anomaly detection for suspicious results
