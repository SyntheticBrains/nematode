# Quantum Nematode — AI Assistant Instructions

## Project Overview

Quantum Nematode simulates a simplified C. elegans navigating dynamic environments to find food, using quantum variational circuits and classical ML alternatives. Research platform for quantum machine learning.

## Tech Stack

- Python 3.12 (strictly >=3.12,\<3.13)
- Quantum: Qiskit 1.0+, Qiskit-Aer 0.17+, Qiskit-IBMRuntime 0.40+ (QPU)
- Classical ML: PyTorch 2.7+
- Config: Pydantic 2.11.4+, PyYAML 6.0+
- Viz: Matplotlib 3.10+, Rich 13.0+
- Optimization: CMA 4.0+ (evolutionary), custom gradient methods (PSR)
- Tooling: uv, Ruff, Pyright, pytest, pre-commit
- Docker with NVIDIA Container Toolkit for GPU support

## Common Commands

- Install: `uv sync --extra cpu --extra torch`
- Test (default, excludes nightly): `uv run pytest -m "not nightly"`
- Test (pre-commit subset, fast): `uv run pytest -m "not smoke and not nightly and not slow"`
- Test (slow integration only): `uv run pytest -m slow -v`
- Test (smoke only): `uv run pytest -m smoke -v`
- Test (nightly E2E only): `uv run pytest -m nightly -v`
- Test (all, including nightly): `uv run pytest`
- Lint/format: `uv run pre-commit run -a`
- Run simulation: `uv run ./scripts/run_simulation.py --config ./configs/scenarios/<scenario>/<config>.yml`

## Key Directories

- `packages/quantum-nematode/quantumnematode/` — Main source code
  - `brain/arch/` — 24 brain architectures: qvarcircuit, qqlearning, qrc, qrh, qef, crh, qrhqlstm, crhqlstm, qsnnreinforce, qsnnppo, qliflstm, hybridquantum, hybridclassical, hybridquantumcortex, mlpreinforce, mlpdqn, mlpppo, lstmppo, spikingreinforce, connectomeppo, feedforwardga, cfcppo, spikingppo, equivariantquantum. Plug-in registry: each Brain self-registers via `@register_brain`; see [docs/architecture/plugin-developer-guide.md](docs/architecture/plugin-developer-guide.md) for how to add a new one.
  - `env/` — Environment simulation
  - `agent/` — Agent orchestration, rewards, metrics
  - `experiment/` — Experiment tracking and benchmarking
  - `optimizers/` — Learning algorithms (PSR, CMA-ES)
- `scripts/` — CLI entry points (run_simulation.py, run_evolution.py, benchmark_submit.py)
- `configs/scenarios/` — YAML config files organized by scenario (`{brain}_{size}[_{variant}]_{sensing}.yml`)
  - Scenarios: `foraging`, `pursuit`, `stationary`, `thermal_foraging`, `thermal_pursuit`, `thermal_stationary`, `oxygen_foraging`, `oxygen_pursuit`, `oxygen_stationary`, `oxygen_thermal_foraging`, `oxygen_thermal_pursuit`, `oxygen_thermal_stationary`, `multi_agent_foraging`, `multi_agent_pursuit`, `multi_agent_stationary`, `foraging_predator_thermal`
  - Sensing suffixes: `_oracle`, `_temporal`, `_derivative`, `_klinotaxis`
  - Variant suffixes: `_classical`, `_fair`, `_separable`, `_modality_paired`, `_pheromone`, `_no_pheromone`, `_social`, `_aggregation`, `_full_social`, `_scarcity`, `_propfood`, `_mixed_phenotype`, etc.
  - Example: `configs/scenarios/foraging/mlpppo_small_oracle.yml`, `configs/scenarios/thermal_pursuit/lstmppo_large_temporal.yml`
- `configs/evolution/` — Evolutionary optimization configs
- `configs/special/` — One-off experimental configs
- `tests/` — Three-tier testing (unit, smoke, nightly)
- `benchmarks/` — Submitted benchmark results
- `openspec/` — Spec-driven development framework

## Code Conventions

- PascalCase classes, snake_case functions, UPPER_SNAKE_CASE constants
- Leading underscore for private modules (`_brain.py`)
- Comprehensive type annotations required
- NumPy-style docstrings
- Pydantic BaseModel for data structures
- Line length: 100 (Ruff)

## Testing

Four tiers:

1. **Unit/Integration** — Default pytest, runs on commits via pre-commit
2. **Slow** (`@pytest.mark.slow`) — Heavy in-process integration (real `EvolutionLoop` runs etc.), excluded from pre-commit, run before push
3. **Smoke** (`@pytest.mark.smoke`) — CLI end-to-end, runs on PRs
4. **Nightly** (`@pytest.mark.nightly`) — Full training benchmarks, runs daily

Pre-commit runs only the fast tier (`not smoke and not nightly and not slow`). Run `uv run pytest -m "not nightly"` (includes slow + smoke) after substantive changes, especially when touching `evolution/`, and `uv run pre-commit run -a` before committing.

## Pull Requests

PR titles MUST use [Conventional Commits](https://www.conventionalcommits.org/) prefixes. Common types in this repo: `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:`. Use `!` after the type for breaking changes (e.g. `feat!: remove legacy preprocessing mode`). Examples from project history: `feat: Add aerotaxis (oxygen sensing) system`, `fix: multi-agent sensing - use agent's own position in BrainParams`, `docs: Klinotaxis Era multi-agent pheromone evaluation (Logbook 011)`.

Commit messages do not require this prefix — only PR titles do.

<!-- markdownlint-disable MD025 -->
