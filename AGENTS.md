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
- Test (smoke only): `uv run pytest -m smoke -v`
- Test (nightly E2E only): `uv run pytest -m nightly -v`
- Test (all, including nightly): `uv run pytest`
- Lint/format: `uv run pre-commit run -a`
- Run simulation: `uv run ./scripts/run_simulation.py --config ./configs/scenarios/<scenario>/<config>.yml`

## Key Directories

- `packages/quantum-nematode/quantumnematode/` — Main source code
  - `brain/arch/` — 19 brain architectures: qvarcircuit, qqlearning, qrc, qrh, qef, crh, qrhqlstm, crhqlstm, qsnnreinforce, qsnnppo, qliflstm, hybridquantum, hybridclassical, hybridquantumcortex, mlpreinforce, mlpdqn, mlpppo, lstmppo, spikingreinforce
  - `env/` — Environment simulation
  - `agent/` — Agent orchestration, rewards, metrics
  - `experiment/` — Experiment tracking and benchmarking
  - `optimizers/` — Learning algorithms (PSR, CMA-ES)
- `scripts/` — CLI entry points (run_simulation.py, run_evolution.py, benchmark_submit.py)
- `configs/scenarios/` — YAML config files organized by scenario (`{brain}_{size}[_{variant}]_{sensing}.yml`)
  - Scenarios: `foraging`, `predators`, `pursuit`, `stationary`, `thermal_foraging`, `thermal_pursuit`, `thermal_stationary`, `health_predators`
  - Sensing suffixes: `_oracle`, `_temporal`, `_derivative`
  - Variant suffixes: `_classical`, `_fair`, `_separable`, `_modality_paired`, etc.
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

Three tiers:

1. **Unit/Integration** — Default pytest, runs on commits
2. **Smoke** (`@pytest.mark.smoke`) — CLI end-to-end, runs on PRs
3. **Nightly** (`@pytest.mark.nightly`) — Full training benchmarks, runs daily

Always run `uv run pytest -m "not nightly"` after changes. Run `uv run pre-commit run -a` before committing.

<!-- markdownlint-disable MD025 -->
