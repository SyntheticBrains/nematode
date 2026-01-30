# Quantum Nematode — AI Assistant Instructions

## Project Overview

Quantum Nematode simulates a simplified C. elegans navigating dynamic environments to find food, using quantum variational circuits and classical ML alternatives. Research platform for quantum machine learning.

## Tech Stack

- Python 3.12 (strictly enforced)
- Quantum: Qiskit 1.0+, Qiskit-Aer
- Classical ML: PyTorch 2.7+
- Config: Pydantic 2.11+, PyYAML
- Tooling: uv, Ruff, Pyright, pytest, pre-commit

## Common Commands

- Install: `uv sync --extra cpu --extra torch`
- Test (unit/integration): `uv run pytest`
- Test (smoke): `uv run pytest -m smoke -v`
- Test (nightly E2E): `uv run pytest -m nightly -v`
- Lint/format: `uv run pre-commit run -a`
- Run simulation: `uv run ./scripts/run_simulation.py --config ./configs/examples/<config>.yml`

## Key Directories

- `packages/quantum-nematode/quantumnematode/` — Main source code
  - `brain/arch/` — 6 brain architectures (modular, qmodular, mlp, ppo, qmlp, spiking)
  - `env/` — Environment simulation
  - `agent/` — Agent orchestration, rewards, metrics
  - `experiment/` — Experiment tracking and benchmarking
  - `optimizers/` — Learning algorithms (PSR, CMA-ES)
- `scripts/` — CLI entry points (run_simulation.py, run_evolution.py, benchmark_submit.py)
- `configs/examples/` — YAML config files ({brain}_{environment}_{size}.yml)
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

Always run `uv run pytest` after changes. Run `uv run pre-commit run -a` before committing.

<!-- markdownlint-disable MD025 -->

<!-- OPENSPEC:START -->

# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:

- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:

- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->
