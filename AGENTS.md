# Quantum Nematode — AI Assistant Instructions

## Project Overview

Quantum Nematode is a closed-loop sensory-motor simulation platform: a simulated C. elegans forages, evades predators and navigates thermal/oxygen gradients while a pluggable brain architecture (MLP, recurrent, spiking, reservoir, quantum, hybrid, GA-evolved, or constrained to the real 302-neuron connectome) is trained by learning or evolution. The research question is which architecture best learns nematode behaviours, with the C. elegans connectome as the focal comparison point; quantum circuits are one architecture family in that comparison, not the organising principle (see docs/roadmap.md). Results are ranked under a paired-seed statistical protocol and validated against published C. elegans behavioural data.

## Tech Stack

- Python 3.13 (strictly >=3.13,\<3.14)
- Quantum: Qiskit 2.x, Qiskit-Aer 0.17+, Qiskit-IBMRuntime 0.40+ (QPU)
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
- Real-worm behavioural-chemotaxis validation: set `sensing.capture_behaviour: true` in a foraging config (default `false` — a byte-identical no-op when off) to log a per-run behavioural trajectory to `exports/<session>/session/data/behaviour_capture.json`, then grade the klinokinesis + weathervane bias curves against the *C. elegans* literature with `uv run python scripts/analysis/behavioural_chemotaxis_validation.py --manifest <manifest.txt, one "<seed> <behaviour_capture.json>" pair per line> --tail-runs 100 --out <behavioural_curves.json> [--figure-dir <dir>] [--theta-sharp 0.45]` (the `--out` path writes the graded summary JSON; see [Logbook 035](docs/experiments/logbooks/035-realworm-chemotaxis-validation.md)). For **thermotaxis** validation set `sensing.capture_behaviour_modality: thermotaxis` (default `food`; it makes the captured drive the homeostatic thermal setpoint error `−|T−Tc|` so the same bias curves apply) and pass `--modality thermotaxis` to the harness (grades against the thermal `data/thermotaxis/` reference set; [Logbook 036](docs/experiments/logbooks/036-realworm-thermotaxis-validation.md)).
- L4 plasticity panel: run the recipe pilot with `uv run python scripts/campaigns/l4_panel_pilot.py --out campaigns/l4-pilot` (grid configs derived beside the results; add `--only <arm> --rate <r> --runs 6000` to extend one arm as a fresh run), summarise it with `uv run python scripts/analysis/l4_panel.py --pilot --campaign-dir campaigns/l4-pilot --out docs/experiments/logbooks/supporting/040-l4-panel/pilot.json` (prints the selected rate and the budget rule's output), then analyse the panel with `uv run python scripts/analysis/l4_panel.py --campaign-dir <campaign-dir> --out panel.json --csv per-seed.csv --curves curves.csv` (confirmatory mode accepts seeds 1–8 only; the four-test family, the band test and the verdict map are fixed in the script).

## Key Directories

- `packages/quantum-nematode/quantumnematode/` — Main source code
  - `brain/arch/` — 26 brain architectures: qvarcircuit, qrc, qrh, qef, crh, qrhqlstm, crhqlstm, qsnnreinforce, qsnnppo, qliflstm, hybridquantum, hybridclassical, hybridquantumcortex, mlpreinforce, mlpdqn, mlpppo, lstmppo, spikingreinforce, connectomeppo, feedforwardga, cfcppo, spikingppo, equivariantquantum, transformerppo, mingruppo, minlstmppo. Plug-in registry: each Brain self-registers via `@register_brain`; see [docs/architecture/plugin-developer-guide.md](docs/architecture/plugin-developer-guide.md) for how to add a new one.
  - `env/` — Environment simulation
  - `agent/` — Agent orchestration, rewards, metrics
  - `experiment/` — Experiment tracking, metadata, and convergence analysis
  - `optimizers/` — Learning algorithms (PSR, CMA-ES)
- `scripts/` — CLI entry points (run_simulation.py, run_evolution.py, experiment_query.py)
- `configs/scenarios/` — YAML config files organized by scenario (`{brain}_{size}[_{variant}]_{sensing}.yml`)
  - Scenarios: `foraging`, `pursuit`, `stationary`, `thermal_foraging`, `thermal_pursuit`, `thermal_stationary`, `oxygen_foraging`, `oxygen_pursuit`, `oxygen_stationary`, `oxygen_thermal_foraging`, `oxygen_thermal_pursuit`, `oxygen_thermal_stationary`, `multi_agent_foraging`, `multi_agent_pursuit`, `multi_agent_stationary`, `foraging_predator_thermal`, `bit_memory`, `associative_memory`
  - Sensing suffixes: `_oracle`, `_temporal`, `_derivative`, `_klinotaxis`
  - Variant suffixes: `_classical`, `_fair`, `_separable`, `_modality_paired`, `_pheromone`, `_no_pheromone`, `_social`, `_aggregation`, `_full_social`, `_scarcity`, `_propfood`, `_mixed_phenotype`, `_ars_depletion`, `_no_respawn_control`, `_rewired_null`, etc.
  - Task suffixes: `_bit_memory` (the `bit_memory` family — a non-spatial delayed-match-to-cue working-memory positive control; spatial/foraging/predator/thermal dynamics are disabled, so it is its own family rather than a variant of a spatial scenario); `_associative_memory` (the `associative_memory` family — a non-spatial chemosensory delayed-associative-match with probabilistic within-trial reversal, a working-memory *update* probe; observation = cue + outcome + go only, so like `bit_memory` it is its own family, not a spatial variant)
  - Example: `configs/scenarios/foraging/mlpppo_small_oracle.yml`, `configs/scenarios/thermal_pursuit/lstmppo_large_temporal.yml`
- `configs/evolution/` — Evolutionary optimization configs
- `configs/special/` — One-off experimental configs
- `tests/` — Four-tier testing (unit, slow, smoke, nightly)
- `openspec/` — Spec-driven development framework

## Code Conventions

- PascalCase classes, snake_case functions, UPPER_SNAKE_CASE constants
- Leading underscore for private modules (`_brain.py`)
- Comprehensive type annotations required
- NumPy-style docstrings
- Pydantic BaseModel for data structures
- Line length: 100 (Ruff)
- **No planning-doc references in implementation code or docstrings**: never cite roadmap
  sections, decision IDs (D1/D7/…), OpenSpec changes, review-finding labels (B1/S3/…), phase or
  milestone names, logbooks, or issue/PR numbers inside `packages/` source. State the technical
  constraint itself — the reader of the code gets the *why* as an intrinsic property, not a
  pointer into planning history. (Tests are the exception: test docstrings SHOULD name the
  OpenSpec spec scenarios they cover.) Planning provenance belongs in commit messages, OpenSpec
  changes, and logbooks.

## Testing

Four tiers:

1. **Unit/Integration** — Default pytest, runs on commits via pre-commit
2. **Slow** (`@pytest.mark.slow`) — Heavy in-process integration (real `EvolutionLoop` runs etc.), excluded from pre-commit, run before push
3. **Smoke** (`@pytest.mark.smoke`) — CLI end-to-end, runs on PRs
4. **Nightly** (`@pytest.mark.nightly`) — Full training benchmarks, runs daily

Pre-commit runs only the fast tier (`not smoke and not nightly and not slow`). Run `uv run pytest -m "not nightly"` (includes slow + smoke) after substantive changes, especially when touching `evolution/`, and `uv run pre-commit run -a` before committing.

### CI sharding and `.test_durations`

CI splits the suite across five parallel shards with `pytest-split`, balanced by
the committed `.test_durations` file and assigned with the `least_duration`
algorithm.

A test missing from that file still runs, but it is **not** placed without an
estimate: `pytest-split` budgets it at the **mean of the recorded durations**. So
drift is silent and it is not free — on 2026-09-13 a three-week-old file left 25%
of the suite unmeasured, including one 100.7s test budgeted at 0.096s, and the
shards' true loads spread 2.36x (135s to 319s) while `pytest-split` reported its
own split as balanced to 1.03x. It cannot see the problem, because it scores each
split with the same stale numbers that produced it.

The `test durations freshness` pre-commit hook is the guard: it fails when more
than 10% of the collected suite has no recorded duration. Regenerate with:

```bash
uv run pytest -m "not nightly" --store-durations --clean-durations --durations-path .test_durations
```

`--clean-durations` drops entries for tests that no longer exist; without it the
file accumulates them. This takes ~2.5 minutes and matches how CI runs, since
`-n logical` comes from `addopts`. `pytest-split` writes the file without a
trailing newline, so run `pre-commit` afterwards — the `fix end of files` hook
adds it, and CI's Code Quality job fails without it.

Locally the suite is unsharded; `-n logical` in `addopts` uses every logical core.

## Pull Requests

PR titles MUST use [Conventional Commits](https://www.conventionalcommits.org/) prefixes. Common types in this repo: `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:`. Use `!` after the type for breaking changes (e.g. `feat!: remove legacy preprocessing mode`). Examples from project history: `feat: Add aerotaxis (oxygen sensing) system`, `fix: multi-agent sensing - use agent's own position in BrainParams`, `docs: Klinotaxis Era multi-agent pheromone evaluation (Logbook 011)`.

Commit messages do not require this prefix — only PR titles do.

User-facing changes add a line to `CHANGELOG.md` under *Unreleased* (Keep a Changelog format; breaking changes first). Releases follow the checklist in CONTRIBUTING.md § Releasing.

<!-- markdownlint-disable MD025 -->
