# Contributing to Quantum Nematode

Thank you for your interest in contributing. This guide covers development setup, the quality tooling, the test tiers, how to extend the platform, and the pull-request process. For *running* simulations, evolution and analysis see the [usage guide](docs/usage.md); for the science see the [roadmap](docs/roadmap.md) and the [experiment logbooks](docs/experiments/README.md).

## Development setup

### Prerequisites

- **Python 3.13** — exactly (`>=3.13,<3.14`); `uv` fetches it if it is not installed
- [**uv**](https://github.com/astral-sh/uv) for dependency management
- [**Git LFS**](https://git-lfs.com) — model weights, evolution checkpoints, curated artifacts and the connectome spreadsheets live in LFS. The committed `.lfsconfig` limits a fresh clone to `data/**` (~4 MB); the curated artifacts (~620 MB) are fetched only on request

```bash
# macOS
brew install uv git-lfs
# Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
sudo apt-get install git-lfs

git lfs install            # once per machine
git clone https://github.com/SyntheticBrains/nematode.git
cd nematode
git lfs ls-files | head    # `*` = present locally, `-` = still a pointer

# Only if you need to reproduce a logbook: fetch the curated artifacts (~620 MB, or a narrower path)
git lfs pull --include='artifacts/**'
```

### Install dependencies

Pick the extras you need — only `cpu` and `gpu` conflict:

| Extra | Adds | Needed for |
|---|---|---|
| `cpu` | Qiskit Aer simulator (CPU) | Any quantum brain on the simulator |
| `gpu` | Qiskit Aer on CUDA 11 (`qiskit-aer-gpu-cu11`) | Quantum simulation on an NVIDIA GPU |
| `torch` | PyTorch | Every classical, recurrent, spiking, hybrid and connectome brain |
| `qpu` | `qiskit-ibm-runtime`, `qiskit-ibm-catalog` | Real IBM Quantum hardware (`--device qpu`) |
| `pixel` | Pygame | The `pixel` and `pixel_continuous` renderers |
| `analysis` | scikit-learn, SciPy | The analysis scripts under `scripts/analysis/` |

```bash
# Typical development install
uv sync --extra cpu --extra torch --extra pixel --extra analysis

# What the pre-commit workflow installs (the test workflow uses cpu, pixel, torch)
uv sync --extra analysis --extra cpu --extra pixel --extra qpu --extra torch --dev
```

> **CUDA 11, not 12.** Qiskit stopped publishing the CUDA-12 `qiskit-aer-gpu` build after 0.15.1 (its newest wheel is cp312), so the `gpu` extra tracks `qiskit-aer-gpu-cu11`, which ships cp313 wheels. Your NVIDIA driver needs CUDA 11 support (driver ≥ 450).

### Environment file

Only needed for quantum hardware. Copy the template and fill in the values from your IBM Quantum and Q-CTRL accounts:

```bash
cp .env.template .env
```

```env
IBM_QUANTUM_API_KEY=…
IBM_QUANTUM_BACKEND=…      # e.g. ibm_brisbane
IBM_QUANTUM_CHANNEL=…
IBM_QUANTUM_CRN=…
QCTRL_API_KEY=…            # for --optimize (Fire Opal)
```

## Quality tooling

Install the pre-commit hooks once; they run on every commit:

```bash
uv run pre-commit install
uv run pre-commit run -a      # run everything manually
```

The hooks run Ruff (lint with `select = ALL` and format, line length 100), Pyright, mdformat and markdownlint for Markdown, YAML/TOML validation, large-file and end-of-file checks, and the fast pytest tier. Configuration lives in [`pyproject.toml`](pyproject.toml), [`.pre-commit-config.yaml`](.pre-commit-config.yaml) and [`.markdownlint.jsonc`](.markdownlint.jsonc). Docstrings follow the NumPy convention.

## Tests

Four tiers, selected with pytest markers:

| Tier | Marker | What | When it runs |
|---|---|---|---|
| Unit / integration | (none) | Fast in-process tests | Pre-commit hook, every PR |
| Slow integration | `slow` | Heavy in-process runs, e.g. a real `EvolutionLoop` | Before pushing, especially for `evolution/` changes; every PR in CI |
| Smoke | `smoke` | The CLI entry points end-to-end with minimal episodes | Every PR in CI |
| Nightly | `nightly` | Full training sessions asserted against benchmark ranges | 03:00 UTC daily, or manually from the Actions tab |

```bash
uv run pytest -m "not smoke and not nightly and not slow"   # fast tier (what pre-commit runs)
uv run pytest -m "not nightly"                               # everything except nightly — run after substantive changes
uv run pytest -m slow -v
uv run pytest -m smoke -v
uv run pytest -m nightly -v                                  # slow: full training sessions
uv run pytest -m nightly -k "foraging_small" -v              # one nightly config
```

Nightly benchmark ranges live in [`e2e_benchmarks.json`](packages/quantum-nematode/tests/quantumnematode_tests/e2e_benchmarks.json) and are derived from the logbooks; if you change a config or training parameter you may need to update them, with the logbook evidence for the new range.

**CI sharding.** The `Tests` workflow splits the suite into five `pytest-split` shards balanced by the committed `.test_durations` file. Tests missing from that file still run, so it only needs regenerating when one shard is visibly slower than the others:

```bash
uv run pytest -m "not nightly" -p no:randomly --store-durations --durations-path .test_durations
```

Locally the suite is unsharded and `-n logical` uses every logical core.

## Repository layout

```text
packages/quantum-nematode/quantumnematode/
  brain/arch/      26 brain architectures, self-registering via @register_brain (see docs/architectures.md)
  brain/modules.py sensory feature modules shared by the brains
  env/             grid and continuous-2D environments, sensing, predators, pheromones, themes
  agent/           agent orchestration, rewards, metrics, multi-agent simulation
  connectome/      Cook et al. 2019 connectome loader and neuron metadata
  evolution/       CMA-ES / GA / TPE loops, inheritance strategies, co-evolution
  optimizers/      parameter-shift rule, CMA-ES and GA optimisers, learning-rate schedules
  executors/       CPU / GPU / QPU backends
  experiment/      experiment tracking, metadata, convergence detection
  validation/      real-worm behavioural validation (bias curves)
  report/          session summaries, plots, CSV export
  utils/           config loader, brain factory, interrupt handling
  logging_config.py  structured logging (logs/simulation_<session-id>.log)
packages/quantum-nematode/tests/   the test suite (all tiers)
scripts/           CLI entry points, analysis scripts, Phase 5 campaign drivers
configs/           scenario, evolution and special configs (see configs/README.md)
docs/              roadmap, logbooks, biology, guides (see docs/README.md)
openspec/          specs and change proposals (spec-driven development)
artifacts/         curated experiment outputs referenced by logbooks (Git LFS)
data/              vendored connectome data and behavioural reference values
```

## Extending the platform

### Adding a brain architecture

Brains are plugins: a vanilla addition touches at most six files and never adds a per-architecture branch to the simulation or training loops. The [plugin developer guide](docs/architecture/plugin-developer-guide.md) is the canonical walkthrough; the essentials:

1. Create `packages/quantum-nematode/quantumnematode/brain/arch/<name>.py` with a Pydantic config class inheriting `BrainConfig` and a brain class inheriting the appropriate base (`QuantumBrain` or `ClassicalBrain`), decorated so it self-registers at import time:

   ```python
   from quantumnematode.brain.arch import ClassicalBrain
   from quantumnematode.brain.arch._registry import register_brain
   from quantumnematode.brain.arch.dtypes import BrainConfig, BrainType


   class MyNewBrainConfig(BrainConfig):
       ...


   @register_brain(
       name="mynewbrain",              # must equal BrainType.MYNEWBRAIN.value
       config_cls=MyNewBrainConfig,
       brain_type=BrainType.MYNEWBRAIN,
       families=("classical",),        # e.g. "classical", "quantum", "spiking"
   )
   class MyNewBrain(ClassicalBrain):
       ...
   ```

2. Add the `BrainType` enum member, import the module in `brain/arch/__init__.py`, and add the config class to the loader's union (the guide lists the exact files).

3. Add tests under `packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/`, and a scenario config under `configs/scenarios/` if you are shipping a runnable baseline.

4. Add the row to [docs/architectures.md](docs/architectures.md) and the enumeration in [AGENTS.md](AGENTS.md).

### Adding a sensory module

1. Define the module in `quantumnematode.brain.modules` with its feature-extraction logic.
2. Add it to the `DEFAULT_MODULES` mapping.
3. Cover it with module tests and check it against the existing brains.

### Adding an environment feature

1. Extend the environment classes in `quantumnematode.env` (`BaseEnvironment` is the base; `DynamicForagingEnvironment` and `Continuous2DEnvironment` are the two substrates).
2. Keep the `BrainParams` interface compatible — every brain reads its inputs through it.
3. Add rendering support for the feature in the relevant theme(s).
4. Track new metrics in `EpisodeTracker` and add the corresponding plots and CSV exports.
5. Anything that changes behaviour when *off* is a bug: new features must be byte-identical no-ops when disabled, and the regression tests check that.

### Adding a scenario config

Follow the naming convention and the "copy the closest config, change only what differs" rule in [configs/README.md](configs/README.md).

## Code style

- **Type hints everywhere**; Pyright runs in pre-commit with `reportMissingImports = "error"`.
- **NumPy-style docstrings** on public functions and classes.
- **Descriptive errors**: build the message, log it, then raise (`logger.error(msg); raise TypeError(msg)`).
- **Structured logging** through `quantumnematode.logging_config.logger`, not `print` (CLI scripts are the exception).
- **Pydantic models** for data structures and configs; PascalCase classes, snake_case functions, UPPER_SNAKE_CASE constants; leading underscore for private modules.

## Workflow and pull requests

1. **Scope the change.** Non-trivial work starts as an [OpenSpec](https://github.com/Fission-AI/OpenSpec) change under `openspec/changes/<name>/` (proposal, design, tasks, spec deltas) and is archived under `openspec/changes/archive/` when it lands. Small fixes do not need one.
2. **Branch** from `main` (`feat/…`, `fix/…`, `docs/…`).
3. **Develop and test.** Write tests for new behaviour; run the fast tier as you go and `uv run pytest -m "not nightly"` before pushing; run `uv run pre-commit run -a`.
4. **Document.** Update docstrings, the relevant guide under `docs/`, and `AGENTS.md` if commands or layout changed. Experimental results go into a numbered logbook (see [docs/experiments/README.md](docs/experiments/README.md)); if they change a phase status, update the roadmap.
5. **Open the PR.** Titles **must** use a [Conventional Commits](https://www.conventionalcommits.org/) prefix — `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:`, with `!` for breaking changes (e.g. `feat!: remove legacy preprocessing mode`). Commit messages need no prefix. Describe the change, link related issues, and say how you tested it.

PR checklist:

- [ ] Tests pass (`uv run pytest -m "not nightly"`)
- [ ] Pre-commit passes (`uv run pre-commit run -a`)
- [ ] Documentation updated (docs, `AGENTS.md`, logbook if there are results)
- [ ] `CHANGELOG.md` has a line under *Unreleased* for any user-facing change
- [ ] Type hints added
- [ ] Disabled-by-default features are byte-identical no-ops when off
- [ ] Benchmark ranges updated if training behaviour changed, with evidence

## Releasing

Releases are tags on `main` plus a GitHub Release; there is no PyPI publication. To cut one:

1. In `CHANGELOG.md`, turn *Unreleased* into the new version with the date, breaking changes first, and update the compare links at the bottom.
2. Set the version in both `pyproject.toml` files and in `CITATION.cff` (with `date-released`), then run `uv lock` so the lock file records it.
3. Run `uv run pytest -m "not nightly"`, `uv run pre-commit run -a`, `uv build`, and `pip-audit` over the exported lock (`uv export --format requirements-txt --no-hashes --no-emit-project --extra cpu --extra torch --extra pixel --extra qpu --extra analysis`).
4. Merge the release PR, then on the merge commit: `git tag -a vX.Y.Z -m vX.Y.Z && git push origin vX.Y.Z`.
5. `gh release create vX.Y.Z --title vX.Y.Z --notes-file <the CHANGELOG section> --generate-notes` — the changelog section leads, GitHub's pull-request list follows.

## Where help is wanted

Aligned with the [roadmap](docs/roadmap.md):

- **Phase 7 plasticity** — STDP/Hebbian and neuromodulator-gated three-factor rules on the connectome substrate; receptor-class metadata from CeNGEN; a minimal metabolic-state model.
- **Environment vectorisation** — a vmappable `Continuous2DEnvironment` is the binding constraint on Phase 6b's NEAT topology search.
- ***P. pacificus* connectome import** — Cook et al. 2025 data through the existing L0/L1 pipeline.
- **Validation** — a predator/mechanosensation behavioural validation arm; named-neuron grounding (NeuroPAL/WormID) for the connectome brain.
- **Reproductions** — rerun a logbook on your hardware and report what you get; discrepancies are findings.
- **Documentation and tutorials**, and test coverage for the scripts under `scripts/`.

## Community

- [Issues](https://github.com/SyntheticBrains/nematode/issues) for bugs and feature requests
- [Discussions](https://github.com/SyntheticBrains/nematode/discussions) for questions and ideas
- Everyone participating is expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md)

## License

By contributing you agree that your contributions are licensed under the [Apache License 2.0](LICENSE).
