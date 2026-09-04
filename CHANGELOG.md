# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow [Semantic Versioning](https://semver.org/): while the project is at 0.x, a minor release may contain breaking changes, and they are listed first under each release.

Releases before 0.5.0 are documented on [GitHub Releases](https://github.com/SyntheticBrains/nematode/releases); this file starts at 0.5.0.

## [Unreleased]

### Added

- The Phase 7 L4 trace substrate on the connectome brain: `enable_activity_traces` and `trace_decay` config fields (off by default; per-synapse eligibility traces accumulated during rollout forwards for the upcoming three-factor rules — training is bit-identical with traces on until a rule consumes them), and the new `quantumnematode.learning_rules` package, whose first citizen `ConnectomePPORule` is the connectome PPO update extracted byte-identically behind the `LearningRule` seam.

### Changed

- Connectome-brain runs now record the mean policy loss per PPO update in the tracked `losses` telemetry (the house convention; the brain previously recorded no loss), so connectome session exports gain a `losses` column.

## [0.5.0] - 2026-08-23

The pre-Phase-7 housekeeping release. It takes the platform that closed Phase 6a, removes the research tooling the platform no longer needs, moves the toolchain to Python 3.13 and Qiskit 2, and prepares the repository for a wider audience.

### Breaking changes

- **NematodeBench removed** ([#274](https://github.com/SyntheticBrains/nematode/pull/274)). The curated benchmark submission workflow, validation, category and leaderboard generation, `BENCHMARKS.md` and `docs/nematodebench/` are gone: across Phases 5 and 6 the architecture-comparison protocol read `--track-experiment` output directly and never used them. The one live component, the convergence detector and composite score, moved to `quantumnematode.experiment.convergence`; the 72 tracked sessions behind the old submissions were migrated to `artifacts/experiments/`; the `composite_benchmark_score` key is unchanged so historical artifacts still load.
- **`qqlearning` brain retired** ([#286](https://github.com/SyntheticBrains/nematode/pull/286)). `QQLearningBrain` was the last architecture with no path to the Phase 7 plasticity work. Configs with `brain.name: qqlearning` no longer load (the loader rejects unregistered brain types); 26 architectures remain, and every count in the docs is now derived from the registry. No tracked artifact carries the retired name.
- **Python 3.13 only, and the `gpu` extra moves to CUDA 11** ([#289](https://github.com/SyntheticBrains/nematode/pull/289)). `requires-python` is `>=3.13,<3.14` (it was pinned to 3.12). The CUDA-12 `qiskit-aer-gpu` build was abandoned upstream after 0.15.1 and has no 3.13 wheel, so the `gpu` extra now installs `qiskit-aer-gpu-cu11` (0.17.2): the bundled CUDA runtime goes from 12 to 11, which needs an NVIDIA driver ≥ 450 rather than ≥ 525. Every dependency was refreshed at the same time (torch 2.13, numpy 2.5, pydantic 2.13, scipy 1.18, scikit-learn 1.9, matplotlib 3.11, rich 15, optuna 4.9).
- **Qiskit 2.x** ([#290](https://github.com/SyntheticBrains/nematode/pull/290)). `qiskit>=2.0,<3`. The platform needed no code changes (it never used the APIs Qiskit 2 removed), but anything of yours built on Qiskit 1.x removals — `BackendV1`, `execute()`, `bind_parameters()`, `opflow`, `qiskit.pulse` — must migrate. Test-suite deprecation warnings dropped from ~382,000 to ~200.

### Added

- A README and documentation set designed for the project's public baseline ([#292](https://github.com/SyntheticBrains/nematode/pull/292)): the README leads with the research question and evidence-linked results; new [usage guide](docs/usage.md), [architecture catalogue](docs/architectures.md), [visualisation reference](docs/visualization.md) and [docs index](docs/README.md); `CITATION.cff`.
- `.lfsconfig` so a fresh clone fetches only the 4 MB of connectome data the code reads, not the 620 MB of curated logbook artifacts; `git lfs pull --include='artifacts/**'` fetches those on demand and CI fetches everything ([#294](https://github.com/SyntheticBrains/nematode/pull/294)).
- `SECURITY.md`, issue forms, a pull-request template, Dependabot for the lock file and GitHub Actions, package authorship metadata and a `.python-version` pin ([#296](https://github.com/SyntheticBrains/nematode/pull/296)).
- This changelog.

### Changed

- The Docker image installs the `torch` extra and copies `configs/` and `data/`, so every brain, scenario and the connectome data are available inside it; it is x86_64-only because the CUDA-11 Aer wheel is ([#292](https://github.com/SyntheticBrains/nematode/pull/292)).
- The continuous-2D screenshot export renders a short walk so the worm's body trail is visible; both documentation screenshots regenerated ([#292](https://github.com/SyntheticBrains/nematode/pull/292)).
- The remaining PPO and REINFORCE brains share `_policy.py` ([#275](https://github.com/SyntheticBrains/nematode/pull/275)); the Phase 5 campaign aggregators share `_common.py` ([#278](https://github.com/SyntheticBrains/nematode/pull/278)).
- CI: the test job runs in about five minutes instead of eighteen, split across five `pytest-split` shards with one BLAS thread per worker ([#288](https://github.com/SyntheticBrains/nematode/pull/288)); the workflows use the current Node 24 action majors ([#297](https://github.com/SyntheticBrains/nematode/pull/297), [#298](https://github.com/SyntheticBrains/nematode/pull/298)); the `phase6-tracking` change passes strict OpenSpec validation ([#287](https://github.com/SyntheticBrains/nematode/pull/287)).
- Documentation drift corrected across the roadmap, README, CONTRIBUTING and AGENTS.md, with the brain counts re-derived from the registry ([#280](https://github.com/SyntheticBrains/nematode/pull/280)); the Claude Code skills carry measured rather than guessed machine numbers ([#291](https://github.com/SyntheticBrains/nematode/pull/291)).

### Fixed

- `spikingreinforce`'s policy gradient was identically zero, so the brain never learned; it trains now, which changes its results relative to earlier logbooks ([#283](https://github.com/SyntheticBrains/nematode/pull/283)).
- The M3/M4 campaign aggregators reported generation-to-target one generation too high; the affected logbooks carry a correction note ([#284](https://github.com/SyntheticBrains/nematode/pull/284)).
- A partial F0 baseline override (or a mistyped `--campaign-root`) now fails closed instead of silently mixing baselines ([#285](https://github.com/SyntheticBrains/nematode/pull/285)).
- A flaky optimizer-identity test that compared `id()`s across object lifetimes, and two warnings the suite emitted on every run ([#293](https://github.com/SyntheticBrains/nematode/pull/293)).

### Removed

- The dead random-predator sprite; `PredatorType` rendering is now exhaustive ([#281](https://github.com/SyntheticBrains/nematode/pull/281)).
- `docs/OPTIMIZATION_METHODS.md`, which still used pre-rename class names; its guidance lives in [docs/architectures.md](docs/architectures.md) ([#292](https://github.com/SyntheticBrains/nematode/pull/292)).

### Security

- `pyarrow` 21.0.0 (CVE-2026-25087, reachable only through the `qpu` extra) is gone from the lock file: `qiskit-ibm-catalog` 0.16 → 0.19 and `qiskit-serverless` 0.32 → 0.35 no longer depend on it, and the catalog floor is raised to 0.19 ([#295](https://github.com/SyntheticBrains/nematode/pull/295)).
- `SECURITY.md` documents private vulnerability reporting and what is unsafe to load from untrusted sources ([#296](https://github.com/SyntheticBrains/nematode/pull/296)).

[0.5.0]: https://github.com/SyntheticBrains/nematode/compare/v0.4.0...v0.5.0
[unreleased]: https://github.com/SyntheticBrains/nematode/compare/v0.5.0...HEAD
