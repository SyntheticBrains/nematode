# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow [Semantic Versioning](https://semver.org/): while the project is at 0.x, a minor release may contain breaking changes, and they are listed first under each release.

Releases before 0.5.0 are documented on [GitHub Releases](https://github.com/SyntheticBrains/nematode/releases); this file starts at 0.5.0.

## [Unreleased]

### Added

- The matched-rule MLP arm (`mlpppo` with `learning_rule: three_factor`): the same reward-modulated three-factor rule, now substrate-generic over a `PlasticTopology` seam (aligned plastic weights, eligibility traces, and edge masks) instead of naming the connectome's attributes. An `MLPTopology` wraps the actor's Linear layers by reference — same object, no new state-dict keys, no re-run of construction — so the PPO path is byte-identical to a pre-refactor frozen reference. Feedforward eligibility is the same-step output-by-input product per layer; every Linear weight is plastic, biases and `log_std` frozen, critic dormant. The plasticity hyperparameters (and `freeze_updates`) move to a mixin both brain configs inherit, so "matched" means the same defaults from one definition. `freeze_updates` is now honoured on the MLP under PPO as well, so the flag means one thing on every brain.

- Two sanity-floor arms for the connectome plasticity panel: a **frozen-weights** floor (`freeze_updates` under the plasticity rule — no learning term, decay or clamp, so weights stay bit-identical to initialisation) and an **unmodulated-Hebbian** floor (`learning_rule: hebbian` — `dw = eta * E`, the reward stream observed but never applied). Both are configured under the plasticity rule rather than PPO so they share its anatomical motor readout: a floor decoding differently from the arm it bounds would confound decoding with learning. The unmodulated arm still computes and reports its prediction error, so the ablation is visible in telemetry rather than only in configuration. Together they separate "this arm learned something" from "this arm learned something from reward".

- A reward-modulated three-factor learning rule for the connectome brain (`learning_rule: three_factor`, requiring `enable_activity_traces`): `dw = eta * delta * E`, where `E` is the topology's eligibility trace and `delta` is a reward prediction error against a running baseline. It computes no gradients, owns no optimiser or value head, and updates once per environment step. Only the chemical synapses change — sensory gains and the motor readout stay frozen, the latter at the anatomical dorsal/ventral and forward/backward contrasts of the motor pools rather than a random draw, since a decoder that is never trained should respect what those pools mean. Plasticity is bounded by weight decay and a magnitude clamp, with prediction error, baseline, mean weight change and saturated fraction reported per update. Dale's law is deliberately not imposed: initial synapse signs here are arbitrary draws rather than neurotransmitter identity, so constraining them would preserve noise. Ships with one plastic wild-type config; the comparison arms follow.

- `scripts/run_campaign.py`: runs a campaign of simulations — one or more configs crossed with a set of seeds — concurrently under a bounded worker pool, with per-run logs, progress reporting, a status summary, `--dry-run`, and a non-zero exit if any run fails. Each run is a subprocess of `run_simulation.py` given byte-for-byte the command line it replaces, so campaigns change only *when* runs happen, never what they compute. Measured 7.29x at 16 workers on an 18-core machine ([Logbook 039](docs/experiments/logbooks/039-runtime-acceleration-audit.md)).

- `DeviceType.MPS` (`--device mps`): Apple's Metal GPU is now selectable for PyTorch brains, and therefore measurable. CPU remains the default and is faster for every brain currently in the repo — see the device benchmark below.

- `scripts/benchmarks/bench_device_backends.py` and `scripts/benchmarks/bench_campaign_parallelism.py`: reproducible measurements behind the device and worker-count guidance, including a large-network control row that distinguishes an unsuitable workload from a broken accelerator.

- The state-dependent continuous action std (roadmap D7): a `continuous_std_mode` field on every brain config (default `state_independent`, byte-identical to previous behaviour; the discrete combination fails validation), per-brain zero-init std heads on the five continuous brains, six `_sdstd` config variants, and a per-update clamped-log-std monitor in the tracked telemetry. Ships **dormant**: the pre-registered klinokinesis validation gate failed (Logbook 038), so the Phase 7 panel substrate froze in the default mode and the mechanism awaits a future pre-registered retry.

- The Phase 7 L4 trace substrate on the connectome brain: `enable_activity_traces` and `trace_decay` config fields (off by default; per-synapse eligibility traces accumulated during rollout forwards for the upcoming three-factor rules — training is bit-identical with traces on until a rule consumes them), and the new `quantumnematode.learning_rules` package, whose first citizen `ConnectomePPORule` is the connectome PPO update extracted byte-identically behind the `LearningRule` seam.

### Changed

- Roadmap decision D2 described the frozen-weights baseline as using "Cook-2019 synapse-count-derived initial weights". The implementation has never done that: the connectome supplies which edges exist, weights along them are drawn `N(0, 1/sqrt(chemical in-degree))`, and the EM synapse count never reaches the weight matrix. The wording is corrected rather than the initialisation — changing the latter would move a substrate that existing results are recorded against.

- Connectome configs now appear in the smoke-test set. None did previously, so the plasticity code path — per-step updates, no value head, an unused rollout buffer — had never been exercised end to end through the run entry point.

- The connectome's eligibility trace is now temporally causal: `E <- lambda*E + M(h_prev (x) h)`, taking the previous step's settled state as the pre-synaptic factor instead of a same-step symmetric outer product. The adjacency mask already separated non-reciprocal edges, but the symmetric form gave both directions of a reciprocal pair identical eligibility and encoded no temporal order, which a rule crediting synapses for causing activity cannot rest on. The first step of an episode now accrues no eligibility. Traces remain off by default and, while no rule consumes them, training remains bit-identical to traces-off.

- Device selection is validated before brain construction and fails with an actionable message. Previously `--device gpu` mapped unconditionally to CUDA and crashed on non-CUDA hosts with a raw `AssertionError: Torch not compiled with CUDA enabled`, despite being an advertised CLI choice. Selection is also checked against the brain family: a PyTorch-only accelerator is rejected for quantum brains, which pass the device to Qiskit — `AerSimulator(device="MPS")` is accepted *without raising*, so the bogus backend would otherwise have been recorded in experiment metadata as though it were real.

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
