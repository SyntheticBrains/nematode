# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow [Semantic Versioning](https://semver.org/): while the project is at 0.x, a minor release may contain breaking changes, and they are listed first under each release.

Releases before 0.5.0 are documented on [GitHub Releases](https://github.com/SyntheticBrains/nematode/releases); this file starts at 0.5.0.

## [Unreleased]

### Added

- **Weekly literature watch** — a scheduled workflow that sweeps for new work bearing on this project and opens a digest issue. Three sources in descending order of precision: new work citing a curated seed set (OpenAlex), then the arXiv and bioRxiv windows. A cheap model scores every candidate against a hand-maintained brief and a capable model writes the entries, so a few hundred abstracts a week reduce to a handful worth reading. See [docs/research/literature-watch/](docs/research/literature-watch/README.md); it needs an `ANTHROPIC_API_KEY` repository secret to run.
- **Campaign progress reader** — `scripts/campaigns/campaign_progress.py --campaign <dir> [--total N] [--watch]` reports a running campaign's finished, in-flight and pending runs with an ETA, reading the campaign directory rather than the runner's redirected stdout, so it works from any shell.

## [0.6.0] - 2026-09-19

The Phase 7 release. Phase 7 asked whether the wild-type *C. elegans* connectome becomes load-bearing under a biologically plausible learning rule, and **closed as a SPLIT**: no rule in that family writes this connectome to any benefit, so the pre-registered question has no interpretable answer here. What the phase established instead is that the wiring is **learning-speed-relevant under gradient descent**, and **legible to a small local readout only at a particular readout width and learning rate**. Thirty logbooks, 040 through 069, carry the evidence.

*Changelog entries in this release are deliberately short.* Earlier drafts restated each logbook's method and figures, duplicating the record they link to; the logbooks are the record, and each entry now says what changed and where to read it.

### Breaking changes

- **Resuming an evolution run now requires `--allow-unsafe-resume`** ([#392](https://github.com/SyntheticBrains/nematode/pull/392), closing the security half of [#16](https://github.com/SyntheticBrains/nematode/issues/16)). Checkpoints are Python pickles, so `--resume` on a file of unknown provenance is arbitrary code execution; the flag defaults off and the refusal fires before any file is read. The format is unchanged and #16 stays open for it — the payload carries CMA-ES and numpy generator state rather than tensors, so `weights_only=True` does not apply.

### Added

**Phase 7 results.** Each logbook carries its own method, registered readings, sensitivity and "may not be cited as" list.

- **[Logbook 069](docs/experiments/logbooks/069-phase7-synthesis.md) — the Phase 7 synthesis: closes as SPLIT.** The flagship MUST is unmet and was unmeetable in the closing scope. Every exit criterion carries one of five statuses with none unmarked, and the roadmap, README and Success Levels are flipped to terminal state.
- **[Logbook 068](docs/experiments/logbooks/068-l1b-rate-calibration.md) — `width_favours_the_shuffle_at_this_rate`.** L.1's interaction does not survive the learning rate: −0.0657 at 0.0001 against +0.2818 at 0.001, a three-way of +0.3475 on 81 of 96 seeds. At the lower rate capacity dominates (+0.6178 width main effect on 96/96) where L.1 detected none.
- **[Logbook 067](docs/experiments/logbooks/067-l4-feature-ablations.md) — both ablations read `carries_the_effect`**, L.4's qualified *carries or unlearnable*. Removing gap junctions helped **both** wirings with the null catching up; grounding the synapse signs cut learnability on both at a rate where the null was already ahead.
- **[Logbook 066](docs/experiments/logbooks/066-l4-readout-width.md) — `pooling_hid_structure`.** Widening the readout from four pooled motor classes to one weight per motor neuron gives an interaction of **+0.2818** on 96 paired seeds, with the wiring effect's sign flipping between widths and no width main effect detected.
- **[Logbook 065](docs/experiments/logbooks/065-wiring-fresh-rewiring.md) — `specific_wiring_efficiency` on both cells.** Block V replicates on rewirings fresh to both prior panels: +55.3% and +40.1%, closing a shared-nulls defect where V.3's rewirings were a subset of V.1's.
- **[Logbook 064](docs/experiments/logbooks/064-l4-frozen-features.md) — `wiring_is_inert_as_features`.** With only the motor readout learning, the wild type shows no advantage over its degree-preserving null at the registered bar, extending the degree-statistics verdict to a third learning regime.
- **[Logbook 063](docs/experiments/logbooks/063-l4-eprop.md) — `learns_without_the_substrate`.** e-prop reaches competence on the connectome (17.570 foods of 20, 52.61% full clear) on the arm whose chemical matrix is **frozen**; every arm that writes the wiring does worse by 3.9 to 15.9 foods. This is the result that closed the rule programme.
- **[Logbook 062](docs/experiments/logbooks/062-l4-frozen-readout.md) — `readout_helps_but_not_enough`.** Substituting the frozen motor readout more than doubles what the rule reaches, 3.751 to 9.639 foods, without any arm becoming competent; the readout's scale does the work, not its direction.
- **[Logbook 061](docs/experiments/logbooks/061-l4-reduced-perturbation.md) — `not_reducible`.** No perturbation set from 1208 draws down to 39 makes the rule learn the connectome, and the per-unit perturbation is now restrictable to a declared set.
- **[Logbook 060](docs/experiments/logbooks/060-l4-perturbation-scale.md) — `mixed`, and the mixture is the finding.** The rule solves a multi-step foraging cell at 8 perturbed units and collapses at 128, matching node perturbation's ~1/N scaling (Werfel, Xie & Seung 2005) against the connectome's 1208 draws per decision.
- **[Logbook 059](docs/experiments/logbooks/059-7a-shipment.md) — SPLIT-shipment.** 7a ships a systematic negative with a diagnosed cause plus block V's positive, with GO unreachable on its own clause and STOP overstating.
- **[Logbook 058](docs/experiments/logbooks/058-wiring-premise-difficulty.md) — `specific_wiring_efficiency`.** The wiring advantage survives removing the thermosensory pathway: +23.5% off time-to-competence on a hard food-only cell, so difficulty is sufficient.
- **[Logbook 057](docs/experiments/logbooks/057-wiring-premise-contrast.md) — `specific_wiring_efficiency`, the project's first positive wiring result.** Under PPO the wild-type wiring reaches competence ~35% sooner than its degree-preserving null on a cell matched to a behaviour the animal performs. A [companion probe](docs/experiments/logbooks/supporting/057-wiring-premise-contrast/probe-v2.md) scored all 64 rewirings on four graph properties fixed before looking and found none predicting learning time.
- **[Logbook 056](docs/experiments/logbooks/056-l4-ladder-reread.md) — no committed verdict changed.** Of 32 registered contrasts across eight logbooks, ten are instrument findings and the rest are not, so "the negatives are artefacts of a broken instrument" covers less than it appears to.
- **[Logbook 048](docs/experiments/logbooks/048-l4-rule-positive-control.md) — the rule fails its own positive control, control valid.** The three-factor rule does not learn a task whose analytic reference closes 99.9% of the gap, with gradient alignment at +0.009. It ran twelfth in the sequence, after seven registered panels.
- **The instrument block** that followed, recorded under [`logbooks/supporting/`](docs/experiments/logbooks/supporting/): a node-perturbation eligibility that passes the control, a σ schedule, endpoints re-evaluated with the perturbation off (bimodal), a contrast family matched to that bimodality with every committed table re-read under it, the eligibility horizon found limiting on the control and **not** transferring to a multi-step task, and a delayed control on which the three settings pinned since A.3 were finally examined.
- **[Logbook 047](docs/experiments/logbooks/047-l4-structured-instruction.md) — `no_routing_effect`**, [046](docs/experiments/logbooks/046-l4-decorrelation.md) — `no_recovery`, [045](docs/experiments/logbooks/045-l4-consolidation.md) — no mechanism passes the clone assay, [044](docs/experiments/logbooks/044-l4-atlas-signs.md) — `degree_statistics`, [043](docs/experiments/logbooks/043-l4-warm-start.md) — `sanity_floor_fail` + `rule_destroys_clone`, [042](docs/experiments/logbooks/042-l4-panel3.md) and [041](docs/experiments/logbooks/041-l4-panel2.md) — `inconclusive`, [040](docs/experiments/logbooks/040-l4-panel.md) — `sanity_floor_fail`, the D10 2×2 panel that opened the phase.

**Substrate and rule mechanisms**, each default-off or byte-identical when unused.

- A reward-modulated **three-factor learning rule** for the connectome brain (`learning_rule: three_factor`): `dw = eta * delta * E` over the topology's eligibility trace, with the **matched-rule MLP arm** (`mlpppo` with the same rule over a `PlasticTopology` seam) and the **plastic degree-preserving rewired-null arm** as the 2×2's other cells.
- Two **sanity-floor arms**: frozen weights (`freeze_updates`) and an unmodulated Hebbian rule, both configured under the plasticity rule so they share its readout.
- **Consolidation** (`plasticity_consolidation`), **decorrelating terms** (`plasticity_decorrelation`) and a **routed third factor** (`third_factor: pathway`) — brakes and instruction mechanisms for a rule shown to drift at near-constant speed through good policies.
- **Neurotransmitter identities** on the substrate (`synapse_signs: atlas`), grounding 3,176 of 3,709 chemical synapses in the Wang et al. 2024 atlas — the fidelity ladder's first rung.
- **`readout_width`** (`pooled` | `per_neuron`), the per-neuron readout initialised by expanding the pooled draw so the RNG stream is untouched and both widths compute the same policy at initialisation.
- **`weight_init`** (`degree_scaled` | `count_scaled`), **`plastic_layers`** on the MLP brain, a **centred compressed modulator**, substrate-invariant scaling switches, the **L4 trace substrate** (`enable_activity_traces`, `trace_decay`), and the **state-dependent continuous action std** (`continuous_std_mode`, roadmap D7).
- **Warm-start tooling** for the imitation arm: `ConnectomePPOBrain` implements `WeightPersistence` (closes [#308](https://github.com/SyntheticBrains/nematode/issues/308)), and `weights_path` accepts `{seed}` so one committed config addresses one clone file per seed.

**Tooling and infrastructure.**

- **`scripts/run_campaign.py`**: configs crossed with seeds under a bounded worker pool, with per-run logs, progress reporting, a status summary and `--dry-run`.
- **Two campaign output controls** on `scripts/run_simulation.py`: `--no-detailed-export` and `--no-file-log`. Without them a run writes ~0.7 GB outside the campaign directory and a 768-run campaign needs ~500 GB, which filled the volume mid-campaign; with them, ~17 MB, and everything an analysis reads is still written.
- **`DeviceType.MPS`** (`--device mps`) so Apple's Metal GPU is selectable and measurable, with `bench_device_backends.py` and `bench_campaign_parallelism.py` behind the device and worker-count guidance.
- **CI test shards rebalanced**, with a guard so the balance cannot rot silently: 1,376 of 5,520 tests had no recorded duration and `pytest-split` was budgeting each at the mean of the rest.

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
[0.6.0]: https://github.com/SyntheticBrains/nematode/compare/v0.5.0...v0.6.0
[unreleased]: https://github.com/SyntheticBrains/nematode/compare/v0.6.0...HEAD
