# Quantum Nematode — AI Assistant Instructions

## Project Overview

Quantum Nematode is a closed-loop sensory-motor simulation platform: a simulated C. elegans forages, evades predators and navigates thermal/oxygen gradients while a pluggable brain architecture (MLP, recurrent, spiking, reservoir, quantum, hybrid, GA-evolved, or constrained to the real 302-neuron connectome) is trained by learning, evolution or fitting. The research question (re-framed 2026-10-23, roadmap v5.0, D33) is **how faithfully a simulated worm reproduces standard C. elegans behaviour, graded against real-worm data, and what the connectome contributes to that fidelity**. Architecture families are the comparison's controllers, compared on fidelity; quantum circuits are one family, not the organising principle (see docs/roadmap.md). Until v5.0 the question was which architecture best learns nematode behaviours; Phase 9 closed that line when the connectome's learning-speed advantage over its nulls turned out to be an artefact of the gap model (Logbook 103). Phase 10 grades simulated worms on standard on-lawn assays. Thermal, predator, multi-agent and oxygen scenarios are frozen (kept and tested, not extended; thermotaxis is the second tier).

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
- Export the connectome for Wormlight: `uv run python scripts/export_wormlight.py --out <path>` (refuses uncommitted changes to tracked files unless `--allow-dirty`)
- Real-worm behavioural-chemotaxis validation: set `sensing.capture_behaviour: true` in a foraging config (default `false` — a byte-identical no-op when off) to log a per-run behavioural trajectory to `exports/<session>/session/data/behaviour_capture.json`, then grade the klinokinesis + weathervane bias curves against the *C. elegans* literature with `uv run python scripts/analysis/behavioural_chemotaxis_validation.py --manifest <manifest.txt, one "<seed> <behaviour_capture.json>" pair per line> --tail-runs 100 --out <behavioural_curves.json> [--figure-dir <dir>] [--theta-sharp 0.45] [--wall-margin-mm 1.0 --arena-mm <world_size_mm>]` (the `--out` path writes the graded summary JSON; the wall margin drops transitions near an arena edge, where the clamped worm slides along the wall, and C.3 and D.1 require it; see [Logbook 035](docs/experiments/logbooks/035-realworm-chemotaxis-validation.md)). For **thermotaxis** validation set `sensing.capture_behaviour_modality: thermotaxis` (default `food`; it makes the captured drive the homeostatic thermal setpoint error `−|T−Tc|` so the same bias curves apply) and pass `--modality thermotaxis` to the harness (grades against the thermal `data/thermotaxis/` reference set; [Logbook 036](docs/experiments/logbooks/036-realworm-thermotaxis-validation.md)).
- **Campaigns: always pass the output controls.** `uv run python scripts/run_campaign.py --config <cfg> [--config ...] --seeds 1-96 --runs 3000 --output-dir campaigns/<name> -- --theme headless --track-experiment --no-detailed-export --no-file-log`. **The two `--no-*` flags are not optional at campaign scale**: without them each run writes ~0.7 GB *outside* the campaign directory (~440 MB of step-level exports plus ~256 MB of verbose log), so a 768-run campaign needs ~500 GB and will fill the volume — which is exactly what happened at run 337 of the L.1 campaign. With them a run writes ~17 MB and everything any analysis reads is still produced: the per-run summary CSVs, `weights/final.pt`, and the tracked-experiment record. The campaign directory itself is small, which is why the cost is invisible until it is not.
- One step's duration: a top-level `step_seconds` (unset by default; byte-identical when unset) sets the point worm's step in worm-seconds, e.g. `step_seconds: 1.0` for the 1-s worm (`configs/scenarios/foraging/*_lawns_*_s1.yml`). The config's per-step values stay written for its reference step, a full-speed step of its `max_step_mm` at 0.2 mm/s (5 s at 1 mm), and the loader converts every step-coupled constant to the configured step (`quantumnematode/utils/step_duration.py`), printing a `Step:` line and each conversion to stdout. Accepted only for the continuous point worm on lawns with `connectomeppo` or `mlpppo` and no within-lifetime plasticity; refused at load elsewhere.
- Locomotor primitives: `environment.continuous.locomotion: primitives` with `brain.config.action_space: primitives` (connectome or MLP-PPO, `signed_speed: false`, `allow_reversal: false`, and `step_seconds: 1.0` or shorter) moves the point worm only by runs, reversal bouts and omega/delta turns. Their kinematics are fixed from data (`data/locomotion/primitives.json`): published off-food values, and for a reversal that starts on a lawn the shorter, slower on-food values measured on Scheer's held-out cohorts (`continuous.reversal_kinematics: off_food` restores the off-food values everywhere, reproducing runs before 2026-10-24; Logbook 108). Each second the brain emits run speed, run curvature and three start hazards, and the environment draws the bouts (`quantumnematode/env/primitives.py`). Sensory modules `food_contact` (lawn density under the worm) and `locomotor_state` (the running primitive) are optional. `quantumnematode/env/hazard_controllers.py` holds the non-learning `ConstantHazards` (Poisson) and `PirouetteHazards` controllers. Configs: `configs/scenarios/foraging/*_s1_primitives.yml`.
- Assay protocols: `environment.assay` runs an episode as phases of food (`lawns` or `none`) and placement (`keep`, `lawn_centre`, `off_lawn`, `arena_centre`). The worm's satiety and the controller's state carry across phases. One phase is the assay phase, with an `analysis_from_s`. Assays need `satiety.starvation_ends_episode: false`, `lawns.start_on_lawn: false`, and `max_steps` × the reference step equal to the phases' length. `continuous.arena_shape: circle` (with `lawns.placement: centre`) gives round wells and plates. `scripts/campaigns/generate_assay_configs.py` writes Scheer's small lawn and uniform well, López-Cruz's and Calhoun's food removal, and Flavell's uniform lawn on the 1-s primitives worm (`configs/scenarios/foraging/*_assay_*.yml`). Captures record each step's `phase` and the protocol's timing.
- Batched evaluation: `quantumnematode/evaluation/` runs N frozen animals of one config in lockstep. `BatchedEvaluator(config, controller, seeds).run()` returns compact per-step arrays, and `to_behaviour_steps(i)` gives one animal's `BehaviourStep` series. Controllers are `BatchedMLP`, `BatchedConnectome`, `BatchedConstantHazards` and `BatchedPirouetteHazards`. Each animal has the per-step runner's random streams for its seed, so it behaves the same alone or in any batch. It covers the 1-s primitives worm on centred lawns or assays (klinotaxis + STAM + fold-change, Fick kernel, satiety never ending the episode) and refuses anything else. 48–98× faster per core than the per-step runner (Logbook 104). Parity tests in `tests/quantumnematode_tests/evaluation/` run on every PR. Re-time with `uv run python scripts/analysis/batched_cost_pilot.py --out <json>`.
- Neuromodulator layer from data: with `modulators: true`, `modulator_layer: data` on the connectome brain replaces the hand-placed serotonin/PDF field (the default, unchanged) with `data/neuromodulators/layer.json`, placed from Bentley 2016 and Ripoll-Sánchez 2023 (CC BY; provenance in its `PROVENANCE.md`): one pool per ligand (55), 121 couples with every cell tagged by source, 16 fixed monoamine signs (`receptor_signs.json`) and 105 fitted signed strengths starting at 0. Knobs: `modulator_timescales_s` (per ligand, default 60 s), `modulator_knockout_genes` (producer gene holds its ligand at zero, receptor gene removes its couples), `modulator_layer_sensitivity: ec50_500nM`, `modulator_layer_sources`, and the receptor-placement null (`modulator_layer_null: receptor_placement` + `modulator_layer_null_seed`). Rebuild with `scripts/analysis/extract_neuromodulator_layer.py fetch|synonyms|build`; audit the null with `scripts/analysis/neuromodulator_null_audit.py audit` (Logbook 106). Neuron classes: `quantumnematode.connectome.neuron_classes` (the 118).
- Benchmark reference data: `data/benchmark/` vendors derived real-worm sets (licences, checksums and units in its `PROVENANCE.md`): Scheer & Bargmann 2023 at 1 s with per-frame flags and every animal's genotype, date and cohort side (`scheer2023_split.json`, recording dates, a third held out of the bands); Calhoun 2014 and Roberts 2016 off-food search tracks; López-Cruz 2019's reorientations per 2-min bin. Read them with `quantumnematode.benchmark.references` (`load_scheer_tracks`, `load_search_tracks`, `load_lopezcruz`). Re-derive from the deposits (kept off-repo) with `scripts/analysis/extract_benchmark_references.py <source>`; the primitives' on-food check is `scripts/analysis/primitive_kinematics_check.py` (Logbook 105).
- Benchmark readouts: `quantumnematode.benchmark.readouts` reads 1-s positions alone, real or simulated: `reversal_onsets` (reversal-initiated reorientations; tuned on Roberts 2016 and registered for **off-food tracks only**, since it misses most on-lawn reversals), `reference_onsets`, `match_onsets`, `local_search_curve`, `fit_local_search` (alpha·exp(−gamma(t−t0))+beta) and `quintile_ratio`. The López-Cruz reference (gamma 0.114/min over 8–44 min after removal) is `data/benchmark/lopezcruz2019_reference.json`. Re-run the detector's calibration with `scripts/analysis/reorientation_detector_calibration.py --out <json>` and the reference with `scripts/analysis/local_search_reference.py --out <json>` (Logbook 107).
- Benchmark grading: `quantumnematode.benchmark.grading` reads real and simulated 1-s tracks through one window rule, tallies each animal, and grades 11 wild-type statistics. Six are on Scheer's small lawn: roaming fraction, censoring-aware roaming and dwelling bout means, exits per on-lawn hour, pre-exit roaming enrichment and time on the lawn. Three are on Scheer's uniform well (protocol `uniform_lawn_scheer`): roaming fraction and both bout means. Two are on López-Cruz's local search: γ and the early/late ratio. Mutants (tph-1, pdfr-1, tax-4) are graded on their difference from their own wild type. Bands are between-date (`data/benchmark/bands.json`, built from band cohorts only by `scripts/analysis/benchmark_bands.py`, with the classifier from `scripts/analysis/benchmark_classifier.py`). A wild-type statistic passes at |z| ≤ 2.84 (Bonferroni over 11), and a mutant's at |z| ≤ 2.24 (Bonferroni over its two). D is the weighted RMS of the capped z's; D_fit leaves local search out. Grade with `uv run python scripts/analysis/benchmark_grade.py batched --controller constant|pirouette --seeds <a-b> --out <json>` or `... captures --manifest <"<family> <behaviour_capture.json>" per line> --out <json>`. Each family is graded at its own size: on the vendored bands at 64 or 200, and otherwise on bands built on demand (each band on its own generator). A mutant's direction is resolved by a paired t-test across recording dates, and local search must be present (early/late above 1 at its own lower bound) (Logbook 109). The benchmark's controls (`scripts/analysis/benchmark_controls.py pilot|read`; `read` refuses until its launch record is committed) found it VALID: held-out real worms pass and six negatives fail (Logbook 110). Negative controllers: `BatchedStateHazards` (open-loop two-state worm) and `BatchedCurvatureWalk`. C.1's mechanistic baseline is `BatchedMechanistic` (20 parameters, `MECHANISTIC_PARAMETERS`). Its NPE recovery is `scripts/analysis/npe_recovery.py pilot|simulate|train|diagnose` (needs the `analysis` extra's `sbi`). F.0a failed its informative gate, whose bar sat below the simulator's own noise (Logbook 111).
- L4 plasticity panel: run the recipe pilot with `uv run python scripts/campaigns/l4_panel_pilot.py --out campaigns/l4-pilot` (grid configs derived beside the results; add `--only <arm> --rate <r> --runs 6000` to extend one arm as a fresh run), summarise it with `uv run python scripts/analysis/l4_panel.py --pilot --campaign-dir campaigns/l4-pilot --out docs/experiments/logbooks/supporting/040-l4-panel/pilot.json` (prints the selected rate and the budget rule's output), then analyse the panel with `uv run python scripts/analysis/l4_panel.py --campaign-dir <campaign-dir> --out panel.json --csv per-seed.csv --curves curves.csv` (confirmatory mode accepts seeds 1–8 only; the four-test family, the band test and the verdict map are fixed in the script).

## Running Experiments

Read [docs/research/phase-protocol.md](docs/research/phase-protocol.md) before designing any panel; its principles bind every
registration. The non-negotiables, each learned from a lost run:

- **Workflow**: plan, then an OpenSpec change, then `/nematode-review-spec`, then implement, archive and PR. Never skip the
  review for a change that launches a campaign over about two hours.
- **Register before running**: a `launch.md` under `docs/experiments/logbooks/supporting/<NNN>-<slug>/` fixing the arms,
  metric, registered minimum (from committed data, on the same cell), sizing, verdict map and gates, committed before any
  scored run. A registered verdict is never re-read under a different rule after the data are seen; anything else is
  reported beside it as description.
- **Gate preflight before launch**: `uv run python scripts/campaigns/gate_preflight.py --panel <analysis module> [--half ppo] --logs <dir of runs at the registered point>` evaluates every level's floor and saturation gates on committed runs,
  or on a pilot, and exits nonzero if any level would be unreadable, sits within 5 points of the bar, or has no evidence.
  A.6t ran 16 hours into an unreadable panel that one minute of this would have caught
  ([Logbook 077](docs/experiments/logbooks/077-thermal-null-strength.md)).
- **Oracle before launch**: a positive control's success bar must be met by its known answer before it registers. Score
  the generating parameters, re-simulated with fresh noise, against the bar. A bar on a statistic compared between
  simulations is set against the simulator's own same-parameter noise. F.0a spent five hours on a gate its oracle could
  not pass ([Logbook 111](docs/experiments/logbooks/111-npe-recovery.md); protocol principles 4 and 6).
- **Pilots** run on seeds disjoint from the registered band, configured exactly as the campaign, and long enough to
  reach the machine's sustained rate; cost is estimated from them. A smoke run that only checks configs execute is not a
  pilot.
- **During a campaign** never switch branches: the runner reads configs and package code from the working tree per run.
- **Retention**: commit the per-seed CSV, analysis JSON and launch record; archive raw logs off-repo.

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
than 10% of the collected suite has no recorded duration.

**Refresh it from CI, not from a laptop.** The split has to balance CI's runners,
and tests do not keep their relative speeds between a Mac and a runner. On
2026-10-24 a file timed locally split into shards of 499s, 1026s, 1046s, 1082s
and 1194s of test time on CI, balanced to 1.00x by its own numbers. So every CI
shard times its own tests and uploads them as the artifact
`test-durations-shard-<n>` (14 days' retention). The union of one run's five
files is the suite timed on CI. Take a run whose five shards all passed:

```bash
dir=$(mktemp -d)
gh run download <run-id> --pattern 'test-durations-shard-*' --dir "$dir"
uv run python scripts/merge_test_durations.py "$dir"
```

Download into a new, empty directory every time. `gh` puts each artifact in its
own `test-durations-shard-<n>/` folder there, which is what the script reads. A
reused directory can still hold an earlier run's folders. The script refuses
fewer than five files, and any test that appears in two of them. Refresh this way when shard times drift apart, and after a PR that adds
heavy tests, from that PR's own run. Then run `pre-commit`.

**Locally**, for example to cover a new module before its first CI run, regenerate with:

```bash
uv run pytest -m "not nightly" --store-durations --clean-durations --durations-path .test_durations
```

These are laptop timings, so prefer the CI refresh as soon as a run exists.
`--clean-durations` drops entries for tests that no longer exist; without it the
file accumulates them. This takes ~2.5 minutes, since `-n logical` comes from
`addopts`. `pytest-split` writes the file without a trailing newline, so run
`pre-commit` afterwards — the `fix end of files` hook adds it, and CI's Code
Quality job fails without it.

Locally the suite is unsharded; `-n logical` in `addopts` uses every logical core.

## Pull Requests

PR titles MUST use [Conventional Commits](https://www.conventionalcommits.org/) prefixes. Common types in this repo: `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:`. Use `!` after the type for breaking changes (e.g. `feat!: remove legacy preprocessing mode`). Examples from project history: `feat: Add aerotaxis (oxygen sensing) system`, `fix: multi-agent sensing - use agent's own position in BrainParams`, `docs: Klinotaxis Era multi-agent pheromone evaluation (Logbook 011)`.

Commit messages do not require this prefix — only PR titles do.

User-facing changes add a line to `CHANGELOG.md` under *Unreleased* (Keep a Changelog format; breaking changes first). Releases follow the checklist in CONTRIBUTING.md § Releasing.

<!-- markdownlint-disable MD025 -->
