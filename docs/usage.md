# Usage Guide

How to run the platform: simulations, outputs, experiment tracking, evolution and inheritance, the analysis and campaign scripts, real-worm validation, quantum hardware, and the GPU container. Installation is covered in the [README](../README.md#quick-start) and [CONTRIBUTING.md](../CONTRIBUTING.md#development-setup).

## Running a simulation

```bash
uv run ./scripts/run_simulation.py --config <scenario.yml> [options]
```

A *session* is `--runs` episodes of one scenario with one brain that learns across them. The scenario YAML carries everything — brain, hyperparameters, environment, sensing mode, reward — so the CLI only controls how the session is executed and rendered (`uv run ./scripts/run_simulation.py --help` is authoritative):

| Option | Effect |
|---|---|
| `--config PATH` | The scenario to run ([configs/README.md](../configs/README.md)) |
| `--runs N` | Number of episodes (default 1) |
| `--seed N` | Master seed for reproducibility (auto-generated and printed if omitted) |
| `--theme NAME` | `pixel` (default), `pixel_continuous`, `ascii`, `emoji`, `unicode`, `colored_ascii`, `rich`, `emoji_rich`, `headless` — see [visualization.md](visualization.md) |
| `--device cpu\|gpu\|qpu` | Backend for the quantum brains (default `cpu`; `gpu` needs the `gpu` extra; `qpu` needs the `qpu` extra and `.env`) |
| `--optimize` | On `qpu`, enable Q-CTRL Fire Opal error suppression |
| `--track-experiment` | Save reproducibility metadata to `experiments/<id>/<id>.json` (see [Experiment tracking](#experiment-tracking)) |
| `--track-per-run` | Write tracked brain data as separate plots per run, in per-run subfolders |
| `--validate-chemotaxis` | Print chemotaxis-index validation against the *C. elegans* literature values in `data/chemotaxis/` |
| `--save-weights PATH` / `--load-weights PATH` | Persist trained weights after the session, or warm-start from a saved file |
| `--show-last-frame-only` | In terminal themes, print only each run's final frame |
| `--manyworlds` | Overlay the top two candidate actions at each step (single run only) |
| `--log-level LEVEL` | `DEBUG`, `INFO` (default), `WARNING`, `ERROR`, `CRITICAL`, or `NONE` |

Examples:

```bash
# Headless training session, fixed seed
uv run ./scripts/run_simulation.py \
  --config configs/scenarios/foraging/mlpppo_small_continuous2d_fick_adaptive_klinotaxis.yml \
  --theme headless --runs 500 --seed 1

# The Phase 6 three-behaviour cell with the connectome brain, tracked for later comparison
uv run ./scripts/run_simulation.py \
  --config configs/scenarios/foraging_predator_thermal/connectomeppo_small_continuous2d_combined_klinotaxis.yml \
  --theme headless --runs 6000 --seed 1 --track-experiment

# Five GRU worms with food-marking and aggregation pheromones and social feeding on a single food cluster, rendered (multi-agent supports pixel and headless only)
uv run ./scripts/run_simulation.py \
  --config configs/scenarios/multi_agent_foraging/lstmppo_large_5agents_single_cluster_pheromone_klinotaxis.yml \
  --theme pixel --runs 10

# Save weights from one session and warm-start another
uv run ./scripts/run_simulation.py --config configs/scenarios/foraging/mlpppo_small_oracle.yml --theme headless --runs 100 --save-weights weights/mlpppo_foraging.pt
uv run ./scripts/run_simulation.py --config configs/scenarios/pursuit/mlpppo_small_oracle.yml --theme headless --runs 100 --load-weights weights/mlpppo_foraging.pt
```

Stopping a session with `Ctrl-C` is safe: the interrupt handler offers a menu to write the summary, plots and tracking data for the runs completed so far before exiting.

## Outputs

Every session gets an ID (`YYYYMMDD_HHMMSS_<hash>`) and writes:

```text
exports/<session-id>/
  session/data/         per-run and session-level CSVs: simulation_results, run_metrics, performance_metrics,
                        foraging_*, paths, distance_efficiencies, tracking_{actions,rewards,probabilities,metadata}
  session/plots/        success rate over time, foods-vs-reward and the other session plots
  weights/final.pt      the trained brain at the end of the session (PyTorch brains)
logs/simulation_<session-id>.log
```

A summary is printed when the session ends (success rate, failure breakdown by cause, average foods, steps, reward, distance efficiency, survival and comfort scores). `exports/`, `logs/`, `experiments/` and `evolution_results/` are git-ignored; curated outputs that a logbook depends on are copied into `artifacts/` (Git LFS) — see [artifacts/README.md](../artifacts/README.md).

## Experiment tracking

`--track-experiment` writes `experiments/<experiment-id>/<experiment-id>.json` with the config file and its hash, the git commit, branch and dirty state, system and dependency versions, the brain and environment parameters, the full results and performance metrics, and the export paths. The convergence detector (`quantumnematode.experiment.convergence`) derives the level-agnostic `post_convergence_success_rate` that the architecture-comparison protocol ranks on.

Query tracked experiments with `scripts/experiment_query.py`:

```bash
uv run scripts/experiment_query.py list                               # recent experiments
uv run scripts/experiment_query.py list --brain-type mlpppo --since 2026-06-01 --limit 20
uv run scripts/experiment_query.py show <experiment-id>               # full detail (add --json to export)
uv run scripts/experiment_query.py compare <exp-id-1> <exp-id-2>      # side-by-side
```

Results that are worth keeping go into a numbered logbook — the workflow is in [experiments/README.md](experiments/README.md).

## Evolution and inheritance

`scripts/run_evolution.py` runs an evolutionary outer loop over a config from `configs/evolution/`:

```bash
uv run python scripts/run_evolution.py --config configs/evolution/<config>.yml [options]
```

| Option | Effect |
|---|---|
| `--algorithm cmaes\|ga\|tpe` | Override the config's optimiser: CMA-ES, genetic algorithm, or TPE (Optuna) |
| `--inheritance none\|lamarckian\|baldwin\|transgenerational` | Override the inheritance strategy between generations |
| `--generations N`, `--population N`, `--episodes N` | Override the budget (generations, population size, episodes per fitness evaluation) |
| `--fitness success_rate\|progress\|learned_performance` | Override the fitness function |
| `--parallel N` | Parallel fitness-evaluation workers |
| `--seed N` | Master seed (per-evaluation seeds are derived from it) |
| `--sigma X` | CMA-ES initial step size |
| `--early-stop-on-saturation N` | Stop if the best fitness has not improved for N consecutive generations (default: run the full budget) |
| `--resume PATH` | Resume from a `checkpoint.pkl` |
| `--output-dir DIR` | Where to write the session (default `evolution_results/`) |

Three kinds of evolution config exist:

- **Weight evolution** — the genome *is* the brain's weights (`feedforwardga_*`, `mlpppo_foraging_small`, `lstmppo_foraging_small_klinotaxis`). The weight-genome encoders cover `mlpppo`, `lstmppo` and `feedforwardga`; the GA arm is the gradient-free floor in the architecture rankings ([Logbook 029](experiments/logbooks/029-continuous-architecture-ranking.md)). CMA-ES's earlier win over parameter-shift gradients for the quantum circuit ([Logbook 002](experiments/logbooks/002-evolutionary-parameter-search.md)) predates this framework, which has no `qvarcircuit` weight encoder yet.
- **Hyperparameter evolution** — the genome patches a brain's config and each evaluation trains a fresh brain for *K* episodes under `learned_performance` fitness (`hyperparam_*`). TPE is the preferred optimiser ([Logbook 012](experiments/logbooks/012-hyperparam-evolution-mlpppo-pilot.md)).
- **Inheritance studies** — `lamarckian_*` (warm-start each generation from the prior elite; the Phase 5 positive result, [Logbook 013](experiments/logbooks/013-lamarckian-inheritance-pilot.md)), `baldwin_*` ([015](experiments/logbooks/015-baldwin-iterative-evaluation.md)), `transgenerational_*` and `tei_prior_*` ([018](experiments/logbooks/018-transgenerational-memory.md)–[020](experiments/logbooks/020-tei-prior-on-lamarckian.md)); the latter two are implemented and closed with STOP verdicts.

Each session writes to `evolution_results/<session-id>/`: the per-generation `history.csv`, the best genome found, and periodic checkpoints to resume from. Configs suffixed `_smoke` or `_pilot` are reduced budgets for framework testing.

```bash
# Evolve feed-forward weights with the GA
uv run python scripts/run_evolution.py --config configs/evolution/feedforwardga_foraging_small.yml --algorithm ga --generations 30 --parallel 4

# Hyperparameter evolution with TPE, Lamarckian warm-starts between generations
uv run python scripts/run_evolution.py --config configs/evolution/hyperparam_mlpppo_pilot.yml --algorithm tpe --inheritance lamarckian --parallel 4

# Resume
uv run python scripts/run_evolution.py --config configs/evolution/feedforwardga_foraging_small.yml --resume evolution_results/<session-id>/checkpoint.pkl
```

**Predator–prey co-evolution** runs through `scripts/run_coevolution.py --config configs/evolution/coevolution_*.yml [--seed N] [--output-dir DIR] [--resume PATH]` (`CoevolutionLoop`; warm-start prey bundles under `configs/evolution/coevolution_warmstart_prey/`). The Red Queen question it was built for closed with a STOP verdict ([Logbook 017](experiments/logbooks/017-coevolution-arms-race.md)); the lag-matrix and cell-grid instruments remain available.

## Analysis and campaign scripts

`scripts/analysis/` holds the statistical tooling behind the Phase 6 logbooks; each script documents its inputs in its module docstring.

| Script | Purpose |
|---|---|
| `weight_search_architecture_ranking.py` | Grid-substrate cross-architecture ranking (T4, [Logbook 025](experiments/logbooks/025-weight-search-architecture-ranking.md)): paired-seed Wilcoxon, bootstrap CIs, BH-FDR |
| `t7_continuous_ranking.py`, `t7_ga_champion_eval.py` | Continuous-substrate ranking on the plateau-tail full-clear metric, and the frozen evaluation of the GA champion (T7, [Logbook 029](experiments/logbooks/029-continuous-architecture-ranking.md)) |
| `bit_memory_separation.py`, `associative_memory_separation.py` | Memory-arm separation statistics for the two working-memory probes ([030](experiments/logbooks/030-bit-memory-positive-control.md), [033](experiments/logbooks/033-associative-memory-probe.md)) |
| `minimal_rnn_reactive_ab.py` | minGRU / minLSTM vs LSTM on the reactive cell ([031](experiments/logbooks/031-minimal-rnn-candidates.md)) |
| `connectome_structure_controls.py`, `connectome_structure_efficiency.py` | Wild-type vs rewired-null connectome comparison on the plateau and learning-efficiency axes ([034](experiments/logbooks/034-connectome-structure-controls.md)) |
| `behavioural_chemotaxis_validation.py` | Real-worm bias-curve grading (see [below](#real-worm-behavioural-validation)) |

Other scripts:

- `scripts/campaigns/` — the Phase 5 campaign drivers (`phase5_*.sh`) and their per-milestone aggregators (`aggregate_m*_pilot.py`); `_common.py` holds the shared helpers.
- `scripts/run_plasticity_test.py` / `compare_plasticity_results.py` — the sequential multi-objective (A → B → C → A′) plasticity protocol and its cross-architecture comparison ([Logbook 008](experiments/logbooks/008-quantum-brain-evaluation.md), QA-7).
- `scripts/qef_mi_analysis.py`, `scripts/qrh_mi_analysis.py` — mutual-information decision gates for the QEF and QRH quantum feature extractors.
- `scripts/export_screenshot.py` — render staged frames of the grid, multi-agent and continuous renderers to PNG (used for the docs images); `scripts/extract_runs.py` — pull run/step counts out of a simulation log.
- `scripts/manage_jobs.py` — check the status of IBM Quantum or Q-CTRL Qiskit Function jobs by ID.
- `scripts/benchmarks/bench_evolution_smoke.py` — wall-clock benchmark of the evolution fitness-evaluation path, for PRs that touch it.
- `scripts/run_health_scaling_study.sh` — the Logbook 005 health-system scaling sweep.

## Real-worm behavioural validation

The validation grades the worm's *own* behaviour against published *C. elegans* navigation strategies, at the behaviour level (no neuron-identity mapping): klinokinesis (turn rate vs dC/dt; Pierce-Shimomura et al. 1999) and weathervane steering (curving rate vs bearing; Iino & Yoshida 2009), each with bootstrap-CI grading (REPRODUCED / PARTIAL / ABSENT). Reference signatures live in `data/chemotaxis/` and `data/thermotaxis/`.

1. Set `sensing.capture_behaviour: true` in a foraging config (default `false`; a byte-identical no-op when off). Each run then logs a behavioural trajectory to `exports/<session>/session/data/behaviour_capture.json`.

2. List the captures in a manifest file — one `<seed> <path>` pair per line, paths resolved relative to the repository root — and grade them:

   ```bash
   printf '1 exports/<session-seed-1>/session/data/behaviour_capture.json\n2 exports/<session-seed-2>/session/data/behaviour_capture.json\n' > _manifest.txt
   uv run python scripts/analysis/behavioural_chemotaxis_validation.py \
     --manifest _manifest.txt --tail-runs 100 --out behavioural_curves.json \
     [--figure-dir figures/] [--theta-sharp 0.45]
   ```

3. For **thermotaxis**, set `sensing.capture_behaviour_modality: thermotaxis` (the captured drive becomes the thermal setpoint error `−|T−Tc|`, so the same bias curves apply) and pass `--modality thermotaxis` to the harness.

Method and results: [Logbook 035](experiments/logbooks/035-realworm-chemotaxis-validation.md) (chemotaxis) and [036](experiments/logbooks/036-realworm-thermotaxis-validation.md) (thermotaxis).

## Quantum hardware

The quantum brains run on the Qiskit Aer simulator by default. To run on IBM Quantum hardware:

1. `uv sync --extra qpu …` (it does not conflict with `cpu`).

2. Copy `.env.template` to `.env` and fill in `IBM_QUANTUM_API_KEY`, `IBM_QUANTUM_BACKEND`, `IBM_QUANTUM_CHANNEL` and `IBM_QUANTUM_CRN`; add `QCTRL_API_KEY` to use Fire Opal.

3. Run with `--device qpu`, and `--optimize` to route through Q-CTRL Fire Opal error suppression:

   ```bash
   uv run ./scripts/run_simulation.py --config configs/scenarios/foraging/qvarcircuit_small_oracle.yml --theme emoji --runs 1 --device qpu --optimize
   ```

4. Check on submitted jobs with `uv run scripts/manage_jobs.py <job-id>`.

Hardware runs are slow and metered; the Phase 2 campaign used them sparingly for validation after simulator results ([Logbook 008](experiments/logbooks/008-quantum-brain-evaluation.md)).

## GPU and Docker

- **GPU simulation**: `uv sync --extra gpu --extra torch --extra pixel` installs `qiskit-aer-gpu-cu11` (CUDA 11; driver ≥ 450) instead of the CPU simulator, then `--device gpu`. The `gpu` and `cpu` extras cannot be installed together.
- **Container**: `docker compose up --build` builds a `python:3.13-slim` image with the `gpu` and `torch` extras, the source, scripts, configs and data, and starts it with all NVIDIA devices attached (requires Docker with the NVIDIA Container Toolkit on an x86_64 host — the CUDA-11 Aer wheel is published for x86_64 only, so on Apple Silicon build it with `docker build --platform linux/amd64 .` for CPU-only checks). Run `git lfs pull` first so the connectome spreadsheets under `data/` are copied in as real files rather than LFS pointers. Run sessions inside it with `docker compose exec quantum-nematode uv run ./scripts/run_simulation.py --config configs/scenarios/… --theme headless --device gpu`; outputs land in the container's `/app/exports` unless you mount a volume for it. The image is not built in CI, so treat it as best-effort.

## Multi-agent sessions

Configs under `configs/scenarios/multi_agent_*/` run 2–10 worms in one arena with food competition, pheromone communication (food-marking, alarm, aggregation), social feeding and collective-behaviour metrics; `mixed_brains_*` configs give each worm a different brain. Multi-agent sessions support the `pixel` and `headless` themes only, and export per-agent metrics alongside the session CSVs. Findings: [Logbook 011](experiments/logbooks/011-multi-agent-evaluation.md).

## Environment reference

The mechanics that scenario configs tune. Defaults are the grid-substrate values; the continuous substrate overrides some of them, as noted. Specs for each capability live under [`openspec/specs/`](../openspec/specs/).

**Substrates.** The grid (`DynamicForagingEnvironment`) is a discrete arena — `small` 20×20, `medium` 50×50, `large` 100×100 — where the action is one of `{forward, left, right, stay}`. The continuous-2D substrate (`Continuous2DEnvironment`, selected by the `_continuous2d` config variants) is a square `world_size_mm` arena — 20 mm in the single-behaviour canary cells, 60 mm in the Phase 6 ranking cells (the C2 pursuit and integrated C3 cells) — with float kinematics, Euclidean geometry, a persistent heading and a `(speed, turn)` action that the environment rescales by `max_step_mm` and `max_turn_rad`; "stay" is emergent (speed ≈ 0). Concentration fields are Fick-shaped (`gradient_field_mode: fick`) and an adaptive, biphasic chemosensor (`adaptive_chemosensor_*`) provides Weber-law fold-change coding ([Logbook 028](experiments/logbooks/028-rung2-gradients-adaptive-sensor.md)).

**Sensing modes** (the config's `_sensing` suffix; [Logbook 009](experiments/logbooks/009-temporal-sensing-evaluation.md)):

- `oracle` — directional gradient vectors for food (attraction) and predators (repulsion); the least biological, the easiest to learn.
- `temporal` (Mode A) — scalar concentration only; the agent must infer direction from its own movement history (klinokinesis), which needs a recurrent brain.
- `derivative` (Mode B) — scalar concentration plus its temporal derivative dC/dt.
- `klinotaxis` (Mode C) — scalar concentration, a lateral head-sweep gradient and dC/dt: the most biologically complete, modelled on ASE head sweeps, and the Phase 6 standard.
- **STAM** — an exponential-decay short-term associative memory buffer of recent readings, position deltas and action entropy, available to every mode.

**Foraging and homeostasis.** Multiple food sources spawn and respawn; eating restores satiety, moving depletes it, and a run ends in success when the target number of foods is collected, or in failure on starvation, health depletion or `max_steps`. Satiety, health and the per-step reward shaping are all config-level (`satiety`, `health`, `reward` blocks). Source depletion (`ars_depletion` variants) lets food patches deplete in-episode ([Logbook 032](experiments/logbooks/032-ars-source-depletion.md)).

**Predators** ([Logbook 024](experiments/logbooks/024-predator-sensing-biology.md)). Two `PredatorType`s: **stationary** (a fixed toxic zone, modelled on constricting-ring fungal traps) and **pursuit** (moves toward the agent inside its detection radius, default 8 units, and wanders outside it; configurable speed, default 1 unit/step). Contact inside `damage_radius` costs health (default 0 — same-cell contact on the grid; the continuous substrate uses 1.0 mm because a zero radius is unreachable in continuous space). Predators are sensed through two biologically-grounded channels — contact mechanosensation (anterior/posterior/lateral zones, ASH/ALM/AVM/PVD/PLM-like) and distal chemosensation (ASH/ASI sulfolipid-like) — plus the legacy oracle gradient. Learning signals: a proximity penalty inside the detection radius, a death penalty on a fatal collision (default −10.0), and per-predator encounter and evasion metrics. Predator behaviour is pluggable through the `PredatorBrain` protocol ([Logbook 016](experiments/logbooks/016-predator-brain-refactor.md)).

**Thermotaxis and aerotaxis** ([Logbooks 007](experiments/logbooks/007-ppo-thermotaxis-baselines.md), [010](experiments/logbooks/010-aerotaxis-baselines.md)). Temperature fields (linear gradients or scattered hot/cold spots) define comfort, discomfort and danger zones around a cultivation setpoint; oxygen fields define an asymmetric five-zone system around the 5–12% O₂ comfort range (URX/BAG-inspired). Both contribute comfort scores and, in their danger zones, health damage.

**Multi-agent** ([Logbook 011](experiments/logbooks/011-multi-agent-evaluation.md)). 2–10 agents step synchronously with food competition policies, three pheromone channels (food-marking, alarm, aggregation) with configurable decay, npr-1-style social feeding (reduced satiety decay near others), social proximity sensing, and collective metrics (aggregation index, alarm evasion, food sharing).

**Working-memory probes** ([Logbooks 030](experiments/logbooks/030-bit-memory-positive-control.md), [033](experiments/logbooks/033-associative-memory-probe.md)). `bit_memory` and `associative_memory` are non-spatial task families: the observation is a cue, (for associative memory) an outcome, and a go signal; there are no gradients or STAM, so only internal recurrent state can bridge the delay.

The biology each of these models is referenced in [nematode_biology.md](nematode_biology.md).
