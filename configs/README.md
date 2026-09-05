# Configuration Files

YAML configuration files for the Quantum Nematode simulation. Every config is a standalone file parsed by Pydantic — there is no inheritance or include mechanism, so each file says everything about its run: the brain and its hyperparameters, the environment, the sensing mode and the reward.

```text
configs/
  scenarios/          Simulation configs, one directory per scenario (environment + behaviour combination)
  evolution/          Evolutionary-optimisation configs for run_evolution.py (weights, hyperparameters, inheritance, co-evolution)
  special/            One-off experimental configs kept for reproducibility (e.g. fine-tuning and curriculum rounds from Logbook 008)
```

## Scenarios

| Directory | Modalities | Description |
|---|---|---|
| `foraging/` | Chemotaxis | Pure food-seeking — the largest directory; includes the continuous-2D calibration configs |
| `pursuit/` | Chemotaxis + pursuit predators | Active predator evasion |
| `stationary/` | Chemotaxis + stationary predators | Static predator avoidance |
| `thermal_foraging/` | Chemotaxis + thermotaxis | Temperature gradient + food-seeking |
| `thermal_pursuit/` | Chemotaxis + thermotaxis + pursuit | Multi-objective with pursuit predators |
| `thermal_stationary/` | Chemotaxis + thermotaxis + stationary | Multi-objective with static predators |
| `oxygen_foraging/` | Chemotaxis + aerotaxis | Oxygen gradient + food-seeking |
| `oxygen_pursuit/`, `oxygen_stationary/` | Chemotaxis + aerotaxis + predators | Oxygen gradient with predators |
| `oxygen_thermal_foraging/`, `oxygen_thermal_pursuit/`, `oxygen_thermal_stationary/` | Chemotaxis + aerotaxis + thermotaxis (+ predators) | Orthogonal oxygen and thermal gradients; the triple-modality cells |
| `foraging_predator_thermal/` | Chemotaxis + pursuit predators + thermotaxis, klinotaxis sensing | The integrated **C3 cell** used for the Phase 6 architecture rankings ([Logbooks 025](../docs/experiments/logbooks/025-weight-search-architecture-ranking.md), [029](../docs/experiments/logbooks/029-continuous-architecture-ranking.md)); `_continuous2d_` variants are the continuous-substrate cell |
| `multi_agent_foraging/` | Multi-agent + chemotaxis (+ pheromones, social feeding, competition) | 2–10 worms sharing an arena |
| `multi_agent_pursuit/`, `multi_agent_stationary/` | Multi-agent + chemotaxis + predators (+ alarm pheromones) | Multi-agent with predators |
| `bit_memory/` | None (non-spatial) | Delayed-match-to-cue working-memory positive control ([Logbook 030](../docs/experiments/logbooks/030-bit-memory-positive-control.md)); its own task family, not a variant of a spatial scenario |
| `associative_memory/` | None (non-spatial) | Chemosensory delayed-associative-match with within-trial reversal — a working-memory *update* probe ([Logbook 033](../docs/experiments/logbooks/033-associative-memory-probe.md)) |

## Naming convention

```text
{brain}_{size}[_{variant}]_{sensing}[_{derived}].yml
```

- **brain** — the registered architecture name (`mlpppo`, `lstmppo`, `connectomeppo`, `qef`, …); see [docs/architectures.md](../docs/architectures.md). Multi-agent configs insert the agent count (`_5agents_`); `mixed_brains_…` runs heterogeneous brains.
- **size** — `small` (20×20), `medium` (50×50), `large` (100×100) grid cells; continuous-2D configs set the arena in millimetres with `world_size_mm` instead (20 mm for the single-behaviour canary cells, 60 mm for the Phase 6 ranking cells — the C2 pursuit and integrated C3 cells).
- **variant** (optional) — what differs from the scenario's default cell. Common ones: `continuous2d` (continuous-2D substrate), `fick_adaptive` (Fick-shaped fields + adaptive chemosensor), `combined` (all scenario modalities active), `classical` / `fair` / `separable` / `rewired_null` / `frozen_control` (ablations and controls), `sdstd` (state-dependent action std, roadmap D7 — a single-key `continuous_std_mode: state_dependent` delta from its parent; dormant capability per Logbook 038's Amendment A), `plastic` (the connectome trained by the reward-modulated three-factor rule instead of PPO — a two-key delta enabling the rule and the eligibility traces it reads), `pheromone` / `no_pheromone` / `social` / `aggregation` / `scarcity` / `competition` (multi-agent conditions), `ars_depletion` / `no_respawn_control` (source-depletion study), `1agent` (single-agent run of a multi-agent config).
- **sensing** — `oracle` (spatial gradient vectors), `temporal` (scalar concentration only), `derivative` (scalar + dC/dt), `klinotaxis` (scalar + head-sweep lateral gradient + dC/dt — the most biologically complete, and the Phase 6 standard).
- **derived** (optional) — a config that is a small, named delta from an existing parent config appends its marker *after* the sensing token, so the parent's full name stays a prefix of the child's and the two sort together: `…_klinotaxis_rewired_null.yml`, `…_klinotaxis_ars_depletion.yml`, `…_klinotaxis_no_respawn_control.yml`, `…_klinotaxis_sdstd.yml`. The unsuffixed parent is left untouched because it is the experimental record its logbook cites.
- Non-spatial task families use a task suffix instead: `_bit_memory`, `_associative_memory`.

Examples:

- `foraging/mlpppo_small_oracle.yml` — MLP-PPO, small grid, oracle sensing
- `foraging/mlpppo_small_continuous2d_fick_adaptive_klinotaxis.yml` — the same brain on the continuous substrate with the fidelity upgrades
- `foraging_predator_thermal/connectomeppo_small_continuous2d_combined_klinotaxis.yml` — the connectome brain on the Phase 6 ranking cell
- `multi_agent_foraging/lstmppo_large_5agents_single_cluster_pheromone_klinotaxis.yml` — five GRU worms with food-marking pheromones on clustered food
- `bit_memory/cfcppo_small_bit_memory.yml` — CfC on the working-memory control

## Evolution configs

`configs/evolution/` holds configs for `scripts/run_evolution.py` and `scripts/run_coevolution.py`: brain-weight evolution (`feedforwardga_*`, `mlpppo_foraging_small`, `lstmppo_foraging_small_klinotaxis`), hyperparameter evolution (`hyperparam_*`), inheritance studies (`lamarckian_*`, `baldwin_*`, `transgenerational_*`, `tei_prior_*`) and predator–prey co-evolution (`coevolution_*`, with warm-start prey bundles under `coevolution_warmstart_prey/`). The `*_smoke` and `*_pilot` suffixes mark reduced budgets. See the [usage guide](../docs/usage.md#evolution-and-inheritance).

## Usage

```bash
uv run ./scripts/run_simulation.py --config ./configs/scenarios/foraging/mlpppo_small_oracle.yml
```

Brain-config keys the selected brain does not accept are reported with a warning at load time (and dropped), and the non-spatial task families reject unknown keys outright — so a typo does not quietly become a default.

## Adding a config

1. Choose the scenario directory, or create one for a new modality combination.
2. Follow the naming convention.
3. Copy the closest existing config from the same scenario and change only the parameters that differ — the comment block at the top of each config explains the calibration choices it inherits.
