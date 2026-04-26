# Configuration Files

YAML configuration files for the Quantum Nematode simulation. Configs are standalone files parsed by Pydantic — no inheritance or include mechanisms.

## Directory Structure

```text
configs/
  scenarios/          Experiment configs organized by scenario
  evolution/          Evolutionary optimization configs
  special/            One-off experimental configs
```

## Scenarios

Each scenario directory represents a distinct environment configuration:

| Directory | Modalities | Description |
|---|---|---|
| `foraging/` | Chemotaxis | Pure food-seeking |
| `pursuit/` | Chemotaxis + pursuit predators | Active predator evasion |
| `stationary/` | Chemotaxis + stationary predators | Static predator avoidance |
| `thermal_foraging/` | Chemotaxis + thermotaxis | Temperature gradient + food-seeking |
| `thermal_pursuit/` | Chemotaxis + thermotaxis + pursuit | Multi-objective (hardest current env) |
| `thermal_stationary/` | Chemotaxis + thermotaxis + stationary | Multi-objective with static predators |
| `multi_agent_foraging/` | Multi-agent + chemotaxis (+ pheromones, social feeding) | Multi-agent food-seeking |
| `multi_agent_pursuit/` | Multi-agent + chemotaxis + pursuit (+ pheromones) | Multi-agent with pursuit predators |
| `multi_agent_stationary/` | Multi-agent + chemotaxis + stationary (+ pheromones) | Multi-agent with stationary predators |

## Naming Convention

Within each scenario directory:

```text
{brain}_{size}[_{variant}]_{sensing}.yml
```

- **brain**: Architecture name (e.g., `mlpppo`, `qef`, `lstmppo`, `crh`)
- **size**: `small` (20x20), `medium` (50x50), `large` (100x100)
- **variant** (optional): `classical`, `fair`, `separable`, `modality_paired`, `ring_compact`, `sensory`, `isothermal`, `satiety`, etc.
- **sensing**: `oracle`, `temporal`, `derivative`

Examples:

- `mlpppo_small_oracle.yml` — MLP PPO, small grid, oracle sensing
- `lstmppo_large_temporal.yml` — LSTM PPO, large grid, temporal sensing
- `qef_small_modality_paired_oracle.yml` — QEF with modality-paired topology
- `crhqlstm_large_classical_oracle.yml` — CRH-QLSTM classical ablation

## Usage

```bash
uv run ./scripts/run_simulation.py --config ./configs/scenarios/foraging/mlpppo_small_oracle.yml
```

## Adding Configs

1. Choose the appropriate scenario directory (or create a new one for new modality combinations)
2. Follow the naming convention: `{brain}_{size}[_{variant}]_{sensing}.yml`
3. Copy an existing config from the same scenario as a starting point
4. Modify only the parameters that differ
