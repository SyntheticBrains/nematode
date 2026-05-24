# Supporting: 024 smoke evaluation (predator-sensing-biology)

Companion data for the [024 — Corrected Biology-Driven Predator Sensing](../../024-predator-sensing-biology.md) logbook's behavioural-difference quantification. Captures the configs, seed, and reproducibility details so a future reader can re-run the 100-episode head-to-head smoke from scratch.

## Setup

- **Seed**: 2026
- **Episodes per run**: 100
- **Brain hyperparameters**: identical between each (new biology, legacy nociception) pair.
- **Env**: identical between pairs — `pursuit` scenario, grid 20×20, 6 foods on grid, 10-food target, predators enabled (count=2, speed=0.5, pursuit movement pattern, detection_radius=6, gradient_decay_constant=10.0, damage_radius default), STAM enabled (buffer_size=30, decay_rate=0.1), klinotaxis sensing mode for chemotaxis.

## Configs

| Variant | Config path | Brain | Sensors |
|---|---|---|---|
| A | `configs/scenarios/pursuit/mlpppo_small_predator_biology_klinotaxis.yml` | MLPPPO | new biology (mechano+chemo klinotaxis, 6-dim) |
| B | `configs/scenarios/pursuit/lstmppo_small_predator_biology_klinotaxis.yml` | LSTMPPO GRU | new biology (mechano+chemo klinotaxis, 6-dim) |
| C (control) | *one-shot tmp/ config* | MLPPPO | legacy `nociception_klinotaxis` (3-dim) |
| D (control) | *one-shot tmp/ config* | LSTMPPO GRU | legacy `nociception_klinotaxis` (3-dim) |

The legacy-nociception control configs (C and D) were one-shot evaluation artefacts kept under the ignored `tmp/evaluations/predator-sensing-biology-smoke/` directory at evaluation time. They are reproducible from the committed new-biology configs (A and B) by swapping:

- `sensory_modules`: replace `predator_mechanosensation_oracle` and `predator_chemosensation_oracle` with the single legacy entry `nociception`.
- `environment.sensing`: remove `predator_mechano_mode` and `predator_distal_mode`; add `nociception_mode: klinotaxis`.

All other fields (brain hyperparameters, reward shape, satiety, env, predator config, health) remain identical between the new-biology and legacy-control variants.

## Command

```bash
uv run python scripts/run_simulation.py \
  --config <config_path> \
  --runs 100 \
  --theme headless \
  --log-level WARNING \
  --seed 2026
```

## Results (100 episodes)

| Variant | Sensors | Success | Foods/ep | Reward/ep | Steps/ep |
|---|---|---|---|---|---|
| **C — MLPPPO legacy** | `nociception_klinotaxis` (3-dim) | **51%** | 7.27 | +22.87 | 250.9 |
| A — MLPPPO new biology | mechano+chemo klinotaxis (6-dim) | 3% | 2.80 | +0.74 | 148.3 |
| **D — LSTMPPO legacy** | `nociception_klinotaxis` (3-dim) | **7%** | 2.25 | -1.76 | 157.4 |
| B — LSTMPPO new biology | mechano+chemo klinotaxis (6-dim) | 0% | 0.93 | -6.83 | 109.7 |

## Per-bin success-rate observations

**A (MLPPPO new biology)** moves from 0% (episode 1) to 3% by episode 100 (episodes 98 and 100 reach 10/10 foods, the rest fail health_depleted). Real signal — gradient flow + visible policy improvement, just slow vs C.

**C (MLPPPO legacy)** reaches 51% over the same 100 episodes — a 48 percentage point advantage at this matched compute. Both runs die to predator contact (`health_depleted`); foraging signal is identical between variants since the food channel is unchanged.

**B (LSTMPPO new biology)** stays at 0% across the 100 episodes. **D (LSTMPPO legacy)** reaches 7%. PPO-with-recurrence typically needs ≥ 500 episodes to converge regardless, so the LSTMPPO gap is partly noise.

## Reproducibility check

Two failure modes worth flagging up-front for any re-run:

1. **STAM-dim inference** (`brain/modules.py::_infer_stam_dim_from_modules`) must include the new `PREDATOR_MECHANOSENSATION_*` and `PREDATOR_CHEMOSENSATION_*` triples in its modality_triples list. Without that, the brain's first Linear layer is sized for the wrong number of STAM channels and crashes with a shape-mismatch at the first forward pass. This was found during the initial smoke and is fixed in the shipped capability.
2. **Sensing-mode keys**: new-biology configs use `sensing.predator_mechano_mode` and `sensing.predator_distal_mode`; legacy configs use `sensing.nociception_mode`. The `apply_sensing_mode` translation in `utils/config_loader.py` routes each appropriately.

## Root-cause hypotheses (carried forward)

The convergence-rate gap finding is recorded in three durable locations:

- [Logbook 024 § Verdict](../../024-predator-sensing-biology.md#verdict).
- The archived [fix-predator-sensing-biology design.md § Modelling caveat 6](../../../../openspec/changes/archive/2026-05-24-fix-predator-sensing-biology/design.md).
- [phase6-tracking T4.0g](../../../../openspec/changes/phase6-tracking/tasks.md) (explicit follow-up sub-task with sparse-signal and information-redundancy hypotheses spelled out).

Future evaluation tranches own the empirical hypothesis tests against canonical compute budgets.
