# 010: Aerotaxis — Oxygen Sensing Baselines and Multi-Modality Evaluation

**Status**: `in_progress` — oracle + derivative complete, temporal in progress

**Branch**: `feat/add-aerotaxis-system` (implementation), `feat/add-aerotaxis-system-eval` (oracle), `feat/add-aerotaxis-system-eval-2` (derivative/temporal)

**Date Started**: 2026-04-02

**Date Resumed**: 2026-04-05

## Objective

Establish performance baselines for the new aerotaxis (oxygen sensing) system across all environment configurations. Determine how oxygen as a third environmental constraint (alongside food and temperature) affects agent performance, and whether temporal sensing can learn oxygen navigation from scalar-only readings.

## Background

Oxygen sensing was implemented in Phase 3 as an extension of the thermotaxis system. Real C. elegans prefers 5-12% O2, using URX/AQR/PQR neurons (hyperoxia detection) and BAG neurons (hypoxia detection). Unlike thermotaxis (symmetric zones around cultivation temperature), oxygen uses an asymmetric 5-zone system with absolute percentage thresholds.

The implementation follows the established thermotaxis pattern:

- **OxygenField** class with linear gradients + high/low oxygen spots (bacterial sinks / ventilation points)
- Full oracle, temporal, and derivative sensing mode support
- STAM expanded from 3 to 4 channels (9→11 dim)
- Zone-based rewards/penalties and HP damage

Oxygen difficulty was tuned to match thermal challenge level: ~46% comfort cell coverage on medium (50×50) grids, ~50% on large (100×100) grids. This was validated against thermal-only baselines where comfort coverage is 34% (medium) and 51% (large).

**Prior work**: Logbook 007 (PPO thermotaxis baselines), Logbook 009 (temporal sensing evaluation).

## Hypothesis

1. Oracle MLP PPO should achieve ≥80% L100 on oxygen-only foraging (comparable to thermal foraging baselines)
2. Adding oxygen to thermal environments should reduce performance by 10-20pp versus thermal-only controls
3. The agent should learn to navigate both oxygen and temperature zones simultaneously in combined environments
4. Temporal GRU sensing should achieve ≥50% of oracle performance on oxygen foraging within 12000 episodes

## Method

### Environments Tested

| Scenario | Grid | Modalities | Episodes | Notes |
|----------|------|------------|----------|-------|
| O2 foraging medium | 50×50 | food + O2 | 2000 | Steepest O2 gradient (0.30%/cell) |
| O2 foraging large | 100×100 | food + O2 | 2000 | O2 spots + gentle gradient |
| O2 + pursuit | 100×100 | food + O2 + 4 pursuit pred | 2000 | |
| O2 + stationary | 100×100 | food + O2 + 5 stationary pred | 2000 | |
| O2 + thermal foraging | 100×100 | food + O2 + temp | 2000 | Orthogonal gradients |
| O2 + thermal + pursuit | 100×100 | food + O2 + temp + 4 pursuit | 2000 | Hardest multi-objective |
| O2 + thermal + stationary | 100×100 | food + O2 + temp + 5 stationary | 2000 | |
| Thermal foraging (control) | 100×100 | food + temp | 2000 | No oxygen — comparison baseline |
| Thermal + pursuit (control) | 100×100 | food + temp + 4 pursuit | 2000 | No oxygen — comparison baseline |
| Thermal + stationary (control) | 100×100 | food + temp + 5 stationary | 2000 | No oxygen — comparison baseline |
| O2 foraging temporal | 50×50 | food + O2 (temporal mode) | 12000 | LSTM PPO GRU, scalar-only |

### Key Configuration

- **Oracle**: MLP PPO (128-256 hidden, 2-4 layers), `sensory_modules: [food_chemotaxis, aerotaxis, ...]`
- **Temporal**: LSTM PPO GRU (64 hidden, chunk 64), `aerotaxis_mode: temporal`, STAM enabled
- **Oxygen field**: `base_oxygen: 10.0%`, gradient orthogonal to temperature (π/2 rad), 2 high + 3 low O2 spots on large grids
- All experiments: 4 seeds (42-45) per configuration

### Code Changes

- **New**: `env/oxygen.py` — OxygenField, OxygenZone (5 zones), OxygenZoneThresholds
- **New**: `brain/modules.py` — `_aerotaxis_core()`, `_aerotaxis_temporal_core()` sensory modules
- **Modified**: `agent/stam.py` — 3→4 channels (MEMORY_DIM 9→11)
- **Modified**: `env/env.py` — AerotaxisParams, oxygen methods, apply_oxygen_effects()
- **Modified**: Full data pipeline (tracker, CSV, plots, experiment metadata)
- **New**: 11 scenario configs across 6 directories
- **New**: 16 oxygen tests + STAM regression fixes

______________________________________________________________________

## Results

### Oracle Baselines — Summary (L100, mean of 4 seeds)

| Scenario | L100 | L500 | O2 Comfort | Temp Comfort |
|----------|------|------|------------|--------------|
| O2 foraging medium | 76% | 75% | 0.87 | — |
| O2 foraging large | **79%** | 78% | 0.97 | — |
| O2 + pursuit | 63% | 60% | 0.95 | — |
| O2 + stationary | 47% | 42% | 0.90 | — |
| O2 + thermal foraging | **89%** | 84% | 0.96 | 0.70 |
| O2 + thermal + pursuit | 70% | 57% | 0.94 | 0.75 |
| O2 + thermal + stationary | 51% | 46% | 0.97 | 0.75 |

### Thermal Controls (no oxygen)

| Scenario | L100 | L500 | Temp Comfort |
|----------|------|------|--------------|
| Thermal foraging (control) | **99%** | 98% | 0.72 |
| Thermal + pursuit (control) | **94%** | 93% | 0.74 |
| Thermal + stationary (control) | **93%** | 93% | 0.73 |

### Impact of Adding Oxygen

| Comparison | Without O2 | With O2 | Drop |
|------------|-----------|---------|------|
| Foraging (large) | 99% L100 | 89% L100 | **-10pp** |
| Pursuit (large) | 94% L100 | 70% L100 | **-24pp** |
| Stationary (large) | 93% L100 | 51% L100 | **-42pp** |

### Temporal Sensing — Medium Foraging (12000 episodes)

| Mode | Brain | L100 | L1000 | O2 Comfort |
|------|-------|------|-------|------------|
| Oracle | MLP PPO | **76%** | 72% | 0.87 |
| Temporal | LSTM PPO GRU | 10% | **13%** | 0.97 |

Oracle→temporal gap: **66pp on L100, 59pp on L1000**.

### Derivative Sensing — Large Grid (6000 episodes, mean of 3-4 seeds)

| Scenario | Oracle L100 | Derivative L100 | Gap |
|----------|-------------|-----------------|-----|
| O2 foraging | 79% | **83%** | +4pp |
| O2+thermal foraging | 89% | **97%** | +8pp |
| O2+pursuit | 63% | **91%** | +28pp |
| O2+thermal+pursuit | 70% | **88%** | +18pp |
| O2+stationary | 47% | **65%** | +18pp |
| O2+thermal+stationary | 51% | **57%** | +6pp |

Derivative exceeds oracle on every scenario.

### Temporal Sensing — Large Grid O2 Foraging (6000 episodes)

| Mode | Grid | Brain | L100 | L500 | O2 Comfort |
|------|------|-------|------|------|------------|
| Oracle | large | MLP PPO | 79% | 78% | 0.97 |
| **Temporal** | **large** | **GRU** | **99%** | **97%** | **0.96** |
| Derivative | large | GRU | 83% | 77% | 0.97 |
| Temporal | medium | GRU | 10% | — | 0.97 |

Large grid temporal achieves **99% L100** — exceeding both oracle and derivative. The medium grid temporal plateau (13%) was a grid-size problem, not a fundamental limitation.

______________________________________________________________________

## Analysis

### H1: Oracle ≥80% L100 on O2 foraging — PARTIALLY MET

Medium oracle achieves 76% L100 (below 80% target), large oracle achieves 79% L100. The medium grid's steep gradient (0.30%/cell, 46% comfort) makes it genuinely harder than expected. Large grid (0.12%/cell + spots, 50% comfort) is closer to the target. The difficulty tuning was intentionally calibrated to match thermal challenge levels.

### H2: O2 adds 10-20pp drop vs thermal-only — EXCEEDED

Foraging drops by 10pp (99→89%), within the predicted 10-20pp range. However, pursuit drops by 24pp (94→70%) and stationary by 42pp (93→51%) — both exceeding the predicted range. The stationary scenario is most severely impacted as toxic zones compound with O2 danger zones to restrict the safe foraging area. Oxygen creates a stronger constraint than anticipated, particularly in predator scenarios.

### H3: Agent learns dual O2+temperature navigation — CONFIRMED

In combined environments, the agent demonstrably learns both modalities:

- O2 comfort rises from 0.85→0.97 over training (episodes 1-200 → 1500-2000)
- Temperature comfort stays at 0.70-0.75 (constrained by hot/cold spot layout, matching thermal-only baselines)
- HP deaths drop 5-6× over training
- Success rate rises from 15-34% → 81-96% (O2+thermal foraging)

### H4: Temporal ≥50% of oracle — REVISED: EXCEEDED (on large grid)

**Medium grid (initial assessment)**: Temporal GRU reaches only 13% L1000 after 12000 episodes (oracle 72%), 18% of oracle. The learning curves plateau in 8000-12000 range.

**Large grid (revised)**: Temporal GRU achieves **99% L100, 97% L500** (oracle 79%) — exceeding oracle by 20pp. The medium grid plateau was entirely a grid-size problem: the 46% comfort coverage and 0.30%/cell gradient was too constraining for scalar-only food-finding. The large grid (50% comfort, 0.12%/cell, more food) resolves this completely.

### H5 (new): Derivative sensing competitive with oracle — EXCEEDED

Derivative GRU exceeds MLP PPO oracle on all 6 scenarios tested (+4pp to +28pp). The dC/dt, dO2/dt, dT/dt signals combined with GRU memory provide more effective navigation than stateless spatial gradients. Pursuit scenarios show the largest gains (+18-28pp) — the dP/dt predator signal is a powerful evasion cue.

### Grid Size as Critical Variable

The medium vs large grid comparison reveals grid size as the most important variable for temporal/derivative sensing:

| Mode | Medium (50×50) | Large (100×100) | Delta |
|------|---------------|-----------------|-------|
| Oracle | 76% | 79% | +3pp |
| Temporal | 13% | **99%** | **+86pp** |
| Derivative | — | 83% | — |

The medium grid's 46% O2 comfort coverage creates a narrow safe band that temporal sensing can't navigate efficiently. The large grid's 50% coverage with gentler gradient gives temporal agents room to learn. This contrasts with thermal temporal (Logbook 009) where medium grids were sufficient — the oxygen comfort range is narrower relative to the field range, making it inherently more grid-size-sensitive.

### Difficulty Calibration Notes

Initial oxygen gradient strengths were too gentle (78% comfort on medium, 73% on large — trivially easy). After tuning to 0.30 (medium) and 0.12 + spots (large), oracle performance dropped to realistic levels. The LR schedule in the medium config also needed fixing (lr_decay_episodes: 200→1500 — was decaying before training started).

______________________________________________________________________

## Conclusions

1. **Oxygen sensing works as intended.** Oracle MLP PPO achieves 76-89% L100 across foraging scenarios, creating genuine navigational challenge.

2. **Oxygen creates meaningful multi-objective pressure.** Adding O2 to thermal environments drops performance by 10-42pp versus thermal-only controls (foraging -10pp, pursuit -24pp, stationary -42pp). The stationary scenario is hardest-hit as toxic zones compound with O2 danger zones.

3. **Dual-modality learning is confirmed.** In combined O2+thermal environments, the agent learns to navigate both oxygen and temperature zones simultaneously, with O2 comfort rising from 0.85→0.97 over training.

4. **Temperature comfort is unaffected by adding oxygen.** Combined scenarios show temp comfort of 0.70-0.75, matching thermal-only controls (0.72-0.74). The two modalities don't interfere.

5. **Stationary predators remain the hardest scenario.** O2+stationary (47% L100) and O2+thermal+stationary (51% L100) are the most challenging oxygen environments, as toxic zones severely constrain safe foraging area.

6. **Derivative mode exceeds oracle on all scenarios.** GRU derivative achieves +4pp to +28pp over MLP PPO oracle across all 6 scenarios. Pursuit scenarios show the largest gains — dP/dt is a powerful evasion cue.

7. **Large grid temporal exceeds oracle on O2 foraging.** GRU temporal achieves 99% L100 vs oracle's 79% on the large grid. The medium grid plateau (13%) was a grid-size problem, not a fundamental limitation.

8. **Grid size is the critical variable for temporal sensing.** Medium grid temporal: 13%. Large grid temporal: 99%. The oxygen comfort range is narrower than thermal, making temporal sensing more grid-size-sensitive. Large grids are required for oxygen temporal evaluation.

9. **`gradient_decay_constant` is critical for temporal/derivative configs.** The O2+thermal foraging temporal config failed (0% success) because it used 12.0 instead of 4.0. Steeper food gradients (4.0) are required for detectable scalar concentration changes.

## Next Steps

Temporal evaluation on large grid in progress. Remaining scenarios:

1. **O2+thermal foraging temporal** — re-run with fixed `gradient_decay_constant: 4.0` (config corrected)
2. **O2+pursuit temporal** — config ready
3. **O2+thermal+pursuit temporal** — config ready
4. **O2+stationary temporal** — config ready
5. **O2+thermal+stationary temporal** — config ready

## Data References

- **Artifacts**: `artifacts/logbooks/010/`
  - Oracle baselines (7 groups):
    - `mlpppo_medium_oracle/` — 4 sessions + config + best-seed weights
    - `mlpppo_large_oracle/` — 4 sessions + config + best-seed weights
    - `mlpppo_oxygen_pursuit/` — 4 sessions + config + best-seed weights
    - `mlpppo_oxygen_stationary/` — 4 sessions + config + best-seed weights
    - `mlpppo_oxygen_thermal_foraging/` — 4 sessions + config + best-seed weights
    - `mlpppo_oxygen_thermal_pursuit/` — 4 sessions + config + best-seed weights
    - `mlpppo_oxygen_thermal_stationary/` — 4 sessions + config + best-seed weights
  - Thermal controls (3 groups):
    - `thermal_foraging_control/` — 4 sessions + config + best-seed weights
    - `thermal_pursuit_control/` — 4 sessions + config + best-seed weights
    - `thermal_stationary_control/` — 4 sessions + config + best-seed weights
  - Derivative (6 groups):
    - `lstmppo_oxygen_foraging_derivative/` — 4 sessions + config + best-seed weights
    - `lstmppo_oxygen_thermal_foraging_derivative/` — 4 sessions + config + best-seed weights
    - `lstmppo_oxygen_pursuit_derivative/` — 4 sessions + config + best-seed weights
    - `lstmppo_oxygen_thermal_pursuit_derivative/` — 3 sessions + config + best-seed weights
    - `lstmppo_oxygen_stationary_derivative/` — 4 sessions (exports-only) + config + best-seed weights
    - `lstmppo_oxygen_thermal_stationary_derivative/` — 4 sessions + config + best-seed weights
  - Temporal (2 groups):
    - `lstmppo_medium_temporal_12k/` — 4 sessions + config + best-seed weights
    - `lstmppo_oxygen_foraging_temporal_large/` — 4 sessions (exports-only) + config + best-seed weights
- **Configs**: `configs/scenarios/oxygen_*/`, `configs/scenarios/oxygen_thermal_*/`
- **Supporting data**: [010/aerotaxis-baselines-details.md](supporting/010/aerotaxis-baselines-details.md)
- **Scratchpad**: `tmp/evaluations/aerotaxis_scratchpad.md`
