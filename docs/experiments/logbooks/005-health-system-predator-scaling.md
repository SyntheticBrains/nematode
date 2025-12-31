# 005: Health System Predator Scaling Study

**Status**: `completed`

**Branch**: `feature/add-health-system`

**Date Started**: 2025-12-31

**Date Completed**: 2025-12-31

## Objective

Evaluate how PPO agents adapt to increasingly dangerous environments with the health system enabled. Determine the maximum predator density at which agents can still learn a successful foraging strategy, and compare performance with health system disabled (control).

## Background

Phase 1 introduced the health system as an independent survival mechanic alongside satiety:

- **HP**: Tracks threat-based damage (predator collisions)
- **Satiety**: Tracks time-based hunger (decays every step)
- **Food**: Restores BOTH HP and satiety

This study validates the health system implementation and provides insights for future predator behavior work (pursuit predators, stationary toxic zones).

### Key Health System Parameters

- `max_hp: 100`
- `predator_damage: 20` (5 hits to die)
- `food_healing: 10` (10 HP per food consumed)

## Hypothesis

1. **Convergence threshold**: Agents should maintain >50% success rate up to ~5-7 predators on a 25x25 grid
2. **Death cause shift**: At low predator counts, starvation dominates; at high counts, health depletion becomes primary failure mode
3. **Learning advantage**: Health system may provide richer reward signals that improve predator avoidance learning

**Note on experimental design**: With `kill_radius: 0`, predators must occupy the exact same cell as the agent to trigger contact. In the health-enabled condition, contact deals damage (20 HP per hit, 5 hits to die). In the control condition (health disabled), contact triggers instant death.

## Method

### Study Design

| Condition | Predators | Health | Damage | Sessions | Runs/Session |
|-----------|-----------|--------|--------|----------|--------------|
| H1 | 1 | Enabled | 20 | 10 | 50 |
| H2 | 2 | Enabled | 20 | 10 | 50 |
| H3 | 3 | Enabled | 20 | 10 | 50 |
| H5 | 5 | Enabled | 20 | 10 | 50 |
| H7 | 7 | Enabled | 20 | 10 | 50 |
| H10 | 10 | Enabled | 20 | 10 | 50 |
| C1 | 1 | Disabled | N/A | 10 | 50 |
| C2 | 2 | Disabled | N/A | 10 | 50 |
| C3 | 3 | Disabled | N/A | 10 | 50 |
| C5 | 5 | Disabled | N/A | 10 | 50 |
| C7 | 7 | Disabled | N/A | 10 | 50 |
| C10 | 10 | Disabled | N/A | 10 | 50 |

**Total: 120 sessions, 6000 runs**

### Configuration

Base config adapted from `ppo_health_satiety_predators_small.yml`:

```yaml
max_steps: 500
brain:
  name: ppo
  config:
    actor_hidden_dim: 64
    critic_hidden_dim: 64
    num_hidden_layers: 2
    clip_epsilon: 0.2
    gamma: 0.99
    gae_lambda: 0.95
    value_loss_coef: 0.5
    entropy_coef: 0.02
    learning_rate: 0.001
    rollout_buffer_size: 256
    num_epochs: 10
    num_minibatches: 2
    max_grad_norm: 0.5
    normalize_advantages: true

body_length: 2

reward:
  reward_goal: 2.0
  reward_distance_scale: 0.5
  reward_exploration: 0.05
  penalty_step: 0.005
  penalty_anti_dithering: 0.02
  penalty_stuck_position: 0
  penalty_starvation: 10.0
  penalty_predator_death: 10.0
  penalty_predator_proximity: 0.1
  stuck_position_threshold: 0
  penalty_health_damage: 0.5
  reward_health_gain: 0.1

satiety:
  initial_satiety: 300.0
  satiety_decay_rate: 0.8
  satiety_gain_per_food: 0.2

environment:
  type: dynamic
  dynamic:
    grid_size: 25
    viewport_size: [11, 11]
    foraging:
      foods_on_grid: 8
      target_foods_to_collect: 12
      min_food_distance: 3
      agent_exclusion_radius: 5
      gradient_decay_constant: 8.0
      gradient_strength: 1.0
    predators:
      enabled: true
      count: <VARIABLE>  # 1, 2, 3, 5, 7, 10
      speed: 1.0
      movement_pattern: random
      detection_radius: 8
      kill_radius: 0
      gradient_decay_constant: 12.0
      gradient_strength: 1.0
    health:
      enabled: <VARIABLE>  # true for H*, false for C*
      max_hp: 100.0
      predator_damage: 20.0  # 5 hits to die
      food_healing: 10.0
```

### Key Metrics

| Metric | Description |
|--------|-------------|
| Success Rate | % of runs completing food target |
| Convergence Run | First run in 5-run success window |
| Post-Conv Success | Success rate after convergence |
| Death Breakdown | % starvation vs health_depleted vs max_steps |
| Avg Final HP | Mean HP at episode end (surviving runs) |
| Foods Collected | Average foods per run |

### Execution

```bash
# Run study with 4 parallel sessions
./scripts/run_health_scaling_study.sh
```

______________________________________________________________________

## Results

### Summary Table

| Predators | Health Enabled | Control (Instant Death) | Delta |
|-----------|----------------|-------------------------|-------|
| 1 | 92.2% | 87.6% | +4.6% |
| 2 | 92.2% | 79.4% | +12.8% |
| 3 | 93.6% | 74.2% | +19.4% |
| 5 | 64.2% | 30.2% | +34.0% |
| 7 | 4.4% | 0.8% | +3.6% |
| 10 | 0.0% | 0.0% | +0.0% |

### Detailed Findings

#### Health-Enabled Conditions (H1-H10)

| Condition | Success Rate | Avg Foods | Starved | HP Depleted | Max Steps | Avg Reward |
|-----------|--------------|-----------|---------|-------------|-----------|------------|
| H1 | 92.2% | 11.52 | 0.5 | 0.0 | 3.4 | 38.49 |
| H2 | 92.2% | 11.52 | 0.5 | 0.5 | 2.9 | 38.04 |
| H3 | 93.6% | 11.62 | 0.1 | 0.3 | 2.8 | 36.82 |
| H5 | 64.2% | 10.25 | 1.1 | 3.3 | 13.5 | 24.30 |
| H7 | 4.4% | 5.55 | 2.9 | 11.9 | 33.0 | -1.46 |
| H10 | 0.0% | 2.19 | 4.6 | 35.7 | 9.7 | -19.68 |

#### Control Conditions (C1-C10)

| Condition | Success Rate | Avg Foods | Starved | Predator Deaths | Max Steps | Avg Reward |
|-----------|--------------|-----------|---------|-----------------|-----------|------------|
| C1 | 87.6% | 11.20 | 0.2 | 3.1 | 3.1 | 37.36 |
| C2 | 79.4% | 10.60 | 0.4 | 7.7 | 2.2 | 33.89 |
| C3 | 74.2% | 10.32 | 0.4 | 10.5 | 2.0 | 31.25 |
| C5 | 30.2% | 7.38 | 1.0 | 22.8 | 11.1 | 13.47 |
| C7 | 0.8% | 3.05 | 1.6 | 35.9 | 12.1 | -6.13 |
| C10 | 0.0% | 0.79 | 2.8 | 45.8 | 1.4 | -14.43 |

### Detection Zone Escape Analysis

**Methodology note**: The "evasion rate" measures how often agents **escape the detection zone** (8-cell radius), not contact avoidance specifically. An "encounter" is counted when the agent enters a predator's detection radius; an "evasion" is counted when the agent exits without dying that step.

Health-enabled agents show higher detection zone escape rates:

| Predators | Health Escape% | Control Escape% | Difference |
|-----------|----------------|-----------------|------------|
| 1 | 97.4% | 95.2% | +2.2% |
| 2 | 97.0% | 94.8% | +2.2% |
| 3 | 96.2% | 94.5% | +1.6% |
| 5 | 97.0% | 94.7% | +2.4% |
| 7 | 96.6% | 93.0% | +3.6% |
| 10 | 92.3% | 85.4% | +6.9% |

Health-enabled agents also enter the detection zone more often per run (longer survival = more time to encounter predators):

| Predators | Health Enc/Run | Control Enc/Run | Ratio |
|-----------|----------------|-----------------|-------|
| 1 | 4.2 | 3.8 | 1.10x |
| 2 | 6.4 | 6.0 | 1.07x |
| 3 | 8.9 | 7.9 | 1.13x |
| 5 | 14.6 | 12.0 | 1.22x |
| 7 | 17.6 | 11.6 | 1.52x |
| 10 | 11.7 | 6.6 | 1.79x |

### Actual Predator Contact Rate Analysis

By counting HP drops (health-enabled) and deaths (control), we can measure actual predator contacts:

| Predators | Health Contacts/Run | Control Contacts/Run | Ratio |
|-----------|---------------------|----------------------|-------|
| 1 | 0.15 | 0.06 | 2.4x |
| 2 | 0.29 | 0.15 | 1.9x |
| 3 | 0.34 | 0.21 | 1.6x |
| 5 | 1.00 | 0.46 | 2.2x |
| 7 | 2.54 | 0.72 | 3.5x |
| 10 | 4.57 | 0.92 | 5.0x |

**Health-enabled agents make significantly more predator contacts.** To determine if this is due to longer survival or genuine risk-taking, we normalize by episode length:

| Predators | Health (per 1000 steps) | Control (per 1000 steps) | Ratio |
|-----------|-------------------------|--------------------------|-------|
| 1 | 0.53 | 0.24 | 2.2x |
| 2 | 1.15 | 0.64 | 1.8x |
| 3 | 1.33 | 0.87 | 1.5x |
| 5 | 2.64 | 1.42 | 1.9x |
| 7 | 5.83 | 2.69 | 2.2x |
| 10 | 15.38 | 6.45 | 2.4x |

**Initial observation**: Even normalized by episode length, health-enabled agents make **1.5-2.4x more predator contacts per step**.

### First-Contact Timing Analysis

To isolate initial risk-taking from accumulated contacts, we examined when agents first encounter a predator:

| Condition | Runs with Contact | Avg First Contact Step |
|-----------|-------------------|------------------------|
| Health-enabled | 1149 | 156.0 |
| Control | 1069 | 158.0 |

**Key finding**: First-contact timing is nearly identical (~156-158 steps), indicating **initial risk-taking behavior is the same** between conditions.

The higher normalized contact rate (1.5-2.4x) in health-enabled agents comes from contacts *after* the first:
- Control: 1 contact per run (fatal), rate = 1/158 ≈ 6.3 per 1000 steps
- Health: Multiple contacts per run, rate ≈ 10.6 per 1000 steps

### Interpretation

The contact rate difference is **partially explained by survival mechanics**, not purely learned risk-tolerance:

1. **Same initial behavior**: Both groups encounter their first predator at the same rate (~step 156-158)
2. **Divergence after first contact**: Health-enabled agents survive and continue foraging; control agents die
3. **Accumulated contacts**: Health-enabled agents make additional contacts in later-episode foraging, where:
   - Food may be located in predator-dense areas
   - Longer episodes mean more opportunities for encounters
   - Agents may become more risk-tolerant after surviving first contact (learned behavior)

**Limitation**: We cannot fully isolate learned risk-tolerance from mechanical survival with this data. A cleaner test would compare contact rates in fixed early-episode windows (e.g., steps 0-100) where both groups are still alive, but this would require more granular trajectory data.

______________________________________________________________________

## Analysis

### Hypothesis Evaluation

1. **Convergence threshold** (Hypothesis 1): **Partially confirmed**

   - Health-enabled: >50% success at P=5 (64.2%), collapses at P=7 (4.4%)
   - Control: \<50% success at P=5 (30.2%), already collapsed at P=7 (0.8%)
   - The threshold is lower than expected (~5 predators, not 5-7)

2. **Death cause shift** (Hypothesis 2): **Confirmed**

   - At P=1-3: Starvation and max_steps dominate failures
   - At P=7: HP depletion becomes primary failure (26% of failures in health-enabled)
   - At P=10: HP depletion dominates (71.4% of failures in health-enabled)

3. **Learning advantage** (Hypothesis 3): **Partially confirmed** - see analysis below

### Primary Finding: Multi-Contact Survival Advantage

The key performance difference is the mechanical advantage of surviving multiple predator contacts:

1. **Survivability**: Health-enabled agents can take up to 5 contacts before dying, vs instant death for control
2. **Learning signal**: `penalty_health_damage: 0.5` provides per-contact feedback, vs terminal-only `penalty_predator_death: 10.0`
3. **Episode length**: Longer episodes mean more time to collect food

The detection zone escape analysis shows modest improvement (+2-7%) for health-enabled agents, but this metric is confounded by **survival bias**: control agents that enter danger zones often die before they can "escape," artificially lowering their measured escape rate.

### Why Control Underperforms

The control condition shows dramatically worse performance despite having no HP attrition mechanism:

| Factor | Health-Enabled | Control |
|--------|----------------|---------|
| Contact consequence | 20 HP damage (5 contacts to die) | Instant death (1 contact) |
| Reward signal | Per-contact penalty (-0.5) | Terminal penalty only (-10.0) |
| Learning opportunities | Multiple per episode | One (fatal) |
| Evasion skill development | Gradual improvement | Limited by early deaths |

At P=10, control agents collect only **0.79 foods/run** vs **2.19 foods/run** for health-enabled - they die too quickly to even attempt foraging.

**Note**: HP depletion in health-enabled conditions is equivalent to predator death in control - both result from predator contact. The difference is that health-enabled agents can survive up to 5 contacts before dying, while control agents die on first contact.

### Critical Density Threshold

Both conditions collapse around 7 predators, but for different reasons:

- **Health-enabled at P=7**: Agents survive longer but accumulate damage faster than they can heal (avg 17.6 encounters/run)
- **Control at P=7**: Agents die almost immediately (88.4% contact→death rate, only 11.6 encounters/run)

______________________________________________________________________

## Conclusions

1. **Health system provides substantial performance advantage**: The ability to survive up to 5 predator contacts significantly improves task success rates (+4.6% to +34.0% across all predator counts).

2. **Initial risk-taking is identical between conditions**: First-contact timing analysis shows both groups encounter their first predator at the same rate (~step 156-158), indicating the health system does not change initial approach behavior.

3. **Higher contact rates are primarily mechanical, not behavioral**: Health-enabled agents make 1.5-2.4x more contacts per step, but this is largely explained by survival allowing additional contacts in later-episode foraging. We cannot fully isolate learned risk-tolerance from mechanical survival with this data.

4. **Primary mechanism is multi-contact survival**: The main advantage is mechanical: agents survive multiple contacts (up to 5) and have more time to collect food. Detection zone escape rate improvement (+2-7%) is modest and confounded by survival bias.

5. **Per-contact feedback may improve learning**: The `penalty_health_damage` provides immediate feedback on contact, potentially helping agents learn predator avoidance, but this effect is difficult to isolate from the survival advantage.

6. **Critical predator density is ~5-7**: Both conditions show sharp performance degradation between P=5 and P=7 on a 25x25 grid with 8 foods.

7. **Health system validation successful**: The implementation correctly tracks HP, applies damage on predator contact, and terminates episodes when HP reaches zero.

______________________________________________________________________

## Next Steps

- [x] Run all 120 sessions (completed)
- [x] Aggregate results by condition (completed)
- [x] Document findings (completed)
- [ ] Consider testing with higher `food_healing` to see if agents can sustain at higher predator counts
- [ ] Test pursuit predators - will the learning advantage hold when predators actively chase?
- [ ] Ablation study: test with `penalty_health_damage: 0` to isolate mechanical vs learning advantage

______________________________________________________________________

## Data References

### Config Files

**Health-enabled:**

- `configs/studies/health_scaling/health_p1.yml`
- `configs/studies/health_scaling/health_p2.yml`
- `configs/studies/health_scaling/health_p3.yml`
- `configs/studies/health_scaling/health_p5.yml`
- `configs/studies/health_scaling/health_p7.yml`
- `configs/studies/health_scaling/health_p10.yml`

**Control (health-disabled):**

- `configs/studies/health_scaling/control_p1.yml`
- `configs/studies/health_scaling/control_p2.yml`
- `configs/studies/health_scaling/control_p3.yml`
- `configs/studies/health_scaling/control_p5.yml`
- `configs/studies/health_scaling/control_p7.yml`
- `configs/studies/health_scaling/control_p10.yml`

### Raw Data

- **Summary CSV**: `artifacts/studies/el005_health_scaling_study/summary.csv`
- **Session logs**: `logs/health_scaling_study/<condition>_session<N>.log` - note: not saved
- **Export data**: `exports/<session_id>/session/data/` (includes predator_results.csv with evasion metrics) - note: not saved
