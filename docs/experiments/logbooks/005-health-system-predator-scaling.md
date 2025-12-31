# 005: Health System Predator Scaling Study

**Status**: `active`

**Branch**: `feature/add-health-system`

**Date Started**: 2025-12-31

**Date Completed**: (pending)

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

1. **Health-enabled agents** will show lower success rates than control at high predator counts, as accumulated damage becomes a significant death cause
2. **Convergence threshold**: Agents should maintain >50% success rate up to ~5-7 predators on a 25x25 grid
3. **Death cause shift**: At low predator counts, starvation dominates; at high counts, health depletion becomes primary failure mode
4. **Control comparison**: Health-disabled runs will show higher success rates at high predator counts (no HP attrition)

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

| Predators | Health Enabled | Health Disabled | Delta |
|-----------|----------------|-----------------|-------|
| 1 | pending | pending | - |
| 2 | pending | pending | - |
| 3 | pending | pending | - |
| 5 | pending | pending | - |
| 7 | pending | pending | - |
| 10 | pending | pending | - |

### Detailed Findings

(To be filled after study completion)

#### Health-Enabled Conditions (H1-H10)

| Condition | Success Rate | Convergence | Post-Conv | Starved | HP Depleted | Max Steps |
|-----------|--------------|-------------|-----------|---------|-------------|-----------|
| H1 | - | - | - | - | - | - |
| H2 | - | - | - | - | - | - |
| H3 | - | - | - | - | - | - |
| H5 | - | - | - | - | - | - |
| H7 | - | - | - | - | - | - |
| H10 | - | - | - | - | - | - |

#### Control Conditions (C1-C10)

| Condition | Success Rate | Convergence | Post-Conv | Starved | Max Steps |
|-----------|--------------|-------------|-----------|---------|-----------|
| C1 | - | - | - | - | - |
| C2 | - | - | - | - | - |
| C3 | - | - | - | - | - |
| C5 | - | - | - | - | - |
| C7 | - | - | - | - | - |
| C10 | - | - | - | - | - |

______________________________________________________________________

## Analysis

(To be completed after study)

### Key Questions to Answer

1. At what predator count does success rate drop below 50%?
2. Does health system create a "ceiling" that control doesn't have?
3. How does death cause distribution shift with predator density?
4. Is there evidence of agents learning predator avoidance vs damage absorption?

______________________________________________________________________

## Conclusions

(To be completed after study)

______________________________________________________________________

## Next Steps

- [ ] Run all 120 sessions
- [ ] Aggregate results by condition
- [ ] Generate comparison plots
- [ ] Document findings

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

### Session IDs

(To be populated during execution)

| Condition | Session 1 | Session 2 | Session 3 | Session 4 | Session 5 | Session 6 | Session 7 | Session 8 | Session 9 | Session 10 |
|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|------------|
| H1 | - | - | - | - | - | - | - | - | - | - |
| H2 | - | - | - | - | - | - | - | - | - | - |
| H3 | - | - | - | - | - | - | - | - | - | - |
| H5 | - | - | - | - | - | - | - | - | - | - |
| H7 | - | - | - | - | - | - | - | - | - | - |
| H10 | - | - | - | - | - | - | - | - | - | - |
| C1 | - | - | - | - | - | - | - | - | - | - |
| C2 | - | - | - | - | - | - | - | - | - | - |
| C3 | - | - | - | - | - | - | - | - | - | - |
| C5 | - | - | - | - | - | - | - | - | - | - |
| C7 | - | - | - | - | - | - | - | - | - | - |
| C10 | - | - | - | - | - | - | - | - | - | - |
