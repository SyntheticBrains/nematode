# 009 Supporting Data: Temporal Sensing Evaluation Details

## Post-Convergence Summary (L100)

| Environment | Oracle L100 | GRU Derivative L100 | GRU Temporal L100 |
|---|---|---|---|
| Foraging (small) | — | 100% | 99% |
| Pursuit pred (small) | 70% | 77% | 78% |
| Pursuit pred (large+thermo) | 95% | 88% | 95% |
| Stationary pred (large+thermo) | 81% | 73% | 70% |

L100 = mean success rate over last 100 episodes across 4 seeds.

## Per-Seed Results

### Small Foraging (20×20, 1000 episodes)

#### GRU Derivative

| Seed | Success | Avg Food |
|------|---------|----------|
| 42 | 90.6% | 9.47 |
| 43 | 95.3% | 9.73 |
| 44 | 82.6% | 9.06 |
| 45 | 94.5% | 9.68 |
| **Mean** | **90.8%** | **9.48** |

#### GRU Temporal

| Seed | Success | Avg Food |
|------|---------|----------|
| 42 | 83.7% | 9.21 |
| 43 | 66.6% | 8.14 |
| 44 | 68.4% | 8.12 |
| 45 | 52.7% | 7.41 |
| **Mean** | **67.9%** | **8.22** |

#### LSTM Derivative (Ablation Control)

| Seed | Success | Avg Food |
|------|---------|----------|
| 42 | 79.6% | 9.03 |
| 43 | 78.0% | 8.65 |
| 44 | 86.9% | 9.22 |
| 45 | 80.6% | 8.93 |
| **Mean** | **81.3%** | **8.96** |

### Small Pursuit Predators (20×20, steep gradients 4.0)

#### Oracle (MLP PPO, 1000 episodes)

| Seed | Success | L1000 Food | Evasion |
|------|---------|-----------|---------|
| 42 | 59.7% | 7.72 | 71% |
| 43 | 63.0% | 8.00 | 70% |
| 44 | 68.1% | 8.29 | 72% |
| 45 | 61.1% | 7.89 | 74% |
| **Mean** | **63.0%** | **7.97** | **72%** |

#### GRU Derivative (8000 episodes)

| Seed | Success | L1000 Success | L1000 Food | Evasion |
|------|---------|-------------|-----------|---------|
| 42 | 65.5% | 81% | 9.16 | 92% |
| 43 | 62.5% | 77% | 8.89 | 91% |
| 44 | 62.7% | 72% | 8.68 | 91% |
| 45 | 58.0% | 76% | 8.94 | 91% |
| **Mean** | **62.2%** | **77%** | **8.92** | **91%** |

#### GRU Temporal (8000 episodes)

| Seed | Success | L1000 Success | L1000 Food | Evasion |
|------|---------|-------------|-----------|---------|
| 42 | 56.6% | 84% | 9.32 | 93% |
| 43 | 49.7% | 78% | 8.96 | 92% |
| 44 | 34.2% | 79% | 9.16 | 94% |
| 45 | 45.5% | 73% | 8.78 | 92% |
| **Mean** | **46.5%** | **78%** | **9.05** | **92%** |

### Large Pursuit Predators (100×100 + Thermotaxis)

#### Oracle (MLP PPO, 1000 episodes)

| Seed | Success | L500 Food | Evasion |
|------|---------|----------|---------|
| 42 | 82.0% | 22.5 | 92% |
| 43 | 87.9% | 23.3 | 92% |
| 44 | 87.7% | 23.0 | 92% |
| 45 | 86.2% | 23.0 | 92% |
| **Mean** | **86.0%** | **22.8** | **92%** |

#### GRU Derivative (4000 episodes)

| Seed | Success | L500 Success | L500 Food | Evasion |
|------|---------|-------------|----------|---------|
| 42 | 65.0% | 88% | 23.8 | 94% |
| 43 | 62.2% | 89% | 23.8 | 94% |
| 44 | 64.0% | 88% | 23.5 | 95% |
| 45 | 59.4% | 88% | 24.0 | 95% |
| **Mean** | **62.6%** | **88%** | **23.8** | **94%** |

#### GRU Temporal (8000 episodes)

| Seed | Success | L500 Success | L500 Food | Evasion |
|------|---------|-------------|----------|---------|
| 42 | 67.0% | 94% | 24.2 | 94% |
| 43 | 64.2% | 93% | 24.3 | 93% |
| 44 | 58.0% | 97% | 24.6 | 94% |
| 45 | 61.6% | 92% | 24.0 | 92% |
| **Mean** | **62.7%** | **94%** | **24.3** | **93%** |

### Large Stationary Predators (100×100 + Thermotaxis)

#### Oracle (MLP PPO, 1000 episodes)

| Seed | Success | L500 Success | Evasion |
|------|---------|-------------|---------|
| 42 | 69.6% | 82% | 94% |
| 43 | 71.3% | 84% | 96% |
| 44 | 63.4% | 74% | 92% |
| 45 | 64.3% | 77% | 91% |
| **Mean** | **67.1%** | **79%** | **93%** |

#### GRU Derivative (4000 episodes)

| Seed | Success | L500 Success | L500 Food | Evasion |
|------|---------|-------------|----------|---------|
| 42 | 52.2% | 71% | 21.7 | 93% |
| 43 | 55.9% | 71% | 21.3 | 93% |
| 44 | 55.1% | 79% | 22.7 | 94% |
| 45 | 55.5% | 73% | 21.8 | 95% |
| **Mean** | **54.7%** | **74%** | **21.9** | **94%** |

#### GRU Temporal (12000 episodes)

| Seed | Success | L500 Success | L500 Food | Evasion |
|------|---------|-------------|----------|---------|
| 42 | 46.6% | 75% | 21.9 | 94% |
| 43 | 39.0% | 72% | 21.2 | 93% |
| 44 | 38.9% | 74% | 22.0 | 95% |
| 45 | 53.5% | 74% | 21.8 | 94% |
| **Mean** | **44.5%** | **74%** | **21.7** | **94%** |

______________________________________________________________________

## BPTT Chunk Length Scaling

### Small Pursuit Predators — Derivative Mode

| Chunk | Mean Success | L1000 Success | Episodes |
|---|---|---|---|
| 16 | 16.1% | 26% | 4000 |
| 32 | 36.6% | 52% | 8000 |
| 48 | 44.2% | 62% | 8000 |
| 64 | 54.1% | 70% | 8000 |
| 96 | 58.7% | 74% | 8000 |

### Small Pursuit Predators — Temporal Mode

| Chunk | Mean Success | L1000 Success | Episodes |
|---|---|---|---|
| 48 | 16.7% | 28% | 8000 |
| 64 | 23.3% | 43% | 8000 |
| 96 | 40.1% | 73% | 8000 |

### Stationary Predators — Temporal Mode (chunk sensitivity)

| Chunk | Mean Success | L500 Success | Note |
|---|---|---|---|
| 48 | 0.1% | 1% | Too short — failed |
| **64** | **18.4%** | **54%** | **Best for stationary** |
| 128 | 0.0% | 0% | Too few chunks per buffer |

______________________________________________________________________

## GRU vs LSTM Ablation Details

### Foraging (Small, 1000 episodes)

| Metric | GRU Derivative | LSTM Derivative | GRU Advantage |
|---|---|---|---|
| Mean success | 90.8% | 81.3% | +9.5pp |
| L100 success | 100% | 100% | 0pp (both perfect) |
| Std across seeds | 5.5% | 3.9% | LSTM more consistent |
| Parameters | Fewer (no cell state) | More | GRU simpler |

### Pursuit Predators (Small, 8000 episodes, chunk=96)

| Metric | GRU Deriv | LSTM Deriv | GRU Temp | LSTM Temp |
|---|---|---|---|---|
| Mean success | 62.2% | 58.7% | 46.5% | 40.1% |
| L1000 success | 77% | 74% | 78% | 73% |
| Evasion rate | 91% | 91% | 92% | 93% |

### Large Pursuit (4000/8000 episodes)

| Metric | GRU Deriv | LSTM Deriv | GRU Temp | LSTM Temp |
|---|---|---|---|---|
| Mean success | 62.6% | 57.4% | 62.7% | 51.1% |
| L500 success | 88% | 90% | 94% | 91% |

______________________________________________________________________

## Learning Curve Analysis

### Breakthrough Transition Pattern (Temporal, Large Stationary)

All temporal seeds on stationary predators exhibit a characteristic pattern:

- **Plateau phase** (0-4000 episodes): Near-zero success, ~2-3 avg foods, agent explores randomly
- **Breakthrough** (4000-6000 episodes): Sudden jump to 15-20+ avg foods within ~1000 episodes
- **Convergence** (6000+ episodes): Steady improvement to 70-75% L500

Example learning curve (best seed):

```text
Eps    1- 1000: food= 1.4, success=  0%
Eps 1001- 2000: food= 1.5, success=  0%
Eps 2001- 3000: food= 2.5, success=  0%
Eps 3001- 4000: food= 3.3, success=  0%
Eps 4001- 5000: food=16.2, success= 37%  ← breakthrough
Eps 5001- 6000: food=20.1, success= 60%
Eps 6001- 7000: food=20.9, success= 67%
Eps 7001- 8000: food=21.5, success= 73%
```

The breakthrough transition occurs when the GRU discovers the "move → compare concentrations → infer direction" strategy. This is an emergent skill that requires substantial random exploration before it can be learned.

### Convergence Speed Comparison

| Config | Oracle Converge | Derivative Converge | Temporal Converge |
|---|---|---|---|
| Foraging (small) | ~100 eps | ~500 eps | ~800 eps |
| Pursuit (small) | ~300 eps | ~4000 eps | ~6000 eps |
| Pursuit (large) | ~300 eps | ~3000 eps | ~6000 eps |
| Stationary (large) | ~300 eps | ~3000 eps | ~8000 eps |

______________________________________________________________________

## Key Hyperparameters

### Final Best Configuration (GRU PPO)

| Parameter | Foraging | Pursuit | Stationary | Rationale |
|---|---|---|---|---|
| rnn_type | gru | gru | gru | Outperforms LSTM everywhere |
| lstm_hidden_dim | 64 | 64 | 64 | Sufficient; 128 hurts |
| bptt_chunk_length | 16 | 96-128 | 64 | Matches task temporal scale |
| actor_lr | 0.0003 | 0.0003 | 0.0003 | Conservative prevents collapse |
| rollout_buffer_size | 512 | 512 | 512 | Small buffer for fresh updates |
| num_epochs | 4 | 4 | 4 | Fewer epochs prevent overfitting |
| entropy_coef_end | 0.005 | 0.01 | 0.01 | Higher floor for stochastic tasks |
| lr_decay_episodes | 1500 | 4000 | 8000 | Matches training duration |
| gradient_decay | 8.0 | 4.0 | 4.0 | Steeper for predator environments |

### Hyperparameters That Don't Help

| Parameter | Tested | Result |
|---|---|---|
| lstm_hidden_dim=128 | Foraging, pursuit | Worse (overfitting) |
| rollout_buffer_size=1024 | All environments | Worse (stale experience) |
| num_epochs=8 | Pursuit | Catastrophic failure |
| gae_lambda=0.99 | Pursuit | Worse (high variance advantages) |
| STAM sensory module | Foraging | No benefit (LSTM memory sufficient) |
| Curriculum learning | Pursuit | Inconsistent benefit |

______________________________________________________________________

## Per-Seed L100 Post-Convergence Results

L100 = success rate over the last 100 episodes of training. This is the primary convergence metric used in logbooks 007 and 008 for consistency.

### Foraging Small (20×20)

| Seed | GRU Deriv L100 | GRU Temp L100 | LSTM Deriv L100 |
|------|---------------|--------------|----------------|
| 1 | 100% | 99% | 100% |
| 2 | 100% | 100% | 99% |
| 3 | 100% | 100% | 100% |
| 4 | 100% | 96% | 100% |
| **Mean** | **100%** | **99%** | **100%** |

### Pursuit Predators Small (20×20)

| Seed | Oracle L100 | GRU Deriv L100 | GRU Temp L100 |
|------|------------|---------------|--------------|
| 1 | 65% | 87% | 90% |
| 2 | 80% | 76% | 73% |
| 3 | 67% | 65% | 83% |
| 4 | 66% | 79% | 67% |
| **Mean** | **70%** | **77%** | **78%** |

### Pursuit Predators Large (100×100 + Thermotaxis)

| Seed | Oracle L100 | GRU Deriv L100 | GRU Temp L100 |
|------|------------|---------------|--------------|
| 1 | 95% | 90% | 96% |
| 2 | 97% | 84% | 94% |
| 3 | 95% | 90% | 97% |
| 4 | 92% | 87% | 94% |
| **Mean** | **95%** | **88%** | **95%** |

### Stationary Predators Large (100×100 + Thermotaxis)

| Seed | Oracle L100 | GRU Deriv L100 | GRU Temp L100 |
|------|------------|---------------|--------------|
| 1 | 80% | 71% | 72% |
| 2 | 88% | 68% | 67% |
| 3 | 76% | 86% | 68% |
| 4 | 80% | 68% | 74% |
| **Mean** | **81%** | **73%** | **70%** |
