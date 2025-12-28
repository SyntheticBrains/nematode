# Legacy Benchmarks

This folder contains benchmark results from before the reproducibility infrastructure
was implemented (pre-Phase 0 completion).

## Important Notes

- These benchmarks **lack per-run seed tracking** and cannot be reproduced exactly
- They will be **removed** once re-run with the new tracking system
- Do not use these as reference for new submissions

## Migration Status

| Benchmark | Status | New Location |
|-----------|--------|--------------|
| dynamic_predator_small/classical | Pending re-run | - |
| dynamic_predator_small/quantum | Pending re-run | - |
| dynamic_small/classical | Pending re-run | - |
| dynamic_small/quantum | Pending re-run | - |
| static_maze/classical | Pending re-run | - |
| static_maze/quantum | Pending re-run | - |

## Re-running Legacy Benchmarks

To re-run a legacy benchmark with full reproducibility:

1. Use the original config file from the experiment exports
2. Run 10+ independent sessions with the new seeding infrastructure
3. Submit using `benchmark_submit.py` which will:
   - Validate seed uniqueness across all sessions
   - Move experiments to `artifacts/experiments/`
   - Create aggregate benchmark in new format

## Legacy Leaderboards

### Static Maze

#### Quantum Architectures

| Brain | Score | Success% | Steps | Converge@Run | Stability | Contributor | Date |
|---|---|---|---|---|---|---|---|
| ✓ modular | 0.980 | 100% | 34 | 20 | 0.000 | @chrisjz | 2025-11-29 |
| ✓ modular | 0.960 | 100% | 32 | 20 | 0.000 | @chrisjz | 2025-11-23 |

#### Classical Architectures

| Brain | Score | Success% | Steps | Converge@Run | Stability | Contributor | Date |
|---|---|---|---|---|---|---|---|
| ✓ mlp | 0.960 | 100% | 24 | 20 | 0.000 | @chrisjz | 2025-11-23 |
| ✓ spiking | 0.932 | 100% | 67 | 34 | 0.000 | @chrisjz | 2025-12-21 |
| ✓ spiking | 0.896 | 100% | 79 | 52 | 0.000 | @chrisjz | 2025-12-19 |

### Dynamic Small (≤20x20)

#### Quantum Architectures

| Brain | Score | Success% | Steps | Converge@Run | Stability | Contributor | Date |
|---|---|---|---|---|---|---|---|
| ✓ modular | 0.762 | 100% | 207 | 20 | 0.000 | @chrisjz | 2025-12-14 |
| ✓ modular | 0.633 | 84% | 350 | 27 | 0.136 | @chrisjz | 2025-11-30 |
| ✓ modular | 0.600 | 81% | 380 | 38 | 0.151 | @chrisjz | 2025-11-30 |
| ✓ modular | 0.598 | 80% | 317 | 43 | 0.162 | @chrisjz | 2025-11-28 |
| ✓ modular | 0.503 | 64% | 370 | 43 | 0.231 | @chrisjz | 2025-11-27 |

#### Classical Architectures

| Brain | Score | Success% | Steps | Converge@Run | Stability | Contributor | Date |
|---|---|---|---|---|---|---|---|
| ✓ ppo | 0.832 | 100% | 178 | 20 | 0.000 | @chrisjz | 2025-12-26 |
| ✓ mlp | 0.822 | 100% | 181 | 20 | 0.000 | @chrisjz | 2025-11-27 |
| ✓ mlp | 0.776 | 100% | 240 | 20 | 0.000 | @chrisjz | 2025-11-23 |
| ✓ spiking | 0.733 | 100% | 267 | 22 | 0.000 | @chrisjz | 2025-12-21 |

### Dynamic Medium (≤50x50)

#### Quantum Architectures

_No benchmarks submitted yet._

#### Classical Architectures

_No benchmarks submitted yet._

### Dynamic Large (>50x50)

#### Quantum Architectures

_No benchmarks submitted yet._

#### Classical Architectures

_No benchmarks submitted yet._

### Dynamic Predator Small (≤20x20)

#### Quantum Architectures

| Brain | Score | Success% | Steps | Converge@Run | Stability | Contributor | Date |
|---|---|---|---|---|---|---|---|
| ✓ modular | 0.675 | 95% | 224 | 29 | 0.045 | @chrisjz | 2025-12-13 |
| ✓ modular | 0.402 | 32% | 320 | 24 | 0.217 | @chrisjz | 2025-11-29 |
| ✓ modular | 0.395 | 31% | 344 | 27 | 0.215 | @chrisjz | 2025-11-27 |

#### Classical Architectures

| Brain | Score | Success% | Steps | Converge@Run | Stability | Contributor | Date |
|---|---|---|---|---|---|---|---|
| ✓ ppo | 0.781 | 93% | 176 | 24 | 0.064 | @chrisjz | 2025-12-27 |
| ✓ mlp | 0.740 | 92% | 199 | 30 | 0.076 | @chrisjz | 2025-11-27 |
| ✓ mlp | 0.618 | 82% | 195 | 78 | 0.148 | @chrisjz | 2025-11-23 |
| ✓ mlp | 0.587 | 87% | 192 | 70 | 0.116 | @chrisjz | 2025-11-23 |
| ✓ spiking | 0.556 | 63% | 247 | 20 | 0.234 | @chrisjz | 2025-12-22 |
| ✓ spiking | 0.390 | 32% | 326 | 20 | 0.219 | @chrisjz | 2025-12-21 |

### Dynamic Predator Medium (≤50x50)

#### Quantum Architectures

_No benchmarks submitted yet._

#### Classical Architectures

_No benchmarks submitted yet._

### Dynamic Predator Large (>50x50)

#### Quantum Architectures

_No benchmarks submitted yet._

#### Classical Architectures

_No benchmarks submitted yet._
