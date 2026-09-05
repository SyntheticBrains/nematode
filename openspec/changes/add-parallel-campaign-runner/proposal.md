# Add a parallel campaign runner; make accelerator selection honest

## Why

Every training run on this project uses **one CPU core**. Profiling the C3 mlpppo cell (10 episodes, 9113 steps, `cProfile`) shows the process pinned at 100% CPU on an 18-core machine, and campaign scripts (`scripts/campaigns/*.sh`) launch their seeds in sequential bash `for` loops — "Runs the MLPPPO hyperparameter-evolution pilot across 4 seeds sequentially". The Phase 7 L4 panel is a 2×2 at n ≥ 8, so the next milestone is dominated by wall-clock that the hardware could already absorb.

Two things are worth fixing, and one tempting thing is deliberately **not** being fixed.

### 1. Seeds run one at a time (the win)

Paired-seed runs are embarrassingly parallel by construction: independent processes, independent RNG streams, independent output directories. `generate_session_id()` already appends a UUID suffix explicitly "to prevent collisions in parallel execution", and `evolution/loop.py` already runs a `multiprocessing.Pool` with a documented worker-init pattern. The training path simply never got the same treatment.

Measured on this machine (16 seeds × 20 episodes; sequential baseline 89.4s):

| workers | wall | speedup | efficiency |
|---|---|---|---|
| 2 | 48.1s | 1.86× | 93% |
| 4 | 27.4s | 3.26× | 82% |
| 6 | 25.1s | 3.56× | 59% |
| 8 | 20.1s | 4.45× | 56% |
| 12 | 17.7s | 5.05× | 42% |
| 16 | **15.7s** | **5.70×** | 36% |
| 18 | 15.8s | 5.67× | 31% |

At a realistic per-run length (8 seeds × 100 episodes) the same pattern holds: 123.3s → 25.9s at 8 workers (4.8×). Process startup is 0.79s, so this is genuine core contention, not spawn overhead.

### 2. `--device gpu` hard-crashes on Apple silicon (a real bug)

`DeviceType.GPU` maps unconditionally to `"cuda"`. On any machine without a CUDA build — which includes this project's primary development machine — `--device gpu` dies with a raw, unactionable assertion:

```
AssertionError: Torch not compiled with CUDA enabled
```

The CLI advertises `gpu` in its `choices` list, so this is a documented option that cannot work. Separately, Apple's GPU is reachable through PyTorch's MPS backend and is currently **unreachable** from any config or flag, which also makes the "should we use the GPU?" question unmeasurable by anyone reproducing this work.

### 3. What is deliberately not changing: the env hot loop

The profile's single largest block is sensory computation — 43% of episode time, with `field_magnitude` called **2,588,092 times** per 10 episodes and `_food_field_magnitude` **2,460,510 times**: scalar `np.exp`/`np.cos` on Python floats inside a 45-source loop, paying roughly 1 µs of NumPy dispatch for a ~5 ns FLOP. A vectorised superposition measures **8.2×** faster on that function (32.66 µs → 4.00 µs), worth ~1.4× overall.

It is not in this change, because it is **not byte-identical**: max abs difference 2.109e-15. The trigonometric rewrite (`cos(atan2(dy,dx)) ≡ dx/dist`) is exact, but pairwise summation reorders float accumulation, and 2400 chaotic steps amplify that into divergent trajectories. Logbook 038's Amendment A froze the L4 substrate mode-off **on the reasoning that "nothing changed underneath 029"**, which is what keeps the 32-run re-baseline descoped. Perturbing environment arithmetic now would falsify that premise and could pull those runs back into scope. The speedup is real and recorded here for a future pre-registered change at a natural re-baseline boundary — not taken opportunistically against a substrate that was frozen four days ago.

## What Changes

### 1. New `scripts/run_campaign.py` — bounded-concurrency campaign runner

Takes one or more `--config` paths and a `--seeds` range/list, forms the **cross product** (the shape the L4 2×2 panel needs: arms × seeds), and executes them as **subprocesses of `run_simulation.py`** with a bounded worker pool. Passthrough args after `--` reach each child unchanged.

The subprocess design is the load-bearing choice: each child receives byte-for-byte the command line a person would type today, so a campaign run is numerically indistinguishable from the sequential runs it replaces. No refactor of `run_simulation.py`, no pickling, no shared interpreter state.

Workers default to `cpu_count - 2` (16 here — the measured wall-clock optimum; 18 regressed), leaving headroom for the OS and interactive work. Children run with BLAS threads pinned to 1, following `evolution/loop.py`'s `_init_worker` precedent and **verified bit-invariant**: forward at batch 1 and 128, gradients, and a connectome-scale matmul all hash identically at 1, 6, and 16 threads.

Operationally: per-run log files, live progress with completion counts and elapsed time, a final per-run status table, non-zero exit if any run failed, `--dry-run`, and SIGINT handling that terminates children rather than orphaning them.

### 2. Honest accelerator selection

- `DeviceType` gains `MPS = "mps"`, so Apple's GPU becomes selectable and therefore measurable.
- `GPU` keeps mapping to `"cuda"` — it is not silently redirected to MPS, because MPS is **measurably slower** for this project's model sizes and a silent redirect would hand users a 2–3.6× regression under the name "gpu".
- Requesting an accelerator that this build cannot provide fails at startup with an actionable message naming the platform-appropriate alternative, instead of a torch assertion mid-brain-construction.

### 3. `scripts/benchmarks/bench_device_backends.py` — reproducible evidence

The device question should not have to be re-litigated from memory. This script reproduces the measurements below on any machine: per-step and batched forward, a full PPO-shaped forward+backward, connectome-scale tensors, and a large-net reference proving the GPU itself is healthy.

Measured here (Apple M5 Max, 40-core GPU, torch 2.13.0):

| workload | CPU | MPS | |
|---|---|---|---|
| mlpppo actor fwd, batch=1 (per-step) | 10.1 µs | 72.6 µs | 0.14× |
| mlpppo fwd+bwd, batch=128 (PPO update) | 172.2 µs | 475.5 µs | 0.36× |
| connectome-scale fwd, batch=128 | 146.6 µs | 80.7 µs | 1.82× |
| reference: fwd+bwd 1024², batch=512 | 4858.8 µs | 748.0 µs | 6.50× |

End-to-end on real simulations: mlpppo 10 episodes **5.73s → 12.41s** (2.2× slower); connectome 5 episodes **2.78s → 10.13s** (3.6× slower).

The cause is structural, not a tuning failure: MPS carries a **~70 µs fixed dispatch floor** while the entire per-step compute is ~10 µs on a **5,186-parameter** actor (13→64→64→2) at **batch size 1**, invoked ~9,113 times per episode against roughly four batched updates. The reference row shows the GPU reaching 6.5× when tensors are actually large. MPS wins exactly one row — connectome-scale batched — and that path is swamped by the batch-1 rollout.

### 4. Documentation + logbook

A usage-guide section on running campaigns and choosing worker counts; an architecture note recording, with numbers, why CPU is the default and when that would change (batched multi-agent rollouts or models two orders of magnitude larger). Logbook 039 records the profile, both device benchmarks, the parallel scaling curve, and the deferred vectorisation with its byte-identity measurement.

## Impact

- **Affected specs**: new `campaign-execution` capability; `cli-interface` gains device-selection requirements.
- **Affected code**: new `scripts/run_campaign.py`, new `scripts/benchmarks/bench_device_backends.py`, `DeviceType` in `brain/arch/dtypes.py`, device validation at CLI entry.
- **Numerical impact: none.** No change to any simulation, brain, environment, or learning-rule code path. Every existing config, weight file, and logbook comparison remains valid, and the Amendment A freeze premise is untouched.
- **Not addressed** (recorded, not scheduled): env-field vectorisation (~1.4× overall, needs a re-baseline boundary); Neural Engine, which has no PyTorch backend at all and is reachable only through CoreML as an inference-only path — a platform constraint, not a gap in this codebase.
