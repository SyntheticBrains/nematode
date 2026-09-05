# 039 — Runtime Acceleration Audit: GPU, Neural Engine, and Where the Time Actually Goes

**Date**: 2026-09-05
**Change**: `add-parallel-campaign-runner`
**Verdict**: GPU **rejected on measurement** (2.2–3.6× slower end-to-end); Neural Engine **unreachable**
(no PyTorch backend); the win is **process parallelism** — 7.29× at 16 workers, with zero numerical
change. Closes [Logbook 037](037-phase6a-synthesis.md)'s open recommendation to settle the
env-vectorisation-vs-GPU question before committing 6b compute.

## Objective

Determine whether the simulation can exploit the development machine's GPU (40-core) or Neural
Engine (16-core), and where training wall-clock actually goes. Logbook 037 recommended settling
this before Phase 6b, asserting that "the binding constraint is environment throughput
(vectorisation), not the GPU" — an unmeasured claim at the time.

## Hypothesis

037's assertion was expected to hold, because the policy networks are small and the rollout is
sequential. The specific quantitative predictions to test: (i) accelerator dispatch overhead
exceeds per-step compute, so the GPU loses; (ii) the environment's per-source field superposition
dominates the profile.

Both were confirmed, and a third fact — that only one of eighteen cores is in use — turned out to
matter more than either.

## Method

Machine: Apple M5 Max, 18 cores (6 Super + 12 Performance), 40-core GPU, 16-core Neural Engine,
128 GB unified memory; torch 2.13.0, MPS available. Workload: the C3 integrated cell
(`foraging_predator_thermal/mlpppo_small_continuous2d_combined_klinotaxis.yml`), headless.

1. **Profile**: `cProfile` over 10 episodes (9,113 steps), ranked by cumulative and internal time.
2. **Device micro-benchmarks**: `scripts/benchmarks/bench_device_backends.py` — per-step and
   batched forward, PPO-shaped forward+backward, connectome-scale tensors, and a large-net control
   row that distinguishes "wrong shapes" from "broken device".
3. **Device end-to-end**: full simulations with the torch device forced to CPU and to MPS.
4. **Thread sensitivity**: identical runs at BLAS thread counts 1, 3, 6.
5. **Parallel scaling**: `scripts/benchmarks/bench_campaign_parallelism.py` — 16 seeds × 20
   episodes, sequential baseline against worker counts 2–18.
6. **Vectorisation headroom**: the food-gradient superposition transcribed and re-implemented with
   array operations, timed and diffed against the scalar loop.

## Results

### Profile — the time is in sensing, not learning

| component | share of episode time |
|---|---|
| `_create_brain_params` (sensory) | 43% (2.86s / 6.70s) |
| `run_brain` (policy forward) | 20% (1.32s) |
| `learn` (PPO update) | 10% (0.67s) |

Hottest leaves, per **10 episodes**: `field_magnitude` **2,588,092 calls**;
`_food_field_magnitude` **2,460,510**; `fick_length` **2,460,510** — scalar `np.exp`/`np.cos` on
Python floats inside a 45-source loop. Roughly 1 µs of NumPy dispatch per ~5 ns of arithmetic.

The mlpppo actor is `13 → 64 → 64 → 2` — **5,186 parameters**, evaluated at **batch 1**.

### GPU — loses on every shape the project actually runs

| workload | CPU | MPS | |
|---|---|---|---|
| policy forward, batch=1 (per-step) | 9.9 µs | 74.3 µs | 0.13× |
| policy forward, batch=128 | 26.6 µs | 73.0 µs | 0.36× |
| policy fwd+bwd, batch=128 (PPO update) | 181.3 µs | 487.7 µs | 0.37× |
| connectome-scale forward, batch=1 | 12.5 µs | 71.8 µs | 0.17× |
| connectome-scale forward, batch=128 | 150.5 µs | 89.8 µs | **1.68×** |
| connectome-scale fwd+bwd, batch=128 | 702.0 µs | 494.4 µs | **1.42×** |
| **control: fwd+bwd 1024², batch=512** | 4836.3 µs | 747.5 µs | **6.47×** |

End-to-end, real simulations: **mlpppo 5.73s → 12.41s (2.2× slower)**; **connectome 2.78s →
10.13s (3.6× slower)**.

MPS shows a **~70 µs floor** independent of workload size — visible as the near-constant 71–74 µs
across four unrelated shapes. Per-step compute is ~10 µs, so the GPU waits seven times longer to be
asked than the work takes.

### Neural Engine — no path exists

`torch.backends` exposes no ANE member; `coremltools`, `mlx`, and `executorch` are absent. The ANE
is reachable only through CoreML, which targets inference, not training.

### Threads — irrelevant

10 episodes at BLAS threads 1 / 3 / 6: **6.48s / 6.51s / 6.61s**. Interpreter-bound, not BLAS-bound.

### Parallelism — the actual win

Every run measured sat at exactly **100% CPU**: one core of eighteen. 16 seeds × 20 episodes:

| workers | wall | speedup | efficiency |
|---|---|---|---|
| 1 | 78.7s | 1.00× | 100% |
| 2 | 41.5s | 1.90× | 95% |
| 4 | 24.0s | 3.28× | 82% |
| 6 | 19.4s | 4.06× | 68% |
| 8 | 15.4s | 5.10× | 64% |
| 12 | 14.9s | 5.28× | 44% |
| 16 | **10.8s** | **7.29×** | 46% |
| 18 | 11.8s | 6.69× | 37% |

### Vectorisation — real, and deliberately not taken

Scalar loop **32.66 µs** → vectorised **4.00 µs** = **8.2×** on the hot function; ~1.4× overall by
Amdahl. Max abs difference **2.109e-15** — **not byte-identical**.

## Analysis

**1. The GPU result is about shape, not hardware, and the control row proves it.** A 6.47× speedup
on a 1024² network at batch 512 rules out a broken device or driver. The losing rows lose because
5,186 parameters at batch 1 cannot amortise a 70 µs dispatch. MPS wins exactly one workload —
connectome-scale batched — and that path is swamped in practice: roughly 9,113 batch-1 rollout
steps per episode against about four batched updates. Hybrid placement would not rescue it, since
transfers cost more than the work.

**2. 037's assertion is confirmed and now quantified.** Environment throughput is indeed the
binding constraint, not the GPU. What was an inference is now two measured numbers: the GPU is
2.2–3.6× *negative*, and vectorisation is worth 8.2× on the hot function.

**3. Capability was never the limit; occupancy was.** The most consequential finding needed no new
hardware. Seventeen of eighteen cores were idle throughout every measurement, and campaign scripts
launch seeds in sequential bash loops. The efficiency knee at 6 workers coincides exactly with the
6 Super cores; the 12 Performance cores keep adding throughput at lower per-core return until 18,
where wall-clock **regresses** as workers contend with each other and the OS.

**4. Vectorisation is correct but mistimed.** The trig identity `cos(atan2(dy,dx)) ≡ dx/dist` is
exact; the 2.1e-15 divergence comes from pairwise summation reordering an accumulation the Python
loop performs sequentially. Over 2400 steps of chaotic closed-loop dynamics that separates
trajectories. [Logbook 038](038-state-dependent-std-gate.md)'s Amendment A froze the L4 substrate
mode-off on the reasoning that *nothing changed underneath Logbook 029*, which is what keeps the
32-run re-baseline descoped. Changing environment arithmetic now would falsify that premise
mid-phase for 1.4×.

## Conclusions

1. **GPU: no.** Rejected on end-to-end measurement, not intuition. Available as `--device mps` so
   the finding stays reproducible, but never the default.
2. **Neural Engine: unreachable.** A platform constraint. Recorded so it need not be re-derived.
3. **Parallelism: adopted.** `scripts/run_campaign.py` runs configs × seeds as isolated
   subprocesses of the standard entry point, each given byte-for-byte the command a person would
   type. **7.29× at 16 workers with no numerical change** — verified by a test asserting a
   campaign-launched run matches a hand-launched one.
4. **Vectorisation: deferred, not rejected.** Recorded with its measurement for a future
   pre-registered change at a natural re-baseline boundary.
5. **A real bug closed.** `--device gpu` mapped unconditionally to CUDA and crashed on Apple
   silicon with a raw `AssertionError`, despite being an advertised CLI choice. Device selection is
   now validated against both availability and brain family — `AerSimulator(device="MPS")` is
   accepted *without raising*, so an unchecked selection would have recorded `aer_simulator_mps` in
   experiment metadata as though it were a real backend.

## Next Steps

- The Phase 7 L4 panel's n ≥ 8 arms launch through the campaign runner (noted in `phase7-tracking`).
- Phase 6b's throughput question is now answered on the GPU side; if 6b proceeds, the
  vectorisation decision is the remaining half, and it needs a re-baseline boundary.
- A future vectorisation change should carry a frozen-reference equivalence test bounding the
  divergence and a dated note that the substrate's arithmetic changed.

## Data References

- Benchmarks: `scripts/benchmarks/bench_device_backends.py`,
  `scripts/benchmarks/bench_campaign_parallelism.py` — both reproduce the tables above on any host,
  skipping absent devices.
- Runner: `scripts/run_campaign.py`. Equivalence and thread-invariance evidence:
  `tests/quantumnematode_tests/campaigns/test_campaign_run_equivalence.py` (marked `slow`) and
  `test_thread_invariance.py`.
- Config under test: `configs/scenarios/foraging_predator_thermal/mlpppo_small_continuous2d_combined_klinotaxis.yml`.
- Profile: `cProfile` over 10 episodes at seed 42, headless. No experiment-tracked sessions — this
  is a performance audit, not a behavioural result, and no simulation output was used as evidence.
