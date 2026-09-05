# Design — parallel campaign runner + honest accelerator selection

## Context

All measurements below were taken on the development machine: Apple M5 Max, 18 cores (6 Super + 12 Performance), 40-core GPU, 16-core Neural Engine, 128 GB unified memory, torch 2.13.0 with MPS available. Numbers are hardware-specific and live here and in Logbook 039, deliberately **not** in the spec deltas, which stay machine-independent.

## Decision 1 — Subprocess isolation, not an in-process pool

**Chosen**: each run is a `subprocess` invocation of `scripts/run_simulation.py` with an explicit `--seed`.

**Rejected**: a `multiprocessing.Pool` over an importable `run_once()` refactor, which is what `evolution/loop.py` does.

The evolution precedent is good and its worker-init pattern is reused, but the two cases differ in what they must protect. Evolution's pool evaluates genomes that only ever existed inside that pool. Here the runner replaces commands a person runs by hand today, and the whole value proposition is that **the results are the same ones**. Subprocess isolation makes that a structural property rather than a claim needing a byte-equivalence test: the child receives the same argv, imports the same modules in the same order, seeds the same RNGs, and writes through the same code. A pool would additionally require `run_simulation.py` to become importable and side-effect-free at import time — a refactor of the most side-effect-heavy script in the repo, taken on for no measured gain, since spawn overhead is 0.79s against runs measured in minutes to hours.

The cost is real and accepted: no shared memory, no in-process aggregation, and per-run startup paid W times. At campaign scale (thousands of episodes per run) that is well under 1% of wall-clock.

## Decision 2 — Worker default reserves headroom

Measured, 16 seeds × 20 episodes, sequential baseline 89.4s:

| workers | wall | speedup | efficiency |
|---|---|---|---|
| 2 | 48.1s | 1.86× | 93% |
| 4 | 27.4s | 3.26× | 82% |
| 6 | 25.1s | 3.56× | 59% |
| 8 | 20.1s | 4.45× | 56% |
| 12 | 17.7s | 5.05× | 42% |
| 16 | **15.7s** | **5.70×** | 36% |
| 18 | 15.8s | 5.67× | 31% |

Efficiency has a knee at 6, matching the 6 Super cores; the 12 Performance cores continue to add throughput at lower per-core return. Wall-clock keeps improving to 16 and then **regresses** at 18, where workers contend with the OS and each other.

**Default: `cpu_count - 2`** (16 here). This tracks the measured optimum, degrades sensibly on other machines, and leaves the machine usable while a campaign runs — a real consideration when a panel occupies the laptop for hours. Efficiency-minded users wanting to run other work can pass a lower `--workers`; the flag is explicit precisely because the right point on this curve is a judgement about the machine's other duties, not a fact about the code.

## Decision 3 — Pin BLAS threads to 1 in children, having proved it inert

With W workers each defaulting to 6 BLAS threads, a 16-worker campaign would request 96 threads on 18 cores. `evolution/loop.py:_init_worker` already sets `torch.set_num_threads(1)` for exactly this reason.

Thread count is **not** numerically inert in general — parallel reductions can reorder float accumulation — so this was verified rather than assumed. At threads 1, 6, and 16, SHA-256 of the raw bytes was identical for: actor forward at batch 1, actor forward at batch 128, concatenated gradients after backward, and a 302×302 batched matmul. The shapes are small enough that torch takes single-threaded paths regardless.

Independently, thread count does not help: 10 episodes at 1/3/6 threads ran in 6.48s/6.51s/6.61s. The workload is Python-interpreter-bound, so pinning costs nothing and prevents oversubscription.

## Decision 4 — `gpu` stays CUDA; `mps` is a new explicit member

Three options were considered for the crash (`AssertionError: Torch not compiled with CUDA enabled` from `--device gpu`):

1. **Map `gpu` → MPS on Apple silicon.** Rejected. It would silently hand the user a **2.2–3.6× slowdown** under a name that means "make this faster", and would make the `gpu` flag mean different backends on different machines — poison for reproducibility in a repo whose comparisons span months.
2. **Remove `gpu` from choices on non-CUDA builds.** Rejected: configs and scripts are shared across machines, and a flag vanishing by platform is harder to diagnose than a flag that explains itself.
3. **Chosen**: add `MPS` as its own member, keep `GPU` → `"cuda"`, and validate availability at startup with an error naming the platform-appropriate alternative.

This keeps device names stable across machines, makes the Apple GPU reachable and therefore measurable by anyone reproducing the benchmark, and converts a mid-construction torch assertion into an actionable startup message.

MPS is added as a **capability, not a recommendation**. The docs record that CPU is the right default for current model sizes, with the measurements and the conditions under which that would change.

### Decision 4a — the check must be brain-family-aware, not just availability-aware

Spec review caught that availability alone is insufficient. `DeviceType` is shared by two unrelated backends: PyTorch tensor placement for classical/spiking brains, and the Qiskit simulator selector for quantum brains, which read it as `self.device.value.upper()`. `AerSimulator(device="MPS")` was tested and is **accepted without raising** — and in Qiskit, MPS already means *Matrix Product State*, a simulation method. An unchecked selection would therefore construct a meaningless backend and propagate `aer_simulator_mps` into experiment metadata as a legitimate-looking backend name: a silent-wrong-data path, the worst failure mode available here.

Validation reads the plugin registry's `families` tuple, which already tags 11 brains `quantum` (one as `("quantum", "classical")`). Hybrids are treated as quantum, because their device value still reaches the simulator. Deriving from registry metadata rather than a device/brain matrix means a new architecture inherits correct validation from its registration alone.

Scope is held tight: this validates only the accelerator this change introduces. The pre-existing behaviour whereby a non-quantum brain selecting `qpu` falls back to CPU placement is documented in the enum and left alone — changing it would be an unrelated behaviour change smuggled into a performance change.

### Decision 4b — availability is checked for every brain (corrected at branch review)

An earlier revision of this design exempted `quantum`-family brains from the *availability* check, reasoning that for them `gpu` selects Aer's GPU device rather than a torch device. Branch review showed that reasoning rests on a false premise: **seven of the eleven** quantum-family brains — `equivariantquantum`, `hybridquantum`, `hybridquantumcortex`, `qliflstm`, `qrc`, `qsnnppo`, `qsnnreinforce` — call `to_torch_device_str()` and allocate torch actors and critics at construction. Exempting them reinstated precisely the raw `Torch not compiled with CUDA enabled` assertion this change exists to remove.

The family tag cannot distinguish the two cases, because it does not encode whether a brain uses torch. Rather than infer it, the availability check now applies to every brain. The cost is a host with a GPU-enabled Aer and a deliberately CPU-only torch build, which is refused with a clear message; that configuration requires a CUDA GPU plus a non-default torch wheel, and the documented GPU install path (`--extra gpu --extra torch`) does not produce it. The alternative cost — an unhandled assertion for seven architectures — is worse and is the failure this requirement was written against.

### Decision 4c — multi-agent runs validate every agent

`_run_multi_agent` is dispatched before the single-brain check in `main` and constructs one brain per agent, each free to name a different architecture. A per-agent loop therefore runs immediately after the agent configs resolve and before any brain is built, reporting which agent is at fault. Without it, a heterogeneous config could carry a brain the device cannot serve straight past validation.

## Decision 5 — The env hot loop is left alone, on purpose

The largest single optimisation available is not taken. Vectorising the food-gradient superposition measures 32.66 µs → 4.00 µs (**8.2×**) on a function reached ~2.5M times per 10 episodes, worth roughly 1.4× overall by Amdahl.

It is deferred because it is **not byte-identical**: max abs difference 2.109e-15. The trig identity `cos(atan2(dy,dx)) ≡ dx/dist` is exact; the divergence comes from NumPy's pairwise summation reordering accumulation that the Python loop performs sequentially. Over 2400 steps of chaotic closed-loop dynamics, 1e-15 is not a rounding detail — trajectories separate.

The blocking consideration is timing rather than merit. Logbook 038's Amendment A froze the L4 substrate mode-off four days ago, and the justification for descoping the 32-run re-baseline was explicitly that *nothing changed underneath Logbook 029*. Changing environment float arithmetic would falsify that premise mid-phase and could pull those runs back into scope — a cost far exceeding 1.4×.

Recorded for a future pre-registered change at a natural re-baseline boundary, where the substrate is being re-measured anyway. If taken then, the house pattern applies: a frozen-reference equivalence test bounding the divergence, and a dated note that the substrate's arithmetic changed.

## Decision 6 — The Neural Engine is out of reach, and that is a platform fact

Not a gap in this codebase. PyTorch exposes no ANE backend (`torch.backends` has no such member). The ANE is reachable only through CoreML; `coremltools`, `mlx`, and `executorch` are all absent from the environment. CoreML is an inference-oriented deployment path with no general training story, while this workload is **training** with batch-1 custom ops and per-step Python control flow.

Even granting a hypothetical training path, the same arithmetic that defeats MPS applies with more force: the ANE is a throughput engine for large batched convolution and matmul, and the per-step workload here is a 5,186-parameter MLP at batch 1. Recorded so the question does not need re-deriving.

## Risks

- **Reduced-fidelity results under load.** Concurrent runs contend for memory bandwidth, so a run inside a 16-wide campaign takes longer in wall-clock than the same run alone. Results are unaffected; only per-run timing telemetry is, which matters if wall-clock is ever a reported metric. Noted in the docs.
- **Disk and memory pressure.** W concurrent runs multiply artefact I/O and resident memory. 128 GB is ample here; the `--workers` flag is the control on smaller machines.
- **Log interleaving.** Addressed by per-run log files rather than multiplexed stdout.
