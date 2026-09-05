# Tasks — parallel campaign runner + honest accelerator selection

## 1. Accelerator selection

- [x] 1.1 Add `MPS = "mps"` to `DeviceType`; `to_torch_device_str()` returns `"mps"` for it, `GPU` keeps `"cuda"`, `QPU`/`CPU` unchanged. Update the enum docstring to state what each maps to and that `GPU` is CUDA-only by design (Decision 4).
- [x] 1.2 Add an availability check that raises an actionable error naming the requested device and the platform-appropriate alternative, invoked at CLI entry before brain construction. CPU takes no check.
- [x] 1.3 Add brain-family validation: reject a PyTorch-only accelerator for a brain whose registry entry carries the `quantum` family tag, reading `Registration.families` rather than a hand-maintained list (11 brains carry it, one as `("quantum", "classical")` — hybrids count as quantum because their device value still reaches the simulator). The error names the device, the brain, and the accepted devices. Without this, `--device mps` reaches `AerSimulator(device="MPS")`, which is accepted **without raising** and whose name collides with Qiskit's Matrix Product State method, so the bogus selection would be recorded as a real backend in experiment metadata.
- [x] 1.4 Wire both checks into `scripts/run_simulation.py` — verified to be the only entry point declaring `--device` — before brain construction. Confirm `--device mps` runs end-to-end on a torch brain, `--device gpu` on a non-CUDA build fails cleanly with no torch assertion, and `--device mps` on a quantum brain fails cleanly.
- [x] 1.5 Update the stale return-contract docstring on `get_device_type_string` (`experiment/system_utils.py`), which enumerates `("cpu", "gpu", "qpu")`.
- [x] 1.6 Tests: `mps` maps correctly; `gpu` still maps to `cuda`; unavailable-accelerator error names device + alternative; quantum brain + `mps` rejected; hybrid `("quantum", "classical")` brain rejected; quantum brain + `cpu`/`gpu`/`qpu` still accepted with an unchanged simulator backend string; CPU needs no check. Skip-guard any test that requires a real MPS device.

## 2. Campaign runner

- [x] 2.1 New `scripts/run_campaign.py`: repeatable `--config`, `--seeds` accepting ranges (`1-8`), comma lists, and space-separated values; cross product to the run plan; passthrough of extra args after `--`.
- [x] 2.2 Bounded concurrency at `--workers`, defaulting to `max(1, cpu_count - 2)`; each run a `subprocess` of `run_simulation.py` with explicit `--seed`.
- [x] 2.3 Child environment pins BLAS/OMP threads to 1 (Decision 3); document the verified bit-invariance at the call site as the reason this is safe.
- [x] 2.4 Per-run log files under the campaign output directory; progress reporting while running; per-run status table at the end.
- [x] 2.5 Non-zero exit if any run failed, after attempting every run; SIGINT terminates children rather than orphaning them.
- [x] 2.6 `--dry-run` prints the planned commands and starts nothing.
- [x] 2.7 Tests: cross-product planning; seed-spec parsing incl. malformed input; worker bound respected; failure surfaces in the summary and exit code; dry-run starts no process; passthrough args reach the command. Use a stub child command so the suite stays fast and hermetic. `scripts/` is not an importable package — load the runner with `importlib.util.spec_from_file_location`, the pattern already used by the campaign-script tests.

## 3. Equivalence evidence

- [x] 3.1 Test asserting a campaign-launched run and a hand-launched run of the same config/seed agree, at a small episode count. Mark it `slow` (heavy integration tests are excluded from pre-commit and run before push) so two real simulations never land in the default suite.
- [x] 3.2 Test pinning the thread-invariance property that licenses task 2.3: batch-1 forward, batch-128 forward, gradients, and a connectome-scale matmul are bit-identical across thread counts.

## 4. Benchmarks

- [x] 4.0 Benchmark scripts ship without unit tests, matching the existing `scripts/benchmarks/bench_evolution_smoke.py` precedent (untested, not run in CI). Deliberate — they are measurement tools, not library code.
- [x] 4.1 New `scripts/benchmarks/bench_device_backends.py`: per-step and batched forward, PPO-shaped forward+backward, connectome-scale tensors, and a large-net reference row proving the accelerator is healthy. Reports per-device timings and speedups; skips absent devices.
- [x] 4.2 New `scripts/benchmarks/bench_campaign_parallelism.py`: sequential baseline vs a sweep of worker counts, reporting wall-clock, speedup, and efficiency.
- [x] 4.3 Re-run both on the final tree and record the numbers in the logbook.

## 5. Documentation

- [x] 5.1 Usage-guide section: running campaigns, choosing `--workers`, per-run logs, and the note that per-run wall-clock inflates under load while results do not change.
- [x] 5.2 Update the existing device documentation to include `mps` and the quantum-family restriction: the `--device` table row in `docs/usage.md`, the hardware line in `docs/architectures.md`, and the "Executor pattern: CPU, GPU, QPU backends" line in `openspec/config.yaml`.
- [x] 5.3 Device guidance recording, with numbers, that CPU is the default for current model sizes; what MPS costs today; the conditions under which that flips (batched multi-agent rollouts, or models two orders of magnitude larger); and that the Neural Engine has no PyTorch path.
- [x] 5.4 Logbook 039: profile, device benchmarks (micro + end-to-end), parallel scaling curve, the deferred vectorisation with its 2.109e-15 byte-identity measurement, and the Amendment A interaction that defers it.
- [x] 5.5 CHANGELOG entry.
- [x] 5.6 Note in `openspec/changes/phase7-tracking/tasks.md` that the L4 panel's n ≥ 8 arms launch through this runner.

## 6. Close-out

- [x] 6.1 `uv run pre-commit run --all-files` green (the mandatory pre-push gate — CI Code Quality runs all-files).
- [x] 6.2 Full non-nightly suite green.
- [x] 6.3 Confirm no implementation code or docstring references planning docs, roadmap IDs, OpenSpec changes, or logbooks; technical rationale stated intrinsically.
- [x] 6.4 Re-review the change for drift, archive, review the branch, open the PR. (Drift found and fixed: the availability requirement was unqualified while the implementation scopes the torch check to torch-backed brains.)
