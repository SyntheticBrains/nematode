# Tasks — parallel campaign runner + honest accelerator selection

## 1. Accelerator selection

- [ ] 1.1 Add `MPS = "mps"` to `DeviceType`; `to_torch_device_str()` returns `"mps"` for it, `GPU` keeps `"cuda"`, `QPU`/`CPU` unchanged. Update the enum docstring to state what each maps to and that `GPU` is CUDA-only by design (Decision 4).
- [ ] 1.2 Add an availability check that raises an actionable error naming the requested device and the platform-appropriate alternative, invoked at CLI entry before brain construction. CPU takes no check.
- [ ] 1.3 Wire the check into `scripts/run_simulation.py` (and any other entry point that accepts `--device`); confirm `--device mps` runs end-to-end and `--device gpu` on a non-CUDA build fails cleanly with no torch assertion.
- [ ] 1.4 Tests: `mps` maps correctly; `gpu` still maps to `cuda`; unavailable-accelerator error names device + alternative; CPU needs no check. Skip-guard any test that requires a real MPS device.

## 2. Campaign runner

- [ ] 2.1 New `scripts/run_campaign.py`: repeatable `--config`, `--seeds` accepting ranges (`1-8`), comma lists, and space-separated values; cross product to the run plan; passthrough of extra args after `--`.
- [ ] 2.2 Bounded concurrency at `--workers`, defaulting to `max(1, cpu_count - 2)`; each run a `subprocess` of `run_simulation.py` with explicit `--seed`.
- [ ] 2.3 Child environment pins BLAS/OMP threads to 1 (Decision 3); document the verified bit-invariance at the call site as the reason this is safe.
- [ ] 2.4 Per-run log files under the campaign output directory; progress reporting while running; per-run status table at the end.
- [ ] 2.5 Non-zero exit if any run failed, after attempting every run; SIGINT terminates children rather than orphaning them.
- [ ] 2.6 `--dry-run` prints the planned commands and starts nothing.
- [ ] 2.7 Tests: cross-product planning; seed-spec parsing incl. malformed input; worker bound respected; failure surfaces in the summary and exit code; dry-run starts no process; passthrough args reach the command. Use a stub child command so the suite stays fast and hermetic.

## 3. Equivalence evidence

- [ ] 3.1 Test asserting a campaign-launched run and a hand-launched run of the same config/seed agree, at a small episode count.
- [ ] 3.2 Test pinning the thread-invariance property that licenses task 2.3: batch-1 forward, batch-128 forward, gradients, and a connectome-scale matmul are bit-identical across thread counts.

## 4. Benchmarks

- [ ] 4.1 New `scripts/benchmarks/bench_device_backends.py`: per-step and batched forward, PPO-shaped forward+backward, connectome-scale tensors, and a large-net reference row proving the accelerator is healthy. Reports per-device timings and speedups; skips absent devices.
- [ ] 4.2 New `scripts/benchmarks/bench_campaign_parallelism.py`: sequential baseline vs a sweep of worker counts, reporting wall-clock, speedup, and efficiency.
- [ ] 4.3 Re-run both on the final tree and record the numbers in the logbook.

## 5. Documentation

- [ ] 5.1 Usage-guide section: running campaigns, choosing `--workers`, per-run logs, and the note that per-run wall-clock inflates under load while results do not change.
- [ ] 5.2 Device guidance recording, with numbers, that CPU is the default for current model sizes; what MPS costs today; the conditions under which that flips (batched multi-agent rollouts, or models two orders of magnitude larger); and that the Neural Engine has no PyTorch path.
- [ ] 5.3 Logbook 039: profile, device benchmarks (micro + end-to-end), parallel scaling curve, the deferred vectorisation with its 2.109e-15 byte-identity measurement, and the Amendment A interaction that defers it.
- [ ] 5.4 CHANGELOG entry.

## 6. Close-out

- [ ] 6.1 `uv run pre-commit run --all-files` green (the mandatory pre-push gate — CI Code Quality runs all-files).
- [ ] 6.2 Full non-nightly suite green.
- [ ] 6.3 Confirm no implementation code or docstring references planning docs, roadmap IDs, OpenSpec changes, or logbooks; technical rationale stated intrinsically.
- [ ] 6.4 Re-review the change for drift, archive, review the branch, open the PR.
