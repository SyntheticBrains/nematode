## 1. The speed-up (behaviour-identical)

- [ ] 1.1 `grade_populations_v2(presence=...)`; the fit's objective without presence.
- [ ] 1.2 `NoiseLibrary`'s stretch offsets computed once; `measure` without per-step `dataclasses.replace`.
- [ ] 1.3 Parity tests: measured steps field for field; statistics, D and D_fit with presence on and off; B v2's reachability reading on a discarded population unchanged.
- [ ] 1.4 Re-time a v2 simulation (discarded seeds), keeping each speed-up only where it is faster and identical.

## 2. The re-run's code

- [ ] 2.1 `cma_recovery.run_fit`: optional simulator and plateau guard, defaults unchanged.
- [ ] 2.2 `scripts/analysis/cma_recovery_v2.py`: `simulate_v2`, `predictive_distance_v2`, the truth selection, `preflight`, `pilot`, `run`, `read`, `smoke`; the launch guard.
- [ ] 2.3 Tests: the distance (weights, undefined statistics, the reported statistic excluded); the guarded plateau; the seeds' disjointness; the candidate order and keep rules on toy records.
- [ ] 2.4 The smoke (CMA 5993, base 99.3 M, discarded).
- [ ] 2.5 The gates' oracle on discarded seeds (99.4–99.6 M): the oracle and unfitted-start gains on the probe's eligible candidates; return to the maintainer if any is within 0.05 of its bar.

## 3. Registration and the runs

- [ ] 3.1 `supporting/116-cma-recovery-v2/launch.md` Part A; committed.
- [ ] 3.2 The preflight; the pilot; Part B with `preflight.json` and `pilot.json`; committed.
- [ ] 3.3 The registered fits; `read`; `recovery.json`.

## 4. Close-out

- [ ] 4.1 Logbook 116; the tracker (F.0, C.1b next); AGENTS.md; the CHANGELOG.
- [ ] 4.2 Full suite; `pre-commit run --all-files` after `git add -A`; `openspec validate --strict`; archive; PR.
