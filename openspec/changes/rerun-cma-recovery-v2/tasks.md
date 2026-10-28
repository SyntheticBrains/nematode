## 1. The speed-up (behaviour-identical)

- [x] 1.1 `grade_populations_v2(presence=...)`; the fit's objective without presence.
- [x] 1.2 `NoiseLibrary`'s stretch offsets computed once; `measure` without per-step `dataclasses.replace`.
- [x] 1.3 Parity tests: measured steps field for field; statistics, D and D_fit with presence on and off; B v2's reachability reading on a discarded population unchanged.
- [x] 1.4 Re-time a v2 simulation (discarded seeds), keeping each speed-up only where it is faster and identical. (Identical end to end on a discarded population, 99.35 M: D 7.0375 both ways; 35.7 s to 20.2 s, 1.76x.)

## 2. The re-run's code

- [x] 2.1 `cma_recovery.run_fit`: optional simulator and plateau guard, defaults unchanged.
- [x] 2.2 `scripts/analysis/cma_recovery_v2.py`: `simulate_v2`, `predictive_distance_v2`, the truth selection, `preflight`, `pilot`, `run`, `read`, `smoke`; the launch guard.
- [x] 2.3 Tests: the distance (weights, undefined statistics, the reported statistic excluded); the guarded plateau; the seeds' disjointness; the candidate order and keep rules on toy records.
- [x] 2.4 The smoke (CMA 5993, base 99.3 M, discarded): two generations and a resume, about 40 s per generation on 16 workers.
- [x] 2.5 The gates' oracle on discarded seeds (99.4–99.6 M): the oracle and unfitted-start gains on the probe's eligible candidates; return to the maintainer if any is within 0.05 of its bar. (Oracle 1.027, start 0.047: far from both.)

## 3. Registration and the runs

- [x] 3.1 `supporting/116-cma-recovery-v2/launch.md` Part A; committed.
- [ ] 3.2 The preflight; the pilot; Part B with `preflight.json` and `pilot.json`; committed.
- [ ] 3.3 The registered fits; `read`; `recovery.json`.

## 4. Close-out

- [ ] 4.1 Logbook 116; the tracker (F.0, C.1b next); AGENTS.md; the CHANGELOG.
- [ ] 4.2 Full suite; `pre-commit run --all-files` after `git add -A`; `openspec validate --strict`; archive; PR.
