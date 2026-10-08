# Tasks: C.1d — the MLP positive control through the body, with the kinematic instruments

Registered in `docs/experiments/logbooks/supporting/083-body-control/launch.md` before any scored run.

- [x] 1. **Body parameters and the reversal threshold**: the wave normalised to ±1 (the C.1c defect), the
  sourced defaults, the threshold, the posture recorder. Tests.
- [x] 2. **Instruments**: `validation/body_kinematics.py` (frequency by band crossing, wavelength,
  speed, reversal fraction, wall exclusion) and the evaluation harness, tested on generated postures of
  known frequency and wavelength.
- [x] 3. **Configs and analysis**: the calibration and control configs through a generator; the rule,
  the gates and the fallback in `body_control.py`. Tests.
- [x] 4. **Calibration pilot** — **done 2026-10-08: third pilot (floored) chose gain 2, frozen as the body's default; every seed ≥ 30% at gain 2, three within 5 points; cost about 41 min per run; kinematics in band but for seed 1503's speed (0.118).** *(first run 2026-10-08 uncapped: gain 1 chosen, but seeds locked into one gait and silenced segments distorted the wavelength; the reversal cap and the undulating-step gate added (design B′, E); re-run under the cap)* *(second run under the cap: gain 1 again, reversals brief, but the MLP silenced 2.7–5 segments per step, the head most; the wave amplitude floored at 0.25 (design B″); re-run a third time)*: seeds 1501–1504, four gains, learning and frozen (32 runs); the rule applied; the gain frozen; cost
  measured; the reversal-rate reference verified (done 2026-10-07: an order of magnitude, recorded in design B).
- [x] 5. **Registration** — **done 2026-10-08: launch.md; spec review (no blocking; speed's band edge, the stale proposal and design passages, the floor's value as a condition, fixed); half-step preflight passes.** Original scope: the launch record, then `/nematode-review-spec`.
- [x] 6. **Control** — **done 2026-10-08: `fallback` at 350 steps (one seed at 26.9%); the fallback pilot chose 500; the re-run on 1517–1524 passes, 92.5% against 0.6%.** Original scope: seeds 1505–1512, learning and frozen; the fallback if needed.
- [x] 7. **Kinematics** — **done: frequency and wavelength in band, speed at the edge, half-step passes.** Original scope: the instruments on the trained controls; the half-step check.
- [x] 8. **Readout** — **done: Logbook 083, tracker C.1d, roadmap C.1, index, CHANGELOG.** Original scope: Logbook 083; tracker C.1d; roadmap C.1.
- [x] 9. **Close-out** — **done 2026-10-08: full suite 7420 passed, 34 skipped; hooks; validate; archived.** Original scope: full suite; hooks by exit code; validate; archive; PR.
