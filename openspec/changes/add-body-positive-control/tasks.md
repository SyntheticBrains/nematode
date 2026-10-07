# Tasks: C.1d — the MLP positive control through the body, with the kinematic instruments

Registered in `docs/experiments/logbooks/supporting/083-body-control/launch.md` before any scored run.

- [x] 1. **Body parameters and the reversal threshold**: the wave normalised to ±1 (the C.1c defect), the
  sourced defaults, the threshold, the posture recorder. Tests.
- [x] 2. **Instruments**: `validation/body_kinematics.py` (frequency by band crossing, wavelength,
  speed, reversal fraction, wall exclusion) and the evaluation harness, tested on generated postures of
  known frequency and wavelength.
- [ ] 3. **Configs and analysis**: the calibration and control configs through a generator; the rule,
  the gates and the fallback in `body_control.py`. Tests.
- [ ] 4. **Calibration pilot**: seeds 1501–1504, four gains, learning and frozen (32 runs); the rule applied; the gain frozen; cost
  measured; the reversal-rate reference verified.
- [ ] 5. **Registration**: the launch record, then `/nematode-review-spec`.
- [ ] 6. **Control**: seeds 1505–1512, learning and frozen; the fallback if needed.
- [ ] 7. **Kinematics**: the instruments on the trained controls; the half-step check.
- [ ] 8. **Readout**: Logbook 083; tracker C.1d; roadmap C.1.
- [ ] 9. **Close-out**: full suite; hooks by exit code; validate; archive; PR.
