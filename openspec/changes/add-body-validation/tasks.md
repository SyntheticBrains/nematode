# Tasks: C.3 — body-level validation

- [ ] 1. **Data**: vendor the eigenworm basis and the real postures with licences and provenance;
  checked against their pinned SHA-256s. Tests.
- [ ] 2. **Instruments**: the frame angle in the posture record; eigenworm variance, peak |κL|, omega
  turns, forward bouts. Tests on generated postures and the vendored real postures.
- [ ] 3. **Harness and analysis**: behaviour capture in the evaluation harness;
  `scripts/analysis/body_validation.py` (evaluation over the arms, the half-step check, the grades, the
  bias curves through Logbook 035's harness, the control's effect size); the control's config. Tests.
- [ ] 4. **Pilot and registration**: evaluate a few runs to measure cost; `086-body-validation/launch.md`
  with every band; `/nematode-review-spec`.
- [ ] 5. **Control and evaluation**: train the derivative control (16 seeds); evaluate every arm.
- [ ] 6. **Readout**: Logbook 086; tracker C.3; roadmap.
- [ ] 7. **Close-out**: full suite; hooks by exit code; validate; archive; PR.
