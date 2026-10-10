# Tasks: C.3 — body-level validation

- [x] 1. **Data** — **done 2026-10-10: both files vendored in LFS at Wormlight's pinned SHA-256s; columns are the modes; four modes capture 96.49% of the real postures; `validation/posture.py` loads and checks them.** Original scope:: vendor the eigenworm basis and the real postures with licences and provenance;
  checked against their pinned SHA-256s. Tests.
- [x] 2. **Instruments** — **done: the posture record carries the frame angle; `head_line_angle`; omega turns (head-swing crossings of the head segment's curvature, 135° net, wall swings skipped); forward bouts; eigenworm variance and amplitude in `validation/posture.py`. Tests.** Original scope:: the frame angle in the posture record; eigenworm variance, peak |κL|, omega
  turns, forward bouts. Tests on generated postures and the vendored real postures.
- [x] 3. **Harness and analysis** — **done: `run_capture` returns postures and behaviour; `body_validation.py` evaluates the arms in parallel, grades the bands, flags edges, runs 035's harness on each arm's captures and gates the control on C.1d's floor gate; the control's two configs and their generator. Tests.** Original scope:: behaviour capture in the evaluation harness;
  `scripts/analysis/body_validation.py` (evaluation over the arms, the half-step check, the grades, the
  bias curves through Logbook 035's harness, the control's effect size); the control's config. Tests.
- [ ] 4. **Pilot and registration**: evaluate a few runs to measure cost; `086-body-validation/launch.md`
  with every band; `/nematode-review-spec`.
- [ ] 5. **Control and evaluation**: train the derivative control, learning and frozen (seeds
  1801–1816, 32 runs); evaluate every arm, the MLP's untrained floor included (416 runs).
- [ ] 6. **Readout**: Logbook 086; tracker C.3; roadmap.
- [ ] 7. **Close-out**: full suite; hooks by exit code; validate; archive; PR.
