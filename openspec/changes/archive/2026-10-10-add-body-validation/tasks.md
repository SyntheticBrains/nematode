# Tasks: C.3 — body-level validation

- [x] 1. **Data** — **done 2026-10-10: both files vendored in LFS at Wormlight's pinned SHA-256s; columns are the modes; four modes capture 96.49% of the real postures; `validation/posture.py` loads and checks them.** Original scope:: vendor the eigenworm basis and the real postures with licences and provenance;
  checked against their pinned SHA-256s. Tests.
- [x] 2. **Instruments** — **done: the posture record carries the frame angle; `head_line_angle`; omega turns (head-swing crossings of the head segment's curvature, 135° net, wall swings skipped); forward bouts; eigenworm variance and amplitude in `validation/posture.py`. Tests.** Original scope:: the frame angle in the posture record; eigenworm variance, peak |κL|, omega
  turns, forward bouts. Tests on generated postures and the vendored real postures.
- [x] 3. **Harness and analysis** — **done: `run_capture` returns postures and behaviour; `body_validation.py` evaluates the arms in parallel, grades the bands, flags edges, runs 035's harness on each arm's captures and gates the control on C.1d's floor gate; the control's two configs and their generator. Tests.** Original scope:: behaviour capture in the evaluation harness;
  `scripts/analysis/body_validation.py` (evaluation over the arms, the half-step check, the grades, the
  bias curves through Logbook 035's harness, the control's effect size); the control's config. Tests.
- [x] 4. **Pilot and registration** — **done: evaluation pilot (about 35 min for 416 runs); the control's pilot reads `readable` (plateaus 0–25%); `launch.md` committed at 3fec9b4f after the spec review (the MLP's untrained floor, paired floor comparison).** Original scope:: evaluate a few runs to measure cost; `086-body-validation/launch.md`
  with every band; `/nematode-review-spec`.
- [x] 5. **Control and evaluation** — **done: 32 control runs succeeded in 1 h 53 min (registered 1.5 h); 416 runs evaluated in 31.5 min, reproduced byte for byte.** Original scope:: train the derivative control, learning and frozen (seeds
  1801–1816, 32 runs); evaluate every arm, the MLP's untrained floor included (416 runs).
- [x] 6. **Readout** — **done: Logbook 086, its index row, tracker C.3 met, the roadmap's C.3 row and note, the changelog.** Original scope:: Logbook 086; tracker C.3; roadmap.
- [x] 7. **Close-out** — **done: full suite 7477 passed; hooks over all files by exit code; validated; archived.** Original scope:: full suite; hooks by exit code; validate; archive; PR.
