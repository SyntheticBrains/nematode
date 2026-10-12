# Tasks: M.9 — the thermal body cell

Registered in `docs/experiments/logbooks/supporting/092-thermal-body/launch.md` before any scored
panel run.

- [x] 1. **Configs**: `generate_thermal_body_configs.py`: the six connectome arms and the two MLP
  arms through the kinematic body (Cook 2019), at targets 35 and 20, and the point-worm gap-only
  check's arms at target 20. Loader tests; a test that a kinematic body's sensed temperature is the
  head's; a short end-to-end test of a thermal body agent stepping.
- [x] 2. **Analysis**: `thermal_body.py`: `STEMS` per null for the gate preflight; the MLP gate
  (C.1d's rule); the one registered reading at 0.058; the beside readings, including the kinematic
  comparison via `body_kinematics_eval.py`. Tests.
- [x] 3. **Plan record**: the D21 note and M.9's entry in the roadmap; the tracker's M.9 entry.
- [ ] 4. **Stage 1**: MLP-PPO and its floor on seeds 9701–9708 at targets 35 and 20; the target rule
  (35 if it passes, else 20, else stop and defer M.9).
- [ ] 5. **Stage 2**: the six connectome arms on seeds 9711–9714 at the chosen target; the gate
  preflight per null; the spread and the cost, re-measured here for the registration. At target 20,
  the point-worm gap-only check on the same seeds.
- [ ] 6. **Registration**: `launch.md`; `/nematode-review-spec`.
- [ ] 7. **Campaign and readout**: the panel from a worktree at the registration commit; the
  kinematics; the logbook; tracker; roadmap; CHANGELOG.
- [ ] 8. **Close-out**: full suite; hooks by exit code; validate; archive; PR.
