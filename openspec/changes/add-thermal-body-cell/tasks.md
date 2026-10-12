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
- [x] 4. **Stage 1**: MLP-PPO and its floor on seeds 9701–9708 at targets 35 and 20; the target rule
  (35 if it passes, else 20, else stop and defer M.9). *(Read 2026-10-12: `fallback` at both, so
  the rule stops M.9; record in `supporting/092-thermal-body/stage1/`.)*
- [x] 4b. **The retry (Decision G)**: MLP-PPO and its floor at target 20, 6,000 episodes, seeds
  9721–9728, from a worktree at the amendment's commit; `mlp-gate --seeds 9721..9728 --targets 20`.
  Pass: set `TARGET = 20` in `thermal_body.py` (so `STEMS`, which the gate preflight reads, names
  the t20 arms), with a test, and commit it before stage 2's worktree is pinned; then stage 2 at
  6,000 episodes. Fail: M.9 stops, deferred-with-destination.
  *(Read 2026-10-13: `passes`, every seed competent, 57–95% against 0% floors; record in
  `supporting/092-thermal-body/retry/`; `TARGET` is 20.)*
- [x] 5. **Stage 2**: the six connectome arms on seeds 9711–9714 at the chosen target and 6,000 episodes; the gate
  preflight per null; the spread and the cost, re-measured here for the registration; the body's median episodes to 30% beside the anchor's (Decision G). At target 20,
  the point-worm gap-only check on the same seeds.
  *(Read 2026-10-13: gates pass on both nulls; the preflight reads `near_bar`, accepted with the
  user and documented in `launch.md`; the anchor saturates at 20; 32 seeds chosen with the user.)*
- [x] 6. **Registration**: `launch.md`; `/nematode-review-spec`. *(2026-10-13: 32 seeds, 2301–2332.)*
- [ ] 7. **Campaign and readout**: the panel from a worktree at the registration commit; the
  kinematics; the logbook; tracker; roadmap; CHANGELOG.
- [ ] 8. **Close-out**: full suite; hooks by exit code; validate; archive; PR.
