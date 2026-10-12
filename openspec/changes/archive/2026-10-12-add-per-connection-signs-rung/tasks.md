# Tasks: the per-connection signs rung

Phase 8, run before the M.9 decision. Registered in
`docs/experiments/logbooks/supporting/091-per-connection-signs/launch.md` before any scored run.

- [x] 1. **Origins**: the rewiring's optional origin record. Tests: null unchanged, presynaptic
  neuron kept, a bijection onto the wild type's edges, counts travel.
- [x] 2. **Sign model**: `synapse_signs: per_connection` in the topology, the null's travelling signs,
  every "grounded" check (validators, construction refusals, PPO and plastic-rule hand-over,
  checkpoint metadata). Tests: AWC → AIY negative; magnitudes equal; differs from atlas exactly on
  the 342; a null's outgoing sign multisets; the atlas null unchanged; PPO holds; off byte-identical.
- [x] 3. **Configs**: the rung's 12 thermal t35 arms and the two per-connection modulated arms, through
  generators, with loader tests.
- [x] 4. **Analysis**: the rung's three readings, Holm across the three, the verdicts and the beside readings, in the gate
  preflight's shape (`STEMS` per sign level), with tests. The modulated arms through `lawn_states.py`.
- [x] 4a. **Plan record**: the rung entered in `phase8-tracking` as M.10 and in the roadmap's Phase 8
  list, with C.5's start noted, before registration.
- [x] 5. **Pilot**: seeds 9501–9504, configured as the panel; gates per level, spreads, cost.
- [x] 5a. **Calibration** — **done**: the first pilot (9501–9504) found grounded signs saturate the network (5 of 16 grounded learners never learned); `chemical_weight_scale` and `sign_scale_calibration.py` (seeds 9601–9616) chose atlas 0.35 and per-connection 0.42; the second pilot (9511–9514) ran at those scales. Tests.
- [x] 6. **Registration**: `launch.md`, the gate preflight, `/nematode-review-spec`.
- [x] 7. **Campaign and readout**: the panel from a worktree at the registration commit; the modulated
  arms; the logbook; tracker; roadmap; CHANGELOG.
- [x] 8. **Close-out**: full suite; hooks by exit code; validate; archive; PR.
