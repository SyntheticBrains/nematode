# Tasks: the per-connection signs rung

Phase 8, run before the M.9 decision. Registered in
`docs/experiments/logbooks/supporting/091-per-connection-signs/launch.md` before any scored run.

- [ ] 1. **Origins**: the rewiring's optional origin record. Tests: null unchanged, presynaptic
  neuron kept, a bijection onto the wild type's edges, counts travel.
- [ ] 2. **Sign model**: `synapse_signs: per_connection` in the topology, the null's travelling signs,
  every "grounded" check (validators, construction refusals, PPO and plastic-rule hand-over,
  checkpoint metadata). Tests: AWC → AIY negative; magnitudes equal; differs from atlas exactly on
  the 342; a null's outgoing sign multisets; the atlas null unchanged; PPO holds; off byte-identical.
- [ ] 3. **Configs**: the rung's 12 thermal t35 arms and the two per-connection modulated arms, through
  generators, with loader tests.
- [ ] 4. **Analysis**: the rung's readings, Holm, the verdicts and the beside readings, in the gate
  preflight's shape (`STEMS` per sign level), with tests. The modulated arms through `lawn_states.py`.
- [ ] 5. **Pilot**: seeds 9501–9504, configured as the panel; gates per level, spreads, cost.
- [ ] 6. **Registration**: `launch.md`, the gate preflight, `/nematode-review-spec`.
- [ ] 7. **Campaign and readout**: the panel from a worktree at the registration commit; the modulated
  arms; the logbook; tracker; roadmap; CHANGELOG.
- [ ] 8. **Close-out**: full suite; hooks by exit code; validate; archive; PR.
