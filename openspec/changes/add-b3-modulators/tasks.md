# Tasks: B.3 — serotonin and PDF on the connectome

- [x] 1. **Circuit data**: `data/neuromodulators/` with provenance, and a loader that maps the classes
  to the connectome's neurons. Tests.
- [x] 2. **Field**: the concentrations; release; the held currents (receptors, ingestion on NSM, PDF's
  inhibition of NSM) in rollout and replay; satiety for the satiety-link arm; appending to the
  preprocessed state; episode reset; checkpoint round-trip; byte-identity when off; an old checkpoint
  still loading. Tests.
- [x] 3. **Knockouts**: `modulator_knockout` (`tph1`, `pdfr1`; *mod-1* and *pdf* would be identical to
  them here); an evaluation override for acute knockouts. Tests.
- [x] 4. **Ingestion**: `BrainParams.food_ingested` from the agent. Tests.
- [x] 5. **Configs and analysis**: the modulated arms (intact, floor, raised *tph-1*, raised *pdfr-1*,
  satiety link); the acute knockout readings in `lawn_states.py` with Holm across the two. Tests.
- [x] 5a. **Re-plan after the first pilot**: atlas signs held under PPO; the speed readout; the
  sign-grounded arms. Tests.
- [x] 6. **Pilot and registration**: the sign-grounded pilot on disjoint seeds (9411–9418) for the
  gates, the knockouts' spreads and the cost; the biological speed minimum; `090-modulators/launch.md`;
  the preflight on the eight pilot seeds; `/nematode-review-spec`.
- [ ] 7. **Campaign and readout**: the panel; Logbook 090; tracker B.3; roadmap.
- [ ] 8. **Close-out**: full suite; hooks by exit code; validate; archive; PR.
