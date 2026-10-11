# Tasks: B.3 — serotonin and PDF on the connectome

- [ ] 1. **Circuit data**: `data/neuromodulators/` with provenance, and a loader that maps the classes
  to the connectome's neurons. Tests.
- [ ] 2. **Field**: the concentrations, release, gating in rollout and replay, the NSM ingestion drive,
  appending to the preprocessed state, episode reset, checkpoint round-trip, and byte-identity when
  off. Tests.
- [ ] 3. **Knockouts**: `modulator_knockout`; an evaluation override for acute knockouts. Tests.
- [ ] 4. **Ingestion**: `BrainParams.food_ingested` from the agent. Tests.
- [ ] 5. **Configs and analysis**: the modulated arms (intact, floor, raised *tph-1*, raised *pdfr-1*,
  satiety link); the acute knockout readings in `lawn_states.py` with Holm across the two. Tests.
- [ ] 6. **Pilot and registration**: a pilot on disjoint seeds for the gates, the knockouts' effect
  size and the minimum; `090-modulators/launch.md`; gate preflight; `/nematode-review-spec`.
- [ ] 7. **Campaign and readout**: the panel; Logbook 090; tracker B.3; roadmap.
- [ ] 8. **Close-out**: full suite; hooks by exit code; validate; archive; PR.
