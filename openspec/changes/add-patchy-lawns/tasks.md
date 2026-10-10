# Tasks: D.1 — patchy lawns, internal state, and the roaming/dwelling readout

- [ ] 1. **Reference data and directions**: vendor Ji et al. 2021's Dryad deposit (CC0) with provenance
  and a pinned SHA-256. Pin each deposited array's condition and units from the paper's legends,
  including the durations' unit. Verify against its source every direction the registration will cite:
  the patch assay, food density, quality (Shtonda & Avery 2006), and the mutants for B.3.
- [ ] 2. **Lawns**: `food_model: lawns`, the density grid, the area-weighted odour field and gradient
  vector, placement, and the scope refusals. The default byte-identical. Tests.
- [ ] 3. **Intake and outcomes**: per-step intake, reward and satiety by quality; the shaping
  validator; survived/starved outcomes; intake in the run summary and the log line. Tests.
- [ ] 4. **Internal state and capture**: the `internal_state` module; behaviour capture's satiety,
  intake and on-lawn fields. Tests.
- [ ] 5. **Instrument**: `validation/roaming_dwelling.py`: the 20-second speed median and variance;
  a two-state Gaussian HMM fitted by expectation-maximisation, written here; the fit on the real track
  and its agreement with the authors' labels (the gate); on-lawn reading only. Tests.
- [ ] 6. **Configs and analysis**: the lawn cell's configs (signed speed, `max_turn_rad` π; learner,
  without `internal_state`, untrained) and their generator; the analysis script and its intake reader
  for the gate preflight. Tests.
- [ ] 7. **Pilot and registration**: a pilot on disjoint seeds for cost, gates and the satiety timescale;
  `087-patchy-lawns/launch.md`; gate preflight; `/nematode-review-spec`.
- [ ] 8. **Campaign and readout**: the positive control; Logbook 087; tracker D.1; roadmap (the
  point-worm decision with its reason).
- [ ] 9. **Close-out**: full suite; hooks by exit code; validate; archive; PR.
