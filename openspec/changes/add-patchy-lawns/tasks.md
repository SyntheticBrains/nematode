# Tasks: D.1 — patchy lawns, internal state, and the roaming/dwelling readout

- [ ] 1. **Reference data and directions**: from Scheer & Bargmann 2023 (CC0), check the wild-type pickle's
  opcodes, then derive and vendor each animal's 5 s windows (speed, angular speed, in-lawn flag, the
  authors' label) with the HMM's parameters and provenance. Vendor Ji et al. 2021's deposit for the
  directions. Pin every array's condition and units from the papers' legends. Check `max_turn_rad`
  against the population's turning. Verify every cited direction against its source.
- [x] 2. **Lawns** — **done: `env/lawns.py` (`LawnField`, `LawnParams`, placement, area-weighted odour and gradient, intake, regrowth); the environment reads it in both field functions and copies it; `LawnConfig` with the food-model, shaping, substrate and multi-agent validators; pixel renderers refused. Tests.** Original scope:: `food_model: lawns`, the density grid, the area-weighted odour field and gradient
  vector, placement, and the scope refusals. The default byte-identical. Tests.
- [x] 3. **Intake and outcomes** — **done: the runner eats from the lawn cell under the worm after each move and regrows the lawns; reward and satiety by quality; a lawn run succeeds by surviving to `max_steps`; intake in the summary CSV and the per-run summary line. Tests.** Original scope:: per-step intake, reward and satiety by quality; the shaping
  validator; survived/starved outcomes; intake in the run summary and the log line. Tests.
- [x] 4. **Internal state and capture** — **done: `internal_state` (satiety over `BrainParams.max_satiety`, width 1); capture's satiety, per-step intake and on-lawn fields, omitted from exports when unrecorded. Tests.** Original scope:: the `internal_state` module; behaviour capture's satiety,
  intake and on-lawn fields. Tests.
- [x] 5. **Instrument** — **done 2026-10-10 on the registered retry.** The line, calibrated at the 5 s step, reached held-out κ = 0.49 against the 0.6 gate and was not used. The retry, registered before it was computed (f015a7ea), is a Gaussian-emission HMM on log speed and angular speed, fitted with the labels on the same calibration half. It reached held-out **κ = 0.632** (accuracy 91.1%), passing, close to the bar. It calls more roaming than the authors (17.4% against 10.6% of on-lawn windows), so absolute fractions are described, never matched. The vendored model reproduces the authors' own decoding on 99.2% of bins. Tests. Original scope:: `validation/roaming_dwelling.py`: the 5 s step measures and 10 s windows,
  the line, the reference HMM's Viterbi decoding per on-lawn run, agreement and slope calibration;
  the calibration on half the real animals and the held-out κ ≥ 0.6 gate. Tests.
- [x] 6. **Configs and analysis** — **done: the three arms' configs and their generator; `scripts/analysis/lawn_states.py` (intake plateaus from the run summary and the paired learning gate; held-out evaluation with capture; on-lawn windows, states by the calibrated model, bout durations, roaming where grazed against fresh). A smoke run on the untrained floor read 96% roaming on lawns. Tests.** Original scope:: the lawn cell's configs (signed speed, `max_turn_rad` π; learner,
  without `internal_state`, untrained) and their generator; the analysis script and its intake reader
  for the gate preflight. Tests.
- [ ] 7. **Pilot and registration**: a pilot on disjoint seeds for cost, gates and the satiety timescale;
  `087-patchy-lawns/launch.md`; gate preflight; `/nematode-review-spec`.
- [ ] 8. **Campaign and readout**: the positive control; Logbook 087; tracker D.1; roadmap (the
  point-worm decision with its reason).
- [ ] 9. **Close-out**: full suite; hooks by exit code; validate; archive; PR.
