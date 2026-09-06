# Tasks: centre the compressed modulator

## 1. The rule

- [x] 1.1 A bias-corrected running mean `c` of `tanh(δ / σ)` at the scale rate, zero before any
  observation, used at its pre-update value; the modulator becomes `tanh(δ / σ) − c` under
  modulator normalisation.
- [x] 1.2 `c` advances under a freeze and in unmodulated mode (modulator stays `1.0`); nothing
  changes with the switch off.
- [x] 1.3 Telemetry key `plasticity_modulator_centre` (NaN when off); history field; recorded by
  the shared recorder.

## 2. Tests

- [x] 2.1 The centred value equals `tanh(δ / σ_prev) − c_prev` against an independent
  bias-corrected mean; `c = 0` on the first step; the range `[−2, 2]`.
- [x] 2.2 Zero mean on a deterministic periodic stream (three foods, one death, small steps summing
  the raw period to zero), measured over whole periods after whole-period warm-up (within `0.005`),
  with the uncentred compression's mean on the same steps exceeding `0.005` in magnitude.
- [x] 2.3 Freeze advances `c` and writes nothing; unmodulated keeps `1.0` and still reports `c`.
- [x] 2.4 The key and the recorder; the existing scaling tests updated to the centred value.
- [x] 2.5 The frozen-reference byte-identity test keeps passing with the switch off.

## 3. Docs and close-out

- [x] 3.1 `docs/architectures.md` plasticity row; `CHANGELOG.md`.
- [x] 3.2 Pre-commit gate on all files exit 0; full suite green. (4576 passed, 2 skipped, 2 xfailed; parallel, nightly excluded as CI runs it)
- [x] 3.3 No implementation code or docstring references a planning document.
- [x] 3.4 Re-review for drift, archive, review the branch, open the PR. (both new scenarios and every kept scenario map to a named test)
