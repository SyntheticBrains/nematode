# Tasks: G.0-B — the modelling question: strength-normalised and ohmic gap coupling

Phase 9, after the STOP condition was met
([Logbook 100](../../../docs/experiments/logbooks/100-body-logcount.md)). The maintainer's decisions of
2026-10-20 were to open the modelling question inside Phase 9, with strength-normalised and ohmic
models, the point worm first, and the body only for a model that reads `placement`.

- [x] 1. **Brain option** `gap_normalisation` (design Decision A):

  - the strength normaliser on each wiring's own matrix;
  - the refusals, by the validator and at build;
  - the docstrings.

  Tests: the bound, the spectral radius, the unit identity, a zero-strength neuron, byte identity, and
  every refusal.
  `.test_durations` is regenerated at close-out.

- [x] 2. **The probe** (Decision C): the strength wild type against the gap-off brain on the pilot's
  seeds, under the half-spread rule, committed as `probe.json`. If it fails, stop for a dated decision.

- [x] 3. **Configs**: the strength model's four configs (thermal t35, Emmons, count weights) from a
  generator, with loader and build tests. The ohmic model reuses G.0-A's configs.

- [x] 4. **Harness** `scripts/analysis/gap_model_question.py` (Decisions D and E):

  - subcommands `plan`, `probe`, `preflight`, `table` and `score`;
  - the family-of-two correction, the verdict map and the joint outcome;
  - the verified manifest, imported from `logcount_contrast.py`.

  Tests: every verdict row and joint outcome, the refusals, and seed disjointness.

- [x] 5. **Before launch**:

  - smokes on 9913–9914;
  - the strength pilot on 9905–9912;
  - the preflight for both models;
  - `table.json`;
  - the operating characteristic, simulated;
  - the cost, from the pilot records;
  - `/nematode-review-spec add-gap-model-question`.

- [x] 6. **Registration**: `docs/experiments/logbooks/supporting/101-gap-model-question/launch.md`,
  committed before any panel run. The seed-band table gains 4201–4264, 9905–9912 and 9913–9914.

- [x] 7. **The panel** from a separate worktree: 512 runs with the output controls. *(512 of 512 exited 0, 7.5 h.)*

- [x] 8. **Readout**: Logbook 101, with what the pair of verdicts does to the STOP statement. Update
  the tracker, the roadmap and the CHANGELOG.

- [x] 9. **Close-out**: full suite, `pre-commit run -a`, `openspec validate --strict`, archive, PR.
