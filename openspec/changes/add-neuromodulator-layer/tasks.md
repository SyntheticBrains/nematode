# Tasks: N — the neuromodulator layer from data

Phase 10, block N: N.1, N.2 and N.3. The maintainer's decisions of 2026-10-24:

- the union of both sources' peptide couples, every edge tagged;
- one pool per ligand (the long-range model);
- monoamine receptor signs fixed, peptide couples fitted;
- a receptor-placement null keeping each receptor's count, by class.

## 1. N.1, licences and extraction

- [ ] 1.1 Record the licences: Bentley's S1 Dataset with its article (CC BY 4.0); Ripoll-Sánchez's
  figshare items (CC BY 4.0); the MIT-only 2024 corrections left out.
- [ ] 1.2 `scripts/analysis/extract_neuromodulator_layer.py`: download by URL with checksums, the
  outer-product check, monoamine producers by synthesis marker, the union of peptide couples with
  source tags, the drives, and the names the connectome lacks.
- [ ] 1.3 `data/neuromodulators/receptor_signs.json`: each monoamine receptor's coupling, verified
  against its citation's text, or marked fitted.
- [ ] 1.4 `data/neuromodulators/layer.json` and its PROVENANCE entry, including what the 2024
  corrections would change.

## 2. N.2, the layer

- [ ] 2.1 `quantumnematode/connectome/neuromodulator_layer.py`: the layer spec, its loader, the
  hand-placed circuit as a spec, and the gene knockouts.
- [ ] 2.2 `connectome_ppo.py`: `modulator_layer`, `modulator_timescales_s` and
  `modulator_knockout_genes`, with load-time validation; the generalised current and concentrations;
  the hand-placed path untouched.
- [ ] 2.3 `BatchedConnectome` on the data layer, with its layer-1 parity test.
- [ ] 2.4 Tests:
  - the default is byte-identical;
  - the restricted spec matches the hand-placed field bitwise, with and without each knockout;
  - gene knockouts;
  - unknown genes are refused;
  - fitted couples start at zero, fixed ones at 1.
- [ ] 2.5 The circuit-agreement table (design Decision F.2).

## 3. N.3, the null

- [ ] 3.1 The receptor-placement null, with its config keys.
- [ ] 3.2 `scripts/analysis/neuromodulator_null_audit.py`: the matched-statistics table over seeds
  7501–7564, and the modulator term in `sensory_motor_hops.py`.
- [ ] 3.3 Tests: exact counts, one placement per receptor gene, reproducible draws.

## 4. Documentation and tracker

- [ ] 4.1 Logbook 106, AGENTS.md, `docs/usage.md` and the CHANGELOG.
- [ ] 4.2 Close N.1–N.3 in the tracker, and add the 7501–7564 band to its seed table.

## 5. Close-out

- [ ] 5.1 Full suite and `pre-commit run -a`.
- [ ] 5.2 `openspec validate add-neuromodulator-layer --strict`.
- [ ] 5.3 Archive.
- [ ] 5.4 PR.
