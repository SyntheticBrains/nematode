## 1. Baselines before the change

- [x] 1.1 Give `scripts/analysis/connectome_identity_check.py` a brain job: a SHA-256 of a built brain's parameters and sign buffers at a fixed seed, for the nine excluded configs and the encoded small-lawn config.
- [x] 1.2 Record `connectome_identity_check.py baseline --build build/aia-ade-signs/identity` before any sign change.

## 2. The table

- [x] 2.1 From López-Cruz et al. 2019's published text, quote verbatim the evidence for the evidence column, and verify from its methods the promoters of the chemosensory (tax-4) and mechanosensory (mec-3) glutamate knockouts.
- [x] 2.2 Add the 16 rows to `data/connectome/sign_overrides_physiology.csv` (sign −1, citation `lopezcruz2019`): ASEL, ASER, ASGL, ASKL, ASKR, AWCL and AWCR onto AIAL; ASER, ASGR, ASKR, AWCL and AWCR onto AIAR; FLPL onto ADEL; FLPL, FLPR and AVM onto ADER.
- [x] 2.3 Add `lopezcruz2019` to `PHYSIOLOGY_CITATIONS`, and update `data/connectome/PROVENANCE.md`'s entry for the table.
- [x] 2.4 `per_connection_signs(..., exclude_citations=())`: drop the listed citations' overrides, and refuse a key the table does not cite.

## 3. The brain and its callers

- [x] 3.1 `ConnectomePPOBrainConfig.sign_override_exclude: list[str] = []`, validated against `PHYSIOLOGY_CITATIONS` and refused unless `synapse_signs` is `per_connection`. It is passed through `_per_connection_edge_signs`, whose new parameter defaults to none excluded.
- [x] 3.2 `generate_lawn_configs.py`'s `modulated_perconn` variant and `generate_sign_rung_configs.py` write `sign_override_exclude: [lopezcruz2019]`; regenerate the five lawn and four thermal configs, and check that nothing else in them changes.
- [x] 3.3 `sign_scale_calibration.py` builds its per-connection brains with the exclusion.

## 4. Tests

- [x] 4.1 `test_signs.py`:
  - the pinned composition becomes 67 / 1,676 / 1,433 / 533, with physiology 48 inhibitory and 19 excitatory, the rule 1,308 and 125, by weight 1,080 and 6,868, and 48 physiology signs opposite to the rule;
  - the Wormlight comparison runs on the table with `lopezcruz2019` excluded, its digest unchanged, and the full table differs from that on exactly the 16 edges.
- [x] 4.2 Test the 16 edges' sign, source and citation, and that the 13 other glutamatergic edges onto AIA and ADE keep the rule.
- [x] 4.3 Test that excluding `lopezcruz2019` reproduces the earlier composition edge for edge, and that an unknown key is refused.
- [x] 4.4 Test the config field: it is refused with an unknown key or another sign model, and the nine configs load with it.
- [x] 4.5 `test_connectome_per_connection_signs.py`: the corrected count becomes 358 (the grounded 3,176 unchanged).
- [x] 4.6 An encoded brain's normalised counts differ from its excluded twin's in exactly the 16 entries, each moving from the excitatory matrix to the inhibitory one at the same magnitude.

## 5. Checks

- [x] 5.1 `connectome_identity_check.py check` against 1.2: every job identical except the encoded brain job. Recorded in `docs/experiments/logbooks/supporting/122-c2-encoder/aia_ade_signs_identity.json`.
- [x] 5.2 `uv run pytest -m "not nightly"` and `uv run pre-commit run -a`.

## 6. Records

- [x] 6.1 An addendum to Logbook 122: the correction, its evidence, that the readiness figures are on the earlier wiring, and the 13 untested inputs as a limit.
- [x] 6.2 A dated note on Logbook 091's counts; AGENTS.md's per-connection line names `sign_override_exclude`; a CHANGELOG line.
- [x] 6.3 The tracker: the correction before the readiness restarts, and the perturbation family's planned use of it.
