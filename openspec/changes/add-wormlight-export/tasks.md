# Tasks: Vendor the Emmons 2024 release, parse neuromuscular connections, and export for Wormlight

Not a Phase 8 task. The decisions are in this change's `design.md`. No campaign runs in this change,
and no experiment's input changes.

- [ ] 1. **Vendor the file**: `data/connectome/emmons_2024_s1_connectome_adjacency.xlsx`, the S1 File
  of Emmons 2024 as the publisher serves it (`pbio.3002939.s001.xlsx`, 4,176,688 bytes, SHA256
  `e866b43f19ba5c70b773c94efd06aff6d6b2887cd24eed4412da80c06986418d`), stored through Git LFS by the
  existing rule.

- [ ] 2. **Provenance**: a `PROVENANCE.md` entry in the house shape (description, original filename,
  size, SHA256, source URL, licence, retrieval date, paper, redistribution rationale), the comparison
  with the 2019 file, and its line in `## Verification`.

- [ ] 3. **Shared parse and the wiring loader**: move the Cook 2019 loader's body into one function
  taking the path and sheet names, with its output unchanged, and add `load_emmons_2024_hermaphrodite()`
  on it with the digest checked.

- [ ] 4. **Neuromuscular parse**: `connectome/muscles.py` naming the 95 body wall muscles,
  `NeuromuscularJunction` in `model.py`, and `load_emmons_2024_neuromuscular()`, refusing a sheet that
  does not list each muscle exactly once. No planning references in package code.

- [ ] 5. **Loader tests**: the digest matches and appears in `PROVENANCE.md`; a changed or missing file
  is refused; 302 neurons; chemical synapses identical to the Cook 2019 loader's (3,709, 20,965
  sections, 38 autapses); 1,095 gap-junction pairs and 5,864 sections, differing from Cook 2019's in
  exactly the four BDU pairs; 956 neuromuscular entries from 162 neurons, 5,515 sections, onto all 95
  muscles; the same neuromuscular entries parsed from the 2019 file; a sheet missing or repeating a
  muscle refused; the Cook 2019 loader's counts unchanged.

- [ ] 6. **The exporter**: `scripts/export_wormlight.py` building `wormlight.connectome/1` from public
  package API, rendering one record per line, and refusing a dirty tree without `--allow-dirty`.

- [ ] 7. **Exporter tests**: counts and section totals; neurons in name order with the table's classes,
  the atlas's identities primary first and the rule's sign; gap pairs canonical; provenance digests
  equal the files'; the rendering parses back to the same object with one record per line; unknown or
  missing keys refused; a dirty tree refused, and `--allow-dirty` recorded.

- [ ] 8. **Docs**: a CHANGELOG line under *Unreleased*, the exporter in `AGENTS.md`'s commands, and the
  connectome package and loader docstrings updated.

- [ ] 9. **Close-out**: the full suite via `uv run pytest -m "not nightly"`; `git add -A` then
  `uv run pre-commit run --all-files`, judged by its exit code; one export written to the scratchpad and
  re-run to confirm identical bytes; `openspec validate --strict`; archive and PR.
