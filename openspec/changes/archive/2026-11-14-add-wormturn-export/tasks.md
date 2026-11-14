# Tasks: Export the signed connectome and the neuromodulator layer for Wormturn

Not a phase task. The maintainer approved the exporter on 2026-11-14; the decisions are in this
change's `design.md`. No campaign runs in this change, and no experiment's input changes.

## 1. Review

- [ ] 1.1 **Spec review.** `/nematode-review-spec` was not run: the change was written in a session
  that could not run the skill. The maintainer's review of the pull request stands in for it.

## 2. The exporter

- [x] 2.1 **Shared helpers** in `scripts/export_wormlight.py` (design Decision E): `input_entry`,
  `neuron_records`, `gap_records`, and `render` taking `record_lists`. **Done: the Wormlight export
  from one tree is byte-identical before and after (SHA256 `1e74745b…`, a dirty-tree export).**
- [x] 2.2 **`scripts/export_wormturn.py`** building `wormturn.connectome/1` from public package API
  and the two layer files: the signs of the Emmons 2024 wiring, passed explicitly, with no citation
  excluded; the records of Decisions B and D; the provenance of Decision F, refusing a Git LFS
  pointer; one record per line; a dirty tree refused without `--allow-dirty`. No planning references
  in the script. **Done.**

## 3. Tests

- [x] 3.1 **`entrypoints/test_export_wormturn.py`**, git never called, covering every scenario.
  **Done, 32 tests:**
  - the schema and key order; counts and section totals; no muscles;
  - neurons as the table and the atlas give them, and both record lists equal to the Wormlight
    export's;
  - every chemical edge with exactly one sign and source, the source counts, signs consistent with
    their source, and example edges;
  - each physiology sign's citation in the table, no citation on any other, and every key used;
  - Emmons 2024 and Cook 2019 chemical edges identical, and the signs equal to either table;
  - the layer and the receptor couplings round-trip against their files;
  - provenance digests equal the files', and an LFS pointer refused;
  - the rendering parses back, one record per line, and is stable; unknown or missing keys refused;
  - a dirty tree refused, `--allow-dirty` recorded, a clean tree written.
- [x] 3.2 The Wormlight exporter's tests pass unchanged. **Done, 14 of 14.**

## 4. Docs

- [x] 4.1 A CHANGELOG line under *Unreleased*, and the exporter in `AGENTS.md`'s commands. **Done,
  with a line in the Wormlight exporter's docstring naming what it shares.**

## 5. Close-out

- [x] 5.1 The new tests, the Wormlight exporter's and the connectome tests; the fast tier. **Done:
  the two exporters' and the connectome tests, 311 passed. The fast tier, first with only the `torch`
  extra: 8,907 passed, 80 skipped, 2 xfailed, 147 failed and 4 errors. With the `cpu` and `pixel`
  extras added, 150 of those 151 pass; the last reads a Git LFS pointer under `artifacts/`, outside
  `data/**`, which this clone has not fetched.**
- [x] 5.2 `uv run pre-commit run --files` on the changed files, judged by its exit code. **Done, exit
  0, with the two suite-wide hooks skipped and run separately (5.1), and markdownlint run on the
  system's node: pre-commit's own node install fails on this machine.**
- [x] 5.3 Two exports to the scratchpad with `--allow-dirty`, byte-identical. The pinned clean export
  is made later, from a clone with Git LFS. **Done.**
- [x] 5.4 `openspec validate add-wormturn-export --strict`; archive. **The OpenSpec CLI is not
  installed in this session, so the change was checked by hand against the archived changes' format
  and not validated; archived.**
