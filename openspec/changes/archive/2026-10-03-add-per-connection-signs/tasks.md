# Tasks: Per-connection chemical signs from vendored sources

Phase 8b housekeeping item H.2. No campaign runs in this change, and no experiment's input changes.

- [x] 1. **Vendor the files**: the Fenyves S1 and S5 Data files, re-downloaded from the publisher with
  their digests checked against Wormlight's pins, through Git LFS; the physiology table copied from
  Wormlight's `data/sign-overrides.csv`.
- [x] 2. **Provenance**: `PROVENANCE.md` entries for all three, in the house shape, with the licence
  and the redistribution rationale.
- [x] 3. **The loader**: `quantumnematode.connectome.signs` with the four-step precedence, the digest and
  agreement checks, and the override validation. No planning references in package code.
- [x] 4. **Tests** — **done, 25 in `connectome/test_signs.py`; the Wormlight agreement is a pinned digest of its table at `1190b3e`, since CI cannot read another repository.** Original scope: digests; the pinned composition; AWCL → AIYL; the Emmons 2024 table equals the Cook
  2019 one; every refusal; precedence on hand-built inputs; and agreement with Wormlight's export on
  every edge, skipped where Wormlight's export is absent.
- [x] 5. **Docs**: a `CHANGELOG.md` line; the Phase 8 tracker's H.1 (confirmed by the first digest) and
  H.2 ticked; the roadmap's C.1e note for Guan et al. 2026, the digest's find.
- [x] 6. **Close-out** — **done: full suite 6,970 passed; hooks pass with everything staged, judged by exit code; the spreadsheets routed through LFS; validated; archived.** Original scope: the full suite; `git add -A` then `uv run pre-commit run --all-files`, judged by
  exit code; `openspec validate --strict`; archive and PR.
