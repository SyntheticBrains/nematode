# Tasks: G.0-A — the corrected model's point-worm contrast on log-compressed counts

Phase 9 G.0-A (roadmap v4.5 § Phase 9; tracker `phase9-tracking` G.0-A), after
[Logbook 097](../../../docs/experiments/logbooks/097-gap-substrate.md) chose log-compressed counts.

- [x] 1. **Harness** `scripts/analysis/logcount_contrast.py` (design Decision F). Its subcommands are
  `plan`, `preflight`, `table` and `score`, and its verdict map follows Decision B. Its D23 rule reuses
  `anchor_minimum`, and its manifest verifies each run's recorded config. It exposes `STEMS` for
  `gate_preflight`. Tests cover every verdict row,
  the D23 cases, refusals and seed disjointness. `.test_durations` is regenerated. *(At close-out, so the regeneration does not compete with a running
  campaign; 19 tests leave the freshness hook far under its 10% bar.)*
- [x] 2. **Before launch** (Decisions D and E):
  - `table.json` on the 64 panel seeds;
  - `preflight.json` on the G.0-A pilot's log arms;
  - the cost from the pilot's own run times;
  - `/nematode-review-spec add-logcount-gap-contrast`, with findings fixed or recorded.
- [x] 3. **Registration**: `docs/experiments/logbooks/supporting/099-logcount-contrast/launch.md`,
  committed before any panel run. It fixes the arms, seeds, reading, minimum, verdict map, operating
  characteristic, gates, D24 audit, cost, launch commands and retention. The tracker's G.1–G.3b items
  gain a note, citing the new protocol requirement, that none registers on log counts before this panel
  has read.
- [x] 4. **The panel** from a separate worktree: 256 runs with the output controls, after M.1(b)'s
  rounds release the machine.
- [x] 5. **Readout**: Logbook 099 (the reading, the D23 minimum if `placement`, and the audit and
  plateaus beside). Update the tracker's G.0-A item, the roadmap's G.0 entry and the CHANGELOG.
- [x] 6. **Close-out**: full suite, `pre-commit run -a`, `openspec validate --strict`, archive, PR.
