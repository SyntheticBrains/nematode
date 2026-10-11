# Tasks: D.1b — patch-leaving

- [x] 1. **Reference** — **done 2026-10-11:** the windows file gains `off_lawn` (other arrays byte-identical); `lawn_exits`, `pre_exit_band` and `last_window_before_exits` are shared by real and simulated tracks. On 212 sustained exits from 184 animals, the classifier's band enrichment is **+0.288** (authors' labels +0.227) and the last-window enrichment +0.467. Vendored with provenance. Tests. Original scope: (first, while the deposit's pickle is on disk):
  `scripts/analysis/patch_leaving_reference.py` derives the real worms' exits (the same rule:
  off-lawn for a full window) aligned to the windows, their 30–180 s band and last-window enrichment
  under the calibrated classifier and under the authors' labels, and the leaving rate; vendored as `data/roaming_dwelling/scheer2023_leaving.json` with
  provenance. Tests.
- [x] 2. **Readings** — **done: exits per episode (all three samples off makes a window off); band and last-window enrichment; leaving rate; the dwelling-end statistic; `read_leaving` (the learning gate, the exits gate, the one-sample reading at a minimum given on the command line); the analysis takes a `--cell`. Tests.** Original scope:: exits, pre-exit enrichment, leaving rate and the dwelling-end statistic in
  `lawn_states.py`; the one-sample registered reading and the leaving gate. Tests.
- [x] 3. **Cell** — **done: the generator's `leaving` cell (8 lawns of 1.0 mm, 5 cells each, placing on all 3,000 seeds tried) for the learner and its floor; D.1's configs unchanged. Tests.** Original scope:: the leaving cell's configs from the generator. Tests.
- [x] 4. **Pilot and registration** — **done 2026-10-11:** two pilots, three cells. The four 1.5 mm lawns pass every gate on every seed and are registered: minimum 0.10, at least 10 exits, roaming 5–95%, at least 12 read seeds, seeds 2101–2116. The likeliest verdict, no coupling, is stated in advance. Gate preflight passes. Original scope:: a small sweep of lawn size and count on disjoint seeds until the
  learner learns and leaves; `088-patch-leaving/launch.md` (the minimum, the exit gate, the
  sizing); gate preflight; `/nematode-review-spec`.
- [x] 5. **Campaign and readout** — **done 2026-10-11:** 32 runs on seeds 2101–2116, all succeeded. The registered reading is **unreadable**: 7 of 16 seeds take both states, against 12 required. Logbook 088; tracker D.1b unmet-with-reason; roadmap note; experiments index; changelog. Original scope:: the panel; Logbook 088; tracker D.1b; roadmap.
- [x] 6. **Close-out** — **done: full suite 7570 passed; hooks over all files by exit code; validated; archived.** Original scope:: full suite; hooks by exit code; validate; archive; PR.
