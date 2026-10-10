# Tasks: D.1b — patch-leaving

- [x] 1. **Reference** — **done 2026-10-11:** the windows file gains `off_lawn` (other arrays byte-identical); `lawn_exits`, `pre_exit_band` and `last_window_before_exits` are shared by real and simulated tracks. On 212 sustained exits from 184 animals, the classifier's band enrichment is **+0.288** (authors' labels +0.227) and the last-window enrichment +0.467. Vendored with provenance. Tests. Original scope: (first, while the deposit's pickle is on disk):
  `scripts/analysis/patch_leaving_reference.py` derives the real worms' exits (the same rule:
  off-lawn for a full window) aligned to the windows, their 30–180 s band and last-window enrichment
  under the calibrated classifier and under the authors' labels, and the leaving rate; vendored as `data/roaming_dwelling/scheer2023_leaving.json` with
  provenance. Tests.
- [ ] 2. **Readings**: exits, pre-exit enrichment, leaving rate and the dwelling-end statistic in
  `lawn_states.py`; the one-sample registered reading and the leaving gate. Tests.
- [ ] 3. **Cell**: the leaving cell's configs from the generator. Tests.
- [ ] 4. **Pilot and registration**: a small sweep of lawn size and count on disjoint seeds until the
  learner learns and leaves; `088-patch-leaving/launch.md` (the minimum, the exit gate, the
  sizing); gate preflight; `/nematode-review-spec`.
- [ ] 5. **Campaign and readout**: the panel; Logbook 088; tracker D.1b; roadmap.
- [ ] 6. **Close-out**: full suite; hooks by exit code; validate; archive; PR.
