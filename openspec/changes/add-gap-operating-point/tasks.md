# Tasks: G.0 — the gap model's operating point

Phase 9's first act (roadmap v4.5 § Phase 9, D23, D24; tracker `phase9-tracking` G.0). Registered in
`docs/experiments/logbooks/supporting/095-gap-operating-point/launch.md` before any scored run.

- [ ] 1. **Rewiring**: a held set and a focal set for `_undirected_double_edge_swap`; `held` draws
  both edges from the non-held pairs, `focal` draws the first from the focal pairs; the accepted count
  returned. Tests: held pairs keep place and count; every focal pair moves when partners allow; degrees
  preserved; the existing nulls draw the graphs pinned before this change.
- [ ] 2. **Brain options**: `gap_weight_mode`, `gap_weight_scale`, `gap_outlier_min_count`, and the
  wirings `rewired_gap_outliers_held` and `rewired_gap_outliers_scattered`; refusals (scale or mode
  away from default with gap junctions off; the threshold read only by the two wirings); the comment
  corrected. Tests: each option's matrix against a hand computation; `plastic_gaps` composes;
  every default byte-identical (parameters, outputs, the next RNG draw).
- [ ] 3. **Configs** through `generate_gap_operating_point_configs.py`: the Emmons t35 anchor and every
  panel arm as stated deltas; loader tests.
- [ ] 4. **The shortcut audit** takes the cell's config; a test on the thermal cell's sensor set.
- [ ] 5. **Analysis** `gap_operating_point.py`: `anchor` (D23's rule), `score` (the readings,
  corrected together, and the verdict map), `predictor` (Decision E), `table` (Decision F), the
  per-seed CSV; in the gate preflight's shape; tests on synthetic manifests, including every verdict
  branch and `no_gap_effect`.
- [ ] 6. **The anchor pilot** on 9801–9816 (64 runs) and the two-seed smokes on 9851–9852; preflight;
  `m` fixed by D23's rule; the cost measured.
- [ ] 7. **Registration**: the launch record (arms, seeds, `m`, the verdict map, the predictor, the
  table, the retention line, D23 and D24 cited), committed; `/nematode-review-spec`; tracker H.2, H.3
  and G.0(a) ticked.
- [ ] 8. **The predictor** on Logbook 082's committed draws, computed only after the launch record is
  committed.
- [ ] 9. **Campaign** from a separate worktree, seeds 2401–2464, ≈ 848 runs, with the output controls.
- [ ] 10. **Readout**: Logbook 095; the verdict recorded in the tracker and the roadmap's exit criteria;
  CHANGELOG.
- [ ] 11. **Close-out**: full suite (`uv run pytest -q -m "not nightly"`), `pre-commit run -a`,
  `openspec validate --strict`, archive, PR.
