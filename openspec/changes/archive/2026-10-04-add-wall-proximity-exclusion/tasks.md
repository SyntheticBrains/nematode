# Tasks: Wall-proximity exclusion for the chemotaxis validation

Phase 8b housekeeping item H.3. Registered in
`docs/experiments/logbooks/supporting/035-realworm-chemotaxis-validation/wall-exclusion/launch.md`
before any capture ran.

- [x] 1. **The filter** — **done, five tests in `validation/test_behavioural_curves.py`.** Original scope:: `away_from_walls` in `behavioural_curves`, with tests.
- [x] 2. **The harness flags** — **done; refused one without the other through `argparse`; two harness tests.** Original scope:: `--wall-margin-mm` and `--arena-mm`, refused one without the other; the
  summary block when on and nothing when off; tests for both paths.
- [x] 3. **Capture configs** — **done: `*_capture.yml`, each its parent plus `capture_behaviour: true`, and the control's `chemotaxis_mode: derivative`.** Original scope:: 035's three arms as one-key deltas from their parents.
- [x] 4. **Re-capture** — **done: 24/24 runs in 75 s, `campaigns/h3-wall-recapture`, archived off-repo.** Original scope:: seeds 42–49, 300 episodes, three arms, through `run_campaign.py` with the
  output controls.
- [x] 5. **The readings** — **done: `wall_exclusion_check.py` with eight tests; identity holds on 21 of 24 seeds, and the three that differ re-run byte-identically; the sensing arms do not move, and the control's thresholded weathervane goes REPRODUCED -> PARTIAL at both margins, marginally at 1.0 mm. Branch review added a floor-held reading: with each seed's creep floor held at its unexcluded value, the 1.0 mm verdict stays REPRODUCED by a lower bound of +0.00001.** Original scope:: identity against 035 with the exclusion off; the 1.0 mm primary and the 2.0 mm
  sensitivity margins; the comparison script and its tests.
- [x] 6. **Docs** — **done.** Original scope:: 035's dated note, the tracker's H.3, a `CHANGELOG.md` line.
- [x] 7. **Close-out** — **done: full suite 6,992 passed; hooks pass with everything staged, judged by exit code; validated; archived.** Original scope: the full suite; `git add -A` then `uv run pre-commit run --all-files`, judged by
  exit code; `openspec validate --strict`; archive and PR.
