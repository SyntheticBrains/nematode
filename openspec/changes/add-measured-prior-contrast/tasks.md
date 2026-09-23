# Tasks: The wiring × weight-prior 2×3

Phase 8 task **B.1c**. The plan is in `docs/roadmap.md` § Phase 8, **D17**. The decisions taken before
implementation are in this change's `design.md`.

## Panel

- [x] 1. **done: `VOCABULARY` holds every level `level_keys` reads, `shuffled` included; the pilot's `LEVELS`, `ALL_LEVELS` and stem map are unchanged; `build_manifest` and `require_complete` take another panel's stem map and levels.** Original scope: **The shuffled level in the shared vocabulary, not in the pilot's panel** (Decision H):
  - `measured_prior_pilot` separates its level vocabulary (read by `level_keys` and `stem_for`, and
    gaining `shuffled`) from its own panel (`LEVELS`, `ALL_LEVELS`, the stem map, unchanged);
  - its `build_manifest` and `require_complete` take another panel's levels and stem map, defaulting
    to the pilot's own;
  - B.1b's tests pass unchanged, and a test asserts the pilot's panel does not contain `shuffled`.
- [x] 2. **done: `scripts/analysis/measured_prior_contrast.py`; the family spans both learners, so both campaigns are scored in one invocation.** Original scope: **The contrast module**, `scripts/analysis/measured_prior_contrast.py`:
  - the stems, levels (random, measured at 1.0, shuffled at 1.0), seeds and arm map;
  - the gates per level against its own floor;
  - the two interactions per learner on both metrics, with `auc_success` as the registered primary
    on both learners and the censoring rule's own choice recorded beside it;
  - the two families' correction;
  - the state classification (Decision D) and the verdict map (Decision E), with the unreadable and
    Lee cases (Decision F);
  - the drift check;
  - the per-seed CSV (`lineterminator="\n"`) and the analysis JSON, including coverage at head
    scope and full scope from `quantumnematode.connectome.measured_weights.coverage`.
- [x] 3. **done: 8 written, 48 kept.** Original scope: **The shuffled configs**: the generator writes the 8 `_measured_shuffled` arms. Nothing
  existing is rewritten.
- [x] 4. **done: 45 tests in `test_measured_prior_contrast.py`; with B.1b's and A.2's, 311 pass.** Original scope: **Tests** (`tests/.../analysis/test_measured_prior_contrast.py`):
  - every new config loads through the real loader and differs from its parent only in
    `weight_prior`;
  - the random and measured levels are the committed B.1b configs;
  - each shuffled arm builds weights distinct from its measured arm, with the same multiset on
    covered edges;
  - each floor is built from its learning arm's weights;
  - the seeds are fresh and disjoint;
  - every state is checked on synthetic interactions, including the three `unresolved` cases;
  - every verdict row is checked, including both value-distribution directions, unreadable and Lee;
  - the primary family is exactly the four registered interactions.

## Registration and run

- [ ] 5. **Pre-launch checks**:
  - the full suite (`uv run pytest -q -m "not nightly"`);
  - `git add -A`, then `uv run pre-commit run --all-files`, judged by its exit code;
  - `openspec validate --strict`;
  - 8-episode smoke runs of one shuffled arm per learner.
- [ ] 6. **The launch record**, `docs/experiments/logbooks/supporting/073-measured-prior-contrast/launch.md`,
  committed **before any seed runs**: arms, seeds, both interactions, the metric departure and its
  reason, the sensitivity table with PPO's fan-in reference, the minimum, the state classification,
  the verdict map, gates, the per-seed shuffle, standing conditions, retention and cost.
- [ ] 7. **The campaigns**: PPO (seeds 225–256), then reading (257–304), with the output controls,
  and no branch switches until both complete.

## Records

- [ ] 8. **Score and commit** the per-seed CSVs and analysis JSON.
- [ ] 9. **Logbook 073**, following the logbook skill, with the verdict per learner, and each
  standing condition in the same sentence as its claim.
- [ ] 10. **Discharge:**
  - the index row;
  - tracker B.1c;
  - the roadmap's B.1 exit line, D17 and the novelty-map row;
  - citation sites the verdict conditions: Logbook 034 if null, block V if hides.
- [ ] 11. **Close-out**: validate, archive and open the PR.
