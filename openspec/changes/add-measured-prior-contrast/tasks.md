# Tasks: The wiring × weight-prior 2×3

Phase 8 task **B.1c**. The plan is in `docs/roadmap.md` § Phase 8, **D17**. The decisions taken before
implementation are in this change's `design.md`.

## Panel

- [ ] 1. **The shuffled level in the shared vocabulary** (Decision H):
  - `measured_prior_pilot` gains the shuffled level's keys and stem rule;
  - its `build_manifest` and `require_complete` take a level table, defaulting to the pilot's own;
  - B.1b's tests pass unchanged.
- [ ] 2. **The contrast module**, `scripts/analysis/measured_prior_contrast.py`:
  - the stems, levels (random, measured at 1.0, shuffled at 1.0), seeds and arm map;
  - the gates per level against its own floor;
  - the two interactions per learner on both metrics, with the per-contrast metric choice;
  - the two families' correction;
  - the state classification (Decision D) and the verdict map (Decision E), with the unreadable and
    Lee cases (Decision F);
  - the drift check;
  - the per-seed CSV (`lineterminator="\n"`) and the analysis JSON, including coverage at head
    scope and full scope from `quantumnematode.connectome.measured_weights.coverage`.
- [ ] 3. **The shuffled configs**: the generator writes the 8 `_measured_shuffled` arms. Nothing
  existing is rewritten.
- [ ] 4. **Tests** (`tests/.../analysis/test_measured_prior_contrast.py`):
  - every new config loads through the real loader and differs from its parent only in
    `weight_prior`;
  - the random and measured levels are the committed B.1b configs;
  - each shuffled arm builds weights distinct from its measured arm, with the same multiset on
    covered edges;
  - each floor is built from its learning arm's weights;
  - the seeds are fresh and disjoint;
  - every state and every verdict row is checked on synthetic interactions, including unreadable
    and Lee;
  - the primary family is exactly the four registered interactions.

## Registration and run

- [ ] 5. **Pre-launch checks**:
  - the full suite (`uv run pytest -q -m "not nightly"`);
  - `git add -A`, then `uv run pre-commit run --all-files`, judged by its exit code;
  - `openspec validate --strict`;
  - 8-episode smoke runs of one shuffled arm per learner.
- [ ] 6. **The launch record**, `docs/experiments/logbooks/supporting/073-measured-prior-contrast/launch.md`,
  committed **before any seed runs**: arms, seeds, both interactions, metric rule, sensitivity table,
  minimum, state classification, verdict map, gates, standing conditions, retention and cost.
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
