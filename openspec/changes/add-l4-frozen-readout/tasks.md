# Tasks

## 1. The preparation

- [ ] 1.1 `scripts/prepare_readout_checkpoints.py`: per seed, write a checkpoint carrying a substituted
  `readout` and **every other tensor at that seed's own fresh initialisation**, built by constructing the
  baseline arm's brain at that seed and swapping one tensor.
- [ ] 1.2 Two sources: `ppo`, harvested from that seed's PPO run on this cell, and `rotated`, a random
  direction at the **same Frobenius norm** as that seed's PPO readout, drawn from a seeded generator so
  the arm reproduces.
- [ ] 1.3 A sidecar per checkpoint recording the source run, the readout's norm before and after, and the
  cosine between them — so a prepared file describes its own provenance.
- [ ] 1.4 Tests: a prepared checkpoint differs from the baseline arm's initialisation **in `readout`
  alone** (`log_std`, `food_gains` and `w_chem` bit-identical); the rotated readout matches the PPO
  readout's norm and differs in direction; preparation is deterministic at a seed; a missing source run
  fails rather than silently producing an unmodified checkpoint.
- [ ] 1.5 **The load-path equivalence test, which decides whether R.1c's arms may serve as the anatomical
  comparator.** Construct the baseline brain at a seed; load a prepared checkpoint carrying the
  **anatomical** readout into a second identically-constructed brain; assert **every tensor bit-identical**.
  A load also calls `reset_state()` and `buffer.reset()`, so this is what establishes that those are inert
  when the loaded tensors match construction — rather than reasoning that they are.
- [ ] 1.6 Record which branch 1.5 selected. **Passing** ⇒ R.1c's committed `motor` pair is the anatomical
  arm. **Failing** ⇒ the anatomical pair is re-run through the load path (16 runs), and R.1c's values
  become a cross-check on what loading changed rather than the comparator.

## 2. The arms

- [ ] 2.1 Five configs from R.1c's committed `motor` pair: `ppo` and `rotated`, each learning and frozen,
  plus the PPO harvest config — the last pinned to **`initial_log_std: -1.0`**, since the committed
  hard-food PPO config trains at the default std 1.0 and a readout adapted to an action distribution the
  rule does not use would be a second co-adaptation. The `anatomical` pair is R.1c's committed arms
  **subject to task 1.5**, and a sixth and seventh config (anatomical learning and frozen, loaded) are
  added only if that test fails.
- [ ] 2.2 Exact-key test: each new arm differs from R.1c's `motor` arm by `weights_path` alone (and
  `freeze_updates` for the frozen ones).
- [ ] 2.3 Confirm σ 0.1, `initial_log_std: -1.0` and `plasticity_perturbation_set: motor` are unchanged
  across all of them — the operating point R.1c measured, held fixed.

## 3. Harness

- [ ] 3.1 A **sibling module**, `scripts/analysis/l4_frozen_readout.py`, not an extension of R.1c's: that
  harness keys its arms by mask with mask-specific dimension columns, and this change's verdict names
  differ. Reuses its statistics layer and its drift reader. Per-arm contrast against **its own** frozen
  control, paired one-sided, BH-FDR across the three readouts.
- [ ] 3.2 Both registered minima, as in R.1c, against each arm's own frozen mean — with the reachable gap
  taken against the **matched** PPO level from the harvest (same 8 seeds, same `initial_log_std`), 058's
  32-seed 19.31 reported beside it, and the more demanding binding.
- [ ] 3.3 **Beats-its-floor and reaches-competence reported separately**, since only the second bears on
  R.1b — the distinction R.1c had to introduce after its verdict condition proved weaker than its
  consequence.
- [ ] 3.4 Credited-synapse drift per arm, to see whether a better readout moves the 1.37–1.38× that was
  invariant across every dimension in R.1c.
- [ ] 3.5 The `rotated` arm's position reported in **every** verdict branch, not only the positive one.
- [ ] 3.6 Tests for 3.1–3.5, including a fixture where `rotated` matches `ppo` and one where `ppo`
  underperforms `rotated`.

## 4. The stop clauses

- [ ] 4.1 Re-score R.1c's committed `motor` logs through this harness and confirm learning **3.751** and
  frozen **3.150**. A mismatch means the harnesses disagree and nothing here is interpretable.
- [ ] 4.2 Confirm the PPO harvest runs learn the cell; a failed harvest voids the arm rather than
  producing a negative.
- [ ] 4.3 `launch.md` committed before anything runs.

## 5. Campaign

- [ ] 5.1 8 PPO harvest runs at seeds 1–8 at `initial_log_std: -1.0`, with `--track-experiment`. These
  supply both the readouts and the matched PPO reference.
- [ ] 5.2 32 arm runs: two readouts × (learning, frozen) × 8 seeds at 3000 episodes, with
  `--track-experiment` so drift has weights to read.
- [ ] 5.3 Per-seed CSV, the per-readout table and the verdict under `supporting/062-l4-frozen-readout/`.

## 6. The record

- [ ] 6.1 Logbook 062: the three-readout table with the `rotated` arm's position, the drift column, the
  verdict against the three registered outcomes, and the co-adaptation caveat stated as a limit on the
  negative branch.
- [ ] 6.2 The experiments index row.
- [ ] 6.3 State plainly that this **cannot** satisfy D1 whatever it returned, and name the plausible
  follow-up a positive result licenses — a better-grounded readout, not PPO.
- [ ] 6.4 State whether R.1b's blocker moved, and whether R.2 is now the live path.

## 7. Close-out

- [ ] 7.1 `CHANGELOG.md`; the tracker (R.1d); the roadmap only if the reading changes.
- [ ] 7.2 Confirm no committed verdict changed, and that R.1c's `motor` values reproduce.
