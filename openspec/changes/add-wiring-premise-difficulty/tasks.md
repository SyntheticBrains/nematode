# Tasks

## 1. The cell

- [ ] 1.1 Four configs from the committed food-only `_t20` base at **each calibration grid point**
  (`max_steps` ∈ {150, 250, 350}) with `satiety_gain_per_food: 0.2`: the wild type, its rewired-null
  variant, and a frozen-weights floor for each wiring. Only the chosen budget's four configs are kept
  once the pilot freezes it; the others are recorded in the launch record as the grid that was run.
- [ ] 1.2 Extend the exact-key test to this family: each config differs from its base by the budget
  keys plus its own arm key and nothing else, with `rewire_seed` unset.

## 2. Harness

- [ ] 2.1 Add the cell to `scripts/analysis/wiring_premise.py` — its `SCORED` metric (full-clear
  success, as the food-only cell was) and its `MIN_EFFECT` peak-axis entry (5 points) even though the
  peak axis now carries only the gates; the registered 20% minimum on time-to-competence for the
  efficiency primary; its four family rows; and its place among the cells whose efficiency contrast
  decides their own campaign. `PRIMARY_CELL` widens from a string to a set — it feeds only a printed
  label, so **no committed number in 057 depends on it**.
- [ ] 2.2 Confirm the family correction is per campaign: a manifest carrying only this cell corrects
  across its four tests, and one carrying two cells corrects across eight. Pin it with a **synthetic
  fixture**, not 057's committed manifests — `campaigns/*` is gitignored, so those manifests' log
  paths do not exist on a clean checkout and the check could not run in CI.
- [ ] 2.3 Tests for the new cell's rows, its verdict order, and the reproduction in 2.2.

## 3. Pilot (disjoint seeds 101–104)

- [ ] 3.1 `launch.md` committed before the pilot runs.
- [ ] 3.2 Per grid point and per pilot seed: does the cell learn, how far are the PPO arms from the
  ceiling, and **do both cross the 30% rolling full-clear threshold the primary metric needs** — the
  band has two edges and a censored metric looks like a null.
- [ ] 3.3 Measured per-run wall time at the chosen budget, and the campaign scheduled from it.
- [ ] 3.4 One budget frozen and recorded with the grid that produced it. If no grid point lands inside
  the band, the campaign does not launch and the change is amended under a dated note.

## 4. Campaign (registered seeds 1–32)

- [ ] 4.1 Launch record committed first; no branch switches while it runs.
- [ ] 4.2 128 runs: four arms × 32 seeds at 3000 episodes, at the frozen budget.
- [ ] 4.3 Per-seed CSV, the family table, both axes and the verdict under
  `supporting/058-wiring-premise-difficulty/`.

## 5. The record

- [ ] 5.1 Logbook 058: which of the cells' differences actually bind (two, not three — starvation
  ends 0.6% of thermal episodes), the calibration and the grid it chose from, what this arm isolates
  and what it cannot, the gates before the contrast, the verdict, and what it licenses.
- [ ] 5.2 The experiments index row.
- [ ] 5.3 State what the result does to 057's claim — generalise it, or leave it narrow — and what
  remains unseparated either way.

## 6. Close-out

- [ ] 6.1 `CHANGELOG.md`; tracker (V.3) and the roadmap if the reading changes.
- [ ] 6.2 Confirm no committed verdict was changed.
