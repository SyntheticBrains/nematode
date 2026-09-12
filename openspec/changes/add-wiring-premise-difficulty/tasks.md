# Tasks

## 1. The cell

- [ ] 1.1 Four configs from the committed food-only `_t20` base: the wild type with
  `max_steps: 500` and `satiety_gain_per_food: 0.2`, its rewired-null variant, and a frozen-weights
  floor for each wiring.
- [ ] 1.2 Extend the exact-key test to this family: each config differs from its base by the budget
  keys plus its own arm key and nothing else, with `rewire_seed` unset.

## 2. Harness

- [ ] 2.1 Add the cell to `scripts/analysis/wiring_premise.py` — its scored metric, its minimum
  effect (the registered 20% on time-to-competence), its four family rows, and its place among the
  cells whose efficiency contrast decides their own campaign.
- [ ] 2.2 Confirm the family correction is per campaign: a manifest carrying only this cell corrects
  across its four tests, and 057's committed manifests still reproduce their committed q values.
- [ ] 2.3 Tests for the new cell's rows, its verdict order, and the reproduction in 2.2.

## 3. Pilot (disjoint seeds 101–104)

- [ ] 3.1 `launch.md` committed before the pilot runs.
- [ ] 3.2 Does the cell learn under the tightened budget, per pilot seed.
- [ ] 3.3 Measured per-run wall time, and the campaign scheduled from it.
- [ ] 3.4 Distance from the ceiling, with the registered remedy applied once if it is met.

## 4. Campaign (registered seeds 1–32)

- [ ] 4.1 Launch record committed first; no branch switches while it runs.
- [ ] 4.2 128 runs: four arms × 32 seeds at 3000 episodes.
- [ ] 4.3 Per-seed CSV, the family table, both axes and the verdict under
  `supporting/058-wiring-premise-difficulty/`.

## 5. The record

- [ ] 5.1 Logbook 058: the three-way difference between the cells, what this arm isolates and what it
  cannot, the gates before the contrast, the verdict, and what it licenses.
- [ ] 5.2 The experiments index row.
- [ ] 5.3 State what the result does to 057's claim — generalise it, or leave it narrow — and what
  remains unseparated either way.

## 6. Close-out

- [ ] 6.1 `CHANGELOG.md`; tracker (V.3) and the roadmap if the reading changes.
- [ ] 6.2 Confirm no committed verdict was changed.
