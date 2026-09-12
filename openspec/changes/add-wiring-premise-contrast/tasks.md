# Tasks

## 1. Configs

- [x] 1.1 The klinotaxis pair: a rewired-null variant of
  `configs/scenarios/foraging/connectomeppo_small_continuous2d_fick_adaptive_klinotaxis.yml`
  differing by `wiring: rewired_degree_preserving` alone, and a frozen-weights variant of each
  wiring differing by `freeze_updates: true` alone.
- [x] 1.2 The thermal pair, the same way, from the committed thermal cell config.
- [x] 1.3 A test that each new config differs from its base by exactly the intended key, so a pair
  cannot silently diverge on anything else, and that no config sets `rewire_seed`, so the null's
  rewiring is derived from the run seed and the pairing holds per seed.

## 2. Harness

- [x] 2.1 `scripts/analysis/wiring_premise.py`, **extending
  `scripts/analysis/connectome_structure_controls.py`** and importing its metric and statistics
  layers (`t7_continuous_ranking.plateau_tail`,
  `weight_search_architecture_ranking.paired_seed_wilcoxon_bootstrap`, `bh_fdr`) rather than
  reimplementing them: the eight-test family corrected together, the gates read before the
  primary, the saturation clause and its named remedy, the registered minimum effect, the verdict
  order for the primary cell with the thermal cell as an annotation, and a completeness flag per
  test. Fixed in code before the campaign runs.
- [ ] 2.2 I.2's mixture family read as the registered secondary through
  `scripts/analysis/l4_mixture_statistic.py`, not reimplemented.
- [x] 2.3 The MLP reference arm carried as a descriptive row that no test can read.
- [x] 2.4 Tests: the verdict order including `no_learning` and `saturated` ahead of the primary; the
  minimum-effect clause; a pinned reproduction of the family from a fixture table.

## 3. Pilot (disjoint seeds 101–104)

- [x] 3.1 `launch.md` committed before the pilot runs.
- [x] 3.2 Convergence of the connectome on C1 at the inherited recipe, reported per pilot seed
  either way, and the config 035's connectome companion ran recorded beside it.
- [x] 3.3 Measured per-run wall time on these exact configs, and the campaign schedule derived from
  it rather than scaled from a lighter run.
- [x] 3.4 Distance from the saturation threshold, with the registered remedy taken if it is met.
- [x] 3.5 The remedy applied once as registered, and its outcome recorded: it did not unsaturate
  either cell, which is what the dated amendment in the design responds to.
- [x] 3.6 The harder-variant arms (`target_foods_to_collect: 20`), each differing from its committed
  base by that key plus its own arm key, with `rewire_seed` unset and both properties tested.
- [x] 3.7 The efficiency axis wired in through the committed 034 harness, with the registered
  minimum effect on time-to-competence, and tested.

## 4. Campaign (registered seeds 1–16)

- [ ] 4.1 Launch record committed first; no branch switches while it runs.
- [ ] 4.2 128 runs at 3000 episodes: four connectome arms × two cells × 16 seeds at the remedy's
  `target_foods_to_collect: 20`, plus the MLP reference.
- [ ] 4.3 Per-seed CSV, the family table, and the verdict per cell under
  `supporting/057-wiring-premise-contrast/`.

## 5. The record

- [ ] 5.1 Logbook 057, following the template: the saturation the pilot found and the amendment it
  forced, the gates before the contrasts, both axes, both verdicts, what each licenses, and the
  committed C3 results carried beside them unchanged.
- [ ] 5.2 The experiments index row.
- [ ] 5.3 What this means for 7b's gate and for the shipment decision, stated rather than implied.

## 6. Close-out

- [ ] 6.1 `CHANGELOG.md`; tracker (new block V) and roadmap.
- [ ] 6.2 Confirm no committed verdict was changed.
