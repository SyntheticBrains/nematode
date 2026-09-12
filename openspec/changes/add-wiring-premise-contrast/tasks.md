# Tasks

## 1. Configs

- [ ] 1.1 The klinotaxis pair: a rewired-null variant of
  `configs/scenarios/foraging/connectomeppo_small_continuous2d_fick_adaptive_klinotaxis.yml`
  differing by `wiring: rewired_degree_preserving` alone, and a frozen-weights variant of each
  wiring differing by `freeze_updates: true` alone.
- [ ] 1.2 The thermal pair, the same way, from the committed thermal cell config.
- [ ] 1.3 A test that each new config differs from its base by exactly the intended key, so a pair
  cannot silently diverge on anything else.

## 2. Harness

- [ ] 2.1 `scripts/analysis/wiring_premise.py`: the eight-test family, both cells corrected
  together, one-sided paired Wilcoxon with 80% bootstrap CIs, the gates read before the primaries,
  the saturation clause, the registered minimum effect, the verdict order, and a completeness flag
  per test. Fixed in code before the campaign runs.
- [ ] 2.2 I.2's mixture family read as the registered secondary through
  `scripts/analysis/l4_mixture_statistic.py`, not reimplemented.
- [ ] 2.3 The MLP reference arm carried as a descriptive row that no test can read.
- [ ] 2.4 Tests: the verdict order including `no_learning` and `saturated` ahead of the primary; the
  minimum-effect clause; a pinned reproduction of the family from a fixture table.

## 3. Pilot (disjoint seeds 101–108)

- [ ] 3.1 `launch.md` committed before the pilot runs.
- [ ] 3.2 Convergence of the connectome on C1 at the inherited recipe, reported either way.
- [ ] 3.3 Measured per-run wall time on these exact configs, and the campaign schedule derived from
  it rather than scaled from a lighter run.
- [ ] 3.4 Distance from the saturation threshold, with the registered remedy taken if it is met.

## 4. Campaign (registered seeds 1–16)

- [ ] 4.1 Launch record committed first; no branch switches while it runs.
- [ ] 4.2 160 runs: four connectome arms × two cells × 16 seeds, plus the MLP reference.
- [ ] 4.3 Per-seed CSV, the family table, and the verdict per cell under
  `supporting/057-wiring-premise-contrast/`.

## 5. The record

- [ ] 5.1 Logbook 057, following the template: the gates before the contrasts, both verdicts, what
  each licenses, and the committed C3 results carried beside them unchanged.
- [ ] 5.2 The experiments index row.
- [ ] 5.3 What this means for 7b's gate and for the shipment decision, stated rather than implied.

## 6. Close-out

- [ ] 6.1 `CHANGELOG.md`; tracker (new block V) and roadmap.
- [ ] 6.2 Confirm no committed verdict was changed.
