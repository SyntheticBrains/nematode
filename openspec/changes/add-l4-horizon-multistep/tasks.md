# Tasks

## 1. The arms

- [ ] 1.1 Six configs from the committed yardstick: the node-perturbation rule at σ = 0.2 with
  `trace_decay` ∈ {0.9, 0.99, 0.999}, each with a `freeze_updates: true` sibling at the same
  horizon. One `freeze_updates` key per file, checked before the run.
- [ ] 1.2 Test: each pair differs from the other only in `trace_decay`, and each frozen arm differs
  from its learning arm in `freeze_updates` alone.

## 2. The reading

- [ ] 2.1 An analysis entry point scoring each horizon's learning arm against **its own** frozen
  control, paired by seed, on plateau-tail mean foods, one-sided, BH-FDR across the three horizons,
  using I.2's family. A verdict of "transferred" requires significance **and** a shift of at least
  **0.5 foods**; a smaller shift is reported and licenses nothing.
- [ ] 2.1b Each arm reports its weight distance from its frozen control beside its score, so a
  horizon too short (little drift, no gain) is distinguishable from one too long (substantial drift,
  no gain).
- [ ] 2.2 The competence-dependent contrasts are reported as undefined where no seed is competent,
  not as a null; the committed 040 yardstick values are carried as a descriptive reference and not
  as the comparator.
- [ ] 2.3 Tests: a floored primary metric does not decide the comparison; a frozen arm from another
  horizon is not accepted as the comparator; a significant shift below the 0.5-foods floor does not
  read as transferred and licenses nothing.

## 3. Run and record

- [ ] 3.1 Launch record before the run: the arms, the comparison, the 0.5-foods floor, the four
  outcomes and what each licenses, the honest prior that this is expected to fail, and the
  disclosure that the I.3 pilot ran on seeds 101–102, disjoint from the registered 1–8.
- [ ] 3.2 Run: 6 arms × 8 seeds × 3000 episodes, from `main`, no branch switch while it runs.
- [ ] 3.3 Records under `supporting/055-l4-horizon-multistep/`.

## 4. Close-out

- [ ] 4.1 `details.md`: whether the horizon transfers off the control, in the terms the outcomes
  registered.
- [ ] 4.2 `CHANGELOG.md`; tracker (I.3b) and roadmap.
- [ ] 4.3 State what I.4 takes from this — on either outcome, since a negative is an input too.
