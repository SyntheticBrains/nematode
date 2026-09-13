# Tasks

## 1. S1 — the width axis on the one-step control

- [x] 1.1 Give `scripts/analysis/l4_rule_positive_control.py` a width parameter. `HIDDEN` stays the
  pinned default so every committed value from I.0–I.3 reproduces bit-for-bit; the width threads
  through `_actor` and `_build` only. Confirm by re-running the committed control at the default and
  diffing against the recorded JSON.
- [x] 1.2 `scripts/analysis/l4_perturbation_scale.py`: the S1 sweep over
  `HIDDEN ∈ {8, 16, 32, 64, 128}`, seeds 1–8, 20 000 trials, at I.1's passing configuration with width
  as the only axis. Per width, the **rule** arm and the **analytic reference** arm.
- [x] 1.3 Trials-to-criterion per seed: the first trial whose trailing 100-trial mean crosses the
  registered halfway threshold. Seeds that never cross are **excluded from the fit and counted**, with
  the censoring rate printed per width — a fit over crossers alone, with the non-crossers unreported,
  is the failure mode V.3 caught.
- [x] 1.4 Reachability: the analytic arm's gap fraction per width, and the rule's fraction normalised
  by it. A width where the reference misses the pass bar is **void at that width**; if `HIDDEN = 8`
  fails, the sweep is void and the run stops.
- [x] 1.5 The 1/N fit: OLS of `log2(trials)` on `log2(N)` over per-seed values, bootstrap CI over
  seeds, plus the descriptive Spearman of per-width medians. The slope bar (≥ 0.5, CI excluding 0) and
  the 128-unit pass/fail are both reported as the verdict's two inputs.
- [x] 1.6 The derived budget for 128 units and for the connectome's 302 units / 1208 draws per
  decision, emitted with an explicit `extrapolation: true` field and printed with that word — not a
  measurement, and both readings of the connectome's dimension given.
- [x] 1.7b **The depth control** *(amendment, 2026-09-13, after S1's grid returned flat)*: one cell at
  the yardstick's exact arrangement — `hidden 64, layers 2`, **128 perturbed units** — on the same
  control, reported beside the matched one-layer 128-unit cell. With the width sweep flat, depth is the
  only shape difference left between the platform the rule passes and the platform it fails. A depth
  parameter on the control with the pinned `HIDDEN_LAYERS = 1` as its default, so the committed values
  still reproduce.
- [x] 1.7 Tests: the default reproduces the committed control; the censoring count; the void clause at
  a failing reference; the fit on a synthetic exact-1/N series recovering slope 1; the extrapolation
  flag present on every derived figure.

## 2. S2 — the width axis on the hard-food cell

- [x] 2.1 Fifteen configs from the committed calibrated MLP food-only base: for each
  `actor_hidden_dim ∈ {4, 8, 16, 32, 64}`, the plastic node-perturbation arm, its **own** frozen
  control, and a PPO capability arm — all three carrying the hard-food cell's `max_steps: 350`,
  `target_foods_to_collect: 20` and `satiety_gain_per_food: 0.2`. The frozen control keeps
  `plasticity_node_noise: 0.2` and freezes only the update, so the perturbation's cost is matched
  across the pair; the capability arm shares `activation: tanh`, the width, `num_hidden_layers` and
  `initial_log_std: -1.0`, and every arm keeps the base's `entropy_coef: 0.05`.
- [x] 2.2 Exact-key test for the family: each config differs from the committed base by the cell keys,
  the recipe keys and its own width and arm keys, and nothing else. `critic_hidden_dim` moves with the
  width on the PPO arm and is stated; on the plastic arms the critic is never used and the record says
  so rather than leaving a stray key to be read as part of the manipulation.
- [x] 2.3 The S2 half of the harness: scan, plateau-tail mean foods through I.2's graded family,
  per-width paired one-sided Wilcoxon against that width's frozen control, BH-FDR across the five
  widths, and the trend against N.
- [x] 2.4 Both effect minima enforced together — 1.0 foods of 20, **and** 10% of that width's own
  PPO-minus-frozen gap — with a downgrade to `below_min_effect` naming which minimum failed.
- [x] 2.5 The capability gate: a width whose PPO arm misses the registered floor is reported
  `uninterpretable`, never as a null, and is excluded from the trend with that exclusion printed.
- [x] 2.5b **NOT NEEDED** — width 4's capability arm reached 19.69 foods and 87.6% full clear, so the
  registered condition never fired and the alternative was not built. The declared alternative was:
  `num_hidden_layers: 1` at width 8, which is also 8 perturbed units without a bottleneck below the
  input dimension. Three more configs, its one-layer architecture stated wherever its number appears,
  reported as an added point rather than a replacement — width 4's failure stays on the record.
- [x] 2.6 Per-width drift: relative weight distance from the frozen control, so starved-and-still is
  distinguishable from writing-a-lot-in-a-worsening-direction. Compared **within** a width — weights
  at different widths have different shapes — and it requires the campaign's `--track-experiment`
  records (task 4.2); the harness reports drift as unavailable rather than as zero when they are
  missing.
- [x] 2.7 Tests for 2.3–2.6, including a synthetic fixture where one width is positive and another is
  uninterpretable, and one where the significance passes and both minima fail.

## 3. Pilot (disjoint seeds 101–104)

- [x] 3.1 `launch.md` committed before anything runs.
- [x] 3.2 Widths 4 and 64, all three arms, 24 runs: is the frozen arm off the floor, is the PPO arm off
  the ceiling, and is there a gap between them at **both** extremes?
- [x] 3.3 Measured per-run wall time, and the campaign scheduled from it.
- [x] 3.4 **NOT APPLIED** — the pilot found PPO at ~19.8 of 20 foods and 90–94% full clear against
  frozen controls at 0.68 and 3.27 and 0% clear, so the platform has room at both extremes and the
  calibrated cell stands. The declared remedy was: if the platform has no room, fall back to the committed C1 food-only cell
  (`max_steps: 800`, target 10) as a **dated amendment** carrying the pilot table that forced it, and
  record that block-V comparability is given up.

## 4. Campaigns

- [x] 4.1 S1 first: it is minutes, it does not depend on S2, and its stop clauses gate the rest.
- [x] 4.2 S2: 120 runs — three arms × five widths × eight seeds at 3000 episodes — run with
  `--track-experiment`. Task 2.6's drift reads `final.pt` through the experiment record's
  `exports_path`, so without the flag every drift figure comes back empty rather than wrong. Launch
  record committed first; no branch switches while it runs.
- [x] 4.3 Per-seed CSVs, the per-width tables, both sweeps' verdicts and the combined verdict under
  `supporting/060-l4-perturbation-scale/`.

## 5. The record

- [x] 5.1 Logbook 060: the confound table with all three platforms' dimensions; S1's slope, its CI, its
  censoring and its reachability normalisation; the 128-unit cell's pass or fail; S2's per-width table
  with its capability column; the drift column; the combined verdict against the three registered
  outcomes, with a mixed reading recorded as mixed.
- [x] 5.2 The experiments index row.
- [x] 5.3 State plainly what the result does to the phase's existing negatives — re-read as
  under-budgeted, or left standing — and that any re-read is a **new registration** rather than a
  re-labelling of a committed verdict.
- [x] 5.4 State what the result may not be cited as. In particular: a sweep topping out at 128 units
  says nothing measured about 302, and the derived budget is an extrapolation.

## 6. Close-out

- [x] 6.1 `CHANGELOG.md`; the tracker's new **R** block with R.1, and R.2 (e-prop) carried as
  registered-not-run from [Logbook 059](../../../docs/experiments/logbooks/059-7a-shipment.md); the
  roadmap only if the reading changes.
- [x] 6.2 Confirm no committed verdict was changed, and that the control's committed values reproduce
  at the default width.
