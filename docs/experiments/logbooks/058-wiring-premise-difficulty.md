# 058: Difficulty, Not Temperature — the Wiring Advantage Survives Removing the Thermosensory Cell (7a-ii V.3 / Phase 7)

**Status**: completed — **`specific_wiring_efficiency`**, and it generalises
[Logbook 057](057-wiring-premise-contrast.md). 057 found the wild-type wiring reaching competence
~35% sooner than its degree-preserving rewired null on a foraging cell under lethal thermal pressure,
and nothing on the food-only cell, which saturates. Two readings fit: the evolved thermosensory
projection does the work, or the wiring helps on any cell hard enough to discriminate. **The cells
differ in three respects and only two bind** — across 057's third panel the thermal cell's episodes
end 14.7% `health_depleted`, 12.4% `max_steps` and **0.6% `starved`**, so the satiety setting is
nearly inert and the separable non-temperature factor is the episode budget. This arm raises
difficulty by the budget alone, with no temperature at all, on a **calibrated** setting: the obvious
inherited value could not have worked. At the frozen `max_steps: 350`, over **32 paired seeds**, the
wild type reaches competence in **892 episodes against 1165 — a +23.5% gain**, above the registered
20% minimum, with three of four efficiency metrics significant (q = 0.003, 0.017, 0.029) and 21–26 of
32 seeds favouring it. Both gates pass **32/32** and the untrained prior is indistinguishable
(−0.01, q = 0.841). **So difficulty is sufficient and the thermosensory projection is not necessary.**
Two honest qualifications travel with it: the fourth efficiency metric does **not** move
(−23.66, q = 0.463, 15/32), where all four moved on the thermal cell; and this arm's 32 rewirings are
**the same graphs** the thermal panels used, so the two cells are independent in task and
initialisation but not in rewiring.

**Branch**: `feat/wiring-premise-difficulty`.

**Date**: 2026-09-13.

**OpenSpec change**: `add-wiring-premise-difficulty` (extends `architecture-comparison-protocol`: an
effect present on one cell and absent on another is attributed only after the cells' differences are
enumerated).

## Objective

Separate the two readings of 057's result. Registered as "pathway or difficulty"; corrected here,
because the two cells differ in more than one respect and the record has to say which ones bind.

## Background

057's pair of cells:

| ending, thermal cell (32 seeds of panel 3) | count | share |
|---|---|---|
| `completed_all_food` | 69,366 | 72.3% |
| `health_depleted` | **14,098** | **14.7%** |
| `max_steps` | **11,927** | **12.4%** |
| `starved` | 609 | 0.6% |

Temperature and the episode budget bind; satiety does not. So the isolable non-temperature factor is
the budget, and `satiety_gain_per_food: 0.2` is carried into this cell for config-matching rather
than as pressure.

[V.2](supporting/057-wiring-premise-contrast/probe-v2.md) had already closed the simplest pathway
story: the wild type's shortest route from AFD to the motor readout is **3 hops against every
rewiring's 1–2**, with a longer characteristic path length, so whatever the projection contributes is
not hop count.

## Method

The food-only cell at `target_foods_to_collect: 20`, no temperature, `max_steps` calibrated, four arms
on 32 paired seeds at 3000 episodes: `wt_ppo`, `rn_ppo` and a frozen-weights floor per wiring. Scored
through `scripts/analysis/wiring_premise.py` — the gates on the peak axis, the **efficiency axis as
the primary** with the registered 20% minimum off time-to-competence, the same bar 057's thermal cell
was held to.

### The budget was calibrated, and the band is one grid step wide

The inherited value would have failed. On this cell the plateau-tail episodes complete in a **mean of
310 steps**, so the thermal cell's `max_steps: 500` cannot bind. A grid declared before the pilot ran
on disjoint seeds 101–104:

| `max_steps` | wt / rn full clear | crossed 30% | inside the band? |
|---|---|---|---|
| 150 | 0.00% / 0.00% | **0% / 0%** | no — nothing full-clears |
| 250 | 10.67% / 6.13% | **0% / 0%** | no — learning happens, metric censored |
| **350** | **77.40% / 69.23%** | **100% / 100%** | **yes** |

**Both edges were load-bearing.** The primary metric returns the horizon for a seed that never
crosses a 30% full-clear rate, so at 250 the contrast would have been censored at 3000 for every seed
and read as a clean null while learning was plainly happening. 350 is the only grid point that works.

## Results

### The registered family, 32 paired seeds

| test | contrast | mean Δ | 80% CI | q | +seeds |
|---|---|---|---|---|---|
| **V9** | `wt_ppo − rn_ppo` (full clear) | +4.05 | +1.98 … +5.78 | **0.001** | 27/32 |
| V10 | `wt_ppo − wt_frozen` (gate) | +76.24 | +74.69 … +77.75 | 0.000 | **32/32** |
| V11 | `rn_ppo − rn_frozen` (gate) | +72.18 | +70.91 … +73.57 | 0.000 | **32/32** |
| V12 | `wt_frozen − rn_frozen` (prior) | −0.01 | −0.02 … +0.00 | 0.841 | 0/32 |

Peak-axis verdict **`below_min_effect`**: V9 is significant at q = 0.001 and **+4.05 is under the
registered 5.0-point minimum**, so it licenses nothing on its own. Recorded because the clause exists
for exactly this case.

### The efficiency axis — the primary

| metric | wild | rewired | Δ | q | wild-better |
|---|---|---|---|---|---|
| `auc_success` | 0.43 | 0.39 | **+0.04** | 0.003 | 26/32 |
| `auc_foods` | 17.33 | 17.05 | **+0.29** | 0.017 | 22/32 |
| `episodes_to_30pct_success` | **892.0** | **1165.3** | **+273.3** | 0.029 | 21/32 |
| `episodes_to_90pct_foods_plateau` | 1181.7 | 1158.0 | **−23.7** | 0.463 | 15/32 |

**Time-to-competence gain +23.5%** against the registered 20% minimum. Crossing rates 100% in both
arms, so the metric is not censored. **Verdict `specific_wiring_efficiency`.**

## Analysis

1. **Difficulty is sufficient; the thermosensory projection is not necessary.** The advantage appears
   on a cell with no temperature at all, no thermosensory projection, and difficulty from an episode
   limit alone. 057's claim generalises from "a foraging cell under thermal pressure" to **"a
   foraging cell hard enough to discriminate"**. It does not establish that *any* hard task shows it —
   only hard foraging cells of this family.
2. **The effect is smaller here: +23.5% against the thermal cell's +35.4%.** Both clear the registered
   bar. Whether the difference is the cell or sampling is not resolvable from two cells.
3. **One metric dissents, and it is the one about an arm's own plateau.** Time to 90% of a run's *own*
   foods plateau does not move (−23.7, 15/32) while time to an absolute 30% full-clear rate does
   (+273.3, 21/32). The wild type gets to a fixed competence bar sooner; it does not converge to its
   own ceiling sooner. On the thermal cell all four metrics moved, so this is a narrowing.
4. **A small endpoint advantage appears here, where the thermal cell had none.** V9 is +4.05 points of
   full clear, significant but under the registered minimum; on the thermal cell the null was
   nominally *ahead* on the endpoint. Neither is licensed as a performance claim.
5. **What this arm cannot separate.** A null here would not have implicated the projection, because
   lethal-zone mortality is temperature-dependent too. It is positive, so that branch is moot — but
   the symmetric point stands: this result says the projection is *unnecessary*, not that it
   contributes nothing.
6. **The rewirings are shared, so "independent confirmation" needs qualifying.** `rewire_seed` derives
   from the run seed, so this arm's 32 rewired graphs are **exactly** those the thermal panels 1 and 2
   used. The cells are independent in task and in weight initialisation, not in rewiring: if a subset
   of these 32 graphs is unusually poor, both cells inherit it. V.2 found no graph property predicting
   learning time, which bounds the concern without removing it. **A fresh-rewiring replication would
   be seeds 65+ on either cell.**

## Conclusions

- **`specific_wiring_efficiency`** on a hard food-only cell: +23.5% off time-to-competence over 32
  paired seeds, three of four efficiency metrics significant, both gates 32/32, prior
  indistinguishable.
- **The wiring advantage does not require the thermosensory pathway.** It requires a cell that
  discriminates.
- **Still speed, not performance.** The peak-axis contrast is real but under its registered minimum.
- **No committed verdict changes.** 057 stands; this widens what it is evidence for.

## Next Steps

- [ ] B.8, which now inherits a wiring advantage shown on **two** cells rather than one.
- [ ] A fresh-rewiring panel (seeds 65+) if the shared-rewiring caveat is to be closed.
- [ ] Separating *whether the projection contributes* from *whether it is necessary* still needs the
  arm that rewires only non-sensory edges, which needs a new rewiring mode and is not registered.

## Data References

- Records: `supporting/058-wiring-premise-difficulty/` — `launch.md` (protocol, the ending counts, the
  calibration grid and both band edges, all registered before the runs), `pilot/` (all three grid
  points), `panel/`, `per-seed-primary.csv`.
- Configs: `configs/scenarios/foraging/connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_hard350*.yml`.
- 176 runs (48 calibration, 128 campaign); all succeeded.
