# V.1 — the wiring premise: the registered protocol

Registered in `openspec/changes/add-wiring-premise-contrast`, reviewed and committed **before** the
pilot ran.

## The question

Does the wild-type wiring beat its degree-preserving rewired null, under an optimiser known to learn
on this substrate, on a cell matched to a behaviour the animal actually performs?

Every wiring contrast in this project — [034](../034-connectome-structure-controls.md), and every
panel in Phase 7 — has run on the integrated C3 cell: food chemotaxis **plus predator evasion plus
thermotaxis**, 2400 steps an episode. 034 recorded the limitation in its own words: "**Single task**
(the continuous integrated-C3 cell); the result is for this multi-objective
foraging/predator/thermotaxis demand, not a universal statement." Both architecture rankings
([025](../025-weight-search-architecture-ranking.md),
[029](../029-continuous-architecture-ranking.md)) place the connectome **competitive on foraging and
behind on predator evasion** — the least worm-like of the three demands. Nothing has separated them.

[Logbook 056](../056-l4-ladder-reread.md) found five registered contrasts uninformative about both
the wiring and the instrument, because this premise has no demonstration behind it. This tests it
where it has the best chance of holding.

## The arms

Four connectome arms per cell, 16 paired seeds (1–16), 3000 episodes, plus a descriptive MLP
reference. Within a wiring pair `wiring` is the only key that differs; within a learning pair
`freeze_updates` is. `rewire_seed` is unset, so each seed's wild-type and rewired arms pair.

| cell | `max_steps` | arms |
|---|---|---|
| C1 klinotaxis foraging | 800 | `wt_ppo`, `rn_ppo`, `wt_frozen`, `rn_frozen`, `mlp_ppo` |
| thermal + foraging survival | 500 | the same |

## The reading

`t7_continuous_ranking.plateau_tail` — the final-quarter window of 029, 034 and every panel — read
through `scripts/analysis/wiring_premise.py`, which calls the same statistics layer the 034 control
does.

**Scored quantity, fixed from the configs before any data existed**: the klinotaxis cell on
**full-clear success (%)**, as 034 and the panels were; the thermal cell on **mean foods**, because
its committed recipe sets `satiety_gain_per_food: 0.2` and its own header calls it "a
lethal-zone-avoidance/survival task rather than a collect-10 budget", so full clears there are
structurally near the floor.

Eight tests, both cells corrected together under BH-FDR at α = 0.05, one-sided and paired, with 80%
bootstrap CIs: the contrast, two learning gates and the untrained prior per cell. **V1 is the only
test that decides a verdict.** The thermal contrast (V5) annotates the 7b reading — 036's
thermotaxis validation ran on the MLP alone, on a cell the connectome has no config for, and the
connectome has never been shown to learn this one.

**The gates are read first.** A contrast against a null presupposes the claim-carrying arm learned;
where V2 fails, the cell's verdict is `no_learning` and no wiring verdict is assigned.

**Significance is not sufficient.** At n = 16 a paired rank test fires on the consistency of the
sign, not the size of the shift, and this contrast has shrunk on every fresh look in its Hebbian
form (+16.2 → +11.9 → +8.1). A significant primary below **+5.0 points** (or **+0.5 foods** on the
thermal cell) is recorded as `below_min_effect` and licenses nothing on its own.

**The ceiling clause.** If both PPO arms reach a plateau-tail full-clear mean ≥ 90%, the cell cannot
discriminate: verdict `saturated`, and the remedy is one change on the same cell —
`target_foods_to_collect` 10 → 20 — run once. The recipe is never adjusted until the arms separate.

## Outcomes, fixed before the run

- **`specific_wiring` on the klinotaxis cell.** The wiring hypothesis is alive on a behaviour the
  animal performs; the phase's headline changes; 7b's gate is re-openable; and the instrument work
  that follows gets a cell on which a working rule would have something to find. It reopens no
  committed C3 verdict, which stands in its own units.
- **`degree_statistics` / `rewired_beats_wildtype` / `inconclusive`.** The premise fails where it was
  most likely to hold. 7b's central measurement has no within-species signal, the shipment decision
  inherits that, and further rule work is instrument characterisation rather than a route to the
  phase's MUST.
- **`no_learning`.** A finding about the platform. It licenses fixing the platform and nothing about
  the wiring.
- **`saturated`.** The named remedy, once.

## Honest prior

**This probably comes out null.** Two PPO regimes have already looked: 034 found the wirings
indistinguishable on C3 (−3.28, q = 0.770, the null nominally higher) and 043's low-noise PPO arms
put the null ahead by 12.6 on 0 of 8. The untrained prior is also indistinguishable between wirings
(+3.3 over 64 seeds; +0.77 after grounding), so any advantage has to be created by learning rather
than inherited from the graph. What is genuinely untested is whether removing the predator component
changes that, and it is worth hours of compute because a negative is the strongest available input
to the shipment decision and a positive changes the phase.

## The pilot, on disjoint seeds 101–104

Registered seeds stay untouched until the protocol is fixed. The pilot answers three questions and
changes nothing else:

1. **Does the connectome converge on C1 at the inherited recipe, seed by seed?** The committed
   config's header says the entropy/lr recipe is "subject to the connectome's own per-seed/entropy C1
   check (Stage 2 step 2a)", and that check is not in the record. If it does not converge, the
   campaign does not launch and the recipe is settled first, on pilot seeds, under a dated amendment.
2. **What does a run cost on these exact configs?** Measured here, not scaled from a lighter one.
   The nearest committed figure is Logbook 026's isolated C1 connectome run at 28.67s per 200
   episodes, post-vectorisation, on the pre-refactor cell.
3. **Where does each cell sit relative to the ceiling threshold, and is the thermal cell at the
   floor on both metrics?** If it is, it is not a platform for this question and is reported as
   such rather than scored on a metric chosen after the fact.

## Reproduce

```bash
# pilot
C1=configs/scenarios/foraging/connectomeppo_small_continuous2d_fick_adaptive_klinotaxis
TH=configs/scenarios/thermal_foraging/connectomeppo_small_continuous2d_thermal_klinotaxis
uv run python scripts/run_campaign.py \
  --config ${C1}.yml --config ${C1}_rewired_null.yml --config ${C1}_frozen.yml \
  --config ${TH}.yml --config ${TH}_frozen.yml \
  --seeds 101-104 --runs 3000 --output-dir campaigns/wiring-premise-pilot \
  -- --theme headless
```

______________________________________________________________________

## Pilot outcome, 2026-09-12 — and the amendment it forced

20 runs on seeds 101–104, then 32 more at the remedy's setting. All 52 succeeded.

**The three pilot questions, answered.**

1. **The connectome converges on C1 at the inherited recipe** on every pilot seed — the check the
   committed config header flagged as open (Stage 2 step 2a) is answered, positively.
2. **Cost, measured on these exact configs**: mean 637 s a run at 3000 episodes, 20 runs in 980 s
   wall on 16 workers; the remedy's 32 runs in 2074 s. The 128-run campaign is scheduled from those
   figures, not from Logbook 026's 28.67 s / 200 episodes on the pre-refactor cell.
3. **Both cells are over the ceiling, and the registered remedy did not fix it.**

| arm | committed target 10 | remedy, target 20 |
|---|---|---|
| klinotaxis `wt_ppo` | **100.00%** | **100.00%** |
| klinotaxis `rn_ppo` | **100.00%** | **100.00%** |
| klinotaxis `wt_frozen` | 57.60% | 25.10% |
| thermal `wt_ppo` | 96.33% | 97.40% (19.80 foods of 20) |
| thermal `rn_ppo` | — | 96.33% (19.69 foods of 20) |

The klinotaxis contrast is **exactly 0.00 on 4/4 seeds** at both settings. Had this run on registered
seeds it would have produced a meaningless null that looked like a result. The remedy was applied
once, as registered, and the recipe was not touched.

**Two things the pilot corrected in the registration**, both recorded rather than quietly applied:

- The thermal cell was registered as scored on mean foods because its survival-dominant satiety
  would put full clears near the floor. **It does not** — 96–97% full clear. The cell is saturated,
  not floored, and that reasoning was wrong.
- The peak axis cannot answer this question on either cell. Re-reading the same runs on the
  committed efficiency harness (034's own follow-up) leaves klinotaxis flat — both wirings reach 30%
  success inside ~40 episodes of 3000 — and shows the thermal cell's only directional signal:
  **301 episodes to competence against the null's 580, a 48.1% gain**, at q = 0.417 and 2/4 seeds,
  which n = 4 cannot resolve either way.

The amendment in the change's design moves the primary to the **efficiency axis on the thermal
cell**, keeps the gates on the peak axis, keeps klinotaxis as a secondary, and registers a
**minimum 20% reduction in time-to-competence** beside significance. The campaign is therefore a
confirmation test of a pilot-generated hypothesis, and is recorded as one.

**The prior, restated after the pilot.** The per-seed consistency behind that 48.1% is 2/4 and 3/4 —
the shape the Hebbian wiring contrast had before it shrank from +16.2 to +8.1 across fresh looks. A
null at n = 16 remains the expected result.

## Reproduce (campaign)

```bash
C1=configs/scenarios/foraging/connectomeppo_small_continuous2d_fick_adaptive_klinotaxis
TH=configs/scenarios/thermal_foraging/connectomeppo_small_continuous2d_thermal_klinotaxis
uv run python scripts/run_campaign.py \
  --config ${C1}_t20.yml --config ${C1}_rewired_null_t20.yml \
  --config ${C1}_frozen_t20.yml --config ${C1}_rewired_null_frozen_t20.yml \
  --config ${TH}_t20.yml --config ${TH}_rewired_null_t20.yml \
  --config ${TH}_frozen_t20.yml --config ${TH}_rewired_null_frozen_t20.yml \
  --seeds 1-16 --runs 3000 --output-dir campaigns/wiring-premise \
  -- --theme headless
```

______________________________________________________________________

## Registered panel outcome, 2026-09-12 — and the replication it triggers

128/128 runs succeeded, seeds 1–16, 3000 episodes, 8026 s wall.

**The primary is positive.** Thermal cell, efficiency axis, `SPECIFIC-WIRING-EFFICIENCY`:

| metric | wild | rewired | delta | q | wild-better |
|---|---|---|---|---|---|
| `auc_success` | 0.80 | 0.67 | **+0.14** | 0.001 | 14/16 |
| `auc_foods` | 18.37 | 17.15 | **+1.21** | 0.009 | 14/16 |
| `episodes_to_30pct_success` | **305.5** | **570.3** | +264.8 | 0.009 | 12/16 |
| `episodes_to_90pct_foods_plateau` | 426.1 | 734.4 | +308.4 | 0.014 | 12/16 |

Time-to-competence gain **+46.4%** against the registered minimum of +20%. Both learning gates pass
at q = 0.000 on 16/16 (V6 +17.31, V7 +16.68), and the untrained prior is indistinguishable
(V8 −0.50, q = 0.912, 7/16) — so the advantage is **created by learning rather than inherited from
the graph**, which is the control that makes the result mean anything.

The secondary cell reads as the pilot predicted: peak `SATURATED` (100.00% on both wirings, contrast
exactly 0.00 on 0/16), efficiency `DEGREE-STATISTICS` with all four metrics nominally negative
(−25.7% on time-to-competence, wild-better 6–7/16). Same axis, same harness, same seeds — so the
efficiency axis is not a positive-generating instrument.

**Checks run beyond the registration.** BH across all eight efficiency metrics jointly rather than
four per cell: q = 0.0017 / 0.0175 / 0.0175 / 0.0290, all surviving. Per-seed, the effect is not
outlier-driven — the four losing seeds are narrow (257/214, 743/316, 502/451, 350/286) while several
wins are large (292/1721, 166/968, 210/878); the wild type's times cluster at 46–743 where the null's
reach 1721. **Lower variance in time-to-competence is much of the effect.**

**What this is not.** The peak axis is saturated, so this is a *learning-speed* claim and not a
performance one: both wirings finish in the same place. It is PPO, not a local rule, so 7b's gate as
literally written is still unmet — what is now supported is the premise behind it. And the thermal
cell enables the thermotaxis projection onto AFDL/AFDR, which rewiring scrambles along with
everything else, so the effect may be specific to the evolved sensory→motor pathway rather than the
wiring at large — a sharper claim than the one registered, and one that needs an arm rewiring only
non-sensory edges before it can be asserted.

## The replication, seeds 17–32

Registered here **before it runs**, and before Logbook 057 is written.

This is the phase's first positive result; it rests on an axis amended after pilot data; and this
project has a committed instance of exactly this effect class not holding — the Hebbian wiring
contrast measured +16.2, then +11.9, then +8.1 across fresh looks, which is why
[panel 3](../042-l4-panel3.md) exists. Fresh seeds are the cheapest answer to both objections.

**No protocol changes**: the same four thermal arms, the same harness, the same four-metric family,
the same registered minimum of 20% off time-to-competence. Seeds 17–32, disjoint from both the pilot
(101–104) and the registered panel (1–16).

**Outcomes, fixed before the run:**

- **Replicates at or above the registered minimum.** The claim is carried by two independent panels
  and the record reports the pooled estimate beside both.
- **Same direction, below the minimum.** Reported as a real but smaller effect, with the shrinkage
  named and panel 3's precedent cited. It licenses the follow-up, not the claim.
- **Does not replicate.** The registered panel is reported as not holding on fresh seeds, and the
  first positive is withdrawn on the record rather than defended.

```bash
TH=configs/scenarios/thermal_foraging/connectomeppo_small_continuous2d_thermal_klinotaxis
uv run python scripts/run_campaign.py \
  --config ${TH}_t20.yml --config ${TH}_rewired_null_t20.yml \
  --config ${TH}_frozen_t20.yml --config ${TH}_rewired_null_frozen_t20.yml \
  --seeds 17-32 --runs 3000 --output-dir campaigns/wiring-premise-replication \
  -- --theme headless
```

______________________________________________________________________

## Replication outcome, 2026-09-12 — directionally consistent, not significant

64/64 runs succeeded, seeds 17–32, protocol unchanged.

| | panel 1 (1–16) | panel 2 (17–32) | pooled (32) |
|---|---|---|---|
| `auc_success` | +0.14, q = 0.001, 14/16 | +0.10, q = 0.115, 10/16 | +0.12, q = 0.000, 24/32 |
| 80% CI | [+0.10, +0.18] | **[+0.04, +0.17]** | [+0.08, +0.15] |
| `episodes_to_30pct_success` | +264.8, q = 0.009, 12/16 | +187.8, q = 0.302, 10/16 | +226.3, q = 0.015, 22/32 |
| 80% CI | [+145, +407] | **[+6, +387]** | [+114, +343] |
| time-to-competence gain | +46.4% | +32.6% | +39.5% |

**The direction replicated on all four metrics; the significance did not.** Panel 2's 80% bootstrap
intervals are entirely above zero on both headline metrics and each contains panel 1's point
estimate, so the two panels are statistically consistent and the drop from +46.4% to +32.6% is
within noise. What fell is seed-level sign consistency, 12–14/16 to 7–10/16, which is what the
paired rank test reads — hence q = 0.115–0.334 and a committed verdict of `degree_statistics`.
**The registered replication did not confirm the verdict.**

The pooled 32-seed estimate is significant, and is recorded as **descriptive only**: it is dominated
by panel 1, and pooling is not a substitute for an independent replication. Logbook 042 treated its
own pooled estimate the same way.

**A defect in this registration, recorded rather than worked around.** The three replication outcomes
fixed above conflate effect *size* with *significance*: the actual result — size above the 20%
minimum, significance absent — fits none of them cleanly. The reading applied is the second
outcome's spirit: a real but smaller effect, the shrinkage named, licensing the follow-up and not
the claim.

**The rank test and the interval disagree, and that is the finding's shape.** The effect is
heterogeneous — large on most seeds, slightly negative on a few. A paired rank test sees only signs;
the bootstrap interval sees magnitude and excludes zero in both panels. This is the mixture problem
I.2 was built for, and panel 3 is read with I.2's family beside the registered rank test.

## Panel 3, seeds 33–64

Registered **before it runs**. Protocol unchanged: the same four thermal arms, the same harness, the
same four-metric family, the same registered minimum of 20% off time-to-competence. Seeds 33–64 are
disjoint from the pilot (101–104), panel 1 (1–16) and panel 2 (17–32).

**Why 32 more rather than 16**: 64 seeds is the sample size [042](../042-l4-panel3.md) used to close
the Hebbian wiring question, and at a true sign-consistency around the pooled 22/32 a paired rank
test has real power there where it has none at 16. This either confirms the effect or exposes it as
marginal; both settle it.

**Outcomes, fixed before the run** — and this time size and significance are separated:

- **Significant on the primary metric at 64 seeds, gain ≥ 20%.** The effect is carried by the full
  panel and Logbook 057 reports it as a finding, with panel 2's non-replication reported beside it.
- **Significant, gain < 20%.** Real and below the registered minimum: reported as such, licensing
  the mechanism follow-up and not the claim.
- **Not significant, intervals still clear of zero.** Reported as a heterogeneous effect the
  registered rank test cannot carry at this sample size, read through I.2's family, and named as
  unresolved rather than either claimed or withdrawn.
- **Not significant, intervals spanning zero.** The first positive is withdrawn on the record.

```bash
TH=configs/scenarios/thermal_foraging/connectomeppo_small_continuous2d_thermal_klinotaxis
uv run python scripts/run_campaign.py \
  --config ${TH}_t20.yml --config ${TH}_rewired_null_t20.yml \
  --config ${TH}_frozen_t20.yml --config ${TH}_rewired_null_frozen_t20.yml \
  --seeds 33-64 --runs 3000 --output-dir campaigns/wiring-premise-panel3 \
  -- --theme headless
```
