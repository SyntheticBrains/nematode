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
