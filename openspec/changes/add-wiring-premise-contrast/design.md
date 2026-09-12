# Design: testing the premise where it has the best chance of holding

## The question, and both answers

**Does the wild-type wiring beat its degree-preserving rewired null, under an optimiser known to
learn, on a single behaviour the animal actually performs?**

- **Positive** — the phase's headline changes. The C3 null becomes a statement about a
  multi-objective cell whose hardest component the animal was never under pressure to solve, rather
  than about the wiring; 7b's gate is re-openable; and the rule question becomes worth solving
  *on that cell*, where a rule that learned would have something to find.
- **Negative** — the premise fails where it was most likely to hold. 7b's central measurement has no
  within-species signal, the shipment decision inherits a much stronger negative than "our rule did
  not learn on our integrated cell", and any further rule work is instrument characterisation
  rather than a route to the phase's MUST.

Both answers are useful. That is the point of running it.

## Why this cell, and not another rung

Three things make the C1 klinotaxis foraging cell the right first rung, and they are independent:

1. **It is the behaviour the animal performs**, and this repository has already validated the
   substrate against the real worm on it — 035 reproduces klinokinesis and the weathervane on both
   the MLP and connectome arms, with a specificity control giving a double dissociation.
2. **It removes the component where the connectome's deficit sits.** 025 and 029 both place the
   connectome competitive on foraging and behind on predator evasion. Nothing has separated the two.
3. **It is cheaper and shorter** — `max_steps: 800` against C3's 2400, no predator projection. The
   nearest measured cost is Logbook 026's isolated C1 connectome run at **28.67s per 200 episodes**
   after vectorisation. That is the arithmetic principle 2 of the phase protocol asks for, and the
   pilot measures it again on the exact configs rather than scaling it.

The thermal cell runs beside it because 7b's MUST names *both* behaviours, and it runs as a
**secondary** because it is not the same kind of cell. `thermal_foraging/connectomeppo_small_continuous2d_thermal_klinotaxis.yml`
is foraging *plus* thermotaxis under a survival-dominant satiety (0.2 per food), `max_steps: 500`;
036's real-worm thermotaxis validation ran on the MLP only, on a thermotaxis-seeking cell that has
no connectome config; and the connectome has never been shown to learn this cell. Its contrast
annotates the 7b reading — a decision about 7b should not rest on one of its two measurements — and
its gate (V6) may well be where it stops, which would itself be a finding 7b needs.

## Arms

Per cell, 16 paired seeds (1–16), **3000 episodes** (the budget 043's PPO arms reached 68.5 and 81.2
in), `wiring` the only key that differs within a wiring pair and `freeze_updates` the only key that
differs within a learning pair. `rewire_seed` is left unset so the brain derives it from the run
seed and the pairing holds per seed. Budgets: C1 klinotaxis `max_steps: 800`; thermal `max_steps: 500`.

| arm | wiring | PPO | role |
|---|---|---|---|
| `wt_ppo` | wild type | yes | the claim |
| `rn_ppo` | rewired null | yes | the contrast |
| `wt_frozen` | wild type | no | the wild type's own floor |
| `rn_frozen` | rewired null | no | the null's own floor |
| `mlp_ppo` | — | yes | the cell's ceiling, descriptive, not a registered test |

## The registered family

Eight tests, both cells corrected together under BH-FDR at α = 0.05, each one-sided, paired,
reported with an 80% bootstrap CI and the count of positive seeds — the bar panels 1–3 and 034 used.
**Through the same code**: `scripts/analysis/connectome_structure_controls.py` already implements
this exact contrast — `t7_continuous_ranking.plateau_tail` for the metric,
`weight_search_architecture_ranking.paired_seed_wilcoxon_bootstrap` and `bh_fdr` for the
statistics, and the `specific_wiring` / `degree_statistics` vocabulary — and the harness extends it
rather than reimplementing it, so commensurability with 034 holds by construction.

| test | contrast | role |
|---|---|---|
| **V1** | klinotaxis: `wt_ppo − rn_ppo` | **primary — the only test that decides the verdict** |
| V2 | klinotaxis: `wt_ppo − wt_frozen` | gate — did the wild type learn on this cell |
| V3 | klinotaxis: `rn_ppo − rn_frozen` | gate — did the null learn on this cell |
| V4 | klinotaxis: `wt_frozen − rn_frozen` | the untrained prior, annotation |
| V5 | thermal: `wt_ppo − rn_ppo` | secondary — annotates the 7b reading |
| V6 | thermal: `wt_ppo − wt_frozen` | gate for V5 |
| V7 | thermal: `rn_ppo − rn_frozen` | gate for V5 |
| V8 | thermal: `wt_frozen − rn_frozen` | prior, annotation |

**The gates are read first and they can stop the cell.** A contrast against a null presupposes that
something learned; principle 6 of the phase protocol exists because the degree-preserving null was
built on a premise nobody tested. If a cell's V2 fails, that cell's contrast is not interpretable
and its verdict is `no_learning` — a finding about the platform, recorded as such.

**The metric** comes from `t7_continuous_ranking.plateau_tail` — the final-quarter window that is
the committed metric of 029, 034 and every panel in the phase — which returns full-clear success (%)
and mean foods from the same tail. Both are reported for every arm.

**The scored quantity differs by cell, and this was fixed from the configs before any data existed.**
The klinotaxis cell is scored on **full-clear success**, as 034 and the panels were. The thermal cell
is scored on **mean foods**, because its committed recipe sets `satiety_gain_per_food: 0.2` — its own
header calls it "a lethal-zone-avoidance/survival task rather than a collect-10 budget" — so
full clears there are structurally near the floor and a contrast on them would read zero against
zero. Its registered minimum effect is **0.5 foods**, the figure I.3b registered for a foods-scored
contrast. If the pilot finds the thermal cell at the floor on *both* metrics, it is not a platform
for this question and is reported as such rather than scored on a third metric chosen after the
fact. **I.2's mixture
family** — competent-fraction discordance at the committed 20.0 threshold and the level among each
arm's own competent seeds, under the pooled-label permutation null — is registered as the secondary
reading of each primary, applied through `scripts/analysis/l4_mixture_statistic.py` rather than
reimplemented. A bimodal outcome is what the phase has met every time it looked.

**A minimum effect is registered beside significance**: **+5.0 percentage points** on the primary,
and **+0.5 foods** on the thermal cell's foods-scored contrast.
At n = 16 a paired rank test fires on the consistency of the sign rather than the size of the
shift, and the phase has one committed example of a significant-looking contrast shrinking on every
fresh look (+16.2 → +11.9 → +8.1). A significant primary below +5.0 is recorded as significant and
below the registered minimum, and licenses nothing on its own.

## The saturation clause

An easier cell can put both arms against the ceiling, where no contrast can resolve and a null means
nothing. Registered before the pilot: **if both PPO arms' plateau-tail full-clear mean is ≥ 90%, the
cell is `saturated`** and its contrast is not read. The named remedy is **one change on the same
cell**: `target_foods_to_collect` 10 → 20 at the committed `foods_on_grid: 5` and `max_steps: 800`,
run once. The recipe is never adjusted until the arms separate, which would be fitting the platform
to the hypothesis.

The pilot measures this on disjoint seeds before any registered seed is spent.

## The pilot, on disjoint seeds 101–108

Registered seeds stay untouched until the protocol is fixed. The pilot answers three questions and
changes nothing else:

1. **Does the connectome converge on C1 at the inherited recipe, seed by seed?** The committed
   config's own header says the entropy/lr recipe is "subject to the connectome's own
   per-seed/entropy C1 check", and that check is not in the record. The pilot reports convergence per
   pilot seed. If it does not converge, the campaign does not launch — the recipe is settled first,
   on pilot seeds, and the change is amended under a dated note. The launch record also states which
   config 035's connectome companion ran, since the base here is the `fick_adaptive` variant.
2. **What does a run cost on these exact cells?** Measured, not scaled from a lighter config. The
   phase has one committed instance of a cost estimate scaled from a lighter pilot missing by 78%.
3. **Where does the cell sit relative to the saturation threshold?**

## Verdicts

Assigned for the primary cell, in order, in the vocabulary 034 and the panels already use so the
record stays comparable; the thermal cell receives the same ordered reading as an annotation:

1. `insufficient_seeds` — any arm incomplete.
2. `no_learning` — the cell's gate (V2) fails: the wild type does not beat its own frozen floor.
3. `saturated` — both PPO arms at or above the ceiling threshold.
4. Otherwise from the primary alone: `specific_wiring` (significant and at or above the minimum
   effect), `below_min_effect` (significant and under it — named, and licenses nothing on its own),
   `rewired_beats_wildtype` (the interval entirely below zero), `degree_statistics` (the interval
   spans zero), `inconclusive` (neither).

The annotations (V3, V4, the mixture family, the MLP reference) never change a verdict.

## What each outcome licenses

Stated before the data exist:

- **`specific_wiring` on the primary** — the wiring hypothesis is alive on a behaviour the animal
  performs. The phase's headline changes, 7b's gate is re-openable on that behaviour, and the
  instrument ladder that follows gets a cell on which a working rule would have something to find.
  It does **not** retroactively reopen any committed C3 verdict; those stand in their own units.
- **`degree_statistics` / `rewired_beats_wild_type` / `inconclusive` on the primary** — the premise
  fails where it was most likely to hold. The thermal annotation then says whether 7b's second
  behaviour offers anything the first did not; where it does not, 7b's central measurement has no
  within-species signal to find, the shipment decision inherits that, and further rule work is
  characterisation of the instrument rather than a route to the phase's MUST.
- **`no_learning`** — a finding about the platform, not about the wiring. It licenses fixing the
  platform, and nothing about the connectome.
- **`saturated`** — the named remedy, once, and no reading of the contrast.

## What this design deliberately does not do

- It does not run any plasticity rule. The rule's status is settled by block I, and mixing it in
  would reintroduce exactly the confound Logbook 056 spent its length separating.
- It does not re-run C3. That cell's verdicts are committed and this change cannot alter them.
- It does not tune the recipe to make the arms separate. The recipe is inherited from the committed
  configs; where it fails, the pilot says so and the campaign waits.

## Amendment, 2026-09-12: the peak axis is saturated; the primary moves to efficiency

The pilot ran on disjoint seeds 101–104 and the registered saturation clause fired. On the
klinotaxis cell **both PPO arms cleared 100.00% on every seed** — a contrast of exactly 0.00 — with
frozen random weights already at 57.60%. The registered remedy (`target_foods_to_collect` 10 → 20,
recipe untouched) was applied once, as registered, and **did not unsaturate either cell**: both
wirings still clear 100.00% on klinotaxis, and the thermal cell reads 19.80 against 19.69 foods of
20\. The peak axis cannot answer this question on cells the animal is wired for, because PPO solves
them.

Re-reading the *same* pilot runs on the committed efficiency harness
(`scripts/analysis/connectome_structure_efficiency.py` — 034's own follow-up, four metrics under
BH-FDR) separates the two cells:

| cell | episodes to 30% success (wild / rewired) | wild-better seeds | q at n = 4 |
|---|---|---|---|
| klinotaxis | 43 / 39 | 2/4 | 0.812 |
| **thermal** | **301 / 580** | 2/4 | 0.417 |

The klinotaxis cell is learned by both wirings inside forty episodes of three thousand: it is flat
on both axes and nothing can be measured on it. The thermal cell is not saturated on the efficiency
axis and carries the only directional signal in the phase.

**So, amended before any registered seed was spent:**

1. **The primary is the efficiency axis**, read through the committed 034 harness with its four
   metrics, its BH-FDR family and its verdict rule unchanged. The **learning gates stay on the peak
   axis**, where "did this arm learn at all" is what they ask.
2. **The thermal cell becomes the primary cell**; klinotaxis stays registered as a secondary and is
   expected to read `degree_statistics` on both axes — a clean statement that at a difficulty where
   both wirings are perfect, neither axis separates them.
3. **A minimum effect is registered for the new primary**: the wild type must shorten
   time-to-competence by **at least 20%** as well as reaching significance. A significant result
   under that is `below_min_effect` and licenses nothing. The pilot's direction was +48.1%.
4. **Both cells run at `target_foods_to_collect: 20`**, the remedy's variant, since that is what the
   pilot characterised.

**What this amendment is and is not.** Changing the scored axis after seeing data is the thing a
registration exists to prevent. It is legitimate here on three conditions, all met: the pilot ran on
**disjoint seeds** and its registered job was to fix the protocol before registered seeds were
spent; the new axis is **not invented for this result** but is 034's own committed follow-up
harness, applied unchanged; and the campaign is therefore an honest **confirmation test of a
pilot-generated hypothesis**, recorded as such, with its minimum effect fixed in advance. The
per-seed consistency in the pilot is weak — 2/4 and 3/4 — which is the shape the Hebbian contrast
had before it shrank from +16.2 to +8.1 across fresh looks. A null at n = 16 is the expected result.
