# Design: isolating the two non-temperature routes to difficulty

## The question, and why the registered framing needed sharpening

057 asked "pathway or difficulty". The two cells it compared differ in temperature, episode length
and satiety gain — but **only two of those bind**. Across 057's third panel the thermal cell's
episodes end 14.7% `health_depleted`, 12.4% `max_steps` and **0.6% `starved`**: the satiety setting is
nearly inert. So the separable non-temperature factor is the **episode budget**, and it is the
manipulation.

**The arm**: the food-only cell at `target_foods_to_collect: 20` — the variant 057's pilot
characterised — with no temperature, `satiety_gain_per_food: 0.2` carried for config-matching, and a
**calibrated** `max_steps`.

**The budget is calibrated, not inherited, because the committed data says the inherited value fails.**
On this cell the plateau-tail episodes complete in a **mean of 310 steps, max 647**, and a policy
trained under a tighter budget needs fewer steps than one trained at 800 — so the thermal cell's
`max_steps: 500` would not bind and the arm would saturate exactly as 057's pilot did. The grid is
declared here and run on the pilot's disjoint seeds: **`max_steps` ∈ {150, 250, 350}**, one chosen,
frozen, and the campaign run at it. This is the sweep-before-pin the phase protocol asks for and
which picking 500 from another cell would have skipped.

**The target band, and why it has two edges.** The primary metric is
`episodes_to_30pct_success` — episodes to first cross a 30% rolling full-clear rate — and the harness
returns the horizon for an arm that never crosses. So the cell must be **hard enough that both PPO
arms stay clear of the ceiling** and **easy enough that both cross 30%**; outside that band the metric
is censored at 3000 for every seed and discriminates nothing while looking like a null. The pilot
reports the ceiling distance *and* the crossing rate per arm, and the chosen budget is the one inside
the band. If no grid point is inside it, the campaign does not launch and the change is amended.

## What each outcome licenses, fixed before any data exists

- **The advantage appears** (significant on the efficiency primary, ≥ 20% off time-to-competence).
  **Difficulty is sufficient and the thermosensory pathway is unnecessary.** The claim generalises
  from "on a foraging cell under thermal pressure" to "on a foraging cell hard enough to
  discriminate", which is broader and much easier to state. It would still not establish that *any*
  hard task shows it — only hard foraging tasks of this family — and the record says so.
- **The advantage is absent and the cell is not saturated.** Difficulty alone is **not** sufficient.
  This does **not** cleanly implicate the sensory projection: lethal-zone mortality is
  temperature-dependent too, so what it narrows to is "something about the thermal configuration",
  with the projection and the mortality pressure both live. The arm that separates *those* rewires
  only non-sensory edges and needs a new rewiring mode; it is named here and not run.
- **The cell saturates, or the metric censors** — both PPO arms at or above the ceiling, or either
  failing to cross the 30% threshold. A fact about the manipulation, not about the wiring. The
  calibration grid is the registered remedy and it is spent on the pilot; if no grid point lands
  inside the band the campaign does not launch and the change is amended under a dated note. The
  recipe is never tuned to make the arms separate.
- **The gate fails** (the wild type does not beat its own frozen floor). A finding about the platform.
  It licenses fixing the platform and nothing about the wiring.

**A null is a real possibility and is worth the compute either way.** The wild type's advantage on
the thermal cell is the phase's only positive result; knowing whether it is a general property of
hard tasks or specific to one cell's configuration is the difference between a broad claim and a
narrow one.

## Arms and the family

Four arms on 32 paired seeds (1–32), 3000 episodes, `wiring` the only key that differs within a
wiring pair and `freeze_updates` the only key within a learning pair, `rewire_seed` unset so each
seed's rewiring derives from its run seed:

| arm | wiring | PPO |
|---|---|---|
| `wt_ppo` | wild type | yes |
| `rn_ppo` | rewired null | yes |
| `wt_frozen` | wild type | no |
| `rn_frozen` | rewired null | no |

The cell also needs peak-axis entries — its scored metric (full-clear success, as the food-only cell
was) and its 5-point minimum — even though after 057's amendment the peak axis carries only the
gates.

Four tests, corrected together under BH-FDR at α = 0.05, one-sided and paired, with 80% bootstrap
CIs — the contrast, two learning gates and the untrained prior — through the committed
`scripts/analysis/wiring_premise.py`, which adds this cell alongside the two it already carries.
The **gates are read before the contrast** and the **efficiency axis is the primary**, exactly as the
thermal cell's campaign was scored after its amendment; the peak axis carries the gates only.

**The minimum effect is the registered 20% off time-to-competence**, unchanged, so this arm and the
thermal one are compared on the same bar.

## Why 32 seeds, decided now

V.1 ran 16, 16, then 32. Its second panel of 16 reached q = 0.115–0.334 on an effect its third panel
of 32 confirmed at q = 0.012–0.027: at this effect size and this seed-level heterogeneity, a paired
rank test at 16 is underpowered. Registering 32 spends about 2.5 hours to avoid discovering that
again.

## The pilot, on disjoint seeds 101–104

The food-only cell has already saturated once, at this exact target, in 057's pilot. The pilot's job
here is larger: **it chooses the episode budget.** For each grid point it reports whether the arms
learn, how far they sit from the ceiling, whether both cross the 30% threshold the primary metric
needs, and what a run costs. One budget is then frozen and the campaign runs at it. If none is inside
the band, the campaign does not launch.

## What this design does not do

- It does not run any plasticity rule.
- It does not re-run or re-read the thermal or food-only panels; their verdicts stand as committed.
- It cannot separate the sensory projection from lethal-zone mortality. That needs an arm this change
  deliberately leaves unregistered.
