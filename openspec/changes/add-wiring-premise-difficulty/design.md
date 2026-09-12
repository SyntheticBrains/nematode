# Design: isolating the two non-temperature routes to difficulty

## The question, and why the registered framing needed sharpening

057 asked "pathway or difficulty". The two cells it compared differ in **temperature, episode length
and satiety gain**, so a difference between them cannot be attributed to temperature while the other
two stand untested. This change tests the conjunction of the other two.

**The arm**: the food-only cell at `target_foods_to_collect: 20` — the variant 057's pilot
characterised — with the thermal cell's own budget: `max_steps: 500` and
`satiety_gain_per_food: 0.2`. Nothing else moves. Food sensing is exactly the cell that saturated;
temperature is absent; the difficulty now comes from a time limit and from food that does not
meaningfully feed the agent.

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
- **The cell saturates** (both PPO arms ≥ 90% full clear). The manipulation did not make it hard
  enough to discriminate, which is a fact about the manipulation and not about the wiring. Registered
  remedy, applied once: `max_steps` 500 → 350, nothing else. The recipe is never tuned until the arms
  separate.
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

The food-only cell has already saturated once, at this exact target, in 057's pilot. So the pilot's
job is the same three questions: does the cell still learn under the tightened budget, what does a
run cost, and where does it sit relative to the ceiling. If it saturates, the registered remedy is
applied once and the campaign waits.

## What this design does not do

- It does not run any plasticity rule.
- It does not re-run or re-read the thermal or food-only panels; their verdicts stand as committed.
- It cannot separate the sensory projection from lethal-zone mortality. That needs an arm this change
  deliberately leaves unregistered.
