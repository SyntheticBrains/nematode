# V.3 — difficulty or temperature: the registered protocol

Registered in `openspec/changes/add-wiring-premise-difficulty`, reviewed and committed **before** the
calibration pilot ran.

## The question

[Logbook 057](../../057-wiring-premise-contrast.md) found the wild-type wiring reaching competence
~35% sooner than its degree-preserving rewired null on a foraging cell under lethal thermal pressure,
and **nothing** on the food-only cell, where both wirings clear 100% and reach competence inside forty
episodes of three thousand.

057 registered that pair as "pathway or difficulty". **How the episodes actually end says the framing
was too coarse.** Across all 32 seeds of 057's third panel:

| ending | count | share |
|---|---|---|
| `completed_all_food` | 69,366 | 72.3% |
| `health_depleted` | **14,098** | **14.7%** |
| `max_steps` | **11,927** | **12.4%** |
| `starved` | 609 | 0.6% |

So the two cells differ in three respects and **only two bind**: temperature and the episode budget.
`satiety_gain_per_food` at 0.2 is nearly inert — starvation ends 0.6% of episodes — and is carried
into this cell for config-matching, not as pressure.

**This arm isolates the budget**: the food-only cell, no temperature, difficulty from the episode
limit alone.

## The calibration, and why the budget is not inherited

The obvious construction takes the thermal cell's `max_steps: 500`. **The committed data says that
cannot bind.** On the food-only `_t20` cell the plateau-tail episodes complete in a **mean of 310
steps (max 647)**, and a policy trained under a tighter budget needs fewer steps than one trained at
800\. At 500 the arm would saturate exactly as 057's pilot did.

So the budget is calibrated on **disjoint pilot seeds 101–104** over a grid declared here —
**`max_steps` ∈ {150, 250, 350}** — and one point is frozen for the campaign. This is the
sweep-before-pin the phase protocol asks for, which taking a number from another cell would have
skipped.

## The band has two edges

The primary metric is `episodes_to_30pct_success`, and the committed harness returns the horizon for a
seed that never crosses the 30% rolling full-clear rate. So the cell must be:

- **hard enough** that both PPO arms stay clear of the 90% full-clear ceiling, and
- **easy enough** that both cross the 30% threshold — registered floor **80% of seeds per arm**, set
  below 1.0 because one non-crossing seed is not censoring (057's own thermal panel sits at 98% and
  100%).

Outside that band the metric is censored at 3000 for every seed and **discriminates nothing while
looking exactly like a null**. The pilot reports both edges per grid point. If no point is inside,
the campaign does not launch and the change is amended under a dated note.

## Arms, family and reading

Four arms on **32 paired seeds (1–32)**, 3000 episodes, at the frozen budget: `wt_ppo`, `rn_ppo` and a
frozen-weights floor per wiring. `wiring` is the only key differing within a wiring pair,
`freeze_updates` within a learning pair, and `rewire_seed` is unset so each seed's rewiring derives
from its run seed.

Four tests (V9–V12) corrected together under BH-FDR at α = 0.05, one-sided and paired, with 80%
bootstrap CIs, through `scripts/analysis/wiring_premise.py`. **The gates are read before the
contrast**; the **efficiency axis is the primary** with the registered **20% minimum off
time-to-competence**, the same bar the thermal cell was held to.

**32 seeds, decided now.** V.1 ran 16, 16, then 32: its second panel of 16 reached q = 0.115–0.334 on
an effect its third panel of 32 confirmed at q = 0.012–0.027. At this effect size a paired rank test
at 16 is underpowered, and registering 32 spends ~2.5 hours to avoid learning that twice.

## Outcomes, fixed before the run

- **The advantage appears** (significant, ≥ 20% gain). **Difficulty is sufficient and the
  thermosensory pathway is unnecessary.** 057's claim generalises from "a foraging cell under thermal
  pressure" to "a foraging cell hard enough to discriminate". It would still not establish that *any*
  hard task shows it — only hard foraging cells of this family.
- **The advantage is absent, cell inside the band.** Difficulty alone is **not** sufficient. This does
  **not** cleanly implicate the sensory projection, because lethal-zone mortality is
  temperature-dependent too: it narrows to "something about the thermal configuration", with the
  projection and the mortality pressure both live. The arm separating those rewires only non-sensory
  edges, needs a new rewiring mode, and is named rather than run.
- **Outside the band** — saturated, or censored below the 80% floor. A fact about the manipulation,
  not the wiring. The grid is the remedy and it is spent on the pilot.
- **The gate fails.** A finding about the platform; it licenses fixing the platform and nothing about
  the wiring.

## Honest prior

**Genuinely uncertain, which is why it is worth running.** V.2 closed the simplest pathway story — the
wild type's route from AFD to the motor readout is *longer* than every rewiring's (3 hops against
1–2), so whatever the projection contributes is not hop count. That makes "the wiring helps whenever
the task discriminates" the more parsimonious reading, and this arm can support it. Against that,
057's effect lives on a cell whose difficulty is mortality, and a pure time limit may simply not
produce the same pressure.

## Reproduce

```bash
# calibration pilot: three budgets x four arms x seeds 101-104
B=configs/scenarios/foraging/connectomeppo_small_continuous2d_fick_adaptive_klinotaxis
for S in 150 250 350; do
  uv run python scripts/run_campaign.py \
    --config ${B}_hard${S}.yml --config ${B}_hard${S}_rewired_null.yml \
    --config ${B}_hard${S}_frozen.yml --config ${B}_hard${S}_rewired_null_frozen.yml \
    --seeds 101-104 --runs 3000 --output-dir campaigns/wiring-premise-hard-pilot-${S} \
    -- --theme headless
done
```
