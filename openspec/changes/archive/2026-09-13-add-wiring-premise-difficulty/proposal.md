# Difficulty or temperature — what the wiring advantage depends on (7a-ii V.3)

## Why

[Logbook 057](../../../docs/experiments/logbooks/057-wiring-premise-contrast.md) found the wild-type
connectome reaching competence ~35% sooner than its degree-preserving rewired null — 396 episodes
against 613 over 64 paired seeds, all four efficiency metrics at q ≤ 0.001, replicated on an
independent panel. It found **nothing** on the food-only cell, where both wirings clear 100% and
reach competence inside forty episodes of three thousand.

057 read that pair as "pathway or difficulty" and registered the question. **Looking at the two
configs and at how episodes actually end, that framing is too coarse: the cells differ in two
respects that bind, not one.** Across all 32 seeds of 057's third panel the thermal cell's episodes
end 69,366 `completed_all_food`, **14,098 `health_depleted` (14.7%)**, 11,927 `max_steps` (12.4%) and
**609 `starved` (0.6%)**:

| | food-only cell | thermal cell | binds? |
|---|---|---|---|
| temperature | absent | lethal zones | **yes** — 14.7% of episodes |
| `max_steps` | 800 | **500** | **yes** — 12.4% |
| `satiety_gain_per_food` | 20.0 | 0.2 | **no** — starvation ends 0.6% |

So the thermal cell is harder for **two** reasons, one of which is temperature. The satiety setting is
carried into the new cell for config-matching, not because it creates pressure. The effect could
depend on the evolved thermosensory projection, or on nothing more than the task being hard enough
to discriminate — and [V.2](../../../docs/experiments/logbooks/supporting/057-wiring-premise-contrast/probe-v2.md)
has already closed the simplest version of the pathway story: the wild type's route from AFD to the
motor readout is **longer** than every rewiring's (3 hops against 1–2), so whatever the pathway
contributes is not hop count.

## What Changes

- **One new cell**: the food-only cell with no temperature, its **episode budget calibrated on the
  pilot's disjoint seeds** and `satiety_gain_per_food: 0.2` carried for config-matching. The step
  budget is the only non-temperature route that binds, so it is the manipulation — and it is
  **calibrated rather than guessed**, because the committed data says the obvious guess fails: on this
  cell the plateau-tail episodes complete in a **mean of 310 steps (max 647)**, so the thermal cell's
  `max_steps: 500` would not bind at all.
- **The same four arms and the same harness**: wild type and rewired null under PPO, each against its
  own frozen-weights floor, scored on the committed efficiency axis with the registered 20%
  minimum on time-to-competence, through `scripts/analysis/wiring_premise.py`.
- **32 paired seeds registered up front**, not 16. V.1 learned that at this effect size a paired rank
  test at 16 seeds is underpowered — panel 2 failed to reach significance on an effect panel 3
  confirmed — so the sample size is set where the test has power rather than discovered again.
- **A pilot on disjoint seeds first**, with the saturation clause and remedy that V.1's pilot proved
  necessary: the food-only cell has already saturated once at this target.
- **What each outcome licenses, and what it does not.** A positive here makes the claim general and
  the pathway unnecessary. A null does **not** cleanly implicate the pathway, because lethal-zone
  mortality is temperature-dependent too — it narrows to "something about the thermal
  configuration", with the projection and the mortality pressure both live.

Out of scope: the arm that would separate those two (rewiring only non-sensory edges), which needs a
new rewiring mode in the substrate; any local rule; and the shipment decision.

## Capabilities

**Modified**: `architecture-comparison-protocol` (an effect present on one cell and absent on another
is not attributed to one difference between them while others stand untested).

## Impact

- New: four configs, the cell wired into the harness, the records under
  `supporting/058-wiring-premise-difficulty/`, Logbook 058.
- Edited: the experiments index, `CHANGELOG.md`, tracker (V.3), roadmap if the reading changes.
- Compute: 128 registered runs plus a pilot, on a cell measured at ~950 s a learning run.
