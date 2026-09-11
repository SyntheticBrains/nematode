# I.3b — the eligibility horizon on a multi-step task: the registered protocol

Registered in `openspec/changes/add-l4-horizon-multistep`, reviewed and committed **before** the run.

## The question

I.3 found the horizon limiting on the one-step control: 89% of the floor-to-optimum gap undelayed,
44.8% at ten steps of delay, **−6.8% at twenty** — below the cue-blind floor — and `trace_decay 0.99`
takes that twenty-step cell to 45.3%.

**Every result in this phase ran at the 0.9 default.** No panel config sets `trace_decay`. Their
episodes run 244 to 2400 steps, two orders beyond where the default already fails on a task with one
scored action. Whether the horizon explains the phase's one-step-success / multi-step-failure
pattern has therefore never been tested on a multi-step task.

## The arms

Six, on the MLP yardstick under the node-perturbation rule at σ = 0.2, seeds 1–8, 3000 episodes:

| `trace_decay` | learning arm | frozen control |
|---|---|---|
| 0.9 (the pinned default) | ✓ | ✓ |
| 0.99 | ✓ | ✓ |
| 0.999 | ✓ | ✓ |

Each horizon carries **its own** frozen control, because what the perturbation costs a policy is not
constant across the setting being varied. The connectome is deliberately excluded: ~10 h a run
against ~8 min, and it is registered only if the yardstick moves.

## The reading

**Plateau-tail mean foods**, with I.2's family, learning arm against its own frozen control, paired
by seed, one-sided, BH-FDR across the three horizons.

The full-clear metric cannot serve: every yardstick arm in the committed 040 table sits at its
floor (mean **1.05%**, no seed competent there). Where a seed does clear the threshold in this
campaign, the level contrast still needs one in **both** arms, so the record derives which arm
lacks one rather than asserting that none exists. The committed 040 yardstick values are a **descriptive
reference only** — they ran under the original rule.

**Significance is not sufficient.** A paired rank test at eight seeds fires on the consistency of
the sign, not the size of the shift: eight seeds moving one way reaches q = 0.012 whether the shift
is 0.05 foods or 2.0. The yardstick's committed foods value is **0.35 of 10** over a seed range of
0.06–0.67 (cv 0.61), so a clean-looking but meaningless result is reachable. **The horizon counts as
transferring only if the shift is significant *and* at least 0.5 foods** — about the I.3 pilot's
frozen-vs-learning gap (0.90 against 0.11) and ~1.5 within-arm sd of the committed table.

Each arm also reports its **weight distance from its frozen control**, so a horizon too short
(little drift, no gain) is distinguishable from one too long (substantial drift, no gain).

## Outcomes, fixed before the run

- **Beats its control at a raised horizon and not at 0.9, by ≥ 0.5 foods.** The mechanism transfers;
  the seven negative results become findings about an instrument at a crippling setting; a
  connectome arm becomes the obvious next registration.
- **Beats its control at every horizon including 0.9.** Something other than the horizon changed
  since the I.3 pilot. Registered as a **stop**, not a result — void until found.
- **Beats its control at no horizon.** The horizon story does not carry off the control. I.4 records
  that it was tested and did not transfer. **This is the expected outcome.**
- **Every arm at the floor and indistinguishable.** The yardstick is not a platform for this
  question either, and the finding is about the platform. Live, not a formality: the yardstick is
  floor-adjacent on *both* metrics.

## Honest prior

**This probably does not work.** I.3's recovery is partial — at twenty steps even `0.999` stays
below the bar. A 0.99 trace effectively never decays inside a 2400-step episode, trading a horizon
too short to bridge the delay for one too long to assign credit. And Logbook 040 located the
yardstick's failure elsewhere: a local rule collapsing a dense stack without decorrelation, which no
horizon setting fixes. It is worth an hour because a negative is a real input to I.4.

## Disclosure

The I.3 pilot that motivated this ran on seeds **101–102**, disjoint from the registered 1–8.

## Reproduce

```bash
P=configs/scenarios/foraging_predator_thermal/mlpppo_small_continuous2d_combined_klinotaxis_plastic_nodepert
uv run python scripts/run_campaign.py \
  --config ${P}_td09.yml --config ${P}_td09_frozen.yml \
  --config ${P}_td099.yml --config ${P}_td099_frozen.yml \
  --config ${P}_td0999.yml --config ${P}_td0999_frozen.yml \
  --seeds 1-8 --runs 3000 --output-dir campaigns/l4-horizon-multistep \
  -- --theme headless --track-experiment
```
