# I.3 — the unexamined knobs

**Run 2026-09-11. The eligibility horizon is the limiting setting, and raising it recovers the
rule.** Homeostasis and the exploration noise are near-neutral. The registered yardstick platform
was not usable and was replaced, as the change records.

## The horizon

Gap closed by delay, at σ = 0.2, 8 seeds, 20,000 trials. In brackets, the **nominal credit
ratio**: the decay weight on the scored step over the total. It weights every step equally and
asks only what the decay does — it is not the scored step's share of the eligibility tensor, whose
contributions are outer products with norms this does not measure.

| `trace_decay` | D = 0 | D = 2 | D = 5 | D = 10 | D = 20 |
|---|---|---|---|---|---|
| **0.9** (pinned) | 89.0% [1.00] | 82.7% [0.30] | 71.0% [0.13] | 44.8% [0.05] ✗ | **−6.8%** [0.01] ✗ |
| 0.99 | 89.0% [1.00] | 84.3% [0.33] | 79.2% [0.16] | 69.3% [0.09] | **45.3%** [0.04] ✗ |
| 0.999 | 89.0% [1.00] | 84.4% [0.33] | 79.8% [0.17] | 70.9% [0.09] | **49.8%** [0.05] ✗ |

✗ marks a cell below the registered bar (half the floor-to-optimum gap). Four cells fail; the
recovery at `0.99` and `0.999` is large but does not clear the bar at twenty steps.

**At the pinned decay the rule dies with delay.** Twenty steps of dilution takes it from 89% of the
floor-to-optimum gap to **below the cue-blind floor** — worse than ignoring the cue entirely. Ten
steps halves it.

**Raising the decay recovers it.** At D = 20, `0.99` closes 45.3% where `0.9` closes −6.8%: a
52-point swing from one setting, on a task where nothing else changed. `0.999` adds little over
`0.99`, so the useful range is bounded.

The nominal credit ratio tracks the collapse closely — Spearman **+0.989** against gap closed over
the fifteen cells — which is what makes dilution the readable account of it: the scored step's
weight is swamped by the steps that follow, and a longer trace keeps more of it.

**It is a correlation, not a threshold, and the grid does not support one.** Four cells fail
(`0.9|10`, `0.9|20`, `0.99|20`, `0.999|20`) and their ratios run to **0.051**, above the 0.043 of
another failing cell and not far below the 0.086 of the lowest passing one. With fifteen cells and
one scored action there is no cutoff to read off, only an ordering.

## The other two

| knob | gap closed | verdict |
|---|---|---|
| homeostasis **on** (pinned) | 89.0%, 8/8 | passes |
| homeostasis **off** | **95.2%**, 8/8 | passes |
| action noise 0.22 | **92.3%**, 8/8 | passes |
| action noise 0.37 (pinned) | 89.0%, 8/8 | passes |
| action noise 0.61 | 77.0%, 8/8 | passes |
| action noise 1.0 | 48.4%, 8/8 | **fails** |

Neither is holding the rule back in any important way. Homeostasis costs about six points of gap —
real, consistent with it being a constraint the rule cannot leave, and small. The pinned exploration
noise is close to best; 0.22 is slightly better and 1.0 fails, which reproduces from the other side
what the panels found when they pinned it (std 1.0 capped every plastic arm near its floor).

## What this establishes

- **The eligibility horizon is the setting that matters, and it is the one nobody had tested.** The
  other two are near-neutral on the platform where the rule works.
- **It offers a mechanical account of the phase's central puzzle.** The rule's only success is a
  one-step task; every multi-step task has failed — the connectome clone, its endpoints, the MLP
  yardstick. Episodes there run 244 to 2400 steps, orders beyond the D = 20 at which the pinned
  decay already falls below the floor. That is not proof the horizon caused those failures, but it
  is the first mechanism that predicts them rather than describing them.
- **It is actionable.** Unlike every earlier negative result, this one names a setting and a
  direction: raise `trace_decay`.

## What it does not establish

- Not that raising the decay fixes the connectome. This is a four-cue association with one scored
  action; the real task has a 2400-step episode, a 302-neuron recurrent substrate and a reward
  stream that arrives throughout. The delayed control shows the horizon *can* be limiting and that
  the setting *can* recover it, on a task where nothing else can explain the result.
- Not a value for `trace_decay`. The grid brackets the effect; it does not select a setting for the
  real task, whose step counts are two orders larger than anything tested here.
- Nothing about homeostasis or exploration noise on a multi-step task. Both were examined at D = 0.

## Method note

The registered platform — the MLP yardstick under I.1's rule — was not usable: on disjoint seeds it
ends below its own frozen control, with no learning signal to vary a knob against. The horizon was
also unmeasurable on the undelayed control, which resets the trace every trial. Both are recorded in
the launch record, with the delayed control that replaced them and the reason its filler must be
nonzero: under trace normalisation a pure scalar decay is divided out, and a zero-filler delay reads
identically at every length.
