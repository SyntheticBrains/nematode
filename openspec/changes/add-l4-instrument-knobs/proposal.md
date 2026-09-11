# The unexamined knobs, and a control that can examine the third (7a-ii I.3)

## Why

Three settings have been pinned since A.3 and never examined: the eligibility horizon
(`trace_decay 0.9`, about ten steps), the homeostatic norm sphere (every unit returned to its
construction norm after every update, a constraint the rule cannot leave), and the exploration noise
(`initial_log_std −1.0`, never trained under the plastic rule). I.3 registered them as "each
examined on the yardstick under I.1's rule".

**That platform does not work, and a pilot says so.** On disjoint seeds 101–102 at 3000 episodes,
the MLP yardstick under the node-perturbation rule ends at **0.00% full clears and 0.11 / 0.00 mean
foods — below its own frozen control at 0.90 / 0.35.** Seed 101 rose to 2.70 foods by the second
sixth of its run and collapsed to 0.11. There is no learning signal on the yardstick to vary a knob
against, and varying knobs on an arm that does not learn is the error block I exists to correct.
Logbook 040 reached the same place under the original rule and explained it structurally: a local
rule on a dense stack collapses the representation without decorrelation.

**And the horizon cannot be measured where the rule does work.** The positive control is one step —
it resets the trace every trial, by design, because that removes the horizon confound. Setting
`trace_decay` to 0.0, 0.9 or 0.99 there gives results identical to four decimals. So the one knob
most likely to matter has no platform at all.

That gap is the interesting part. **The rule's only demonstrated success is on a task with no
temporal credit assignment**: it covers 88% of the floor-to-optimum gap on a one-step association
and has failed on every multi-step task tried — the connectome clone, its endpoints, and now the
yardstick. An eligibility that decays to 0.35 over ten steps, against episodes of 244 to 2400, is
the standing suspect for that gap, and it is exactly the knob nothing can currently test.

## What Changes

- **Two knobs on the control that can hold them**, where I.1's rule demonstrably learns:
  homeostasis on and off, and a grid over the exploration noise, with the pinned values as the
  baseline arm. Cheap, and decisive about whether either is holding the rule back.
- **A delayed positive control, so the horizon becomes measurable.** The task gains a delay `D`:
  the scored action is taken while the cue is visible, then `D` further steps elapse before the
  reward arrives, so the eligibility for that action has decayed by roughly `trace_decay^D` when
  the modulator reaches it. **The closed-form bounds are unchanged** — they depend on the targets
  and the action noise, not on when the reward lands — so the floor, the optimum and the pass rule
  carry over exactly, and **`D = 0` SHALL reproduce the committed one-step numbers**, which is the
  regression anchor that makes the extension trustworthy.
- **The horizon examined on it**: `trace_decay` × `D`, with the pinned 0.9 as the baseline, over
  delays that bracket the ten-step scale the setting implies.
- Records under `supporting/054-l4-instrument-knobs/`, tests, docs.

Out of scope: any panel; the low-σ programme, which 052 licensed and which stays open; any change
to a committed verdict. The yardstick pilot is disclosed as a pilot on disjoint seeds and is not
part of any registered result.

## Capabilities

**Modified**: `plasticity-evaluation` (a control that can measure the eligibility horizon, and what
examining a pinned setting requires).

## Impact

- New: the delay in `positive_control.py`, arms in `l4_rule_positive_control.py`, the supporting
  directory. Edited: `CHANGELOG.md`, docs.
- No substrate or rule change. `D = 0` and the pinned knob values leave every existing arm
  bit-identical, and the committed control records stay reproducible.
