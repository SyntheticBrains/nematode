# Tasks

## 1. The delayed control

- [ ] 1.1 A delay on the control task: cue, scored action, `D` steps against the **uniform
  observation over the cue channels** (nonzero, same dimension, identical every trial), then the
  reward. The bounds are untouched — they depend on the targets and the action noise alone.
- [ ] 1.2 The harness runs a delayed arm: the trace accumulates across the intervening steps, the
  modulator arrives once at the end, and the **alignment gradient is taken at the scored step and
  held** until the update, not recomputed against the neutral observation.
- [ ] 1.3 Tests: `D = 0` reproduces the committed one-step arm's score at a registered seed
  **bit-for-bit**; the floor, optimum and gap are unchanged at every delay; the intervening
  observation carries nothing about the cue; the scored action is the one taken while the cue was
  visible; **the credited step's share of the normalised trace falls with `D`** — the test a
  zero-filler delay fails, since normalisation divides out a pure scalar decay.

## 2. The knobs that the existing control can hold

- [ ] 2.1 A homeostasis arm (on, off) at the pinned values otherwise.
- [ ] 2.2 An exploration-noise grid, std ∈ {0.22, 0.37, 0.61, 1.0}, the panels' own history around the pinned 0.37.
- [ ] 2.3 Each scored by the control's registered pass rule, with the pinned values as the baseline
  arm, and recorded per cell.

## 3. The horizon

- [ ] 3.1 The horizon grid on the delayed control: `trace_decay` ∈ {0.9, 0.99, 0.999} ×
  `D` ∈ {0, 2, 5, 10, 20}, baseline `0.9` at `D = 0`.
- [ ] 3.2 The record states, per cell, whether the arm cleared the registered bar, so the delay at
  which a horizon-limited rule fails is visible rather than inferred.

## 4. Run and record

- [ ] 4.1 Launch record written before the run: the grids, the pass rule, the outcomes and what each
  licenses, and the yardstick pilot disclosed as a pilot on disjoint seeds.
- [ ] 4.2 Run. Records under `supporting/054-l4-instrument-knobs/`.
- [ ] 4.3 `details.md`: what each knob showed, and whether the horizon explains the gap between the
  rule's one-step success and its multi-step failures.

## 5. Close-out

- [ ] 5.1 `docs/architectures.md` or `docs/experiments/README.md`: the delayed control and what it
  is for.
- [ ] 5.2 `CHANGELOG.md`; tracker (I.3) and roadmap, including that the registered yardstick
  platform was not usable and why.
- [ ] 5.3 State what I.4 takes from this: which of the three settings, if any, the seven negative
  results should be re-read against.
