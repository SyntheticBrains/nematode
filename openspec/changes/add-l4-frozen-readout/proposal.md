# Is the frozen readout what stops the rule? (R.1d)

## Why

[R.1c](../../../docs/experiments/logbooks/061-l4-reduced-perturbation.md) closed the perturbation
dimension: from **1208 draws per scored decision down to 39**, no declared set beats its own frozen
control by the registered minima and none reaches competence, while credited synapses drift
**1.37–1.38× their own norm at every set**. With σ and the action noise also calibrated on this
substrate and both null, **three independent axes leave the learning arm between 2.4 and 4.4 foods of
20** against PPO's 19.31 on the same cell. That is a structural limit, not a hyperparameter one.

**One structural asymmetry is measured rather than guessed.** The connectome's `plastic_weights` is
**`w_chem` alone** — `food_gains`, `readout` and `log_std` live in the optimiser's parameter list, used
**only under PPO** — so under the rule all three are frozen. After 300 PPO episodes on this cell:

| tensor | relative norm change | cosine to init | writable by the rule |
|---|---|---|---|
| **`readout`** (2×4, motor classes → action) | **0.783** | +0.865 | **no** |
| `w_chem` (3709 chemical synapses) | 0.486 | +0.899 | yes |
| `food_gains` (sensory projection) | 0.177 | +0.985 | no |

Of the three frozen tensors, two are already eliminated: the sensory projection barely moves under PPO,
and the action-noise scale was swept by hand in R.1c's calibration and is null. **The readout is the one
left**, and it shows the largest relative norm change of the three — which is not the same as being the
most important, since eight matrix entries and 3709 synapses are not commensurable. Hence a diagnostic.

## What Changes

- **A preparation step** that writes, per seed, a checkpoint carrying a **substituted readout** and
  everything else at that seed's **own fresh initialisation**. Only the readout moves: a wholesale load
  would also bring PPO's `log_std`, changing the action noise the rule runs at, and PPO's `w_chem`,
  which would make the arm a clone assay rather than a readout test.

- **Three readouts at one mask**, held at R.1c's best measured operating point (`motor`, σ 0.1,
  `initial_log_std: -1.0`):

  | readout | what it is | status |
  |---|---|---|
  | `anatomical` | the committed default — speed as the B-vs-A motor-class contrast, turn as D-vs-V, unit-normed | R.1c's committed pair (learning 3.751, frozen 3.150), **reused only if a load-path equivalence test passes** |
  | `ppo` | harvested from a PPO run on this cell at the same seed, **at the arm's own action scale** | new |
  | `rotated` | a random direction at the **same Frobenius norm** as that seed's PPO readout | new |

- **The `rotated` arm is the discriminator, and it is why this is worth running.** Without it a positive
  `ppo` result cannot distinguish "PPO found a good readout" from "the anatomical prior is bad and almost
  any change helps" — and those have different follow-ups.

- **Each new readout carries its own frozen control.** A better readout raises the do-nothing floor too,
  so the anatomical arm's floor is not the right null for it. R.1c's committed `motor` arms supply the
  anatomical pair; only the two new readouts need new floors.

- **Eight PPO harvest runs** on this cell at seeds 1–8, **pinned to the arm's own action scale**, which
  also supply a **matched** PPO reference: 058's 19.31 foods was measured over 32 seeds at an action std
  of 1.0 against these arms' 0.368, and its weights are no longer on disk either way.

- **What a positive result may not be cited as**, registered before the run: a rule that needs a
  gradient-trained tensor is **not** a biologically plausible local learner, so this cannot satisfy D1
  whatever it returns. Its positive branch points at **substrate fidelity** — a better-grounded
  anatomical readout — and explicitly not at PPO.

Out of scope: making the readout plastic, which [Logbook 040](../../../docs/experiments/logbooks/040-l4-panel.md)
already measured as destructive (a 96% forager collapsed to zero within three episodes once its readout
learned); any other mask; and R.2.

## Capabilities

**Modified**: `plasticity-evaluation` — a diagnostic that borrows a gradient-trained component states
that it cannot satisfy a plausibility deliverable, and a substituted component is tested against a
same-magnitude random control so "this component" is separable from "not the default".

## Impact

- New: the preparation script and its tests; six configs; records under
  `supporting/062-l4-frozen-readout/`; Logbook 062.
- Edited: the experiments index, `CHANGELOG.md`, the tracker (R.1d), the roadmap only if the reading
  changes.
- Compute: **40 runs** — 8 PPO harvests at ~950 s and 32 arm runs at ~1800 s — about **1h 15m** at the
  measured parallelism, plus 16 more only if the load-path equivalence test fails.
