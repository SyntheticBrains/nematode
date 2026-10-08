## Why

C.1e's first probe ran the connectome through the kinematic body at C.1d's 500-step cell: the wild type
and the chemical-only null, learning and frozen, seeds 1601–1604. **Every arm plateaued at 0%.** The
learning arms ate 3–4 of 20 foods against the frozen arms' 2.3, with no trend over 3,000 episodes. The
MLP reaches 92.5% on the same cell, and the connectome reached 80% as a point worm (C.0).

The cause is the readout's scale:

- PPO's entropy bonus drives every learner's exploration noise to its cap, σ = 7.4, the MLP's included.
- **The MLP answers it** with an unbounded mean. Its trained means sit at the edge of the action
  range: |mean| 0.95 after the squash.
- **The connectome cannot.** Its mean is its settled rates through the column-normalised anatomical
  map, within [−1, 1] before the squash, with nothing to scale it. Its trained means average 0.28 and
  never exceed 0.67, so its drive is noise.

D18 specified the remedy, and C.1 left it out: *"The learnable part of the muscle readout is the gain
vector over muscle groups, identical in size across arms."* C.1 read D18's intent as no learnable
readout at all.

## What Changes

- **A learnable gain vector on the body drive.** One log-gain per drive output, 25 in all, starting at
  zero (a gain of 1), multiplies the anatomical drive before the policy's squash. The map itself (which
  cells, their signs, the column weights) stays fixed anatomy. The gain takes the readout's place in
  the learnable parameters, so the trainable count is identical across wirings.
- **The probe re-runs** on the same seeds and configs, to read whether the connectome now learns the
  500-step cell, and how readable a wiring contrast would be there. C.1e is planned on what it shows.

## Capabilities

**Modified**: `connectome-ppo-brain`, its requirement "The anatomical neuromuscular readout".

## Impact

- `brain/arch/connectome_ppo.py`: the gain parameter, its use, and its place in the learnable
  parameters.
- Tests. Nothing registered has run on the body-drive connectome, and the MLP has no anatomical readout,
  so C.1d's control is unaffected.
