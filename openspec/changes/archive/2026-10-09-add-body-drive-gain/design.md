## Overview

The connectome must learn through the body before C.1e can read a wiring contrast there. Three probes,
each on seeds 1601–1604 at C.1d's 500-step cell, found why it did not and what it needs.

## The probes

| probe | change | wild type, learning | chemical-only null, learning | frozen floors |
|---|---|---|---|---|
| 1 | none (C.1 as merged) | 0, 0, 0, 0 | 0, 0, 0, 0 | 0% |
| 2 | gain vector | 0, 0, 0, 0 | 0, 0, 62.1, 52.7 | 0% |
| 3 | gain vector, dimension-matched entropy | 58.4, 76.1, 48.7, 83.5 | 27.7, 80.7, 18.8, 82.0 | carried from 2 |

Plateaus in percent, seeds 1601–1604. Probe 3 adds an MLP learning arm at the same setting: 74.9, 27.6, 60.0, 69.3. Probe 3's frozen floors carry over from probe 2, since a frozen arm is untouched by the entropy setting.

### Probe 1: the readout cannot scale

Every learner's exploration noise rose to its cap, σ = 7.4. The MLP's trained means sat at the edge of
the action range (|mean| 0.95 after the squash); the connectome's averaged 0.28 and never exceeded
0.67, because its mean is settled rates through a column-normalised map with nothing to scale it.

### Decision A: the gain vector (D18)

D18 specified it: *"the learnable part of the muscle readout is the gain vector over muscle groups,
identical in size across arms."* 25 log-gains, starting at a gain of 1, scale the anatomical drive. The
map stays fixed anatomy; the gains take the readout's place among the learnable parameters.

### Probe 2: learning where the gains outgrow the noise

Two chemical-only seeds learned, climbing from 6–10 foods per episode to about 18 of 20. They are
exactly the seeds whose gains grew furthest: mean gain about 4, direction gain about 21. Every other
seed stayed near a mean gain of 2 and learned nothing. σ stayed at its cap throughout.

### Decision B: a dimension-matched entropy bonus

PPO's entropy bonus is the policy's entropy summed over action dimensions, so its coefficient pushes on
σ in proportion to the action's width. The 0.05 in the hard350 configs was set for the point worm's two
numbers; on the body's 25 it pushes 12.5 times harder, which is what holds σ at its cap. Every
body-drive arm, connectome and MLP alike, runs at **0.05 × 2 / 25 = 0.004**, the same per-dimension
pressure the point-worm configs were set at. C.1d's MLP control ran at 0.05 and passed; C.1e's MLP arm
runs at 0.004 beside the connectome, so the two share the setting.

### Probe 3: both wirings learn

With the dimension-matched bonus every seed of both wirings learns: the wild type averages 66.7% and the
chemical-only null 52.3%, against 0% floors. Exploration noise settles near σ = 2 for the connectome and
1 for the MLP instead of at its cap, and the gains settle near 1.5. No seed reaches the 90% saturation
bar (best 83.5%), so the 500-step cell is readable for a wiring contrast. The MLP averages 58.0% at this
setting, below C.1d's 92.5% at 0.05 on other seeds, with one seed at 27.6%. A connectome learning run
costs about 51 minutes at 12 workers.
