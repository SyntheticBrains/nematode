## Why

C.1 is 8b's body: the brain's motor neurons drive the body-wall muscles through the anatomical
neuromuscular map, and a segmented body turns that drive into movement. This replaces the learned
two-number readout (roadmap § Phase 8, item 9). C.0 froze its prerequisites (Logbook 081): a 5
worm-second step, reversal, Emmons 2024, and D19 decided. This change builds the substrate:

- **C.1a**: the brain-side neuromuscular tensor.
- **C.1b**: the muscle readout.
- **C.1c**: the kinematic body with D19's generator.

C.1d's MLP positive control and C.1e's wiring contrast are their own changes.

The maintainer chose four settings before this change was written:

- **12 body segments.**
- **All 162 cells** with a neuromuscular junction drive the muscles.
- **Zero weight** for the 32 cells that release neither acetylcholine nor GABA.
- **One drive vector for every arm.** The MLP control emits the same drive vector the connectome's
  readout produces.

## What Changes

- **Muscle anatomy.** `connectome/muscles.py` gains a name parser and a segment map. Each of the 95
  body-wall muscles maps to a quadrant (dorsal or ventral, left or right) and to one of 12 segments,
  head to tail.
- **The signed neuromuscular drive map** (C.1a).
  - It maps each presynaptic cell's rate to quadrant × segment drive.
  - Weights are EM counts signed by the cell's transmitter: acetylcholine +1, GABA −1, anything else 0.
  - Each quadrant-segment column is normalised by its total count.
- **A body-drive action space** (C.1b): `action_space: body_drive` on the brain, 25 numbers.
  - Dorsal and ventral drive for each of the 12 segments, and one direction channel, each in
    `[-1, 1]`.
  - The connectome's policy mean comes from its cell rates through the map, with no learnable readout.
    Its direction channel is the anatomical forward-minus-backward motor-class contrast.
  - MLP-PPO emits the same 25 numbers.
  - The other brains refuse it.
- **A kinematic body** (C.1c): `continuous.body_model: kinematic`.
  - A 12-segment body whose curvature comes from D19's generator: a head relaxation switch for the
    phase, and a relay along the body for propagation, mirrored when the direction channel is
    negative.
  - The drives set each segment's amplitude and bias.
  - Displacement within each step comes from resistive-force theory: the body moves so that the net
    drag force and torque on it vanish.
  - The head carries the sensors and the heading.
  - The step integrates the 5 worm-seconds C.0a recorded.
- **Load-time agreement.** A `body_drive` brain needs a `kinematic` environment and the reverse.
  `kinematic` requires `allow_reversal`, the frozen substrate's setting.
- **Off by default.** The point worm and every existing action space are byte-identical.

## Capabilities

**Modified**:

- `continuous-2d-environment`: the kinematic body.
- `continuous-action-policy`: the body-drive action space.
- `connectome-ppo-brain`: the neuromuscular readout.

## Impact

- `connectome/muscles.py`; `connectome/neuromuscular.py` (new, the drive map).
- `env/body.py` (new, the body, the generator and the resistive-force step); `env/continuous_2d.py`
  (dispatching to the body); `utils/config_loader.py` (`body_model` and the agreement check).
- `brain/arch/_policy.py` (body-drive bounds); `brain/arch/connectome_ppo.py` (the readout);
  `brain/arch/mlpppo.py` (action width); `brain/actions.py` (the action type);
  `agent/runners.py` (the dispatch).
- Tests for each part.
- After the change: tracker C.1a, C.1b and C.1c.

## Breaking Changes

None. Every new key defaults to today's behaviour.
