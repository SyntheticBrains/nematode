## Overview

The brain's cells drive the muscles through the anatomical map, the drives modulate a body-level wave,
and the wave moves the body through resistive-force theory. Every arm acts through the same 25-number
drive vector. The connectome reaches it through its anatomy and the MLP directly, so C.1d tests the body
and C.1e the wiring.

## Decisions

### Decision A: The neuromuscular drive map (C.1a)

**Muscles to segments.** `muscle_position(name)` parses each body-wall muscle into its quadrant
(`dBWML`, `dBWMR`, `vBWML`, `vBWMR`) and its position, 1 being most anterior. Each quadrant's positions
pair into 12 segments: positions 1–2 make segment 1, and so on. Ventral-left has 23 positions, so its
last segment holds one muscle.

**The map.** For the 162 cells with a neuromuscular junction (Emmons 2024, identical to Cook 2019):

- **Each entry** is `count × sign` from the cell to each quadrant-segment. The sign follows body-wall
  muscle receptors (Richmond & Jorgensen 1999): acetylcholine +1, GABA −1. Glutamate, dopamine and
  unknown transmitters get 0, since none has an established body-wall effect.
- **The 32 zero-weight cells are listed in the record**, as the roadmap asks of anything dropped.
- **Each quadrant-segment column is divided by its total absolute count.** A column's drive from rates in
  `[-1, 1]` is then a weighted mean in `[-1, 1]`, comparable across segments of different innervation.
- **The map is fixed.** It is anatomy, and it is never learned.

### Decision B: The body-drive action (C.1b)

`action_space: Literal["speed_turn", "body_drive"]`, defaulting to `speed_turn` (byte-identical). Under
`body_drive` the action is 25 numbers in `[-1, 1]`:

- `dorsal_i`, the mean of the dorsal-left and dorsal-right drive for segment `i`;
- `ventral_i`, the same for the two ventral quadrants;
- `direction`.

The policy is the same tanh-squashed Gaussian over 25 dimensions, with a learnable `log_std` of width 25.

- **The connectome's policy mean** is its settled rates through the map, and its direction is the
  forward-minus-backward motor-class contrast `mean(VB, DB) − mean(VA, DA)`. There is no learnable
  readout: PPO learns the chemical weights and sensor gains, which shape the rates the anatomy reads.
  That is D18's intent, and the trainable-parameter count cannot differ by wiring.
- **MLP-PPO** emits the 25 numbers from its actor, sized from the action space.
- **LSTM, CfC and transformer PPO** refuse `body_drive` until C.4 needs them.

### Decision C: The body (C.1c)

`continuous.body_model: Literal["point", "kinematic"]`, defaulting to `point`. The kinematic body has 12
segments of `body_length_mm / 12`. Its state is each segment's curvature, the generator's phase state,
the head position and the head heading. The segment positions follow from integrating curvature from
the head.

**The generator** (D19 as recorded).

- **Phase.** The head's curvature relaxes toward a target `±1` with time constant `τ_h`, and the target
  flips sign when the curvature crosses `±θ`. That is Ji et al. 2021's relaxation switch. The period is
  set by `P = 2 τ_h ln((1+θ)/(1−θ))`, so `τ_h` follows from the period.

- **Propagation.** A relay passes each segment's normalised wave value to the next with a delay
  `Δ = P / (12 λ)`, `λ` being the wavelength in body lengths. That is Wen et al. 2012's front-to-back
  relay. A negative direction channel runs the relay tail-to-head, with the switch moved to the tail,
  which is the mirrored backward relay.

- **Drive modulation.** Segment `i`'s curvature is

  ```text
  κ_i = A₀ · a_i · w_i + B₀ · b_i
  a_i = (1 + (dorsal_i + ventral_i) / 2) / 2   ∈ [0, 1]   amplitude, which sets speed
  b_i = (dorsal_i − ventral_i) / 2             ∈ [−1, 1]  bias, which sets steering
  ```

  with `w_i` the relay's wave value at the segment.

**Locomotion: resistive-force theory.** Within a step, the generator advances in `n_sub` sub-steps
(default 20) across 5 worm-seconds, the step C.0a recorded. At each sub-step the shape change gives each
segment a velocity relative to the body frame. Each segment's drag is `−(c_t v_t t̂ + c_n v_n n̂)`. The
body's rigid translation and rotation are the solution of a 3 × 3 linear system that zeroes the net drag
force and torque, the standard force-free swimmer of the Gray–Hancock class. The head's new position becomes
`pos_continuous`, so food capture, sensing and contact zones sit at the head. **`heading_rad` is the
direction from the body's midpoint to the head**, not the head segment's tangent. A step is 5
worm-seconds, not a whole number of periods, so the tangent would sample the head swing at an arbitrary
phase each step, adding noise to sensing and contact zones that the worm does not have. The direction
from midpoint to head is stable at the wave's scale. The environment's lateral sample still models the
head sweep, as it does for the point worm. A head step that would leave the arena is clamped there, and the
body follows.

**Parameters and their sources.**

| parameter | default | source |
|---|---|---|
| `P` | 3.1 s | C.3's crawl-frequency band, 0.2–0.45 Hz, midpoint |
| `λ` | 0.65 body lengths | C.3's crawl-wavelength band, 0.5–0.8, midpoint |
| `θ` | 0.5 | a shape parameter of the switch; the period is held by `τ_h` |
| `A₀` | peak curvature × body length | **to be sourced** before C.1d, then calibrated |
| `B₀` | steering gain | **to be calibrated** on C.1d's MLP and frozen |
| `c_n / c_t` | drag anisotropy on agar | **to be sourced** before C.1d, then calibrated |

The defaults written into code for `A₀`, `B₀` and the anisotropy are placeholders. The design does not
claim them as measured. Plausible ranges are recalled, not checked here, so they are verified against
their sources before C.1d registers. Every free parameter is calibrated **once on C.1d's MLP positive
control and frozen across arms**, with a registered sensitivity check (D18).

**Feasibility arithmetic** (protocol principle 2).

- At `λ = 0.65` and `P = 3.1` the wave travels about 0.21 body lengths per second.
- A crawler at high drag anisotropy moves at a large fraction of its wave speed, so expect roughly
  0.1–0.2 body lengths per second. That is inside C.3's 0.12–0.3 band, and about 0.5–1 mm per 5-second
  step.
- The point worm's full-speed step is 1 mm, so hard350's difficulty will move. C.1d's pilot measures it,
  and C.1e re-establishes its floors and baselines in the new reference frame.

**Cost.** A step adds `n_sub` small linear solves on 12 segments. C.1d's pilot measures the wall-clock
before any panel is sized.

### Decision D: Agreement and refusals

At load:

- `body_drive` needs `kinematic`, and `kinematic` needs `body_drive` on a continuous brain.
- `kinematic` needs `allow_reversal: true`, 8b's frozen setting.
- `signed_speed` is not read under `body_drive` and is refused away from its default there.
- `body_drive` is refused under leaky dynamics and the plastic rules, which need a two-number readout
  this change does not give them.
- `body_drive` is refused with the state-dependent std head, which is sized for two outputs.

### Decision E: What is held fixed

Every point-worm path, the `speed_turn` action space, and every existing config and pinned result are
byte-identical. The renderer keeps drawing its cosmetic trail; drawing the body's own curvature is C.5.

## Risks

- **The body may be slower than the point worm**, as the feasibility arithmetic expects, which makes
  hard350 harder. C.1d measures it, and if the MLP cannot learn the cell in 3,000 episodes, that is
  C.1d's diagnosis.
- **The resistive-force step adds per-step cost.** It is measured on the pilot. If it is too slow, the
  sub-step count is the first lever, with a check that motion is unchanged at half the sub-steps.
- **Unsourced parameters.** `A₀` and the drag anisotropy are flagged and verified before C.1d. Nothing
  here reads a learning result that depends on them.
