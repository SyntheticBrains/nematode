## Why

C.1d blocks every connectome arm through the body: MLP-PPO must forage through it first (roadmap
§ Phase 8, item 9; protocol principle 4). The roadmap asks for that control to be read with C.3's
adopted kinematic instruments, so that it is a kinematic pass and not only a foraging one. The body
from C.1a–c carries placeholder parameters, and its branch review recorded that an untrained policy
barely moves.

The maintainer chose four settings before this change was written:

- **Neutral drive gives the sourced crawl.** Peak curvature is 18 body-lengths⁻¹, so zero drive gives a
  W-shape and full drive an Ω-shape.
- **Fix the sourced parameters and calibrate only the steering gain.**
- **A fixed forward bias**, the same for every arm.
- **One change for the control and the instruments.**

## What Changes

- **A defect fixed.** C.1c's relayed wave spanned ±0.5, the head switch's threshold, not ±1, so
  every posture was half as curved as intended. It is normalised, with a test.

- **The body takes its sourced parameters:**

  | parameter | value | source |
  |---|---|---|
  | period | 3.33 s | crawl frequency 0.30 ± 0.02 Hz (Fang-Yen et al. 2010) |
  | wavelength | 0.65 body lengths | 0.65 ± 0.03 (Fang-Yen et al. 2010) |
  | drag anisotropy | 10 | 9.4 ± 0.6 (Shen et al. 2012); 222 / 22.1 (Rabets et al. 2014) |
  | peak curvature | 18 at full drive | crawl at A/q ≈ 1 and Ω-shapes at A/q ≈ 2, with qL ≈ 9 (Bilbao et al.) |

- **A reversal threshold.** The wave runs backward only when the direction channel is below −0.5. It is
  the same for every arm, so an untrained worm mostly crawls forward.

- **The steering gain is calibrated once.** A pilot of MLP-PPO through the body at four candidate gains
  chooses it under a rule fixed before the pilot runs. It is then frozen, with its neighbours reported
  as the sensitivity check (D18).

- **The MLP positive control.**

  - MLP-PPO through the body on hard350 with reversal on, on 8 disjoint seeds, learning and frozen.
  - It passes if it beats its frozen floor and every seed reaches competence.
  - A pre-written fallback re-sizes the cell's episode length if the slower body makes 350 steps too
    few.

- **The kinematic instruments**, read on the trained MLPs in evaluation episodes with sub-step posture
  capture:

  - undulation frequency, by the band-crossing rule;
  - wavelength;
  - speed;
  - reversal fraction;
  - each against C.3's adopted bands, with wall-proximal steps excluded;
  - a half-step check that re-reads the same policies at twice the sub-steps.

## Capabilities

**Modified**: `continuous-2d-environment`, with two added requirements: the reversal threshold, and
sub-step posture capture.

## Impact

- `env/body.py`: the sourced defaults, the reversal threshold and the posture recorder.
- `validation/body_kinematics.py`: new; the instruments.
- `scripts/analysis/body_control.py`: new; the calibration rule, the control's gates and the kinematic
  reading.
- `scripts/campaigns/generate_body_control_configs.py`: new.
- Configs, tests, Logbook 083, and tracker C.1d.

## Breaking Changes

The kinematic body's defaults change. Nothing has been registered or run on them; the point worm is
untouched.
