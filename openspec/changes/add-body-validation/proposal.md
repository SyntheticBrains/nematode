## Why

C.3 asks whether what the connectome and the MLP do through the kinematic body looks like a worm, at
the level of posture and behaviour, graded against thresholds fixed in advance. C.1d graded the MLP's
crawl on four kinematic instruments. C.1e trained 64 wild-type, 64 chemical-only-null and 64 MLP runs
through the body. C.3 reads those trained runs, so it needs no new training except one control.

The roadmap names C.3's measures: the eigenworm posture spectrum (Stephens et al. 2008), undulation
frequency and amplitude, omega-turn geometry, and Logbook 035's klinokinesis and weathervane curves
re-derived from emergent kinematics, with Wormlight's thresholds and instruments where they apply.

## What Changes

- **The eigenworm basis and real postures are vendored**: the Stephens group's eigenworms as
  distributed with WormPose (BSD-3-Clause) and 6,655 real postures from the OIST Physics of Behavior
  tutorials (CC BY 4.0), with licences and SHA-256s in `data/` provenance.
- **New instruments**:
  - the variance the first four eigenworms capture in body postures;
  - undulation amplitude in the first two eigenworms' plane, beside the real postures';
  - omega turns, by Wormlight's definition;
  - forward bouts.
- **The posture record carries the body's frame angle**, which omega turns need.
- **The evaluation harness captures behaviour**, so the trained runs' turning can be read by Logbook
  035's bias-curve harness.
- **One control is trained**: a derivative-mode MLP through the body (no synthetic head-sweep), as
  Logbook 035's specificity control, learning and frozen (its floor).
- **A registered validation**: Wormlight's pass and partial bands and 035's sign-only curve grading,
  fixed before any reading, per arm. The readings the body's generator sets by construction are graded
  as body checks, separate from the emergent behaviour readings.

## Capabilities

**Modified**:

- `realworm-behavioural-validation`: posture instruments, and behaviour capture in the evaluation
  harness.
- `continuous-2d-environment`: the posture record's frame angle.

## Impact

- `data/posture/`: the basis and the real postures, with provenance.
- `env/body.py`, `env/continuous_2d.py`: the frame angle in the posture record.
- `validation/body_kinematics.py`: the new instruments.
- `scripts/analysis/body_kinematics_eval.py`: behaviour capture.
- `scripts/analysis/body_validation.py`: new; evaluation over the arms, the grades.
- The control's two configs and their generator, tests, Logbook 086, tracker C.3.
