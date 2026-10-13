## Why

C.3 validated chemotaxis through the kinematic body and left thermotaxis to M.9: "Not established:
anything for thermotaxis. Logbook 036's curves are out of scope (M.9)" (Logbook 086). M.9's design
never picked it up. Its panel has now trained 32 wild-type and 32 gap-only-null bodies on the
thermal cell, so the reading needs no new training.

Logbook 092 also describes its lead as "part speed, part stability". A breakdown of the same panel
by time shows that every collapse comes after episode 4,500, and that over the first 3,000 episodes
the lead is already +0.115 of its +0.126. The synthesis should state that precisely.

## What Changes

- **The evaluation harness captures the thermal drive when asked.** `run_capture` gains the capture
  modality, so an evaluation on the thermal cell records the setpoint error `−|T − Tc|`, not the
  food drive.
- **A thermotaxis validation through the body**, on M.9's trained weights. Logbook 036's four bias
  statistics are graded against the thermal reference set (`data/thermotaxis/`), per arm, with the
  conventions fixed beforehand:
  - **Arms:** the wild type and the gap-only null, trained and as untrained floors.
  - **Floor comparison:** trained against floor, paired by seed.
  - **Beside:** the two wirings' difference.
- **A descriptive addendum to Logbook 092** that splits the lead into speed and stability, from the
  panel's committed logs. It adds no verdict.
- `thermal_body.py` gains a `thermotaxis` subcommand.

## Capabilities

**Modified**:

- `realworm-behavioural-validation`: the evaluation harness's behaviour capture takes a modality.

## Impact

- `scripts/analysis/body_kinematics_eval.py`: the capture modality.
- `scripts/analysis/thermal_body.py`: the `thermotaxis` subcommand. Tests.
- Logbook 093 (thermotaxis through the body), with an addendum to Logbook 092, and the tracker.
- No package code changes. The agent already records the thermal drive when the modality asks for
  it.
