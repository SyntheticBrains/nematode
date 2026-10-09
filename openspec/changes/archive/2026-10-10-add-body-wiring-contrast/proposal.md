## Why

C.1e is Phase 8b's central reading: does the wild-type wiring help a connectome forage **through the
body**, against nulls that differ from it only in the chemical wiring? C.1d's MLP control passed, and
the body-drive gain and a dimension-matched entropy bonus (`add-body-drive-gain`) let the connectome
learn the 500-step cell on every probe seed. Nothing committed yet measures a wiring effect on this
cell, so its minimum cannot be taken from data; the protocol requires it to be.

## What Changes

- **A frozen-wiring PPO learner.** A `freeze_wiring` option keeps the chemical weights at their
  initial draw while PPO trains the sensor gains, the body-drive gains and the policy's noise. It is the
  body's analogue of the reading learner: the wiring is read through the anatomy, never written. The
  e-prop reading learner cannot run through the body, whose readout is fixed anatomy.
- **The boundary-preserving null under the body.** Its motor boundary becomes every cell the body drive
  reads: the cells with junctions onto the muscles, and the four motor classes the direction channel
  reads.
- **A registered pilot on this cell** sets the panel's minimum and sizing: the wild type against the
  chemical-only null under both learners, learning and frozen, on disjoint seeds.
- **The panel**: four wirings under both learners, with their frozen floors and the MLP beside,
  registered from the pilot's committed effect.
- Configs through a generator, and an analysis module for the pilot and the panel.

## Capabilities

**Modified**: `connectome-ppo-brain`, with two added requirements (the frozen-wiring learner; the
boundary null's body-drive boundary).

## Impact

- `brain/arch/connectome_ppo.py`: `freeze_wiring`; the body-drive boundary.
- `scripts/campaigns/generate_body_wiring_configs.py`, `scripts/analysis/body_wiring.py`: new.
- Configs, tests, Logbooks 084 (pilot) and 085 (panel), tracker C.1e, roadmap.
