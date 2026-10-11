## Why

B.3 puts serotonin and PDF on the connectome and reads their knockouts against roaming and dwelling.
D.1 (Logbook 087) validated the roaming/dwelling readout with the MLP only, so the connectome needs
its own positive control on the same cell first. That control is also the baseline B.3's modulator
arms are read against. The phase protocol requires both: a new component on a validated platform
gets its own positive control, and a baseline is re-established on the new operating point before
anything is ablated. This is **B.3 stage 0**, agreed with the user on 2026-10-11.

## What Changes

- **The connectome on D.1's lawn cell.** The connectome PPO brain (Emmons 2024, reversal, settling to
  depth 4) runs on D.1's cell unchanged. There are no modulators and no satiety input: satiety
  reaches the connectome through B.3's modulators.
- **An entropy calibration.** A pilot at 0.005 (D.1's MLP setting) and 0.05 (the connectome's
  point-worm default), on disjoint seeds. The registration uses the level where the connectome
  learns and dwells.
- **A registered positive control.** The connectome's on-lawn dwelling share minus its untrained
  floor's, paired by seed, on D.1's seeds 2001–2016. Beside it, the connectome is paired by seed with
  D.1's MLP.
- **The lawn analysis reads any learner,** with its learner and floor arms named.

## Capabilities

**Modified**:

- `realworm-behavioural-validation`: the lawn readout reads a named learner arm against its floor.

## Impact

- `scripts/campaigns/generate_lawn_configs.py`: the connectome's lawn arms at two entropy levels.
- `scripts/analysis/lawn_states.py`: named learner and floor arms, and the pairing with D.1's MLP.
- Tests, Logbook 089, tracker B.3 stage 0, roadmap.
