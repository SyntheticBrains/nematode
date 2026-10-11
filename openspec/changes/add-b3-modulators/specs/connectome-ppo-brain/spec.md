## ADDED Requirements

### Requirement: The neuromodulator field

The connectome PPO brain SHALL offer an opt-in modulator field: serotonin released by NSM and acting on
AIY, ASI and RIF with an inhibitory current, and PDF released by AVB, PVP and SIAV and acting on AIY,
RIM and RIA with an excitatory current. Each concentration is carried across steps, decaying with a
60 s timescale toward its release cells' activity above rest, and is reset at each episode's start.
NSM SHALL receive a held current proportional to the food ingested on the worm's last move, as a
fraction of the largest one move can yield, and an inhibitory current proportional to the PDF
concentration. With the field off, the brain SHALL be
byte-identical to before, and checkpoints saved before the field existed SHALL still load.

#### Scenario: Off is unchanged

- **WHEN** a connectome brain is built without the modulator field
- **THEN** its forward pass, parameters and checkpoint SHALL be identical to before

#### Scenario: Serotonin inhibits its receptor cells

- **WHEN** the serotonin concentration is positive
- **THEN** AIY, ASI and RIF SHALL settle lower than with no serotonin, all else equal

#### Scenario: Ingestion reaches serotonin through settling

- **WHEN** the worm ate on its last move
- **THEN** NSM's activity at the end of settling SHALL be higher than with nothing eaten, all else
  equal

#### Scenario: PDF inhibits NSM

- **WHEN** the PDF concentration is positive
- **THEN** NSM SHALL settle lower than with no PDF, all else equal

#### Scenario: An old checkpoint still loads

- **WHEN** a connectome checkpoint saved without the field is loaded into a brain without the field
- **THEN** it SHALL load and act identically

#### Scenario: A concentration persists and decays

- **WHEN** release stops after a burst
- **THEN** the concentration SHALL decay by a factor of exp(−Δt/60 s) per step

#### Scenario: Replay sees the rollout's concentrations

- **WHEN** PPO replays a buffered step
- **THEN** the replayed action distribution SHALL equal the rollout's

### Requirement: Modulator knockouts

The brain SHALL accept a *tph-1* knockout (no serotonin) or a *pdfr-1* knockout (no PDFR-1 effect,
including PDF's inhibition of NSM), for evaluation of a trained network or from the start of
training.

#### Scenario: A tph-1 knockout removes serotonin

- **WHEN** a trained network is evaluated with the `tph1` knockout
- **THEN** the serotonin concentration SHALL stay at 0 and no MOD-1 current SHALL flow

#### Scenario: A pdfr-1 knockout removes every PDFR-1 effect

- **WHEN** a trained network is evaluated with the `pdfr1` knockout
- **THEN** no PDFR-1 current SHALL flow on AIY, RIM or RIA, and PDF SHALL not inhibit NSM
