# Spec: connectome-ppo-brain

## ADDED Requirements

### Requirement: Plastic wiring arms share everything but the wiring

The panel's primary contrast — plastic wild-type against plastic degree-preserving rewired-null — is interpretable only if the two arms differ in wiring and in nothing else. The plastic rewired-null arm SHALL therefore be the plastic wild-type configuration with the wiring selector changed and no other key, and the two brains SHALL satisfy the following at any one run seed.

**Identical**: the motor readout; every sensory-projection gain; the action-noise parameters; the trace configuration; and every plasticity hyperparameter. These hold because the rewiring draws from a dedicated generator and preserves the chemical synapse count, so the brain's own initialisation stream is consumed identically in both arms, and because the readout is anatomical and consumes no randomness.

**Preserved**: the chemical edge count, every neuron's chemical in-degree and out-degree, every neuron's gap-junction degree, and therefore each weight's initialisation scale.

**Different**: the chemical edge mask, the placement of the initial chemical weights, and the gap-junction matrix.

The requirement SHALL NOT claim that per-neuron initial weight energy is preserved. It is not: the same draws land on different pre/post pairs, so realised per-neuron sums differ even though per-neuron scale is identical.

Under the plastic rule, every weight update and every non-zero eligibility-trace entry SHALL lie on the rewired edge set, so the arm is the null wiring in what learns and not only in what is wired. The rewired arm SHALL honour the freeze exactly as the wild-type arm does.

Rewiring SHALL pair by run seed: the rewiring seed defaults to the run seed, so seed *k* of each arm is a matched pair, and the rewiring SHALL be deterministic under it.

#### Scenario: The periphery is identical at one seed

- **WHEN** the plastic wild-type and plastic rewired-null configurations are loaded and their brains constructed at the same seed
- **THEN** the motor readout, every sensory-projection gain, and the action-noise parameters SHALL be bit-identical between the two brains

#### Scenario: Degrees and scale are preserved; wiring differs

- **WHEN** the two brains are inspected
- **THEN** the chemical edge count and every neuron's chemical in- and out-degree SHALL be equal
- **AND** the chemical edge mask, the initial chemical weights, and the gap-junction matrix SHALL differ

#### Scenario: Plasticity is confined to the null wiring

- **GIVEN** the plastic rewired-null brain trained for several steps
- **WHEN** its weights and traces are inspected
- **THEN** every changed chemical weight SHALL lie on the rewired edge mask
- **AND** every non-zero trace entry SHALL lie on the rewired edge mask

#### Scenario: The null arm learns, and freezes

- **GIVEN** the plastic rewired-null configuration
- **WHEN** an episode runs
- **THEN** the chemical weights SHALL have changed
- **AND** with updates frozen the same episode SHALL leave them bit-identical to initialisation

#### Scenario: Hyperparameters are shared

- **WHEN** the two configurations are loaded
- **THEN** every plasticity field and the trace configuration SHALL be equal
- **AND** the mask mode SHALL be strict on both, inherited rather than restated

#### Scenario: Pairing is by seed and deterministic

- **WHEN** the rewired arm is constructed twice at one seed and once at another
- **THEN** the two same-seed masks SHALL be identical
- **AND** the other-seed mask SHALL differ

#### Scenario: The config is a one-key delta

- **WHEN** the plastic rewired-null configuration is compared with the plastic wild-type configuration
- **THEN** the only difference SHALL be the wiring selector
