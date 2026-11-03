## ADDED Requirements

### Requirement: C.2's compact encoder behind a byte-identical default

`ConnectomePPOBrainConfig` SHALL accept `encoder: compact`, default `none`. With it unset, every tensor, draw and action SHALL be unchanged. With it set, the brain SHALL be a function of 196 named parameters:

- the wiring: excitatory, inhibitory and gap scales, and a signed gain for each of 4 presynaptic transmitter groups;
- the sensory inputs: 36 class-tied gains;
- the motor readout: 20 weights and 5 biases;
- the 5 action log-SDs;
- the neuromodulator layer's 104 signed and 17 fixed-sign strengths;
- the monoamine and neuropeptide timescales.

The brain SHALL refuse the encoder:

- unless `dynamics` is settling and `sensing_mode` is klinotaxis;
- unless the data layer is used with the satiety link;
- unless `synapse_signs` is `per_connection`, `gap_normalisation` is `strength` with additive coupling, and `weight_prior` is unset;
- unless `modulator_step_seconds` equals the configured step.

#### Scenario: The encoder is off

- **WHEN** a connectome brain is built without `encoder`
- **THEN** its actions on fixed senses SHALL be byte-identical to those before the encoder existed

#### Scenario: A 5-s modulator step is refused

- **WHEN** the encoder is set with `modulator_step_seconds` 5.0 on a 1-s worm
- **THEN** the brain SHALL refuse to build

### Requirement: The wiring is fixed from data

With the encoder on:

- a chemical edge's weight SHALL be its scale × its grounded sign × its synapse count, normalised by the root sum of squares of the counts onto its postsynaptic neuron;
- an edge without a grounded sign SHALL take its presynaptic transmitter group's signed gain in place of the scale and sign;
- a gap pair's weight SHALL be its count divided by √(s_i·s_j), s a neuron's summed pair counts, × the gap scale.

No weight SHALL depend on the brain's seed. On a rewired null:

- each edge SHALL carry its own count and its origin's sign;
- every normalisation SHALL be recomputed on the null's counts;
- the parameters' names SHALL be the wild type's.

#### Scenario: A null is encoded with the wild type's parameters

- **WHEN** an encoded brain is built on any existing rewired null, or under the receptor-placement null
- **THEN** it SHALL build, with the same parameter names as the wild type, and its mutant encodings SHALL build on it

#### Scenario: Two seeds, one vector

- **WHEN** two encoded brains are built with different seeds and the same vector is written into both
- **THEN** their chemical and gap weights SHALL be equal

### Requirement: Class-tied sensory inputs with an antisymmetric lateral gain

With the encoder on, the 11 amphid chemosensory classes (AWC, AWA, ASE, AWB, ASK, ADF, ASI, ASJ, ASG, ADL, ASH) SHALL each receive:

- concentration and dC/dt through one gain each, shared by the class's left and right neurons;
- the lateral difference through one gain applied with opposite signs to its right and left neurons.

Food contact SHALL reach only CEP, ADE and PDE, through one gain per class.

#### Scenario: Lateral drives the two sides oppositely

- **WHEN** only the lateral difference is non-zero
- **THEN** each encoded class's right and left neurons SHALL receive equal and opposite inputs

#### Scenario: Food contact reaches only the dopaminergic mechanosensors

- **WHEN** only food contact is non-zero
- **THEN** only CEP, ADE and PDE neurons SHALL receive input

### Requirement: A unit box and a vector interface

The encoder SHALL expose its parameters' names in a fixed order and each one's range. It SHALL write a unit-box vector into a brain in place and read it back, and `BatchedConnectome` SHALL act on the written values without a rebuild. The box's centre SHALL be the registered start for C.2's fits. Under PPO the policy's learnable parameters SHALL be exactly the encoder's, with the PPO rule's critic unchanged.

`encoder_modulator_tying` SHALL choose between:

- `per_couple`, 196 parameters;
- `per_ligand`, one signed gain per fitted ligand and one strength per fixed-sign ligand: 132 parameters.

#### Scenario: A vector round-trips

- **WHEN** a unit-box vector is written and read back
- **THEN** the read vector SHALL equal the written one

#### Scenario: A reused brain equals a fresh one

- **WHEN** one brain has a sequence of vectors written into it, and a fresh brain is built for the last
- **THEN** the two SHALL act identically on the same senses and generators

#### Scenario: Per-ligand tying shares a ligand's strengths

- **WHEN** `per_ligand` tying is set
- **THEN** every couple of one fitted ligand SHALL carry the same signed strength, and the parameter count SHALL be 132

#### Scenario: The batched controller sees a write

- **WHEN** a vector is written into a brain after its `BatchedConnectome` is built
- **THEN** the controller's next actions SHALL be those of a brain built with that vector

### Requirement: Two tied timescales

With the encoder on, one timescale SHALL set the decay of the four monoamine pools and another the 51 neuropeptide pools, each decaying by exp(−step / τ).

#### Scenario: The monoamine timescale changes only monoamines

- **WHEN** τ_mono is written and τ_pep is unchanged
- **THEN** only the dopamine, octopamine, serotonin and tyramine pools' decay SHALL change

### Requirement: The mutant encodings bind the registered table

With the encoder on:

- tph-1 and pdfr-1 SHALL be built with their knockout genes and the wild type's vector;
- tax-4 SHALL zero the sensory gains of AWC, AWB, ASE, ASG, ASI, ASJ and ASK, and leave every other parameter as written.

#### Scenario: tax-4 keeps the non-tax-4 classes

- **WHEN** C.2's tax-4 encoding is applied
- **THEN** the 21 gains of the seven tax-4 classes SHALL be 0, and AWA's, ADF's, ADL's and ASH's gains and food contact's SHALL be unchanged

### Requirement: The encoder is checked before F.0b-ii registers

Before F.0b-ii's registration, on discarded seeds:

- a cost pilot SHALL time a generation on D_fit's assays and on all four;
- a feasibility check SHALL show whether a hand-set vector carries a post-removal decay of 6–12 min without reading López-Cruz's data;
- a readiness probe SHALL fit up to 3 restarts to v3's band centres on D_fit's families, stopping once 2 pass D_fit's families in 3 of 5 populations. Each plateau SHALL wait until every median in the last 40 generations is below 100. Each answer SHALL be graded with presence on 5 populations on all five families and the three mutants;
- the restart-to-restart spread of the held-out families and of the mutant differences SHALL be recorded against W's minimum;
- the cost pilot SHALL be profiled, and SHALL time a whole generation's settling step on CPU and on MPS.

Each check's result SHALL be recorded.

#### Scenario: A restart with half its candidates undefined does not stop

- **WHEN** a probe restart's medians hold near 500 for 40 generations and σ ≤ σ₀
- **THEN** the plateau SHALL NOT stop it
