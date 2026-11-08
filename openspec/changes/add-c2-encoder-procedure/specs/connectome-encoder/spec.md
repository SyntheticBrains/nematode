## MODIFIED Requirements

### Requirement: A unit box and a vector interface

The encoder SHALL expose its parameters' names in a fixed order and each one's range. It SHALL write a unit-box vector into a brain in place and read it back, and `BatchedConnectome` SHALL act on the written values without a rebuild. The box's centre SHALL be the registered start for C.2's fits. Under PPO the policy's learnable parameters SHALL be exactly the encoder's, its two timescales held (three with `encoder_speed_noise: ou`, the speed noise's among them), with the PPO rule's critic unchanged. The pool centres, with `encoder_pool_centres: true`, SHALL be among the learnable parameters.

`encoder_modulator_tying` SHALL choose between:

- `per_couple`, 196 parameters;
- `per_ligand`, one signed gain per fitted ligand and one strength per fixed-sign ligand: 132 parameters. A couple signed through `modulator_receptor_signs` is a fixed-sign group of its own per ligand and receptor, which leaves the count at 132 for PDFR-1 and SER-5.

With `encoder_speed_noise: ou` each count is one more: 197 and 133. With `encoder_pool_centres: true` each is four more: 200 and 136, or 201 and 137 with both.

#### Scenario: A vector round-trips

- **WHEN** a unit-box vector is written and read back
- **THEN** the read vector SHALL equal the written one to float32 precision (1e-6)

#### Scenario: A reused brain equals a fresh one

- **WHEN** one brain has a sequence of vectors written into it, and a fresh brain is built for the last
- **THEN** the two SHALL act identically on the same senses and generators

#### Scenario: Per-ligand tying shares a ligand's strengths

- **WHEN** `per_ligand` tying is set
- **THEN** every couple of one fitted ligand SHALL carry the same signed strength, and the parameter count SHALL be 132

#### Scenario: The batched controller sees a write

- **WHEN** a vector is written into a brain after its `BatchedConnectome` is built
- **THEN** the controller's next actions SHALL be those of a brain built with that vector

#### Scenario: The counts with the new options

- **WHEN** an encoded brain is built under per-ligand tying with `encoder_pool_centres: true`, with and without `encoder_speed_noise: ou`
- **THEN** it SHALL have 136 and 137 parameters, the centres last

## ADDED Requirements

### Requirement: A soft-capped readout and pool centres behind byte-identical defaults

`ConnectomePPOBrainConfig` SHALL accept `encoder_readout`, `linear` by default or `soft_cap`, and `encoder_pool_centres`, false by default. With both at their defaults, every output SHALL be byte-identical to the brain before the options.

- **With `soft_cap`,** each output's pre-squash value SHALL be 2·tanh(x/2) of the readout's value.
- **With pool centres,** the readout SHALL read b + Σ_j w_j (pool_j − c_j). The four centres are entries linear on [−1, 1] named `readout.centre.VB`, `.DB`, `.VA` and `.DA`, and SHALL be learnable under PPO.
- **Both apply in the encoded forward** that the per-step runner, the batched evaluator and PPO share.

#### Scenario: Off is byte-identical

- **WHEN** an encoded brain is built with `encoder_readout: linear` and `encoder_pool_centres: false`
- **THEN** its actions, log-probs and encoder vector SHALL equal those of the brain before the options

#### Scenario: The cap bounds each output

- **WHEN** the readout's value for an output is x
- **THEN** that output's pre-squash value SHALL be 2·tanh(x/2), within (−2, 2)

#### Scenario: Centres at zero leave the readout unchanged

- **WHEN** pool centres are on and every centre is 0
- **THEN** the readout SHALL equal the readout without centres on the same pools

### Requirement: A graded box and an inhibition floor

`encoder_box` SHALL be `wide` by default or `graded`.

- **`graded` SHALL hold** the excitatory, inhibitory and gap wiring scales on [0, 0.75] and every sensory gain on [−2, 2]. Every other entry's range SHALL be unchanged.
- **`encoder_inhibition_floor: f`,** None by default, SHALL set the inhibitory scale's range to [f, high]. A floor outside \[0, high) SHALL be refused.
- **Neither binds PPO,** which trains the raw tensors.

#### Scenario: The graded box's centre

- **WHEN** an encoded brain is built with `encoder_box: graded` and its box centre written
- **THEN** each wiring scale SHALL be 0.375 and each sensory gain 0

#### Scenario: The floor bounds the inhibitory scale

- **WHEN** `encoder_inhibition_floor: 0.09` is set and the inhibitory entry's unit is 0
- **THEN** the inhibitory scale SHALL be 0.09

#### Scenario: A floor outside the box is refused

- **WHEN** `encoder_inhibition_floor` is negative, or at or above its box's wiring maximum
- **THEN** the config SHALL be refused at validation

### Requirement: The vector's order keeps stored vectors a prefix

The encoder's entries SHALL be ordered:

1. the base entries;
2. `timescale.speed_noise`, with `encoder_speed_noise: ou`;
3. the four pool centres, with `encoder_pool_centres: true`.

The count SHALL be the base's (196, or 132 under per-ligand tying), plus 1 for the speed noise, plus 4 for the centres.

#### Scenario: Prefix property

- **WHEN** a brain with `encoder_pool_centres: true` is written with a vector extended by four centre entries at their unit centre 0.5
- **THEN** it SHALL act as a brain with the option off written with the shorter vector

### Requirement: The probe procedure's answer reproduces in the package

`c2_encoder_probe.package_vector` SHALL map a vector in the probe scripts' layout into the package's encoding, entry by entry by name. The package's encoded forward SHALL compute what the probe's patched forward computed on the answer in `supporting/129-c2-wild-type-and-mutants/probes/grade_nudged.json`. The answer's five-population grade SHALL be re-run and reported as description, not as a gate: the probe's pass sits at the threshold, and float32-level differences can move a population across it.

#### Scenario: The forwards agree

- **WHEN** the converted answer is written into a package brain with the procedure's options, and the probe's answer into the probe's patched brain, and both run the recorded states of one reference episode on the small lawn and on food removal
- **THEN** every output SHALL agree within 1e-5
- **AND** the primitive chosen at every step of that episode SHALL be the same

#### Scenario: The grade is reported

- **WHEN** the converted answer is graded on 795,000,000 + 200·g + i (g < 5) with presence
- **THEN** its record SHALL report each population's wild-type result, its D and its mutants beside the probe's

### Requirement: An optional forward observer

`ConnectomeTopology` SHALL accept an optional observer, `None` by default, called after each encoded settling with that step's outputs and its settled hidden state. With no observer every output SHALL be byte-identical. The observer SHALL NOT change what the forward returns. It is how a fit records outputs and motor saturation for its guards, without replacing the forward.

#### Scenario: The observer sees each step and changes nothing

- **WHEN** an observer that records its arguments is attached and an episode runs
- **THEN** it SHALL be called once per settling with the step's outputs and hidden state
- **AND** the episode's actions SHALL equal those without the observer
