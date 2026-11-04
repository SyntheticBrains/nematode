# connectome-encoder Specification

## Purpose

The compact encoder makes the connectome brain a function of a few hundred named parameters, its wiring fixed from data, so a controller can be fitted to real-worm behaviour and its nulls fitted at equal budget with the same parameters.

## Requirements

### Requirement: C.2's compact encoder behind a byte-identical default

`ConnectomePPOBrainConfig` SHALL accept `encoder: compact`, default `none`. With it unset, every tensor, draw and action SHALL be unchanged. With it set, the brain SHALL be a function of 196 named parameters (197 with `encoder_speed_noise: ou`, which appends the speed noise's timescale):

- the wiring: excitatory, inhibitory and gap scales, and a signed gain for each of 4 presynaptic transmitter groups;
- the sensory inputs: 36 class-tied gains;
- the motor readout: 20 weights and 5 biases;
- the 5 action log-SDs;
- the neuromodulator layer's 104 signed and 17 fixed-sign strengths;
- the monoamine and neuropeptide timescales.

The brain SHALL refuse the encoder:

- unless `dynamics` is settling, `sensing_mode` is klinotaxis and the action space is the primitives';
- unless the full data layer is used with the satiety link and without `modulator_timescales_s`;
- unless `synapse_signs` is `per_connection`, `gap_normalisation` is `strength` with additive coupling, and `weight_prior` is unset;
- while any weight setting it replaces (the chemical draw and scales, the gap weighting and scale, plastic gaps, the readout width, the sensory projections, traces and node noise) is away from its default;
- unless `modulator_step_seconds` is at most 1 s; and the loader SHALL refuse it unless the converted `modulator_step_seconds` equals the configured step.

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

The encoder SHALL expose its parameters' names in a fixed order and each one's range. It SHALL write a unit-box vector into a brain in place and read it back, and `BatchedConnectome` SHALL act on the written values without a rebuild. The box's centre SHALL be the registered start for C.2's fits. Under PPO the policy's learnable parameters SHALL be exactly the encoder's, its two timescales held (three with `encoder_speed_noise: ou`, the speed noise's among them), with the PPO rule's critic unchanged.

`encoder_modulator_tying` SHALL choose between:

- `per_couple`, 196 parameters;
- `per_ligand`, one signed gain per fitted ligand and one strength per fixed-sign ligand: 132 parameters.

With `encoder_speed_noise: ou` each count is one more: 197 and 133.

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
- a readiness probe SHALL establish whether sep-CMA from the box centre reaches the wild type on D_fit's families, each fit's answer re-simulated at checkpoints on fresh animals, and SHALL record what stops it. Its plateau rule SHALL wait until every median in the last 40 generations is below 100;
- *(amended 2026-11-04)* the readiness restarts (up to 3, each answer graded with presence on 5 populations on all five families and the three mutants, with their restart-to-restart spread recorded against W's minimum) SHALL run under the fitting procedure the probe establishes, in the change that registers it;
- the cost pilot SHALL be profiled, and SHALL time a whole generation's settling step on CPU and on MPS;
- a W sizing record SHALL state what is measured and what is not.

Each check's result SHALL be recorded.

#### Scenario: A restart with half its candidates undefined does not stop

- **WHEN** a probe restart's medians hold near 500 for 40 generations and σ ≤ σ₀
- **THEN** the plateau SHALL NOT stop it

### Requirement: Slow speed noise behind a byte-identical default

`ConnectomePPOBrainConfig` SHALL accept `encoder_speed_noise`, `none` by default or `ou`. With `none`
every output SHALL be byte-identical to the brain before the option, and the encoder SHALL keep its 196
parameters (132 under per-ligand tying). With `ou` the speed output's standardised noise SHALL follow
x_t = ρ·x\_{t−1} + √(1−ρ²)·ε_t, ρ = exp(−Δ/τ), ε_t the step's own draw and x_0 = ε_0 at each episode's
start; the other outputs SHALL keep independent noise; and the encoder SHALL append one parameter,
`timescale.speed_noise`, log-scaled over 0.25–300 s, after every existing one.

#### Scenario: Off is byte-identical

- **WHEN** a connectome brain is built with `encoder_speed_noise: none`
- **THEN** its actions, log-probs and encoder vector SHALL equal those of the brain before the option

#### Scenario: On appends one parameter

- **WHEN** an encoded brain is built with `encoder_speed_noise: ou`
- **THEN** it SHALL have 197 parameters (133 under per-ligand tying), the last `timescale.speed_noise`
- **AND** a 196-entry vector SHALL be the first 196 entries of a 197-entry one, writing the same tensors

#### Scenario: The process is a stationary OU on speed alone

- **WHEN** the slow-noise sampler runs for 20,000 steps at ρ = 0.9
- **THEN** the speed noise's variance SHALL be 1 and its lag-1 autocorrelation 0.9, each within its
  sampling error
- **AND** the other four outputs, and every output on an episode's first step, SHALL equal the
  independent sampler's on the same draws

#### Scenario: PPO evaluates the exact conditional

- **WHEN** a rollout under slow noise is re-evaluated before any update
- **THEN** every log-prob SHALL equal the one stored at sampling, so every ratio is 1
- **AND** after the parameters change, the speed dimension's log-density SHALL be that of
  N(m(s_t) + ρ·(u\_{t−1} − m(s\_{t−1})), σ²(1−ρ²)), with both means recomputed under the new parameters

#### Scenario: τ is held under PPO

- **WHEN** the encoded brain with slow noise is trained by PPO
- **THEN** `timescale.speed_noise` SHALL not be among its learnable parameters

#### Scenario: Unsupported settings are refused

- **WHEN** `encoder_speed_noise: ou` is set without `encoder: compact`, without continuous primitives,
  with leaky dynamics or with activity traces, in a validated config or one derived by `model_copy`
- **THEN** the brain SHALL refuse it, naming the setting
- **AND** a weight file recorded under the other speed-noise mode SHALL be refused on load

### Requirement: The slow noise is shown to reach the window statistics before it is relied on

The positive control SHALL grade an answer with independent noise and with slow noise at τ ∈ {0.25, 2,
8.7, 30, 100, 300} s on the same 200 animals, and SHALL be met before the readiness restarts rely on
slow noise. It is read only on an answer whose speed output responds: one whose speed output's
pre-squash median over a reference episode lies within ±2, where the squash is not flat. On the
stage-2 answer, whose speed output's median is −3.9, it was run as first registered and failed; that
reading SHALL be reported as failed, and the control SHALL be re-run on the first fit whose speed
output responds.

#### Scenario: The positive control on a responsive answer

- **WHEN** the sweep is graded on an answer whose speed output responds
- **THEN** the roaming speed's 90th percentile SHALL rise by at least 3 band scales above its value at
  τ = 0.25 at some τ, with the dwelling speed's 90th percentile at that τ reported beside it
- **AND** at τ = 0.25 every kinematics statistic SHALL equal the independent model's, on the same
  seeds, within twice its band scale

#### Scenario: An answer whose speed is pinned cannot carry it

- **WHEN** an answer's speed output's pre-squash median lies outside ±2
- **THEN** the control's reading on it SHALL be reported as failed, not passed and not re-read, with
  the speed output's median beside it
