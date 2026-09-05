# Spec: learning-rules

## ADDED Requirements

### Requirement: Minimal rate-based three-factor plasticity rule

The project SHALL provide a reward-modulated Hebbian learning rule satisfying the `LearningRule` Protocol, updating a topology's chemical synapses from an eligibility trace gated by a global neuromodulatory signal.

The update SHALL be `Δw = η · δ · E`, where `E` is the topology's eligibility trace, `η` is a configurable plasticity rate, and `δ` is a reward prediction error `r − b` against a running baseline `b`. The baseline SHALL be maintained as an exponential moving average of observed reward, so that the modulator encodes reward *surprise*; without it a predominantly one-signed reward stream drives weight change irrespective of behaviour.

The rule SHALL compute no gradients, own no optimiser, and require no value head. Its update SHALL execute entirely under `torch.no_grad()`.

The rule SHALL apply updates **once per environment step**, at the point the reward for that step becomes available, so the modulator is aligned with the eligibility it gates.

The rule SHALL update **only** the chemical synaptic weights over which the trace is defined. Sensory-projection gains, the motor readout, and action-noise parameters SHALL be left at their initial values.

Every update SHALL be projected through the topology's mask seam, so no update creates support outside the topology's edge set.

#### Scenario: Update follows the three-factor product

- **WHEN** the rule steps with a known trace, reward, and baseline
- **THEN** the change in chemical weights SHALL equal the plasticity rate times the reward prediction error times the trace, before stabilisation terms
- **AND** a zero prediction error SHALL produce no weight change from the Hebbian term
- **AND** a zero trace SHALL produce no weight change from the Hebbian term

#### Scenario: The modulator is a prediction error, not a reward

- **GIVEN** a constant non-zero reward stream
- **WHEN** the rule has observed enough steps for the baseline to converge
- **THEN** the magnitude of the weight change per step SHALL tend toward zero
- **AND** an unexpected reward SHALL produce a larger weight change than an expected one of the same magnitude

#### Scenario: No gradient machinery is engaged

- **WHEN** the rule steps
- **THEN** no chemical-weight tensor SHALL acquire a gradient
- **AND** the update SHALL succeed with autograd globally disabled

#### Scenario: Only chemical synapses change

- **WHEN** the rule steps
- **THEN** the chemical weights MAY change
- **AND** sensory gains, motor readout, and action-noise parameters SHALL be bit-identical to their pre-step values

#### Scenario: Updates respect the topology mask

- **WHEN** the rule steps on a topology with a restricted edge set
- **THEN** every weight outside that edge set SHALL remain zero

### Requirement: Bounded plasticity

An unbounded Hebbian rule diverges. The rule SHALL therefore provide configurable stabilisation and SHALL make saturation observable rather than silent.

A weight-decay term SHALL be applied alongside the Hebbian term, and updated weights SHALL be clamped to a configurable magnitude bound. Both SHALL be configurable, and both SHALL be validated at load time so an unusable setting fails before a run rather than during one.

The rule SHALL preserve each synapse's sign: an update SHALL NOT change an existing synapse from excitatory to inhibitory or the reverse. A chemical synapse's sign follows from its neurotransmitter rather than from experience, so sign inversion would model a transition the animal cannot make.

#### Scenario: Weights stay bounded under sustained drive

- **GIVEN** a sustained positive prediction error and a non-zero trace
- **WHEN** the rule steps many times
- **THEN** no chemical weight magnitude SHALL exceed the configured bound

#### Scenario: Decay pulls unreinforced weights down

- **GIVEN** a zero prediction error
- **WHEN** the rule steps repeatedly
- **THEN** chemical weight magnitudes SHALL be non-increasing

#### Scenario: Synapse signs are preserved

- **GIVEN** a synapse with a known sign and an update that would invert it
- **WHEN** the rule steps
- **THEN** the synapse SHALL retain its original sign
- **AND** the occurrence SHALL be counted in the rule's telemetry

#### Scenario: Invalid stabilisation settings fail at load

- **WHEN** a configuration sets a plasticity rate, decay, or bound outside its valid range
- **THEN** loading SHALL fail with an error naming the offending field

### Requirement: Plasticity telemetry

The rule SHALL report, per update, the prediction error, the running baseline, the mean absolute weight change, the fraction of synapses at the magnitude bound, and the number of sign-clamp events, so that a saturating or inert rule is visible during a run.

#### Scenario: Health signals are reported every update

- **WHEN** the rule steps
- **THEN** its report SHALL carry the prediction error, baseline, mean absolute weight change, saturated fraction, and sign-clamp count

#### Scenario: Saturation is visible

- **GIVEN** a run driven until weights reach the magnitude bound
- **WHEN** the telemetry is inspected
- **THEN** the saturated fraction SHALL be non-zero
