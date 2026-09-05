# Spec: learning-rules

## ADDED Requirements

### Requirement: Minimal rate-based three-factor plasticity rule

The project SHALL provide a reward-modulated Hebbian learning rule satisfying the `LearningRule` Protocol, updating a topology's chemical synapses from an eligibility trace gated by a global neuromodulatory signal.

The update SHALL be `Δw = η · δ · E`, where `E` is the topology's eligibility trace, `η` is a configurable plasticity rate, and `δ` is a reward prediction error `r − b` against a running baseline `b`. The baseline SHALL be maintained as an exponential moving average of observed reward, so that the modulator encodes reward *surprise*; without it a predominantly one-signed reward stream drives weight change irrespective of behaviour.

The rule SHALL compute no gradients, own no optimiser, and require no value head. Its update SHALL execute entirely under `torch.no_grad()`.

The rule SHALL apply updates **once per environment step**, at the point the reward for that step becomes available, so the modulator is aligned with the eligibility it gates.

The alignment SHALL be **inclusive of the current step**: the trace is updated during the forward pass that selects the step's action, and the reward earned by that action gates the trace *including* that step's contribution. This is the intended semantics — the synapses that produced the action are the ones credited with its outcome — and it is stated explicitly because the alternative (gating only eligibility accrued strictly before the action) is equally implementable and would silently change what the rule credits.

The baseline SHALL persist across episode boundaries. It estimates the task's prevailing reward level, not one episode's; resetting it per episode would make every episode's opening steps register as surprising regardless of behaviour.

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

#### Scenario: The baseline survives episode boundaries

- **GIVEN** a rule that has observed a run of rewards
- **WHEN** the episode ends and per-episode state is reset
- **THEN** the baseline SHALL retain its value

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

**Synapse signs are deliberately unconstrained.** A Dale's-law constraint — forbidding a synapse from crossing between excitatory and inhibitory — would be the biologically correct restriction *if* synapse signs carried neurotransmitter identity. In this substrate they do not: initial chemical weights are drawn from a zero-mean distribution, so each synapse's sign is an arbitrary draw. Freezing it would preserve noise rather than biology, and would prevent the rule from correcting a synapse whose initial sign was simply wrong. Dale's law becomes enforceable once synapse signs are derived from neurotransmitter identity, which is a prerequisite this change does not create.

#### Scenario: Weights stay bounded under sustained drive

- **GIVEN** a sustained positive prediction error and a non-zero trace
- **WHEN** the rule steps many times
- **THEN** no chemical weight magnitude SHALL exceed the configured bound

#### Scenario: Decay pulls unreinforced weights down

- **GIVEN** a zero prediction error
- **WHEN** the rule steps repeatedly
- **THEN** chemical weight magnitudes SHALL be non-increasing

#### Scenario: A synapse may cross zero

- **GIVEN** a synapse whose accumulated updates drive it through zero
- **WHEN** the rule steps
- **THEN** the update SHALL NOT be clamped on account of the sign change

#### Scenario: Invalid stabilisation settings fail at load

- **WHEN** a configuration sets a plasticity rate, decay, or bound outside its valid range
- **THEN** loading SHALL fail with an error naming the offending field

### Requirement: Frozen controls apply to every rule

A configuration that freezes weight updates SHALL be honoured by every learning rule, not only by the gradient rule. A paired frozen control whose weights kept changing would be indistinguishable from its plastic counterpart in configuration and materially different in results, which is the failure a control exists to prevent.

Under a freeze the rule SHALL write nothing at all — no learning term, no decay, and no stabilising clamp. A clamp alone would still edit a weight that began outside the bound, silently changing the substrate the control is supposed to hold fixed.

Reporting SHALL continue while frozen, so the control remains comparable step-for-step with the arm it is paired against.

#### Scenario: A frozen arm does not learn

- **GIVEN** a configuration selecting a plasticity rule with updates frozen
- **WHEN** an episode is run
- **THEN** the trainable weights SHALL be bit-identical to their initial values

#### Scenario: The clamp does not fire under a freeze

- **GIVEN** a frozen arm whose magnitude bound is below its largest initial weight
- **WHEN** an episode is run
- **THEN** the weights SHALL still be bit-identical to their initial values

#### Scenario: A frozen arm still reports

- **GIVEN** a frozen arm
- **WHEN** updates are applied
- **THEN** telemetry SHALL be recorded for every step
- **AND** the reported weight change SHALL be zero

### Requirement: Plasticity telemetry

The rule SHALL report, per update, the prediction error, the running baseline, the mean absolute weight change, and the fraction of synapses at the magnitude bound, so that a saturating or inert rule is visible during a run.

#### Scenario: Health signals are reported every update

- **WHEN** the rule steps
- **THEN** its report SHALL carry the prediction error, baseline, mean absolute weight change, and saturated fraction

#### Scenario: Saturation is visible

- **GIVEN** a run driven until weights reach the magnitude bound
- **WHEN** the telemetry is inspected
- **THEN** the saturated fraction SHALL be non-zero
