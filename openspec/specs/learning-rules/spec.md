# learning-rules Specification

## Purpose

This capability specifies the **update mechanisms** that change a brain topology's weights from experience — the rules themselves, separated from the substrates they act on.

A rule owns whatever machinery its own update needs: optimiser state, value heads, advantage estimators, hyperparameters, gradient clippers. The paired topology is pure structure. That separation is what lets one substrate be trained by different rules and compared, which is the whole basis of the rule × wiring comparisons this project is built around: if the rule and the substrate were entangled in one class, a difference between two arms could always be blamed on something other than the dimension under test.

Two rules live here. The clipped-surrogate PPO update is the gradient baseline that Phase 6's results were measured under. The reward-modulated three-factor rule is the biologically-motivated alternative: pre- and post-synaptic activity via a decaying eligibility trace, gated by a global neuromodulatory signal, with no backward pass, no weight transport, and no per-synapse error signal. That locality is the property a gradient method cannot claim, and the reason this rule family — rather than PPO — is the instrument for asking whether a real connectome's wiring is legible to a learner the animal could plausibly host.

Not to be confused with `quantumnematode.plasticity`, which is the quantum-plasticity **evaluation protocol** (sequential multi-objective and catastrophic-forgetting metrics), not a learning rule.

## Requirements

### Requirement: Minimal rate-based three-factor plasticity rule

The project SHALL provide a reward-modulated Hebbian learning rule satisfying the `LearningRule` Protocol, updating a topology's **plastic weights** — those over which it maintains eligibility traces — from those traces gated by a global neuromodulatory signal. The rule SHALL read the substrate through the plastic-topology seam and SHALL NOT name any substrate's attributes directly; the connectome's chemical synapses and the MLP's layer weights are both plastic weights to it.

The update SHALL be `Δw = η · δ · E` for each plastic tensor `w` and its aligned trace `E`, where `η` is a configurable plasticity rate and `δ` is a reward prediction error `r − b` against a running baseline `b`. The baseline SHALL be maintained as an exponential moving average of observed reward, so that the modulator encodes reward *surprise*; without it a predominantly one-signed reward stream drives weight change irrespective of behaviour.

The rule SHALL compute no gradients, own no optimiser, and require no value head. Its update SHALL execute entirely under `torch.no_grad()`.

The rule SHALL apply updates **once per environment step**, at the point the reward for that step becomes available, so the modulator is aligned with the eligibility it gates.

The alignment SHALL be **inclusive of the current step**: the trace is updated during the forward pass that selects the step's action, and the reward earned by that action gates the trace *including* that step's contribution. This is the intended semantics — the synapses that produced the action are the ones credited with its outcome — and it is stated explicitly because the alternative (gating only eligibility accrued strictly before the action) is equally implementable and would silently change what the rule credits.

The baseline SHALL persist across episode boundaries. It estimates the task's prevailing reward level, not one episode's; resetting it per episode would make every episode's opening steps register as surprising regardless of behaviour.

The rule SHALL update **only** the plastic weights the seam exposes. Every other parameter of the substrate — on the connectome its sensory gains, motor readout, and action-noise parameters; on the MLP its biases, action-noise parameters, and any feature-gating weights — SHALL be left at its initial value.

Every update SHALL be projected through the topology's mask seam, so no update creates support outside the topology's edge set.

*(The scenario titled "Only chemical synapses change" below keeps its historical name from when the connectome was the only substrate; it now specifies that only the seam's plastic weights change on any substrate.)*

#### Scenario: Update follows the three-factor product

- **WHEN** the rule steps with a known trace, reward, and baseline
- **THEN** the change in each plastic tensor SHALL equal the plasticity rate times the reward prediction error times its trace, before stabilisation terms
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
- **THEN** no plastic tensor SHALL acquire a gradient
- **AND** the update SHALL succeed with autograd globally disabled

#### Scenario: Only chemical synapses change

- **WHEN** the rule steps on either substrate
- **THEN** the plastic weights MAY change
- **AND** every non-plastic parameter SHALL be bit-identical to its pre-step value — on the connectome the sensory gains, motor readout, and action-noise parameters; on the MLP the biases and action-noise parameters

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

The reported weight change SHALL be the **effective** change — measured against the weights as they stood before the update, after any stabilising clamp — and not the change the update proposed. A rule whose synapses have reached the bound writes nothing while still proposing a large update, so reporting the proposal would show a healthy learning signal for a rule that has become a constant function, defeating this requirement's purpose at exactly the point it is needed.

#### Scenario: Health signals are reported every update

- **WHEN** the rule steps
- **THEN** its report SHALL carry the prediction error, baseline, mean absolute weight change, and saturated fraction

#### Scenario: Saturation is visible

- **GIVEN** a run driven until weights reach the magnitude bound
- **WHEN** the telemetry is inspected
- **THEN** the saturated fraction SHALL be non-zero
- **AND** the reported weight change SHALL be zero, because the clamp discarded the update

### Requirement: Unmodulated Hebbian mode

The plasticity rule SHALL support an **unmodulated** mode in which the neuromodulatory third factor is not applied: the weight change is the plasticity rate times the eligibility trace, with the stabilisation terms unchanged.

This mode exists as the ablation that isolates the rule's central claim. An arm that beats an untrained network has learned something; an arm that beats unmodulated Hebbian has learned something **from reward**. Without the comparison, an advantage attributable to correlation structure alone is indistinguishable from one attributable to reward-driven learning.

The mode SHALL differ from the modulated rule in the modulator alone. Eligibility accumulation, masking, decay, clamping, and reporting SHALL be identical, so that a difference between the two arms is attributable to the third factor and not to an incidentally different code path.

The reward prediction error and its baseline SHALL still be computed and reported in this mode, even though the update does not apply them. Both arms then record what the reward stream was doing, and only one records having used it, making the ablation visible in telemetry rather than inferable only from configuration.

#### Scenario: The update omits the modulator

- **WHEN** the rule steps in unmodulated mode with a known trace
- **THEN** the weight change SHALL equal the plasticity rate times the trace, before stabilisation terms
- **AND** it SHALL NOT depend on the reward

#### Scenario: Reward changes nothing in this mode

- **GIVEN** two rules in unmodulated mode with identical traces and identical initial weights
- **WHEN** one is stepped with a large reward and the other with a small one
- **THEN** their resulting weights SHALL be identical

#### Scenario: The reward stream is still observed

- **WHEN** the rule steps in unmodulated mode
- **THEN** its report SHALL carry the prediction error and baseline the reward stream implies
- **AND** those values SHALL match what the modulated rule would have reported for the same rewards

#### Scenario: Only the modulator differs from the modulated rule

- **GIVEN** a modulated and an unmodulated rule over identical topologies with identical traces
- **WHEN** each is stepped once with a reward whose prediction error is exactly one
- **THEN** their weight changes SHALL be identical

#### Scenario: Stabilisation still applies

- **GIVEN** an unmodulated rule driven until weights reach the magnitude bound
- **WHEN** it steps again
- **THEN** no weight magnitude SHALL exceed the bound
- **AND** the reported weight change SHALL be the effective change, which is zero once saturated

### Requirement: Plastic-topology seam

The project SHALL define a `PlasticTopology` Protocol carrying exactly what a local plasticity rule touches, so the same rule can drive different substrates without naming any of them: an ordered list of plastic weight tensors, an aligned list of eligibility traces of the same shapes, an aligned list of boolean edge masks, a flag stating whether traces are enabled, and the mask projector applied per tensor.

The seam SHALL be a list from the outset, so that a substrate with one plastic tensor and a substrate with one per layer are handled by the same code path. A dense substrate SHALL expose an all-true mask rather than omitting one, so mask-dependent telemetry has the same meaning on every substrate.

The connectome topology SHALL expose the seam as views over its existing chemical-weight, edge-mask, and trace tensors, adding no state and leaving its trace update unchanged.

#### Scenario: Both substrates satisfy the seam

- **WHEN** the connectome topology and the MLP topology are inspected
- **THEN** each SHALL satisfy the `PlasticTopology` Protocol at runtime
- **AND** each SHALL expose aligned lists whose traces and masks match their weights' shapes entry for entry

#### Scenario: The connectome seam is a view, not a copy

- **WHEN** the connectome's seam is read
- **THEN** its plastic weight SHALL be the same tensor object as `w_chem`
- **AND** its trace SHALL be the same tensor object as the topology's eligibility buffer

#### Scenario: A dense substrate exposes a full mask

- **WHEN** the MLP topology's masks are read
- **THEN** every mask SHALL be all-true with the shape of its weight

### Requirement: Matched-rule invariance across substrates

The three-factor rule SHALL be one implementation driving every substrate through the seam, so that "matched rule" is a property of the code and not only of the equation. Two arms trained by the matched rule SHALL share the same rule class, the same update arithmetic, and the same hyperparameter values.

The plasticity hyperparameters — rule selection, plasticity rate, weight decay, magnitude bound, baseline rate, trace enablement, and trace decay — SHALL be defined **once** and inherited by every brain configuration that offers the plastic rules, together with their validation, so the defaults cannot drift between arms without a single edit being visible.

Telemetry SHALL carry the same keys with the same semantics on every substrate: the mean absolute effective weight change aggregated over all plastic entries, and the saturated fraction over all masked entries.

#### Scenario: The same update lands on both substrates

- **GIVEN** a connectome topology and an MLP topology, each with a scripted eligibility trace
- **WHEN** the rule steps each with the same reward and baseline
- **THEN** each plastic tensor SHALL change by the plasticity rate times the prediction error times its trace, before stabilisation terms

#### Scenario: Plasticity defaults are identical across brain configurations

- **WHEN** the connectome and MLP brain configurations are constructed with no plasticity fields set
- **THEN** every plasticity field SHALL hold the same value on both

#### Scenario: The trace-pairing validator applies to every plastic brain

- **WHEN** either brain configuration selects a plastic rule without enabling traces
- **THEN** loading SHALL fail with the same error

#### Scenario: Telemetry means the same thing on both arms

- **WHEN** the rule reports after stepping each substrate
- **THEN** the report SHALL carry the same keys
- **AND** the saturated fraction SHALL be measured over masked entries only on both
