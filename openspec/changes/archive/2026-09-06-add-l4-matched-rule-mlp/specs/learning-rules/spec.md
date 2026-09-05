# Spec: learning-rules

## ADDED Requirements

### Requirement: Plastic-topology seam

The project SHALL define a `PlasticTopology` Protocol carrying exactly what a local plasticity rule touches, so the same rule can drive different substrates without naming any of them: an ordered list of plastic weight tensors, an aligned list of eligibility traces of the same shapes, an aligned list of boolean edge masks, and a flag stating whether traces are enabled. Masking is expressed through the aligned masks: the rule multiplies each tensor's update by its own mask, which on a 0/1 mask is bitwise-identical to the connectome's projector and needs no projector member on the seam.

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

## MODIFIED Requirements

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
