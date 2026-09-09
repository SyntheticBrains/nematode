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
- **THEN** no plastic weight magnitude SHALL exceed the configured bound

#### Scenario: Decay pulls unreinforced weights down

- **GIVEN** a zero prediction error
- **WHEN** the rule steps repeatedly
- **THEN** plastic weight magnitudes SHALL be non-increasing

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

The project SHALL define a `PlasticTopology` Protocol carrying exactly what a local plasticity rule touches, so the same rule can drive different substrates without naming any of them: an ordered list of plastic weight tensors, an aligned list of eligibility traces of the same shapes, an aligned list of boolean edge masks, an aligned list of fan-in axes (`plastic_fan_in_axes`, for each tensor the axis to reduce over to obtain one unit's incoming weights), an aligned list of post-synaptic activities (`plastic_post_activities`, for each tensor the activity vector of its post-synaptic units from the step the trace was last accumulated on, indexed along the axis complementary to the fan-in axis), and a flag stating whether traces are enabled. Masking is expressed through the aligned masks: the rule multiplies each tensor's update by its own mask, which on a 0/1 mask is bitwise-identical to the connectome's projector and needs no projector member on the seam.

The seam SHALL be a list from the outset, so that a substrate with one plastic tensor and a substrate with one per layer are handled by the same code path. A dense substrate SHALL expose an all-true mask rather than omitting one, so mask-dependent telemetry has the same meaning on every substrate.

The connectome topology SHALL expose the seam as views over its existing chemical-weight, edge-mask, trace and activity tensors, adding no state and leaving its trace update unchanged. Its chemical matrix is indexed `[pre, post]`, so its fan-in axis SHALL be `0` and its post-synaptic activity is indexed along axis `1`; a `Linear` weight is `[out, in]`, so the MLP topology's fan-in axis SHALL be `1` for every layer and its post-synaptic activity is indexed along axis `0`. The post-synaptic activity SHALL be the same vector the trace's post-synaptic factor was built from, so that a term reading it and the Hebbian term agree on what the unit did.

#### Scenario: Both substrates satisfy the seam

- **WHEN** the connectome topology and the MLP topology are inspected
- **THEN** each SHALL satisfy the `PlasticTopology` Protocol at runtime
- **AND** each SHALL expose aligned lists whose traces and masks match their weights' shapes entry for entry
- **AND** each SHALL expose one fan-in axis per plastic tensor: `0` for the chemical matrix, `1` for each MLP layer
- **AND** each SHALL expose one post-synaptic activity vector per plastic tensor whose length equals the weight's extent along the axis complementary to its fan-in axis

#### Scenario: The connectome seam is a view, not a copy

- **WHEN** the connectome's seam is read
- **THEN** its plastic weight SHALL be the same tensor object as `w_chem`
- **AND** its trace SHALL be the same tensor object as the topology's eligibility buffer

#### Scenario: A dense substrate exposes a full mask

- **WHEN** the MLP topology's masks are read
- **THEN** every mask SHALL be all-true with the shape of its weight

#### Scenario: The post-synaptic activity is the trace's own

- **WHEN** a trace has just been accumulated from pre-synaptic activity `x` and post-synaptic activity `y`
- **THEN** the seam's post-synaptic activity for that tensor SHALL equal `y`

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

### Requirement: Substrate-invariant scaling of the three-factor update

The three-factor rule SHALL offer two independent, default-off scaling modes shared by every
plastic brain through the plasticity configuration mixin. With **modulator normalisation** on,
the third factor SHALL be `tanh(δ / σ) − c`, where `σ` is a running root-mean-square of the raw
prediction error `δ` maintained by exponential moving average at a configurable scale rate,
bias-corrected so its first observation counts fully, used at its pre-update value for the
current step, and floored at a configurable positive floor before division; and `c` is a
bias-corrected running mean of `tanh(δ / σ)` at the same scale rate, zero before any
observation and used at its pre-update value for the current step, so that the modulator is
zero-mean under the agent's own policy as a prediction error must be. With **trace
normalisation** on, the Hebbian term for each plastic tensor SHALL be divided by `ρ`, a running
root-mean-square of that tensor's eligibility trace over its masked entries, maintained,
bias-corrected, used and floored the same way, with an all-zero trace neither updating it nor
counting toward its correction. The decay term and the magnitude clamp SHALL be unchanged by
either mode. The scales and the centre SHALL advance under a freeze and in unmodulated mode,
where the modulator SHALL remain `1.0`. With both modes off, the rule SHALL be bit-identical to
the rule without this requirement. The rule SHALL report the effective modulator, the modulator
scale, the modulator centre and the mean trace scale beside its existing telemetry, and the raw
prediction error SHALL still be reported.

#### Scenario: The modulator is bounded and scale-free

- **GIVEN** modulator normalisation on and a warmed scale `σ`
- **WHEN** the rule steps with prediction error `δ`
- **THEN** the modulator SHALL equal `tanh(δ / σ_prev) − c_prev`, with the scale and the centre
  from before this step
- **AND** it SHALL lie in `[−2, 2]` for any `δ`
- **AND** the centre SHALL be zero before any observation and the bias-corrected running mean
  of `tanh(δ / σ)` after
- **AND** the raw `δ` SHALL still be reported as the prediction error, and the centre SHALL be
  reported beside it

#### Scenario: The modulator is zero-mean under a skewed reward stream

- **GIVEN** modulator normalisation on and a deterministic periodic stream of prediction errors
  with many small values, frequent moderate positives and rare large negatives, whose raw values
  sum to zero over each period
- **WHEN** the rule steps through whole periods past a whole-period warm-up
- **THEN** the mean of the modulator over those steps SHALL be within `0.005` of zero
- **AND** the mean of the uncentred `tanh(δ / σ)` on the same steps SHALL exceed `0.005` in
  magnitude, so the centring is shown to be load-bearing

#### Scenario: The trace step is invariant to the trace's scale

- **GIVEN** trace normalisation on and two topologies whose traces differ by a constant factor
- **WHEN** both scales have warmed and the rule steps each with the same modulator
- **THEN** the Hebbian steps SHALL be equal within floating-point tolerance
- **AND** each plastic tensor SHALL carry its own scale, computed over its masked entries only

#### Scenario: Scales are bias-corrected from the first observation

- **GIVEN** a freshly constructed rule with a scaling mode on
- **WHEN** it takes its first step
- **THEN** the modulator scale SHALL equal the first `|δ|` (floored) and the trace scale the first
  non-zero trace's root-mean-square (floored)
- **AND** after `t` observations each scale SHALL equal its bias-corrected running mean square
- **AND** an all-zero trace SHALL leave the trace scale and its observation count unchanged

#### Scenario: Both modes off is bit-identical

- **GIVEN** a configuration with the default scaling fields
- **WHEN** the rule steps in any mode (modulated or not, frozen or not)
- **THEN** the weights and the existing telemetry SHALL be bit-identical to the frozen reference

#### Scenario: Freeze and unmodulated mode keep the scales comparable

- **GIVEN** a frozen arm and an unmodulated arm with a scaling mode on
- **WHEN** each steps
- **THEN** the scales and the centre SHALL advance exactly as on the plastic modulated arm
- **AND** the frozen arm SHALL write no weight
- **AND** the unmodulated arm's modulator SHALL be `1.0` while its trace step is normalised

#### Scenario: Matched rule across substrates under normalisation

- **GIVEN** the connectome and the MLP with both scaling modes on and the same hyperparameters
- **WHEN** each steps from traces whose magnitudes differ by orders of magnitude
- **THEN** the root-mean-square Hebbian step per unit modulator SHALL be the same on both within
  tolerance

#### Scenario: The scaling fields are shared and bounded

- **WHEN** a plastic brain configuration is loaded
- **THEN** the four scaling fields SHALL come from the shared mixin with identical defaults on
  every plastic brain
- **AND** a scale rate outside `(0, 1]` or a non-positive floor SHALL fail at load

### Requirement: Homeostatic incoming-norm scaling

The three-factor rule SHALL offer a default-off homeostatic mode, shared by every plastic brain
through the plasticity configuration mixin. When on, the rule SHALL capture at construction, for
every unit of every plastic tensor, the norm of that unit's incoming plastic weights over the
tensor's masked entries, and after each plastic update SHALL rescale each unit's incoming weights over the masked
entries only — off-edge entries SHALL never be written — so that their norm returns to that
target, dividing by the unit's current norm; a unit with a zero target, or whose incoming weights are all zero (no direction to scale along), SHALL be left untouched; the magnitude clamp
SHALL be applied after the rescale. Nothing SHALL be rescaled under a freeze. The rule SHALL obtain each
tensor's fan-in axis from the plastic-topology seam and SHALL NOT name any substrate's layout.
The decay term SHALL remain in place; under homeostasis its uniform shrink is undone by the
rescale. The rule SHALL report the mean relative norm deviation
before the rescale (`plasticity_norm_drift`), NaN when the mode is off. With the mode off, the
rule SHALL be bit-identical to the rule without this requirement.

#### Scenario: Incoming norms return to their targets on both substrates

- **GIVEN** homeostasis on and a rule stepped over the connectome and over the MLP with
  non-trivial traces and modulators
- **WHEN** an update has been applied
- **THEN** every unit with a non-zero target SHALL have incoming norm equal to its target within
  floating-point tolerance, on the connectome over the chemical mask and on the MLP over each
  layer's rows
- **AND** under the connectome's soft-prior mask the off-edge entries SHALL be unchanged by the
  rescale

#### Scenario: Units without inputs and the bound are respected

- **GIVEN** homeostasis on
- **WHEN** a unit has no incoming edges
- **THEN** its (zero) incoming weights SHALL be unchanged
- **AND** no weight SHALL exceed the magnitude bound after the rescale
- **AND** a unit whose norm is positive but tiny SHALL be restored to its target by its actual norm
- **AND** a unit whose incoming weights are all zero SHALL be left at zero

#### Scenario: Homeostasis off is bit-identical and a freeze rescales nothing

- **GIVEN** the default configuration
- **WHEN** the rule steps in any mode
- **THEN** the weights and the existing telemetry SHALL be bit-identical to the frozen reference
- **AND** with homeostasis on under a freeze, no weight SHALL change while the norm drift is still
  reported

#### Scenario: The connectome's targets follow its initialisation scale

- **GIVEN** the connectome with homeostasis on
- **WHEN** the rule captures its targets
- **THEN** every neuron with chemical inputs SHALL have a target near 1, the incoming norm its
  `1/√k` initialisation gives by construction

### Requirement: Consolidation mechanisms for the three-factor update

The three-factor rule SHALL offer a **consolidation mechanism** selected by a single
configuration value shared by every plastic brain through the plasticity configuration mixin,
one of `none`, `anchor`, `rigidity` or `oracle`, defaulting to `none`. With `none` selected the
rule SHALL be bit-identical to the rule without this requirement, SHALL allocate no consolidation
state, and SHALL add no operation to the update. A selector other than `none` that its own
parameters leave inert SHALL be rejected at load, since it names a mechanism that would not act:
`anchor` with a stiffness of zero, `rigidity` with a growth or a strength of zero.

With `anchor` selected, each plastic tensor SHALL carry an anchor of its own shape, initialised
to the weights the rule was constructed over. The update SHALL gain a restoring term
`− η · κ_a · (w − a)` applied inside the same masked update as the weight decay, and after the
write the anchor SHALL advance as `a ← a + ρ_a · (w − a)` from the weights as they then are. An
anchor rate of zero SHALL hold the anchor fixed at the weights the rule started from.

With `rigidity` selected, each plastic tensor SHALL carry a non-negative protective variable of
its own shape, zero at construction. The Hebbian term's rate SHALL be divided by `1 + κ_c · c`
using the protective variable's value from **before** this step's growth, and after the update the
protective variable SHALL advance as `c ← (1 − λ_c) · c + γ_c · max(m, 0) · |E| / ρ_E`, where `m`
is the effective modulator, `E` the eligibility trace and `ρ_E` the running trace scale when trace
normalisation is on and `1` otherwise, so the growth is measured on the trace as the update sees
it and a pinned growth rate means the same thing on every substrate. Under the unmodulated mode `m` is `1.0` and the
growth term SHALL follow the trace alone, so the two arms differ in the modulator and nowhere
else.

With `oracle` selected, the rule SHALL maintain a trailing episode-success rate as an exponential
moving average of an episode-success flag supplied by the brain at the end of each episode, and
SHALL scale the plasticity rate by `clamp(1 − s / s_ref, 0, 1)` for a configured reference rate
`s_ref`. The reference SHALL default to `1.0`, at which the gate stays open for every trailing estimate below `1.0` — closing only on an estimate saturated at `1.0`, which the scaling already implies — so that a run selecting
this mechanism pins its reference explicitly. This mechanism consumes a quantity that is a
property of the task's scoring rather than of the reward stream the rule observes; its
implementation SHALL say so, and it SHALL NOT be offered as a default or presented as a
biologically plausible mechanism.

Consolidation SHALL compose with, and never replace, the existing terms: the Hebbian term, the
weight decay, the mask, Dale's-law sign projection, the homeostatic rescale and the magnitude
clamp SHALL keep their order, with the clamp last, and consolidation state SHALL be updated after
the write.

#### Scenario: The default path is unchanged

- **WHEN** the consolidation selector is `none`
- **THEN** the weight trajectory SHALL be bit-identical to the rule without consolidation
- **AND** no anchor and no protective variable SHALL be allocated

#### Scenario: A named mechanism with no effect is rejected

- **GIVEN** a configuration selecting `anchor` with a stiffness of zero, or `rigidity` with a
  growth of zero or a strength of zero
- **WHEN** the configuration is loaded
- **THEN** loading SHALL fail with a message naming the parameter that leaves the mechanism inert

#### Scenario: The anchor opposes departure and follows slowly

- **GIVEN** `anchor` selected with a positive stiffness and a positive anchor rate
- **WHEN** a weight has moved away from its anchor and a step is applied
- **THEN** the update SHALL include a term proportional to the negative departure
- **AND** the anchor SHALL then move toward the weight by the anchor rate

#### Scenario: A fixed anchor holds the starting policy

- **GIVEN** `anchor` selected with an anchor rate of zero
- **WHEN** any number of steps are applied
- **THEN** the anchor SHALL remain the weights the rule was constructed over

#### Scenario: Rigidity grows where reward gated a large trace and slows later writes

- **GIVEN** `rigidity` selected
- **WHEN** steps with a positive modulator and a large trace are applied to the same synapses
- **THEN** those synapses' protective variable SHALL increase
- **AND** a subsequent step's Hebbian term at those synapses SHALL be smaller than the same step
  applied with the mechanism off
- **AND** the divisor SHALL use the pre-growth value, so a step is not charged for the rigidity it
  creates

#### Scenario: Rigidity decays so nothing is frozen permanently

- **GIVEN** a synapse whose protective variable is positive
- **WHEN** steps are applied that neither reinforce it nor drive its trace
- **THEN** its protective variable SHALL decrease toward zero at the configured decay

#### Scenario: The oracle stops updating at its reference success rate

- **GIVEN** `oracle` selected with a reference rate `s_ref`
- **WHEN** the trailing episode-success rate reaches or exceeds `s_ref`
- **THEN** the effective plasticity rate SHALL be zero and no weight SHALL change
- **AND** when the trailing rate is far below `s_ref` the rate SHALL be the configured rate
- **AND** with the reference at its default of `1.0` the rate SHALL NOT reach zero while the trailing estimate is below `1.0`

#### Scenario: Consolidation is applied before the bound and after the sign

- **WHEN** a consolidated update is applied on a substrate with grounded signs and homeostasis on
- **THEN** the sign projection, the homeostatic rescale and the magnitude clamp SHALL apply in
  their existing order, with the clamp last

### Requirement: Consolidation state follows a loaded policy

When the rule's running state is reset — the path taken when weights are loaded into a brain —
the elastic anchor SHALL be re-anchored to the weights the rule now starts from and the protective
variable SHALL be returned to zero, alongside the existing baseline, scale, centre, trace and
homeostatic-target resets. An anchor left at the values of a previous substrate would pull a
loaded policy toward weights it no longer has, with a force proportional to the distance between
them.

Consolidation state SHALL NOT be persisted with weights: a checkpoint carries the policy, and a
rule constructed over a loaded policy SHALL anchor to that policy.

#### Scenario: A loaded clone is anchored to itself

- **GIVEN** a rule with `anchor` selected, constructed over a random initialisation
- **WHEN** a cloned competent policy is loaded and the rule's state is reset
- **THEN** the anchor SHALL equal the loaded weights
- **AND** the first update's restoring term SHALL be zero

#### Scenario: The protective variable does not survive a load

- **GIVEN** a rule with `rigidity` selected whose protective variable has grown
- **WHEN** weights are loaded and the rule's state is reset
- **THEN** the protective variable SHALL be zero

### Requirement: Consolidation telemetry

The rule SHALL report, beside its existing plasticity telemetry, the effective rate multiplier it
actually applied, the mean absolute departure of the weights from their anchor over the edge set,
and the mean protective variable over the edge set. Under `none` the multiplier SHALL be `1` and
the two means SHALL be reported as not-a-number, in the same way the scaling telemetry reports an
inactive estimator. Every key SHALL be recorded by the shared plasticity report so the arms of a
screen remain comparable step for step.

#### Scenario: A variant that held a policy by not moving is distinguishable

- **WHEN** a screen reads a run's telemetry
- **THEN** the effective rate multiplier and the anchor departure SHALL be available per step
- **AND** they SHALL make a run that consolidated distinguishable from one whose updates were
  suppressed outright

### Requirement: Decorrelating terms for the three-factor update

The three-factor rule SHALL offer a **decorrelating term** selected by a single configuration
value shared by every plastic brain through the plasticity configuration mixin, one of `none`,
`anti_hebbian_inhibitory` or `oja`, defaulting to `none`. With `none` selected the rule SHALL be
bit-identical to the rule without this requirement and SHALL add no operation to the update.

With `anti_hebbian_inhibitory` selected, the Hebbian term SHALL be negated for every synapse whose
grounded sign is inhibitory, and left unchanged for grounded excitatory synapses and for synapses
the substrate grounds no sign for. The term's magnitude SHALL be unchanged, so the variant
redirects the update rather than resizing it. Selecting it SHALL be rejected at load on a
substrate whose signs are not grounded, since the negation would otherwise key on an arbitrary
draw.

With `oja` selected, the update SHALL gain a term `− η · γ · y² · w` inside the same masked update
as the weight decay, where `y` is the post-synaptic activity the eligibility trace was built from,
broadcast along each weight's post-synaptic axis, and `γ` is a configured coefficient. A
coefficient of zero SHALL be rejected with the mechanism selected, since it names a term that
would not act.

The decorrelating term SHALL compose with every existing term: the Hebbian term, the weight decay,
consolidation, the mask, Dale's-law projection, the homeostatic rescale and the magnitude clamp
SHALL keep their order, with the clamp last.

#### Scenario: The default path is unchanged

- **WHEN** the decorrelation selector is `none`
- **THEN** the weight trajectory SHALL be bit-identical to the rule without decorrelation

#### Scenario: Inhibitory synapses learn with the opposite sign

- **GIVEN** `anti_hebbian_inhibitory` selected on a substrate with grounded signs
- **WHEN** a step is applied
- **THEN** the update at each grounded inhibitory synapse SHALL equal the negation of the update
  the rule would have applied there with the selector off
- **AND** the update at grounded excitatory and ungrounded synapses SHALL be unchanged

#### Scenario: The variant redirects the update without resizing it

- **WHEN** `anti_hebbian_inhibitory` is selected
- **THEN** the total absolute magnitude of the Hebbian term over the edge set SHALL equal the
  magnitude the same step would have produced with the selector off

#### Scenario: Flipping an arbitrary sign is refused

- **GIVEN** a configuration selecting `anti_hebbian_inhibitory` on a substrate whose synapse signs
  are not grounded in the atlas
- **WHEN** the configuration is loaded
- **THEN** loading SHALL fail with a message saying the variant keys on grounded signs

#### Scenario: The Oja term opposes growth in proportion to post-synaptic activity

- **GIVEN** `oja` selected with a positive coefficient
- **WHEN** a step is applied with a post-synaptic unit active and its incoming weight non-zero
- **THEN** that weight's update SHALL include a term proportional to the negative of the weight
  times the square of that unit's activity
- **AND** a unit with zero activity SHALL receive no such term

#### Scenario: A named term with no coefficient is rejected

- **GIVEN** a configuration selecting `oja` with a coefficient of zero
- **WHEN** the configuration is loaded
- **THEN** loading SHALL fail with a message naming the coefficient

### Requirement: Decorrelation telemetry

The rule SHALL report the share of the update's total absolute magnitude carried by the
decorrelating term — the negated subset's share under `anti_hebbian_inhibitory`, the Oja term's
share under `oja`, and zero under `none` — beside its existing plasticity telemetry, recorded by
the shared plasticity report. An arm that improved by decorrelating and an arm that improved by
updating less are different results, and this quantity separates them.

#### Scenario: A recovery attributable to something other than the term is visible

- **WHEN** a screen reads a run's telemetry
- **THEN** the decorrelation share SHALL be available per step
- **AND** a run whose share is near zero SHALL be distinguishable from one whose share is large
