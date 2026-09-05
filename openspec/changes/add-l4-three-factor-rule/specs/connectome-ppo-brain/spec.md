# Spec: connectome-ppo-brain

## ADDED Requirements

### Requirement: Learning-rule selection on the connectome brain

`ConnectomePPOBrainConfig` SHALL expose `learning_rule` selecting between the PPO update and the three-factor plasticity rule, defaulting to PPO. The selection SHALL be a configuration option on the existing brain rather than a separate brain class, so that panel arms differ in exactly the dimension under test and share every other code path.

Selecting the three-factor rule SHALL require activity traces to be enabled; the pairing SHALL be validated at load time, because a plasticity rule with no eligibility trace to read would train silently and produce no weight change at all.

Under the three-factor rule the brain SHALL invoke the rule **once per environment step** as rewards arrive, rather than accumulating a rollout and updating when full. The rollout buffer, advantage estimation, and value head SHALL NOT be exercised.

The default selection SHALL be byte-identical to the pre-change brain: identical seeds SHALL produce identical parameters after the same number of updates, verified against a frozen reference of the pre-change behaviour.

#### Scenario: Default selection is byte-identical

- **WHEN** a brain is constructed with `learning_rule` unset
- **THEN** every constructed tensor SHALL be byte-identical to a pre-change construction at the same seed
- **AND** training SHALL produce bit-identical parameters to the frozen pre-change reference

#### Scenario: Three-factor without traces fails at load

- **WHEN** a configuration selects the three-factor rule without enabling activity traces
- **THEN** loading SHALL fail with an error naming both fields and stating that the rule requires a trace

#### Scenario: Three-factor updates once per step

- **GIVEN** a brain configured with the three-factor rule
- **WHEN** a sequence of environment steps with rewards is run
- **THEN** one rule update SHALL occur per step
- **AND** the update SHALL NOT wait for a rollout buffer to fill

#### Scenario: PPO machinery is dormant under the three-factor rule

- **GIVEN** a brain configured with the three-factor rule
- **WHEN** an episode runs to completion
- **THEN** no value estimate, advantage, or clipped-surrogate loss SHALL be computed
- **AND** the chemical weights SHALL have changed from their initial values

#### Scenario: Selecting a rule does not change construction

- **WHEN** two brains are constructed at the same seed with different rule selections
- **THEN** all topology parameters SHALL be byte-identical at initialisation

### Requirement: Anatomically-derived motor readout under the plastic rule

When the three-factor rule is selected the motor readout is never updated, so what it is frozen at determines what the plastic arms can express. It SHALL therefore be set to the anatomical contrast implied by the motor classes it reads, rather than to a random draw.

The four pooled motor classes carry fixed anatomical meaning: two denote dorsal versus ventral muscle innervation, and two denote forward versus backward locomotion drive. The readout SHALL map the dorsal-minus-ventral contrast to the turn action and the forward-minus-backward contrast to the speed action.

The assignment SHALL consume no additional randomness and SHALL occur after the existing initialisation draw, so that every other parameter remains byte-identical across rule selections at the same seed. Under the PPO rule the readout SHALL keep its existing random initialisation, so previously measured results remain reproducible bit-for-bit.

#### Scenario: Plastic readout encodes the anatomical contrasts

- **WHEN** a brain is constructed with the three-factor rule
- **THEN** the readout row driving turn SHALL weight dorsal classes and ventral classes with equal magnitude and opposite sign
- **AND** the readout row driving speed SHALL weight forward-drive classes and backward-drive classes with equal magnitude and opposite sign

#### Scenario: The anatomical readout preserves initialisation scale

- **WHEN** the anatomical readout is constructed
- **THEN** its rows SHALL be unit-norm and mutually orthogonal, matching the scale and conditioning of the random initialisation it replaces

#### Scenario: Rule selection consumes identical randomness

- **WHEN** two brains are constructed at the same seed under different rule selections
- **THEN** every parameter other than the readout SHALL be byte-identical
- **AND** the random-number stream SHALL have advanced identically

#### Scenario: PPO arms keep the random readout

- **WHEN** a brain is constructed with the PPO rule
- **THEN** its readout SHALL be byte-identical to a pre-change construction at the same seed

#### Scenario: The readout is identical across wiring arms

- **WHEN** a wild-type brain and a rewired-null brain are constructed at the same seed under the three-factor rule
- **THEN** their readouts SHALL be byte-identical, so the wiring contrast is not confounded by the decoder

## MODIFIED Requirements

### Requirement: Persistent activity-trace substrate (eligibility traces)

`ConnectomePPOBrainConfig` SHALL expose `enable_activity_traces: bool = False` and `trace_decay: float = 0.9`, the latter pydantic-bounded to `0.0 ≤ trace_decay < 1.0` so an out-of-range value fails at load time (a decay ≥ 1 is a divergent accumulator). When enabled, `ConnectomeTopology` SHALL maintain a per-synapse eligibility buffer `E` of shape `(n_neurons, n_neurons)`, allocated **only when enabled** (no state-dict or RNG footprint otherwise, constructed after all RNG-consuming parameter blocks), updated on each **rollout forward invocation** (`forward_with_hidden`) under `torch.no_grad()` — once per environment step by call-site convention, documented on the method.

**Update formula (v2, amended 2026-09-05 by the A.3 three-factor rule change, exercising the v1 requirement's explicit amendment clause):**

`E ← trace_decay · E + M_chem ∘ (h_prev ⊗ h)`

where `h` is the settled post-K hidden state of the current step, `h_prev` is the settled hidden state of the **previous** step within the same episode, and `E[i, j]` is oriented pre-`i` → post-`j`, matching `w_chem`'s edge layout. The previous-step pre-synaptic term makes the trace **causal** (pre-synaptic activity precedes the post-synaptic activity it is credited with) and **directional** (reciprocal edge pairs, which the v1 symmetric `h hᵀ` gave identical eligibility, are distinguished). The superseded v1 formula is retained in the archived A.2 change as the historical record.

`h_prev` SHALL be maintained as topology state, reset alongside `E` at episode start. On the first step of an episode there is no previous state, so that step SHALL contribute no eligibility; accrual begins on the second step.

The SHALL-protected invariants are unchanged and formula-independent: masked support, `torch.no_grad()`, the rollout-forward update site, conditional allocation, the episode-reset lifecycle, PPO-path independence, and bit-invariance while unconsumed. The batched (PPO replay) forward SHALL NOT read or mutate `E` or `h_prev`. Traces SHALL reset via `prepare_episode()` at episode start (`reset_traces()` SHALL be a no-op when `E` is unallocated); they therefore persist through the terminal `learn(..., episode_done=True)` call of the preceding episode, and this ordering SHALL be documented. While no learning rule consumes `E`, training SHALL remain bit-invariant to the flag: identical seeds with traces on and off SHALL produce identical parameters.

#### Scenario: Traces off is byte-identical

- **WHEN** a brain is constructed with `enable_activity_traces` unset or `false`
- **THEN** every constructed tensor SHALL be byte-identical to a default-config construction at the same seed
- **AND** neither a trace buffer nor previous-state storage SHALL be allocated

#### Scenario: Trace recurrence matches the closed form

- **WHEN** traces are enabled and a scripted sequence of steps is run
- **THEN** `E` SHALL equal the closed form accumulating decayed outer products of each step's previous and current settled states
- **AND** `E` SHALL be zero on every non-edge

#### Scenario: The first step of an episode contributes no eligibility

- **GIVEN** an episode that has just started
- **WHEN** exactly one step is taken
- **THEN** `E` SHALL be all zeros

#### Scenario: The trace distinguishes reciprocal edges

- **GIVEN** a topology containing a reciprocal edge pair
- **WHEN** the two endpoints carry different activity across consecutive steps
- **THEN** the eligibility of the two directions SHALL differ

#### Scenario: Traces reset at episode boundaries

- **WHEN** `prepare_episode()` is called
- **THEN** `E` SHALL be all zeros
- **AND** the stored previous state SHALL be cleared
- **AND** the rule's `reset_episode()` SHALL have been invoked

#### Scenario: Training is bit-invariant while no rule consumes traces

- **WHEN** two brains train under the PPO rule with identical seeds, one with traces enabled and one without
- **THEN** all learnable parameters SHALL be `torch.equal` after the same number of updates
- **AND** the batched update path SHALL leave `E` unchanged
