# Spec: connectome-ppo-brain

## ADDED Requirements

### Requirement: Persistent activity-trace substrate (eligibility traces)

`ConnectomePPOBrainConfig` SHALL expose `enable_activity_traces: bool = False` and `trace_decay: float = 0.9`, the latter pydantic-bounded to `0.0 ≤ trace_decay < 1.0` so an out-of-range value fails at load time (tracker Decision B.6; a decay ≥ 1 is a divergent accumulator). When enabled, `ConnectomeTopology` SHALL maintain a per-synapse eligibility buffer `E` of shape `(n_neurons, n_neurons)`, allocated **only when enabled** (no state-dict or RNG footprint otherwise, constructed after all RNG-consuming parameter blocks), updated on each **rollout forward invocation** (`forward_with_hidden`) under `torch.no_grad()` — once per environment step by call-site convention, documented on the method. The **v1 update formula** is `E ← trace_decay · E + M_chem ∘ (h hᵀ)`, where `h` is the settled post-K hidden state and `E[i, j]` is oriented pre-`i` → post-`j`, matching `w_chem`'s edge layout; the formula is the **amendable part** of this requirement — the A.3 rule change MAY replace it with a dated amendment if its three-factor design needs directional or injection-aware pre-terms (`h hᵀ` is symmetric, and settled `h` overwrites the sensory injection; alternatives recorded in the change's design). The SHALL-protected invariants are: masked support, `torch.no_grad()`, the rollout-forward update site, conditional allocation, the episode-reset lifecycle, PPO-path independence, and bit-invariance until consumed. The batched (PPO replay) forward SHALL NOT read or mutate `E`. Traces SHALL reset via `prepare_episode()` at episode start (`reset_traces()` SHALL be a no-op when `E` is unallocated); they therefore persist through the terminal `learn(..., episode_done=True)` call of the preceding episode, and this ordering SHALL be documented. Until a learning rule consumes `E`, training SHALL be bit-invariant to the flag: identical seeds with traces on and off SHALL produce identical parameters.

#### Scenario: Traces off is byte-identical

- **WHEN** a brain is constructed with `enable_activity_traces` unset or `false`
- **THEN** every constructed tensor (`m_chem`, `w_chem`, `g_gap`, gains, readout) SHALL be byte-identical to a default-config construction at the same seed (the 034 `TestWiringControl` template; pre-change identity follows because the trace path consumes no RNG — argued in design)
- **AND** no trace buffer SHALL be allocated

#### Scenario: Trace recurrence matches the closed form

- **WHEN** traces are enabled and a scripted sequence of steps is run
- **THEN** `E` SHALL equal the closed-form `Σ_t trace_decay^(T-t) · M_chem ∘ (h_t h_tᵀ)` within float32 exactness of the recurrence
- **AND** `E` SHALL be zero on every non-edge (masked)

#### Scenario: Traces reset at episode boundaries

- **WHEN** `prepare_episode()` is called
- **THEN** `E` SHALL be all zeros
- **AND** the rule's `reset_episode()` SHALL have been invoked

#### Scenario: Training is bit-invariant while no rule consumes traces

- **WHEN** two brains train with identical seeds, one with traces enabled and one without
- **THEN** all learnable parameters SHALL be `torch.equal` after the same number of PPO updates
- **AND** the batched update path SHALL leave `E` unchanged

### Requirement: PPO update routed through the learning-rule seam

The connectome brain's PPO update SHALL execute inside `ConnectomePPORule` (package `learning_rules`), which owns the optimiser, critic, PPO hyperparameters (including `gamma`/`gae_lambda` — the rule therefore computes returns and advantages itself), and the epoch × minibatch loop, invoking the buffer's `get_minibatches` **once per epoch** so the per-epoch permutation draws are preserved; the rule SHALL receive the continuous flag, action bounds, `chemical_mask_mode`, and device at construction. The brain SHALL retain experience collection (`RolloutBuffer`, pending tuples) and feature unpacking, surfaced to the rule through `batch` (the buffer handle, `_unpack_state_batched` bound as a callable, and `last_value`). The extraction SHALL be behaviour-preserving to the house byte-equivalence bar: identical tensor operations in identical order (the same update-path `_policy.py` helpers with the same arguments; the strict-mask projection at the same post-optimiser point; `freeze_updates` and empty-buffer short-circuits preserved — still returning a `RuleStepReport` with `None` loss fields), verified against a frozen in-tree reference of the pre-change update with bit-equal parameters and identical RNG streams. `brain.critic` and `brain.optimizer` SHALL remain available as delegating properties (an existing test reads `brain.critic`; existing suites stay green unmodified). `step` SHALL return a `RuleStepReport`; the brain SHALL append its mean **policy loss** to `history_data.losses` — matching the house PPO convention (lstmppo/cfc/spiking all record `avg_policy`) — with a `None`-guard so frozen updates append nothing (a declared additive telemetry change; the brain previously recorded no loss).

#### Scenario: Extraction is bit-equivalent to the frozen reference

- **WHEN** the extracted rule and the frozen pre-change reference are driven from deep-copied identical state (same seeds, same buffer contents)
- **THEN** every learnable parameter and critic weight SHALL be `torch.equal` after the update
- **AND** the torch RNG state SHALL be identical afterwards

#### Scenario: Strict-mask projection survives the extraction

- **WHEN** a PPO update runs with `chemical_mask_mode: strict`
- **THEN** `w_chem` SHALL be zero on every non-edge after the update, exactly as pre-change

#### Scenario: Loss telemetry flows to tracking

- **WHEN** at least one PPO update has run
- **THEN** `history_data.losses` SHALL contain finite values (exported by the generic CSV path)
- **AND** with `freeze_updates: true` it SHALL remain empty
