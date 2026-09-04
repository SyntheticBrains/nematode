# Spec: connectome-ppo-brain

## ADDED Requirements

### Requirement: Persistent activity-trace substrate (eligibility traces)

`ConnectomePPOBrainConfig` SHALL expose `enable_activity_traces: bool = False` and `trace_decay: float = 0.9`. When enabled, `ConnectomeTopology` SHALL maintain a per-synapse eligibility buffer `E` of shape `(n_neurons, n_neurons)`, allocated **only when enabled** (no state-dict or RNG footprint otherwise, constructed after all RNG-consuming parameter blocks), updated **once per environment step** inside the rollout forward under `torch.no_grad()` as `E ← trace_decay · E + M_chem ∘ (h hᵀ)`, where `h` is the settled post-K hidden state and `E[i, j]` is oriented pre-`i` → post-`j`, matching `w_chem`'s edge layout. The batched (PPO replay) forward SHALL NOT read or mutate `E`. Traces SHALL reset via `prepare_episode()` at episode start; they therefore persist through the terminal `learn(..., episode_done=True)` call of the preceding episode, and this ordering SHALL be documented. Until a learning rule consumes `E`, training SHALL be bit-invariant to the flag: identical seeds with traces on and off SHALL produce identical parameters.

#### Scenario: Traces off is byte-identical

- **WHEN** a brain is constructed with `enable_activity_traces` unset or `false`
- **THEN** every constructed tensor (`m_chem`, `w_chem`, `g_gap`, gains, readout) SHALL be byte-identical to the pre-change brain at the same seed
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

The connectome brain's PPO update SHALL execute inside `ConnectomePPORule` (package `learning_rules`), which owns the optimiser, critic, PPO hyperparameters, and the epoch × minibatch loop; the brain SHALL retain experience collection (`RolloutBuffer`, pending tuples) and feature unpacking, surfaced to the rule through `batch`. The extraction SHALL be behaviour-preserving to the house byte-equivalence bar: identical tensor operations in identical order (the same `_policy.py` helpers with the same arguments; the strict-mask projection at the same post-optimiser point; `freeze_updates` and empty-buffer short-circuits preserved), verified against a frozen in-tree reference of the pre-change update with bit-equal parameters and identical RNG streams. `step` SHALL return a `RuleStepReport`; the brain SHALL append its `total_loss` to `history_data.losses` (a declared additive telemetry change — the brain previously recorded no loss).

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
