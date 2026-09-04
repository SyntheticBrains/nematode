# Add the L4 trace substrate (rule seam + persistent activity traces)

## Why

Phase 7's first milestone (`phase7-tracking` tasks A.1 + A.2; roadmap D1/D8/D10 context). The L4 plasticity programme needs two prerequisites before any three-factor rule can exist:

1. **A real rule seam (A.1).** The `LearningRule` / `BrainTopology` Protocols shipped at L1 have **zero code consumers** — and reconnaissance shows the conformance story is worse than "unused": `ConnectomeTopology` does not satisfy `BrainTopology` at runtime (`isinstance` → `False`; it exposes `n_neurons`/`n_food_features`, not the Protocol's mandated `n_inputs`/`n_outputs`/`n_hidden`, and its multi-channel forward signature cannot match `forward(x)`). The plugin-developer guide's conformance claim is currently aspirational. Meanwhile the connectome brain's PPO update is inlined (`connectome_ppo.py:1404-1493`), so the A.3 three-factor rule would otherwise have no seam to plug into.
2. **Cross-step plasticity state (A.2).** `forward_with_hidden` zero-initialises `h` every env step (`connectome_ppo.py:709`) — there is no state on which an eligibility trace could accumulate. Three-factor rules need per-synapse traces that persist across steps within an episode.

Both are pure substrate: byte-identical-when-off, no learning-rule behaviour change, per the `phase7-tracking` spec's substrate-freeze and mechanism requirements.

## What Changes

### 1. Protocol reconciliation (A.1, part 1)

Amend `BrainTopology` to the seam a rule actually needs — `apply_weight_mask` + `learnable_parameters` — dropping the shape attributes and `forward(x)` that nothing implements or consumes (forward signatures are topology-specific). `ConnectomeTopology` then genuinely satisfies the Protocol at runtime. `LearningRule` keeps `step(topology, batch) -> RuleStepReport` + `reset_episode()`, with the ownership docstring softened so a brain-owned experience-collection buffer may be passed through `batch`. The plugin-developer guide's claims become true.

### 2. PPO update extracted through the seam (A.1, part 2)

The inlined `_perform_ppo_update` moves verbatim into `ConnectomePPORule` in the new **`learning_rules/` package** (roadmap D8 — `quantumnematode/plasticity/` is the unrelated quantum-plasticity eval protocol and is not touched). The rule owns the optimiser, critic, and PPO hyperparameters; the brain retains experience collection (`RolloutBuffer`, pending tuples) and feature unpacking. Every tensor op, `_policy.py` helper call, RNG draw, and the strict-mask projection execute in the same order — the extraction is held to the house **byte-equivalence bar** (M1 frozen-reference pattern: bit-equal parameters after updates, identical RNG streams).

### 3. Persistent activity traces (A.2)

A per-synapse eligibility buffer `E` (302×302, masked to the chemical edge set) on `ConnectomeTopology`: updated once per env step in the rollout forward under `torch.no_grad()` as `E ← λ·E + M_chem ∘ (h hᵀ)` (settled post-K hidden; `E[i,j]` aligned pre-`i` → post-`j` with `w_chem`), allocated only when enabled, reset at `prepare_episode`. The batched PPO re-forward never reads or mutates `E`, so no lstmppo-style replay buffering is needed and PPO gradients are untouched. Off by default (`enable_activity_traces: false`); until A.3's rule consumes `E`, training with traces **on** is also bit-identical to off.

### 4. Telemetry

`ConnectomePPORule.step` returns a `RuleStepReport`; the brain appends the mean **policy loss** to `history_data.losses` (the house PPO convention — lstmppo/cfc/spiking all record `avg_policy`, so the CSV column stays one quantity across brains). The connectome brain currently records no loss at all, so this is a declared **additive** tracking change with zero effect on training bits.

### 5. Tracker + docs

`phase7-tracking` tasks.md: shipment 7a-i status → 🟡 in progress; A.1/A.2 ticked at completion. Plugin-developer guide updated to the reconciled Protocols. CHANGELOG line under *Unreleased* for the new config fields.

## Capabilities

**Modified**: `brain-architecture` (Brain Topology Protocol + Learning Rule Protocol requirements reconciled to an implementable, consumed contract); `connectome-ppo-brain` (adds the activity-trace substrate requirement and the PPO-update-via-rule-seam requirement).

**Added**: none. (The `learning-rules` capability is added by the A.3 rule change, once there is a biologically-plausible rule to specify; this change only creates the package with the extracted PPO rule as its first citizen.)

## Impact

**Code**: `brain/arch/_topology.py`, `_rule.py` (Protocol reconciliation); `brain/arch/connectome_ppo.py` (trace buffer + reset hook + update delegation; config fields `enable_activity_traces`, `trace_decay`); new `quantumnematode/learning_rules/{__init__,ppo.py}`; no `_policy.py` changes (D7 pends against that module).

**Tests**: new `brain/arch/test_connectome_traces.py`, `test_connectome_rule_extraction.py` (+ frozen `_legacy_connectome_update_reference.py`, M1 pattern); `test_topology_rule_protocols.py` (the conformance test the L1 change promised but never shipped); a fresh-interpreter import-order test for the new package (Decision 3b); existing connectome suites must stay green unmodified (`brain.critic`/`brain.optimizer` survive as delegating properties).

**Docs**: `docs/architecture/plugin-developer-guide.md`; `openspec/changes/phase7-tracking/tasks.md`; `CHANGELOG.md`.

**Configs**: none — traces default off; the A.3 change adds the plastic arms.

## Breaking Changes

None. Byte-identical-when-off is the gating bar; with traces off (default), every existing config trains bit-identically.

## Backward Compatibility

The external `Brain` Protocol surface is unchanged (`prepare_episode` gains a body but keeps its signature). The only observable addition is the new `losses` telemetry column for connectome runs, which the CSV exporter already handles generically.
