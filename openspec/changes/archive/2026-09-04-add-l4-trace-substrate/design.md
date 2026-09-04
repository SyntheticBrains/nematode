# Design — L4 trace substrate (rule seam + persistent activity traces)

## Context

First 7a-i milestone under `phase7-tracking` (tasks A.1/A.2; execution-protocol standards in that change's design.md § Decision B apply — byte-identical-when-off, load-time config validation, no substrate change after the freeze). Reconnaissance facts this design rests on, verified against source:

- `LearningRule` / `RuleStepReport` / `BrainTopology` have zero code consumers; `isinstance(ConnectomeTopology(...), BrainTopology)` is `False` (missing `n_inputs`/`n_outputs`/`n_hidden`; incompatible `forward` signature). The promised `test_topology_rule_protocols.py` was never shipped.
- The connectome PPO update is inlined at `connectome_ppo.py:1404-1493`: epochs × `buffer.get_minibatches(...)` → `_unpack_state_batched` → `forward_with_hidden_batched` → critic → `_policy.py` helpers (`categorical_evaluate_torch` / `continuous_evaluate_tanh_gaussian`, `ppo_clip_policy_loss`) → Adam step → strict-mask re-projection (`:1489-1493`). It records no loss telemetry.
- `forward_with_hidden` zero-initialises `h` per env step (`:709`); the batched twin zero-initialises per minibatch row (`:831`) — the PPO replay path is stateless by construction.
- `prepare_episode()` on the connectome brain is a no-op (`:1391`); the runner calls it at every episode start (`agent/runners.py:958`), and `learn(..., episode_done=True)` fires **before** `post_process_episode` at episode end.
- House byte-equivalence precedents: M1 frozen-reference (`env/test_predator_brain_byte_equivalence.py` + `_legacy_predator_reference.py`), the wiring control's `TestWiringControl` byte-identical tests, and the policy-migration in-process-reference pattern (no golden float constants — they drift ~1e-8 across BLAS builds).
- Opt-in blocks in `ConnectomeTopology.__init__` are constructed **after** the always-on parameters so RNG-stream consumption order is preserved when flags flip (`:180-184`, `:453-457`, `:534-538`) — the trace buffer must follow the same discipline (trivially: it consumes no RNG).

## Goals

- Give `LearningRule` its first genuine consumer and `BrainTopology` a contract that its focal implementation actually satisfies, so A.3's three-factor rule is a second rule behind an existing seam, not a fork.
- Land cross-step eligibility state on the connectome topology, byte-identical-when-off — and bit-inert even when on, until a rule consumes it.

## Non-Goals

- The minimal three-factor rule itself (A.3, next change — it adds the `learning-rules` capability spec).
- Migrating any other brain onto the seam (the L1 factoring stays single-consumer by design until a second rule exists).
- Changing hidden-state semantics: `h` stays within-step settling. Persisting `h` across steps would change the 2×2 substrate mid-programme and confound the D10 panel against Logbooks 029/034.
- Weight persistence for the connectome brain (it has none today; explicitly out of scope).
- `extra="forbid"` on `BrainConfig` (would affect all 26 brains; the existing `_warn_unknown_brain_config_keys` path covers the new fields once they are model fields).
- Any `_policy.py` restructuring (roadmap D7 pends against that module).

## Decisions

### Decision 1 — Seam-first extraction, not a bespoke plastic brain

A.1 resolves to: extract the PPO update through the `LearningRule` seam on the **existing** brain. Rejected alternatives: (a) a bespoke `ConnectomePlasticBrain` — duplicates ~1,500 lines, defers the seam debt to A.3, and forks the exact code paths the D10 panel must hold constant; (b) implementing the Protocols as currently written — they mandate attributes (`n_inputs` etc.) and a `forward(x)` signature that no topology in the repo has or needs. Precedent: the wiring control's D1 ("config options on the existing brain, not a new brain class").

### Decision 2 — Reconcile the Protocols to the seam a rule actually needs

`BrainTopology` becomes: `apply_weight_mask(weights)` + `learnable_parameters` (a **property**, matching the implementation, so static conformance holds too), dropping shape attributes and `forward` from the Protocol. Forward stays out because signatures are topology-specific — *not* because rules never forward: PPO must re-forward under current weights once per minibatch per epoch, and it does so against the **concrete** `ConnectomeTopology` (`forward_with_hidden_batched`), a surface beyond the Protocol (see Decision 3). `ConnectomeTopology` then passes `runtime_checkable` `isinstance`. `LearningRule` keeps `step(topology, batch) -> RuleStepReport` + `reset_episode()`; its ownership docstring is softened: the rule owns optimiser / value head / hyperparameters, while an experience-collection buffer owned by the brain MAY be surfaced through `batch`. The `brain-architecture` spec deltas carry the rewritten requirements; the plugin guide's conformance claims become true and are tested (`test_topology_rule_protocols.py`, finally).

### Decision 3 — Ownership split: rule owns the update, brain owns experience

`ConnectomePPORule` (in `learning_rules/ppo.py`, per D8) owns: the Adam optimiser, the critic, the PPO hyperparameters — **including `gamma`/`gae_lambda`, so the rule computes returns and advantages itself** (the brain cannot precompute GAE without violating the ownership split) — and the epoch × minibatch loop. Constructor inputs (from the config, at brain-`__init__` time): the continuous flag, action bounds (`_action_low`/`_action_high`), `chemical_mask_mode`, and device (for the `last_value` fallback tensor). The brain retains: `RolloutBuffer` + pending-tuple collection (including `buffer.reset()` in `learn()`), feature unpacking, and `last_value`. `batch` exposes exactly what the inlined code consumed — the **buffer handle** (the rule calls `get_minibatches` once **per epoch**, preserving the fresh per-epoch permutation draws the byte-equivalence bar protects), `_unpack_state_batched` bound as a callable, and `last_value` — so every tensor op moves **verbatim** and the bit-order of the update is unchanged. The three update-path `_policy.py` helpers (`categorical_evaluate_torch`, `continuous_evaluate_tanh_gaussian`, `ppo_clip_policy_loss`) are called with identical arguments in identical order (the rollout-side samplers stay on the brain); the strict-mask projection stays at the same point (post-optimiser, per update). `brain.critic` / `brain.optimizer` remain as thin delegating properties — one existing test reads `brain.critic` (`test_connectome_ppo_continuous.py:120`) and the suites must stay green unmodified.

### Decision 3b — Import discipline (cycle avoidance)

`brain/arch/__init__.py` imports `connectome_ppo` at package load. A module-level `from quantumnematode.learning_rules.ppo import ConnectomePPORule` in `connectome_ppo.py` therefore creates a genuine cycle whenever `learning_rules.ppo` is the *entry point* (exactly what the new tests do): `learning_rules.ppo` → `brain.arch.__init__` (fresh, runs fully) → `connectome_ppo` → import from a **partially-initialised** `learning_rules.ppo` → `ImportError`. Discipline, pinned: (a) `connectome_ppo.py` imports the rule **lazily inside `ConnectomePPOBrain.__init__`**; (b) `learning_rules/ppo.py` imports only **leaf modules** (`quantumnematode.brain.arch._policy`, `._rule`, `._ppo_buffer`), never the `brain.arch` package; (c) a fresh-interpreter test (subprocess) imports `quantumnematode.learning_rules.ppo` first and asserts success.

### Decision 4 — Trace semantics: masked co-activity eligibility, rollout-side only

`E` is a `(302, 302)` buffer on `ConnectomeTopology`, allocated **only when** `enable_activity_traces` is true (no state-dict or RNG footprint when off). Once per env step, inside `forward_with_hidden` under `torch.no_grad()`:

```text
E ← trace_decay · E + M_chem ∘ (h hᵀ)        # h = settled post-K hidden
```

with `E[i, j]` aligned pre-`i` → post-`j`, matching `w_chem`'s edge orientation (`(WᵀM h)_j = Σ_i W[i,j] h_i`). This is the rate-based co-activity accumulator the A.3 three-factor rule multiplies by its neuromodulator signal — no spike times, no BPTT. **The rollout forward runs grad-enabled** (`run_brain` wraps nothing in `no_grad`; detachment happens later in `RolloutBuffer.add`), so the `torch.no_grad()` block around the trace update is **load-bearing** — it is what prevents an autograd leak into `E`; do not drop it. **The batched PPO forward never reads or mutates `E`**: traces are rollout-time observables, so none of the lstmppo per-step-hidden replay machinery is needed, PPO gradients are untouched by construction, and byte-identity holds trivially. The update lives in `forward_with_hidden`, so "once per env step" is a call-site convention — documented on the method; the diagnostic `forward()` shim would also update `E` if traces were on, which no diagnostic path enables.

**The v1 formula is amendable; two structural limitations are recorded for A.3** (definitional choices, not calibration): (i) `h hᵀ` is symmetric, so `E` carries no pre→post directional information beyond the mask — reciprocal edges always receive equal eligibility; (ii) the sensory injection only seeds the initial `h` and is overwritten by the K settling iterations, so eligibility on sensory-neuron outgoing edges does not reflect the stimulus that drove the step. Alternatives A.3 may adopt with a dated spec amendment: pre = the *previous* step's settled `h` (directional, temporal); pre = the pre-recurrence injected state (stimulus-aware); rectification of tanh-negative rates. The SHALL-protected invariants (masking, `no_grad`, lifecycle, inertness) are formula-independent. `trace_decay` defaults to `0.9` as an inert placeholder, pydantic-bounded to `[0, 1)` (a decay ≥ 1 is a divergent accumulator); its biological calibration is A.3's problem and is explicitly not claimed here.

### Decision 5 — Trace lifecycle pinned to the existing episode hooks

`prepare_episode()` (currently a no-op) gains: `topology.reset_traces()` + `rule.reset_episode()`. Consequence, stated rather than left implicit: traces **persist through the terminal `learn(..., episode_done=True)` call** (which fires before `post_process_episode`) and reset at the *next* episode's start — matching the runner's hook ordering. A.3 may revisit whether the terminal update should see traces; this change just makes today's ordering explicit and tested.

### Decision 6 — Byte-equivalence bar for the extraction

M1 pattern: a frozen copy of the pre-change `_perform_ppo_update` lives in the test tree (`_legacy_connectome_update_reference.py`), **copied from the pre-change tree (this branch's merge-base)** and committed *before* the delegation commit — which is what makes "suite written before the extraction lands" enforceable inside one PR. It is driven against a deep-copied brain state with identical buffer contents and pinned seeds; assert **bit-equal** (`torch.equal`) parameters after the update and identical RNG bit-generator state. No golden float constants (policy-migration precedent — they drift across BLAS builds). Plus the two inert-mechanism invariants: traces **off** ⇒ construction byte-identical to pre-change (034 `TestWiringControl` template); traces **on** ⇒ post-update parameters bit-equal to traces-off at the same seed (nothing consumes `E` yet).

### Decision 7 — Telemetry is additive and declared

`step` returns a `RuleStepReport` aggregating means across the epoch × minibatch iterations; the brain appends `report.policy_loss` to `history_data.losses` — the **house PPO convention** (lstmppo, cfc, spiking, reservoir, equivariant all record `avg_policy`), so a cross-brain read of the `losses` CSV column stays a single quantity. On the `freeze_updates` / empty-buffer short-circuits the rule still returns a `RuleStepReport` (loss fields `None`) and the brain `None`-guards before appending. The connectome brain currently records nothing, so connectome runs gain a `losses` CSV column via the exporter's generic key-union — an observable but training-inert change, called out in the CHANGELOG.

## Risks / open questions

- **Extraction touch-surface**: `_perform_ppo_update` reads seven brain attributes. Mitigation: the rule is constructed inside `ConnectomePPOBrain.__init__` from the config (no `_build_infra_kwargs` change; the brain-factory seam stays untouched), and the byte-equivalence suite is written **before** the extraction lands.
- **Protocol softening ripple**: loosening `BrainTopology` is a spec change to `brain-architecture`; the delta rewrites the two requirements in full and the L1 archive change is unaffected (archived changes are historical record).
- **`copy()` remains `NotImplementedError`** on the connectome brain; the rule object makes a future implementation slightly larger. Recorded, not solved here.
- **Trace memory**: one extra `(302, 302)` float32 buffer (~365 KB) when enabled — negligible.
