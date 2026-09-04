# Tasks — L4 trace substrate (rule seam + persistent activity traces)

Scope: `phase7-tracking` A.1 (rule-seam decision + extraction) and A.2 (persistent activity
traces), byte-identical-when-off. The minimal three-factor rule (A.3) and the D10 panel arms are
**not** in this change.

## 1. Protocol reconciliation

- [ ] 1.1 `_topology.py`: `BrainTopology` = `apply_weight_mask(weights)` + `learnable_parameters`
  (list of `nn.Parameter`); drop `n_inputs`/`n_outputs`/`n_hidden` and `forward` from the Protocol;
  update docstrings to the Decision-2 contract.
- [ ] 1.2 `_rule.py`: keep `step(topology, batch) -> RuleStepReport` + `reset_episode()`; soften the
  ownership docstring (brain-owned experience buffer may arrive via `batch`); no signature changes.
- [ ] 1.3 New `tests/.../brain/arch/test_topology_rule_protocols.py`: `isinstance(topology,
  BrainTopology)` and `isinstance(rule, LearningRule)` both pass for the connectome pair (the
  conformance test the L1 change promised at `archive/2026-05-24-.../tasks.md:85` but never shipped).
- [ ] 1.4 Update `docs/architecture/plugin-developer-guide.md` §§ topology/rule (~L234-301) so the
  conformance and consumer claims are true as written.

## 2. `learning_rules/` package + PPO-rule extraction

- [ ] 2.1 Create `quantumnematode/learning_rules/{__init__.py, ppo.py}` (D8 — do NOT touch
  `quantumnematode/plasticity/`, the quantum-plasticity eval protocol).
- [ ] 2.2 `ConnectomePPORule`: owns Adam optimiser, critic (`nn.Linear(n_neurons, 1)`, orthogonal
  init at the same construction point so RNG streams are unchanged), PPO hyperparameters, and the
  epoch × minibatch loop; `step(topology, batch)` executes the ops of `_perform_ppo_update`
  **verbatim** — same six `_policy.py` symbols, same argument order, `freeze_updates` /
  empty-buffer short-circuits preserved, strict-mask projection at the same post-optimiser point.
- [ ] 2.3 `batch` object: buffer minibatch iterator (same RNG stream + draw order) + the brain's
  `_unpack_state_batched` bound as a callable + `last_value` fallback — exactly what the inline code
  consumed, nothing more.
- [ ] 2.4 Brain delegation: `ConnectomePPOBrain` constructs the rule in `__init__` (no
  `_build_infra_kwargs` change), `learn()` calls `rule.step(...)` where it called
  `_perform_ppo_update()`; `optimizer`/`critic` attributes move onto the rule (grep for external
  readers first; tests use `_drive_one_ppo_update`, not the attributes).

## 3. Byte-equivalence suite (written BEFORE the extraction lands)

- [ ] 3.1 Freeze the pre-change update as `tests/.../brain/arch/_legacy_connectome_update_reference.py`
  (M1 pattern), operating on a deep-copied brain state.
- [ ] 3.2 `test_connectome_rule_extraction.py`: identical seeds + identical buffer contents ⇒
  `torch.equal` on every learnable parameter and critic weight after the update, and identical
  torch RNG state; parametrised over discrete/continuous heads and strict/soft-prior mask modes.
  No golden float constants (policy-migration precedent).
- [ ] 3.3 Existing suites green **unmodified**: `test_connectome_ppo.py`,
  `test_connectome_ppo_continuous.py`, `test_connectome_vectorisation.py`, projection tests,
  `TestWiringControl`, `TestFrozenUpdates`.

## 4. Trace substrate (A.2)

- [ ] 4.1 Config fields on `ConnectomePPOBrainConfig`: `enable_activity_traces: bool = False`,
  `trace_decay: float = 0.9` (inert placeholder; biological calibration is A.3's). Fields on the
  model ⇒ the `_warn_unknown_brain_config_keys` load-time path covers them.
- [ ] 4.2 `ConnectomeTopology`: conditionally-allocated `(n_neurons, n_neurons)` float32 buffer `E`
  (zeros; allocated only when enabled; consumes no RNG; constructed after all RNG-consuming
  blocks per the in-file discipline at L180-184/453-457/534-538); `reset_traces()` method.
- [ ] 4.3 Trace update in `forward_with_hidden` under `torch.no_grad()`, once per env step:
  `E ← trace_decay·E + m_chem ∘ (h hᵀ)` with `E[i,j]` pre-`i`→post-`j` aligned to `w_chem`; the
  **batched** forward does not touch `E`.
- [ ] 4.4 `prepare_episode()`: `topology.reset_traces()` + `rule.reset_episode()` (was a no-op);
  terminal-`learn`-sees-old-traces ordering documented in the method docstring.
- [ ] 4.5 `test_connectome_traces.py`: off ⇒ constructed tensors byte-identical to default (034
  `TestWiringControl` template) and no `E` attribute allocated; on ⇒ decay recurrence matches the
  closed form over a scripted step sequence; masked (E is zero off-edges); reset at
  `prepare_episode`; **training-bit-invariance** — same seed, traces on vs off ⇒ `torch.equal`
  parameters after `_drive_one_ppo_update`; deterministic across identical runs; batched update
  leaves `E` unchanged.

## 5. Telemetry

- [ ] 5.1 `step` returns `RuleStepReport` (means of `policy_loss`, `value_loss`, `entropy`,
  `total_loss`, `grad_norm` across the epoch × minibatch iterations); brain appends
  `report.total_loss` to `history_data.losses`.
- [ ] 5.2 Test: after one update, `history_data.losses` is non-empty and finite; with
  `freeze_updates` it stays empty.

## 6. Docs + tracker

- [ ] 6.1 `CHANGELOG.md` *Unreleased*: new connectome config fields + the new `losses` telemetry
  column for connectome runs.
- [ ] 6.2 `openspec/changes/phase7-tracking/tasks.md`: tick A.1 + A.2 with the shipment status
  header updated (done in this change's final PR state).

## 7. Pre-merge gates

- [ ] 7.1 `openspec validate add-l4-trace-substrate --strict` passes.
- [ ] 7.2 Full test suite green (`uv run pytest -m "not nightly"`).
- [ ] 7.3 Grep gate: no remaining reference to `_perform_ppo_update` outside the frozen test
  reference; no import of `quantumnematode.plasticity` from `learning_rules/`.
