# Tasks — matched-rule MLP arm

## 1. The plastic-topology seam (Decision 1)

- [ ] 1.1 Add a `PlasticTopology` Protocol beside `BrainTopology`: aligned lists of plastic weight tensors, eligibility traces, and edge masks; the traces-enabled flag; the mask projector applied per tensor. Dense substrates expose an all-true mask.
- [ ] 1.2 Expose the seam on `ConnectomeTopology` as views over `w_chem`, `m_chem`, and `activity_traces` — no new state, no change to the trace update.
- [ ] 1.3 Rewrite the rule to read the seam, iterating the aligned lists; aggregate telemetry across every plastic entry (mean absolute change) and every masked entry (saturated fraction). Rename to reflect generality; keep the connectome-named class as an alias, and export **both** names from `learning_rules/__init__.py` — the alias is what existing tests and the connectome brain import.
- [ ] 1.4 Tests: the connectome path is bit-identical before and after the rewrite (frozen-reference pattern — parameters equal after identical update sequences); `isinstance` conformance for both topologies; alias resolves to the same class.

## 2. Shared plasticity configuration (Decision 5)

- [ ] 2.1 Move `learning_rule`, the four `plasticity_*` fields, `enable_activity_traces`, `trace_decay`, the trace-pairing validator, and the `_PLASTIC_RULES` / `_UNMODULATED_RULES` sets to a mixin module both `ConnectomePPOBrainConfig` and `MLPPPOBrainConfig` inherit; update the connectome brain's imports of the two sets. (Composition probed before authoring: a `BaseModel` mixin with fields and an after-validator composes with `BrainConfig`, both validators fire, and the loader compares field *sets*, so the reorder is inert.)
- [ ] 2.3 **Freeze the MLP reference before any change to `mlpppo.py`.** Capture, from the untouched brain at a fixed seed, the parameters after a scripted sequence of PPO updates on both the discrete and continuous paths, into a frozen reference module (the connectome's `_legacy_*` pattern). This task MUST complete before section 3 begins: a reference captured after the refactor passes by construction and proves nothing.
- [ ] 2.2 Tests: both configs report identical plasticity defaults; the pairing validator fires on both; the connectome config's field set is unchanged.

## 3. MLP topology and traces (Decisions 2, 3)

- [ ] 3.1 `MLPTopology` wrapping the actor's `Linear` layers by reference; one `(out, in)` trace per layer registered on the topology, allocated only when traces are enabled; `reset_traces()`.
- [ ] 3.2 Traced forward: same modules, same order as `self.actor`, recording `pre_l`/`post_l` and updating traces under `torch.no_grad()`, once per environment step. Post is the post-nonlinearity output where one exists.
- [ ] 3.3 Tests: traced forward bitwise-equal to `self.actor(features)`; closed-form recurrence per layer; trace shape equals weight shape per layer; traces-off allocates nothing; `self.actor` is the same object and its `state_dict` keys are unchanged; reset lifecycle; **after `copy.deepcopy` of a plastic brain, `topology.layers[i] is brain.actor[j]` still holds** (`copy()` raises and points users at deepcopy — this is the one place wrap-by-reference could silently split).

## 4. MLP brain integration (Decisions 4, 6)

- [ ] 4.1 `learning_rule` selection on the MLP brain; construct the generic rule under a plastic selection; critic and optimiser still constructed, never used.
- [ ] 4.2 Skip the critic call on both the discrete and continuous action paths under a plastic rule; leave the per-step and bootstrap value unset; do not append to the buffer.
- [ ] 4.2a **Exactly one traced forward per environment step.** The discrete path forwards the actor twice per step — once in `get_action_and_value` to select the action, and again in `run_brain` to record probabilities. Only the action-selecting call updates traces; the probability call MUST use the untraced actor (or the already-computed logits). Routing both through the traced path would accrue eligibility twice per step and break the once-per-step invariant the rule's alignment semantics rest on. The continuous path forwards once and needs no special handling.
- [ ] 4.3 Per-step dispatch in `learn()` under a plastic rule; `prepare_episode()` resets traces and rule state.
- [ ] 4.4 Plastic set: every `Linear` weight; biases, `log_std`, gate weights, and the critic untouched.
- [ ] 4.5 Route plasticity telemetry into `history_data` through a shared helper used by both brains, so the four keys and their meaning cannot drift between arms.
- [ ] 4.6 Tests: PPO path byte-identical to the reference frozen in 2.3 (construction and training, both action paths); per-step cadence; buffer empty and value unset under the plastic rule; **one discrete step accrues exactly one outer product per layer** (traces after a single step equal `post ⊗ pre` from the action-selecting forward, not twice that); only `Linear` weights change; biases and `log_std` bit-identical after training; unmodulated and frozen selections behave as on the connectome; the initial-weight bound check across seeds.

## 5. Matched-arm evidence

- [ ] 5.1 Test: the same rule object type, the same update arithmetic, and the same hyperparameter values drive both substrates — a scripted trace and reward produce the update `η·δ·E` on an MLP layer exactly as on `w_chem`.
- [ ] 5.2 Test: telemetry keys and semantics are identical across the two arms.

## 6. Configuration and coverage

- [ ] 6.1 Plastic MLP config derived from the C3 MLP cell by the minimal key delta, named so the parent stays a prefix.
- [ ] 6.2 Minimal-delta and load tests; smoke entry.

## 7. Documentation

- [ ] 7.1 `docs/architectures.md`: the MLP's plastic mode and its role as the yardstick, including the deliberate output-layer asymmetry.
- [ ] 7.2 `configs/README.md` variant vocabulary; CHANGELOG entry.
- [ ] 7.3 Tracker: tick A.5 with a dated note recording the ratified scope decision.

## 8. Close-out

- [ ] 8.1 `git add -A`, then `uv run pre-commit run --all-files` **unfiltered**, exit code 0.
- [ ] 8.2 Full non-nightly suite green.
- [ ] 8.3 No planning references in implementation code or docstrings.
- [ ] 8.4 Re-review for drift, archive, review the branch, open the PR.
