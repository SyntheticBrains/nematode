# Tasks — matched-rule MLP arm

## 1. The plastic-topology seam (Decision 1)

- [ ] 1.1 Add a `PlasticTopology` Protocol beside `BrainTopology`: aligned lists of plastic weight tensors, eligibility traces, and edge masks; the traces-enabled flag; the mask projector applied per tensor. Dense substrates expose an all-true mask.
- [ ] 1.2 Expose the seam on `ConnectomeTopology` as views over `w_chem`, `m_chem`, and `activity_traces` — no new state, no change to the trace update.
- [ ] 1.3 Rewrite the rule to read the seam, iterating the aligned lists; aggregate telemetry across every plastic entry (mean absolute change) and every masked entry (saturated fraction). Rename to reflect generality; keep the connectome-named class as an alias.
- [ ] 1.4 Tests: the connectome path is bit-identical before and after the rewrite (frozen-reference pattern — parameters equal after identical update sequences); `isinstance` conformance for both topologies; alias resolves to the same class.

## 2. Shared plasticity configuration (Decision 5)

- [ ] 2.1 Move `learning_rule`, the four `plasticity_*` fields, `enable_activity_traces`, `trace_decay`, the trace-pairing validator, and the plastic/unmodulated rule sets to a mixin both `ConnectomePPOBrainConfig` and `MLPPPOBrainConfig` inherit.
- [ ] 2.2 Tests: both configs report identical plasticity defaults; the pairing validator fires on both; the connectome config's field set is unchanged.

## 3. MLP topology and traces (Decisions 2, 3)

- [ ] 3.1 `MLPTopology` wrapping the actor's `Linear` layers by reference; one `(out, in)` trace per layer registered on the topology, allocated only when traces are enabled; `reset_traces()`.
- [ ] 3.2 Traced forward: same modules, same order as `self.actor`, recording `pre_l`/`post_l` and updating traces under `torch.no_grad()`, once per environment step. Post is the post-nonlinearity output where one exists.
- [ ] 3.3 Tests: traced forward bitwise-equal to `self.actor(features)`; closed-form recurrence per layer; trace shape equals weight shape per layer; traces-off allocates nothing; `self.actor` is the same object and its `state_dict` keys are unchanged; reset lifecycle.

## 4. MLP brain integration (Decisions 4, 6)

- [ ] 4.1 `learning_rule` selection on the MLP brain; construct the generic rule under a plastic selection; critic and optimiser still constructed, never used.
- [ ] 4.2 Skip the critic call on both the discrete and continuous action paths under a plastic rule; route the forward through the traced path; leave the per-step and bootstrap value unset; do not append to the buffer.
- [ ] 4.3 Per-step dispatch in `learn()` under a plastic rule; `prepare_episode()` resets traces and rule state.
- [ ] 4.4 Plastic set: every `Linear` weight; biases, `log_std`, gate weights, and the critic untouched.
- [ ] 4.5 Route plasticity telemetry into `history_data` as on the connectome.
- [ ] 4.6 Tests: PPO path byte-identical to a frozen reference (construction and training); per-step cadence; buffer empty and value unset under the plastic rule; only `Linear` weights change; biases and `log_std` bit-identical after training; unmodulated and frozen selections behave as on the connectome; the initial-weight bound check across seeds.

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
