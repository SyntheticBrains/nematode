# Tasks — minimal rate-based three-factor rule

## 1. Causal eligibility trace (Decision 2)

- [ ] 1.1 Add previous-settled-state storage to `ConnectomeTopology`, allocated only when traces are enabled (no state-dict or RNG footprint otherwise), cleared by `reset_traces()` alongside `E`.
- [ ] 1.2 Change the rollout-forward trace update to `E ← trace_decay · E + M_chem ∘ (h_prev ⊗ h)`, keeping the `torch.no_grad()` block, the mask seam, and the once-per-step call-site convention. First step of an episode contributes nothing.
- [ ] 1.3 Confirm the batched (PPO replay) forward still neither reads nor mutates `E` or the previous state.
- [ ] 1.4 Tests: closed-form recurrence under the causal formula; zero eligibility after a single step; reciprocal edge pair receives different eligibility in each direction; masked support; reset lifecycle; traces-off construction byte-identical.
- [ ] 1.5 Re-verify inertness: PPO training with traces on and off remains bit-identical (the formula change touches the path that guarantees it). Update the existing traces-on/off equivalence test rather than adding a second one.

## 2. The rule (Decisions 4–6)

- [ ] 2.1 New `learning_rules/three_factor.py`: `ConnectomeThreeFactorRule` satisfying `LearningRule`; owns plasticity rate, decay, magnitude bound, baseline rate; no optimiser, no critic, no gradients.
- [ ] 2.2 Batch dataclass carrying the per-step reward, mirroring how `ConnectomePPOBatch` surfaces brain-owned experience.
- [ ] 2.3 Update: `Δw = η·δ·E − η·λ_w·w`, `δ = r − b`, EMA baseline; whole step under `torch.no_grad()`; projected through `apply_weight_mask`.
- [ ] 2.4 Magnitude clamp. No sign constraint: synapse signs are arbitrary draws in this substrate, so a Dale's-law clamp would freeze noise and block correction of a wrongly-signed synapse.
- [ ] 2.5 `reset_episode()` clears per-episode rule state; the baseline persists across episodes (a running estimate of the task's reward level, not one episode's) — pinned by a test, since resetting it would make every episode's opening steps read as surprising.
- [ ] 2.6 `RuleStepReport.extra` carries prediction error, baseline, mean |Δw|, saturated fraction.
- [ ] 2.7 Export from the package `__init__`.

## 3. Brain integration (Decision 1, Decision 5)

- [ ] 3.1 `learning_rule` config field defaulting to PPO; validator rejecting the three-factor selection without `enable_activity_traces`, naming both fields.
- [ ] 3.2 Construct the selected rule; keep RNG draw order unchanged so every parameter **other than the readout** stays byte-identical across selections (the readout differs by design — task 3b.1).
- [ ] 3.3 Per-step dispatch in `learn()` under the three-factor rule; rollout buffer and GAE unexercised.
- [ ] 3.3a **Skip the per-step value computation** under the plastic rule: `run_brain` currently calls `self.critic(hidden)` unconditionally on every step via a property delegating to the PPO rule, so a rule owning no critic would fail on the first action. Leave the per-step and bootstrap value state unset and do not append to the experience buffer. Skip rather than stub — a null value head would keep a dead tensor in the action path and make "no value head" true only by wording.
- [ ] 3.3b Make the `critic` / `optimizer` back-compat accessors raise an error naming the active rule under the plastic selection, instead of surfacing an attribute error from inside the delegation. Four existing tests read `brain.critic`; all construct PPO brains, so they stay green.
- [ ] 3.4 Route the new telemetry into `history_data` alongside the existing loss/monitor series.
- [ ] 3.5 Tests: default byte-identity against a frozen reference; validator; per-step cadence; PPO machinery dormant (buffer stays empty, no value computed); action selection succeeds with no value head; accessors fail informatively; construction identical across selections apart from the readout; only `w_chem` changes.

## 3b. Anatomical motor readout (Decision 7)

- [ ] 3b.1 Under the three-factor rule, write the readout as the anatomical contrast (turn ← dorsal − ventral, speed ← B-type − A-type) **after** the existing orthogonal draw, so no extra randomness is consumed and the RNG stream is unperturbed.
- [ ] 3b.2 Derive the contrast from the motor-class index structure already built at construction rather than from hard-coded column positions, so it cannot silently desynchronise if the class order changes.
- [ ] 3b.3 Tests: rows encode the two contrasts; rows are unit-norm and mutually orthogonal; PPO readout byte-identical to pre-change; all non-readout parameters byte-identical across rule selections; wild-type and rewired-null readouts identical at the same seed.

## 4. Evidence the rule is what it claims

- [ ] 4.1 Test: update equals `η·δ·E` before stabilisation, and vanishes when either `δ` or `E` is zero.
- [ ] 4.2 Test: under a constant reward the per-step weight change tends to zero as the baseline converges; an unexpected reward moves weights more than an expected one.
- [ ] 4.3 Test: no chemical-weight tensor acquires a gradient; the update succeeds with autograd globally disabled.
- [ ] 4.4 Test: sustained drive leaves every weight within the bound; zero prediction error is non-increasing in magnitude; a synapse driven through zero is not clamped on account of the sign change.
- [ ] 4.5 Test: sensory gains, readout, and action-noise parameters are bit-identical after a step.
- [ ] 4.6 Test pinning the trace/reward alignment: the reward gating an update credits the trace **including** the step whose action earned it, not only eligibility accrued strictly before the action.

## 5. Config and documentation

- [ ] 5.1 One plastic wild-type smoke config (three-factor + traces on), derived from the C3 connectome cell by the minimal key delta; the panel's remaining arms are A.4–A.6.
- [ ] 5.2 Config-variant test pinning the smoke config as a minimal delta from its parent.
- [ ] 5.3 `docs/architectures.md` note that the connectome brain supports two update regimes; `configs/README.md` variant vocabulary.
- [ ] 5.4 CHANGELOG entry.
- [ ] 5.5 Tracker: tick A.3 with a dated note; record the `trace_decay` calibration hand-off and cross-reference the filed class-naming issue.
- [ ] 5.6 Record two findings for the arm changes that inherit them: (a) roadmap D2 pins the frozen-weights baseline as "Cook-2019 synapse-count-derived initial weights", but the implementation derives *edge existence* from the connectome and draws weights from a zero-mean distribution scaled by chemical in-degree — the arm change should either correct the wording or change the initialisation deliberately; (b) the connectome brain implements no weight-component persistence at all, which the frozen-weights arm will need.

## 6. Close-out

- [ ] 6.1 `git add -A` **first**, then `uv run pre-commit run --all-files` green (hooks skip untracked files, so staging is what makes new files visible).
- [ ] 6.2 Full non-nightly suite green.
- [ ] 6.3 No implementation code or docstring references planning docs, roadmap IDs, OpenSpec changes, or logbooks; rationale stated intrinsically.
- [ ] 6.4 Re-review for drift, archive, review the branch, open the PR.
