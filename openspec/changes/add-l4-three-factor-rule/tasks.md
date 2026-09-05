# Tasks — minimal rate-based three-factor rule

## 1. Causal eligibility trace (Decision 2)

- [ ] 1.1 Add previous-settled-state storage to `ConnectomeTopology`, allocated only when traces are enabled (no state-dict or RNG footprint otherwise), cleared by `reset_traces()` alongside `E`.
- [ ] 1.2 Change the rollout-forward trace update to `E ← trace_decay · E + M_chem ∘ (h_prev ⊗ h)`, keeping the `torch.no_grad()` block, the mask seam, and the once-per-step call-site convention. First step of an episode contributes nothing.
- [ ] 1.3 Confirm the batched (PPO replay) forward still neither reads nor mutates `E` or the previous state.
- [ ] 1.4 Tests: closed-form recurrence under the causal formula; zero eligibility after a single step; reciprocal edge pair receives different eligibility in each direction; masked support; reset lifecycle; traces-off construction byte-identical.
- [ ] 1.5 Re-verify inertness: PPO training with traces on and off remains bit-identical (the formula change touches the path that guarantees it).

## 2. The rule (Decisions 4–6)

- [ ] 2.1 New `learning_rules/three_factor.py`: `ConnectomeThreeFactorRule` satisfying `LearningRule`; owns plasticity rate, decay, magnitude bound, baseline rate; no optimiser, no critic, no gradients.
- [ ] 2.2 Batch dataclass carrying the per-step reward, mirroring how `ConnectomePPOBatch` surfaces brain-owned experience.
- [ ] 2.3 Update: `Δw = η·δ·E − η·λ_w·w`, `δ = r − b`, EMA baseline; whole step under `torch.no_grad()`; projected through `apply_weight_mask`.
- [ ] 2.4 Magnitude clamp and sign preservation; count sign-clamp events.
- [ ] 2.5 `reset_episode()` clears per-episode rule state; the baseline persists across episodes (it is a running estimate of the task's reward level, not of one episode's).
- [ ] 2.6 `RuleStepReport.extra` carries prediction error, baseline, mean |Δw|, saturated fraction, sign-clamp count.
- [ ] 2.7 Export from the package `__init__`.

## 3. Brain integration (Decision 1, Decision 5)

- [ ] 3.1 `learning_rule` config field defaulting to PPO; validator rejecting the three-factor selection without `enable_activity_traces`, naming both fields.
- [ ] 3.2 Construct the selected rule; keep RNG draw order unchanged so construction stays byte-identical across selections.
- [ ] 3.3 Per-step dispatch in `learn()` under the three-factor rule; rollout buffer, GAE, and value head unexercised.
- [ ] 3.4 Route the new telemetry into `history_data` alongside the existing loss/monitor series.
- [ ] 3.5 Tests: default byte-identity against a frozen reference; validator; per-step cadence; PPO machinery dormant; construction identical across selections; only `w_chem` changes.

## 4. Evidence the rule is what it claims

- [ ] 4.1 Test: update equals `η·δ·E` before stabilisation, and vanishes when either `δ` or `E` is zero.
- [ ] 4.2 Test: under a constant reward the per-step weight change tends to zero as the baseline converges; an unexpected reward moves weights more than an expected one.
- [ ] 4.3 Test: no chemical-weight tensor acquires a gradient; the update succeeds with autograd globally disabled.
- [ ] 4.4 Test: sustained drive leaves every weight within the bound; zero prediction error is non-increasing in magnitude; signs never invert.
- [ ] 4.5 Test: sensory gains, readout, and action-noise parameters are bit-identical after a step.

## 5. Config and documentation

- [ ] 5.1 One plastic wild-type smoke config (three-factor + traces on), derived from the C3 connectome cell by the minimal key delta; the panel's remaining arms are A.4–A.6.
- [ ] 5.2 Config-variant test pinning the smoke config as a minimal delta from its parent.
- [ ] 5.3 `docs/architectures.md` note that the connectome brain supports two update regimes; `configs/README.md` variant vocabulary.
- [ ] 5.4 CHANGELOG entry.
- [ ] 5.5 Tracker: tick A.3 with a dated note; record the class-naming debt and the `trace_decay` calibration hand-off.

## 6. Close-out

- [ ] 6.1 `git add -A` **first**, then `uv run pre-commit run --all-files` green (hooks skip untracked files, so staging is what makes new files visible).
- [ ] 6.2 Full non-nightly suite green.
- [ ] 6.3 No implementation code or docstring references planning docs, roadmap IDs, OpenSpec changes, or logbooks; rationale stated intrinsically.
- [ ] 6.4 Re-review for drift, archive, review the branch, open the PR.
