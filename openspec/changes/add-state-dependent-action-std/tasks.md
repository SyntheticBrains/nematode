# Tasks — state-dependent action std (D7) + pre-panel platform freeze

Scope: `phase7-tracking` P.1–P.5. The L4 panel itself is NOT in this change; this change ends with
the substrate declared frozen.

## 1. P.1 verification (D5 — no code)

- [ ] 1.1 Verify-and-record: zero `normalize_advantages` keys in `configs/` (was 60+ per #254),
  zero lstmppo renamed-key residue (`chunk_length`/`hidden_dim`); tick `phase7-tracking` P.1 with a
  dated already-done note (shipped in the Phase 6 bit-memory pre-work). Advantage normalization
  stays unimplemented (closed-issue constraint; 029 raw-GAE lineage).

## 2. Config field + std heads (D7 mechanism)

- [ ] 2.1 `continuous_std_mode: Literal["state_independent", "state_dependent"] =
  "state_independent"` on `BrainConfig` (`dtypes.py`), docstring beside `action_mode`; covered by
  the #253 unknown-key warning automatically (test: a typo'd key warns; the real key parses).
- [ ] 2.2 Per-brain std heads, allocated only when `continuous` AND mode on, **after** all existing
  parameters, zero-initialised (weight + bias) post-construction so `std = 1` at step 0: mlpppo
  (gated-feature trunk), lstmppo (`h_out`), cfcppo (`h_new`, both `actor_head` modes),
  transformerppo (pooled `d_model`), connectome (`nn.Linear(4, 2)` off `_pool_motor(h)` on
  `ConnectomeTopology`, added to `learnable_parameters` conditionally).
- [ ] 2.3 In state-dependent mode the `log_std` parameter is **not allocated**; sample and evaluate
  call sites pass the head's output where they passed the parameter (`_policy.py` untouched);
  optimiser + grad-clip lists swap accordingly; `learning_rules/ppo.py` evaluate path reads the
  head via the topology when on.
- [ ] 2.4 Weight persistence: `log_std_head` `WeightComponent` (weight + bias) when on, `log_std`
  when off; save→load round-trip test per brain.
- [ ] 2.5 The frozen `_legacy_connectome_update_reference.py` is exempt (mode-off only) — assert no
  edits to it in this change's diff.

## 3. Tests (byte-identity + parity)

- [ ] 3.1 Per-brain off-mode byte-identity: default vs explicit `state_independent` construction ⇒
  `torch.equal` on every parameter (034 template); negative space: no `log_std_head` attribute off,
  no `log_std` attribute on (transformer discrete-mode precedent).
- [ ] 3.2 On-at-init parity: mode-on brain at step 0 produces the same action distribution as
  mode-off given the same mean (pinned RNG; sampled action byte-identical — the
  consolidate-change bar, no tolerance).
- [ ] 3.3 `(B, 2)` broadcast test in `test_policy_continuous.py`: batched state-dependent `log_std`
  through `continuous_evaluate_tanh_gaussian` matches per-row manual change-of-variables; entropy
  is the batch mean of per-state entropies.
- [ ] 3.4 Shape pin on cfc/lstm per-step paths: head output shape equals `mean`'s exactly
  (`(1, 2)`-shaped hiddens are the soft spot).
- [ ] 3.5 Gradient-flow test: with mode on, a training step moves the head's weights; existing
  continuous suites green **unmodified**; full suite + `pre-commit run --all-files` (the final
  gate, per house rule) before every push.

## 4. Configs

- [ ] 4.1 Six `_sdstd` variants (4 × C3: mlpppo/cfcppo/transformerppo/connectomeppo
  `*_small_continuous2d_combined_klinotaxis_sdstd.yml`; 2 × thermal gate: the 036 mlpppo
  klinotaxis + derivative pair), each differing from its parent by exactly
  `continuous_std_mode: state_dependent` (config-diff test asserts single-key delta); parents
  untouched (029/036 records).

## 5. The D7 validation gate (P.3)

- [ ] 5.1 Run the gate pair: n=4 seeds, 300ep, `--track-experiment`, capture-behaviour on; assay via
  `behavioural_chemotaxis_validation.py --modality thermotaxis --theta-sharp 0.45 --tail-runs 100`.
- [ ] 5.2 Grade against the pre-registered PASS (design Decision 4): klinotaxis-arm klinokinesis
  **PRESENT** (both ratio CIs' lower bounds > 1.0) AND weathervane non-regression AND derivative
  control non-PRESENT. On failure: one entropy-only tuning pass; still failing ⇒ dated D7
  amendment in `phase7-tracking` + roadmap — the substrate does NOT freeze on a failed gate.

## 6. The post-D7 re-baseline (P.4)

- [ ] 6.1 Launch 4 arms × n=8 (seeds 1–8) on the `_sdstd` C3 configs, uniform 6000ep,
  `--track-experiment`, per the `nematode-run-experiments` skill pattern; convergence audit
  (level-agnostic metric) before any ranking read; 029's top-up rule for still-climbing seeds
  (8000ep).
- [ ] 6.2 Frozen recipes: no tuning drift; if an arm's converged-fraction falls below its 029
  value, one bounded entropy-only repair pass, dated. lstmppo / minimal-RNN / GA exclusions
  recorded.
- [ ] 6.3 Analysis via `t7_continuous_ranking.py` (fresh manifest → ranking.json, per-seed CSVs,
  BH-FDR pairwise); comparison to 029 reported **qualitatively** (non-commensurability rule).
- [ ] 6.4 A/B probes, if any, from a clean `git worktree` only (consolidate-change incident note).

## 7. Logbook + freeze + close-out

- [ ] 7.1 Logbook (next number): gate verdict with curves, re-baseline table + converged-fraction,
  entropy-meaning shift discussion, exclusions, qualitative-vs-029 read.
- [ ] 7.2 `phase7-tracking` P.1–P.5 ticked; substrate declared **FROZEN** for the L4 panel in the
  tracker status header; roadmap Technical Debt item 8 updated; CHANGELOG entry (new config field
  + variants).
- [ ] 7.3 `openspec validate add-state-dependent-action-std --strict`; full suite green; grep gate:
  no `_policy.py` diff in this change.
