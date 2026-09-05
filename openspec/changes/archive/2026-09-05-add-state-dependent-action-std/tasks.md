# Tasks — state-dependent action std (D7) + pre-panel platform freeze

Scope: `phase7-tracking` P.1–P.5. The L4 panel itself is NOT in this change; this change ends with
the substrate declared frozen.

## 1. P.1 verification (D5 — no code)

- [x] 1.1 Verify-and-record: zero `normalize_advantages` keys in `configs/` (was 60+ per #254),
  zero lstmppo renamed-key residue (`chunk_length`/`hidden_dim`); tick `phase7-tracking` P.1 with a
  dated already-done note (shipped in the Phase 6 bit-memory pre-work). Advantage normalization
  stays unimplemented (closed-issue constraint; 029 raw-GAE lineage).

## 2. Config field + std heads (D7 mechanism)

- [x] 2.1 `continuous_std_mode: Literal["state_independent", "state_dependent"] = "state_independent"` on `BrainConfig` (`dtypes.py`), docstring beside `action_mode`; covered by
  the #253 unknown-key warning automatically (test: a typo'd key warns; the real key parses).
- [x] 2.2 Per-brain std heads, allocated only when `continuous` AND mode on, **after** all existing
  parameters, with weight and bias built **directly as zero `nn.Parameter`s** — the allocation
  consumes zero RNG draws (review B1; no `nn.Linear` default init), so `std = 1` at step 0 and
  every other parameter draw stays matched on/off: mlpppo
  (gated-feature trunk), lstmppo (`h_out`), cfcppo (`h_new`, both `actor_head` modes),
  transformerppo (pooled `d_model`), connectome (a zero-Parameter `(4 → 2)` head off
  `_pool_motor(h)` on `ConnectomeTopology`, added to `learnable_parameters` conditionally).
- [x] 2.3 In state-dependent mode the `log_std` parameter is **not allocated**; sample and evaluate
  call sites pass the head's output where they passed the parameter (`_policy.py` untouched);
  optimiser + grad-clip lists swap accordingly. The connectome accessor is the **public**
  `ConnectomeTopology.state_dependent_log_std(hidden)` (handles `(302,)` and `(B, 302)`; review
  B2) — `forward_with_hidden*` signatures unchanged (frozen-reference 2-tuple arity), and
  `learning_rules/ppo.py` + `_run_brain_continuous` call it only when the mode is on (mode-off
  paths byte-identical).
- [x] 2.4 Weight persistence (the four persisting brains — the connectome has no
  `WeightPersistence` surface, exempt with a note; review M2): `log_std_head` `WeightComponent`
  (weight + bias) when on, `log_std` when off; same-mode save→load round-trip test per brain;
  **cross-mode load raises a descriptive mode-mismatch error** (never AttributeError or a silent
  skip; review S3) + test.
- [x] 2.5 The frozen `_legacy_connectome_update_reference.py` is exempt (mode-off only) — assert no
  edits to it in this change's diff.
- [x] 2.6 Config honesty (review S5): pydantic `model_validator` on `BrainConfig` rejects
  `continuous_std_mode: state_dependent` with `action_mode: discrete` at load; test.

## 3. Tests (byte-identity + parity)

- [x] 3.1 Per-brain off-mode byte-identity: default vs explicit `state_independent` construction ⇒
  `torch.equal` on every parameter (034 template); negative space: no `log_std_head` attribute off,
  no `log_std` attribute on (transformer discrete-mode precedent).
- [x] 3.2 On-at-init parity: mode-on brain at step 0 produces the same action distribution as
  mode-off given the same mean (pinned RNG; sampled action byte-identical — the
  consolidate-change bar, no tolerance).
- [x] 3.3 `(B, 2)` broadcast test in `test_policy_continuous.py`: batched state-dependent `log_std`
  through `continuous_evaluate_tanh_gaussian` matches per-row manual change-of-variables; entropy
  is the batch mean of per-state entropies.
- [x] 3.4 Shape pin on cfc/lstm per-step paths: head output shape equals `mean`'s exactly
  (`(1, 2)`-shaped hiddens are the soft spot).
- [x] 3.5 Gradient-flow test: with mode on, a training step moves the head's weights; existing
  continuous suites green **unmodified** with **skip/xfail counts exactly invariant** (the
  consolidate-change clause; review M5); full suite + `pre-commit run --all-files` (the final
  gate, per house rule) before every push.
- [x] 3.6 Clamp-ceiling monitor (review S4): mode-on runs record per-update mean/max of the
  clamped `log_std` batch (`RuleStepReport.extra` for the connectome; a logged stat for the PPO
  brains); surfaced in the gate + re-baseline logbook.

## 4. Configs

- [x] 4.1 Six `_sdstd` variants (4 × C3: mlpppo/cfcppo/transformerppo/connectomeppo
  `*_small_continuous2d_combined_klinotaxis_sdstd.yml`; 2 × thermal gate: the 036 mlpppo
  klinotaxis + derivative pair), each differing from its parent by exactly
  `continuous_std_mode: state_dependent` (config-diff test asserts single-key delta); parents
  untouched (029/036 records).

## 5. The D7 validation gate (P.3) — RUN 2026-09-05; **FAILED** → Amendment A (Logbook 038)

- [x] 5.1 Run the gate pair: n=4 seeds, 300ep, `--track-experiment`, capture-behaviour on; assay via
  `behavioural_chemotaxis_validation.py --modality thermotaxis --theta-sharp 0.45 --tail-runs 100`.
- [x] 5.2 Grade against the pre-registered PASS (design Decision 4): klinotaxis-arm klinokinesis
  **PRESENT** (both ratio CIs' lower bounds > 1.0) AND weathervane non-regression (both slope
  statistics stay REPRODUCED) AND the derivative control's combined klinokinesis verdict
  **∉ {PRESENT, PRESENT_PARTIAL}** (no REPRODUCED ratio statistic in the control; review S1). On failure: one entropy-only tuning pass; still failing ⇒ dated D7
  amendment in `phase7-tracking` + roadmap — the substrate does NOT freeze on a failed gate.

## 6. The post-D7 re-baseline (P.4) — **DESCOPED under Amendment A** (gate failed; substrate froze mode-off; Logbook 029 stands; house dropped-scope ticks below)

- [x] 6.1 Launch 4 arms × n=8 (seeds 1–8) on the `_sdstd` C3 configs, uniform 6000ep,
  `--track-experiment`, per the `nematode-run-experiments` skill pattern; convergence audit
  (level-agnostic metric) before any ranking read; 029's top-up rule for still-climbing seeds
  (8000ep).
- [x] 6.2 Frozen recipes: no tuning drift; if an arm's converged-fraction falls below its 029
  value, one bounded entropy-only repair pass, dated. lstmppo / minimal-RNN / GA exclusions
  recorded.
- [x] 6.3 Analysis via `t7_continuous_ranking.py` (fresh manifest → ranking.json, per-seed CSVs,
  BH-FDR pairwise); expected missing-seed WARNs for the absent `PPO_ARCHS` arms
  (lstmppo/mingruppo/minlstmppo) noted so they are not misread (review M4); comparison to 029
  reported **qualitatively** (non-commensurability rule).
- [x] 6.4 A/B probes, if any, from a clean `git worktree` only (consolidate-change incident note).

## 7. Logbook + freeze + close-out

- [x] 7.1 Logbook (next number): gate verdict with curves, re-baseline table + converged-fraction,
  entropy-meaning shift discussion, exclusions, qualitative-vs-029 read.
- [x] 7.2 `phase7-tracking` P.1–P.5 ticked (P.5's plasticity-key half as a dated
  verified-already-done note — the trace fields are pydantic-validated since #303; review M7);
  substrate declared **FROZEN for the L4 panel in `continuous_std_mode: state_dependent`** —
  panel configs derive from the `_sdstd` variants, head ownership (PPO vs three-factor rule)
  decided in the panel change (review S2); roadmap Technical Debt item 8 updated **including the
  now-wrong "via `_policy.py`" phrase** (review M6); `configs/README.md` variant vocabulary gains
  `sdstd` (review M1); the stale `log_std` example in the base `brain-architecture` spec's
  learnable-parameters scenario is noted in the logbook as non-exhaustive "e.g." (review M3);
  CHANGELOG entry (new config field + variants).
- [x] 7.3 `openspec validate add-state-dependent-action-std --strict`; full suite green; grep gate:
  no `_policy.py` diff in this change.
