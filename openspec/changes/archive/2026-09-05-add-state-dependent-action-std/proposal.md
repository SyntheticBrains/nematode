# Add the state-dependent action std (D7) + pre-panel platform freeze

## Why

`phase7-tracking` shipment 7a-i's pre-panel platform tranche (tasks P.1–P.5; roadmap D5/D7). [Logbook 036](../../../docs/experiments/logbooks/036-realworm-thermotaxis-validation.md) root-caused the missing thermotaxis klinokinesis to the **state-independent `log_std`**: every continuous brain samples from a tanh-Gaussian whose std is a free parameter, so the policy can steer but cannot modulate its own stochasticity by state — the biased-random-walk component of real-worm navigation is structurally unreachable, and the Leifer/Chen co-primary validation target (roadmap § validation targets) is unmeasurable until it is. D7 (ratified, amended with its validation gate 2026-08-27) lands the capability *before* the L4 panel and then freezes the substrate — the substrate-vs-rule confound row demands the panel never spans a platform change.

Two scope facts established at recon (2026-09-05):

- **P.1 (D5, the #254 dead-key removal) is already done.** Issue #254 was closed with the "remove" decision and the cleanup was folded into the Phase 6 bit-memory pre-work: zero configs carry `normalize_advantages` today, and the lstmppo renamed-key residue is likewise clean (verified by grep across `configs/`). This change verifies-and-ticks P.1 with a dated note; no code. The issue's standing constraint carries: advantage normalization stays **unimplemented** so the re-baseline remains comparable to 029's raw-GAE lineage.
- **`_policy.py` needs no changes.** The tanh-Gaussian helpers are shape-generic (all reductions `.sum(-1)`; docstring already promises "batched or broadcastable log-std"), so a `(B, 2)` state-dependent `log_std` flows through sample/evaluate unchanged — the mechanism is per-brain std *heads*, not shared-module surgery. This matters because `_policy.py` carries the strictest regression bar in the repo (the consolidate-ppo-policy-helpers spec).

## What Changes

### 1. `continuous_std_mode` on the shared `BrainConfig`

One field — `continuous_std_mode: Literal["state_independent", "state_dependent"] = "state_independent"` — beside `action_mode` in `dtypes.py`, inherited by all five continuous brains (mlpppo, cfcppo, lstmppo, transformerppo, connectomeppo; minimal-RNN inherits via lstmppo). Load-time validation is automatic via `model_fields` (#253 path).

### 2. Per-brain state-dependent std heads, byte-identical-when-off

When the mode is on (and `action_mode: continuous` — the discrete combination fails at load), each brain allocates a small linear **std head** off the same trunk feature that feeds its mean head, its weight and bias built directly as zero `nn.Parameter`s so the allocation consumes **no RNG draws** and `log_std ≡ 0 → std = 1` at step 0 — exactly the current parameter's init, making "on-at-init ≡ off-at-init" a byte-identical **run property** (no test-side re-seed) and leaving every other parameter draw matched across modes. The connectome's accessor is the public `ConnectomeTopology.state_dependent_log_std(hidden)`; forward signatures are unchanged. Off (the default), nothing is allocated and every brain is byte-identical to today. The `log_std` parameter is not allocated when the head is (mirroring the transformer's discrete-mode negative-space precedent); optimiser/grad-clip lists and `WeightComponent` persistence swap accordingly. The frozen `_legacy_connectome_update_reference.py` is explicitly exempt (byte-equivalence tests run mode-off).

### 3. The D7 validation gate (P.3)

Re-run the 036 thermal pair (mlpppo klinotaxis arm + derivative specificity control, n=4, 300ep) with the mode **on**, graded by the same assay (`behavioural_chemotaxis_validation.py --modality thermotaxis --theta-sharp 0.45 --tail-runs 100`). **PASS** = the klinotaxis arm's combined klinokinesis verdict reaches **PRESENT** (both ratio statistics' 80% CI lower bounds > 1.0; 036 scored PARTIAL at 1.015/1.006 with CIs spanning 1.0), weathervane does not regress below its 036 grades (both slope statistics REPRODUCED), and the derivative control's combined klinokinesis verdict is neither PRESENT nor PRESENT_PARTIAL (specificity — no REPRODUCED ratio statistic in the control). Fail ⇒ one bounded entropy-only tuning pass, then a dated D7 amendment if still failing — never a silent slide.

### 4. The post-D7 re-baseline (P.4)

Four reference arms × n=8 (seeds 1–8) = **32 runs** on state-dependent-std variants of the C3 continuous configs: **mlpppo, cfcppo, transformerppo, connectomeppo** (the L4 panel's descriptive reference frame per D2/D10; lstmppo/minimal-RNN/GA are not panel reference arms and are excluded — recorded, not silent). 029 protocol carried verbatim: frozen recipes (no per-arm retuning; entropy-only bounded repair if an arm's converged-fraction collapses, documented), uniform 6000ep with the level-agnostic top-up rule, plateau-tail metric, `t7_continuous_ranking.py` analysis. The result becomes the panel's reference frame — compared to 029 **qualitatively** (substrate change ⇒ non-commensurable, per the 2026-06-14 house rule) — and the freeze is explicitly **mode-on**: the L4 panel configs derive from the `_sdstd` variants.

### 5. Freeze declaration (P.5) + logbook

A new logbook (gate verdict + re-baseline table) closes the tranche; `phase7-tracking` P.1–P.5 tick and the substrate is declared **FROZEN** for the L4 panel; roadmap Technical Debt item 8 updates.

## Capabilities

**Modified**: `brain-architecture` (continuous-action head requirements gain the std-mode contract, the per-brain head placement, the gate, and the re-baseline reference frame).

**Added**: none.

## Impact

**Code**: `brain/arch/dtypes.py` (field); `mlpppo.py`, `cfc_ppo.py`, `lstmppo.py`, `transformer_ppo.py`, `connectome_ppo.py` (+ the connectome head on `ConnectomeTopology`, feeding from the pooled motor 4-vec so `learnable_parameters` stays conditional); `learning_rules/ppo.py` (evaluate path reads the head when on). No `_policy.py` changes.

**Configs**: state-dependent variants of the four C3 reference configs + the two 036 thermal-gate configs (suffix `_sdstd`); originals untouched (they are the 029/036 record).

**Tests**: per-brain off-byte-identity + on-at-init parity + head-persistence tests; a `(B,2)` broadcast test in `test_policy_continuous.py`; negative-space (`not hasattr`) checks; existing continuous suites green unmodified.

**Runs**: gate 2×4×300ep (hours); re-baseline 32×6000–8000ep (the tranche's real calendar cost; launched per the `nematode-run-experiments` skill, `--track-experiment` required).

**Docs**: new logbook; `phase7-tracking` tasks; roadmap Technical Debt item 8; CHANGELOG.

## Breaking Changes

None. Mode off (default) is byte-identical everywhere; no existing config changes behaviour.

## Backward Compatibility

Existing weights files load unchanged (the `log_std` component only exists in mode-off brains, as today). The 029/036 configs and artifacts are untouched records.
