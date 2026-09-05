# Design — state-dependent action std (D7) + pre-panel platform freeze

## Context

Roadmap D7 (ratified + gate-amended 2026-08-27); `phase7-tracking` P.1–P.5; execution-protocol standards (that change's design.md § Decision B) apply. Logbook 036's mechanism finding: with a state-independent `log_std`, turn-rate cannot co-vary with thermal drive — the policy migrates-and-parks instead of biased-random-walking, so klinokinesis grades PARTIAL (ratios 1.015/1.006, CIs spanning the 1.0 null) while weathervane is merely weak. The Chen/Leifer navigation-reweighting validation target is unmeasurable until klinokinesis exists.

Recon facts this design rests on (verified 2026-09-05):

- `_policy.py`'s tanh-Gaussian helpers are shape-generic (`.sum(-1)` reductions; clamp elementwise; `Normal` broadcasts) — a `(B,2)` `log_std` needs **zero shared-module changes**. The one soft spot is per-step tensor shapes on cfc/lstm (`(1, units)`-shaped hiddens): the head output must match `mean`'s shape exactly; pinned by test.
- Five brains own continuous heads; each creates `log_std = nn.Parameter(zeros(2))` and has a clear trunk feature feeding its mean head (mlpppo: gated features; lstmppo: `h_out`; cfc: `h_new` in both `actor_head` modes; transformer: pooled `d_model`; connectome: `_pool_motor(h)` 4-vec feeding the `readout`).
- Entropy is a pure function of `log_std` (`Normal.entropy = c + log_std`); `mean_entropy = entropy().sum(-1).mean()` becomes a genuine batch mean of per-state entropies under the new mode — the *intended* mechanism (the policy can trade exploration across states), but it changes what a given `entropy_coef` buys, which is exactly why P.4 re-baselines rather than assumes.
- P.1 (#254) is already done in-tree; the closed issue pins "advantage normalization stays unimplemented" for 029-lineage comparability.
- The `consolidate-ppo-policy-helpers` spec sets the bar for anything near `_policy.py`: same-process pre/post comparison under pinned RNG, no stored goldens, no tolerance-widening, exact skip/xfail count invariance. Its worktree note is copied into the tasks (A/B measurements only from a clean `git worktree`).

## Goals / Non-Goals

**Goals**: the D7 capability byte-identical-when-off; the klinokinesis gate passed (or honestly failed into a dated amendment); the post-D7 reference frame measured at n=8; the substrate frozen for the L4 panel.

**Non-Goals**: advantage normalization (stays unimplemented per #254); entropy-schedule redesign; any `_policy.py` restructuring; retuning frozen recipes (bounded entropy-only repair excepted); lstmppo/minimal-RNN/GA re-baselines (not L4-panel reference arms); the L4 panel itself.

## Decisions

### Decision 1 — One shared config field, no per-brain duplication

`continuous_std_mode: Literal["state_independent", "state_dependent"] = "state_independent"` on `BrainConfig` (`dtypes.py`), beside `action_mode` — same scope (all five brains need it identically), same default-off semantics, one docstring, auto-covered by the #253 unknown-key warning via `model_fields`. Rejected: five per-brain fields (drift surface); a `log_std_init` float (unneeded — Decision 2 pins init to the current behaviour exactly).

### Decision 2 — RNG-free zero-Parameter heads make "on-at-init ≡ off-at-init" a run property

*(Amended at spec review — B1: `nn.Linear`-then-zero overwrites values but not the RNG stream, which would have shifted the connectome critic's draws and reduced step-0 parity to a test artifact.)* When on, each brain builds the head's **weight and bias directly as zero `nn.Parameter`s** (exactly how `log_std` is allocated today) — the allocation SHALL consume **zero RNG draws**. The head outputs `log_std ≡ 0 → std = 1` for every state at step 0, bit-for-bit the current parameter's init. Consequences: (a) step-0 sampling is byte-identical on/off **in a real run**, not just under a test re-seed — `Normal.sample()` consumes the same two draws either way; (b) mode-on construction leaves *every* other parameter draw matched (including the connectome critic, initialised after the topology inside the rule), and mode-off construction is byte-identical trivially. Allocation order still sits after all existing parameters as defence-in-depth.

### Decision 3 — Head placement per brain; `log_std` parameter not allocated when on

The head hangs off the exact trunk feature that feeds the mean head (recon table): mlpppo gated features; lstmppo `h_out`; cfc `h_new` (both `actor_head` modes — the head is mode-independent); transformer pooled `d_model`; **connectome: the `_pool_motor(h)` 4-vec, on the `ConnectomeTopology`** — mirroring the readout's input, keeping the parameters under `learnable_parameters`' conditional contract and the `BrainTopology` seam intact. *(Amended at spec review — B2: the accessor seam is pre-registered.)* The topology gains one **public method**, `state_dependent_log_std(hidden)`, computing `head(_pool_motor(hidden))` for both `(302,)` and `(B, 302)` hiddens; the rollout path and `learning_rules/ppo.py`'s evaluate path call it when the mode is on. The `forward_with_hidden*` signatures are **unchanged** (the frozen legacy reference unpacks a 2-tuple and must not be edited — a 3-tuple return is foreclosed), and mode-off paths in the rule and `_run_brain_continuous` remain byte-identical (the `topo.log_std` read untouched). In state-dependent mode the `log_std` parameter is **not allocated** (transformer's discrete-mode `not hasattr` precedent): optimiser and grad-clip lists carry the head instead, and weight persistence gains a `log_std_head` component in place of `log_std`. The frozen `_legacy_connectome_update_reference.py` reads `topology.log_std` and is **exempt by design** — it must not be modernised, and every byte-equivalence test runs mode-off.

### Decision 4 — The gate is pre-registered, with a bounded failure path

Gate configs: `_sdstd` variants of the 036 thermal pair (mlpppo klinotaxis + derivative control), n=4, 300ep, `--theta-sharp 0.45 --tail-runs 100`, same grading machinery (80% bootstrap CIs, sign-only refs). **PASS** requires all three, pre-registered: (i) klinotaxis-arm combined klinokinesis verdict **PRESENT** (both ratio statistics REPRODUCED — CI lower bounds > 1.0); (ii) weathervane grades ≥ their 036 values (thresholded REPRODUCED, θ-free REPRODUCED — no regression); (iii) the derivative control's combined klinokinesis verdict is **neither PRESENT nor PRESENT_PARTIAL** — equivalently, no klinokinesis ratio statistic grades REPRODUCED in the control (a PRESENT_PARTIAL control would be a significant klinokinesis signal without head-sweep sensing, eroding the specificity dissociation while passing a naive "non-PRESENT" reading; 036's control was ABSENT). On failure: exactly one entropy-coefficient-only tuning pass on the gate configs; still failing ⇒ the finding is recorded and D7 is amended with a dated note (the roadmap's Leifer target then explicitly stays unmeasurable) — the substrate does NOT freeze on a silently-failed gate.

### Decision 5 — Re-baseline: four arms, frozen recipes, qualitative vs 029

Arms = the L4 panel's reference frame (D2/D10): **mlpppo** (leader anchor + matched-rule counterpart), **cfcppo + transformerppo** (the D2 descriptive band), **connectomeppo** (the 2×2's PPO-regime row). n=8, seeds 1–8, uniform 6000ep + the 029 top-up rule (level-agnostic convergence audit before ranking; still-climbing seeds to 8000ep), plateau-tail full-clear metric, `t7_continuous_ranking.py` with a fresh manifest. Recipes carry over **frozen** — `entropy_coef` untouched even though its meaning shifts (that shift is part of what is being measured); if an arm's converged-fraction collapses below its 029 value, one bounded entropy-only repair pass per arm, dated in the logbook. The comparison to 029 is **qualitative** (substrate change; non-commensurability rule) — the deliverable is the new reference frame, not a delta. lstmppo/minGRU/minLSTM/GA exclusions are recorded in the logbook, with lstmppo's config left mode-off so the 031/033 memory-arm lineage stays coherent.

### Decision 6 — New `_sdstd` config variants; originals are records

Six new YAMLs (4 C3 + 2 thermal-gate), each differing from its parent by exactly `continuous_std_mode: state_dependent` (asserted by a config-diff test). Parents untouched — they are the 029/036/034 record, and the rewired-null precedent set the variant-file pattern.

### Decision 7 — P.1 verified done; P.5 is fields + declaration; the frozen substrate is mode-ON

P.1 ticks with a dated verified-already-done note (house style; the work shipped in the Phase 6 bit-memory pre-work). P.5 = the Decision-1 field (load-time-validated automatically), a dated verified-already-done note for its plasticity-key half (the trace fields are pydantic-validated since #303), and the explicit tracker/roadmap **FROZEN** declaration once the gate passes and the re-baseline lands. *(Added at spec review — S2.)* The freeze statement SHALL say which mode it freezes: **the L4 panel runs `continuous_std_mode: state_dependent`** — panel configs derive from the `_sdstd` variants, so the panel and its reference frame share one substrate (running the panel mode-off would span exactly the delta the freeze exists to prevent). Whether the three-factor rule updates the std head or leaves it PPO-owned is a **panel-change decision**, recorded there, not silently defaulted here.

### Decision 8 — Config honesty: mode-mismatch fails at load; cross-mode weights fail loudly

*(Added at spec review — S5/S3.)* `continuous_std_mode: state_dependent` beside `action_mode: discrete` is meaningless and SHALL fail at load (a pydantic `model_validator` on `BrainConfig` — the #253/#254 config-honesty stance, made an error rather than a warning since no legitimate combination exists). Cross-mode weight loading SHALL raise a descriptive error (a mode-on brain fed a mode-off file, or vice versa, must not AttributeError or silently skip the std component); same-mode round-trips are the supported path.

## Risks / open questions

- **The gate may fail** — state-dependent std is the 036-diagnosed *necessary* condition, not a proven sufficient one. Decision 4's bounded failure path keeps that honest; the likeliest repair lever (entropy coefficient) is pre-registered as the only one.
- **Seed fragility may worsen** under per-state entropy (the coef's meaning changes on the fragile recurrent arms). Contained: cfc is the only fragile re-baseline arm; the repair pass + converged-fraction reporting (house standard) carry it.
- **Per-state clamp-ceiling trap** *(added at spec review — S4)*: `torch.clamp` has zero gradient outside `[-5, 2]`, so a state whose head output drifts past +2 receives **no restoring gradient through the std path** — a per-state stuck-at-max-entropy failure the single shared parameter could not exhibit (its one value is pulled by all states; cf. the 027-era over-exploration history). Pre-registered guard, no mechanism change: every mode-on run records per-update **mean and max of the clamped `log_std` batch** (`RuleStepReport.extra` for the connectome; a logged stat for the PPO brains), reported in the logbook; the pre-registered response to ceiling-pinning is the bounded entropy-only repair pass.
- **Compute**: 32 runs × 6000–8000ep continuous C3 is the tranche's real cost (days of wall-clock at 4–6-way parallelism on the dev machine); the gate is hours. Launches via the `nematode-run-experiments` skill pattern; all runs `--track-experiment`.
- **A/B hygiene**: any pre/post-D7 probe measurements come from a clean `git worktree` (the consolidate-change's recorded incident).
