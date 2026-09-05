# 038 — State-Dependent Action Std: D7 Validation Gate (FAILED → Amendment A)

**Date**: 2026-09-05
**Change**: `add-state-dependent-action-std` (Phase 7 tracker P.2/P.3; roadmap D7)
**Verdict**: mechanism **SHIPPED**; klinokinesis gate **FAILED** after the one pre-registered
entropy-only pass; **Amendment A applied** — the substrate freezes **mode-off**, Logbook 029
remains the L4 panel's reference frame, and the post-D7 re-baseline (P.4) is descoped.

## Background

[Logbook 036](036-realworm-thermotaxis-validation.md) root-caused the missing thermotaxis
klinokinesis to the state-independent `log_std`: the policy could steer but not modulate its own
stochasticity by state. Roadmap D7 (ratified 2026-08-27) landed the capability — per-brain
state-dependent std heads, byte-identical-when-off — with a pre-registered gate: re-run the 036
thermal pair with the mode on; freeze the substrate only if the klinotaxis arm's combined
klinokinesis verdict reaches **PRESENT** (both down/up ratio statistics' 80% CI lower bounds

> 1.0), weathervane holds its 036 grades (both slope statistics REPRODUCED), and the derivative
> control stays ∉ {PRESENT, PRESENT_PARTIAL}. Failure path, also pre-registered: exactly one
> entropy-coefficient-only tuning pass, then a dated D7 amendment — never a silent freeze.

## Method

036 protocol verbatim: mlpppo thermal pair (`thermal_foraging/*_seeking_{klinotaxis,derivative}`),
`_sdstd` variants (single-key delta: `continuous_std_mode: state_dependent`), n=4 seeds (42–45),
300 episodes, `--track-experiment`; assay
`behavioural_chemotaxis_validation.py --modality thermotaxis --theta-sharp 0.45 --tail-runs 100`.
Attempt 2 changed exactly one value: `entropy_coef 0.05 → 0.01` (scratch configs; the committed
`_sdstd` records untouched). The clamp-ceiling monitor (`log_std_clamped_mean/max` per update,
review S4) ran on every session.

## Results

| statistic (klinotaxis arm, n=4) | attempt 1 (ent 0.05) | attempt 2 (ent 0.01) | 036 baseline (mode-off) |
|---|---|---|---|
| klinokinesis turn-rate ratio | 0.995 [0.993, 0.997] | 1.000 [0.992, 1.007] | 1.015 [0.996, 1.034] |
| klinokinesis magnitude ratio | 0.997 [0.996, 0.998] | 1.000 [0.997, 1.004] | 1.006 [0.999, 1.013] |
| weathervane slope (thresholded) | +0.003 [+0.001, +0.006] | +0.009 [+0.002, +0.017] | +0.011 [+0.007, +0.015] |
| weathervane slope (θ-free) | +0.014 [+0.012, +0.017] | +0.005 [−0.003, +0.013] | +0.020 [+0.013, +0.027] |
| **combined klinokinesis / weathervane** | ABSENT / PRESENT | EQUIVOCAL / PRESENT_PARTIAL | PARTIAL / PRESENT |
| std monitor (mean clamped log_std) | **pinned at +2.0** from ~¼ of training | healthy: drifts to −0.46, max ~0.07 | n/a (free parameter) |

Derivative control: klinokinesis ABSENT in both attempts (specificity intact throughout).

## The two findings

1. **The S4 clamp-ceiling trap is real, and the monitor caught it live.** At `entropy_coef 0.05`
   the per-state entropy bonus drove the head to the +2 clamp ceiling for every state within ~75
   updates — `std ≈ 7.4` everywhere, no restoring gradient beyond the clamp (`torch.clamp` has
   zero gradient outside its bounds), a per-state stuck-at-max-entropy failure the single shared
   parameter could not exhibit. The near-random policy explains attempt 1's below-null
   klinokinesis lean and weakened weathervane. This validates both the monitor's design and the
   coefficient-meaning-shift warning the change pre-registered.
2. **Capability is not pressure: the 036 diagnosis was necessary, not sufficient.** With healthy
   std dynamics (attempt 2), the policy *could* modulate its stochasticity by state — and did not.
   Both ratio statistics sit dead on the 1.0 null. Nothing in the reward, at 300 episodes on this
   task, differentiates turn-rate by gradient direction strongly enough for the head to learn the
   biased-random-walk pattern. Weathervane simultaneously regressed to PRESENT_PARTIAL (θ-free
   PARTIAL), failing the non-regression clause independently.

**Pre-registered verdict: gate FAILED** (attempt 2: klinokinesis EQUIVOCAL ≠ PRESENT; weathervane
non-regression failed). The one permitted tuning pass is spent.

## Amendment A (dated 2026-09-05, per Decision 4's failure path)

- The substrate freezes **mode-off** (`state_independent`, the default): the L4 panel runs on the
  unchanged 029-lineage substrate.
- **Logbook 029 remains the panel's descriptive reference frame**; the post-D7 n=8 re-baseline
  (P.4, ~32 runs) is **descoped** — nothing changed underneath 029, so there is nothing to
  re-measure.
- The D7 mechanism ships as a tested, dormant capability: byte-identical when off (48 tests), with
  the `_sdstd` config variants and the ceiling monitor available to any future attempt.
- The Chen/Leifer navigation-reweighting validation target **remains unmeasurable** (klinokinesis
  absent on the substrate), exactly as the roadmap's D7 row anticipated for a failed gate. The
  dopamine-gated-forgetting target stands alone as the primary biological-validation candidate,
  with its preprint caveat now more load-bearing.
- **Future work (recorded, not scheduled)**: klinokinesis emergence likely needs *pressure*, not
  capability — candidate levers are reward-side (a turn-rate-sensitive shaping term),
  longer-horizon training, or a structurally stochastic motor readout; any retry is its own
  pre-registered change with the gate protocol amended explicitly.

## Limitations

- n=4 at 300 episodes per the 036 protocol; a capability this subtle may need more training than
  the gate budget allows — that is precisely why the failure is recorded as "not sufficient under
  these training conditions" rather than "impossible".
- The entropy lever was explored at exactly two points (0.05, 0.01) per the pre-registration; the
  space between was deliberately not searched.
- Attempt 2's healthy-std run shows the head *tracking* something (mean drifts to −0.46 — the
  policy learned to *reduce* exploration globally); state-differentiation, not std learning per
  se, is what failed to emerge.

## Supporting artefacts

`docs/experiments/logbooks/supporting/038-state-dependent-std-gate/`:
`attempt1-ent005/` and `attempt2-ent001/` each hold the assay outputs
(`curves_{klinotaxis,derivative}.json`) and the seed-42 std-monitor trajectory
(`monitor_log_std_mean_seed42.csv`). Experiment IDs `20260905_003738_*` (attempt 1) and the
attempt-2 set are in the local (gitignored) `experiments/` store; gate configs are the committed `_sdstd`
thermal pair (attempt 2 = the same with `entropy_coef: 0.01`, reproducible by that one edit).
