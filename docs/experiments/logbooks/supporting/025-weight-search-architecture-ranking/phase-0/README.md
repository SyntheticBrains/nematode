# Supporting: 025 Phase 0 — Predator-sensing convergence investigation

Companion data for the upcoming [025 — weight-search architecture ranking](../../../025-weight-search-architecture-ranking.md) logbook. Captures the methodology, per-variant per-seed results, root-cause findings, and canonical-variant lock-in for the Phase 0 predator-sensing investigation deferred from [`fix-predator-sensing-biology` Modelling caveat 6](../../../../../../openspec/changes/archive/2026-05-24-fix-predator-sensing-biology/design.md).

The Phase 0 work answers: **does the corrected two-channel predator-sensing biology actually train competitively against legacy `nociception_klinotaxis` at canonical Phase 4 budget, or is there a structural learning gap that would distort the downstream architecture comparison?**

Short version: there was a 44pp gap at canonical budget; root-cause was a silent-gating bug in `predator_lateral_gradient` population (`agent.py`); post-fix the canonical two-channel biology beats legacy by +14pp once paired with the new `distal_chemo_contact_trigger` reward.

## Files in this directory

| File | Purpose |
|---|---|
| `README.md` | This narrative + ranking tables (load-bearing forensics). |
| `per-seed-results.csv` | Per-variant per-seed last-25 metrics (success%, foods/ep, death%, reward/ep). |
| `summary-stats.csv` | Aggregated mean ± std per variant. |

## Methodology

**Canonical Phase 4 budget**: 500 episodes × max_steps 500, n=4 seeds (42, 43, 44, 45), `--theme headless`. All runs use the same brain (MLPPPO small, identical hyperparameters), the same env (pursuit scenario, grid 20×20, 6 foods, 10-food target, 2 predators speed 0.5, detection_radius 6), and the same `satiety` + `health` blocks. Only sensors + reward shape vary across the 6 variants.

The brain choice (MLPPPO small) is deliberate: it's the cheapest predator-evasion config, so n=4 × 6 variants × 500 episodes fits within ~10h wall-clock. Phase 4 inherits the canonical sensor + reward chosen here for every architecture cell.

A separate Step A pre-flight (n=1, seed 2026) at canonical budget gated the decision to invest the n=4 sweep — see the [Step A pre-flight section](#step-a-pre-flight-n1) below.

### Variants evaluated

| ID | Sensors | Reward |
|---|---|---|
| A1 | legacy `nociception_klinotaxis` (3-dim) | `gradient_proximity` (default) |
| A2 | new biology: `predator_mechanosensation_klinotaxis` + `predator_chemosensation_klinotaxis` (6-dim) | `gradient_proximity` (default) |
| B0.3 | sparse-fix mechano (`predator_mechanosensation_klinotaxis_sparse_fix`) + canonical chemo | `gradient_proximity` (default) |
| B0.4 | composite single-channel `predator_biology_klinotaxis` (4-dim) | `gradient_proximity` (default) |
| B0.5 | canonical new biology (same as A2) | `distal_chemo_contact_trigger` (new dual-mechanism) |
| B0.6 | sparse-fix mechano + canonical chemo (same as B0.3) | `distal_chemo_contact_trigger` (new) |

The sparse-fix mechano variant (B0.3, B0.6) injects `predator_distal_concentration` into the strength field when `predator_contact_zone == ContactZone.NONE`, replacing the canonical channel's mostly-zero off-contact signal.

The composite variant (B0.4) collapses the two parallel 3-dim mechano + chemo channels into a single 4-dim feature vector (intensity + zone-as-angle + distal_concentration + dconcentration_dt), dropping the chemo channel's `predator_lateral_gradient` by design.

The `distal_chemo_contact_trigger` reward replaces the default `gradient_proximity`'s distance-scaled evasion term + flat-fallback with two biologically-motivated terms matching the dual-channel sensor split: continuous distal-chemo penalty (`env.get_predator_concentration`) + binary contact damage trigger at `dist <= 1`.

### Configs

The five Phase 0 evaluation YAMLs (legacy A1 control + four ablation variants B0.3–B0.6) are persisted alongside this README at `./configs/` as reproducibility artefacts. A2 reuses the canonical pursuit config under `configs/scenarios/pursuit/` (which post-Phase 0 has been updated to use `reward_mode: distal_chemo_contact_trigger` as the canonical reward — to reproduce the A2 pre-Phase-0 baseline form, remove that one line from the canonical pursuit config).

| Variant | Config |
|---|---|
| A1 | [`./configs/mlpppo_small_legacy_nociception_klinotaxis_control.yml`](./configs/mlpppo_small_legacy_nociception_klinotaxis_control.yml) |
| A2 | [`configs/scenarios/pursuit/mlpppo_small_predator_biology_klinotaxis.yml`](../../../../../../configs/scenarios/pursuit/mlpppo_small_predator_biology_klinotaxis.yml) (canonical pursuit config; per note above, the A2 baseline used the pre-Phase-0 default-reward form) |
| B0.3 | [`./configs/mlpppo_small_predator_sparse_fix_klinotaxis.yml`](./configs/mlpppo_small_predator_sparse_fix_klinotaxis.yml) |
| B0.4 | [`./configs/mlpppo_small_predator_composite_klinotaxis.yml`](./configs/mlpppo_small_predator_composite_klinotaxis.yml) |
| B0.5 | [`./configs/mlpppo_small_predator_biology_klinotaxis_distal_chemo_contact_trigger.yml`](./configs/mlpppo_small_predator_biology_klinotaxis_distal_chemo_contact_trigger.yml) (canonical winner) |
| B0.6 | [`./configs/mlpppo_small_predator_sparse_fix_distal_chemo_contact_trigger_klinotaxis.yml`](./configs/mlpppo_small_predator_sparse_fix_distal_chemo_contact_trigger_klinotaxis.yml) |

### Command (per run)

```bash
uv run python scripts/run_simulation.py \
  --config <config_path> \
  --theme headless --runs 500 --seed <seed> \
  --log-level INFO --show-last-frame-only
```

Phase 0 launched all 40 canonical-budget runs in parallel via `run_in_background` task batches (Step A: 2 runs; Step B pre-fix: 20 runs; Step B post-fix re-run: 12 runs; Extended validation: 8 runs). Total wall-clock ≈ 8-10 hours across batches.

## Step A pre-flight (n=1)

Goal: settle whether a 5× budget jump (100 ep smoke → 500 ep canonical) closes the gap. Decision gate: gap ≤ 10pp on last-25 mean success → budget artefact (skip Step B ablations); gap > 10pp → real structural gap (run full Step B).

**Result** (seed 2026):

| Variant | last-25 success | last-25 foods/ep | last-25 death rate |
|---|---|---|---|
| A1 legacy | 60.0% | 8.68 | 40.0% |
| A2 new biology | 16.0% | 5.28 | 84.0% |
| **Gap (legacy − new)** | **44 pp** | 3.40 | -44 pp (worse) |

Gate fired (gap > 10pp): proceeded to Step B at n=4 × 5 variants.

## Step B results (pre-Bug-1-fix)

20 of 20 runs complete. None of the three initial ablations closed the gap to legacy:

| Variant | last-25 success | Δ vs A2 |
|---|---|---|
| A1 legacy | 67.0% ± 7.6 | — |
| B0.5 reward variant | 25.0% ± 5.0 | +2.0pp |
| B0.3 sparse_fix sensor | 24.0% ± 11.3 | +1.0pp |
| A2 new biology | 23.0% ± 14.0 | — |
| B0.4 composite | 16.0% ± 7.3 | -7.0pp |

This null result triggered a deeper bio-fidelity audit before committing to bigger-network / reward-tuning experiments. Three diagnostic investigations ran in parallel — learning trajectory, failure mode, and sensor encoding.

## Root-cause findings

Three findings from the diagnostic investigations:

1. **Learning trajectory**: all four new-biology variants plateau by ep 350-400 (none still climbing). Eliminates the "longer budget would help" hypothesis.

2. **Failure-mode diagnosis**: per-encounter evasion is similar (legacy 83%, new biology 77%) — only 5-7pp gap. Death rate accumulates across ~4 encounters/ep: legacy 0.83⁴≈47% cumulative death vs new biology 0.77⁴≈65% cumulative death. The brain is *almost* evading; the per-step probability compounds.

3. **Smoking gun (Bug 1)**: `agent.py:_compute_temporal_data` gated `predator_lateral_gradient` population on the legacy `nociception_mode == KLINOTAXIS`. New-biology configs set `predator_distal_mode: klinotaxis` but leave `nociception_mode` at ORACLE (the legacy field is irrelevant for new modules). Result: `predator_lateral_gradient` was silently `None` in A2 / B0.3 / B0.5; the chemo channel's directional `angle` field was permanently 0.0. The brain had **no head-sweep directional information about predators** despite the chemo channel claiming klinotaxis mode — emitting `[distal_concentration, 0.0, dC/dt]` instead of `[distal_concentration, lateral_gradient, dC/dt]`.

   Legacy A1 explicitly set `nociception_mode: klinotaxis` so the field IS populated — explaining the entire 44pp gap.

**Bug 1 fix** (commit `65a5b517`): `agent.py:_compute_temporal_data` now populates `predator_lateral_gradient` under EITHER legacy `nociception_mode == KLINOTAXIS` OR new `predator_distal_mode == KLINOTAXIS`. Three regression tests added (`TestPredatorLateralGradientPopulation` in `test_agent.py`); broader 1650-test sweep clean.

**Bug 2 (deferred)**: `get_predator_sulfolipid_concentration` is an alias of `get_predator_concentration`. Per archived [`fix-predator-sensing-biology` Decision T3.5](../../../../../../openspec/changes/archive/2026-05-24-fix-predator-sensing-biology/design.md) this is an explicit T6/T7 deferral pending Liu et al. 2018 plate-assay calibration. Not blocking; not addressed in this work.

A separate pre-batch bug was fixed before Step B launched (commit `c25588a1`): `agent/stam.py` was missing explicit recognition of `predator_biology_klinotaxis` in `resolve_active_channels` — the composite module fell through to the legacy fallback channel, causing brain-side STAM dim (9) to disagree with env-side STAM dim (7) and crashing the first forward pass. The fix activates both `predator_mechano` and `predator_distal` channels when the composite module is in the module list. Five regression tests added to `test_stam.py`.

## Step B re-run (post-Bug-1-fix)

12 of 12 runs complete. Three variants re-run (B0.4 NOT re-run — composite drops `lateral_gradient` by design so Bug 1 doesn't affect it; A1 legacy NOT re-run — legacy gate unchanged):

| Variant | Pre-fix | Post-fix | Δ | vs Legacy 67% |
|---|---|---|---|---|
| B0.5 reward variant (canonical sensors + new reward) | 25.0% | **81.0% ± 5.0** | +56pp | **+14pp** |
| B0.3 sparse_fix sensor + default reward | 24.0% | 73.0% ± 6.0 | +49pp | +6pp |
| A2 canonical new biology + default reward | 23.0% | 65.0% ± 16.1 | +42pp | -2pp (≈ tied) |
| A1 legacy (reference) | 67.0% ± 7.6 | — | — | (reference) |

**Bug 1 fix closed the convergence gap entirely.** Two variants now beat legacy: B0.3 by +6pp, B0.5 by +14pp. The corrected biology is fully viable when its directional signal is properly wired.

## Extended validation (orthogonal stack + composite re-run)

User flagged two additional candidates worth evaluating: B0.6 (orthogonal stack of the two best individual changes) and B0.4 re-run (does composite improve post-fix, or is it genuinely structurally inferior?). 8 of 8 runs complete:

| Variant | last-25 success | death% | foods/ep |
|---|---|---|---|
| B0.6 sparse_fix + new reward (orthogonal stack) | 78.0% ± 6.9 | 22.0% | 9.03 |
| B0.4 composite (post-fix re-run) | 16.0% ± 7.3 | 83.0% | 4.70 |

Two clean findings:

**Finding 1: Orthogonal stack does NOT compound** — B0.6 (78.0%) is within noise of B0.5 (81.0%), Δ = -3.0pp. Combining sparse_fix mechano with the new reward gives slightly *worse* performance than the canonical mechano + new reward. Interpretation: once the new reward provides the dual-mechanism predator signal (continuous distal + binary contact), the sparse_fix mechano's "always-on distal fallback off-contact" becomes redundant with the reward's already-continuous distal penalty — the brain sees the same information from two channels. B0.5 is the simpler choice (biology-default mechano; only reward differs from A2).

**Finding 2: B0.4 composite is structurally inferior, not Bug-1-bound** — pre-fix 16.0% vs post-fix 16.0% (byte-identical means). The composite's failure is structural: its 4-dim output drops `lateral_gradient` by design, so Bug 1 never affected it. The 83% death rate stays put — the brain has no head-sweep directional info via the composite's encoding. Confirms the redundancy hypothesis was wrong: `lateral_gradient` IS load-bearing for klinotaxis predator-evasion.

## Final ranking (all 6 variants, n=4 × 500 ep, post-Bug-1-fix)

| Rank | Variant | last-25 success | death% |
|---|---|---|---|
| 1 | **B0.5 canonical sensors + new reward** | **81.0% ± 5.0** | **18.0%** |
| 2 | B0.6 sparse_fix sensor + new reward (orthogonal stack) | 78.0% ± 6.9 | 22.0% |
| 3 | B0.3 sparse_fix sensor + default reward | 73.0% ± 6.0 | 27.0% |
| 4 | A1 legacy nociception_klinotaxis (reference) | 67.0% ± 7.6 | 32.0% |
| 5 | A2 canonical new biology + default reward | 65.0% ± 16.1 | 32.0% |
| 6 | B0.4 composite | 16.0% ± 7.3 | 83.0% |

## Canonical decision: B0.5

**Locked canonical** = canonical two-channel new biology (`predator_mechanosensation_klinotaxis` + `predator_chemosensation_klinotaxis`) + `distal_chemo_contact_trigger` reward mode.

Rationale:

- **Beats legacy by +14pp** (81% vs 67%) — the headline result Phase 0 was investigating.
- **Tightest variance** of all variants (std 5.0 vs A1's 7.6).
- **Lowest death rate** of all variants (18% vs A1's 32%).
- **Biological fidelity**: uses the corrected sensors as designed, with all features properly wired post-Bug-1-fix. The reward shape mirrors the dual-channel sensor split — continuous distal aversion (matching the chemo channel's role) + sharp contact pain (matching the mechano channel's role).
- **Design simplicity**: sensors are biology-default (no sensor variants needed); only the reward mode differs from the A2 baseline.

Orthogonal stacking (B0.6) does not compound, so the sparse_fix sensor variant is not needed in Phase 4. B0.4 composite is structurally inferior — the redundancy hypothesis it tested (collapse 6 dims to 4) is empirically refuted; `lateral_gradient` is load-bearing for klinotaxis predator-evasion.

## Carry-forward implications

- **Phase 4 (architecture comparison) C-curriculum cells** SHALL consume the B0.5 canonical sensors + reward jointly per the spec scenario "Canonical variant locked before Phase 4 C-cells launch".
- **Phase 6 T7 L2 re-run** (env-fidelity upgrade) SHALL consume the same canonical unless the env-upgrade work plausibly invalidates the Phase 0 evidence — in which case a re-evaluation is itself a documented decision.
- **Phase 6 T8 NEAT topology search** SHALL consume the same canonical by default.
- **`predator_mechanosensation_klinotaxis_sparse_fix` and `predator_biology_klinotaxis` modules** stay shipped and registered as opt-in alternatives for future ablation work — they are not Phase 4 canonical but the empirical evidence justifying their non-selection lives in this directory.

## Reproducibility checklist

1. **STAM-dim inference** (`brain/modules.py::_infer_stam_dim_from_modules`) must include the sparse_fix mechano triple AND handle composite double-counting. Without that, the brain's first Linear layer is sized wrong and crashes with a shape-mismatch at the first forward pass. Covered by regression tests in `test_stam.py`.
2. **`predator_lateral_gradient` gating** must trigger under EITHER `nociception_mode == KLINOTAXIS` OR `predator_distal_mode == KLINOTAXIS`. Bug 1 root-cause; covered by regression tests in `test_agent.py::TestPredatorLateralGradientPopulation`.
3. **Reward-mode literal**: `distal_chemo_contact_trigger` must be accepted by `RewardConfig.reward_mode`'s `Literal` type. Covered by regression tests in `test_reward_calculator.py`.
4. **Per-seed determinism**: all variants use the same seed set (42, 43, 44, 45) and the same brain hyperparameters. Per-seed variance reflects training stochasticity, not config drift.

## Related artefacts

- [Phase 0 canonical-variant selection in `design.md`](../../../../../../openspec/changes/weight-search-architecture-ranking/design.md) — the normative selection record.
- [Spec scenario "Canonical variant selected (Phase 0 outcome)"](../../../../../../openspec/changes/weight-search-architecture-ranking/specs/predator-sensing-biology/spec.md) — the spec-level lock.
- [Archived `fix-predator-sensing-biology` Modelling caveat 6](../../../../../../openspec/changes/archive/2026-05-24-fix-predator-sensing-biology/design.md) — the original deferred question.
- [Supporting/024 smoke evaluation](../../024/smoke-evaluation.md) — the T3 100-episode smoke that surfaced the convergence-rate question.
