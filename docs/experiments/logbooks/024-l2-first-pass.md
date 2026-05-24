# 024: L2 First Pass — Grid Substrate × Four MUST Architectures × Three Behaviours

**Status**: stub. The T3 corrected biology-driven predator sensing surface lands first as the env-correctness prerequisite for the L2 first pass (per phase6-tracking design.md Decision 1). The full L2 evaluation (four MUST architectures × three behaviours × n ≥ 4 seeds, with the gradient_proximity-vs-distal-chemo reward ablation per T4.0g) is the body of this logbook and is authored when T4 work begins.

**Branch**: `feat/predator-sensing-biology` (T3) → `feat/l2-first-pass` or similar (T4, not yet created).

**OpenSpec changes**: `fix-predator-sensing-biology` (T3, this PR) → `add-l2-first-pass` or similar (T4, not yet created).

**Date Started**: 2026-05-24 (T3 prerequisite section).

**Date Completed**: TBD (T4 body completes the logbook).

This stub records the T3 prerequisite work and the empirical findings T3 surfaced that T4 must consume. The L2 first-pass cells themselves (4 architectures × 3 behaviours, with all the canonical T4 ablations) land below the T3 section as a separate body when T4 begins.

## T3 prerequisite: Corrected biology-driven predator sensing

### Objective

Replace the single chemosensory `nociception_klinotaxis` predator-detection channel — which Logbook 011 flagged as biologically wrong for the contact-based ASH/ALM/AVM/PLM nociception circuit — with a biologically-grounded two-channel model. Land the corrected env + BrainParams + STAM + sensor-module surface so the T4 L2 predator-evasion cells consume the right signal from the start instead of being re-run later.

### What shipped

Implementation across 11 commits on `feat/predator-sensing-biology`. Full architectural surface in [`fix-predator-sensing-biology` OpenSpec change](../../../openspec/changes/archive/) (archived post-merge).

**Two new sensor channels**:

- **Contact-mechanosensory** (ASH / ALM / AVM / PVD / PLM) via three new modules (`predator_mechanosensation_oracle` / `_temporal` / `_klinotaxis`). Emits graded `predator_contact_intensity ∈ [0, 1]` (replaces the boolean `predator_contact` for the new channel — legacy field kept frozen) and a new `predator_contact_zone ∈ {NONE, ANTERIOR, POSTERIOR, LATERAL}` enum derived from the predator's bearing-vs-heading on the grid.
- **Distal-chemosensory** (ASH + ASI sulfolipid signal per Liu et al. 2018, *Nat. Commun.*) via three new modules (`predator_chemosensation_oracle` / `_temporal` / `_klinotaxis`). Emits `predator_distal_concentration` (env-side alias of the existing exp-decay sum, framed as the Liu 2018 sulfolipid analogue with literature-calibrated decay deferred to T6/T7) and `predator_distal_dconcentration_dt` (STAM-computed temporal derivative).

**Naming convention shift**: the new oracle variants carry an explicit `_oracle` suffix (`predator_mechanosensation_oracle`, etc.), departing from the legacy bare-named convention (`food_chemotaxis`, `nociception`) that's a historical accident from before multi-mode support. The legacy modules stay frozen-in-place for the 22 archived Phase 5 evolution configs.

**STAM channel split**: `predator` channel splits into `predator_mechano` + `predator_distal` so habituation kinetics (matching Hilliard et al. 2005's seconds-to-minutes ASH adaptation timescale) ride on the existing STAM exponential-decay buffer per channel. Legacy `predator` channel kept as deprecated alias for configs that select legacy `nociception_*` modules.

**Test coverage**: 38 ContactZone-discrimination tests + 24 sensor-module extraction tests + 45 archived-config regression assertions all green.

### Behavioural difference under matched conditions (phase6-tracking T3.2)

100-episode head-to-head smoke evaluation on `mlpppo_small` and `lstmppo_small` predator-pursuit configs, seed 2026, identical env / brain hyperparameters / sensing modes except for the sensor module names:

| Brain | Sensors | Success | Foods/ep | Reward/ep |
|---|---|---|---|---|
| **MLPPPO legacy** | `nociception_klinotaxis` (3-dim) | **51%** | 7.27 | +22.87 |
| MLPPPO new biology | mechano+chemo klinotaxis (6-dim) | 3% | 2.80 | +0.74 |
| **LSTMPPO legacy** | `nociception_klinotaxis` (3-dim) | **7%** | 2.25 | -1.76 |
| LSTMPPO new biology | mechano+chemo klinotaxis (6-dim) | 0% | 0.93 | -6.83 |

Full results + per-episode breakdown in `tmp/evaluations/predator-sensing-biology-smoke/predator-sensing-biology-smoke_scratchpad.md`.

### Verdict

The corrected biology surface ships and works end-to-end: no crashes, gradient flow, training visible, MLPPPO's success-rate moves from 0% (episode 1) to 3% (episodes 98-100 contain 2 successes). The convergence-rate gap vs the legacy `nociception_klinotaxis` is a real empirical finding worth recording but **not a T3 blocker** — T3's deliverable was the corrected biology surface, and that surface is biologically faithful, well-tested, and confirmed to work end-to-end. The convergence-rate question is the right T4 empirical evaluation.

### T4 carry-forwards

Three explicit T4 sub-tasks added to [phase6-tracking/tasks.md](../../../openspec/changes/phase6-tracking/tasks.md) following from T3 evidence:

- **T4.0g** — investigate the new-biology predator-sensing convergence-rate gap. Run new-biology cells at canonical T4 compute budget; ablate the sparse-contact-signal hypothesis (try injecting distal sulfolipid concentration into the mechano-strength field when not in contact); ablate the information-redundancy hypothesis (try a single composite predator module emitting 4 dims instead of two parallel 3-dim modules); decide whether the gap is acceptable substrate-finding or whether sensor encoding needs revisiting before T7.
- **T4.\*.predator_evasion_reward_ablation** — four new sibling sub-tasks (one per MUST architecture row) comparing the existing `gradient_proximity` reward against a proposed `distal_chemo_penalty + binary_contact_damage_trigger` variant. The joint sensor-encoding × reward-shape ablation space (T4.0g × these four rows) is the canonical T4 way to evaluate the new biology under matched compute.

The reward-shape ablation lives at T4 (not T3) because reward changes alter the training signal everyone learns against; doing both sensor + reward shape changes in T3 unilaterally would conflate two confounded effects.

### Modelling caveats inherited by T4

Documented in [fix-predator-sensing-biology design.md § Modelling caveats](../../../openspec/changes/archive/) (archived post-merge):

1. ContactZone is bearing-vs-heading on a 1-cell grid agent, not anatomical head-vs-tail. T5/T7 may revisit when continuous-2D body extent exists.
2. Distal-chemo decay constant is the placeholder exp-decay sum, not Liu et al. 2018-calibrated. T6/T7 owns the parameter sweep.
3. SpikingReinforceBrain stays on the legacy oracle predator gradient (bypasses the sensory_modules pipeline by design). Known asymmetry in the T4 comparison.
4. Reward stays env-coupled in T3; biological-faithfulness reward-shape ablation is the T4.\*\_reward_ablation work above.
5. ADL ascaroside pheromone channel deferred to T6/T7.
6. New biology learns substantially slower than legacy `nociception_klinotaxis` at 100-episode budget. Convergence-rate study is T4.0g.

______________________________________________________________________

## L2 First-Pass Body (TBD — authored when T4 begins)

The full L2 evaluation (four MUST architectures × three behaviours × n ≥ 4 seeds + the T4.0g convergence-rate study + the T4.\*\_reward_ablation rows + the T4.0c sensor-projection ablation choice for connectome) lands as the body of this logbook below this stub when T4 starts. The T3 prerequisite section above becomes the "Background" or equivalent.
