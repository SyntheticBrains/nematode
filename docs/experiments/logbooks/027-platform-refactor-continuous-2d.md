# 027: Platform Refactor — Continuous-2D Substrate + Continuous-Action Heads (Phase 6 Tranche 5)

**Date:** 2026-06-07
**Tranche:** Phase 6 T5 (platform refactor). **Gate:** Gate 2 (L1 plugin parity in practice).
**Change:** [`add-continuous-2d-and-action-heads`](../../../openspec/changes/add-continuous-2d-and-action-heads/).
**Status:** complete — **Gate 2 decision: GO**.

## Objective

Take the architecture-comparison platform from the discrete grid onto a
**continuous-2D substrate with continuous-action control**, without breaking the
existing brains or the L1 plugin interface, and close **Gate 2** — the
pre-registered check that adding a new architecture family is still cheap
(≤ 6 files, no per-architecture branches) on the upgraded platform. T5 is the
*platform* half of the env upgrade; env *fidelity* (T6) and the apples-to-apples
ranking (T7) build on it.

## Background

Per [`phase6-tracking` Decision 1](../../../openspec/changes/phase6-tracking/design.md),
the env upgrade is split T5 (platform) / T6 (fidelity) so Gate 2 closes against a
single verifiable platform outcome rather than a bundled fidelity tranche. The
MUST set is 5 families (connectome, MLP-PPO, LSTM-PPO, CfC-PPO, NEAT — the last
built at T8); CfC was promoted SHOULD→MUST at the 2026-06-04 checkpoint.

## What shipped

Across seven PRs (the change stays active until archived at T5 close):

| PR | Scope |
|----|-------|
| #205 | §1 shared action-policy module (`_policy.py`) + §2 MUST-brain discrete migration onto it (byte-equivalent / declared ~1e-7 tolerance) + §8.0 pull-forward of the two SHOULD rows |
| #207 | §3 `Continuous2DEnvironment` (subclasses `DynamicForagingEnvironment`; kinematic `(speed, turn)` movement, capture-radius, Euclidean distances; grid env byte-stable) |
| #209 | §4.1 tanh-squashed Gaussian helpers + §4.2 MLP-PPO continuous head |
| #210 | §4.2 LSTM-PPO + CfC-PPO continuous heads (recurrent buffers / BPTT) |
| #211 | §4.3 connectome continuous-output adapter + §4.4 strict-mask invariant tests |
| #212 | §5 Transformer (temporal-window) PPO brain — the Gate-2 parity vehicle |
| #213 | §6 G2.d floor check + continuous-PPO entropy calibration |

Continuous tanh-squashed Gaussian heads now exist on **5 brains**: MLP-PPO,
LSTM-PPO, CfC-PPO, connectome-PPO, and the new Transformer. Brains emit a
*normalized* `(speed, turn)`; the environment rescales to physical units
(decouples brains from env scale; PPO-equivalent).

## Gate 2 verification (G2.a–G2.d)

- **G2.a — engineer-hours (documented, NOT load-bearing).** The Transformer
  addition (the measured new family) was a single focused implementation, well
  under the informal "≤ 1 week" target. Recorded as the post-refactor baseline
  for future additions, not a verdict on this one (no pre-refactor baseline
  exists). **Met (documented).**
- **G2.b — files-touched ≤ 6.** The Transformer-addition commit touched **5
  files** (`git show --name-only`): new module, `dtypes.py` enum, `brain/arch/__init__.py`,
  `config_loader.py` `BrainConfigType` union, test. No `_build_infra_kwargs`
  branch (default infra shape); `BRAIN_CONFIG_MAP` auto-derives from the registry.
  The CLI-default `match brain_type` block was refactored to a registry-driven
  default in a **separate prior commit** so it is not counted. **PASS.**
- **G2.c — no per-architecture branches.** The Transformer routes through the
  registry + the env-mode action dispatch; no `isinstance(brain, …)` was added to
  the simulation/training loops, and the entrypoint default is registry-driven.
  **PASS.**
- **G2.d — continuous-substrate floor check.** Connectome + MLP-PPO train without
  breaking on continuous-2D klinotaxis. **PASS** (with the entropy-0.10
  calibration below). Evaluation: scratchpad R5/R6 in
  `tmp/evaluations/continuous-2d-action-heads/`.

### G2.d detail (the floor check did real work)

The literal "mean episode return ≥ 50% of grid" criterion proved **confounded**
and was not used: on the grid a converged agent eats its 10-food target and the
episode ends early (short, food-dominated return ~35), while on continuous-2D
agents rarely hit the target and run the full ~200-step budget accruing shaping
(return ~130) — the ratio "passes" trivially while foraging is ~3–5% of grid. We
evaluated G2.d per its stated **"didn't break training"** intent (a learning
signal with no mid-training collapse), agreed with review.

Running it surfaced a real **continuous-training instability**: at the
discrete-tuned entropy (0.03) the continuous tanh-Gaussian's `log_std` collapses
to the clamp floor mid-training (exploration dies) and the policy collapses — the
**MLP canary itself exhibits it**, so it is substrate-level, not per-architecture
(same hyperparameters converge cleanly on the discrete grid). Reducing epochs/lr
made it *worse* (refuted). **Fix (config-only, no `_policy.py` change): continuous
PPO needs a higher entropy_coef (~0.10)** — the entropy bonus holds `log_std` off
the floor. At 0.10:

| Arch | Continuous-2D (entropy 0.10, 500-ep, seed 0) | Grid baseline |
|------|----------------------------------------------|---------------|
| MLP-PPO | ramps ~0 → ~0.5, sustains ~0.37 foods/ep, no collapse | 9.87 foods/ep, +35.67 |
| Connectome-PPO | ramps 0.12 → ~0.3 foods/ep, no collapse, strict-mask preserved | 9.52 foods/ep, +33.44 |

Foraging is far below grid (continuous-2D is genuinely harder: capture radius +
continuous control) — but G2.d is a floor, not a ranking; the full multi-brain ×
multi-seed continuous ranking is **T7**.

## Gate 2 decision: **GO**

All four criteria are met (G2.a documented; G2.b/G2.c PASS; G2.d PASS with the
entropy calibration). Neither PIVOT trigger fired (G2.b and G2.c both passed) and
the STOP trigger does not apply (no MUST family is interface-incompatible — the
connectome, the highest-risk family, added its continuous head as a readout swap
with the strict-mask/gap-junctions provably untouched). **Proceed to T6.**

## Key findings

1. **L1 plugin parity holds in practice on the upgraded platform** — a real new
   architecture (Transformer) added in 5 files, no per-arch branches. The
   continuous-action contract did not leak a per-arch branch (the G2.c risk):
   action mode flows through `BrainConfig` + the env-mode dispatch, not the brain
   class.
2. **Continuous PPO needs ~3× the entropy of discrete** to avoid `log_std`
   exploration collapse. This is the load-bearing tuning lesson for T7.
3. **Continuous-action design**: tanh-squashed Gaussian with Jacobian-corrected
   log-prob; brains emit normalized actions, the env rescales (substrate-decoupled,
   PPO-equivalent); `pre_tanh` stored for byte-stable PPO re-scoring; `log_std`
   clamped for finiteness.
4. **Connectome continuous adapter = readout swap only** — `(4,4)→(2,4)` mean +
   state-independent `log_std`; the chemical strict-mask and fixed gap junctions
   are upstream of the readout and byte-identical across output modes
   (test-covered). This retires the roadmap's stated T5 failure mode
   ("continuous-output adapter incompatible with the strict-mask").
5. **Transformer = temporal-window self-attention** (window-as-state → reuses the
   shared shuffled-minibatch buffer, no BPTT chain). It learns cleanly on discrete
   klinotaxis (reward 5.2 → 36.2; foods 3.1 → 9.9/10) — a strong temporal
   comparator, not a throwaway parity vehicle.

## Deferrals & known limitations

- **§3.4 float source placement + Euclidean pheromone/predator-mechano distance
  → T6.** Sources stay on the integer lattice within the continuous arena (worm +
  capture are continuous); predator-mechano distance stays Manhattan (shared with
  the grid path; predators-on-continuous not yet exercised). Env-fidelity work.
- **§8.1 SHOULD/MAY continuous heads → T7-prep (after T6).** Not front-loaded:
  T6 shifts the continuous substrate, so adding+tuning them now would force
  re-tuning. The cheap path is banked (§8.0 migrated the two SHOULD rows onto the
  shared `_policy`; the entropy-0.10 lesson transfers).
- **LSTM/CfC/Transformer continuous arms — multi-seed tuning → T7.** Wired +
  smoke-validated; their continuous configs still sit at the older (collapse-prone)
  entropies pending the entropy-0.10 carry-over + multi-seed validation.
- **Many-worlds on the continuous substrate** — `env.copy()` plumbing exists but
  the runner path is `NotImplementedError`-guarded; enablement is future work.
- **Broader PPO-family discrete consolidation** ([#204](https://github.com/SyntheticBrains/nematode/issues/204))
  and **god-class env decomposition** ([#206](https://github.com/SyntheticBrains/nematode/issues/206))
  — tracked as GitHub issues, out of T5 scope.

## Conclusions

Gate 2 is **GO**. The platform refactor delivered a continuous-2D substrate +
continuous-action heads on all current MUST brains, verified L1 plugin parity in
practice (5-file Transformer addition, no per-arch branches), and confirmed the
substrate doesn't break training (with a continuous-PPO entropy calibration). The
deliberate T5/T6 split paid off — the parity outcome is clean and unbundled from
fidelity tuning.

## Next steps

- **T6 (env fidelity):** signal-specific static Fick gradients + adaptive/biphasic
  chemosensory sensor; pick up the §3.4 float-source-placement + Euclidean-field
  deferrals.
- **T7-prep:** add SHOULD/MAY continuous heads on the finalized substrate; carry
  the entropy-0.10 lesson; multi-seed-tune all continuous arms for the ranking.

## Data references

- Change: [`add-continuous-2d-and-action-heads`](../../../openspec/changes/add-continuous-2d-and-action-heads/) (PRs #205, #207, #209–#213).
- G2.d evaluation trail: `tmp/evaluations/continuous-2d-action-heads/` (R1–R6).
- Configs: `configs/scenarios/foraging/{mlpppo,connectomeppo,lstmppo,cfcppo,transformerppo}_small_continuous2d_klinotaxis.yml` (+ the transformer grid klinotaxis config).
- Grid baselines: `tmp/evaluations/connectome-ppo-gate1-evaluation/` (MLP 9.87 foods/+35.67; connectome 9.52/+33.44).
