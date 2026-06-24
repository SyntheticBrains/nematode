# 029: Cross-Architecture Ranking on the Continuous-2D Substrate (T7)

**Status**: integrated-C3 ranking complete for the **6 MUST architectures** (n=8 paired seeds, uniform
6000ep + a 5-seed 8000ep convergence top-up). T7 is **NOT closed** — SHOULD/MAY arms and a memory-bound
control are deferred to a follow-up (see Next Steps).

**Branch**: `openspec/t7-n8-ranking`.

**OpenSpec changes**: the T7-prep arc (`add-continuous-2d-and-action-heads`, `add-continuous-predator-kinematics`,
`extend-fick-chemical-fields`, the continuous reward/sensing fixes); **`add-level-agnostic-convergence-metric`**
(#250, the ranked-metric fix this logbook depends on); ranking methodology per `architecture-comparison-protocol`.

**Date**: 2026-06-24.

______________________________________________________________________

## Objective

The **T7 (high-fidelity continuous-physics) analogue of Logbook 025** — rank the brain-architecture
families on the integrated **C3 cell** (food chemotaxis + predator evasion + thermotaxis, all active
under klinotaxis sensing), now on the **continuous-2D substrate** (float kinematics, Euclidean
geometry, Fick fields, adaptive sensor) rather than the discrete grid. Three questions:

1. Where does the **wild-type connectome** rank against the trained-net and gradient-free baselines on
   high-fidelity integrated control? (Gate 3 G3.b.)
2. Which architecture juggles the three pressures best on the continuous substrate, and how reliably?
3. Does the higher-fidelity cell **discriminate** the architectures, or reproduce T4's flat top-cluster tie?

## Background

The C3 cell is the load-bearing comparison: an identical env + reward + satiety block across all arms —
only the brain differs. The continuous difficulty is calibrated **on its own terms** (predator count 2 /
speed 0.4 / detection 15, steep predator field for a real flee-direction signal; thermal linear gradient
0.4; `distal_chemo_contact_trigger` reward; 60 mm world) — *not* number-matched to T4, which is
non-commensurable (the two-regimes caveat, `phase6-tracking`). So the T4↔T7 comparison is **qualitative**
(does the repertoire/ranking character survive the fidelity jump), and the within-T7 ranking is the clean
result.

**Ranked metric.** Full-clear (10/10-food) success over the **plateau tail** (final-quarter mean), with
the level-agnostic `post_convergence_success_rate` (#250) as the per-seed convergence verdict. The metric
fix was load-bearing: the prior detector required a near-100% window, so on the T7 *sub-saturation* band
(arms deliberately spread ≈35–90%) it returned null for sub-50% plateaus and silently fell back to a
noisy last-10-run mean — which mis-ranked arms before #250. Statistics: paired-seed one-sided Wilcoxon
(seeds 1–8), 80% bootstrap CIs (1000 resamples, seeded), BH-FDR across the pairwise set — the
`architecture-comparison-protocol` standard, identical to Logbook 025.

## Hypotheses

- **H1 (connectome)**: mid-pack — competitive on foraging, behind the trained nets on predator evasion
  (its T4 evasion-lag; fixed wiring, no learned recurrence). Carried from 025.
- **H2 (discrimination)**: the calibrated sub-saturation cell **discriminates** the arms (unlike T4's
  flat ~84% top-cluster tie), because the difficulty is set below saturation by design.

## Method

**6 MUST arms, n=8 (seeds 1–8):** MLP-PPO, Transformer-PPO, CfC, LSTM-PPO (GRU), connectome-PPO
(strict chemical-synapse mask + fixed gap junctions + predator/thermo projections), FeedforwardGA.
Per-arch recipes were tuned **on the integrated cell** (recipes do not port between cells); the GA trains
via `run_evolution.py` (no `--track-experiment` JSON) and is scored by its evolved-champion full-clear
rate over a frozen eval.

**Budget — uniform 6000ep (+5-seed 8000ep top-up).** The arms learn at very different *rates* on this
hard cell (MLP plateaus by ~3000ep; the others climb far longer), so a budget that plateaus only the fast
learner under-ranks the slow ones. Budget was raised iteratively (3000 → 4000 → uniform 6000ep) once a
convergence audit showed slow-climbers; a final targeted bump of the 5 still-climbing seeds to 8000ep
confirmed all converged. Non-uniform budgets are protocol-allowed (each arm ranked on its own plateau);
the plateau-tail metric normalises across budgets. Detail + the budget-iteration story:
[supporting/029](supporting/029-continuous-architecture-ranking/details.md).

**Analysis harness**: `scripts/analysis/t7_continuous_ranking.py` (reuses the committed
Wilcoxon/bootstrap/BH-FDR helpers from `weight_search_architecture_ranking.py`) +
`scripts/analysis/t7_ga_champion_eval.py`.

## Results

### Cross-architecture C3 ranking (plateau-tail full-clear, n=8)

| rank | architecture | success % (mean ± sd) | converged | foraging (foods) | evasion (%) | thermal (comfort) |
|---|---|---|---|---|---|---|
| 1 | **mlpppo** | **89.0 ± 7.3** | 7/8 | **9.59** | **80.5** | **0.775** |
| 2 | cfcppo | 75.8 ± 9.7 | 7/8 | 9.00 | 69.0 | 0.716 |
| 2 | transformerppo | 74.0 ± 7.7 | 6/8 | 8.62 | 65.5 | 0.754 |
| 4 | lstmppo | 60.1 ± 15.8 | 5/8 | 8.24 | 60.9 | 0.706 |
| 5 | connectomeppo | 52.2 ± 9.0 | 8/8 | 7.79 | 50.8 | 0.614 |
| 6 | feedforwardga | 15.0 ± 16.4 | n/a | — | — | — |

**Three significant tiers (BH-FDR).** MLP is the unambiguous leader — significant over every arm
(q = 0.007–0.024 \*\*\*). CfC and Transformer are a **statistical tie for #2** (Δ+1.8, q=0.53 ns), both
significantly above everything below. LSTM (4th) is not separable from connectome (q=0.21 ns) and is the
**stability laggard** (5/8 converged; its non-converged seeds *wobble*, not climb). The connectome (5th)
is significantly below the trained-net cluster and significantly above the GA. GA is the significant floor
(every arm beats it, q=0.007 \*\*\*).

### Connectome verdict (Gate 3 G3.b)

The wild-type connectome (52.2%, **8/8 converged**) ranks **5th of 6** — significantly below the
trained-net cluster (vs Transformer/CfC Δ−21.8 / −23.6 \*\*\*) but significantly above gradient-free
search (vs GA Δ+37.2 \*\*\*). Its deficit is specifically **predator evasion** (50.8%, the field's lowest)
— exactly its T4 evasion-lag, now confirmed on the high-fidelity substrate. **H1 confirmed.** This is a
real mid-low rank, **not** a "failed to train" STOP: it learns the integrated cell (52%, with the
**highest converged fraction of any arm — 8/8 seeds**).

### What survives the fidelity jump (T4 → T7, qualitative)

T4's integrated cell produced a **flat ~84% four-way top-cluster tie** (Logbook 025). The T7 cell, set
below saturation, **discriminates** — MLP separates as a clear leader, a CfC/Transformer #2 pair forms,
and LSTM/connectome/GA spread into distinct lower tiers. **H2 confirmed.** The connectome-mid-pack and
GA-floor character carries over; the within-cluster ties resolve. (The reactive-task / memory caveat from
025 still applies — see Limitations.)

## Analysis

**The result only became trustworthy after two methodology fixes — both caught by deliberate dig-ins:**

1. **The ranked metric (#250).** The convergence detector required a ≈10/10 window (variance < 0.05 ∧
   mean ≥ 0.5) — fine at T4's 73–84% band, but on T7's sub-saturation band it returned null for sub-50%
   plateaus, which then silently fell back to a *last-10-run mean*. That fallback corrupted the ranking
   (connectome read 20% vs its true ~48% plateau; a Transformer seed read 100% off a lucky streak).
   Replaced with a level-agnostic plateau detector + a plateau-tail ranked metric; a per-seed
   final-window cross-check guards it.
2. **The budget.** A convergence audit found the arms learn at *very different rates*. At 3000–4000ep
   several arms (CfC especially — all its non-converged seeds) were **still climbing**, so the early
   ranking under-ranked the slow learners. A uniform 6000ep re-run lifted **CfC 65 → 76** (moving it from
   #3 to the #2 tie) — the under-ranking was real. A final 8000ep top-up of the 5 remaining climbers
   confirmed convergence; the order was unchanged and the values barely moved (CfC 76.3 → 75.8), so the
   6000ep ranking was already sound.

**Seed-stability is itself a finding, not tuned away.** Per-arch exploration/exploitation tuning was
already done twice (T7-prep + the integrated-cell pass); the residual seed-basin variance is documented as
*only partly dial-able*, so the protocol *reports* it (converged-fraction + per-seed) rather than chasing
8/8. LSTM is the clear stability laggard (5/8, one collapse-to-25 seed); connectome/CfC/MLP converge
cleanly (8/8, 7/8, 7/8). **A phantom recipe** surfaced en route: the MLP config's entropy schedule
(`entropy_coef_end`/`entropy_decay_episodes`) was silently dropped — `mlpppo` never implemented it, so MLP
ran flat `entropy 0.08` throughout (it is still the leader). That class of silently-dropped config key is
now caught at load time (#253) and the wider debt tracked (#254).

## Conclusions

1. **Ranking: MLP 89 ≫ {CfC 76 ≈ Transformer 74} > LSTM 60 > connectome 52 ≫ GA 15** — three significant
   tiers, robust across three budget levels and the 8000ep top-up.
2. **MLP is the clear leader** and best on all three behaviours (forages 9.6/10, evades 80%, best thermal).
   On this *reactive* cell, the all-rounder feed-forward net wins; recurrence/attention/biology don't beat it.
3. **CfC ≈ Transformer for #2** (statistical tie); both significantly clear of the rest.
4. **Connectome ranks 5th** — competitive-ish foraging, behind on predator evasion (its T4 lag persists),
   significantly below the trained nets but significantly above gradient-free search. Learns the cell
   (52%, 8/8); not a STOP. **H1 confirmed.**
5. **GA collapses to the floor** (15%) — gradient-free weight search does not solve the integrated cell
   (reproduces T4's ~0–collapse; optimiser-fundamental, action-space-agnostic).
6. **The sub-saturation cell discriminates** where T4's flat cell tied (**H2 confirmed**) — the deliberate
   below-saturation difficulty is what makes the ranking meaningful.
7. **Methodology**: the trustworthy ranking required the level-agnostic metric (#250) + a fair uniform
   budget; two dig-ins caught a fallback-contamination bug and a slow-climber under-ranking before they
   reached this writeup.

## Limitations

- **Reactive-task / memory caveat (carried from 025).** Klinotaxis foraging is a reactive hill-climb with
  a one-step temporal derivative (STAM) every arm shares, so the cell under-tests the working-memory axis
  on which these architectures most differ. MLP-leads is consistent with that: nothing here rewards
  recurrence. A memory-bound positive control is the planned probe (Next Steps).
- **Seed fragility.** Continuous tanh-Gaussian PPO is seed-fragile; LSTM (5/8 converged) especially. The
  ranking handles this via n=8 + per-seed/converged-fraction reporting, not by forcing convergence.
- **T4↔T7 non-commensurable.** The substrates differ on many axes at once; the cross-regime comparison is
  qualitative, not a controlled single-variable delta.
- **GA action space.** The GA runs discrete (4-way) via the env's discrete-action fallback (no continuous
  head) — its identity as the gradient-free arm, but an action-space asymmetry vs the continuous-control
  PPO arms (it ranks last regardless; the ceiling is optimiser-fundamental).
- **Config debt.** ~71 configs carry silently-dropped brain-config keys (notably `normalize_advantages`
  on 60+ mlpppo configs — never implemented); pre-existing, tracked in #254.

## Next Steps

- **SHOULD/MAY arms** (quantum / spiking / reservoir / hybrid) — opportunistic continuous-substrate
  bring-up, decided *after* this MUST ranking (non-gating; `T7.prep.should_may_continuous`). Quantum is
  settled-negative at T4 (025) so low priority.
- **Memory-bound control** — an explicitly-artificial bit-memory / area-restricted-search task to confirm
  the comparison *can* separate working memory in principle (the open hypothesis from 025).
- **Config debt** (#254): decide implement-vs-remove for `normalize_advantages` (a re-validation-grade
  decision, since it would shift all MLP results).
- **T7 closure** is deferred until the above are resolved — this logbook records the MUST ranking, not the
  tranche close.

## Data References

- **Analysis outputs**: [supporting/029](supporting/029-continuous-architecture-ranking/) —
  `ranking-per-seed.csv`, `per-behaviour.csv`, `cross_arch_pairwise.json` (Wilcoxon/bootstrap/BH-FDR),
  `ga_c3_results.json`, and `details.md` (budget-iteration story + per-seed analysis).
- **Harness**: `scripts/analysis/t7_continuous_ranking.py` + `scripts/analysis/t7_ga_champion_eval.py`.
- **C3 configs**: `configs/scenarios/foraging_predator_thermal/*_small_continuous2d_combined_klinotaxis.yml`
  (5 PPO arms) + `configs/evolution/feedforwardga_small_continuous2d_combined_klinotaxis.yml`.
- **Ranked-metric fix**: archived `2026-06-21-add-level-agnostic-convergence-metric` (#250).
