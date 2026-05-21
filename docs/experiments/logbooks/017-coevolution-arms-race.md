# 017: Co-Evolution Arms Race — M5 (CoevolutionLoop + Red Queen primitives + screen-sweep pilot)

**Status**: `STOP` — M5 closes without Red Queen verdict. The screen-sweep pilot (13 single-seed lever ablations + an R1 re-audit) decisively falsified strict Red Queen entanglement at this substrate: own-vs-cross fitness lag delta landed at +0.017 to +0.024 across every candidate that produced a full champion-archive snapshot (8 of 13 screens, including the GA optimiser fallback design D2 reserved for "if CMA-ES diversity collapses"). Methodology contributions (lag-matrix, cell-grid, fair-test instruments) ship intact and motivate M6.5 NEAT as the natural next Red Queen attempt.

**Branch**: `feat/m5-coevolution-pr7-logbook`.

**OpenSpec change**: archived as `2026-05-15-add-coevolution-arms-race` (no spec sync per STOP convention; the screened-but-untested specs remain in the change archive for reference rather than promoted to main specs).

**Date Started**: 2026-05-08 (substrate work began earlier under PR 1-5; pilot screening began 2026-05-09).

**Date Last Updated**: 2026-05-15 — M5 closed STOP, logbook + roadmap synced, change archived.

This logbook covers M5 across PRs 1–7:

- **PRs 1–5**: substrate ship (predator MLPPPO brain + dispatcher + encoder + fitness, `CoevolutionLoop` orchestrator with alternating-K-block schedule + 70/30 HoF mix, Red Queen metric primitives, configs + warmstart bundles + aggregator). Logbook coverage of these is brief — the substrate landed clean, the screening exercised it, and the screening's results are the headline finding.
- **PR 6**: pilot screening trajectory + substrate revisions discovered mid-flight. The originally planned 2-seed × 30-gen pilot expanded into a 13-screen lever sweep (plus an R1 re-audit under modern probe semantics) across env knobs, optimiser choice, alternation cadence, predator bootstrap, and prey regularisation when the canonical pilot's verdict gate produced ambiguous signal.
- **PR 7 (this one)**: synthesis + STOP verdict + closure.

## Objective

Demonstrate Red Queen dynamics in a co-evolved prey/predator system using the M3-validated LSTMPPO+klinotaxis prey substrate against a new learnable MLPPPO predator. The original gate (softened-disjunctive per design D6): GO if **(a) phenotypic cycling** OR **(b) trait escalation** fires in ≥2 of 4 seeds over a 30–70 generation campaign. Pilot was 2 seeds × 30 gens × pop 24/16 × K=10 alternating with HoF + generality probe.

A successful M5 would have been the headline scientific milestone for Phase 5 — published Red Queen demonstrations (Sakana DRQ 2025; Stanley & Miikkulainen 2002 NEAT-vs-NEAT predator-prey; Sci Reports 2026 trait-decoupling) all rely on architectures that evolve together, and our LSTMPPO+klinotaxis prey is more biologically grounded than the substrates those papers use. The hypothesis: if Red Queen reliably emerges in simpler substrates, it should be at least as detectable in a richer one.

## Background

Phase 5 had already shipped:

- **M0** (evolution framework — `EvolutionLoop`, `GenomeEncoder` Protocol, `LearnedPerformanceFitness`, Lamarckian inheritance)
- **M1** (predator-brain refactor — `PredatorBrain` Protocol seam, byte-equivalent against the legacy heuristic predator)
- **M2** (hyperparameter evolution — RQ1 closed, TPE wins for hyperparam, CMA-ES wins for weights)
- **M3** (Lamarckian inheritance pilot — +47pp on 4 seeds; the strongest concrete Phase 5 result)
- **M4** (Baldwin inheritance — STOP at single-task substrate; the diagnosis pointed at "task distributions" as the missing axis)

M5's framing (co-evolution as a task-distribution generator) followed directly from M4's STOP diagnosis. The agent's "task" under co-evolution is "survive the current predator population" — that population shifts every K predator generations, supplying exactly the non-stationarity Fernando 2018 / Chiu 2024 use to surface Baldwin signal. M5 was thus positioned to provide both a Red Queen finding (primary) and a secondary Baldwin readout (the dropped M5.7 instrumentation discussed below).

## What shipped

### Substrate (PRs 1–5)

| Component | File | Purpose |
|---|---|---|
| Learnable predator brain | [`env/mlpppo_predator_brain.py`](../../../packages/quantum-nematode/quantumnematode/env/mlpppo_predator_brain.py) | MLP + PPO actor-critic; 11-D input (predator pos + k_nearest=2 agent positions + step-fraction); 5-way action (STAY/UP/DOWN/LEFT/RIGHT) |
| Predator-brain dispatcher | [`env/env.py:_build_predator_brain`](../../../packages/quantum-nematode/quantumnematode/env/env.py) | YAML `predators.brain_config.kind ∈ {heuristic, mlpppo_predator}` |
| Predator factory + encoder + fitness | [`evolution/_predator_brain_factory.py`](../../../packages/quantum-nematode/quantumnematode/evolution/_predator_brain_factory.py), [`evolution/predator_encoders.py`](../../../packages/quantum-nematode/quantumnematode/evolution/predator_encoders.py), [`evolution/predator_fitness.py`](../../../packages/quantum-nematode/quantumnematode/evolution/predator_fitness.py) | `MLPPPOPredatorEncoder` (subclasses `_ClassicalPPOEncoder`), `PredatorEpisodicKillRate` (mean kill-rate over N frozen-weight episodes; secondary proximity fallback) |
| Hall-of-Fame buffer | [`evolution/hall_of_fame.py`](../../../packages/quantum-nematode/quantumnematode/evolution/hall_of_fame.py) | Bounded deque with quality-eviction; opposition mix `mix_with_pop(rng, pop, frac_hof=0.3)` |
| Red Queen metric primitives | [`evolution/redqueen_metrics.py`](../../../packages/quantum-nematode/quantumnematode/evolution/redqueen_metrics.py) | Phenotypic cycling (autocorrelation + FFT), trait escalation (linear-regression slope/SE), fitness lag, coupled rate, generality |
| `CoevolutionLoop` | [`evolution/coevolution.py`](../../../packages/quantum-nematode/quantumnematode/evolution/coevolution.py) | Two-side alternating-K-block orchestrator, opposition sampling, generality probe, per-side champion archive + inheritance plumbing |
| Configs | [`configs/evolution/coevolution_{pilot_arm_a,pilot_arm_b,full,smoke}.yml`](../../../configs/evolution) | Pretrain on/off pilot arms, full-run scaffold, smoke validation |
| Aggregator | [`scripts/campaigns/aggregate_m5_pilot.py`](../../../scripts/campaigns/aggregate_m5_pilot.py) | Per-seed cycling + escalation + generality + wall-time reconciliation |

Test coverage: 430 unit tests across the substrate (predator brain conformance, encoder round-trip, fitness conformance, HoF eviction/sampling, CoevolutionLoop K-block boundaries + probe cadence + HoF push, Red Queen metric correctness on synthetic series). `uv run pytest` smoke runs clean. Spec deltas validated `--strict`.

### Substrate revisions landed during PR 6 (mid-pilot)

The screening pivot surfaced gaps that required code changes outside the original "run-only" PR 6 scope. Each amends a design.md decision:

1. **Probe-semantics gaps 1+2+3** ([commits](https://github.com/SyntheticBrains/nematode/commits/feat/m5-coevolution-pr6-pilot) `caa7848f`, `96dec3ea`, `cea7ef7f`): in-run prey-side probe now uses `EpisodicSuccessRate` (frozen-weight) not `LearnedPerformanceFitness` (would let prey adapt at measurement time); probe-env explicitly overrides training-env predator settings to enforce the calibrated probe; K-block elite `.pt` copied to `champion_archive/` outside the inheritance dir GC scope so the probe loads post-PPO-trained weights instead of raw CMA-ES samples.
2. **Predator PPO substrate + Lamarckian opt-in** (commits `9e5394a6`, `e7c60f36`, `01cc03e0`): `MLPPPOPredatorBrain` gains `enable_learning=True` mode (Adam + RolloutBuffer + standard PPO hyperparams); `_build_predator_state` reads `predator_evolution.inheritance ∈ {none, lamarckian}` from YAML; multi-agent runner's per-step `predator.brain.learn(reward, episode_done)` hook fires inside the predator-learning pass when enabled. Reverses design.md D13's "predator = frozen-weight" decision for the YAML-configurable case (D13 default unchanged).
3. **CMA-ES persistence across K-blocks** (commit `caa7848f`): new `predator_evolution.persist_cma_across_kblocks: bool = false` knob; default behaviour (reset on K-block boundary) unchanged.
4. **GA optimiser allowance** (commit `02dac939`): `CoevolutionConfig._validate_invariants` previously hardcoded `algorithm == "cmaes"` on both sides — a stale guard that blocked the GA optimiser reserved by design D2. Relaxed to `algorithm ∈ {cmaes, ga}`; TPE still rejected. Unblocks the C16 GA screen.
5. **PR-review-driven fixes** (commit `e40d3f38`): brain `learn()` episode-end flush bug (partial buffers stranded when terminal `learn(episode_done=True)` arrived with no pending transition); shared persistent-brain guard in `predator_fitness.py` (multi-predator scenarios with `inheritance: lamarckian` had silently corrupted per-slot rollout buffers — guard now fails fast with `ValueError`); `cma_diagonal` validator gated behind `algorithm == "cmaes"` (was unconditional, blocking GA configs that didn't set the irrelevant field).

The PR-review fixes carry a non-trivial caveat for the screening results: **every M5 screen with `predator.inheritance: lamarckian` and `count ≥ 2` trained on corrupted per-slot rollout data.** That's all of X4, Y1, Z1-Z5, P1, C4, C12, C14, C15, C16 except R1 (which predates the predator PPO substrate). The STOP conclusion holds — fixing the corruption wouldn't have made the substrate Red-Queen-capable — but the predator-side fitness numbers from those screens shouldn't be trusted as clean training signal. The lag-matrix evidence (cross-pairing prey elites vs frozen predator elites) is unaffected because it operates on the K-block-end elite weights themselves, not on the inflight rollout dynamics.

## Pilot trajectory

The canonical pilot (2 seeds × 30 gens at pop 24/16, K=10 alternating, heuristic-imitation pretrain) ran first under `configs/evolution/coevolution_pilot_arm_a.yml`. The R4 per-gen reaggregation gate fired (cycling-or-escalation detected on the per-generation lineage mean rather than the original K-block-elite series; this was the first observability fix). But the prey-side generality probe scored 0.000 across every fire, raising the obvious diagnostic question: is the prey learning real or is it overfitting to MLPPPO predators?

The pilot pivoted into a **screen sweep** to chase that question. Thirteen single-seed screens explored levers across four axes:

| Axis | Screens | Lever |
|---|---|---|
| Env knobs | Z1, Z2, Z3, Z4, Z5 | grid_size 16→18 / pred speed 0.5→0.7 / pred count {3, 4, 6} / K_per_block 10→5 |
| Density | P1, C12 | predator count {2 vs 4} × grid {16 vs 20} (P1 = count=2 grid=16; C12 = full M3 parity count=2 grid=20) |
| Schedule | C4 | K_per_block 10→5, generation_pairs 2→4 (faster alternation, same total gens) |
| Substrate | X4, Y1, C14, C15, C16 | base config + prey regulariser (Y1) / cold-start predator (C14) / seed-43 reproducibility (C15) / GA optimiser (C16) |

R1 was screened earliest under the pre-substrate-fix probe; re-audited at 25-eps under the modern probe (commit `cea7ef7f`-era) for direct comparison.

Each screen produced a post-hoc 25-eps fair-test against a calibrated probe env (count=2 speed=0.5 grid=20 — same as M3 baseline) using the K-block-end prey elite. The cell grid is (det, dam) where det ∈ {4, 6, 8, 10} and dam ∈ {0, 1}.

## Headline results

### Fair-test ranking ([summary/fair_test_summary.csv](../../../artifacts/logbooks/017-coevolution-arms-race/summary/fair_test_summary.csv))

| Rank | Candidate | Config delta vs X4 | Fair-test mean | dam=0 mean | dam=1 mean |
|--:|---|---|--:|--:|--:|
| 1 | **X4** | baseline (count=4 grid=16 speed=0.5 K=10) | **0.120** | **0.200** | 0.040 |
| 2 | C12 | grid 16→20, count 4→2 (M3 env parity) | 0.105 | 0.180 | 0.030 |
| 3 | P1 | count 4→2 | 0.080 | 0.150 | 0.010 |
| 4 | Y1 | prey `learn_episodes_per_eval` 8→4, `entropy_coef_end` 0.005→0.015 | 0.075 | 0.150 | 0.000 |
| 5 | C15 | C12 base, seed 42→43 | 0.025 | 0.030 | 0.020 |
| 6 | Z2 | pred speed 0.5→0.7 | 0.020 | 0.030 | 0.010 |
| 7 | Z3 | pred count 4→6 | 0.015 | 0.030 | 0.000 |
| 8 | C16 | optimiser cmaes→ga | 0.010 | 0.020 | 0.000 |
| 9 | Z1 | grid 16→18 | 0.005 | 0.010 | 0.000 |
| 10-14 | R1, Z4, Z5, C4, C14 | various | 0.000 | 0.000 | 0.000 |

### Cell-grid pattern: damage_radius cliff

Every screen exhibits the same shape: prey holds up at `dam=0` (contact-radius damage; takes 5 hits to kill) and falls to floor at `dam=1` (one-cell-radius reach; same hit count but doubled effective hit volume). The cliff isn't about instant-kill — `predator_damage=20` and `max_hp=100` are constant across all conditions — it's about ranged-vs-contact predator reach. Prey learns a touch-avoidance policy that generalises poorly to ranged predators.

**The dam=0 column is the realistic-physics ceiling**: X4 hits 0.200, C12 hits 0.180. M3 single-task baseline scored ~0.50 in roughly comparable conditions, so M5 sits at ~40% of M3 on the realistic-physics column. Not zero, not full success.

### Lag matrix: no Red Queen entanglement ([summary/lag_matrix.csv](../../../artifacts/logbooks/017-coevolution-arms-race/summary/lag_matrix.csv))

The c2 fitness-lag analysis cross-pairs each candidate's gen-19 prey elite against each candidate's gen-19 predator elite:

```text
prey \ pred         X4      P1      Y1      Z2      Z3     C12     C15     C16
X4               1.000   0.933   0.800   0.933   0.933   0.733   0.933   0.933
P1               1.000   0.667   0.933   0.733   0.867   0.933   0.467   0.800
Y1               0.200   0.467   0.533   0.467   0.667   0.533   0.333   0.267
Z2               0.933   0.467   0.133   0.467   0.333   0.133   0.733   0.933
Z3               0.000   0.067   0.000   0.133   0.067   0.000   0.000   0.000
C12              0.867   0.933   1.000   0.867   0.933   1.000   1.000   0.733
C15              0.867   1.000   1.000   1.000   1.000   1.000   0.933   0.933
C16              0.867   0.933   0.867   1.000   1.000   0.800   0.933   0.933

own-pair mean: 0.700   cross-pair mean: 0.683   delta: +0.017
```

Diagonal = each prey vs the predator it co-evolved with. Off-diagonal = each prey vs predators from other screens it never trained against.

**Delta interpretation**: a real Red Queen arrow would show **negative** delta (prey performs *worse* against its own predator, because that predator learned specifically to exploit *this* prey lineage). The observed delta of **+0.017** means pairings don't matter — the grid is purely explained by "X4 is a stronger prey" and "X4 is a weaker predator", not by who-trained-against-whom. **Strict Red Queen entanglement is absent.**

This is the same delta sign and magnitude across the entire sweep including the GA fallback (C16). The failure isn't an optimiser problem.

### Reproducibility: C12 → C15

C12 (the most dynamically interesting candidate — smooth prey climb 0.01 → 0.90 across K-block-1, predator non-monotone trough+rebound, dual positive escalation) **did not reproduce at seed 43**:

| Signal | C12 (seed 42) | C15 (seed 43) |
|---|--:|--:|
| Fair-test mean | 0.105 | 0.025 |
| Predator escalation slope | +0.049 (p\<0.0001) | −0.021 (p=0.049) |
| Predator KB2 trough → KB3 recovery | yes | NO |
| R4 gate fires | yes | yes (different way) |
| Own-cross lag delta | +0.024 | +0.017 |

C12's "first cleanly co-escalating M5 result" framing was a seed-42-specific quirk; the predator non-monotone trajectory (most exciting feature) did not reproduce. The reproducibility failure is the second decisive negative finding: even the strongest single-seed result is fragile.

### Verdict (per design D6 + lag-matrix instrument)

**STOP M5**. Two decisive negatives:

1. **Lag-matrix delta = +0.017 across 8 candidates** (all screened candidates with full posthoc + champion archive). No candidate achieved the strict Red Queen criterion (delta ≤ −0.05). The GA optimiser fallback that design D2 reserved for "if CMA-ES diversity collapses" also produced +0.017 — same direction, same magnitude.
2. **Best single-seed result did not reproduce** (C12 0.105 → C15 0.025 at seed 43; predator non-monotone trajectory disappeared).

Going to a 4-seed full pilot on the best-screened config (X4) would have produced an R4-gate fire (cycling-or-escalation by the soft criterion) but the lag-matrix would have stayed positive across all 4 seeds, and the fair-test reproducibility risk would have remained. The screening pivot generated the same evidence at 1/4 the compute cost of the canonical campaign.

## Why no Red Queen — four candidate root causes

1. **Predator-side capacity gap (dominant hypothesis)**. Brain asymmetry:

   - Prey: LSTMPPO + klinotaxis (GRU recurrence, ~30k weights, bilateral chemo/nociception sensing, STAM short-term memory)
   - Predator: MLPPPO (stateless MLP, ~10k weights, position-only 11-D input, k_nearest=2)

   Prey can remember predator behaviour patterns across timesteps; predator sees only a position snapshot. The lag-matrix consistently shows each candidate's prey beating *every other candidate's predator* in row mean — that's the fingerprint of one side having higher policy capacity, not paired specialisation. The M5 design picked MLPPPO predator partly for CMA-ES weight-count tractability; that trade-off was never revisited under co-evolution stress.

2. **Fitness-function asymmetry**. Prey = `LearnedPerformanceFitness` (multi-component: foraging + survival + health + exploration). Predator = `PredatorEpisodicKillRate` (kill count + secondary proximity fallback). Prey optimises a generalist objective; predator optimises a specialist objective where all prey strategies that reduce kill rate are equally penalised. No incentive for predator to develop prey-specific counters. Literature Red Queen models use richer predator fitness (e.g. kill latency, prey-specific bonuses) — we never implemented those.

3. **Episode-coupling weakness**. Within each K-block, the trained side faces a stationary opposition distribution (frozen K-block elite + HoF samples). Prey trains against a fixed-but-structured opponent for 10 generations — PPO converges fast on a stationary distribution. Paired specialisation requires both sides to move together. Alternating training breaks this rally; design D1 explicitly chose alternating ("TPE handles non-stationarity poorly") accepting this trade-off.

4. **Brain-vs-population evolution**. CMA-ES outer loop + PPO inner loop on the prey side. PPO dominates fitness improvement within a K-block — the "evolution" we observe is mostly PPO learning under stationary opposition, not actual genetic drift in coupled phenotypes. We never tested "CMA-ES only, no PPO" to isolate the genuine evolutionary signal.

**My read**: cause #1 is dominant. The substrate is biologically lopsided; we tried to compensate with optimiser tweaks (which don't change capacity asymmetry) but never with architecture symmetry. The single substrate change most likely to flip the lag-delta sign is **LSTMPPO predator at matched hidden_dim with k_nearest=5+ and bilateral sensing** — about a week of implementation effort, out of M5 scope. This naturally motivates M6.5 NEAT (next section).

## Methodology contributions

Even with STOP M5, the screening produced reusable methodology for future co-evolution attempts:

1. **Cross-species fair-test cell grid** — post-hoc 25-eps evaluation of K-block-end elite weights against a calibrated probe env (count=2 speed=0.5 grid=20 matching M3 baseline) across an 8-cell (det, dam) grid. The dam=0 vs dam=1 split exposed the touch-vs-ranged-avoidance generalisation gap; the mean alone hides this. Per-screen cell-grid outputs live under [`pilot/<screen>/posthoc_25eps/matrix.csv`](../../../artifacts/logbooks/017-coevolution-arms-race/pilot/x4/posthoc_25eps/matrix.csv) (X4 example link); the aggregated table is at [`summary/fair_test_summary.csv`](../../../artifacts/logbooks/017-coevolution-arms-race/summary/fair_test_summary.csv). The forensic helper scripts (`c2_cellgrid_audit.py`, `c2_fitness_lag.py`) lived in the gitignored screening-pivot output tree and are recoverable from commits `cea7ef7f` through `8436082c` on the merged PR #153 branch if a future analysis needs to reproduce them.
2. **Own-vs-cross lag matrix** — cross-pair every candidate's prey-elite vs every candidate's predator-elite. Asymmetric quality vs paired entanglement appear in clearly different signatures: pure quality dominance shows as constant rows/columns; real Red Queen shows as negative diagonal-vs-off-diagonal delta. The default R4 gate (cycling-or-escalation per design D6) fires on prey-saturation patterns indistinguishable from Red Queen at the per-gen-lineage level; the lag matrix discriminates. The 8×8 final matrix is at [`summary/lag_matrix.csv`](../../../artifacts/logbooks/017-coevolution-arms-race/summary/lag_matrix.csv); the captured stdout from the c2 cross-pairing run is at [`c2/lag_matrix_final.log`](../../../artifacts/logbooks/017-coevolution-arms-race/c2/lag_matrix_final.log).
3. **Per-gen re-aggregation vs K-block-elite series** ([scripts/campaigns/screen_r4_per_gen_reaggregate.py](../../../scripts/campaigns/screen_r4_per_gen_reaggregate.py)) — the production aggregator's K-block-elite series is too short for cycling/escalation tests when `generation_pairs=2` (only 4 K-blocks → 4 points/side). Per-gen mean-fitness reaggregation gives ~10× more samples and exposes signal the K-block-elite series masks when prey saturates at the population's success-rate ceiling. **Both gates fire on Red Queen and non-Red-Queen alike; the lag matrix is the discriminative instrument, not the R4 gate.**

These instruments are the M5 contribution that survives the STOP verdict.

## Compute envelope

- Substrate dev (PRs 1–5): zero pilot compute; all unit tests + smoke runs.
- Pilot screening (PR 6): 13 single-seed screens × ~2–3 wall-hours each at parallel_workers=4 = **~30 wall-hours total** (vs the canonical pilot's ~7–14 wall-hours; the screens explored more lever space at the cost of single-seed noise). C4 + C14 killed at gen ≤12 (wipeout pattern; recovery impossible). R1 re-audited under modern probe semantics at zero compute (re-ran post-hoc analysis against existing weights).
- Post-hoc analysis: ~30 min per candidate × 14 (incl. R1 re-audit and c2 cross-pairing analyses).
- Full-run campaign (tasks 10.x in the OpenSpec change): **NOT executed**. STOP recommendation derived from screening before committing the canonical 30–60 wall-hour 4-seed full run. This decision saves ~30 wall-hours; the screening evidence is decisive.

Reconciliation table (per-seed wall):

| Screen | parallel_workers | mean eval (prey) | mean eval (predator) | total wall |
|---|--:|--:|--:|--:|
| X4 | 4 | 2.22s | 41.7s | 4.0h |
| C12 | 4 | 2.20s | 39.8s | 2.2h |
| C16 (GA) | 4 | 2.18s | 40.5s | 2.2h |

The GA optimiser ran at materially the same per-eval cost as CMA-ES (the optimiser overhead is dominated by PPO inner-loop training cost on the prey side, not the outer-loop sampling). Wall-time was not the bottleneck in this pilot.

## M5.7 secondary Baldwin readout — formally dropped

Design D11 / task 11 originally scoped a secondary Baldwin instrumentation: per-gen elite-vs-schema-prior signal-delta against the current predator population, evaluated at K′ ∈ {10, 25} PPO inner-loop training episodes per saved prey elite at generations G ∈ {5, 10, 15, 20, 25, 30}. Readout was scoped to run "regardless of M5 verdict" since the signal is informative even on a STOP.

**Dropped**: the readout requires "per-gen elite snapshots from the full-run lineage CSV" plus "per-gen predator pop from champion_history.json contemporaneous with gen G". The full 4-seed × 30+ gen run that produces those snapshots was not executed (per the screening pivot). The 13 single-seed screens produced lineage CSVs but they're single-seed, not the 4-seed protocol the readout expects. Re-scoping to screening data would lose the statistical power the readout was designed against and muddle the protocol.

**Implication**: M4.7 (deferred Baldwin retry under hyperparameter evolution) stays armed but unmotivated by M5 evidence. The dropped hyperparam-spread condition (already dropped pre-pilot as not-observable under weight evolution) confirms that definitive Baldwin closure requires a dedicated hyperparameter-evolution milestone.

## What's next: M6.5 NEAT as natural Red Queen follow-up

The capacity-gap hypothesis (root cause #1 above) points directly at M6.5 (NEAT architecture evolution, currently optional in the Phase 5 plan):

- **NEAT evolves both topology and weights** — the predator brain could *grow* its own complexity over generations rather than being locked at our current MLPPPO architecture.
- This is the published-literature regime where Red Queen reliably emerges (Stanley & Miikkulainen 2002 NEAT-vs-NEAT predator-prey; Sakana DRQ 2025 uses architecture-evolving agents).
- Our M5 setup is "fixed-architecture, parameter-only co-evolution" — a constrained version of the literature regime.
- **TensorNEAT 2024–2025** materially lowers M6.5's compute cost (GPU-accelerated topology + weight evolution).
- Honest probability estimate of Red Queen emerging under M6.5 with architecture symmetry: **~40–50%**. Much higher than M5's near-zero outcome under fixed asymmetric architectures.

M6.5 stays optional in the plan; the recommendation if it's scheduled is to use M5's lag-matrix instrument as the verdict primitive (it's discriminative where the R4 gate isn't).

## Verdict line

**M5: STOP.** No Red Queen entanglement at this substrate under any of 13 screened lever configurations including the GA-fallback design reserve. Best single-seed fair-test (X4 at 0.120, ~40% of M3 baseline on the realistic-physics column) did not reproduce at seed 43 (C12 → C15). Methodology contributions (lag-matrix + cell-grid + per-gen re-aggregation) ship intact and discriminate Red Queen from asymmetric-quality patterns. M3 (Lamarckian inheritance pilot, +47pp / +79pp) remains the strongest concrete Phase 5 result. M6.5 NEAT is the natural follow-up for the architecture-symmetry hypothesis if Red Queen remains a Phase 5 goal.
