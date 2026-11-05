# Parked: C.1 refitted on benchmark v4 (draft design, not registered)

*Parked 2026-11-05 by the maintainer's decision ([Logbook 125](../../125-c1-v4-diagnosis.md)): the refit is deferred until C.4 approaches, and is then run once on the benchmark current at that time. Nothing below is registered. It is the draft OpenSpec change as it stood when the box-centre probes failed, kept so the refit starts from it.*

**Open when the refit resumes:**

- **The start.** Three box-centre runs failed v4 (D 1.78–1.86 with 25 parameters; 1.85–1.92 with 26, its states swapped; 2.87–3.02 with 26 and the states ordered), while starts continued from C.1 (v3)'s answer reach the passing basin (4 of 5). A warm start conflicts with D38 as amended ("its search box and x₀ are set without search data"), since C.1 (v3)'s answer was fitted on every v3 family, local search and chemotaxis included; it needs a dated D38 amendment, or a box-centre design whose odds are measured first.
- **The states' order.** Ordering them (the dwelling speed a fraction of the roaming speed) removed the swapped basin but not the box-centre failure.
- **The [BOX] items** below (the cap, the recovery's second truth) follow from whichever start is chosen.

______________________________________________________________________

## Why

Benchmark v4 ([Logbook 124](../../../docs/experiments/logbooks/124-benchmark-v4.md)) fails C.1's v3 answer in all 5 of its registered populations. The answer reverses too rarely on the lawn: 0.78–0.85 onsets per on-lawn minute against real worms' 2.28 (z −6.4 to −6.7). C.1 must be refitted on v4 before the fidelity table.

Discarded-seed probes (2026-11-05, description; `supporting/125-c1-v4-fit/probes/`) show that C.1's 25-parameter model cannot reach v4, and why:

- **It never reverses while dwelling on food.** Real worms make most on-lawn onsets while dwelling (2.12 per minute in dwelling windows, which are 83% of on-lawn time), from intermittent movement: over half their dwelling seconds are paused, yet 31% of their 1-s steps clear the detector's 0.03 mm floor. C.1 dwells at a near-constant 0.013 mm/s with its dwelling reversal rate at its floor (0.001/s), so its dwelling onsets are 0.78 per minute.
- **Raising `r_dwell` breaks the off-food assays.** The same rate sets reversals off food: at 0.03 it lifts on-lawn onsets to z −2.2, but flattens López-Cruz's local search (log early/late z −3.1) and over-flips Chen's chemotaxis (+2.1). The conflict is on food against off food.

Four candidates were probed, each one warm-started restart graded on 5 populations:

| Candidate | Parameters | Passing | D | What fails |
|---|---|---|---|---|
| none (readiness, warm / box centre) | 25 | 0 / 0 | 1.39–1.46 / 1.78–1.86 | the on-lawn onset rate (z −3.3 to −5.8) |
| reversal–speed coupling | 26 | 0 | 1.56–1.72 | the onset rate (z −3.7 to −4.1) |
| intermittent dwelling | 27 | 0 | 1.30–1.50 | the onset rate (z −4.0 to −4.2) |
| a state-free off-food reversal rate | 26 | 0 | 3.58–3.80 | local search's decay (γ undefined) |
| **an on-food dwelling reversal rate** | **26** | **4** | **1.28–1.44** | one small-lawn roaming-bout length |
| the same, from the box centre | 26 | 0 | 1.85–1.92 | its states swapped: the fast state is "dwelling", so the on-food rate acts on the wrong one |
| the same, states ordered, box centre | 26 | 0 | 2.87–3.02 | the kinematics (dwelling speed p90 z +10, median +7; roaming speed −5), its roaming state at three bounds |

The maintainer's decisions (2026-11-05):

- refit C.1 on v4 with the 26-parameter model, the 25 plus `r_dwell_on_food`;
- drop the coupling and intermittent dwelling, committed with the probes so they reproduce;
- probe the 26-parameter model from the box centre before registering;
- tighten the plateau's stop;
- after the box-centre probe fell into the swapped basin, order the states: the dwelling speed is the roaming speed times a fraction, so state 0 is always the slower.

## What Changes

- **`BatchedMechanistic` gains `r_dwell_on_food`:** a dwelling animal's reversal rate wherever food contact is above 0, in place of `r_dwell`, before the klinokinesis, edge and search terms. Off food, and roaming, the state's rate applies. Off by default: without it the model is byte-identical (59 of 59 identity jobs). The reversal–speed coupling and intermittent dwelling, probed at commit `886e7bfc`, are removed.
- **C.1 (v4), the 26-parameter model:** C.1 (v3)'s 25 parameters on their boxes, plus `r_dwell_on_food` on `r_dwell`'s prior (log, 0.001–0.5), with one change: the coordinate that held `v_dwell` holds the dwelling speed's fraction of the roaming speed, log-uniform on [0.025, 1], so state 0 is always the slower state, as the classifier defines dwelling.
- **The mutant encodings:** C.1 (v3)'s. `r_dwell_on_food` is a locomotor rate, not chemosensory, and every mutant keeps it.
- **A reduced recovery** (F.0b-i's rules, 2 truths × 2 restarts) on the 26-parameter model, before any registered fit. [BOX: the truths.]
- **The registered fit:** `scripts/analysis/c1_v4_fit.py`, C.1 (v3)'s design on v4:
  - 12 box-centre restarts;
  - the ranking score against v4's bands;
  - the doubly guarded plateau, tightened: it stops a restart only while every median in its last 40 generations is below 100. [BOX: the cap.]
  - selection on 5 separate populations per plateaued restart;
  - a verdict at 3 of 5 registered populations with presence, at v4's |z| ≤ 3.0781.
- **The probes are the pilot.** They are disclosed, they set the cap, and they show reachability.
- **Launch record 125** (Part A before the recovery; Part B after it, before the fits) and **Logbook 125**.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `mechanistic-baseline`: an optional on-food dwelling reversal rate.
- `controller-fit`: C.1 refitted on benchmark v4 with its 26-parameter model, a reduced recovery first, the probes as the pilot, and a tightened stop.

## Impact

- **Code:** `quantumnematode/evaluation/controllers.py` (`BatchedMechanistic`); `scripts/analysis/c1_v4_fit.py` (new); tests.
- **Records:** `docs/experiments/logbooks/supporting/125-c1-v4-fit/` (the probes as disclosure, the launch record, the recovery, the fit); Logbook 125; the tracker (C.1, C.4, the seed table); AGENTS.md; the CHANGELOG.
- **Cost** at about 30 s per generation on 16 workers: the recovery about 3.5–7 h; the fits about 10–20 h typical [BOX: at most]; grading under 10 minutes.

______________________________________________________________________

## Context

- **Benchmark v4** ([Logbook 124](../../../docs/experiments/logbooks/124-benchmark-v4.md)) is VALID and the default: 24 statistics in six equal families at z\* 3.0781, the new activity family (pause share, flip fraction, on-lawn onset rate) in D and D_fit. A faithful controller passes 3 of 5 populations with probability 0.975.

- **C.1 (v3)** ([Logbook 121](../../../docs/experiments/logbooks/121-c1-v3-fit.md)) passed v3 at 3 of 5, and fails v4 at 0 of 5 on the on-lawn onset rate (z −6.4 to −6.7), on the same animals.

- **The probes** (2026-11-05, description, discarded seeds 620–642 M, CMA 9441–9447; records and scripts in `supporting/125-c1-v4-fit/probes/`). Each is one sep-CMA restart against v4's ranking score with C.1b's doubly guarded stop and a cap of 300, its answer graded on v4 with presence on 5 populations:

  | Probe | Model | Start | Stop | Passing | D | What fails |
  |---|---|---|---|---|---|---|
  | readiness, warm | 25 | C.1 (v3)'s answer, σ 0.1 | 100 | 0 | 1.39–1.46 | the onset rate (z −3.3 to −3.9) |
  | readiness, box | 25 | centre | 200 | 0 | 1.78–1.86 | the onset rate (z −5.6 to −5.8) |
  | coupling | 26 | readiness warm, σ 0.1 | 124 | 0 | 1.56–1.72 | the onset rate (z −3.7 to −4.1) |
  | intermittent dwelling | 27 | readiness warm, σ 0.1 | 126 | 0 | 1.30–1.50 | the onset rate (z −4.0 to −4.2) |
  | off-food rate | 26 | readiness warm, r_dwell 0.03, σ 0.1 | cap 300 | 0 | 3.58–3.80 | local search's γ undefined in all 5 |
  | **on-food dwelling rate** | **26** | **readiness warm, σ 0.1** | **101** | **4** | **1.28–1.44** | **a small-lawn roaming-bout length, in 1** |
  | on-food dwelling rate, box | 26 | centre, tightened stop | 172 | 0 | 1.85–1.92 | the onset rate (z −4.1 to −4.6), dwelling speed p90 (+3.4 to +3.7), flips (+2.9 to +3.3): its fast state is state 0 |
  | ordered states, box | 26, v_dwell as a fraction of v_roam | centre, tightened stop | 115 | 0 | 2.87–3.02 | dwelling speed p90 (+10), median (+7), roaming speed (−5); the roaming spread, omega and reversal rates at their bounds |

  **The diagnosis** (`state_diagnosis.json`): on Scheer's band cohorts, 83% of classified on-lawn seconds are dwelling, with 2.12 onsets per minute; the readiness answer's dwelling has 0.78. Real dwelling 1-s steps are 55% below 0.02 mm and 31% at least 0.03 mm; the answer's 63% and 20%.

  **Two sensitivity checks** (one population each on shared discarded animals):

  - on the intermittent-dwelling answer (`dwell_sensitivity.json`), raising `r_dwell` lifts on-lawn onsets (z −4.0 → −2.2 at 0.03 → +0.1 at 0.1) but flattens local search (log early/late −3.1 at 0.03) and over-flips Chen's chemotaxis (+2.1); faster dwelling movement breaks the dwelling window speeds (+7);
  - on the readiness warm answer (`ondwell_sensitivity.json`), `r_dwell_on_food` at 0.03 lifts on-lawn onsets to z −2.7 and moves no off-food statistic; the small lawn's roaming fraction and roaming bouts then fail, which the fit retunes.

- **Cost:** about 30 s per generation on 16 workers.

## Goals / Non-Goals

**Goals:**

- C.1's fit to v4's wild type, with the one added mechanism answering the named failure;
- a recovery check on the 26-parameter model;
- rules registered before any registered reading, and selection kept apart from the verdict.

**Non-Goals:**

- mutant grading, which is C.4's;
- changing the benchmark;
- further mechanisms, including the two probed and dropped;
- grading the reversal share, which stays reported.

## Decisions

1. **The model's option** (`BatchedMechanistic`):

   - **`r_dwell_on_food`:** a dwelling animal's reversal rate wherever food contact is above 0, in place of `r_dwell`, before the klinokinesis, edge and search terms. Off food, and while roaming, the state's rate applies.
   - **Removed:** the reversal–speed coupling and intermittent dwelling, probed at commit `886e7bfc`, where their probes reproduce.
   - **Identity:** `speedup_identity_check.py check --build build/c1-v4/identity` against the baseline recorded on the code before any of the three options; all 59 jobs must be byte-identical. Tests cover the option's scope.

2. **C.1 (v4): 26 parameters.** C.1 (v3)'s 25 on their boxes (`c1_v3_fit.parameters`), plus:

   | Parameter | Range | From |
   |---|---|---|
   | `r_dwell_on_food` | log 0.001–0.5 per s | `r_dwell`'s prior |
   | the dwelling speed, as `v_dwell` / `v_roam` | log 0.025–1, on the coordinate that held `v_dwell` | the speeds' prior: 0.005 / 0.2 to 1 |

   **The states are ordered.** The box-centre probe's answer swapped them (state 0 fast), and `r_dwell_on_food` acts only on state 0, so there it acts on the fast state and the slow one cannot reverse on food. With the dwelling speed a fraction of the roaming speed, state 0 is always the slower, as the classifier defines dwelling. The warm answer maps inside the box (fraction 0.177, unit 0.53).

   It is not set from any reference's data. The mechanism was chosen by probes on all six families. That is disclosed, and it is why C.1's mutant predictions, not its wild-type fit, are its held-out test.

3. **The mutant encodings:** C.1 (v3)'s, registered in launch record 121 on `113-c1-fit/encodings.md`, both bound byte-identical:

   - tax-4 sets κ_w = κ_off = β_r = β_o = 0;
   - tph-1 sets a_intake = b_intake = 0;
   - pdfr-1 is not encodable;
   - every mutant keeps `r_dwell_on_food`, a locomotor rate set by food contact, which no mutant's encoding reads.

4. **The reduced recovery** (F.0b-i's rules on the 26-parameter model, on v4's D), as Logbook 121's:

   - **The truths:** [BOX: the on-food dwelling probe's warm answer (t = 0) and the box answer (t = 1).]
   - **Each truth's record:** the observation and 20 re-simulations; the floor, the median capped D from the observation to re-simulations 1–10; the oracle, the same over 11–20; the prior D, the median capped D to 200 prior draws of the 26-parameter box (`default_rng(9467)`).
   - **Gain** = (prior D − D) / (prior D − floor), every D the registered capped D on v4's scales and family map.
   - **Before the fits:** the oracle's median gain at least 0.85, the box centre's (10 re-simulations) at most 0.25, and each truth's prior D above its floor by at least 1.0. Otherwise the recovery stops and returns to the maintainer.
   - **The fits:** 2 restarts per truth (fit = 2t + j), sep-CMA from the box centre, σ₀ 0.2, population 16, CMA seeds 9463 + fit, ranked by the definedness-first score against the truth's observation, the tightened stop and the cap of decision 5.
   - **The gate:** per truth the better restart by final objective is re-simulated 10 times and its gain read. The median gain over the two truths must be at least **0.75**; otherwise no registered fit runs until the maintainer decides.
   - C.1's states can swap roles, so each parameter error is read in whichever state order is closer; the gain is unaffected.

5. **The registered fit** (`c1_v4_fit.py`): C.1 (v3)'s design on v4.

   - **The objective:** the ranking score against v4's band centres at n = 200, over v4's assays through its measurement models, without presence.
   - **sep-CMA:** 12 restarts r = 0–11 from the box centre, σ₀ 0.2, population 16, bounds [0, 1], CMA seeds 9451 + r; the answer is `xfavorite`.
   - **The stop, tightened:** the plateau rule once σ ≤ σ₀, no median of 1,000 or more in the last 40 generations, **and every median in the last 40 generations below 100**. Otherwise the cap. [BOX: the cap, ⌈1.5 × the 26-parameter probes' longest plateau⌉, rounded up to a multiple of 10.]
   - **Selection:** every plateaued restart graded with presence on 5 selection populations; the restart passing most is chosen, ties going to the lower median D and then the lower r; a capped restart cannot be chosen; if none plateaus, no verdict is read.
   - **The verdict:** every restart graded with presence on 5 registered populations; a population passes when every family passes at |z| ≤ 3.0781 with local search and chemotaxis present; **the chosen restart passes the wild type when at least 3 of its 5 pass.**
   - **The verdict map:** pass, C.1 (v4) enters C.4; fail, the failing statistics are the diagnosis, and the roadmap's C.1 risk row goes to the maintainer.
   - **Reported:** every restart's stop, final score, selection and registered grades, and median D over 10 re-simulations; the restart spread; the chosen parameters and any held at a bound; each restart's reversal share against Roberts' 17.3%.

6. **Seeds.** Each restart's block spans 2,000,000 animals (k = 10,000·r + …).

   | Base | Use |
   |---|---|
   | 650,000,000 + 200·k + i | the fit: k = 10,000·r + 16·(n − 1) + m; answers at k = 10,000·r + 8,000 + q (q < 10), above every generation's index under the cap |
   | 675,000,000 + 200·g + i | the registered populations, g < 5 (the fit's block ends below 674 M) |
   | 676,000,000 + 200·(5r + s) + i | the selection populations, s < 5 |
   | 680,000,000 + 200·k + i | the recovery's fits, k = 10,000·fit + …, answers at 10,000·fit + 8,000 + q |
   | 690,000,000 + 200·(100·t + q) + i | the truths' observation (q = 0) and re-simulations (1–20) |
   | 691,000,000 + 200·j + i | the prior draws, j < 200 |
   | 692,000,000 + 200·q + i | the box centre's re-simulations, q < 10 |
   | 99,640,000 and 99,650,000; CMA 9468 | the smoke, discarded |
   | 620,000,000–642,001,000; 627,000,000; 632,000,000; 99,620,000; 99,630,000; CMA 9441–9447 | the probes, the diagnosis and the sensitivity checks, discarded |

7. **The registration:** launch record 125.

   - **Part A** fixes every rule above, discloses the probes, and is committed before the recovery.
   - **Part B**, after the recovery and before any fit, records its result and changes no rule.
   - **The guards:** `recovery-preflight` and `recovery` refuse unless Part A is committed and unchanged; `run`, `select`, `grade` and `read` need Part B with the recovery record bound and passing; every stage binds the encoding tables byte-identical.

## Risks / Trade-offs

- **[The mechanism was chosen on the benchmark]** → Disclosed, with every candidate probed and its record committed. It answers a measured failure: real worms reverse while dwelling on the lawn and C.1 cannot. C.1's mutant predictions remain its held-out test.
- **[Thin margins]** → The warm probe's passing populations hold the onset rate at z −2.6 to −3.0 and the small lawn's roaming bouts at +2.2 to +3.2, against a bar of 3.0781. The verdict's chance is below the 4 of 5 the warm probe saw, and both 26-parameter box-centre probes passed 0 of 5.
- **[Under-reversal off food]** → The on-lawn onset rate is now graded and met; the reported reversal share over the off-food windows stays near 5%, against Roberts' 17.3%. It is reported beside the verdict.
- **[Parameters at their bounds]** → The warm answer holds `b0` at its upper bound (log 0.1) and `r_dwell` near its floor. The box is not widened; a bound is reported where it holds.
- **[Cost]** → about 15–30 h in all, the fits resumable and the recovery gating them.

______________________________________________________________________

## 1. The model's option

- [ ] 1.1 `BatchedMechanistic`: keep `r_dwell_on_food`, off by default, and remove the reversal–speed coupling and intermittent dwelling (probed at `886e7bfc`) with their tests. Tests for each scenario.
- [ ] 1.2 `speedup_identity_check.py check --build build/c1-v4/identity --out supporting/125-c1-v4-fit/identity.json`: all 59 jobs byte-identical.

## 2. The fit script

- [ ] 2.1 `scripts/analysis/c1_v4_fit.py`:

  - the 26-parameter box;
  - the v4 band objective and simulation;
  - the tightened stop;
  - the seed scheme;
  - the encodings, bound;
  - the stages `recovery-preflight`, `recovery`, `run`, `select`, `grade`, `read` and `smoke`, guarded as registered.

  Tests:

  - disjoint seed blocks, the answers' offset above every generation's index under the cap;
  - the tightened stop: a restart whose last 40 medians include one of 100 or more does not stop;
  - the guards;
  - the recovery gate's arithmetic, on the capped v4 D;
  - selection and ties.

- [ ] 2.2 The smoke (CMA 9468, animals 99,640,000, grading 99,650,000, discarded): two generations of each stage.

## 3. Registration and the recovery

- [ ] 3.1 The probes' records and scripts committed to `supporting/125-c1-v4-fit/probes/` (all but the box probe's at `886e7bfc`).
- [ ] 3.2 Launch record 125, Part A (rules, seeds, the encodings bound, the probes disclosed), committed.
- [ ] 3.3 `recovery-preflight`: the observations, floors, oracle, prior D and the box centre's gain. Stop if a check fails.
- [ ] 3.4 `recovery`: 4 fits, the gains, the gate. Return to the maintainer if the median gain is below 0.75.
- [ ] 3.5 Part B: the recovery's result, committed.

## 4. The fit and the verdict

- [ ] 4.1 `run` (12 restarts), `select` (5 populations per plateaued restart), `grade` (every restart on 5 registered populations), `read`: `fit.json` with the verdict.

## 5. Close-out

- [ ] 5.1 Logbook 125; the tracker (C.1 on v4, C.4, the seed table); AGENTS.md and the CHANGELOG.
- [ ] 5.2 Full suite; `pre-commit run --all-files` after `git add -A`; `openspec validate --strict`; archive; PR.

______________________________________________________________________

## Draft spec delta: controller-fit

### (ADDED Requirements)

#### Requirement: C.1 is refitted on benchmark v4 with its 26-parameter model

C.1 (v4) SHALL be C.1 (v3)'s 25 parameters on their boxes plus `r_dwell_on_food` on `r_dwell`'s prior, set from no reference's data. It SHALL be fitted to the real wild type on benchmark v4 by sep-CMA from the box centre, with 12 restarts, the definedness-first ranking score against v4's bands at n = 200, the doubly guarded plateau tightened so that a restart stops only while every median in its last 40 generations is below 100, and a cap set from the probes. Selection on 5 separate populations per plateaued restart SHALL choose the restart, and the verdict SHALL be at least 3 of 5 registered populations passing with presence at |z| ≤ 3.0781.

##### Scenario: A restart above the tightened level does not stop

- **WHEN** a restart's medians are flat and its σ is at most σ₀, but one of its last 40 medians is 100 or more
- **THEN** the plateau SHALL NOT stop it

##### Scenario: A capped restart is not chosen

- **WHEN** a restart reaches the cap without the plateau
- **THEN** it SHALL NOT be chosen, though its registered grades SHALL be reported

##### Scenario: Seed blocks do not overlap

- **WHEN** the fit's, the answers', the selection's, the registered populations' and the recovery's animal seeds are enumerated
- **THEN** no animal seed SHALL appear in two of them

#### Requirement: A reduced recovery gates the v4 fit

Before any registered v4 fit, sep-CMA SHALL recover C.1 (v4) at two known truths from the 26-parameter probes. Each truth SHALL be fitted twice from the box centre on its observation, and the better restart by final objective SHALL be scored by its gain fraction, (prior D − D) / (prior D − floor). The truths' oracle SHALL reach a median gain of at least 0.85 and the box centre at most 0.25 before any recovery fit runs. The fit SHALL run only if the median gain over the two truths is at least 0.75.

##### Scenario: The recovery falls short

- **WHEN** the two truths' median gain is 0.70
- **THEN** no registered fit SHALL run until the maintainer decides

#### Requirement: The probes are the v4 fit's pilot

The discarded-seed probes recorded in launch record 125 SHALL set the v4 fit's cap and stand as its reachability evidence. Every probe SHALL be disclosed in Part A with its seeds and outcome, the dropped candidates included.

##### Scenario: The cap comes from the probes

- **WHEN** the cap is registered
- **THEN** it SHALL be at least 1.5 × the longest plateau generation among the 26-parameter probes of the chosen model

#### Requirement: The 26-parameter model's mutant encodings

C.1 (v4)'s mutant encodings SHALL be C.1 (v3)'s registered encodings, and every mutant SHALL keep `r_dwell_on_food`. The tables SHALL stay byte-identical, bound by every stage.

##### Scenario: tax-4 keeps the on-food dwelling rate

- **WHEN** C.1 (v4)'s tax-4 encoding is built
- **THEN** both weathervane gains and both klinokinesis gains SHALL be 0, and `r_dwell_on_food` SHALL be unchanged
