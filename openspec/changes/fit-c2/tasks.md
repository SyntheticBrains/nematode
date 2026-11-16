## 1. The grading arm

- [x] 1.1 `generate_assay_configs.py`:

  - the arm `encoded_following`: the encoded arm's configs with `continuous.on_food_reversal_speed: follows_run_speed`;
  - its six configs generated and committed;
  - `benchmark_grade.ARMS["connectome_c2_following"]`.

  Tests:

  - the arm differs from the encoded arm only in the option;
  - the committed-configs test and the assay smoke cover it through the generator's `ARMS`.

## 2. The fit script

- [x] 2.1 `scripts/analysis/c2_fit.py`, the objective:

  - the ranking score against a truth (the bands, or an observation);
  - the mutant terms against the truth's differences, on the encoding table's graded statistics (tph-1's leaving recorded, not scored), each weighing n_f / n_m;
  - both guards through the forward observer, on the wild type's simulation only;
  - the failure probability with the noise step's SDs and the bands' bars.

  Tests:

  - the half-weight and at-the-bar scenarios;
  - tph-1's leaving not scored;
  - an undefined statistic;
  - a leave-one-out candidate's score not changed by tax-4.

- [x] 2.2 The stages:

  - stage 1's subset;
  - the 1-s timescale at 2a;
  - the search state carried from 2a to 2b and from 2b to the final stage (σ and `CMA_stds` = stds/σ) *(from 2026-11-16, task 4.1n: 2b starts a new isotropic search at 2a's answer, σ₀ 0.05, and only 2b's state is carried)*;
  - fixed budgets, with the final stage's plateau reported;
  - readouts every 10 generations recording every statistic;
  - atomic state every generation, and a resume that refuses changed settings.

  Tests: the hand-off, the subset, and the plateau not stopping a stage.

- [x] 2.3 The noise step: 30 populations without presence, SDs floored at 0.05. Test: the floor, and a statistic defined in fewer than 2 populations.

- [x] 2.4 The grade and the verdict:

  - 5 registered populations with presence, and each mutant on its population's own animals;
  - 10 more for the pass rate;
  - the verdict on the wild type, tph-1 and tax-4 on the table's graded statistics, with pdfr-1 reported;
  - the restart spread against the pilot.

  Tests: the verdict's scenarios.

- [x] 2.5 The held-out reading on W's rule, for N2's unread and PD1074's held-out dates:

  - the 15 graded populations;
  - the 15 band-cohort populations (`default_rng(2048)`);
  - the real halves.

  Tests:

  - (slow) the real halves reproduce `w_minimum_v4`'s fitted 95th percentile;
  - the band-cohort draws come only from band dates.

- [x] 2.6 F.0b-ii:

  - the observation as the truth;
  - the recovery distance;
  - the truth's stability: no statistic's definedness flips across re-simulations 1–10, or the change returns to the maintainer;
  - the preflight record and its checks, at the bars the stand-in preflight sets, run once;
  - the band-centre reference's gain, reported;
  - the gate on 10 re-simulations.

  Tests: the gain arithmetic; the stability check; the preflight's refusals.

- [x] 2.7 The seed layout, with the smoke's own layout inside 99,880,000–99,889,999. Tests:

  - every block disjoint from the others and from the seed table's earlier blocks;
  - 2a's last candidate inside its sub-block;
  - the smoke inside its range.

- [x] 2.8 The guards:

  - `pilot`, `recovery-preflight`, `recovery`, `run`, `grade`, `read` and `loo`, each as registered;
  - the digest manifest, checked at start and at each resume;
  - v4's validity;
  - HEAD in every record.

  Tests: each refuses before its part, on a failed go rule or gate, and on a changed manifest file.

- [x] 2.9 `speed-noise-control`: the registered control's rule on Logbook 131's answer under the procedure settings and the speed-following arm (888,000,000 + i). The answer runs at each of the control's timescales, and without the OU noise with the timescale entry dropped.

- [x] 2.10 The smoke, discarded (CMA 9508–9511): run once end to end on its own layout. It covers:

  - two generations of each stage;
  - the noise step;
  - the grade;
  - the preflight arithmetic;
  - the held-out reading on band-cohort date sets.

## 3. The control and the registration

- [x] 3.1 Run `speed-noise-control` (about 30 min). Return to the maintainer if it fails.

- [ ] 3.2 Launch record 133, Part A:

  - every rule, gate and seed in the design;
  - the go rule and the kill rules;
  - F.0b-ii's bars, with the stand-in preflight that set them disclosed (its script and record);
  - the speed-noise control's result;
  - Logbook 131's probes disclosed;
  - the digest manifest;
  - the encoding table bound.

  Committed with:

  - the dated D37 and D38 amendments in the roadmap;
  - the tracker's notes: C.2, F.0, W.2 superseded, and the seed table (C.2's row: animals at 885–999 M, not 5101–5356).

## 4. The readiness pilot

- [x] 4.1 `pilot` (about 30 h, discarded seeds). Stop and return to the maintainer on a kill rule. *(2026-11-11: stopped by 2a's kill rule, D_fit 3.13 against 2.5.)*

- [x] 4.1a Diagnose the first pilot on discarded seeds, each run's rule fixed before it ran: the two screens, the curriculum's 2a replayed on the registered arm, another CMA seed, the continuations, start (c) and the lawn-guard probe.

- [x] 4.1b The amendment of 2026-11-12 (design decision 3a), on the maintainer's decision:

  - `c2_fit.py`: stage 1 at 64 animals and population 24; up to six starts of the first half; the lawn-holding reading at 2a's generations 40, 50 and 60 (bar 0.95); each start's CMA seeds and animals; a run with no answer in the pilot, the recovery, the fit and the leave-one-out;
  - the pilot on a new base (1,100,000,000) and new CMA seeds.

  Tests:

  - a start that does not hold the lawn is dropped, and the next begins from the box's centre on its own seeds;
  - no start holds the lawn;
  - the reading is a mean of its three readouts;
  - an interrupted drop resumes to the same answer;
  - each start has its own animals and CMA seeds, inside the run's block and C.2's CMA band;
  - a start's candidates are simulated on its own animals;
  - stage 1's checkpoint stops only the pilot;
  - a registered fit with no answer fails, and a recovery with no answer does not pass; the pilot, the leave-one-out and the smoke with no answer;
  - a run resumes inside any start to an uninterrupted run's answer;
  - each start is read against its own bar, and every start's 2a opens at the 1-s timescale;
  - F.0b-ii's preflight refuses a truth below the lawn-holding bar;
  - deliberate bugs in the start loop, each caught;
  - the smoke runs a dropped start and a kept one.

- [x] 4.1c Part A amended and committed before the pilot runs again, with:

  - the first pilot's record (`pilot_first.json`) and every diagnosis script and record under `probes/`;
  - the manifest rebuilt;
  - D38's note in the roadmap (stage 1 below 200 animals) and the tracker's seed row.

- [x] 4.1d `pilot` again (about 29 h, and about 2.1 h for each dropped start). *(2026-11-13: no start of six held the lawn, and the pilot stopped with no answer after 15.8 h; `pilot_second/`.)*

- [x] 4.1e Diagnose the second pilot on discarded seeds, each run's rule fixed before it ran: the guard probe resumed, two of its dropped starts' 2a again under the lawn guard, and a first half from the box centre with the guard in stage 1 and 2a.

- [x] 4.1f The amendment of 2026-11-13 (design decision 3b), on the maintainer's decision:

  - `c2_fit.py`: the lawn guard in the objective, in every stage, in the settings a resume checks, and each generation's share of candidates paying it recorded;
  - the pilot on a new base (1,140,000,000) and new CMA seeds (9591–9594, restarts 9651–9660), clear of the seeds the Phase 10 table gives other work.

  Tests:

  - the lawn guard's values below, at and above 0.95, undefined, beside the other guards and in every stage's objective;
  - a resume with a changed lawn guard is refused;
  - each generation's share of candidates paying it is recorded in every stage;
  - the second pilot's and its probes' seed blocks, and the CMA seeds the seed tables give other work, are earlier blocks;
  - deliberate bugs in the guard and the pilot's seeds, each caught, and the first amendment's again on the new code.

- [x] 4.1g Part A amended and committed before the pilot runs a third time, with the second pilot's record, every diagnosis script and record under `probes/`, the manifest rebuilt, D38's note in the roadmap and the tracker's seed row.

- [x] 4.1h `pilot` a third time (about 30 h with its first start kept). *(2026-11-14: its first start held the lawn and was kept, and 2a's kill rule stopped it at 2a's end, D_fit 3.83 against 2.5, after 9.9 h; `pilot_third.json`.)*

- [x] 4.1i Diagnose the third pilot on discarded seeds, its rule fixed before it ran: every guarded 2a against 2a's bar, and the dopamine wiring gain mirrored at five answers (`probes/dopamine_flip.py.txt`).

- [x] 4.1j The amendment of 2026-11-14 (design decision 3c), on the maintainer's decision:

  - `c2_fit.py`: 2a's descent reading (generations 130–150, D_fit at most 4.25) and bar reading (280–300, at most 2.5), each dropping a start in every run; 2a's kill rule removed; every reading a start takes recorded; the readings' readouts and bars in the settings a resume checks; a run with readings out of order, past 2a or without a readout refused;
  - each start's own animals for all of 2a, so 40,000,000-animal blocks and new bases for every run (pilot 1,200,000,000, recovery 1,240,000,000, fit 1,280,000,000, leave-one-out 1,320,000,000);
  - the pilot's new CMA seeds (9661–9664, restarts 9671–9680), clear of the fit-first investigation's;
  - F.0b-ii's preflight requires the truth's D_fit at most 2.5 against the bands, as a kept start's must be;
  - the neural reading declared in Part A as description, with its paired test and wiring-only control;
  - the smoke's four starts, failing the lawn, the descent and the bar in turn.

  Tests:

  - a start dropped at the descent and at the bar, each at its own generation, with every reading it took recorded, and each start read against its own bars;
  - no start kept when every start misses the bar;
  - the descent reading's mean, its undefined D_fit and its bar, and each reading in a run taking the mean of all its readouts at its last one;
  - the preflight refusing a truth past 2a's bar;
  - a resume across each drop, and an interrupted drop at each reading, each resuming rather than restarting;
  - a resume with a changed reading refused, and a run with a reading out of order, past 2a or without a readout, or a 2a past its seeds, refused;
  - every start's 2a on its own animals in every generation, inside the run's block;
  - the earlier blocks and CMA seeds, including the third pilot's, its diagnosis's and the investigation's;
  - deliberate bugs in the readings, the layout and the seeds each caught, and both earlier amendments' again on the new code.

- [x] 4.1k Part A amended and committed before the pilot runs a fourth time, with the third pilot's record, its diagnosis's script and record and start 3's guarded 2a under `probes/`, the manifest rebuilt, D38's note in the roadmap and the tracker's notes (C.2's seed row and its neural reading).

- [x] 4.1l `pilot` a fourth time (about 30 h with its first start kept; about 4.4 h for a start dropped at the descent and 8 h at the bar). *(2026-11-16: it dropped starts 0 and 1 at the lawn and the descent and kept start 2, which passed 2a's bar at 1.85; its answer failed the go rule, the wild type 0 of 5 and every mutant 5 of 5, after 36.6 h; `pilot_fourth.json`.)*

- [x] 4.1m Diagnose the fourth pilot on discarded seeds, its rule fixed before it ran: 2b again from the pilot's 2a answer as a new isotropic search at σ₀ 0.05, without the mutants (arm A) and with them at half (arm B), 60 generations each (`probes/two_b_screen.py.txt`). Both moved local search; by the rule, B's moving decides.

- [x] 4.1n The amendment of 2026-11-16 (design decision 3d), on the maintainer's decision:

  - `c2_fit.py`: 2b starts a new isotropic search at 2a's answer, σ₀ 0.05, on the run's third CMA seed; the final stage still carries 2b's state;
  - the pilot on a new base (1,360,000,000) and new CMA seeds (9682–9685, restarts 9686–9695);
  - the neural reading: its share of the measured ceiling (0.423), the paired test's class resample (`default_rng(9811)`) with the neuron resample beside it (`default_rng(9812)`), off 9781, and Logbook 135's calibrated references quoted.

  Tests:

  - 2b's strategy isotropic at 0.05 at 2a's answer on the run's third seed, whichever start was kept, and only 2b's state, its own final σ, carried to the final stage;
  - a resume with 2b's start changed refused;
  - the pilot alone has kill rules;
  - the fourth pilot's, the screen's and the timing check's blocks, and their CMA seeds and Logbook 135's generators, as earlier blocks;
  - the paired test's generators clear of every earlier one and of every run's CMA seeds;
  - deliberate bugs in 2b's start, the pilot's seeds and the neural reading's generators each caught, and the third amendment's again.

- [ ] 4.1o Part A amended and committed before the pilot runs a fifth time, with the fourth pilot's record, the screen's script and records and the timing check under `probes/`, the manifest rebuilt, D38's note in the roadmap and the tracker's notes (C.2's seed row and its neural reading).

- [ ] 4.1p `pilot` a fifth time (about 31 h with its first start kept).

- [ ] 4.2 Read the go rule. Part B: the result, the cost per stage and the plateau reading, committed. If the go rule fails, diagnose it and return to the maintainer before anything registered.

## 5. F.0b-ii

- [ ] 5.1 `recovery-preflight`: the observation, its stability, floor, oracle, prior D, box centre and the band-centre reference. Return to the maintainer if a check fails. Commit the record before `recovery` runs.
- [ ] 5.2 `recovery` (about 30 h, and 2.3–8 h for each dropped start): the gate. Return to the maintainer if the gain is below 0.75.
- [ ] 5.3 Part C, the recovery's result, committed.

## 6. The registered fit

- [ ] 6.1 `run` (about 30 h, and 2.3–8 h for each dropped start), `grade` (5 + 10 populations) and `read`: `fit.json` with the verdict, the pass rate, the held-out reading, the restart spread against the pilot, and the cost.

- [ ] 6.2 The neural reading's paired test and wiring-only control written to Part A's rules, tested on Logbook 131's answer and committed with Part B or Part C before `run`; then the reading on the registered answer, as Part A declares it: the answer's r and the box centre's, chance, the paired test, the wiring-only control and sign agreement, as description beside `fit.json`.

## 7. The tax-4 leave-one-out

- [ ] 7.1 `loo` (about 26 h), read beside the registered fit as description.

## 8. Close-out

- [ ] 8.1 Docs:
  - Logbook 133;
  - the tracker (C.2, F.0, C.4, W.2's compute input, the seed table) and the logbook index row;
  - AGENTS.md;
  - `configs/README.md` (the option set by the registered arm);
  - the CHANGELOG.
- [ ] 8.2 Checks and merge:
  - the full suite;
  - `pre-commit run --all-files` after `git add -A`;
  - the privacy check on every diff (no off-repo archive path);
  - `openspec validate --strict`;
  - archive;
  - PR.
