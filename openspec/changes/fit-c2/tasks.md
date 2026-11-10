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
  - the search state carried from 2a to 2b and from 2b to the final stage (σ and `CMA_stds` = stds/σ);
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

- [ ] 2.10 The smoke, discarded (CMA 9508–9511): run once end to end on its own layout. It covers:

  - two generations of each stage;
  - the noise step;
  - the grade;
  - the preflight arithmetic;
  - the held-out reading on band-cohort date sets.

## 3. The control and the registration

- [ ] 3.1 Run `speed-noise-control` (about 30 min). Return to the maintainer if it fails.

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

- [ ] 4.1 `pilot` (about 30 h, discarded seeds). Stop and return to the maintainer on a kill rule.
- [ ] 4.2 Read the go rule. Part B: the result, the cost per stage and the plateau reading, committed. If the go rule fails, diagnose it and return to the maintainer before anything registered.

## 5. F.0b-ii

- [ ] 5.1 `recovery-preflight`: the observation, its stability, floor, oracle, prior D, box centre and the band-centre reference. Return to the maintainer if a check fails. Commit the record before `recovery` runs.
- [ ] 5.2 `recovery` (about 30 h): the gate. Return to the maintainer if the gain is below 0.75.
- [ ] 5.3 Part C, the recovery's result, committed.

## 6. The registered fit

- [ ] 6.1 `run` (about 30 h), `grade` (5 + 10 populations) and `read`: `fit.json` with the verdict, the pass rate, the held-out reading, the restart spread against the pilot, and the cost.

## 7. The tax-4 leave-one-out

- [ ] 7.1 `loo` (about 27 h), read beside the registered fit as description.

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
