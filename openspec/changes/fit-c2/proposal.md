## Why

C.2's encoded connectome reaches benchmark v4 from a clean start. On 15 fresh populations graded with presence, Logbook 131's last answer passes all 24 wild-type statistics and tph-1, pdfr-1 and tax-4 in every one. Under the package's speed-following reversal (Logbook 132), it passes 10 of 10.

That answer is readiness evidence, not a verdict:

- the route was adapted after each reading;
- it ran once, on discarded seeds;
- part of it ran on a body patched in scripts.

The registered fit has to fix its procedure in advance, run it fresh, and pass the recovery control D38 requires (F.0b-ii).

The roadmap also no longer matches what C.2 needs:

- **D37** fits the wild type only and holds out local search, with the mutants as predictions. Logbook 128 showed that wild-type data do not identify C.2's perturbation responses, so C.2 fitted that way fails its mutants.
- **D38** has no staged procedure, and its recovery control for C.2 (F.0b-ii) has not run.

The maintainer's decisions (2026-11-10), from the change plan:

1. **Procedure B.** The fitted families (300 generations), then all six families with the mutants at half the objective (200), then a noise measurement, then the failure-probability stage (50). The search state is carried between stages, and the speed-following reversal is on. The design puts B04's stage 1 before them: 2a starts from its answer, and it is the clean start's only evidence.
2. **A readiness pilot first,** on discarded seeds. Its go rule: the wild type in at least 3 of 5 populations, and tph-1 and tax-4 each in at least 3 of 5.
3. **F.0b-ii as one truth, before the fit.** The truth is the pilot's answer, refitted from the box centre by the same procedure, with a floor-relative gain gate preflighted against the oracle. The order is pilot, then F.0b-ii, then the registered fit.
4. **Symmetric fit splits.** C.2, and C.1's refit before C.4, each fit their encodable mutants at half the objective. Both are read on Scheer's held-out recording dates beside W's minimum, plus one tax-4 leave-one-out as description.

## What Changes

- **Slow speed noise's positive control, owed since Logbook 123,** re-run under its registered rule on Logbook 131's answer, whose speed output responds, before Part A. Every stage relies on that noise.

- **A grading arm with the registered body.** `generate_assay_configs.py` gains the arm `encoded_following`. Its assay configs are the encoded arm's with `continuous.on_food_reversal_speed: follows_run_speed`, and `benchmark_grade.ARMS["connectome_c2_following"]` names them. The body is in the config, not in a hook.

- **`scripts/analysis/c2_fit.py`**, the procedure fixed in advance, in the shape of `c1_v3_fit.py`:

  - the objective: the ranking score, the mutants at half the objective on the encoding table's graded statistics, the saturation and graded responsiveness guards, a lawn guard below 0.95 on the lawn (from the amendment of 2026-11-13), and each statistic's chance of failing a population;
  - four stages at population 48 and 200 animals (stage 1 at population 24 and 64 animals, and the first half started up to six times, from the amendment of 2026-11-12; a start kept only when 2a holds the lawn, keeps up its descent and reaches its bar, from the amendment of 2026-11-14): stage 1 on the 77 parameters that carry senses to outputs, 2a, 2b with the mutants, and the failure-probability stage;
  - each later stage starts from the one before's mean, σ and per-parameter steps;
  - the noise measurement as a step of its own;
  - readouts every 10 generations recording every statistic, and state saved every generation;
  - the stages `smoke`, `pilot`, `recovery-preflight`, `recovery`, `run`, `grade`, `loo` and `read`, each guarded by the launch record.

- **F.0b-ii, C.2's recovery control:** one truth, the pilot's answer.

  - Its floor, oracle, prior and box-centre gains are preflighted before its fit, with bars set from a stand-in preflight on discarded seeds.
  - The fit runs from the box centre by the registered procedure, and the gate is a median gain of at least 0.75.
  - The truth sits beside the wild type, so the gate cannot tell recovering it from fitting the wild type. A record at the bands' centres is scored beside the gate as a reference, and the D38 amendment states the limit: F.0b-ii tests reaching the region and the run-to-run spread, not identifying the truth.

- **The held-out reading:** the fitted worm against Scheer's held-out dates, on the same splits, scales and families as W's minimum. It reuses the registered populations and needs no new simulation. Two real references are read beside it:

  - the real halves against each other, which is W's minimum;
  - 200-animal populations drawn from the band cohorts against the same halves, which is what a faithful simulator would score.

  The verdict itself is in-sample, since every graded statistic and both resolved mutants are fitted.

- **Launch record 133** in three dated parts: A before the pilot, B after it, C after the recovery. Part A binds the digests of the script, configs and data every stage reads, and each stage refuses on a mismatch. **Logbook 133** records the chain.

- **Dated roadmap amendments:**

  - D37, for the fit split and what is held out;
  - D38, for the staged procedure with fixed per-stage budgets, the plateau reported rather than used as a stop, and F.0b-ii's one truth before the fit.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `controller-fit`: C.2's registered fit, comprising its procedure, the readiness pilot, its verdict, its held-out reading and leave-one-out, its grading arm, and its registration in three parts.
- `fit-recovery`: F.0b-ii, sep-CMA's recovery of C.2's procedure at one truth.

## Impact

- **Code:**
  - `scripts/analysis/c2_fit.py` (new);
  - `scripts/campaigns/generate_assay_configs.py` and six generated configs;
  - `scripts/analysis/benchmark_grade.py` (one arm);
  - tests.
- **Records:**
  - `docs/experiments/logbooks/supporting/133-c2-fit/` (the launch record, the pilot, the recovery, the fit, the readings) and Logbook 133;
  - the roadmap's D37 and D38 rows;
  - the tracker (C.2, F.0, C.4, W.2, the seed table);
  - AGENTS.md, `configs/README.md` and the CHANGELOG.
- **Cost** on 16 workers, at the measured rates (2a's 77.7 s a generation on two assays, and the speed-following refit's 309.5 s on six with the mutants, readouts included):
  - a run of the procedure takes about 31.5 h: stage 1 3.2 h, 2a 6.5 h, 2b 17.2 h, the final stage 4.3 h, plus noise and grading;
  - the pilot, the recovery and the registered fit are three runs;
  - the speed-noise control and the recovery's preflight add about 0.5 h each;
  - the leave-one-out, after the fit, is a fourth run of about 28.5 h;
  - in all, about 124 h, or 5.2 days of machine time. *(After the amendment of 2026-11-12: about 114 h with every first start kept, about 127 h at 2 starts in 5; design decision 3a. After the amendment of 2026-11-14: about 119 h with every first start kept, and about 3 h more a run on average for starts dropped in 2a; design decision 3c.)*
- **Engineering:** about 2 days.
