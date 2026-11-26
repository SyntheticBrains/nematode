## Why

C.1 has no answer that passes benchmark v4, the current benchmark. C.4's fidelity table needs one, beside C.2's failing in-sample row.

C.1 (v3)'s answer ([Logbook 121](../../../docs/experiments/logbooks/121-c1-v3-fit.md)) fails v4 in all 5 of its registered populations. Its on-lawn onset rate is 0.78–0.85 per minute, against real worms' 2.28 (z −6.4 to −6.7; [Logbook 124](../../../docs/experiments/logbooks/124-benchmark-v4.md)).

[Logbook 125](../../../docs/experiments/logbooks/125-c1-v4-diagnosis.md) found the missing mechanism and deferred the refit:

- **The mechanism:** a dwelling reversal rate of its own on food, `BatchedMechanistic`'s `r_dwell_on_food`. It is already in the package, off by default and byte-identical when off.
- **A route from C.1 (v3)'s answer reaches the basin:**
  - 100 generations without the new rate passed 0 of 5;
  - then 101 more, with the rate started at 0.03, passed 4 of 5 (D 1.28–1.44);
  - neither run had the mutants in its objective.
- **The box centre does not:** from there, 0 of 5 in each of three runs.

Three things have changed since the deferral:

- **The objective now includes the mutants.** D37's amendment of 2026-11-10 fits C.1's refit on all six families together with its encodable mutants, tph-1 and tax-4, which make up half the objective, as in C.2's fit.
- **The body has changed.** Logbooks 131 and 132 fixed the on-food reversal that follows the worm's own speed as the registered body for C.1's refit.
- **The maintainer has chosen the start.** The direction decision of 2026-11-23, confirmed 2026-11-26, is a warm start from C.1 (v3)'s answer, under a dated D38 amendment, with a new pilot.

D38 as amended in 2026-10-25 sets C.1's "search box and x₀ … without search data". A warm start needs that amended.

**A probe on discarded seeds** (2026-11-26, before any registration) read two more things:

- **The second truth is too close.** Logbook 125's passing answer sits 0.91 above its floor in distance from the warm start, short of the recovery's 1.0, so the maintainer chose one recovery truth, the pilot's answer.
- **The mutants go the wrong way at the start.** tph-1's knockout lowers roaming there (Δ −0.19, against real worms' +0.27), so the fit must reverse it.

## What Changes

- **D38 amended (2026-11-26, the maintainer's decision):**
  - C.1's refit on v4 starts from C.1 (v3)'s registered answer at σ₀ 0.1, the new parameter at its prior box's centre.
  - Its recovery's gain is measured from that start.
  - Its mutant rows follow the encoding table's `unidentified` rule across the restarts that pass the wild type, beside a verdict read on the chosen restart.
  - The fit is disclosed as a local refinement.
- **A speed-following grading arm for C.1:**
  - `generate_assay_configs.py` gains the arm `internal_following`: the `internal` arm's configs with `continuous.on_food_reversal_speed: follows_run_speed`, their only difference.
  - `benchmark_grade.ARMS["mlp_following"]` names it.
  - Every other config stays byte-identical.
- **`scripts/analysis/c1_v4_fit.py`,** in the shape of `c1_v3_fit.py`:
  - **The model:** C.1 (v3)'s 25 parameters plus `r_dwell_on_food`.
  - **The start:** C.1 (v3)'s answer, the new rate at its box's centre.
  - **The objective:** the ranking score against v4's bands, with the tph-1 and tax-4 differences at half by C.2's formula.
  - **The stop:** the doubly guarded plateau, tightened, with σ guarded at the registered σ₀.
  - **The stages:** `pilot`, `recovery-preflight`, `recovery`, `run`, `select`, `grade`, `read` and `loo`, each guarded by the launch record and by the fit's code being unchanged since Part A, plus a discarded `smoke`.
- **A pilot** on discarded seeds, configured as the fit:
  - **Its go rule:** the wild type, tph-1 and tax-4 each in at least 3 of 5.
  - **Its other checks:** the recovery's preflight checks on its own answer, and a cost bar of 48 h of projected registered compute.
  - **It sets the cap,** max(300, 1.5 × its stop), rounded up to a multiple of 10, at most 500.
- **The recovery, this refit's F.0 control:**
  - **The truth:** the pilot's answer, fitted twice by the registered procedure from the registered start.
  - **The gain:** measured from the start, (start D − D) / (start D − floor).
  - **The gate:** the better restart's gain at least 0.75.
  - **Beside it:** the prior-relative gain and the band reference's gain from the start are reported.
- **The registered fit:**
  - **The restarts:** 12, with selection on 5 separate populations per plateaued restart.
  - **The verdict:** the wild type, tph-1 and tax-4 each pass in at least 3 of 5 registered populations. Every branch of the verdict map names what C.4 receives.
  - **Reported beside the verdict:**
    - the mutant rows under the `unidentified` rule;
    - 10 more populations' pass rates;
    - the held-out reading on Scheer's held-out recording dates, as C.2's was.
- **A tax-4 leave-one-out,** as description: one restart with tax-4 out of the objective, graded on 15 populations. It is C.1's one predicted mutant row, and both outcomes are always shown.
- **benchmark-controls' held-out rule** is scoped to the benchmark's own validation. A registered fit's held-out reading under D37 is the exception, read only in that fit's `read` stage.
- **Launch record 136,** in three parts:
  - **Part A** before the pilot;
  - **Part B** after it, before the recovery;
  - **Part C** after the recovery, before the fit.
- **Logbook 136,** with the close-out.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `controller-fit`: C.1 refitted on benchmark v4 from a warm start, with its encodable mutants fitted at half the objective, on the speed-following arm. It adds a pilot with a go rule and a cost bar, a verdict on the wild type and both mutants, mutant rows under the `unidentified` rule, a held-out reading and a tax-4 leave-one-out.
- `fit-recovery`: a warm-started fit's recovery, its gain measured from the registered start, on the pilot's answer.
- `benchmark-controls`: the held-out cohorts' no-reread rule is scoped to the benchmark's validation, with D37's registered held-out reading as the exception.

## Impact

- **Code:**
  - `scripts/campaigns/generate_assay_configs.py` (one arm);
  - `scripts/analysis/benchmark_grade.py` (one `ARMS` entry);
  - `scripts/analysis/c1_v4_fit.py` (new);
  - tests.
- **Configs:** six `mlpppo_small_continuous2d_fick_adaptive_klinotaxis_lawns_internal_s1_primitives_following_assay_*.yml`.
- **Records:**
  - `docs/experiments/logbooks/supporting/136-c1-v4-fit/`: the probe, the launch record, the pilot, the recovery, the fit and the leave-one-out;
  - Logbook 136 and its index row;
  - the roadmap (the D38 amendment, C.1, C.4);
  - the tracker (C.1 on v4, the seed table);
  - AGENTS.md, `configs/README.md` and the CHANGELOG.
- **No package code changes.** `BatchedMechanistic` already carries `r_dwell_on_food`, and the batched evaluator already carries the speed-following reversal.
- **Cost:** 35 s per evaluation with the mutants, from the probe.
  - At a 200-generation route the registered compute is about 31 h: 2 recovery restarts, 12 registered restarts and the leave-one-out. The pilot adds about 2 h.
  - The pilot measures the cost, and Part B reads it against the 48 h bar.
  - Every fit resumes from its saved state.
