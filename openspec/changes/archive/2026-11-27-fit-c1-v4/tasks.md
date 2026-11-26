## 1. The speed-following arm

- [x] 1.1 `generate_assay_configs.py`: the arm `internal_following` (stem `…_internal_s1_primitives_following`), its configs the `internal` arm's with `continuous.on_food_reversal_speed: follows_run_speed`. Generate the six configs. Every other arm's output stays byte-identical.
- [x] 1.2 `benchmark_grade.ARMS["mlp_following"]`.
- [x] 1.3 Tests:
  - both following arms differ from their bases only in the key;
  - every committed assay config re-derives byte-identically;
  - `BatchedMechanistic` runs on the arm in the batched evaluator, checked on 16 discarded animals.
- [x] 1.4 `configs/README.md`: the six new configs, and the following option now set by two arms.

## 2. The fit script

- [x] 2.1 `scripts/analysis/c1_v4_fit.py`:
  - **the model and the start:** the 26-parameter model, and the start from `121-c1-v3-fit/fit.json`;
  - **the evaluation:** the wild type, tph-1 and tax-4 on the same animals;
  - **the objective,** with the mutants at half;
  - **the stop and the cap:** the stop, the cap rule and the cost bar;
  - **the seed scheme;**
  - **the recovery:** its distance, the gain from the start, the preflight's separation, oracle and stability checks, the band reference's gain, the gate line and the gate;
  - **the readings:** selection, the verdict, the mutant rows under the `unidentified` rule, the pass rates, the held-out reading (via `c2_fit`, imported in the main process only) and the leave-one-out with both labels;
  - **the stages:** `smoke`, `pilot`, `recovery-preflight`, `recovery`, `run`, `select`, `grade`, `read` and `loo`, each guarded by its parts, its bound records and the code at Part A's commit, and each writing its record once.
- [x] 2.2 Tests, each naming its spec scenario, the stage logic factored into pure functions and the guards tested through `guarded()`. The independent review's 25 planted bugs caught only 2 of the first tests; after the rewrite, 37 of 37 planted bugs are caught (`review136/mutate2.py` in the session's scratch, not committed).
- [x] 2.3 `smoke` (CMA 9469, animals 699,000,000–699,400,199, discarded): it ran end to end in 4.9 min.
  - The first run read the held-out sets through C.2's defaults. That is disclosed in Part A.
  - The smoke now passes band-cohort stand-ins, which a test checks.
- [x] 2.4 Re-run the smoke on the final script: exit 0, D 4.2527 as before.

## 3. Registration and the pilot

- [x] 3.1 The separation probe before Part A (`probes/separation.py.txt` and `.json`, 647 M, discarded): separation 0.91, the mutants the wrong way, 35 s per evaluation. The maintainer chose one truth.

- [x] 3.2 The roadmap's D38 amendment (2026-11-26), and the D37 note keeping C.1's leave-one-out.

- [x] 3.3 Launch record 136, Part A:

  - every rule;
  - the bound records and code;
  - the probes, the separation probe and the first smoke's held-out read disclosed;
  - the held-out reading authorised;
  - the tracker's seed-table row.

  An independent review, then commit.

- [x] 3.4 `pilot`: the restart, its 5 grades, the go rule, the preflight's check on its answer, the cost bar and the cap. Stop and return to the maintainer if any part fails. *(2026-11-27, 1.34 h: it stopped on its plateau at generation 113, and the check and the cost bar (20.1 h) held. **It failed the go rule:** the wild type 0 of 5 (roaming windows' speed median), tph-1 5 of 5, tax-4 0 of 5 (exits per on-lawn hour); `pilot.json`. It returned to the maintainer. A probe on discarded seeds (`probes/tax4_retention`) finds tax-4 out of reach for C.1's model. The maintainer closed the refit as a no-go.)*

- [x] 3.5 Part B: the pilot's result, the cap, the truth bound, the measured cost. Commit. *(2026-11-27: Part B records the no-go, the probes seen after it and the maintainer's decision. It changes no rule.)*

## 4. The recovery

- [x] 4.1 *(Not run: the pilot's go is false, 2026-11-27.)* `recovery-preflight`: the observation, floor, oracle, start D, prior D, band reference, gate line and checks. Stop if a check fails.
- [x] 4.2 *(Not run.)* `recovery`: 2 fits, the gain, the gate. Return to the maintainer if the gain is below 0.75.
- [x] 4.3 *(Not written.)* Part C: the recovery's result. Commit.

## 5. The fit, the verdict and the leave-one-out

- [x] 5.1 *(Not run: no registered fit, verdict or held-out reading.)* `run` (12 restarts), `select`, `grade` (every restart on 15 registered populations) and `read`: `fit.json`, with the verdict, the mutant rows and the held-out reading.
- [x] 5.2 *(Not run.)* `loo`: the leave-one-out and its 15 grades, `loo.json`.

## 6. Close-out

- [x] 6.1 Logbook 136 and its index row. Update:
  - the roadmap (C.1, C.4, the C.1 risk row, D37 and D38);
  - the tracker (C.1 on v4);
  - AGENTS.md;
  - the CHANGELOG.
- [ ] 6.2 The full suite, then `pre-commit run --all-files` after `git add -A`, then `openspec validate --strict`. Archive the change and open the PR.
