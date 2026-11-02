## 1. The model's options

- [ ] 1.1 `BatchedMechanistic`: `v_off_food`, `sigma_off_food`, and the paired speed spread, each off by default. Tests for each scenario. (The code and two tests exist uncommitted from the probes; they are committed with this task.)
- [ ] 1.2 `speedup_identity_check.py check --build build/benchmark-v3/identity --out supporting/121-c1-v3-fit/identity.json`: all 55 jobs byte-identical.

## 2. The fit script

- [ ] 2.1 `scripts/analysis/c1_v3_fit.py`:

  - the 25-parameter box;
  - the v3 band objective and simulation;
  - the seed scheme;
  - the tax-4 encoding;
  - the stages `recovery-preflight`, `recovery`, `run`, `select`, `grade`, `read` and `smoke`, guarded as registered.

  Tests:

  - disjoint seed blocks, including the answers' offset of 8,000 above every generation's index under the cap;
  - the guards;
  - the encoding;
  - the recovery gate's arithmetic, on the capped v3 D;
  - parameter errors read modulo the dwelling/roaming swap;
  - selection and ties.

- [ ] 2.2 The smoke (CMA 5990, animals 86,000,000, grading 86,970,000, discarded): two generations of each stage, `recovery-preflight` and `recovery` included.

## 3. Registration and the recovery

- [ ] 3.1 The probes' records and scripts, the sensitivity tests and the recovery gate's preflight, committed to `supporting/121-c1-v3-fit/probes/`.
- [ ] 3.2 Launch record 121, Part A (rules, seeds, the encoding extension, the probes disclosed), committed.
- [ ] 3.3 `recovery-preflight`: the observations, floors, oracle, prior D and the box centre's gain. Stop if a check fails.
- [ ] 3.4 `recovery`: 4 fits, the gains, the gate. Return to the maintainer if the median gain is below 0.75.
- [ ] 3.5 Part B: the recovery's result, committed.

## 4. The fit and the verdict

- [ ] 4.1 `run` (12 restarts), `select` (5 populations per plateaued restart), `grade` (every restart on 5 registered populations), `read`: `fit.json` with the verdict.

## 5. Close-out

- [ ] 5.1 Logbook 121, the tracker (C.1 on v3, C.4, the seed table), the index row, AGENTS.md and the CHANGELOG.
- [ ] 5.2 At archive, `mechanistic-baseline`'s Purpose names the optional off-food kinematics and speed spread. Full suite; `pre-commit run --all-files` after `git add -A`; `openspec validate --strict`; archive; PR.
