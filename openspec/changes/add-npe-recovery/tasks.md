# Tasks: F.0a — NPE recovers C.1's baseline

Phase 10, block F, tracker item F.0 (first half). The maintainer's decisions of 2026-10-25:

- two changes, NPE first;
- `sbi`;
- calibration and statistic-space gates;
- sep-CMA at 4 truths × 2 seeds, in F.0b.

## 1. The model

- [ ] 1.1 `BatchedMechanistic` in `evaluation/controllers.py`:
  - 20 named parameters;
  - per-animal state, previous contact and time off food;
  - one generator call per animal per second;
  - I = 0 where `max_intake` is undefined.
- [ ] 1.2 Tests:
  - klinokinesis lowers hazards up-gradient;
  - a falling contact raises reversals by exp(e × fall);
  - search decays off food;
  - the same alone and in a batch;
  - mean action refused.

## 2. NPE tooling

- [ ] 2.1 Add `sbi` to the `analysis` extra; `uv lock`. Confirm it imports under Python 3.13 and the
  project's torch.
- [ ] 2.2 `scripts/analysis/npe_recovery.py`:
  - the prior;
  - the simulator (θ → statistics at 200 animals, three assays);
  - the sentinel-and-flag summary;
  - a resumable process pool writing each simulation as it completes;
  - training two flows;
  - SBC coverage, contraction, the informative gate (posterior- against prior-predictive D) and
    coverage on the truths nearest the wild type.
- [ ] 2.3 Tests:
  - the prior's ranges;
  - the summary's sentinels and flags;
  - coverage on a toy problem with a known posterior (a Gaussian), where the gate passes;
  - an overconfident toy posterior fails;
  - a prior-returning posterior passes coverage but fails the informative gate.

## 3. Registration

- [ ] 3.1 A pilot: 64 simulations from the prior, on its own seed scheme. Seconds per simulation and
  the projected wall time are written to the launch record.
- [ ] 3.2 `supporting/111-npe-recovery/launch.md`: the model, priors, summary, seeds, NPE, both gates,
  the pilot, and why `gate_preflight.py` does not apply. Committed before 4.1.

## 4. The run

- [ ] 4.1 10,000 training simulations and 100 truths.
- [ ] 4.2 Train both flows; read the gate; write the record.

## 5. Documentation and tracker

- [ ] 5.1 Logbook 111: the gate, coverage and contraction per parameter, the posterior-predictive
  check, the cost.

- [ ] 5.2 The tracker:

  - F.0's NPE half;
  - the seed scheme in the seed table;
  - C.1's note: simulate at the observation's size, or add between-date noise.

  AGENTS.md and the CHANGELOG.

## 6. Close-out

- [ ] 6.1 Full suite and `pre-commit run --all-files` after `git add`.
- [ ] 6.2 `openspec validate add-npe-recovery --strict`.
- [ ] 6.3 Archive, with the new specs' Purposes written.
- [ ] 6.4 PR.
