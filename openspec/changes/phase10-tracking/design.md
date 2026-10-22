## Overview

Phase 10, *fidelity on the plate*, is planned in `docs/roadmap.md` § Phase 10 (v5.0, 2026-10-23) from
the plan review of 2026-10-23 and its adversarial critique. The review is gitignored, at
`build/phase10-review-2026-10.md`. This design holds the execution standards, the seed bands, and the
places the roadmap leaves to the per-milestone changes.

## Goals / Non-Goals

**Goals**

- Track Phase 10 block by block, with every milestone PR updating `tasks.md`.
- Make D33–D41 binding on every Phase 10 change.

**Non-goals**

- Implementation. Each block lands as its own change.

## Design Decisions

### Decision A: The roadmap's D33–D41 are binding here, amended only here

A Phase 10 change that contradicts a D-decision is refused at its spec review. An amendment is dated,
here and in the roadmap, before the change that needs it registers.

### Decision B: Execution-protocol standards (inherited, non-optional)

Inherited from Phase 9's tracker, where they still apply:

- paired-seed statistics with bootstrap intervals and BH-FDR;
- byte-identical-when-off for every new mechanism;
- load-time validation of every new key;
- a positive control for every new component (protocol principle 4);
- sweep before pin;
- re-establish a moved baseline in its own campaign;
- a minimum registered in both directions;
- artefact retention (A.0);
- gate preflight before launch;
- sizing from an upper bound on a pilot's spread;
- a failed positive control retried only as a new registration;
- every null with its matched-statistics table and shortcut audit (D24);
- the collapse statistic beside every curve beyond 3,000 episodes, should any reward-trained arm run
  that long;
- a harness reads a run only if its log records the config its name claims.

Phase 10's additions:

1. **One step-duration value.** Every step-coupled constant derives from it, and a test asserts the
   derivation (D35). Results at the 5-s default are byte-identical to before.
2. **Same primitives for every arm and every negative** (D36). Reorientations and states are read from
   trajectories, never from primitive labels.
3. **The benchmark's bands come from the project's own classifier on real tracks**, from batch-matched
   cohorts. They never come from authors' labels or absolute levels across labs (D34).
4. **Positive controls hold out whole cohorts**, not animals (D37).
5. **No fit before the benchmark reads valid**, and no fit on real data before its method clears
   parameter recovery (F.0).
6. **The fitting budget is pre-registered** (weights, population, generations, x₀, σ₀, animals per
   evaluation, sep- or full CMA), with a plateau criterion every arm must meet. A wiring reading is
   never a fitting-speed contrast (D38, D40).
7. **Mutant encodings are registered per controller before any fit, and are never refitted** (D37).
   Mutant predictions carry posterior-predictive intervals, with an `unidentified` outcome.
8. **The wiring reading's minimum is the held-out distance between two real wild-type cohorts**,
   registered in both directions. It is read on graded distance whatever any pass gate says (D40).
9. **Every vendored dataset carries its licence**; only CC0 and CC BY are vendored. A dataset whose
   licence cannot be confirmed is not vendored, and the change says what stands in for it.

### Decision C: External precedent is carried as convergent, with what was verified stated

The nearest neighbours are BAAIWorm, Roberts 2016, WormWars and Chen, Leifer & Pillow 2025, each cited
with what it did and did not do. NeuroSparkSNT (Research Square rs-10240803) is unverified and is
checked at B's registration.

## Seed-band table

Phase 10 uses seeds for simulated animals and fit restarts, not for training panels. The bands reserved
here are cited by each registration.

| Registration | Band | Notes |
|---|---|---|
| E + T parity and cost pilot | 5001–5032 | the per-step runner against the batched evaluator, same seeds |
| B the benchmark's simulated negatives and controller grading | 5101–5356 | 256 simulated animals per graded assay |
| F.0 parameter recovery | 5401–5464 | known-parameter synthetic data and refit restarts |
| C.1 posterior-predictive checks | 5501–5756 | |
| C.2 fit restarts and grading | 5801–5864 | fit seeds; grading animals drawn from 5101–5356 |
| C.3 reward-trained comparator | 5901–5916 | 16 training seeds |
| W rewiring draws and fits | 6001–6256 | one fresh draw per null fit, 8–16 per arm, sized after the cost pilot |
| Smokes | 5991–5999 | execution checks only, discarded |

## Open Questions (resolved in per-milestone changes, not here)

- E.2: whether the point worm senses the lawn edge mechanically.
- E.3: the exact protocol matched to each reference, including time on food before removal, and the
  plate geometry.
- B: the composite pass rule per assay, and the weights in the benchmark's distance.
- N: the null design for receptor placement, and how a peptide's diffusion range (Ripoll-Sánchez's
  three models) is chosen.
- C.2: the encoder's parameterisation; the slow variable and its feasibility check.
- W: the number of draws per arm, set after the cost pilot and F.0.

## Risks

The roadmap's risk table is authoritative. The tracker adds one: **scope creep into the body**. Any
body or posture work is Phase 11, by D41.
