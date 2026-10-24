# 104: Phase 10 — The Batched Evaluator: Many Worms per Call, at the Per-Step Runner's Behaviour

**Status**: completed. **T.1 ships.** Frozen controllers now run on hundreds of simulated animals in lockstep, 48–96 times faster per core than the per-step runner. Their behaviour matches that runner's, layer by layer, and the target of 25 µs per animal-step at 256 animals is met by every controller on every assay.

> At 256 animals on one core:
>
> - the MLP runs at 2.7–3.8 µs per animal-step (76–96×);
> - the connectome runs at 6.1–7.3 µs (48–51×);
> - the hazard controllers run at 1.6–2.8 µs (62–87×).
>
> One 200-animal evaluation over all four assays takes 12.5 s on the MLP and 25 s on the connectome. At an illustrative fit of 20 × 200, W's connectome panel costs about 1,150 core-hours at 8 draws per arm, and 2,270 at 16.

**Date**: 2026-10-24.

**OpenSpec change**: `add-batched-evaluator`.

**Records**: [`supporting/104-batched-evaluator/cost_pilot.json`](supporting/104-batched-evaluator/cost_pilot.json) (commit `93161183`).

## Objective

Phase 10's fits and its wiring reading evaluate frozen controllers on many animals:

- C.2 fits the connectome by sep-CMA on at least 200 animals per evaluation;
- W repeats that fit for every null arm and rewiring draw;
- B grades every controller and every negative on 256 animals per assay.

The per-step runner, one animal at a time, made that out of reach. T.1 was to supply:

- a batched evaluator;
- a parity test against the per-step runner;
- a cost pilot on seeds 5001–5032 that sizes W.

These were the maintainer's decisions of 2026-10-24:

- frozen evaluation only;
- layered parity;
- sampled noise from per-animal streams;
- a target of at least 20× per core (25 µs per animal-step at 256 animals).

## What was built

`quantumnematode/evaluation/`:

- **`BatchedEvaluator`** holds every per-animal state as arrays and runs a step for all animals in the per-step runner's order. The state covers position, heading, primitive bout, lawn density, satiety, the short-term memory buffer, the adaptive baseline and the connectome's modulators. The step order is:
  1. assay phase;
  2. sensing;
  3. controller;
  4. primitive move;
  5. intake and satiety's restoration;
  6. satiety's decay.
- **Per-animal randomness.** Each animal keeps the environment the per-step runner would build for its seed, used only for its random draws and its phase starts. Its policy noise comes from a torch generator seeded the same, five draws a step. An animal therefore behaves the same alone or in any batch.
- **Controllers:**
  - `BatchedMLP`;
  - `BatchedConnectome`, with a new batch-safe `modulator_release_batched`;
  - the two hazard rules.
- **Captures** come back as compact arrays, readable as the per-step runner's `BehaviourStep` series on request.
- **A declared surface**, refused at construction otherwise:
  - the 1-s primitives worm;
  - centred lawns or assay protocols;
  - klinotaxis sensing through the short-term memory and the fold-change sensor;
  - the Fick kernel;
  - a satiety that never ends the episode.

Two exact rewrites took the evaluator from about 20 µs to under 8 µs per animal-step on the connectome:

1. **The odour.** The Fick kernel is separable on the lawn's cell grid, so the odour at a point is two small matrix products over a dense density grid.
2. **The cell under each worm.** It is found by rounding to the grid. Only near the lawn's rim, where the nearest grid point can fall outside the disc, does it fall back to searching every cell.

## Parity

The parity tests compare each animal with one per-step episode on its seed. They covered four controllers on all four E.3 assays, on seeds 5001–5004:

| Layer | Requirement | Result |
|---|---|---|
| 1. State | features, action means, modulators and the fold-change derivative within 1e-5 over 50 steps | met on all 16 combinations |
| 2. Trajectory | bouts identical over 300 steps; positions and headings within 1e-3 over 150, satiety and lawn density within 1e-6 | met on all 16 |
| Batch size | a seed alone and inside a batch of 64 make the same bouts | met |
| 3. Statistics | full Scheer and Flavell assays on 5001–5032 (MLP and pirouette rule); each paired difference's 95% bootstrap interval contains zero | met for all five statistics in all four tests |

Layer 3's five statistics are on-lawn fraction, roaming fraction, bout rate, final satiety and total intake.

**Two deliberate faults were caught.** Leaving the derivative scale at its written 50 fails layer 1. Swapping the head-sweep sides fails layers 1 and 2.

**Why the tolerances are what they are.** Batched and single-animal forward passes differ in the last bits of float32: by 2 × 10⁻⁷ on the MLP and 1.5 × 10⁻⁶ on a 302 × 302 settling product. Around the closed sensory loop, such differences grow. On Scheer's lawn edge, where the odour gradient is steepest, the MLP's difference grew about tenfold every 40 steps: 10⁻⁷ at step 0, 10⁻⁵ by step 180 and 10⁻³ by step 228. Its bouts still all matched at step 300. Continuous values are therefore held over 150 steps, where the worst case measured was 1.6 × 10⁻⁵, and bouts over 300.

Over a full hour some pairs decorrelate, so layer 3 checks for bias rather than identity. An earlier draft also required each mean difference to stay below 0.02. That bound failed on the MLP's Scheer bout rate: a mean of 0.036 per minute against paired differences of about ±0.15. Thirty-two independent animals cannot hold a mean that tightly, so the bound was dropped.

## The cost pilot

These are per-core timings, with every thread pool pinned to one, on an Apple-silicon Mac (`cost_pilot.json` records the machine). The per-step runner was timed on seeds 5001–5008; the evaluator on 5001–5032, repeated to N.

| Controller | Per-step runner (µs) | Batched at N = 32 | at 128 | at 256 | Speedup at 256 | 200-animal evaluation |
|---|---|---|---|---|---|---|
| MLP | 247–293 | 6.1–8.2 | 3.2–4.5 | 2.7–3.8 | 76–96× | 1.9–4.1 s |
| Connectome | 304–347 | 11.5–13.5 | 7.0–8.2 | 6.1–7.3 | 48–51× | 4.4–7.8 s |
| Constant hazards | 131–175 | 4.1–6.5 | 2.0–3.4 | 1.6–2.8 | 62–85× | 1.1–3.1 s |
| Pirouette rule | 135–179 | 4.2–6.2 | 2.0–3.3 | 1.6–2.8 | 65–87× | 1.1–3.0 s |

Each range runs over the four assays. The fastest is Scheer's, with its 9-cell lawn; the slowest is Flavell's 90-minute run or López-Cruz's 1,963-cell fed lawn.

- **Default threads.** With the machine's default thread settings the N = 256 figures are unchanged, so the evaluator gains nothing from threads. Parallelism for the fits should come from processes.
- **The per-step runner** was faster unprofiled (250–350 µs) than the profiler suggested before the change (about 500 µs). The 20× target was set against the higher figure, and is met against the lower one.

## What this means for W

The record's budget formula is:

> one fit = population × generations × (sum over assays of a 200-animal evaluation)
>
> panel = (1 + null arms × draws per arm) × one fit

At an illustration of population 20, 200 generations and 5 null arms:

| Controller | All-assay evaluation | One fit | Panel, 8 draws | Panel, 16 draws |
|---|---|---|---|---|
| Connectome | 25.3 s | 28 core-hours | 1,150 core-hours | 2,270 core-hours |
| MLP | 12.5 s | 14 core-hours | 571 | 1,128 |

This machine has 18 logical cores, 6 of them performance cores. Spread across all 18, the connectome panel is about 3–5 days of wall time at those settings, and longer to the extent the efficiency cores run slower. C.2's registration sets the real population, generations and assay subset. F.0's recovery spread sets the draws per arm. This record supplies the per-evaluation cost both multiply.

## Conclusions

1. **T.1 ships.** Frozen controllers run 48–96 times faster per core, matched to the per-step runner on all four assays.
2. **W is affordable but not cheap.** At illustrative settings the connectome panel is 3–5 days of the machine, or more. C.2's and W's registrations should size their budgets from this record.
3. **The parity tests guard the evaluator against drift.** Layers 1 and 2 run in the default test tier on every PR. A later change to the per-step runner's sensing that the evaluator does not mirror fails them.

## Next steps

- [ ] B (the benchmark): grade real tracks, the negatives and the controllers with the evaluator, starting with B.1's reference re-extraction.
- [ ] N (the neuromodulator layer from data), in parallel with B.
- [ ] C.2 and W: register budgets from this record's per-evaluation costs.
