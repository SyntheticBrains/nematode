# I.1b — the annealed schedule fails the positive control

**Run 2026-09-10. Verdict: `fail`, control valid. The clone assay did not run.**

## The result

| arm | mean | gap closed | seeds above floor | alignment | verdict |
|---|---|---|---|---|---|
| analytic reference | −0.1361 | 99.8% | 8/8 | +0.4536 | passes — control valid |
| unmodulated floor | −0.8753 | — | 3/8 | −0.0104 | does not pass — control valid |
| constant σ = 0.2 | −0.1962 | **89.0%** | 8/8 | +0.2628 | **passes** |
| **annealed 0.2 → 0.02** | **−0.4795** | **38.1%** | 7/8 | +0.1756 decay / +0.0510 floor | **does not pass** |
| constant σ = 0.05 | −0.4773 | 38.4% | 7/8 | +0.1184 | does not pass |
| constant σ = 0.01 | −0.7769 | — | 4/8 | +0.0274 | does not pass |

Floor −0.6909, optimum −0.1353, pass threshold −0.4131. The control is valid: the analytic
reference passes and the unmodulated floor does not.

**The annealed arm lands on the constant σ = 0.05 arm** — 38.1% of the gap against 38.4% — not on
the σ = 0.2 arm it starts from.

## Why: it annealed *through* the learning phase, not after it

A geometric decay spends most of its length near its floor. With σ₀ = 0.2, σ_final = 0.02 and
E = 10,000:

| trial | 0 | 2,000 | 5,000 | 6,021 | 10,000 | 20,000 |
|---|---|---|---|---|---|---|
| σ | 0.200 | 0.126 | 0.063 | **0.050** | 0.020 | 0.020 |

σ falls below 0.05 at trial **6,021**, so **70% of the budget runs at a scale the grid had already
shown cannot clear the bar**. The arm did not explore at the σ that learns and then settle; it
spent under a third of its run anywhere near it.

The alignment split says the same thing from the estimator's side, and it is the reason that split
was registered. Decay-phase alignment is **+0.1756** — above the σ = 0.05 arm's +0.1184, below the
σ = 0.2 arm's +0.2628 — so the estimator *was* aimed while the scale was still large. Floor-phase
alignment is **+0.0510**: nearly silent, exactly as predicted for a small scale, which is why a low
number there is not by itself a failure signature. The registered signature is a decay-phase
alignment that does not rise **together with** a floor-phase score below the bar. Here the
alignment did rise and the score still fell short, which locates the failure in how the budget was
spent rather than in the estimator being broken by the schedule.

## What the seeds say

| seed | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| score | −0.489 | −0.298 | −0.338 | −0.535 | −0.387 | **−1.024** | −0.414 | −0.351 |

Seed 6 sits far below the floor and decides the mean clause: the median is **−0.4004** and the
mean without that seed **−0.4017**, both marginally *above* the −0.4131 threshold. Reported because
it is true, not as a reason to discount the verdict — the registered rule is a mean over eight
seeds and it is not met, the seed clause (7/8 above floor) is met, and no seed is dropped. Even
read at its most favourable the arm closes 38% of a gap the constant σ = 0.2 arm closes 89% of.

## What this establishes

- **The registered schedule fails, and the sequence stops.** The clone assay did not run. Under the
  registered response this is reported as a property of *this* schedule rather than resolved by
  re-tuning its bounds or its length.
- **It is not a refutation of annealing.** The design named this outcome in advance and how to read
  it: a constraint on `E` relative to the task's horizon. The evidence supports that reading —
  alignment rose while the scale was large, and the arm's score is what a run spending 70% of its
  budget below the learning scale should produce.
- **The tension I.1 exposed is unresolved.** A constant σ = 0.2 learns and destroys; a schedule
  that reaches a runnable scale within this budget does not learn. Nothing here says both are
  achievable at once.

## What it does not establish

- Not that no schedule works. A later `E`, a floor between 0.05 and 0.2, or a shape that holds σ₀
  before decaying are all untested, and the design's stated successor — compensating the rate by
  `1/σ²`, the textbook unbiased estimator — is untouched by this result.
- Not anything about retention. The clone assay did not run, so this arm has never met a competent
  policy.
- Not a re-read of I.1. The constant-σ arm's pass reproduces here exactly (−0.1962, 8/8, +0.2628),
  so the instrument is unchanged and the two results sit side by side.
