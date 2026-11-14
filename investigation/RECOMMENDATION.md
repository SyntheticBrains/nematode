# Recommendation: a slimmed hybrid, gated on what the wiring has already shown

Written 2026-11-14, on SPIKE_REPORT.md, COMPARISON.md and Part A. Revised the same day after the
independent check (INDEPENDENT_CHECK.md) and the reading it proposed (H0, `spike/H0_RESULT.md`),
with the maintainer's agreement. The first version is in git history at 54d96b73.

**Keep Phase 10's behaviour fitting, and add a small neural track beside it.** The track starts
with a standing neural readout, floored by a degree-preserving null. It runs a neural fit only if
the fit is tested against causal data, not just against covariation. Each step has a kill criterion.

## What H0 found

The independent check argued that the neural readout's signal might be degree statistics, as two
earlier connectome effects in this project turned out to be (Logbooks 034 and 103). If so, the
neural track would have had nothing to offer. H0 tested this before anything else, under a
registration that mapped each outcome to a recommendation.

- **The test:** the unfitted encoder on the true wiring against 19 degree-preserving rewirings, on
  the measured responses to single-neuron stimulation.
- **The true wiring:** r = 0.0653, first of 20.
- **The rewirings:** median 0.0050, maximum 0.0219. The margin over their median is about 5.5 times
  the registered minimum.
- **Training matrix and C.2's fitted answer:** both give the same ordering.

So the readout's signal is the specific wiring, not the degree sequence. Under H0's registered map,
that is a PASS: the slimmed hybrid. Two further points:

- **Behaviour fitting does not detectably add to it.** C.2's fitted answer scores 0.0534 on the true wiring,
  against 0.0653 unfitted. The intervals overlap, but they treat the matrix's entries as
  independent and both networks are scored on the same matrix, and no paired test was run. So
  there is no detectable gain, which is not the same as evidence of none.
- **The absolute level is modest.** The true wiring reaches 38% of the data's own ceiling of 0.173,
  where Creamer's fitted model reaches 82%.

## Why not the others

- **Not a switch.**
  - Fit-first's direct reach is M-a and M-b, and M-a is blocked on licensed data today: the CC-BY
    atlas names the target of only half its stimulations.
  - A fair control panel at that scale costs about as much machine time as W.
  - Neither published connectome-constrained fit can be reproduced as specified on licensed
    materials. One has a wiring bug, and neither had been tested against a degree-preserving null.
  - M-c and M-d sit close to the machinery Phase 10 already has.
- **Not stay.**
  - H0 shows the connectome's specific wiring predicts measured neural responses, with no fitting.
  - Behaviour fitting shows no detectable gain on it, and is not graded on it.
  - A readout that costs minutes per model, and grades the connectome's contribution on neural data
    the controllers never saw, answers part of Phase 10's own question: what the connectome
    contributes to fidelity.

## The first three milestones

**H1. A standing neural readout, floored by a degree-preserving null (about 2 sessions; minutes of
machine time per model).**

- **What it scores:** a connectome model's in-silico stimulation responses, against Creamer's
  measured arrays (MIT-licensed). Each model is read against two floors: the same model on 19
  degree-preserving rewirings, and its own unfitted wiring.
- **Where it is applied:**
  - C.2's registered answer when it exists;
  - C.1 where it is defined;
  - each arm of W.
- **Its status:** description, beside the behaviour grade, never an objective. That keeps the
  readout held out.
- **Kill:** if the readout cannot tell any two models apart beyond the spread of its own null,
  across the next three models it reads, it is retired. That spread is measured from the 19 draws.

**H2. A connectome-constrained neural fit, with a causal bar (about 5 to 8 sessions, hard stop at 8;
about 40 to 50 machine-hours, about 15 h of wall-clock on 8 processes).**

- **What:** Mi et al.'s latent-voltage model, rewritten from the paper in PyTorch (their code has no
  licence). It is fitted to their training traces, which are MIT-licensed on Zenodo 3906530.
- **The arms:**
  - the correctly parsed connectome;
  - Mi's mis-parsed one;
  - 8 degree-preserving nulls;
  - a dense model.
- **The compute, as an assumption:** 11 arms with one held-out-neuron run per arm for each of 3
  held-out folds, so 33 runs. Session 0b measured one run at 1.2 to 1.5 h, so this is about 40 to 50
  machine-hours, or about 15 h of wall-clock with 8 runs at a time (3.5 h per run when 8 share the
  machine).
- **Two bars, both registered in advance with a synthetic-data oracle and a fresh review:**
  - **imputation:** withheld neurons, beating the nulls by the registered minimum;
  - **causal:** the fitted model's in-silico stimulation responses must beat the same responses
    from the null-wiring fits on H1's measured arrays. This is the independent check's point:
    imputation alone shows that partners covary, not that the coupling is causal.
- **Kill:** if the causal bar fails, the neural-fitting track stops, whatever imputation shows, and
  only H1 remains.

**H3. Head coupling into the closed-loop model (gated on H2's causal bar and on licensed data;
rented compute).**

- **Precondition:** H2's causal bar passes, and complete licensed atlas metadata exist.
- **What:** fit the head's effective coupling to the perturbation atlas with its nulls on a rented
  float64 GPU, after a one-hour cost pilot with a budget cap set beforehand. Then test whether
  importing it into C.2 lifts H1's readout above both floors without failing a behaviour family
  C.2 passes.
- **Kill, any one of:**
  - no licensed complete data within three months of asking;
  - the pilot projects the panel above its cap;
  - the import does not lift the readout;
  - the import breaks a behaviour family.

**For the whole track:** if it delays a Phase 10 step by more than two weeks, it pauses until that
step is through. Each milestone's sessions are logged in an effort ledger, as the spike's were, so
the diversion of attention is visible before it shows up as a slip.

## What happens to the current phase's work in progress

- **C.2's sequence continues.** The stopped third readiness pilot needs the maintainer's decision.
  H1 scores its registered answer when it exists.
- **W is not registered at its registered size.** At about 1,200 to 2,900 machine-hours it is beyond
  what the machine can run as planned, as the tracker itself says. Choose one of the tracker's cuts.
  Add H1's readout to every arm, which costs minutes. W then asks its wiring question of a
  behaviour grader and a neural grader at once. H0 has already shown that the neural grader
  separates the true wiring from degree-preserving nulls.
- **C.1's refit and C.3 are unchanged.**
- **The atlas note.** Before H0, I suggested sending it only if H0 passed, because H3's route depends
  on complete licensed data. H0 passed, so that condition is now met. The maintainer has deferred the
  note, and it can go whenever they choose. The note to Mi's authors stays a courtesy that can go any
  time.

## What would change this recommendation

- **H1 fails to separate models across its first three readings.** The recommendation becomes stay,
  with H0 recorded as a property of the wiring, not a tool.
- **H2's causal bar fails.** Stay, plus H1.
- **H2 passes, and H3's import lifts the readout without behavioural cost.** A larger neural track
  becomes worth planning as its own phase, and PLACEMENT.md's extraction question reopens.
