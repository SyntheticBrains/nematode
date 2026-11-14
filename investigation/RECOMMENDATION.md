# Recommendation: a hybrid, small and gated

Written 2026-11-14, on SPIKE_REPORT.md, COMPARISON.md and Part A.

**Keep Phase 10's behaviour fitting, and add a small neural track beside it.** The track starts with
a standing neural readout, which costs almost nothing. It then runs one gated neural fit on licensed
data, aimed at the question the field has not answered: does the connectome help predict neural
activity beyond what its degree statistics give? Each step has a kill criterion. If the answer is
no, the track shrinks back to the readout.

## Why not the others

- **Not a switch.**
  - Fit-first's direct reach is M-a and M-b, and M-a is blocked on licensed data today: the CC-BY
    atlas names only half its stimulation targets.
  - A fair control panel at that scale costs about as much machine time as W.
  - Neither published connectome-constrained fit can be reproduced as specified on licensed
    materials. One has a wiring bug, and neither has been tested against a degree-preserving null.
  - M-c and M-d sit far from fit-first and close to the machinery Phase 10 already has.
- **Not stay.**
  - The roadmap never measures neural fidelity.
  - The spike's one reading shows behaviour fitting does not supply it: C.2's fitted head scores
    0.053 against measured responses, its unfitted wiring 0.065, and chance tops out at 0.022.
  - If the long-run goal includes neural fidelity, neural data have to enter somewhere. The cheapest
    place is a readout that grades every connectome model the project already fits.

## The first three milestones

**H1. A standing neural readout (about 2 sessions; no machine time).**

- **What:** the M-a reading, packaged as a reusable readout. It scores any connectome model's
  in-silico stimulation responses against Creamer's measured response arrays, against the model's
  own unfitted wiring and a relabelling null.
- **Where it is applied:**
  - C.2's registered answer when it exists;
  - C.1 where it is defined;
  - each arm of W.
- **Its status:** description, beside the behaviour grade, never an objective.
- **Kill:** none needed. If it ever fails to separate any model from chance, it is retired.

**H2. A connectome-constrained neural fit on licensed data, with the nulls the field lacks (about 5
to 8 sessions, hard stop at 8; about 10 to 20 machine-hours).**

- **What:** Mi et al.'s latent-voltage model, rewritten from the paper in PyTorch (their code has no
  licence). It is fitted to their own training traces, which are MIT-licensed on Zenodo 3906530, and
  scored on held-out neurons.
- **The arms:**
  - the correctly parsed connectome;
  - Mi's mis-parsed one;
  - 8 degree-preserving nulls with reciprocity kept;
  - a dense model.
- **Before launch:** registered in advance with a synthetic-data oracle, after a fresh independent
  review (the spike's experience: plans need it).
- **Kill:** if the correct connectome does not beat its degree-preserving nulls on held-out neurons
  by the registered minimum, the neural-fitting track stops at this data size, and only H1 remains.
  The same applies if it does not beat the dense model.

**H3. Head coupling into the closed-loop model (gated on H2 and on licensed data; rented compute).**

- **Precondition:** H2 passes, and the atlas authors supply the missing target identities or license
  the OSF exports.
- **What:** fit the head's effective coupling to the perturbation atlas with its nulls on a rented
  float64 GPU, after a one-hour cost pilot with a budget cap set before the pilot. Then test whether
  importing that coupling into C.2, as fixed weights or as a prior, raises H1's readout above the
  unfitted floor without failing the behaviour families C.2 passes.
- **Kill, any one of:**
  - no licensed complete data within three months of asking;
  - the pilot projects the panel above its cap;
  - the import does not lift the readout above the floor;
  - the import breaks a behaviour family that C.2 passed.

**For the whole track:** if it delays a Phase 10 step by more than two weeks, it pauses until that
step is through.

## What happens to the current phase's work in progress

- **C.2's sequence continues.** The stopped third readiness pilot needs the maintainer's decision, as
  it did before this investigation. H1 scores its registered answer when it exists.
- **W is not registered at its registered size.** At about 1,200 to 2,900 machine-hours it is beyond
  what the machine can run as planned, as the tracker itself says. Choose one of the tracker's cuts:
  a stopping rule on draws, fewer draws, or a cheaper fit. Then add H1's readout to every arm, which
  costs minutes. W then asks its wiring question of both graders at once.
- **C.1's refit and C.3 are unchanged.**
- **Housekeeping, for the maintainer:**
  - correct the Randi entry in `data/connectome/PROVENANCE.md`: the atlas is CC-BY on DANDI 001075,
    though its metadata are incomplete;
  - decide whether to send the two drafted notes (`spike/NOTE_TO_AUTHORS_DRAFT.md`,
    `spike/NOTE_TO_MI_AUTHORS_DRAFT.md`);
  - decide BRIEF.md's formatting before any pull request of this branch.

## What would change this recommendation

- **The atlas authors supply complete licensed data, and a GPU pilot shows a controlled panel costs
  little.** H3 moves up, but the recommendation stays a hybrid, because M-c and M-d still sit with
  Phase 10's machinery.
- **H2 fails its null.** The recommendation becomes stay, plus H1.
- **H2 passes strongly, and H3's import lifts the readout without behavioural cost.** A larger
  neural track becomes worth planning as its own phase, and PLACEMENT.md's extraction question
  reopens.
