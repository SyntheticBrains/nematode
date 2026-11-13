# Decisions log

Every non-trivial decision taken during the fit-first investigation, in the order taken. Each entry
gives the decision, the reason, and what was considered instead.

## Part A

### A1. Work in a separate git worktree (2026-11-14)

**Decision.** The branch `investigation/fit-first` lives in its own worktree at
`.claude/worktrees/fit-first-investigation`, not in the main checkout.

**Reason.** When the investigation started, a registered C.2 fit pilot (`c2_fit.py pilot --workers 16`)
was running from the main checkout on `feat/fit-c2`. The campaign runner reads configs and package
code from the working tree, so switching branches there would have broken it, as it once did to 7 of
64 runs.

**Considered instead.** Waiting for the pilot to end (about 28 hours), or switching branches anyway.

### A2. Branch from `main`, not from `feat/fit-c2` (2026-11-14)

**Decision.** The investigation branch starts at `main` (`2c80cfcd`). The five commits on
`feat/fit-c2` (the C.2 fit procedure and launch record 133) are read from the main checkout, not
merged into this branch.

**Reason.** The brief says not to change the current phase's work. Branching from `main` keeps the
investigation independent of unmerged work and lets either branch merge without the other.

**Considered instead.** Branching from `feat/fit-c2`, which would have made this branch carry the
current phase's unreviewed commits.

### A3. No compute in Part A (2026-11-14)

**Decision.** Part A runs nothing heavier than file reads, `git log` and web lookups.

**Reason.** Part A is evidence only, by the brief. The machine's cores were also fully booked by the
16-worker pilot, and the project's core budget allows at most 16 workers in total.

### A4. Parallel research agents, with access levels recorded (2026-11-14)

**Decision.** Five research agents ran in parallel: worm neural-data fitting; analogues,
identifiability and tooling; neuromodulation, plasticity and the calcium link; the dataset inventory;
and a read-only audit of the codebase. Each was told to cite only what it opened and to record, per
source, whether it read the full text, the abstract, or a second-hand report. Their raw reports are
working notes; the investigation documents are written from them, and load-bearing claims are
re-checked against the primary source before they are relied on.

**Reason.** The literature and dataset sweeps are broad and independent. Running them in parallel
keeps each step reviewable in one sitting.

**Considered instead.** One sequential sweep, which would have taken several times longer with no
gain in independence.

### A5. The prior direction review is cited, not re-litigated (2026-11-14)

**Decision.** The October 2026 direction review (`build/direction-review-2026-10.md`, gitignored)
already weighed a version of this hypothesis as its option C ("data-constrained neural dynamics,
embodied": Creamer, Leifer and Pillow's fitted whole-brain model in the loop) and deferred it to
"B's stage 3". ORIENTATION.md records that, and the investigation treats fit-first as a broader
question than option C: fitting to neural data as the primary objective, not one fitted model
dropped into the loop.

**Reason.** Candour about the current approach includes saying that this was considered before and
why it was set aside. The brief's hypothesis differs from option C in kind, so the earlier deferral
does not settle it.

### A6. The brief stays byte-for-byte verbatim (2026-11-14)

**Decision.** `BRIEF.md` is kept exactly as pasted, even though the repository's `mdformat` hook would
insert a blank line before each of its nested lists (whitespace only; no word changes).

**Reason.** The brief asks for a verbatim copy. If this branch ever goes to a pull request, CI's Code
Quality job runs `mdformat` on every file, so at that point either the blank lines are accepted or
`investigation/BRIEF.md` is excluded from the hook. That choice is the maintainer's and is deferred
until then. The other investigation documents are written to pass the hook.

### A7. Accept the literature sweeps' coverage gap, and check the load-bearing claims (2026-11-14)

**Decision.** All three literature agents ran out of web-search budget near the end of their sweeps,
so work from late 2025 to November 2026 is under-covered. LANDSCAPE.md says so at the top and lists
what was looked for and not found. Rather than re-running the sweeps, I re-opened the sources that
carry the most weight and confirmed the quoted claims (Creamer v4, Randi 2023, r-elegans, flyvis,
Beiran and Litwin-Kumar, Li et al. 2026, Mehta et al. 2023, Ding et al. 2023). Those are marked
"checked" in the source table.

**Reason.** A second sweep would cost as much as the first for a marginal gain, and the user's seed
list (RECONCILE.md, next) is the brief's own mechanism for catching what the sweep missed.

**Considered instead.** A fourth agent restricted to 2025 and 2026 work. Worth doing if the seed list
shows the gap matters.

### A8. Commit the agents' raw notes beside the curated documents (2026-11-14)

**Decision.** The six research reports and two sub-audits are committed under `investigation/notes/`,
with em dashes replaced and bare URLs wrapped so they pass the Markdown hooks, and a README saying they
are unreviewed agent output superseded by the curated documents.

**Reason.** The curated documents compress about 6,000 lines of notes into a reviewable size. Keeping
the notes lets every claim be traced to the URL opened and the access level reached, which the brief's
citation rule needs, and matches the project's habit of committing every reading.

**Considered instead.** Leaving them in the session scratchpad, which is deleted with the session.

### A9. Correct the project's licence finding on the Randi atlas (2026-11-14)

**Decision.** ORIENTATION.md and DATA_INVENTORY.md record that the Randi 2023 atlas is available under
CC-BY-4.0 on DANDI 001075, against `data/connectome/PROVENANCE.md`, which judged it unusable. The
project file itself is not changed on this branch.

**Reason.** I confirmed the licence on DANDI's API (`spdx:CC-BY-4.0`, version 0.240930.1859). The
correction removes what looked like a hard blocker for fit-first inside this repository. Editing
`data/` is outside this investigation's remit, so the fix to PROVENANCE.md is left to the maintainer.

### A10. The spike reproduces Creamer, Leifer and Pillow (2026-11-14)

**Decision.** SPIKE_PLAN.md proposes reproducing Creamer, Leifer and Pillow's connectome-constrained
linear dynamical system on the Randi atlas, refitted by gradient descent.

**Reason.** It is the only published fit of a connectome-constrained model to worm perturbation data,
its numbers are stated in the text, its code is MIT, and its data are CC-BY-4.0 on DANDI.

**Considered instead.** Mi et al.'s latent-voltage model (wormvae): gradient-based already, but its
paper could not be opened and its data and numbers are unknown. Morrison and Young's premotor model:
no held-out split to reproduce. BAAIWorm: no held-out data and an unpublished gradient method.

### A11. PyTorch on CPU with a steady-state Kalman filter (2026-11-14)

**Decision.** The spike fits by gradient on the exact marginal likelihood in PyTorch, float64 on CPU,
using a steady-state filter per animal.

**Reason.** PyTorch is already a dependency and the model is small. Apple's GPU offers no float64 in
PyTorch, and jax-metal is abandoned. With each animal's recorded set fixed, the steady-state filter
cuts a gradient step from minutes to seconds; the published EM fit takes about 10 hours on 10 CPUs.

**Considered instead.** JAX (a new dependency for no gain at this size); the exact time-varying filter
throughout (used only where a recording breaks mid-session).

### A12. The bridge's interface is fixed in advance, from neural data only (2026-11-14)

**Decision.** The bridging test adds a sensory input map and a locomotion readout, both fixed before
any run and neither fitted to the behaviour benchmark: inputs through the model's own fitted filters,
scaled to one optogenetic unit; the readout either fitted to freely moving recordings (option A) or
fixed from literature signs (option B), chosen at approval.

**Reason.** The fitted model has optogenetic inputs and calcium outputs, and no motor neurons, so
"unchanged" can only mean its fitted parameters are frozen. An interface fitted to the benchmark
would break the brief's rule that behaviour is held out.

## Reconciliation

### A13. Mi et al. upgraded, and made the spike's alternative target (2026-11-14)

**Decision.** Mi et al. 2022 was read in full from the Janelia-hosted PDF (text extracted locally with
`pypdf`) after the user's seed list named it. LANDSCAPE.md and IDENTIFIABILITY.md now carry its
numbers, and SPIKE_PLAN.md names it, on Yemini 2021 (DANDI 000541, CC-BY-4.0), as the target if
neither data route to the Randi atlas opens. Its bars are r ≥ 0.25 on held-out neurons (about four
seed standard errors below the published 0.319) and the dense variant below it.

**Reason.** It is the one gradient-fitted connectome-constrained worm model with a held-out test, and
its held-out-worm result (no connectome benefit on whole animals) qualifies what Part A said. Switching
target keeps the spike inside the project's licence rule, where the unlicensed OSF fallback would not.

**Considered instead.** Making Mi the primary target. Not done: it has no optogenetic perturbation, and
the brief puts perturbation data first.

## Review of Part A

### A14. Three independent reviewers, corrections held for the maintainer (2026-11-14)

**Decision.** At the maintainer's request, three reviewers read Part A, each with no context from
this investigation and each with one scope: project facts, literature and data, and the spike's
design and fairness. Their findings, my re-checks of the load-bearing ones, and what I propose are in
REVIEW_PART_A.md. The Part A documents are left as reviewed until the maintainer decides how the
corrections are applied, because the maintainer is reading them at the same time.

**Why.** Separate scopes keep each reviewer's report checkable in one sitting. Starting them with no
context keeps their reading independent of mine. I re-checked four findings myself before recording
them: the W sizing against the tracker and launch 133, the vendored Creamer weights, the spike
plan's session count, and Mi et al.'s Table 2.

**Considered instead.** One reviewer for everything. Not done: the literature check alone took about
130 tool calls, and a single report would have mixed fact errors with design judgements.

## The maintainer's decisions after the review (2026-11-14)

Each was put to the maintainer with options and a recommendation; each took the recommendation.

### A15. Every correction applied at once

All of REVIEW_PART_A.md's findings are applied to the Part A documents in one commit, so the
maintainer can read one diff after their review. Considered instead: waiting until their read was
done, or applying only the load-bearing findings.

### A16. The spike plan revised and re-presented before Part B

SPIKE_PLAN.md is rewritten with the design review's fixes, and Part B starts only when the
maintainer approves the revised plan. Considered instead: approving now and fixing in session 0, or
holding Part B.

### A17. A hard stop at 10 sessions

The fixes and the three readings Part B needs (M-a for C.2, M-b, M-d) are kept. Nine sessions are
planned, so one is in reserve. Considered instead: keeping 8 by cutting scope, or keeping 8 with the
fixes only.

### A18. Compute locally now, yielding to C.2

The third readiness pilot was stopped on 2026-11-14, so the machine is idle. The spike runs in the
gap, checkpoints, and exits whenever a C.2 run starts, within 16 workers in total. Considered
instead: waiting for the C.2 sequence, renting CPU, or renting a GPU.

### A19. Creamer's pickles opened in isolation

They are listed with `pickletools`, loaded in an empty directory with `python -I` through an
allow-list unpickler, and exported to a checksummed `.npz` in the spike's folder. Nothing is added to
`data/`. Considered instead: an oracle generated from a model the spike fits itself, which would
inherit any error in the fitter.

### A20. Mi on Yemini as the only fallback, and no OSF data

If DANDI cannot supply Randi's traces, the target becomes Mi et al. on Yemini 2021 under its own
committed amendment. The unlicensed OSF deposits are not used, even locally. Considered instead: OSF
data locally, or stopping. This also supersedes A13's wording that the switch keeps the spike
"inside the project's licence rule": the project record's reasons for the Randi exclusion were the
missing licence and the GPL package (REVIEW_PART_A.md P3).

### A21. Bridge readout B, with an interface-only row

Literature signs and unit gains, plus the coupling-free model bridged identically, so a pass cannot
be the readout's doing alone. The time saved goes to scoring C.2 on Randi's held-out responses. This
reverses my earlier recommendation of option A. Option A cost two to three sessions on a bridge
expected to fail, and AWC and AIB are missing from the released weights. Considered instead: option
A, or both.

## The revised plan's own design choices

### A22. How the revision answers the design review (2026-11-14)

**Decisions.**

- **Hyperparameters.** Every arm uses Creamer's own ridge and pruning, and only the learning rate is
  chosen, on synthetic data. This replaces per-arm selection on validation, which would have cost
  about 30 extra fits. It also matches the paper's own comparisons, since his dense and relabelled
  models used his recipe.
- **R1.** R1 becomes non-inferiority to Creamer's released model scored through the spike's own
  pipeline, with the tolerance from a bootstrap over test animals. The v3-to-v4 shift is no longer
  the tolerance.
- **The minimum effect.** For the wiring bars it is a quarter of what coupling adds over none. It is
  scale-free, and it makes the wiring question unreadable, by a named branch, when coupling itself
  is not detectable.
- **No null on the held-out-perturbation folds.** Those would be 24 more fits, so the limit is stated
  instead.
- **Seeds.** The spike's seeds are listed in the plan, outside every band on both branches. The
  tracker's seed table is current-phase work, so the spike does not edit it.

**Why.** Each choice keeps a fix the reviewer asked for within the 10-session stop.

**Considered instead.**

- Per-arm selection.
- A fixed 0.10 tolerance.
- A minimum effect anchored to the paper's analysis revisions.

### A23. The re-review's fixes, and two choices of mine (2026-11-14)

**Decision.** Every fix the design reviewer named on re-reading the revised plan (N1 to N13,
REVIEW_PART_A.md § 6) is applied before the plan goes to the maintainer. Two choices are mine:

- **The degree-preserving null** rewires the model's own mask. Reciprocal entries are swapped among
  themselves and the others by directed swaps, and any swap that creates a duplicate, a
  self-connection or a new reciprocal pair is rejected. This preserves exactly the number of free
  weights, every in- and out-degree, and the number of reciprocal pairs, and it always terminates.
- **R1a in the unrecovered branch** compares against a spike fit initialised at Creamer's
  parameters on the same 80 animals.

**Why.**

- **The null.** The model gives every mask entry its own free directed weight, whatever the synapse
  type, so the mask is the structure it sees. The reviewer's options were to rewire the typed edge
  classes and keep their overlap, or to equalise counts after rewiring. The first is harder to make
  terminate, and the second perturbs degrees.
- **R1a.** His parameters were fitted to a different 80 animals, so on the spike's 80 they are not
  an optimum, and beating them would not show that the fitter works.

**Considered instead.**

- `rewire_degree_preserving` on a typed `Connectome`.
- Accepting draws whose count lies within 1% of the wild type's.
- R1a against his parameters in both branches.
