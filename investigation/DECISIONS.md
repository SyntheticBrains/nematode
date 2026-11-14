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

## After the fresh review (2026-11-14)

### A24. Session 0 runs as fact-finding only, before the plan is registered

The maintainer's decision, on the fresh reviewer's recommendation (REVIEW_PART_A.md § 7, M15).
Session 0 runs no fits and reads no test animals. It establishes the facts that the plan's rules
depend on:

- Creamer's ridge, pruning, missing-value handling, split, neuron set, preprocessing stage and
  input encoding;
- the pickles' contents, opened under A19's isolation;
- what the 110 DANDI segmentation files hold, and how they compare with his example recording;
- the missing values per neuron and frame;
- the measured cost of one exact-filter likelihood pass;
- DANDI 000776's layout.

One revision then applies every fix (X1 to X3, M1 to M15 and the minor findings) together with
those facts, and goes to the maintainer for registration. Considered instead: revising the whole
plan first, or stopping the spike.

### A25. Two departures from the brief's stop rule, approved

The brief says "if reproduction fails within the timebox, record why and stop". The maintainer
approved two departures, each to be named in the report:

- if Creamer's split is not recovered and the held-out reproduction fails while the fitter itself is
  verified, the spike continues;
- C.2 is scored on Randi's held-out responses even if reproduction fails, since that needs only the
  verified pipeline and scorer.

Considered instead: allowing only the second, or neither.

### A26. Phase 10 is not paused yet; W waits for Part B

No Phase 10 work is paused now, because session 0 needs no machine time. If session 0 shows the
spike's fits cost 10 to 50 times the estimate, I bring the measured cost and a float64 GPU quote,
and the maintainer chooses between renting and pausing C.2. W is not registered until Part B
recommends switch, hybrid or stay. Considered instead: pausing Phase 10 now, or keeping it going
fully with no hold on W.

## After session 0 (2026-11-14)

### A27. The Creamer target stops; the spike switches to Mi on Yemini

Session 0 (SESSION_0.md) found that an exact reproduction of Creamer et al. cannot run on licensed
data within the spike's budget:

- the CC-BY DANDI deposit names the stimulated neuron for only 2,815 of 5,614 stimulations, which I
  re-checked independently;
- its labels do not reproduce his 154 neurons;
- missing values everywhere make the exact filter necessary, at 44 to 154 h per fit here, or about
  1,000 to 2,500 machine-hours for the planned panel.

The maintainer chose A20's fallback: Mi et al. 2022 on Yemini 2021 (DANDI 000541), held-out neurons
only, with a degree-preserving null that Mi did not run. It starts with fact-finding (session 0b).
The Creamer route is recorded as stopped, with its reasons, and those reasons are themselves
evidence for Part B. Considered instead:

- stopping and going to Part B;
- Creamer on DANDI's identified half with a rented GPU;
- Creamer on the unlicensed OSF arrays.

### A28. C.2 is scored against Creamer's measured responses now, as its own step

The reading is described in `investigation/spike/MA_C2_READING.md`, whose rules were committed before
it was computed. It uses the measured STAM arrays in Creamer's MIT repository and needs no
reproduction (A25). Considered instead: keeping it inside the spike, or dropping it.

### A29. A note to the data's authors, drafted for the maintainer to send

The note asks whether the missing target identities, or a licence for the OSF exports, could be
provided. The maintainer decides whether to send it. Considered instead: leaving the data as they are.

## The spike's end (2026-11-14)

### A30. The spike stops at fact-finding; Part B follows

Session 0b found that Mi et al.'s code has no licence, parses polyadic chemical synapses in reverse,
and records no scoring protocol, so neither target can be reproduced as specified on licensed
materials. The maintainer chose to stop under the brief's stop rule and write SPIKE_REPORT.md, then
Part B. A re-implementation of Mi, with correct wiring, Mi's wiring and degree-preserving nulls,
becomes a candidate milestone for Part B to weigh. Considered instead: that re-implementation as a
new study now, or pausing the investigation.

### A31. A note to Mi's authors, drafted for the maintainer to send

It reports the parsing bug, with the line and the counts, and asks for a licence. The maintainer
decides whether to send it. Considered instead: recording the bug in our documents only.

## After the independent check (2026-11-14)

### A32. H0 run, and the recommendation finalised from it

The maintainer adopted the independent check's proposed first test (H0) and the revision mapped to
its outcomes. H0 passed: the unfitted encoder on the true wiring scores 0.0653 against the measured
responses, against a median of 0.0050 for 19 degree-preserving rewirings. RECOMMENDATION.md is
revised accordingly:

- H1 is floored by the degree-preserving null;
- H2 carries a causal bar beside imputation;
- H3 is gated on that bar.

The first version is kept in git history at 54d96b73. Considered instead: keeping the recommendation
as written, or staying without running H0.

### A33. The merge, and the handover

The merge contains everything except the raw agent notes, which stay on a tagged branch:

- the conclusions and the Part A documents;
- the spike's records;
- a logbook entry;
- the provenance correction;
- the seed-table entries.

A handover brief for the Phase 10 session is added as a file and also sent to that session. Both
notes to authors are deferred. Considered instead: merging everything, or the conclusions only;
the handover as a file only.
