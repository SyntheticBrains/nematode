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
