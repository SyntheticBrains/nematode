# Handover to the Phase 10 work

Written 2026-11-14 for the session continuing Phase 10 (C.2's registered fit, launch record 133, on
`feat/fit-c2`). It lists what the fit-first investigation found that bears on Phase 10, and what it
leaves for Phase 10 to decide. Nothing here changes a registered rule or verdict. The full record is
`investigation/`; the conclusion is `RECOMMENDATION.md`; the readings are Logbook 134.

## What changes for Phase 10

1. **W's sizing is about 1,200 to 2,900 machine-hours at its registered size.**

   - **The arithmetic:** the tracker counts fits two ways. Its sizing formula (1 + 5 null arms × 8 to
     16 draws) gives 41 to 81 fits; its later cost line (6 arms × 8 to 16 draws) gives 48 to 96. At
     about 30 h a fit, launch 133's measured rate, that is 1,230 to 2,430 h or 1,440 to 2,880 h. The
     tracker's 232 to 815 h assumed 4.8-h fits.
   - **The tracker already says** W at this size is "beyond what it can run as planned". Its three
     cuts are a stopping rule on draws, fewer draws, or a cheaper fit.
   - **The recommendation:** do not register W at its registered size. Pick a cut first.
   - **Phase 10's close:** 12 to 27 weeks from now with W at its registered size; 7 to 14 weeks if W
     is cut to a quarter of it (ORIENTATION.md § 5.4).

2. **A neural readout now exists. It separates the true wiring from its degree-preserving
   rewirings, though not C.2's fitted answer from its unfitted wiring** (Logbook 134). It scores a connectome
   model's in-silico single-neuron stimulation responses against Creamer et al.'s measured responses
   (MIT).

   - **The unfitted encoder:** at the box centre, on the true wiring, it scores r 0.065, against a
     median of 0.005 for 19 degree-preserving rewirings.
   - **C.2's Logbook 131 answer** scores 0.053, with no detectable gain over unfitted: the intervals
     overlap, and no paired test was run.
   - **It costs minutes per model** and needs no simulation.

3. **The investigation recommends three uses** (RECOMMENDATION.md, H1). All are description beside
   the behaviour grade, never an objective. Adopting them is Phase 10's decision:

   - on C.2's registered answer, once it exists (launch 133 is unaffected);
   - on C.1 where it is defined;
   - on every arm of W, so that W asks its wiring question of a behaviour grader and a neural
     grader at once. A fair floor for each arm is its own degree-preserving draws.

   If adopted, it belongs in the tracker and the roadmap as a task, with Logbook 134 as its basis.
   The code is spike-local (`investigation/spike/ma_c2/` and `h0/`, as `.py.txt`). It would need a
   proper module, and PLACEMENT.md suggests one behind an optional extra.

4. **The vendored Creamer weights are confirmed and characterised** (SESSION_0.md, B):

   - `data/connectome/creamer_lds_2026_model_weights.csv` equals his released connectome model to
     5 × 10⁻⁷;
   - its 2,011 off-diagonal entries are 8.53% of pairs, not the paper's 9.4%;
   - it covers 125 of his 154 neurons, because 29 have no NemaNode partner.

5. **The Randi atlas is CC-BY on DANDI 001075.** The provenance record now says so (this merge), and
   notes that the deposit names the target of only half its stimulations.

## What is left for the Phase 10 session

- **The stopped third readiness pilot** (launch 133) still needs the maintainer's decision. The
  investigation did not touch it.
- **The seed table.** This merge adds one row for the investigation's generators and reserved
  ranges, placed after "D36's speed-following reversals". `feat/fit-c2`'s own new row sits near the
  top of the table, so a merge should be clean; check it when `feat/fit-c2` next merges `main`.
- **The logbook index** has no rows for Logbooks 123 to 132; 134's is added after 122's. That was
  found while adding 134's, and filling them is left to Phase 10's record-keeping.
- **Two notes to outside authors are drafted and deferred.** One asks the atlas authors for the
  missing target identities or an OSF licence; the other reports a parsing bug in Mi et al.'s code
  (`investigation/spike/NOTE_TO_*_DRAFT.md`). The maintainer decides whether to send them. Neither
  affects Phase 10.

## Machine and environment

The investigation used no Phase 10 compute, and ran nothing while a C.2 run was active. Its worktree
(`.claude/worktrees/fit-first-investigation`) holds about 3.5 GB of gitignored downloads and a 1.1 GB
`.venv`. Both can be deleted after the merge.

## Addendum, 2026-11-16: the readout's ceiling (Logbook 135)

A registered spike, run after the merge, measured what the neural readout can reach. Nothing here
changes a registered rule or verdict; the tracker and roadmap entries stay with the Phase 10 session.

- **The readout's ceiling is about 0.42, not 0.17.** √(Cov(T, S)/Var(S)) is 0.423 (0.369 to 0.471).
  The 0.173 that Logbook 134 called the ceiling is r(training, test), a reference. On unseen
  stimulations, no model so far reaches more than 31% of the ceiling.
- **The reference to beat is the calibrated unfitted network.** On held-out stimulations, H0-cal
  (the box centre, calibrated with the row/column-mean and transpose floors on visible entries)
  scores 0.132. A linear model learned on the anatomical support scores 0.099 and is not above it.
- **A learned model cannot test anatomy on these targets.** The positive oracle showed that, at this
  reliability, the test cannot tell a true anatomical model from rewirings. The unfitted network
  against H0's degree-preserving null (Logbook 134) remains the readout's valid floor for W's arms.
- **C.2's behaviour fitting has a measured neural cost.** Paired, C.2's Logbook 131 answer is below
  its unfitted network by about 0.012 on every split, with the interval below 0 on two of three. If
  the readout is adopted, W's arms can be read for the same paired difference.
- **The spike's code** is `investigation/neural-ceiling/` (kept as `.py.txt`). It used 8 workers for
  about 5 h on 2026-11-16, after the machine was quiet, and ran nothing while a C.2 run was active.
