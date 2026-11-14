# Handover to the Phase 10 work

Written 2026-11-14 for the session continuing Phase 10 (C.2's registered fit, launch record 133, on
`feat/fit-c2`). It lists what the fit-first investigation found that bears on Phase 10, and what it
leaves for Phase 10 to decide. Nothing here changes a registered rule or verdict. The full record is
`investigation/`; the conclusion is `RECOMMENDATION.md`; the readings are Logbook 134.

## What changes for Phase 10

1. **W's sizing is about 1,200 to 2,900 machine-hours at its registered size.**

   - **The arithmetic:** 41 to 96 fits (5 null families, 8 to 16 draws, plus the wild type; the
     tracker's own cost line counts 6 arms) at about 30 h a fit, launch 133's measured rate. The
     tracker's 232 to 815 h assumed 4.8-h fits.
   - **The tracker already says** W at this size is "beyond what it can run as planned". Its three
     cuts are a stopping rule on draws, fewer draws, or a cheaper fit.
   - **The recommendation:** do not register W at its registered size. Pick a cut first.
   - **Phase 10's close:** 12 to 27 weeks from now with W at its registered size; 7 to 14 weeks if W
     is cut to a quarter of it (ORIENTATION.md § 5.4).

2. **A neural readout now exists, and it separates wirings** (Logbook 134). It scores a connectome
   model's in-silico single-neuron stimulation responses against Creamer et al.'s measured responses
   (MIT).

   - **The unfitted encoder:** at the box centre, on the true wiring, it scores r 0.065, against a
     median of 0.005 for 19 degree-preserving rewirings.
   - **C.2's Logbook 131 answer** scores 0.053, not distinguishable from unfitted.
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
   notes that the deposit names only half its stimulation targets.

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
