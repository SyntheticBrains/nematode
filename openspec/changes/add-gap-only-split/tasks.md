# Tasks: The gap-only split

A.6's follow-up, before the 8a synthesis. The decisions taken before implementation are in this
change's `design.md`.

## Code

- [ ] 1. **The wiring value:** add `wiring: "rewired_gap_junctions_held"`, which calls the rewiring
  function with both options off. No planning references in package code.
- [ ] 2. **Tests.** At a fixed seed:
  - `m_chem` is bit-identical to the current null's;
  - `g_gap` is bit-identical to the wild type's;
  - the autapse diagonal equals the current null's;
  - the value validates;
  - a measured prior and the fan-in draw build on it.

## Panel

- [ ] 3. **The analysis**, `scripts/analysis/gap_split.py` (Decisions B, D, F, G):
  - the stem table and levels, extending A.6's with `gap_held`;
  - a manifest drawn from two campaign directories;
  - the identity comparator;
  - the gates, the interaction and the family correction;
  - `classify` and the verdict map, and `honour_drift`;
  - the per-seed breakdown;
  - the CSV and the JSON.
- [ ] 4. **The configs:** the generator writes the 4 gap-held arms from their current-null parents,
  changing `wiring` alone.
- [ ] 5. **Panel tests:**
  - the configs differ from their parents in `wiring` alone;
  - each gap-held arm builds on its current-null parent's `m_chem`, with the wild type's `g_gap`;
  - the seeds equal A.6's;
  - the manifest reads both campaigns;
  - the identity comparator passes identical logs and fails a single changed `Run:` line;
  - every verdict row is checked, with the gates and a void drift;
  - the breakdown's two terms sum to A.6's interaction per seed;
  - the family is the two primaries.

## Registration and run

- [ ] 6. **Pre-launch checks:**
  - the full suite passes;
  - `git add -A`, then pre-commit, judged by its exit code;
  - `openspec validate --strict`;
  - 8-episode smokes of the two gap-held arms.
- [ ] 7. **The launch record**, `docs/experiments/logbooks/supporting/075-gap-only-split/launch.md`,
  committed before any new seed runs. It covers:
  - what each of the three nulls preserves;
  - the pairing;
  - the reuse and the identity check;
  - the interaction, the minimum and the verdicts;
  - the pessimistic sensitivity;
  - the breakdown;
  - retention and cost.
- [ ] 8. **The identity check:** 12 runs, compared field by field against A.6's logs, with the
  evidence committed. The new arms launch only if all 12 are identical.
- [ ] 9. **The campaigns:** the gap-held arms on PPO (305–336), then on the reading learner
  (337–384), with the output controls and no branch switches.

## Records

- [ ] 10. **Score and commit** the CSV and the JSON; re-scoring reproduces them.
- [ ] 11. **Logbook 075.**
- [ ] 12. **Discharge:**
  - the index row;
  - tracker A.6 and S8a;
  - the split's reading added to Logbook 074, the roadmap's A.6 note and block V's condition.
- [ ] 13. **Close-out:** validate, archive, and open the PR.
