# Tasks: The gap-only split

A.6's follow-up, before the 8a synthesis. The decisions taken before implementation are in this
change's `design.md`.

## Code

- [x] 1. **done.** Original scope: **The wiring value:** add `wiring: "rewired_gap_junctions_held"`, which calls the rewiring
  function with both options off. No planning references in package code.
- [x] 2. **done: 5 tests in `test_connectome_chemical_null.py` — the gap-held null's `m_chem`, autapse diagonal and drawn `w_chem` are the current null's bit for bit, its `g_gap` the wild type's.** Original scope: **Tests.** At a fixed seed:
  - `m_chem` is bit-identical to the current null's;
  - `g_gap` is bit-identical to the wild type's;
  - the autapse diagonal equals the current null's;
  - the value validates;
  - a measured prior and the fan-in draw build on it.

## Panel

- [x] 3. **done: `scripts/analysis/gap_split.py`, with subcommands for the identity check, the reused-evidence check and scoring; scoring also re-derives A.6's move from the reused runs and checks it against A.6's committed per-seed CSV every time (`a6_reproduced`), rather than only in a test CI cannot run.** Original scope: **The analysis**, `scripts/analysis/gap_split.py` (Decisions B, D, F, G):
  - the stem table and levels, extending A.6's with `gap_held`;
  - a manifest drawn from two campaign directories;
  - the identity comparator;
  - the gates, the interaction and the family correction;
  - `classify` and the verdict map, and `honour_drift`;
  - the per-seed breakdown;
  - the CSV and the JSON.
- [x] 4. **done: the generator now writes both panels' arms from their own panel definitions; 4 written, A.6's 4 kept.** Original scope: **The configs:** the generator writes the 4 gap-held arms from their current-null parents,
  changing `wiring` alone.
- [x] 5. **done, 28 tests.** Original scope: **Panel tests:**
  - the configs differ from their parents in `wiring` alone;
  - each gap-held arm builds on its current-null parent's `m_chem`, with the wild type's `g_gap`;
  - the seeds equal A.6's;
  - the manifest reads both campaigns;
  - the identity comparator passes identical runs and fails a single changed `Run:` line or a changed
    final `w_chem`;
  - every verdict row is checked, with the gates and a void drift;
  - the breakdown's two terms sum to A.6's interaction per seed, and re-scoring the current and
    chemical-only nulls from `campaigns/a6-*` reproduces A.6's committed per-seed interaction in
    `074-null-strength-control/per-seed.csv`;
  - the family is the two primaries.

## Registration and run

- [x] 6. **done: full suite 6,924 passed; hooks pass; validated strict; 8-episode smokes of both gap-held arms completed; drift evidence resolves for all 480 reused runs (192 PPO, 288 reading).** Original scope: **Pre-launch checks:**
  - the full suite passes;
  - `git add -A`, then pre-commit, judged by its exit code;
  - `openspec validate --strict`;
  - 8-episode smokes of the two gap-held arms;
  - drift evidence resolves for all 480 reused A.6 runs (their experiment records and exports on
    disk).
- [x] 7. **done: committed before any new seed ran.** Original scope: **The launch record**, `docs/experiments/logbooks/supporting/075-gap-only-split/launch.md`,
  committed before any new seed runs. It covers:
  - what each of the three nulls preserves;
  - the identity check's exact command line and its fields (`Run:` lines and final `w_chem`);
  - the pairing;
  - the reuse and the identity check;
  - the interaction, the minimum and the verdicts;
  - the pessimistic sensitivity;
  - the breakdown;
  - retention and cost.
- [x] 8. **done: all 12 identical — 3,000 `Run:` lines each and the final `w_chem` bit for bit; evidence in `identity-ppo.json` and `identity-reading.json`.** Original scope: **The identity check:** 12 runs with A.6's exact command line, compared against A.6's runs
  on every `Run:` line and on the final `w_chem`, with the evidence committed. The new arms launch only if all 12 are identical.
- [x] 9. **done: 160/160, none failed; PPO 58 min, reading 2h52m.** Original scope: **The campaigns:** the gap-held arms on PPO (305–336), then on the reading learner
  (337–384), with the output controls and no branch switches.

## Records

- [x] 10. **done: `split.json` and `per-seed.csv` (80 rows); `a6_reproduced` true on both learners — A.6's move re-derived from the reused runs matches its committed per-seed CSV at every seed.** Original scope: **Score and commit** the CSV and the JSON; re-scoring reproduces them.
- [x] 11. **done: Logbook 075.** Original scope: **Logbook 075**, with the gap junctions' share reported beside A.6's own uncertainty.
- [x] 12. **done: index row; tracker A.1, A.6 and S8a; the sharpened condition at the roadmap's A.1 and A.6 and Logbooks 067, 070 and 074.** Original scope: **Discharge:**
  - the index row;
  - tracker A.6 and S8a;
  - the split's reading added to Logbook 074, the roadmap's A.6 note and block V's condition.
- [x] 13. **done: validated strict and archived; re-scoring reproduces the committed JSON and CSV byte for byte; every new link resolves; the PR is opened from this branch.** Original scope: **Close-out:** validate, archive, and open the PR.
