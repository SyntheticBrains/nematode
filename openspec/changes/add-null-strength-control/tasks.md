# Tasks: The null-strength control

Phase 8 task **A.6**. The plan is in `docs/roadmap.md` § Phase 8, Required deliverable 5, added by
PR #405. The decisions taken before implementation are in this change's `design.md`.

## Code

- [ ] 1. **The rewiring function** (Decision A): add `rewire_gap_junctions` and `preserve_autapses` as
  keyword arguments. Under `preserve_autapses`, self-loops leave the directed swap's list and are
  re-added unchanged. The defaults take the same draws in the same order.
- [ ] 2. **The brain:** add `wiring: "rewired_chemical_only"`, and test `wiring != "wild_type"` at the
  rewiring step and at the measured prior's `rewired=`. No planning references in package code.
- [ ] 3. **Tests:**
  - the default null is unchanged edge for edge and count for count at fixed seeds, against pinned
    counts;
  - the chemical null keeps chemical in- and out-degree exactly, and keeps all 38 autapses with their
    counts;
  - the chemical null keeps its gap junctions identical to the wild type's, so per-neuron gap
    totals are unchanged;
  - the chemical null's non-autapse chemical edges differ from the wild type's;
  - as a contrast, per-neuron gap totals still move under the full null;
  - a brain built under the new value has `g_gap` bit-identical to the wild type's and a different
    `m_chem`;
  - a measured prior and the fan-in draw both build under the new value;
  - the value is refused nowhere it should be accepted.

## Panel

- [ ] 4. **The analysis**, `scripts/analysis/null_strength_control.py` (Decisions B–F):
  - the stems, seeds and levels, with the wild-type arms under both levels in the manifest;
  - the gates per level and the drift check;
  - the interaction per learner on both metrics;
  - the family correction;
  - `classify`, the verdict map and `honour_drift`;
  - the per-seed CSV and the analysis JSON.
- [ ] 5. **The configs:** a generator writes the 4 `_rewired_chemical_null` arms from their
  full-null parents, changing `wiring` alone.
- [ ] 6. **Panel tests:**
  - the configs load through the real loader and differ from their parents in `wiring` alone;
  - the reused arms are the committed configs;
  - each chemical-null frozen floor is built on its learning arm's weights, with `g_gap` equal to the
    wild type's;
  - the seeds are fresh;
  - every verdict row is checked on synthetic states, including the gates and a void drift;
  - the family is exactly the two primary interactions.

## Registration and run

- [ ] 7. **Pre-launch checks:**
  - the full suite passes;
  - `git add -A`, then `pre-commit --all-files`, judged by its exit code;
  - `openspec validate --strict` passes;
  - 8-episode smokes run one chemical-null arm per learner.
- [ ] 8. **The launch record**, `docs/experiments/logbooks/supporting/074-null-strength-control/launch.md`,
  committed **before any seed runs**. It covers:
  - the arms and seeds;
  - what each null preserves and does not (the new protocol requirement);
  - the interaction, the metric and its reason;
  - the sensitivity from the committed CSVs;
  - the minimum, the verdict map and the gates;
  - the combined-control reading;
  - retention and cost.
- [ ] 9. **The campaigns:** PPO (seeds 305–336), then the reading learner (seeds 337–384), with the
  output controls, and no branch switches until both finish.

## Records

- [ ] 10. **Score and commit** the per-seed CSV and the analysis JSON; re-scoring reproduces them.
- [ ] 11. **Logbook 074**, with each learner's verdict read as combined.
- [ ] 12. **Discharge:**
  - the index row;
  - tracker A.6 ticked;
  - block V's standing-condition notes resolved at every citation site — the roadmap's A.1 and
    A.6, the tracker's A.1, and Logbooks 067 and 070;
  - the roadmap's exit line.
- [ ] 13. **Close-out:** validate, archive, and open the PR.
