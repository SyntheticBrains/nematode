## Why

This is A.6's registered follow-up, and it runs before the 8a synthesis.

[Logbook 074](../../../docs/experiments/logbooks/074-null-strength-control.md) found that holding a
rewired null's gap junctions and autapses at the wild type's moves the wiring gap toward the null on
both learners:

- **PPO:** −0.028 `auc_success`, about half of block V's lead against the current null;
- **reading learner:** −0.099.

Both moves are significant (q = 0.019), and both are below the registered minimum. A.6 was a combined
control, so it cannot say whether the gap junctions or the autapses carried the move. A.5 will cite
block V, and it needs to know which.

**The gap-only null answers this with an exactly paired contrast.** It runs the current null's
chemical swap unchanged and skips only the gap-junction swap. At the same seed it therefore has **the
current null's chemical graph, edge for edge**, including the autapses that graph loses, while holding
the wild type's gap junctions. Against the current null it differs in its gap junctions alone:
placement and strength, jointly.

## What Changes

- **A fourth wiring value, `rewired_gap_junctions_held`.** It calls the rewiring function with the
  gap-junction swap off and autapse preservation off; both options already exist, from A.6. A test
  asserts the exact pairing: `m_chem` is the current null's, and `g_gap` is the wild type's.
- **The split, on A.6's seeds and runs.** Only the gap-held arms, learning and frozen, are new: 160
  runs across PPO (seeds 305–336) and the reading learner (337–384).
  - Every reused A.6 arm is licensed by the parsed-field identity check first: one seed per arm,
    re-run and compared on every `Run:` line, 12 runs.
  - Any difference means the reused baseline is re-run in full, never mixed.
- **The reading, registered before launch.** One paired interaction per learner on `auc_success`,
  read against 2/3 of A.6's committed move:
  - `gap_junctions` if holding them reproduces at least that much;
  - `partial` if a significant move falls short of it;
  - `not_gap_junctions` if no move is detected within it.
- **Sensitivity, stated honestly.** A.6's spread, the only committed proxy, puts the minimum
  detectable effect at about A.6's whole move. That proxy is pessimistic, because exact pairing
  removes the chemical-graph sampling noise in it. The achieved figure is reported beside it, never
  used to re-read a verdict.
- **A descriptive breakdown of A.6's move** into the gap junctions' share and a remainder. The
  remainder mixes the autapses with the chemical-graph difference in A.6's chemical-only null, and is
  reported as such.
- **What stays out of reach:** placement against strength. Separating them needs a control that moves
  one without the other; this change notes it and builds nothing for it.

## Capabilities

### Modified Capabilities

- `connectome-ppo-brain`: the rewired-null requirement gains `rewired_gap_junctions_held` and its
  exact-pairing guarantee.

## Impact

- **Code:** the wiring literal and the rewiring call in `brain/arch/connectome_ppo.py`. The default
  and every existing value are unchanged.
- **Scripts and configs:**
  - `scripts/analysis/gap_split.py`, holding the panel, the identity comparator and the breakdown;
  - the config generator, extended;
  - 4 configs;
  - A.6's, B.1c's, B.1b's and A.2's helpers, reused.
- **Data:** `campaigns/a6-ppo` and `campaigns/a6-reading` stay on disk until the logbook is committed.
- **Records:**
  - Logbook 075, with the identity-check evidence committed;
  - tracker A.6's follow-up and S8a;
  - the split's reading added to Logbook 074, the roadmap's A.6 note, and block V's condition.
