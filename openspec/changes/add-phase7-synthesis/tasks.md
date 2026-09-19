# Tasks

## 1. The exit-criterion walkthrough

- [ ] 1.1 Every Phase 7 criterion gets exactly one of **met**, **unmet-with-reason**,
  **deferred-with-destination**, **superseded-by-result**, **unreachable-with-reason**, with its
  logbook or decision named. MUSTs and SHOULDs are gates on the close; **MAYs are marked without being
  gates**, as the roadmap states. No criterion left unmarked.
- [ ] 1.2 **D4's learnable-gap-junction ablation: deferred-with-destination, narrowed** — B.6's
  recorded status names two destinations, and R.2 (every substrate-writing arm does worse) closes the
  first while L.5 (removing gap junctions helped both wirings) lowers its priority, but **the second,
  a PPO arm on the block-V cells, is untouched and live**. Recorded as deferred with that destination
  in the roadmap's exit-criteria line and the tracker, with the narrowing stated. Not "superseded":
  that was this change's own first draft, and it is the error requirement 1 exists to catch.
- [ ] 1.3 **Co-primary biological validation: unreachable-with-reason** — both predictions are about a
  plastic wiring, and no rule that writes the wiring to a benefit exists. Recorded in both places.
- [ ] 1.4 The three 7b SHOULD lines (dauer pathfinder, weight-transplant transfer, third behaviour)
  marked **deferred-with-destination**, naming D14 and the phase after 7 on their own lines rather
  than relying on the block header.
- [ ] 1.5 The 6a-preprint line corrected: the tracker records S.1 **cancelled** 2026-09-15, and the
  roadmap shows it unresolved. The roadmap follows the tracker, with the publication decision named as
  relooked after the close.

## 2. The terminal reading, and the conditions that travel with it

- [ ] 2.1 The D10 2×2's terminal reading against the claim-discipline bars: the primary (plastic
  wild-type beats plastic rewired-null) is **unmet and unmeetable within the closing scope**, since
  every learner that reaches competence leaves `w_chem` frozen. State what the phase settled instead.
- [ ] 2.2 The three shipped results, each with its standing condition **in the same sentence as the
  claim**: block V's learning-speed advantage (rewiring varies with initialisation); L.1's
  pooling result (`plasticity_rate` 0.001, reversing a decade lower); L.4/L.5's feature ablations
  (each reading's qualification, and L.5 measured at 0.001 only).
- [ ] 2.3 **The nulls are rate-robust**, and it is reported as the terminal reading's firmest part:
  the pooled null leads or levels at both rates tested, so L.0's `wiring_is_inert_as_features` does
  not depend on the pin that moved L.1.
- [ ] 2.4 The phase's negatives with their diagnosed causes — the rule programme's failure mode, the
  instrument block's finding, the ladder re-read — stated as diagnoses rather than as absence.
- [ ] 2.5 The **unexamined-pins** table as a stated condition on every fixed-features result:
  `forward_pass_depth` 4 never varied anywhere, with its mechanism argument; `trace_decay` 0.9
  examined only on the three-factor rule's control though it decays the readout's own eligibility
  trace; and `initial_log_std` −1.0 swept by hand **under the rule** by R.1d and left by that record
  as an open question it called cheap to ask — quoted as R.1d's words, not presented as new.
- [ ] 2.6 The **wide readout's missing positive control** named, with the gates that stood in and why
  that is defensible here.

## 3. Block V's confound becomes a standing condition

- [ ] 3.1 A dated note in [Logbook 065](../../../docs/experiments/logbooks/065-wiring-fresh-rewiring.md):
  promoted from open caveat to standing condition, with Dhiman 2026 named as the external precedent
  and the control named as Phase 8's first act.
- [ ] 3.2 The same condition in the tracker's block V entries and in the roadmap wherever the +35% /
  +55% figures are cited, in the same sentence as the claim.
- [ ] 3.3 Why the control needs its own design decision — "the same initialisation" has no single
  meaning once the mask changes — so it opens a phase rather than closing one.

## 4. Reproducibility

- [ ] 4.1 A reproducibility statement in the synthesis: re-derivable from git alone (every headline
  figure, from the committed per-seed CSVs); needs the local campaign logs (re-derivation from raw);
  gone (exports before the readout-width era, and one block V campaign's tracked-experiment records).
- [ ] 4.2 Name the concrete instance it already cost: `peak_action_density` uncomparable in L.1b's
  pilot because L.0's export had been deleted.

## 5. The roadmap and README terminal state (Z.2)

- [ ] 5.1 Timeline Overview: Phase 7 → **✅ COMPLETE (SPLIT)**, the status cell delegating to the
  running-history section as it already does.
- [ ] 5.2 The exit-criteria block flipped to terminal, every line carrying its status from task 1.
- [ ] 5.3 Success Levels flipped: which level Phase 7 reached, stated against the criteria as written.
- [ ] 5.4 `README.md`'s "Phases 0–6a are complete" line flipped, with Phase 7's shipped results named
  at the level of detail the README uses.

## 6. What Phase 8 opens on

- [ ] 6.1 One inventory in the roadmap, in dependency order: **(a)** the init-vs-rewiring control,
  first, with its design question stated; **(b)** the calibration rung over rate, width,
  `forward_pass_depth` and `initial_log_std`; **(c)** the dynamics rung (gap-junction coupling),
  behind (b); **(d)** the placed-plasticity rung; **(e)** methodology-spec consolidation and the
  byte-identity → parsed-field rename; **(f)** the publication decision S.1 deferred to after the
  close.
- [ ] 6.2 Each item carries what it inherits from Phase 7 — the condition or the result that motivates
  it — so the inventory is not a wish list.
- [ ] 6.3 The carried SHOULD, L.2, appears in the inventory with its new precondition (the
  calibration rung) rather than as an unmarked leftover.

## 7. Bookkeeping

- [ ] 7.1 Logbook 069 written, the experiments index row added, `CHANGELOG.md` entry.
- [ ] 7.1a `docs/research/phase-protocol.md` principle 12 gains the **five-status vocabulary** and the
  lesson that deferred and superseded are not interchangeable, with this change's own first-draft
  error on D4 as the dated example.
- [ ] 7.2 The tracker: Z.1 and Z.2 ticked, L.2 settled as **carried** with its destination, and every
  status from task 1 recorded. No task left unmarked when the tracker archives.
- [ ] 7.3 `uv run openspec validate --all --strict` (only `benchmark-management` may fail — a
  deliberately retired spec with zero requirements), then `pre-commit run --all-files` clean.
