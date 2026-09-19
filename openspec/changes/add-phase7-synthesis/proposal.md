# Close Phase 7: the synthesis, the terminal roadmap state, and what Phase 8 opens on

## Why

Phase 7's closing scope under D14 has shipped: L.0, V.4, L.1, and — after L.1's positive reopened
them — L.4, L.5 and L.1b. Three tracker items remain: L.2 (a SHOULD the tracker already allows
carrying), Z.1 and Z.2. Nothing outstanding needs a run.

An audit before closing found the close would be untidy in checkable ways, and the audit's findings
are what this change fixes rather than defers:

- **Six SHOULD-level exit criteria in the roadmap carry no status**, and Z.1's scope as written claims
  to settle only four ladder items. One line is stale (the 6a preprint shows unresolved where the
  tracker records it cancelled on 2026-09-15); three are 7b work that D14 deferred collectively
  without their own lines saying so; and two need a real decision — the **learnable-gap-junction
  ablation (D4)** and **co-primary biological validation**.
- **Block V's result still varies rewiring and initialisation together.** Logbook 065 registered the
  gap and left the control unregistered. After L.1b, block V is the phase's strongest citable result,
  and the one published critique aimed at this design — Dhiman 2026 — is exactly that confound.
- **Three more pinned settings sit under every `readout_only` result without ever being examined
  there**: `forward_pass_depth` 4 (never varied anywhere), `trace_decay` 0.9 (examined only on the
  three-factor rule's control, though it decays the readout's own eligibility trace), and
  `initial_log_std` −1.0 — which R.1d swept by hand *under the rule* and whose fixed value that same
  record named as an open question it called cheap to ask. L.1b showed an inherited pin can set the
  sign of a registered primary, so the three are a condition on the fixed-features results.
- **The record's raw inputs have partly decayed.** `campaigns/`, `exports/` and `experiments/` are all
  gitignored; exports are gone for every campaign before the readout-width era and one campaign has
  lost its tracked-experiment records. The committed per-seed CSVs are what the record rests on, and
  the close should say so rather than leave a future reader to discover it.
- **The wide readout never had its own positive control**, which the phase protocol requires of a new
  component. Its gates stood in, which is defensible and should be stated as such.

Two facts also came out of the audit in the project's favour, and the terminal reading should carry
them: **the nulls are rate-robust** (the pooled null leads or levels at both rates tested, so L.0's
`wiring_is_inert_as_features` does not depend on the pin that moved L.1), and the phase's negative
results have a diagnosed cause rather than an unexplained one.

## What changes

- **Logbook 069, the Phase 7 synthesis (Z.1)**: an exit-criterion walkthrough that assigns every
  MUST, SHOULD and MAY one of **met / unmet-with-reason / deferred-with-destination /
  superseded-by-result / unreachable-with-reason**, so no criterion is left unmarked; the terminal
  reading of the D10 2×2 against the claim-discipline bars; the three shipped results with their
  standing conditions stated in the same sentence as each claim; the phase's negatives with their
  diagnosed causes; a **reproducibility statement** naming what is re-derivable from git alone; and
  the limitations, including the unswept pins and the wide readout's missing control.
- **The six SHOULD lines settled**, each in the roadmap and the tracker: the preprint line corrected
  to cancelled; the three 7b lines marked deferred with D14 named; **D4's learnable-gap-junction
  ablation recorded deferred-with-destination, narrowed** — R.2 closed the class of rules that write
  the wiring and L.5 has now tested gap junctions as fixed features, so the **local-rule destination is
  closed**, while **the PPO arm on the block-V cells that B.6's own status names stays live and
  unattempted**; it is therefore an unresolved narrowed question, inherited by Phase 8's dynamics rung,
  and *not* superseded; and
  **co-primary biological validation recorded unreachable with its reason** — it needed a rule that
  writes the wiring to a benefit in order to make a sign-level prediction, and no such rule exists.
- **Block V's confound becomes a standing condition**, carried at every citation site (Logbook 065,
  the tracker, the roadmap, the synthesis) in the same sentence as the claim, with the control named
  as **Phase 8's first act** rather than left as an open caveat.
- **The roadmap's terminal state (Z.2)**: Phase 7 → ✅ COMPLETE (SPLIT), the Timeline Overview cell,
  the exit-criteria block and the Success Levels flipped; the README's "Phases 0–6a are complete"
  line flipped with it.
- **A Phase 8 opening inventory**, in one place in the roadmap: the init-vs-rewiring control first; a
  calibration rung over rate, width, `forward_pass_depth` and `initial_log_std`; the dynamics rung
  (gap-junction coupling) behind it; the placed-plasticity rung; the methodology-spec consolidation
  and the byte-identity rename; and the publication decision S.1 deferred to "after the close".

**No runs, and no package code.** The one substantive judgement is the status assigned to each
unresolved criterion, and each is argued from the committed record rather than asserted.

## Impact

- Affected specs: `plasticity-evaluation` (two requirements about how a close records scope)
- Also affected: `docs/research/phase-protocol.md` (principle 12 gains the five-status vocabulary)
- **Deliberately not here**: `AGENTS.md`'s stale 7a-i pilot recipe and the CHANGELOG shortening go
  with the v0.6.0 release change, so that this change touches only the record of what Phase 7 found
- Affected docs: `docs/experiments/logbooks/069-phase7-synthesis.md` (new), `docs/experiments/README.md`,
  `docs/roadmap.md`, `README.md`, `docs/experiments/logbooks/065-wiring-fresh-rewiring.md` (a dated
  standing-condition note), `CHANGELOG.md`, `openspec/changes/phase7-tracking/tasks.md`
- **No package code, no configs, no harnesses.** Every analysis script stays read-only; this change
  reads the committed record and nothing else.
