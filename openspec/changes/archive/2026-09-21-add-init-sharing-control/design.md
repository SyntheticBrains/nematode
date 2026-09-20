## Overview

A.1 registers the init-vs-rewiring control. `docs/roadmap.md` § Phase 8 **D15** is authoritative for what "the same initialisation" means; this document records what the implementation turned out to do, why the primary is an interaction, and the scope decisions taken with the maintainer before anything ran.

## Design Decisions

### Decision A: What is actually confounded, read off the code rather than the prose

D15 was corrected at PR review against `rewire_degree_preserving`, and reading the initialisation path narrows it further. Established during planning:

| | |
|---|---|
| The draw | per-edge, 3,709 Gaussian values, walked in `(pre, post)` sorted order (`connectome_ppo.py:616-634`) |
| The scale | `1/sqrt(chemical in-degree)`, computed **per post-synaptic neuron** (`:588-601`), applied as the draw's `scale` |
| Edge count | identical across graphs (the swap rewires in place), so **both consume the same standard-normal stream** and the *n*-th value is the same in both |
| Periphery | already matched — rewiring uses a dedicated generator (`:2129-2133`), and `test_rewired_matches_wild_type_init_at_same_seed` asserts the readout and food gains are byte-identical across wirings |

So the residual confound is the **value-to-edge pairing** and the per-edge scale sequence that follows from it. Nothing else about initialisation differs. This is worth stating plainly because the standing condition's own wording — "rewiring varies with initialisation" — is broader than what the code does, and a reader could reasonably expect A.1 to be controlling more than it is.

### Decision B: The primary is an interaction, not a comparison against remembered numbers

The tempting design is to run the shared-init arms and compare their wiring effect to Block V's published +35.4/+23.5/+55.3/+40.1. That would be a cross-campaign comparison against results produced on a different seed set and a different dependency set.

Instead the draw mode is **crossed** with wiring and the **interaction** is the primary, one per new mode per cell, on `episodes_to_30pct_success`. This follows the requirement that a manipulation crossed with a structure contrast is read as an interaction and never as a main effect: a draw mode that moves both arms equally is a fact about the draw, not about the wiring. The three other efficiency metrics are reported beside it.

### Decision C: A.1 answers the pairing question only

Block V's standing condition covers two confounds:

1. **Within a seed** — the two graphs put the same drawn values on different edges. D15's definitions address this.
2. **Across seeds** — the null's graph and its weights both derive from the run seed, so the null arm carries graph-variance the wild-type arm does not. Fixing this means pinning `rewire_seed` to a constant.

Logbook 065 calls (2) "a different experiment" in its own words, and it is: pinning the graph across seeds reintroduces the shared-nulls caveat that V.4 closed by moving to fresh rewirings. **A.1 takes (1).** `rewire_seed` stays unset and is documented as equal to the run seed — a declared rule rather than a declared value, since a static YAML cannot express a per-seed expression. (2) is recorded as a follow-up in the tracker.

### Decision D: The `edge_order` baseline is re-run fresh, not reused

V.4's committed per-seed data for seeds 65–96 carries all four metrics and could serve as the baseline level. Against that: the governing requirement permits reuse only under a parsed-field identity check with **no partial reuse**, and PR #397 bumped five dependencies since those runs, which makes a field mismatch plausible. A failed check forces a full re-run anyway, after spending the check.

Re-running all three levels on one fresh seed set costs roughly two extra hours and buys a within-campaign contrast with uniform provenance. Taken.

**This runs a baseline in the same campaign as the manipulation, which principle 6's 2026-09-19 note warns against, so the difference has to be stated rather than assumed.** What that note forbids is reading an ablation against a baseline the same campaign is establishing at a *moved operating point* — L.4's error, where the rate check moved the arms and the ablation assumed the new baseline. Here the operating point does not move: `edge_order` is the committed block-V configuration, unchanged, and its wiring effect has been measured four times across V.1, V.3 and V.4. The in-campaign arm therefore **replicates a known baseline** rather than establishing an unknown one, and the crossed interaction is the registered form for this design. If the `edge_order` arm fails to reproduce block V's effect on fresh seeds, that is itself the finding and the interaction is not read — a branch task 8 registers.

### Decision D2: The primary metric is chosen after the censoring rates are known, by a rule fixed before they are

`episodes_to_30pct_success` is right-censored at the horizon, and the interaction is a difference of differences across four cells. The moved metric requirement forbids exactly that pairing unless censoring is equal across the cells, and it was written for L1b, where a 0.604 censoring spread voided a registered secondary. A draw mode that slows the null would censor more of its seeds and corrupt the interaction silently.

So the *rule* is registered in advance and the *choice* follows the data: censoring counted per cell, never pooled; rates differing means the uncensored `auc_success` is the primary with the censored metric beside it; rates matching means the reverse. Both readings are reported either way, which is what the requirement asks of a departure.

### Decision E: The instrument is not modified

`wiring_premise.py` hard-codes its cells, arms, family and minimum effects as tuples. Adding a third factor level by editing them would forfeit the property V.4's replication rests on — that the same analysis scored the original and the replication unmodified.

A new driver, `scripts/analysis/init_sharing_control.py`, therefore sits in the mould `wiring_fresh_rewiring.py` established: it maps config stems to (cell, arm), builds manifests, calls the unmodified instrument per draw mode, computes the interaction from its per-seed output, and reports branches. Its test asserts both instrument files are byte-identical to `main`, as `test_wiring_fresh_rewiring.py:145` already does.

### Decision F: One new requirement, deliberately

Seven requirements in `architecture-comparison-protocol` already govern this campaign, and are cited rather than restated: a wiring contrast under a learner that does not write the wiring; power stated in advance when a null carries a consequence; a control drawn from the same seeds stating that coupling; replication on the unmodified instrument; multi-panel disagreement fixed in advance; committed-baseline reuse; standing conditions.

The one case none covers: **a claim that two arms share an initialisation is a claim about what the code did**, and construction arguments about RNG streams are exactly the kind of reasoning that looks sound and is wrong. It is added as a requirement that the shared quantity be asserted by test.

Only one is added. A.4 has just finished redistributing 38 rules that accumulated one per rung; the inherited discipline is to add a requirement only where no existing one reaches.

## Open Questions (resolved during implementation)

- **Minimum effect on the interaction.** Registered as a fraction of the **baseline wiring effect measured in this campaign**, not of Block V's published figure — the ablation-minimum requirement asks for a fraction of the established effect, and the within-campaign baseline is the one the interaction is actually formed against. Registered in both directions before the panel runs. **Its power comes from V.4's committed per-seed spread**, 128 rows across both cells, which is the frozen prior-committed source the requirement names; four pilot seeds confirm the machinery and cannot estimate a spread.
- **Whether both new modes need both cells.** The plan runs both on both. If the pilot shows a mode is unrunnable on a cell, that is recorded as a limitation of that mode rather than resolved by dropping the cell.
- **Whether `dense_mask`'s larger draw needs a seed-offset guard.** `self.rng` feeds nothing but this loop, so consuming more values disturbs nothing downstream; the test asserting the periphery is untouched is what establishes it rather than the argument.

## Risks

- **The likeliest outcome is a shrunken effect, not a dissolved one**, since the periphery was already matched and only the pairing moves. The registered minimum in both directions is what stops a shrunken effect being reported as the original, and it is also what stops a small reduction being reported as dissolution.
- **The pilot band is nearly exhausted.** Seeds 101–104 are programme-wide pilots; this change uses 105–108 to avoid contaminating the registered set, and declares 129–144 for the panel.
- **A draw mode could fail a learning gate**, leaving a cell unscoreable at that mode. The registered family's gates are read before the contrast, as they already are; a gate failure is reported as one rather than worked around.
- **384 runs on one machine.** The campaign is launched with detailed export off per the runner's own warning; the retention rule (A.0) is recorded before it starts so the artefacts are not lost the way Logbook 069 records for the pre-readout-width era.
