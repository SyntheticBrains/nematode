# Tasks: The init-vs-rewiring control

Phase 8 task **A.1** (with **A.0** riding along). The plan is authoritative in `docs/roadmap.md`
§ Phase 8 **D15**; the design decisions taken before anything ran are in this change's `design.md`.

**Registration discipline**: tasks 1–6 land and the protocol is registered **before any panel run**.
Task 7's pilot confirms the modes run and the cost estimate holds; the minimum effect and the power
come from V.4's committed per-seed spread, not from four pilot seeds. Task 8 registers all of it —
including the rule that picks the primary metric — and only then does the panel launch. **No panel
seed is touched before task 8**, and the pilot uses its own band so it cannot contaminate one.

- [x] 1. **A.0 — the artefact-retention rule** — done 2026-09-20, recorded in the Phase 8 tracker with the cost Logbook 069 records for its absence., recorded in `openspec/changes/phase8-tracking/tasks.md`
  before the first Phase 8 campaign: parsed per-seed CSVs committed under
  `docs/experiments/logbooks/supporting/`, raw campaign logs and step-level exports archived
  off-repo. Cited by every campaign that follows.

- [x] 2. **The two draw modes** — done: `WeightDraw` alias, the `weight_draw` field, a model validator refusing the `count_scaled` pairing, the constructor parameter, and the branch. The sharing modes pre-draw (a dense matrix, or one block per post-synaptic neuron paired in pre-synaptic-index order) so what two wirings share is a property of the draw rather than of the edge order. — `packages/quantum-nematode/quantumnematode/brain/arch/connectome_ppo.py`.
  A `weight_draw: Literal["edge_order", "dense_mask", "per_neuron_fanin"] = "edge_order"` field beside
  `weight_init`, threaded to `ConnectomeTopology.__init__` as `readout_width` is, and branched at the
  draw loop. `edge_order` is untouched. `dense_mask` draws one dense matrix and takes each edge's own
  cell scaled per post-synaptic neuron. `per_neuron_fanin` draws each post-synaptic neuron's fan-in
  values together and assigns them to its incoming edges in pre-synaptic-index order. Refuse
  `weight_draw != edge_order` with `weight_init: count_scaled` at load, per the
  `plasticity_plastic_readout` + `freeze_updates` precedent.

- [x] 3. **Brain tests** — done, 19 tests. Both sharing properties asserted against constructed brains; bitwise identity on both axes over **every** topology parameter except `w_chem`, so a tensor added later is covered without anyone remembering. One test asserts the default path does *not* share the common edges, which is what makes the control worth running. 1,907 brain/arch tests pass. — new
  `packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/test_connectome_weight_draw.py`,
  modelled on `test_connectome_count_init.py` and `test_connectome_readout_width.py`'s
  `TestTheRngStreamIsUntouched`. It asserts: the default is **bit-identical** to the pre-option brain;
  under `dense_mask`, every edge present in both graphs carries an identical value at one seed; under
  `per_neuron_fanin`, every neuron's incoming multiset matches across wirings; and the `count_scaled`
  combination raises. Bitwise identity is asserted on **both axes** — across the two wirings at one
  mode, and across modes for one wiring on the readout, food gains, critic and gap junctions — since
  naming only one leaves the other free to move unnoticed. The sharing property is **asserted, not
  argued**; that is the requirement this change adds.

- [x] 3b. **Docs for the new key** — done: the configs README variant list and the architectures catalogue row. — `configs/README.md`'s variant list gains `densemask` and
  `fanin` beside the `countinit` precedent it already documents as "the second panel's
  initialisation factor", and `docs/architectures.md`'s `connectomeppo` row names `weight_draw`
  where it already names `weight_init`.

- [x] 4. **Configs** — done: 16 new YAMLs, each verified through the real loader to differ from its parent in **exactly** `weight_draw` and to leave `rewire_seed` unset. The eight committed block-V configs serve `edge_order` unchanged. The YAML-compatibility suite picked them up automatically, 408 → 424. — 16 new YAMLs, one key off their parent, across four arms
  (`wt_ppo`, `rn_ppo`, `wt_frozen`, `rn_frozen`) × two new modes × two cells (thermal `_t20`,
  `hard350`). The eight committed block-V configs serve the `edge_order` level **unchanged**, which
  keeps the `_PANEL_COMMITS` identity `test_wiring_fresh_rewiring.py` pins. `rewire_seed` stays unset
  in every rewired arm, with a comment recording that it therefore equals the run seed.

- [x] 5. **The driver** — done: `scripts/analysis/init_sharing_control.py`. Builds one manifest per draw mode, drives the **unmodified** instrument, and reads the interaction off its per-seed block paired by seed. Metric orientation comes from the instrument's own `_METRICS` table rather than a second copy, so "positive means the wild type is better" means here what it means in every committed block-V record. The censoring rule is a function fixed in advance, not a judgement at reading time. — new `scripts/analysis/init_sharing_control.py` in the
  `wiring_fresh_rewiring.py` mould: stem → (cell, arm, mode) mapping, manifest building, calls to the
  **unmodified** `wiring_premise.py` and `connectome_structure_efficiency.py` per mode, the
  interaction computed from their per-seed output, branch reporting, and a refusal to score an
  incomplete panel without an explicit flag.

- [x] 6. **Driver test** — done, 46 tests: the stem mapping is the full 2×4×3 cross product with no duplicate targets, the baseline level uses the committed configs unchanged, both seed bands are unburnt and disjoint, all 16 new configs declare their mode and leave `rewire_seed` unset, the censoring rule moves the primary in both directions, a partial panel and an unknown stem both raise, and both instrument files are byte-identical to `main`. — new
  `packages/quantum-nematode/tests/quantumnematode_tests/analysis/test_init_sharing_control.py`,
  asserting the stem mapping, the seed freshness (129–144 disjoint from 1–96 and 101–108), the
  incomplete-panel refusal, that **`rewire_seed` stays unset in all sixteen new rewired configs**
  (as `test_wiring_premise.py` and `test_wiring_fresh_rewiring.py` already assert for the committed
  eight), and that both instrument files are **byte-identical to `main`**.

- [x] 7. **Pilot** — done 2026-09-20: 48 runs, 48/48 succeeded in 2,769 s at 14.8×, 24 MB. The modes run, the driver scores end to end, and the censoring spread was 0.000 on the thermal cell. It also measured the across-mode correlation the panel's power turns on and **found nothing usable** (−0.66 to +0.34 at n = 4), which is what moved the panel to 32 seeds. Two driver bugs it caught: the completeness check demanded both cells when the pilot is one by design, and the instrument expects its caller to create the temp directory. — seeds **105–108**, the **thermal `_t20`** cell (V.1's cell: the larger effect
  and the fuller committed spread), all three modes, configured the way the campaign will be, not
  lighter, per the cheapest-platform-first principle's cost clause. Its job is to confirm the modes
  run, the learning gates fire, and the wall-clock estimate holds. It is **not** the spread source —
  see task 8.

- [x] 8. **Register the protocol** — done 2026-09-20, before any panel seed ran: [`supporting/070-init-sharing-control/launch.md`](../../../docs/experiments/logbooks/supporting/070-init-sharing-control/launch.md). Carries the metric rule with its 0.10 censoring tolerance, the power table at ρ = 0 from V.4's committed spread, the two-thirds minimum **in both directions**, the separate-cells rule, the non-reproduction branch, and A.0's retention line. — before the panel, and this is the task the campaign's
  credibility rests on:

  - **The primary metric, chosen for the contrast it must support.** `episodes_to_30pct_success` is
    **right-censored** at the horizon, and the interaction is a **difference of differences** across
    four cells. The governing requirement forbids a censored metric as the primary for that shape
    unless censoring is known equal across the cells. So: censoring is counted **per cell, never
    pooled**; if the rates differ, the uncensored **`auc_success`** is the interaction's primary and
    `episodes_to_30pct_success` is reported beside it; if they match, the reverse. The rule and its
    trigger are fixed here, before any rate is known.
  - **The minimum effect**, as a fraction of the **within-campaign** baseline wiring effect, and
    **registered in both directions** — a reduction that would count as the effect dissolving, and a
    reverse effect that would count as anything at all.
  - **The power**, computed from V.4's committed per-seed spread
    (`supporting/065-wiring-fresh-rewiring/per-seed-primary.csv`, 128 rows across both cells) — the
    frozen prior-committed source the requirement asks for. Four pilot seeds cannot estimate a
    spread; they confirm the machinery. The exact sign-test floor is carried as `wiring_fresh_rewiring._power` does.
  - **The comparator and detectable effect size**, stated in advance because a null here carries a
    registered consequence.
  - **The two cells read separately**, a split reported as a split and never pooled toward whichever
    cell supports the original claim.

- [ ] 9. **The panel** — 2 cells × 4 arms × 3 modes × **32** seeds (**129–160**) = **768 runs** through
  `scripts/run_campaign.py`, 16 workers. ≈ 10–13 h. (Raised from 16 seeds at task 8: at 16 the censored metric could not have detected a total dissolution.)

- [ ] 10. **Score and read** — through the unmodified instrument; assign the registered branch; if the
  effect shrinks rather than dissolving, report it as shrunken against the registered minimum.

- [ ] 11. **Restate the claim wherever it is cited** — Block V's learning-speed result carries its
  status into `docs/roadmap.md`, the Phase 8 tracker and the logbook, in the same sentence as the
  claim. Whatever the outcome, the standing condition is either discharged or restated, not dropped.

- [ ] 12. **Logbook** — the next numbered record, with per-seed CSVs under `supporting/` per A.0.

## Not in this change

- **Pinning `rewire_seed` to a constant** to isolate the graph from the weights across seeds. This is
  the second half of Block V's standing condition and a genuinely different experiment, which
  Logbook 065 already calls one: it would reintroduce the shared-nulls caveat V.4 closed. Recorded as
  a follow-up in the Phase 8 tracker.
- **Any change to `wiring_premise.py` or `connectome_structure_efficiency.py`.** The replication
  property depends on them staying byte-identical.
