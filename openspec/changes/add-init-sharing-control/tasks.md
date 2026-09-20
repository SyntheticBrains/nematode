# Tasks: The init-vs-rewiring control

Phase 8 task **A.1** (with **A.0** riding along). The plan is authoritative in `docs/roadmap.md`
§ Phase 8 **D15**; the design decisions taken before anything ran are in this change's `design.md`.

**Registration discipline**: tasks 1–6 land and the protocol is registered **before any panel run**.
Task 7's pilot informs the minimum effect and the power arithmetic; task 8 registers them; only then
does the panel launch. No panel seed is touched before task 8.

- [ ] 1. **A.0 — the artefact-retention rule**, recorded in `openspec/changes/phase8-tracking/tasks.md`
  before the first Phase 8 campaign: parsed per-seed CSVs committed under
  `docs/experiments/logbooks/supporting/`, raw campaign logs and step-level exports archived
  off-repo. Cited by every campaign that follows.

- [ ] 2. **The two draw modes** — `packages/quantum-nematode/quantumnematode/brain/arch/connectome_ppo.py`.
  A `weight_draw: Literal["edge_order", "dense_mask", "per_neuron_fanin"] = "edge_order"` field beside
  `weight_init`, threaded to `ConnectomeTopology.__init__` as `readout_width` is, and branched at the
  draw loop. `edge_order` is untouched. `dense_mask` draws one dense matrix and takes each edge's own
  cell scaled per post-synaptic neuron. `per_neuron_fanin` draws each post-synaptic neuron's fan-in
  values together and assigns them to its incoming edges in pre-synaptic-index order. Refuse
  `weight_draw != edge_order` with `weight_init: count_scaled` at load, per the
  `plasticity_plastic_readout` + `freeze_updates` precedent.

- [ ] 3. **Brain tests** — new
  `packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/test_connectome_weight_draw.py`,
  modelled on `test_connectome_count_init.py` and `test_connectome_readout_width.py`'s
  `TestTheRngStreamIsUntouched`. It asserts: the default is **bit-identical** to the pre-option brain;
  under `dense_mask`, every edge present in both graphs carries an identical value at one seed; under
  `per_neuron_fanin`, every neuron's incoming multiset matches across wirings; in all three modes the
  readout, food gains, critic and gap junctions are untouched; and the `count_scaled` combination
  raises. The sharing property is **asserted, not argued** — that is the requirement this change adds.

- [ ] 4. **Configs** — 16 new YAMLs, one key off their parent, across four arms
  (`wt_ppo`, `rn_ppo`, `wt_frozen`, `rn_frozen`) × two new modes × two cells (thermal `_t20`,
  `hard350`). The eight committed block-V configs serve the `edge_order` level **unchanged**, which
  keeps the `_PANEL_COMMITS` identity `test_wiring_fresh_rewiring.py` pins. `rewire_seed` stays unset
  in every rewired arm, with a comment recording that it therefore equals the run seed.

- [ ] 5. **The driver** — new `scripts/analysis/init_sharing_control.py` in the
  `wiring_fresh_rewiring.py` mould: stem → (cell, arm, mode) mapping, manifest building, calls to the
  **unmodified** `wiring_premise.py` and `connectome_structure_efficiency.py` per mode, the
  interaction computed from their per-seed output, branch reporting, and a refusal to score an
  incomplete panel without an explicit flag.

- [ ] 6. **Driver test** — new
  `packages/quantum-nematode/tests/quantumnematode_tests/analysis/test_init_sharing_control.py`,
  asserting the stem mapping, the seed freshness (129–144 disjoint from 1–96 and 101–108), the
  incomplete-panel refusal, and that both instrument files are **byte-identical to `main`**.

- [ ] 7. **Pilot** — seeds **105–108**, one cell, all three modes, configured the way the campaign
  will be (not lighter), per the cheapest-platform-first principle's cost clause. Confirms the modes
  run, the learning gates fire, and yields the observed per-seed spread.

- [ ] 8. **Register the protocol** — before the panel: the minimum effect on the interaction as a
  fraction of the **within-campaign** baseline wiring effect, **registered in both directions**; the
  power to detect it from the pilot's spread; the detectable effect size and comparator stated in
  advance because a null here carries a registered consequence; and the rule that the two cells are
  read **separately**, a split reported as a split.

- [ ] 9. **The panel** — 2 cells × 4 arms × 3 modes × 16 seeds (**129–144**) = **384 runs** through
  `scripts/run_campaign.py`, 16 workers, detailed export off per the runner's warning. ≈ 5–6 h.

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
