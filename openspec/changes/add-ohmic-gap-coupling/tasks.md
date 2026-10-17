# Tasks: G.0-A — ohmic gap coupling and the substrate selection

Phase 9 G.0-A (roadmap v4.5 § Phase 9; tracker `phase9-tracking` G.0-A), after G.0 read `artefact`
([Logbook 095](../../../docs/experiments/logbooks/095-gap-operating-point.md)). The maintainer's
decisions of 2026-10-17: ohmic coupling in the settling map, selected against log-compressed counts in a
gated pilot; the point-worm contrast now, 092's body contrast after M.1 reads out.

- [ ] 1. **Brain option** `gap_coupling` (`additive` | `ohmic`): the operator `(I + L)⁻¹` built once
  from the gap matrix; both settling paths (single and batched; the body drive reads the same settled
  state); the refusals (gap junctions off, `dynamics: leaky`, `plastic_gaps`, the plastic learning
  rules), by the validator and at build. Tests: the operator against a hand computation; no latch at
  high conductance; single and batched agree; `additive` byte-identical (pinned digests, outputs, the
  next draw); every refusal.
- [ ] 2. **Configs** through `generate_gap_substrate_configs.py`: the pilot's arms on the thermal cell at
  target 35 and on hard350, on Emmons 2024 (each model's wild type and gap-only null, learning and
  frozen; the MLP control, learning and frozen, its frozen floor derived where none is committed);
  loader and build tests.
- [ ] 3. **Analysis** `scripts/analysis/gap_substrate_selection.py`: the cells' readability, each model's
  eligibility per cell, the speed comparison and the selection (design Decision C), never reading the
  wiring gap; tests on synthetic manifests covering every branch of the rule.
- [ ] 4. **Registration**: the pilot's launch record (`docs/experiments/logbooks/supporting/097-gap-substrate/launch.md`)
  with the selection rule, the seeds, the cost from G.0's measured rates and the retention line,
  committed before any pilot run; `/nematode-review-spec`; the seed-band table gains 9871–9878 and
  3901–3964.
- [ ] 5. **The selection pilot** from a separate worktree, 160 runs, with the output controls.
- [ ] 6. **Readout**: the selection recorded in the launch record's results, the tracker's G.0-A item
  and the roadmap; Logbook 097 if the pilot finds something worth recording beyond the choice.
- [ ] 7. **Close-out**: full suite, `pre-commit run -a`, `openspec validate --strict`, archive, PR. The
  point-worm contrast on the chosen model registers next, in its own change.
