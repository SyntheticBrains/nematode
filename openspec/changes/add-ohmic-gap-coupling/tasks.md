# Tasks: G.0-A — ohmic gap coupling and the substrate selection

Phase 9 G.0-A (roadmap v4.5 § Phase 9; tracker `phase9-tracking` G.0-A), after G.0 read `artefact`
([Logbook 095](../../../docs/experiments/logbooks/095-gap-operating-point.md)). The maintainer's
decisions of 2026-10-17: ohmic coupling in the settling map, selected against log-compressed counts in a
gated pilot; the point-worm contrast now, at a minimum of 0.0577; 092's body contrast after M.1 reads out.

- [ ] 1. **Brain option** `gap_coupling` (`additive` | `ohmic`): the operator `(I + L)⁻¹` built once from
  the gap matrix as a non-persistent buffer; the sensors equilibrated through it; both settling paths
  (single and batched); the modulator field's held current through it; the refusals (gap junctions off,
  `dynamics: leaky`, `plastic_gaps`, the plastic learning rules, node perturbation, activity traces,
  e-prop), by the validator and at build; the module and `forward_with_hidden` docstrings. Tests (design
  Decision E): the operator and its row sums; the no-gain test; the sensor relay; the held current; single
  and batched agree; `additive` byte-identical (pinned digests, state-dict keys, outputs, the next draw);
  every refusal. `.test_durations` regenerated.
- [ ] 2. **Configs** through `generate_gap_substrate_configs.py`: on the thermal cell at target 35 and on
  hard350, on Emmons 2024, each model's wild type and gap-only null learning and frozen, the additive-count
  wild type learning, the MLP control learning and frozen (frozen floors derived); loader and build tests.
- [ ] 3. **The operating-point probe** (design Decision D), at no run cost, before any pilot run: the
  saturated fraction, the action-mean spread and `M⁻¹`'s self-weights for the additive, log and ohmic wild
  types on both cells; the quarter-spread stop rule applied.
- [ ] 4. **Analysis** `scripts/analysis/gap_substrate_selection.py`: the cells' readability, each model's
  eligibility per cell, the shown-slower test against each cell's margin and the selection (design
  Decision C), never reading the wiring gap; the additive arm described; tests on synthetic manifests
  covering every branch of the rule.
- [ ] 5. **Registration**: the launch record (`docs/experiments/logbooks/supporting/097-gap-substrate/launch.md`)
  with the probe's result, the selection rule and its operating characteristic, the seeds, the cost from
  measured rates, and the retention line, committed before any pilot run; two-seed smokes on 9891–9892;
  `/nematode-review-spec`; the seed-band table gains 9871–9886, 9891–9892 and 3901–3964; the tracker's
  G.0-A item and the roadmap's G.0 entry carry the dated sequencing amendment (092's body re-run after M.1
  reads out).
- [ ] 6. **The selection pilot** from a separate worktree, 352 runs, with the output controls.
- [ ] 7. **Readout**: Logbook 097 (the probe, the pilot, the selection, the additive arm beside); the
  tracker's G.0-A item and the roadmap; CHANGELOG.
- [ ] 8. **Close-out**: full suite, `pre-commit run -a`, `openspec validate --strict`, archive, PR. The
  point-worm contrast on the chosen model registers next, in its own change, with D24 defined for ohmic
  coupling if ohmic is chosen.
