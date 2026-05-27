## ADDED Requirements

### Requirement: Canonical Predator Sensor + Reward Variant Selected by Phase 0 Investigation

The `weight-search-architecture-ranking` change's Phase 0 investigation SHALL select one canonical predator-sensing variant (sensor encoding + reward shape) for use across all Phase 4 C-curriculum cells. The investigation SHALL evaluate at minimum: the matched-compute baseline (legacy `nociception_klinotaxis` vs new two-channel biology at canonical T4 episode budget); a sparse-signal ablation (distal sulfolipid concentration injected into the mechano-strength field when not in contact); a redundancy ablation (single-channel composite predator-biology module collapsing 6 input dims to 4); and a reward-shape ablation (existing `gradient_proximity` vs a `distal_chemo_penalty + binary_contact_damage_trigger` variant). The chosen canonical variant SHALL be documented in writing in the change's logbook AND in the change's design.md before any Phase 4 C-curriculum cell launches.

This requirement establishes that "the canonical predator-evasion sensor + reward shape for Phase 4 onwards" is a deliberate, evidence-backed selection — not a default carried over silently from the corrected-biology shipping change. The selection is normative for downstream Phase 6 work: T7's L2 re-run on the upgraded substrate SHALL consume the same canonical variant unless the env-upgrade work plausibly invalidates it.

**Relationship with the existing "Two-Channel Predator-Sensing Model" requirement.** The existing requirement (line 15 of the main spec) mandates two orthogonal sensor channels (contact-mechanosensory + distal-chemosensory) and is the load-bearing biological-fidelity claim for the predator-sensing-biology capability. The redundancy-ablation variant (single-channel composite `predator_biology_klinotaxis` module collapsing 6 dims to 4) is an explicitly non-two-channel encoding. If Phase 0 selects the composite as canonical, the implementer SHALL ALSO ship a `MODIFIED Requirements` delta to the "Two-Channel Predator-Sensing Model" requirement that documents the single-channel composite as an alternative valid encoding alongside (not replacing) the existing two-channel architecture. The sparse-signal ablation variant remains two-channel and would NOT require a MODIFIED delta. If Phase 0 selects either the existing two-channel baseline or the sparse-signal variant, no MODIFIED delta is needed.

#### Scenario: Phase 0 investigation evaluates at minimum the four variant axes

- **WHEN** the change's Phase 0 work completes
- **THEN** the Phase 0 logbook section (or equivalent OpenSpec change artefact) SHALL document evaluation results for at minimum:
  - Matched-compute baseline of legacy `nociception_klinotaxis` vs new two-channel biology at the canonical Phase 4 episode budget (≥ 500 episodes, n ≥ 4 seeds)
  - Sparse-signal ablation: a variant injecting `predator_distal_concentration` into the `predator_contact_intensity` field when `predator_contact_zone == ContactZone.NONE`
  - Redundancy ablation: a single-channel composite `predator_biology` module emitting a 4-dim feature vector (intensity + zone-as-angle + distal_concentration + dconcentration_dt) replacing the two parallel 3-dim mechano + chemo modules
  - Reward-shape ablation: the existing `gradient_proximity` reward vs a `distal_chemo_penalty + binary_contact_damage_trigger` reward variant
- **AND** each variant's result SHALL include the canonical Phase 4 metrics (last-25 mean success, foods-collected, predator-deaths, etc.) with per-seed numbers preserved for downstream re-analysis

#### Scenario: Canonical variant locked before Phase 4 C-cells launch

- **GIVEN** Phase 0 investigation results in hand
- **WHEN** the change's design.md is amended with the canonical-variant decision
- **THEN** the design.md SHALL name exactly one canonical sensor encoding (one of: two-channel-as-shipped, sparse-signal-ablation variant, single-channel-composite variant) AND one canonical reward shape (one of: `gradient_proximity` as-shipped, `distal_chemo_penalty + binary_contact_damage_trigger` variant)
- **AND** the choice SHALL include written rationale citing the Phase 0 evidence (e.g. matched-compute deltas, statistical significance, biological-fidelity considerations)
- **AND** Phase 4 C-curriculum configs SHALL consume only the canonical variant
- **AND** non-canonical variants SHALL NOT appear in Phase 4 C-cells (they MAY remain available as opt-in modules for future ablation work)

#### Scenario: Canonical variant selected (Phase 0 outcome)

- **GIVEN** Phase 0 investigation completed 2026-05-27 with 40 canonical-budget runs (n=4 seeds × 500 episodes × 6 variants on MLPPPO small + klinotaxis sensing)
- **WHEN** the canonical variant is selected per the preceding scenario
- **THEN** the canonical sensor encoding SHALL be the two-channel-as-shipped variant: `predator_mechanosensation_klinotaxis` + `predator_chemosensation_klinotaxis` (the canonical biology-default new-biology sensor pair shipped by `fix-predator-sensing-biology`)
- **AND** the canonical reward shape SHALL be `reward_mode: distal_chemo_contact_trigger` (the new dual-mechanism reward shipped in this change — continuous distal-chemo penalty via `env.get_predator_concentration` + binary contact damage trigger at `dist <= 1`, with distance-scaled evasion and flat-fallback paths dropped)
- **AND** Phase 4 C-curriculum predator-evasion configs SHALL use both jointly (sensors + reward); neither alone produces the +14pp gain over legacy nociception_klinotaxis observed in Phase 0 ranking
- **AND** the rationale documented in design.md § "Phase 0 canonical-variant selection" SHALL cite at minimum: Phase 0 ranking placing this combination first at 81.0% ± 5.0 last-25 mean success vs legacy 67.0% ± 7.6 (+14pp); lowest death rate of all variants (18.0% vs legacy 32.0%); tightest variance of all variants; orthogonal stacking with the sparse_fix sensor variant (B0.6) does not compound (78.0% ± 6.9, within noise of B0.5); composite single-channel variant is structurally inferior (16.0% pre- and post-Bug-1-fix, confirming `lateral_gradient` is load-bearing for klinotaxis predator-evasion)

#### Scenario: Canonical variant becomes the carry-forward for Phase 6 downstream work

- **GIVEN** the change's design.md documents the canonical variant
- **WHEN** subsequent Phase 6 changes (T7 L2 re-run; T8 NEAT topology search) consume the predator-sensing capability
- **THEN** those changes SHALL consume the same canonical variant by default
- **AND** any departure from the canonical variant SHALL be documented as a deliberate per-change scope decision with rationale (e.g. T6's env-fidelity upgrades may plausibly invalidate the Phase 0 evidence and require a re-evaluation; that re-evaluation is itself a documented decision)
