# predator-sensing-biology Specification

## Purpose

The `predator-sensing-biology` capability provides a biologically-grounded two-channel predator-detection model that replaces the single chemosensory-at-distance `nociception` channel originally flagged as biologically wrong in [Logbook 011](../../../docs/experiments/logbooks/011-multi-agent-evaluation.md). The capability owns the env-side `ContactZone` enum + zone-discrimination method, the `predator_mechanosensation_{oracle,temporal,klinotaxis}` and `predator_chemosensation_{oracle,temporal,klinotaxis}` sensor modules, the `predator_mechano` and `predator_distal` STAM channels, the new `BrainParams` predator fields (`predator_contact_intensity`, `predator_contact_zone`, `predator_distal_concentration`, `predator_distal_dconcentration_dt`, `predator_mechano_dintensity_dt`), and the `SensingConfig.predator_mechano_mode` + `predator_distal_mode` fields.

**Contact-mechanosensory channel** models the ASH (nose) / ALM / AVM (anterior body) / PVD (body harsh) / PLM (posterior body) mechanosensory pathway, with anterior/posterior/lateral receptive-field discrimination supporting the canonical reversal-vs-acceleration escape circuit (Pirri & Alkema 2012; Kawano et al. 2011). Contact intensity is graded — replacing the prior boolean `predator_contact` field — with STAM-driven habituation kinetics matching the Hilliard et al. 2005 ASH adaptation timescale.

**Distal-chemosensory channel** models the ASH + ASI detection of predator-secreted sulfolipids documented by Liu et al. 2018 (*Nat. Commun.*) — a genuine distal-chemosensory escape signal *C. elegans* uses to evade predators before physical contact. The current env-side decay constant is the existing exp-decay-sum placeholder; literature-calibrated decay deferred to a future env-fidelity tranche.

The capability lives alongside the legacy `nociception` / `nociception_temporal` / `nociception_klinotaxis` modules in [`brain-architecture`](../brain-architecture/spec.md), which stay frozen in place to preserve byte-equivalent loading of archived predator-evasion configs.

## Requirements

### Requirement: Two-Channel Predator-Sensing Model

The system SHALL provide biologically-grounded predator detection via two orthogonal sensor channels: a contact-mechanosensory channel modelling ASH (nose) / ALM / AVM (anterior body) / PVD (body harsh) / PLM (posterior body) function, and a distal-chemosensory channel modelling ASH + ASI detection of predator-derived chemical signals (Liu et al. 2018 *Nat. Commun.* sulfolipid analogue). The two channels SHALL be independently configurable and independently consumable by brain architectures.

#### Scenario: Contact-mechanosensory channel emits graded intensity

- **GIVEN** an env with predators enabled and an agent positioned within `damage_radius` of a predator
- **WHEN** the agent's `BrainParams` are populated for the current step
- **THEN** `BrainParams.predator_contact_intensity` SHALL be set to `max(0, 1 - manhattan_dist_to_nearest_predator / damage_radius)` clipped to `[0, 1]`
- **AND** the intensity SHALL be exactly `1.0` when the agent literally overlaps the predator (Manhattan distance 0)
- **AND** the intensity SHALL be exactly `0.0` at the damage-radius edge
- **AND** the intensity SHALL be exactly `0.0` outside the damage radius

#### Scenario: Contact-mechanosensory channel distinguishes anterior / posterior / lateral zones

- **GIVEN** an env with a predator in physical contact with the agent
- **WHEN** `env.get_agent_predator_contact_zone_for(agent_id)` is called
- **THEN** the method SHALL return a `ContactZone` enum value chosen by the predator's relative bearing vs the agent's forward heading:
  - `ANTERIOR` if the predator is within ±45° of the agent's forward heading
  - `POSTERIOR` if the predator is within ±45° of the opposite-of-heading
  - `LATERAL` otherwise (any other approach angle within the damage radius)
- **AND** the method SHALL return `NONE` when no predator is within the damage radius
- **AND** the agent's `BrainParams.predator_contact_zone` SHALL be populated with the returned value when env predator is enabled

#### Scenario: Distal-chemosensory channel emits concentration + temporal derivative

- **GIVEN** an env with predators enabled
- **WHEN** the agent's `BrainParams` are populated for the current step
- **THEN** `BrainParams.predator_distal_concentration` SHALL be set to `env.get_predator_sulfolipid_concentration(agent_pos)`
- **AND** `BrainParams.predator_distal_dconcentration_dt` SHALL be set to the STAM `predator_distal` channel's derivative output when STAM is enabled
- **AND** `predator_distal_dconcentration_dt` SHALL be `None` when STAM is disabled or the agent's first step in an episode (no history)

### Requirement: STAM Predator Channel Split

The Short-Term Associative Memory (STAM) registry SHALL provide two separate predator channels — `predator_mechano` and `predator_distal` — so that habituation kinetics for the mechanosensory channel can ride independently of the distal-chemosensory channel's temporal averaging. The legacy `predator` channel SHALL remain in the registry as a frozen alias for backward compatibility.

#### Scenario: STAM activates predator channels per sensory-modules selection

- **WHEN** `resolve_active_channels(env, sensory_modules)` is called with `env.predator.enabled = True`
- **THEN** the resolver SHALL return the `predator_mechano` channel iff `sensory_modules` contains any variant of `predator_mechanosensation` (oracle / temporal / klinotaxis)
- **AND** SHALL return the `predator_distal` channel iff `sensory_modules` contains any variant of `predator_chemosensation` (oracle / temporal / klinotaxis)
- **AND** SHALL return the legacy `predator` channel when **either** (a) `sensory_modules` contains any variant of `nociception` (oracle / temporal / klinotaxis), **or** (b) no new-family predator module (`predator_mechanosensation*` / `predator_chemosensation*`) is selected — the latter rule guards the byte-equivalent-load invariant for archived configs that pre-date the new-family modules and may pass `sensory_modules=None` or an empty list to `resolve_active_channels`
- **AND** the three channels MAY all activate simultaneously if a config lists modules from multiple families (no rejection; STAM memory dim grows accordingly)

#### Scenario: STAM dimension formula absorbs new channels

- **WHEN** `compute_memory_dim(num_channels)` is invoked with the count of active channels
- **THEN** the formula `2 * num_channels + 3` SHALL be applied unchanged
- **AND** brain consumers of `BrainParams.stam_state` SHALL absorb the dim change via the existing dynamic-shape interface (no per-brain code edits required)

### Requirement: New Sensor Modules in Registry

The `SENSORY_MODULES` registry SHALL provide six new modules — `predator_mechanosensation_oracle`, `predator_mechanosensation_temporal`, `predator_mechanosensation_klinotaxis`, `predator_chemosensation_oracle`, `predator_chemosensation_temporal`, `predator_chemosensation_klinotaxis` — exposed via the corresponding `ModuleName` StrEnum entries. The new modules SHALL adopt an explicit `_oracle` suffix on the oracle variant (differing from the legacy bare-named convention used by `food_chemotaxis`, `nociception`, etc. — see [design.md § Decision T3.1](../../design.md)).

#### Scenario: New modules registered with expected dimensions

- **WHEN** the `brain.modules` module finishes import
- **THEN** the `SENSORY_MODULES` dict SHALL contain entries for all six new module names
- **AND** the oracle variants (`predator_mechanosensation_oracle`, `predator_chemosensation_oracle`) SHALL have `classical_dim = 2`
- **AND** the temporal variants SHALL have `classical_dim = 2`
- **AND** the klinotaxis variants SHALL have `classical_dim = 3`
- **AND** each module's `extract` callable SHALL produce a finite-valued array matching its declared `classical_dim`

#### Scenario: apply_sensing_mode translates new oracle-base module names by suffix

- **GIVEN** a YAML config specifying `sensory_modules: [predator_mechanosensation_oracle, predator_chemosensation_oracle]` and `sensing.predator_mechano_mode: klinotaxis` and `sensing.predator_distal_mode: temporal`
- **WHEN** `apply_sensing_mode(sensory_modules, sensing)` is called during config loading
- **THEN** the returned list SHALL contain `predator_mechanosensation_klinotaxis` (mode-translated from `predator_mechanosensation_oracle` by the `predator_mechano_mode` value)
- **AND** SHALL contain `predator_chemosensation_temporal` (mode-translated by the `predator_distal_mode` value)

### Requirement: Legacy Nociception Compatibility

The existing `nociception` / `nociception_temporal` / `nociception_klinotaxis` modules SHALL remain in the registry permanently, byte-identical to their pre-T3 behaviour, so that archived predator-evasion configs continue to LOAD and RUN without behavioural change.

#### Scenario: Archived configs load via legacy modules

- **GIVEN** any archived config under `configs/evolution/` that names `nociception*` modules in its `sensory_modules` list
- **WHEN** `configure_brain(load_simulation_config(<path>))` is invoked
- **THEN** the call SHALL succeed without error
- **AND** the resulting brain SHALL consume the legacy `nociception*` module's extract function
- **AND** the brain's input shape SHALL be identical to the pre-T3 result

#### Scenario: Legacy nociception modules are never auto-substituted

- **GIVEN** a config selecting `nociception_klinotaxis` in its `sensory_modules` list
- **WHEN** the config is loaded and the brain is constructed
- **THEN** the resulting brain SHALL consume the legacy `nociception_klinotaxis` extract function (NOT the new `predator_mechanosensation_klinotaxis` or `predator_chemosensation_klinotaxis` modules)
- **AND** no deprecation warning SHALL be raised at config-load time (silent legacy support preserves Phase 5 reproducibility)

### Requirement: BrainParams Predator Fields

`BrainParams` SHALL expose two parallel sets of predator-sensing fields: a frozen-legacy set (consumed by `nociception*` modules and the reward calculator), and a new set (consumed by the two-channel `predator_mechanosensation*` + `predator_chemosensation*` modules). The two sets SHALL be populated independently — populating one MUST NOT mutate the other.

#### Scenario: BrainParams exposes new predator fields alongside legacy fields

- **WHEN** a `BrainParams` instance is constructed with default arguments
- **THEN** the legacy fields SHALL exist with defaults `predator_contact: bool | None = None`, `predator_concentration: float | None = None`, `predator_dconcentration_dt: float | None = None`, `predator_gradient_strength: float | None = None`, `predator_gradient_direction: float | None = None`, `predator_lateral_gradient: float | None = None`
- **AND** five new fields SHALL exist with defaults `predator_contact_intensity: float | None = None`, `predator_contact_zone: ContactZone | None = None`, `predator_distal_concentration: float | None = None`, `predator_distal_dconcentration_dt: float | None = None`, `predator_mechano_dintensity_dt: float | None = None`
- **AND** `predator_mechano_dintensity_dt` SHALL carry the STAM-computed temporal derivative of `predator_contact_intensity` for the mechanosensation temporal + klinotaxis sensor modules (independent of the legacy `predator_dconcentration_dt` field, which the legacy `nociception_*` modules continue to read)
- **AND** populating the new fields SHALL NOT affect the legacy fields' values (and vice versa)

### Requirement: SensingConfig Mode Fields

`SensingConfig` SHALL accept three independent predator-related sensing-mode fields: the existing `nociception_mode` (frozen, used by legacy configs), plus two new fields `predator_mechano_mode` and `predator_distal_mode` (used by configs selecting the new sensor modules).

#### Scenario: SensingConfig accepts new mode fields with defaults

- **WHEN** a `SensingConfig` instance is constructed with no overrides
- **THEN** `predator_mechano_mode` SHALL default to `SensingMode.ORACLE`
- **AND** `predator_distal_mode` SHALL default to `SensingMode.ORACLE`
- **AND** the existing `nociception_mode` SHALL default to `SensingMode.ORACLE` (unchanged from pre-T3)
- **AND** all three mode fields SHALL accept the four `SensingMode` enum values (`ORACLE`, `TEMPORAL`, `DERIVATIVE`, `KLINOTAXIS`)

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

- **GIVEN** Phase 0 investigation completed 2026-05-27 with 40 canonical-budget runs across 6 sensor + reward variants on MLPPPO small + klinotaxis sensing (n=4 seeds × 500 episodes per variant; 2 Step A pre-flight + 20 Step B pre-fix + 12 Step B re-run after Bug 1 fix + 8 extended-validation)
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
