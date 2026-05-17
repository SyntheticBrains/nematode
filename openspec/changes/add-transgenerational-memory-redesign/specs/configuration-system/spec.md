# Configuration System — Delta for `add-transgenerational-memory-redesign` (M6.9+ PR-A)

## ADDED Requirements

### Requirement: Bias-Network Schema in TransgenerationalConfig

`TransgenerationalConfig` SHALL accept an optional `bias_network` sub-block specifying the substrate's sensory-conditional MLP architecture. Fields: `hidden_dim: int ≥ 0` (default 8; `0` means linear projection — no hidden layer), `activation: Literal["tanh", "relu", "gelu"]` (default `"tanh"`), `input_features: list[str]` (default `["predator_gradient_strength", "predator_gradient_direction_sin", "food_gradient_strength"]`). When the sub-block is absent or `null`, the substrate SHALL use the legacy M6 constant `logit_bias` path (byte-equivalent).

#### Scenario: default bias_network values

- **WHEN** a `TransgenerationalConfig` is loaded from YAML with `bias_network: {}` (empty sub-block, all defaults)
- **THEN** `bias_network.hidden_dim` SHALL be 8
- **AND** `bias_network.activation` SHALL be `"tanh"`
- **AND** `bias_network.input_features` SHALL be the 3-element default list

#### Scenario: hidden_dim 0 means linear projection

- **WHEN** `bias_network.hidden_dim: 0` is configured
- **THEN** the substrate's bias-network SHALL be a single linear layer (no hidden layer; no activation)

#### Scenario: input_features must reference known BrainParams field names

- **WHEN** `bias_network.input_features: ["nonexistent_field"]` is configured
- **AND** the config is loaded
- **THEN** the validator SHALL reject the config with an operator-friendly message naming the unknown field
- **AND** the suggested-fix message SHALL list the supported `BrainParams` fields

#### Scenario: input_features accepts \_sin / \_cos derived transforms

- **WHEN** `bias_network.input_features: ["predator_gradient_direction_sin", "food_gradient_direction_cos"]` is configured
- **AND** the underlying `BrainParams` fields `predator_gradient_direction` and `food_gradient_direction` exist
- **THEN** the validator SHALL accept the config without error
- **AND** at runtime the substrate's `apply_to_logits` and `extract_from_brain` SHALL derive the feature values by reading the underlying radian field and applying `math.sin` / `math.cos` respectively
- **AND** the raw radian field (without suffix) SHALL also remain a valid input_features entry (no transform applied)

#### Scenario: input_features rejects unknown \_sin / \_cos stem

- **WHEN** `bias_network.input_features: ["nonexistent_field_sin"]` is configured
- **AND** the config is loaded
- **THEN** the validator SHALL reject the config (the stem `nonexistent_field` is not a known `BrainParams` radian field)
- **AND** the error message SHALL clarify that only radian-valued `BrainParams` fields support the `_sin` / `_cos` suffix

#### Scenario: absent bias_network block preserves M6 byte-equivalence

- **WHEN** a `TransgenerationalConfig` is loaded WITHOUT a `bias_network` sub-block
- **THEN** the substrate SHALL fall back to the M6 legacy `logit_bias` constant tensor path
- **AND** behaviour SHALL be byte-equivalent to PR #166

### Requirement: Decay Shape Field in TransgenerationalConfig

`TransgenerationalConfig.decay_shape: Literal["geometric", "linear", "sigmoid"]` SHALL select the substrate cascade decay schedule. Default `"geometric"` preserves M6 byte-equivalence. The field SHALL be validated at YAML-load time; unknown values rejected.

#### Scenario: default decay_shape is geometric

- **WHEN** `TransgenerationalConfig` is loaded without `decay_shape`
- **THEN** `decay_shape` SHALL default to `"geometric"`
- **AND** the cascade SHALL be byte-equivalent to PR #166

#### Scenario: unknown decay_shape rejected at YAML load

- **WHEN** `decay_shape: "exponential"` is configured (not a member of the Literal)
- **THEN** the config loader SHALL reject the YAML with a clear validation error naming the unsupported value and listing valid options

### Requirement: Probe-Ring Schema in TransgenerationalConfig

`TransgenerationalConfig.probe_ring` SHALL accept the env-derived F0 extraction probe ring's shape: `count: int ≥ 1` (default 8), `radius_offset: int ≥ 0` (default 1), `include_food_gradient_variants: bool` (default `false`). When `probe_ring` is absent, the loader SHALL default to the canonical 8-position ring at `damage_radius + 1`.

#### Scenario: default probe_ring values

- **WHEN** a `TransgenerationalConfig` is loaded without a `probe_ring` sub-block
- **THEN** `probe_ring.count` SHALL be 8
- **AND** `probe_ring.radius_offset` SHALL be 1
- **AND** `probe_ring.include_food_gradient_variants` SHALL be `false`

#### Scenario: probe_ring.count must be at least 1

- **WHEN** `probe_ring.count: 0` is configured
- **THEN** the validator SHALL reject the config (would produce zero probes per predator)

#### Scenario: include_food_gradient_variants doubles probe count

- **WHEN** `probe_ring.count: 8` and `probe_ring.include_food_gradient_variants: true` are configured
- **AND** the F0 extraction runs
- **THEN** the probe builder SHALL emit `2 × 8 = 16` probes per predator (one ring with food-gradient set, one with food-gradient zero)

### Requirement: Reward Mode Switch in RewardConfig

`RewardConfig.reward_mode: Literal["default", "gradient_only"]` SHALL select the predator-evasion reward shape. Default `"default"` preserves byte-equivalence with M3 / M4 / M5 / M6. Under `"gradient_only"`, the distance-scaled evasion term SHALL be dropped from the per-step reward computation while the contact penalty and `HEALTH_DEPLETED` termination SHALL be preserved.

#### Scenario: default reward_mode preserves M3 byte-equivalence

- **WHEN** `RewardConfig` is loaded without `reward_mode` (or with `reward_mode: "default"`)
- **AND** a step is computed with a predator in detection range
- **THEN** the evasion reward SHALL include the distance-scaled term `penalty_predator_proximity × (curr_dist − prev_dist)` (M3/M4/M5/M6 byte-equivalent)
- **AND** the M3 reproduction test SHALL pass with byte-identical reward sequences

#### Scenario: gradient_only mode drops distance term

- **WHEN** `RewardConfig.reward_mode: "gradient_only"` is configured
- **AND** a step is computed with a predator in detection range at `curr_dist > 1`
- **THEN** the evasion reward SHALL NOT include the distance-scaled term
- **AND** the reward SHALL equal the food-approach + per-step cost components only

#### Scenario: gradient_only mode preserves contact penalty

- **WHEN** `RewardConfig.reward_mode: "gradient_only"` is configured
- **AND** a step is computed with a predator at `curr_dist ≤ 1` (contact)
- **THEN** the contact penalty `reward -= penalty_predator_proximity` SHALL still fire
- **AND** the `HEALTH_DEPLETED` termination path SHALL be unchanged

#### Scenario: unknown reward_mode rejected at YAML load

- **WHEN** `reward_mode: "sparse"` is configured (not a member of the Literal)
- **THEN** the config loader SHALL reject the YAML with a clear validation error naming the unsupported value

### Requirement: Three-Arm YAML Structural Pairing

The M6.9+ PR-A campaign SHALL run three structurally-distinct YAML configs corresponding to the `tei_on`, `weights_only`, and `control` arms. The existing `_validate_inheritance` pairing validator on `EvolutionConfig` (M6 invariant) SHALL NOT be relaxed. Instead, the `weights_only` and `control` YAMLs SHALL omit the `transgenerational:` block entirely (set to `None` via absence) so the validator's `transgenerational.enabled ↔ inheritance` rule does not fire on the non-TEI arms.

#### Scenario: tei_on YAML carries the full transgenerational block

- **WHEN** `transgenerational_m69_tei_on.yml` is loaded
- **THEN** `evolution.inheritance` SHALL equal `"transgenerational"`
- **AND** the `transgenerational:` block SHALL be present with `enabled: true`, `bias_network`, `decay_shape`, `probe_ring`, and `lawn_schedule` populated

#### Scenario: weights_only YAML omits the transgenerational block

- **WHEN** `transgenerational_m69_weights_only.yml` is loaded
- **THEN** `evolution.inheritance` SHALL equal `"lamarckian"`
- **AND** the `transgenerational:` block SHALL be absent (parsed as `None`)
- **AND** the `_validate_inheritance` pairing rule SHALL NOT fire (it only checks when the block is present)

#### Scenario: control YAML omits the transgenerational block

- **WHEN** `transgenerational_m69_control.yml` is loaded
- **THEN** `evolution.inheritance` SHALL equal `"none"`
- **AND** the `transgenerational:` block SHALL be absent

#### Scenario: lamarckian + transgenerational.enabled=False is still rejected

- **WHEN** a YAML mixes `inheritance: lamarckian` with `transgenerational: {enabled: false}` (the block is present but disabled)
- **THEN** the existing M6 `_validate_inheritance` pairing rule SHALL reject the config
- **AND** the operator-friendly message SHALL suggest either omitting the `transgenerational:` block entirely OR setting `inheritance: none`

### Requirement: Fitness Survival Weight Parity Across Arms

When the M6.9+ PR-A three-arm campaign is launched, all three YAMLs SHALL set `evolution.fitness_survival_weight` to the same value (default `1.0` for the M6.9+ campaign). The composite fitness `success_rate × (1 − fitness_survival_weight × death_rate)` is the legacy `fitness_metric: "composite"` formulation; under M6.9+ default `fitness_metric: "survival_rate"` this field is ignored at fitness computation but SHALL still match across arms for round-trip / artefact-symmetry. Parity remains a launch precondition because future ablation runs may switch back to composite, and a divergent value in the YAMLs is a config drift smell regardless of which dispatch is active.

#### Scenario: all three arm YAMLs set fitness_survival_weight: 1.0

- **WHEN** the three arm YAMLs (`tei_on`, `weights_only`, `control`) are loaded
- **THEN** each SHALL have `evolution.fitness_survival_weight` set to `1.0`
- **AND** the value SHALL be identical across all three so F0 elite selection uses the same composite-fitness rule

#### Scenario: campaign shell sanity-checks parity at launch

- **WHEN** `phase5_m69_transgenerational_lstmppo_klinotaxis.sh --pilot` or `--full` launches
- **AND** the three arm YAMLs are loaded
- **THEN** the shell SHALL extract `evolution.fitness_survival_weight` AND `evolution.fitness_metric` from each YAML
- **AND** if any `fitness_survival_weight` value differs from the others, the shell SHALL exit with a parity-violation error BEFORE any worker is dispatched
- **AND** if any `fitness_metric` value differs from the others, the shell SHALL exit with a parity-violation error BEFORE any worker is dispatched

### Requirement: Fitness Metric Primary-Scalar Selector

`EvolutionConfig.fitness_metric: Literal["composite", "success_rate", "survival_rate"]` SHALL select the per-eval primary scalar that `LearnedPerformanceFitness.evaluate` returns. Default `"composite"` preserves the M3 / M6 byte-equivalent fitness `success_rate × (1 − fitness_survival_weight × death_rate)`. Under `"survival_rate"` the function SHALL return `1 − deaths / eval_count`; under `"success_rate"` it SHALL return `successes / eval_count`. `fitness_survival_weight` SHALL be ignored under both non-default dispatches — the dispatch is the single point of metric selection.

#### Scenario: default composite preserves M3/M6 byte-equivalence

- **WHEN** `EvolutionConfig` is loaded without `fitness_metric` (or with `fitness_metric: "composite"`)
- **THEN** `LearnedPerformanceFitness.evaluate` SHALL return `success_rate × (1 − fitness_survival_weight × death_rate)` exactly as the pre-M6.9+ implementation did
- **AND** the M3 / M5 / M6 regression tests SHALL pass without YAML changes

#### Scenario: survival_rate dispatch returns survival fraction

- **WHEN** `EvolutionConfig.fitness_metric: "survival_rate"` is configured
- **THEN** `LearnedPerformanceFitness.evaluate` SHALL return `1 − (deaths / eval_count)`
- **AND** the return value SHALL be in `[0.0, 1.0]`
- **AND** `fitness_survival_weight` SHALL NOT influence the returned scalar

#### Scenario: success_rate dispatch returns foraging fraction

- **WHEN** `EvolutionConfig.fitness_metric: "success_rate"` is configured
- **THEN** `LearnedPerformanceFitness.evaluate` SHALL return `successes / eval_count` where `successes` counts episodes ending in `TerminationReason.COMPLETED_ALL_FOOD`
- **AND** `fitness_survival_weight` SHALL NOT influence the returned scalar

#### Scenario: unknown fitness_metric rejected at YAML load

- **WHEN** `fitness_metric: "fastest_food"` (or any value not in the Literal) is configured
- **THEN** Pydantic SHALL raise a `ValidationError` at config load with the offending value

#### Scenario: M6.9+ three arms set fitness_metric: survival_rate

- **WHEN** the three M6.9+ PR-A arm YAMLs (`tei_on`, `weights_only`, `control`) are loaded
- **THEN** each SHALL have `evolution.fitness_metric: "survival_rate"`
- **AND** the value SHALL be identical across all three so the cross-arm primary verdict compares the same scalar

### Requirement: Per-Genome Eval Diagnostics Side-Channel

`LearnedPerformanceFitness.evaluate` SHALL accept an optional `diagnostics_path: Path | None = None` keyword argument. When set to a non-`None` path, the function SHALL append one JSON-Lines row per call recording `genome_id`, `generation`, `seed`, `eval_count`, `successes`, `deaths`, `success_rate`, `survival_rate`, `composite`, `fitness` (the returned scalar), `fitness_metric`, and `fitness_survival_weight`. The `EvolutionLoop` SHALL pass `output_dir / "eval_diagnostics.jsonl"` for every per-genome eval so the decision-gate machinery (T1 envelope check, T3 M6-floor-to-beat, post-hoc aggregator) can read the per-eval `survival_rate` directly.

#### Scenario: diagnostics row carries all metric variants

- **WHEN** `LearnedPerformanceFitness.evaluate` is called with `diagnostics_path` set
- **THEN** one JSON-Lines row SHALL be appended to that path
- **AND** the row SHALL contain `success_rate`, `survival_rate`, `composite`, and the returned `fitness` scalar
- **AND** the row SHALL be reconstructable into a `dict` without parsing the whole file

#### Scenario: diagnostics path None preserves byte-equivalence

- **WHEN** `LearnedPerformanceFitness.evaluate` is called without `diagnostics_path` (or with `diagnostics_path=None`)
- **THEN** no file I/O SHALL be performed by the diagnostics side-channel
- **AND** the returned scalar SHALL be identical to the same call with `diagnostics_path` set (the side-channel is observation-only)
