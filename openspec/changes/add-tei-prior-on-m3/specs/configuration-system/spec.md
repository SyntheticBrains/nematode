# Configuration System — Delta for `add-tei-prior-on-m3` (M6.13)

## ADDED Requirements

### Requirement: Composed Inheritance Literal Value

`EvolutionConfig.inheritance` SHALL accept a fifth Literal value `"weights+transgenerational"` (in addition to the existing four `"none"`, `"lamarckian"`, `"baldwin"`, `"transgenerational"`). When set, this value SHALL select the composed `LamarckianTransgenerationalInheritance` strategy that runs the M3 weight-inheritance path AND the M6.9+ substrate-flow path in parallel for every F1+ child.

The widened Literal SHALL be the canonical source of truth for the inheritance dispatch — every consumer (loop construction, CLI override resolution, checkpoint validation, validator pairing rules) reads from this single Literal.

#### Scenario: composed Literal value loads via YAML

- **WHEN** a YAML file sets `evolution.inheritance: weights+transgenerational` AND `evolution.transgenerational.enabled: true` AND `evolution.lawn_schedule` is populated with F1+ entries having `ppo_train_episodes > 0`
- **AND** the YAML is loaded via `load_simulation_config`
- **THEN** the resolved `EvolutionConfig.inheritance` SHALL equal the string `"weights+transgenerational"`
- **AND** no `ValidationError` SHALL be raised by the pairing validator (`_validate_inheritance`)
- **AND** the loop's strategy factory SHALL construct a `LamarckianTransgenerationalInheritance(elite_count=1)` instance for this value

#### Scenario: composed Literal value rejects unknown stem

- **WHEN** a YAML file sets `evolution.inheritance: weights+transgenerational_typo` (or any value not in the widened Literal)
- **AND** the YAML is loaded
- **THEN** Pydantic SHALL raise a `ValidationError` at config load with the offending value listed
- **AND** the error message SHALL list all five legal values: `"none"`, `"lamarckian"`, `"baldwin"`, `"transgenerational"`, `"weights+transgenerational"`

### Requirement: Composed Inheritance + Substrate-Enabled Pairing

The `_validate_inheritance` pairing rule on `EvolutionConfig` SHALL accept `transgenerational.enabled=True` paired with EITHER `inheritance == "transgenerational"` OR `inheritance == "weights+transgenerational"`. The pre-M6.13 rule (substrate-enabled requires exactly `inheritance == "transgenerational"`) SHALL be widened by one cell. The `transgenerational.enabled=False` case is unchanged — still requires `inheritance == "none"`.

The full cross-product validation matrix (positive cells; everything else MUST raise):

| `inheritance` | `transgenerational` block | Pairing verdict |
|---|---|---|
| `"none"` | absent / `None` | ✅ accepts |
| `"none"` | `enabled: false` | ✅ accepts |
| `"lamarckian"` | absent / `None` | ✅ accepts |
| `"baldwin"` | absent / `None` | ✅ accepts |
| `"transgenerational"` | `enabled: true` | ✅ accepts |
| `"weights+transgenerational"` | `enabled: true` | ✅ accepts (NEW under M6.13) |
| any other combination | — | ❌ rejects with named offending cell |

#### Scenario: composed inheritance + substrate-enabled is accepted

- **WHEN** a YAML sets `evolution.inheritance: weights+transgenerational` AND `evolution.transgenerational.enabled: true` (with all other M6.9+ schema fields populated correctly)
- **AND** the YAML is loaded
- **THEN** the validator SHALL NOT raise
- **AND** the resolved `EvolutionConfig.transgenerational.enabled` SHALL be `True`

#### Scenario: composed inheritance without transgenerational block is rejected

- **WHEN** a YAML sets `evolution.inheritance: weights+transgenerational` AND the `transgenerational:` block is absent (or `None`)
- **AND** the YAML is loaded
- **THEN** the validator SHALL raise a `ValidationError` with a clear message naming the missing requirement
- **AND** the message SHALL point the user to either populate the `transgenerational:` block (with `enabled: true` + bias_network + probe_ring) OR set `inheritance` to a non-composed value

#### Scenario: composed inheritance with transgenerational disabled is rejected

- **WHEN** a YAML sets `evolution.inheritance: weights+transgenerational` AND `evolution.transgenerational.enabled: false`
- **AND** the YAML is loaded
- **THEN** the validator SHALL raise a `ValidationError`
- **AND** the message SHALL explain that composed mode REQUIRES the substrate to be active (otherwise it's equivalent to pure Lamarckian; the user should set `inheritance: lamarckian` for that case)

#### Scenario: legacy non-composed pairings remain unchanged

- **WHEN** a YAML sets `evolution.inheritance: transgenerational` AND `evolution.transgenerational.enabled: true`
- **AND** the YAML is loaded
- **THEN** the validator SHALL accept (legacy M6.9+ pairing, unchanged)
- **AND** the loop SHALL construct a `TransgenerationalInheritance` strategy (NOT the composed strategy)

#### Scenario: Lamarckian with transgenerational block is rejected

- **WHEN** a YAML sets `evolution.inheritance: lamarckian` AND `evolution.transgenerational.enabled: true`
- **AND** the YAML is loaded
- **THEN** the validator SHALL raise a `ValidationError`
- **AND** the message SHALL state that to combine M3 weight inheritance with substrate inheritance, the user MUST set `inheritance: weights+transgenerational` (the composed mode), not `inheritance: lamarckian` with a separate substrate block

#### Scenario: Baldwin with transgenerational block is rejected

- **WHEN** a YAML sets `evolution.inheritance: baldwin` AND `evolution.transgenerational.enabled: true`
- **AND** the YAML is loaded
- **THEN** the validator SHALL raise a `ValidationError`
- **AND** the message SHALL state that Baldwin inheritance is mutually exclusive with substrate inheritance (no current composition; future work could add a `baldwin+transgenerational` mode if motivated)

### Requirement: F1+ Train Episode Coverage Under Composed Inheritance

When `evolution.inheritance: weights+transgenerational` is configured, the `_validate_inheritance` validator SHALL enforce that every `lawn_schedule` entry with `generation > 0` has `ppo_train_episodes > 0`. The validator SHALL emit a `ValidationError` listing every offending F1+ entry. Composed mode REQUIRES F1+ retraining — the prior must act on actual training to be a prior; `ppo_train_episodes: 0` at F1+ is reserved for pure-TEI (`inheritance: transgenerational`).

The gen-0 `lawn_schedule` entry SHALL be unaffected by this rule (gen 0 always trains under any inheritance setting; the existing M6.9+ rule that `ppo_train_episodes > 0` at gen 0 continues to apply under composed mode).

#### Scenario: composed mode with K=0 at F1+ is rejected

- **WHEN** a YAML sets `evolution.inheritance: weights+transgenerational` AND a `lawn_schedule` entry has `generation: 1, ppo_train_episodes: 0`
- **AND** the YAML is loaded
- **THEN** the validator SHALL raise a `ValidationError` naming the offending entry (`generation=1, ppo_train_episodes=0`)
- **AND** the message SHALL state that composed mode requires F1+ retraining
- **AND** the message SHALL point the user to either set `ppo_train_episodes > 0` for every F1+ entry OR set `inheritance: transgenerational` for the pure-TEI K=0 arm

#### Scenario: composed mode with K>0 at every F1+ entry passes

- **WHEN** a YAML sets `evolution.inheritance: weights+transgenerational` AND every `lawn_schedule` entry with `generation > 0` has `ppo_train_episodes >= 500`
- **AND** the YAML is loaded
- **THEN** the validator SHALL accept the config
- **AND** the K-coverage check SHALL NOT raise

#### Scenario: pure-TEI K=0 at F1+ remains valid

- **WHEN** a YAML sets `evolution.inheritance: transgenerational` (NOT composed) AND a `lawn_schedule` entry has `generation: 1, ppo_train_episodes: 0`
- **AND** the YAML is loaded
- **THEN** the validator SHALL accept the config (pure-TEI is the M6.9+ K=0 arm; this rule does NOT apply to pure-TEI)

### Requirement: M6.13 Three-Arm YAML Structural Pairing

The M6.13 campaign SHALL run three structurally-distinct YAML configs corresponding to the `tei_weights`, `weights_only`, and `control` arms. The pairing validator SHALL accept the three arm YAMLs with the following structural contract:

- `transgenerational_m613_tei_weights.yml`: `inheritance: weights+transgenerational` AND the `transgenerational:` block SHALL be PRESENT with `enabled: true`, `bias_network`, `decay_shape`, `probe_ring`, and `lawn_schedule` entries with `ppo_train_episodes >= 500` for every F1+ entry.
- `transgenerational_m613_weights_only.yml`: `inheritance: lamarckian` AND the `transgenerational:` block SHALL be ABSENT (parsed as `None`).
- `transgenerational_m613_control.yml`: `inheritance: none` AND the `transgenerational:` block SHALL be ABSENT.

The three arm YAMLs SHALL share the env, brain, reward, and `fitness_metric` subtrees as plain YAML duplication (no shared-base + overlay pattern — the validator constraint prevents that). The campaign shell SHALL sanity-check the three files at launch time (mirroring M6.9+'s parity check): `fitness_survival_weight`, `fitness_metric`, and the env's `grid_size` / `predators.count` / `predator_damage` / `min_food_predator_distance` MUST match across the three arms; otherwise the shell SHALL exit before any worker is dispatched.

#### Scenario: tei_weights YAML carries the full composed inheritance block

- **WHEN** `transgenerational_m613_tei_weights.yml` is loaded
- **THEN** `evolution.inheritance` SHALL equal `"weights+transgenerational"`
- **AND** the `transgenerational:` block SHALL be present with `enabled: true` AND all M6.9+ sub-blocks (`bias_network`, `decay_shape`, `probe_ring`) populated
- **AND** every `lawn_schedule` F1+ entry SHALL have `ppo_train_episodes >= 500`

#### Scenario: weights_only YAML omits the transgenerational block (M3 reproduction)

- **WHEN** `transgenerational_m613_weights_only.yml` is loaded
- **THEN** `evolution.inheritance` SHALL equal `"lamarckian"`
- **AND** the `transgenerational:` block SHALL be absent (parsed as `None`)
- **AND** the `_validate_inheritance` pairing rule SHALL NOT fire (it only checks when the block is present)

#### Scenario: control YAML omits the transgenerational block (TPE-fresh)

- **WHEN** `transgenerational_m613_control.yml` is loaded
- **THEN** `evolution.inheritance` SHALL equal `"none"`
- **AND** the `transgenerational:` block SHALL be absent

#### Scenario: launch-time parity check fires on env-config divergence across arms

- **WHEN** the campaign shell `phase5_m613_tei_prior_lstmppo_klinotaxis.sh --pilot` is invoked
- **AND** the three arm YAMLs have divergent `predators.count` (e.g. tei_weights uses 4 but control uses 3)
- **THEN** the shell SHALL exit with a non-zero status BEFORE dispatching any worker
- **AND** stderr SHALL name the divergent field and the per-arm values

#### Scenario: launch-time parity check fires on fitness_metric divergence across arms

- **WHEN** the campaign shell is invoked with divergent `fitness_metric` across arms (e.g. tei_weights uses `survival_rate` but weights_only uses `composite`)
- **THEN** the shell SHALL exit with a non-zero status BEFORE dispatching any worker
- **AND** stderr SHALL name the divergent field and the per-arm values
