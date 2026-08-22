## ADDED Requirements

### Requirement: Shared action-policy module is the single source of truth for discrete policy scoring

Every brain architecture that scores a discrete categorical policy SHALL obtain its action log-probability, its policy entropy, and its policy-gradient loss term from the shared action-policy module [`brain/arch/_policy.py`](../../../packages/quantum-nematode/quantumnematode/brain/arch/_policy.py), rather than from an inline per-brain implementation. This extends the T5 D6 decision from the four MUST PPO brains to the whole PPO family plus the REINFORCE-family brains.

The module SHALL expose scorers for both distribution forms the brains use: a **logits-based** scorer for policies that are a softmax of network outputs, and a **probability-vector-based** scorer for policies that are constructed as an explicit mixture (such as an ε-greedy blend of a tempered softmax with a uniform distribution). Neither scorer SHALL apply an additive epsilon floor inside the logarithm; numerical stability is provided by `torch.distributions.Categorical`.

Brains that sample via a NumPy RNG SHALL retain their sampler verbatim; only the scoring and loss terms move to the shared module.

#### Scenario: No inline clipped surrogate remains

- **WHEN** the `brain/arch/` and `env/` trees are searched for a hand-written PPO clipped surrogate (a `torch.clamp(ratio, 1 - ε, 1 + ε)` paired with a `-torch.min(...).mean()`)
- **THEN** the only occurrence SHALL be inside `_policy.py` itself and its unit test
- **AND** every PPO-family brain SHALL obtain its policy loss from `ppo_clip_policy_loss`

#### Scenario: Mixture policies are scored by the shared module

- **GIVEN** a brain whose action distribution is an explicit mixture rather than a softmax of any available logits vector
- **WHEN** it computes the log-probability of the taken action and the policy entropy
- **THEN** it SHALL do so through the shared probability-vector scorer
- **AND** the construction of the mixture (including any temperature or exploration schedule) SHALL remain the brain's own concern, not the shared module's

#### Scenario: NumPy samplers are preserved

- **WHEN** a brain that samples via `rng.choice` is migrated onto the shared module
- **THEN** the sampling call and the probability vector passed to it SHALL be unchanged
- **AND** the sampled-action trajectory for a pinned seed SHALL be byte-identical to the pre-migration trajectory

### Requirement: Policy-consolidation migration regression bar

Migrating a brain's inline discrete-policy code onto the shared action-policy module SHALL preserve its behaviour to a tolerance declared before the migration, verified by a test that compares the migrated path against the exact pre-migration expression computed in the same process under the same pinned RNG state.

Brains whose pre-migration code already uses `torch.distributions.Categorical` SHALL be **byte-exact** (`torch.equal`). All other brains SHALL satisfy `torch.allclose(rtol=0, atol=1e-7)`, the same tolerance the standing *"Migration Regression Bar — Other 17 Architectures Numerical Equivalence"* requirement already binds these architectures to. Any architecture exceeding its declared tolerance SHALL be fixed, not have the tolerance widened after the fact.

Regression tests for this bar SHALL NOT assert against stored absolute floating-point constants, which drift across BLAS and torch builds without indicating a change in computation.

#### Scenario: Byte-exact families verified against the inline reference

- **GIVEN** a brain whose pre-migration scoring used `torch.distributions.Categorical`
- **WHEN** the pre-migration expression and the migrated brain path are evaluated on the same logits under the same pinned RNG state
- **THEN** the action, log-probability, and entropy SHALL satisfy `torch.equal`

#### Scenario: Declared-tolerance families verified against the inline reference

- **GIVEN** a brain whose pre-migration scoring used a manual `log(probs + ε)` or `-Σ p·log(p + ε)`
- **WHEN** the pre-migration expression and the migrated brain path are evaluated on the same distribution
- **THEN** the log-probability and entropy SHALL satisfy `torch.allclose(rtol=0, atol=1e-7)`
- **AND** the deviation SHALL be attributable to removing the epsilon floor and to torch's numerically stabler log-space evaluation

#### Scenario: Existing per-brain suites are unaffected

- **WHEN** the full non-nightly test suite is run after the migration
- **THEN** every migrated brain's existing test module SHALL pass without modification
- **AND** no previously-passing test SHALL fail, and no existing test SHALL be weakened, skipped, or otherwise altered to accommodate the migration
- **AND** the skipped and xfailed counts SHALL match the pre-migration baseline exactly, while the passing count SHALL increase by exactly the number of tests this change adds

### Requirement: Both halves of the PPO importance ratio use one formula

For any brain using a PPO clipped surrogate, the log-probability stored at rollout time and the log-probability recomputed at update time SHALL be produced by the same scoring formula, so that an unchanged policy yields an importance ratio of exactly 1.

#### Scenario: Rollout and update scoring are symmetric

- **GIVEN** a brain that stores an old log-probability during rollout and re-scores the same action during its PPO update
- **WHEN** the update is performed with policy parameters unchanged since rollout
- **THEN** the importance ratio SHALL carry no systematic offset attributable to a difference of *formula* between the two paths
- **AND** where both paths already operate on the same tensor dtype, the ratio SHALL be exactly 1 up to float round-off
- **AND** where the two paths straddle a pre-existing NumPy-float64 / torch-float32 boundary, the residual offset SHALL be bounded by the declared per-brain tolerance and SHALL be reported separately rather than averaged together with the same-dtype paths

#### Scenario: Rollout and update migrate together

- **WHEN** a brain's policy scoring is migrated onto the shared action-policy module
- **THEN** its rollout-side and update-side scoring SHALL be migrated in the same change, never one without the other
