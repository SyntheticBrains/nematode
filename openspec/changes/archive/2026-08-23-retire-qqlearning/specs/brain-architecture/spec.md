## RENAMED Requirements

- FROM: `### Requirement: Migration Regression Bar — Other 17 Architectures Numerical Equivalence`
- TO: `### Requirement: Migration Regression Bar — Non-MUST Architectures Numerical Equivalence`

## MODIFIED Requirements

### Requirement: Migration Regression Bar — Non-MUST Architectures Numerical Equivalence

The brain architectures outside the MUST set SHALL produce parameter tensors after
a 5-step smoke training pre-refactor and post-refactor that satisfy
`torch.allclose(rtol=0, atol=1e-7)`, with all RNG seeds pinned.

The architectures in scope are the registered non-MUST brains at the time the
refactor runs, taken from the registry rather than enumerated here. The count is
deliberately no longer in the requirement's title: it was "17" and became wrong the
moment an architecture was retired, which is the same drift the
`configuration-system` capability carried for `qqlearning` and its non-existent
legacy aliases.

`QQLEARNING` is no longer among them — `QQLearningBrain` was retired after the
roadmap recorded it as "evaluated, not competitive; deprioritised".

#### Scenario: Per-architecture numerical-equivalence smoke

- **GIVEN** a registered brain architecture outside the MUST set
- **AND** a 5-step smoke training config with a pinned seed
- **WHEN** the brain is trained for 5 steps pre-refactor and post-refactor on that config
- **THEN** every Pydantic-exposed parameter tensor SHALL satisfy `torch.allclose(post, pre, rtol=0, atol=1e-7)`
- **AND** any architecture exceeding the tolerance SHALL be either fixed or have its tolerance widened with explicit written justification in the T2 logbook

#### Scenario: Quantum architectures use deterministic simulator

- **WHEN** a quantum-family architecture's regression-equivalence test is executed
- **THEN** the test SHALL use a deterministic statevector simulator (not the noisy AerSimulator)
- **AND** SHALL pin shot-RNG seeds so QPU-shot variance does not introduce drift

#### Scenario: A retired architecture leaves the bar

- **WHEN** an architecture is removed from the registry
- **THEN** it SHALL cease to be in scope for this bar without requiring an edit to this requirement
