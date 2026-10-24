## ADDED Requirements

### Requirement: Registered before the held-out data are read

A launch record SHALL fix every rule below, the matched negatives' parameters and the seeds before
any held-out value is read. The controls script SHALL refuse to run unless the launch record exists
in the repository's committed history.

#### Scenario: No launch record, no read

- **WHEN** the controls script runs without a committed launch record
- **THEN** it SHALL exit nonzero before loading any held-out row

### Requirement: Positive controls on held-out data

Fifteen control units SHALL be graded, each at its own size on bands built on demand:

- the held-out OD 2 wild type on the small lawn (six statistics) and the uniform well (three), at the
  wild type's bar;
- each held-out López-Cruz batch, on γ and early/late at the wild type's bar, and on presence;
- each mutant's held-out Δ against its held-out same-date wild type, at the mutant's bar, with the
  bands' resolved directions.

#### Scenario: A López-Cruz batch without search fails

- **WHEN** a held-out batch's statistics pass but its search is absent
- **THEN** that unit SHALL fail

### Requirement: The allowance across controls

The positive controls SHALL pass when at most 2 of the 15 units fail. The allowance SHALL be the
largest count that real data exceed with probability at most 0.05, under a Poisson-binomial of the
units' failure rates: 0.075 (small lawn), 0.038 (each López-Cruz batch), and 0.05 (the uniform well
and each mutant). Which units fail SHALL be reported.

#### Scenario: Three failing units void the controls

- **WHEN** three units fail
- **THEN** the positive controls SHALL fail and the benchmark SHALL be VOID

### Requirement: Matched negatives from band data

The open-loop HMM worm SHALL switch roaming and dwelling every 10 s by the authors' HMM transitions,
regardless of food. In each state it SHALL use that state's speed, curvature noise, and reversal and
omega hazards, measured on the OD 2 band cohorts. The Poisson worm SHALL use the same quantities
pooled over both states, held constant. No held-out date SHALL enter their parameters.

#### Scenario: The HMM worm ignores food

- **WHEN** the HMM worm runs on López-Cruz's assay
- **THEN** its state switching and kinematics SHALL be the same on and off food

### Requirement: Trivial negatives

The random walk and the correlated random walk SHALL run at the band wild type's mean on-lawn speed
with minimum hazards. The random walk's curvature SHALL be drawn uniformly over its full range each
second. The correlated walk's SHALL be an AR(1) with the wild type's heading persistence. The
untrained policy SHALL be 8 independently initialised networks × 25 animals for each of the MLP and
connectome arms, graded pooled.

#### Scenario: Every negative is graded at n = 200

- **WHEN** a negative is graded
- **THEN** each assay SHALL hold 200 animals, on animal seeds 5101–5300

### Requirement: The verdict

The benchmark SHALL be VALID when the positive controls pass by the allowance and every negative fails
the wild type, and VOID otherwise. Each negative's D and D_fit SHALL be reported beside its verdict,
and W's minimum SHALL be read by B.3's registered rule.

#### Scenario: One negative passes

- **WHEN** every positive control passes but one negative passes the wild type
- **THEN** the benchmark SHALL be VOID
