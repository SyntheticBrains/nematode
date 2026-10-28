# benchmark-controls Specification

## Purpose

Decide whether the benchmark can tell worms from non-worms. Held-out real worms must pass it, within a
registered allowance across control units, and matched and trivial negatives must fail it. The rules
are fixed in a launch record committed before any held-out value is read, and W's minimum effect is
read from real half-splits.

## Requirements

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
pooled over both states, held constant. Curvature noise and heading persistence SHALL be read only on
seconds whose displacement is at least the readouts' heading floor. Hazards SHALL be onsets per second
outside every bout. No held-out date SHALL enter their parameters.

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
- **AND** each animal's randomness SHALL come from the evaluator's per-animal generator, so it is the
  same alone or in a batch

### Requirement: The verdict

The benchmark SHALL be VALID when the positive controls pass by the allowance and every negative fails
the wild type, and VOID otherwise. Each negative's D and D_fit SHALL be reported beside its verdict.
W's minimum SHALL be read by B.3's registered rule. For local search, the scale of a half is the band
batches' SD divided by the square root of the half's batch count.

#### Scenario: One negative passes

- **WHEN** every positive control passes but one negative passes the wild type
- **THEN** the benchmark SHALL be VOID

### Requirement: B v2 is valid only if a faithful point worm can pass and the known exploits fail

B v2 SHALL be VALID only when all of the following hold:

- the measurement-model checks pass: the losses and the jitter on Scheer's rig, and the round trip on Chen's;
- the reachability oracle passes;
- the fresh N2 control passes, with at most 1 of its 4 units failing;
- the chemotaxis leave-one-recording-out validation passes within its registered allowance, a recording's unit failing where its chemotaxis is not present;
- B.4's six negatives each fail the wild type, and its four open-loop negatives (`hmm`, `poisson`, `random_walk`, `correlated_walk`), which ignore the odour, each fail the chemotaxis family;
- the blind worm fitted against v2's D fails the wild type at 3 of 5;
- C.1's v1 chosen answer fails the wild type at 3 of 5.

Otherwise it SHALL be VOID.

#### Scenario: The known exploit still passes

- **WHEN** C.1's v1 answer passes v2's wild type
- **THEN** B v2 SHALL be VOID

#### Scenario: One unlucky N2 unit

- **WHEN** one of the four N2 units fails and the other three pass
- **THEN** the N2 control SHALL pass

#### Scenario: A real-rate point worm cannot reach a statistic

- **WHEN** the reachability worm, through the measurement model, fails a speed statistic in more than 10% of 40 populations, or its mean z lies outside [−1, 1]
- **THEN** the run SHALL stop before Part B

### Requirement: The new statistics are validated on unread data

The new statistics SHALL be validated on N2's held-out dates that no reading has touched, against N2 bands from N2's band dates, read once under a guard. PD1074's held-out cohorts SHALL NOT be read again.

#### Scenario: A held-out PD1074 cohort is requested

- **WHEN** any v2 stage would read a PD1074 held-out date
- **THEN** it SHALL refuse

### Requirement: W's minimum v2 is recorded on both family sets

B v2 SHALL record W's minimum on the fitted families (small lawn and kinematics, between halves of N2's unread held-out dates) and on the held-out families (local search, between halves of López-Cruz's band batches; chemotaxis, between every 3-against-3 split of Chen's recordings). W SHALL read its arms on the same family set as the minimum it cites.

#### Scenario: The families agree

- **WHEN** W registers
- **THEN** its distance SHALL be over the same family set as its minimum
