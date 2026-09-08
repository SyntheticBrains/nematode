## ADDED Requirements

### Requirement: The sign-grounding test asks whether real signs change the earlier answers

The sign-grounding test SHALL re-run two committed protocols with atlas-grounded synapse signs and
compare them against the committed random-sign values rather than re-running those arms. The
**prior sweep** SHALL run the frozen arms on both wirings under grounded signs on paired seeds
1–64 at 600 episodes, panel 2's protocol; enforcement SHALL NOT be run on a frozen arm. The
**Hebbian contrast** SHALL run the unmodulated-Hebbian arms on both wirings under grounded signs,
once with sign enforcement off and once with it on, on paired seeds 1–16 at 1000 episodes, panel
2's protocol, each with the single registered extension at 1.5× for a run the committed plateau
detector marks non-converged. The per-seed ranked metric SHALL be the committed plateau-tail
full-clear success and the comparator SHALL be panel 2's committed per-seed table. No pilot SHALL
precede the test.

The confirmatory family SHALL be exactly four one-sided paired tests corrected together at
α = 0.05: (G1) the grounded wild-type frozen arm over panel 2's random-sign wild-type frozen arm
on seeds 1–64; (G2) the grounded wild-type over the grounded rewired-null Hebbian arm without
enforcement, the primary; (G3) the same with enforcement; (G4) the enforced over the unenforced
wild-type Hebbian arm. The verdict SHALL be assigned in order as `insufficient_seeds`,
`substrate_fail` when the grounded frozen arms' competent fraction falls below a fifth of the
committed random-sign value, and then from G2 alone as `specific_wiring`,
`rewired_beats_wild_type`, `degree_statistics` or `inconclusive`; G1, G3 and G4 SHALL annotate the
verdict and never change it. A launch record SHALL be committed before any run.

#### Scenario: Grounded arms are compared against the committed table

- **WHEN** the harness computes G1
- **THEN** the comparator SHALL be panel 2's committed per-seed values for the same arm and seeds,
  read from its published table, and no random-sign arm SHALL be re-run

#### Scenario: The family has four members and the verdict follows the order

- **WHEN** the harness applies multiple-comparisons correction
- **THEN** exactly G1–G4 SHALL be corrected together
- **AND** `substrate_fail` SHALL take precedence over every outcome but `insufficient_seeds`
- **AND** the remaining verdict SHALL follow G2 alone, with G1, G3 and G4 recorded as annotations

#### Scenario: A broken substrate is named, not rationalised

- **GIVEN** grounded frozen arms whose competent fraction is below a fifth of the committed
  random-sign value
- **WHEN** the verdict is assigned
- **THEN** it SHALL be `substrate_fail` and the record SHALL state that nothing downstream of it is
  interpretable

#### Scenario: Enforcement is reported against the flip rate it removes

- **WHEN** the test reports its descriptive layer
- **THEN** it SHALL include the fraction of synapses whose sign changed under the unenforced arms
  and confirm that it is zero for grounded synapses under the enforced arms
