## ADDED Requirements

### Requirement: Panel 2 tests the Hebbian wiring contrast and measures the prior over policies

The second L4 panel SHALL compare eight arms on the frozen C3 cell under the recipe panel 1
pinned: wiring {wild-type, rewired-null} × initialisation {degree-scaled, count-scaled} ×
rule {frozen, unmodulated Hebbian}. The four Hebbian arms SHALL run on paired seeds 1–16 at a
uniform 1000 episodes, seeds 1–8 reproducing panel 1's Hebbian floors bit for bit; the four
frozen arms SHALL run on paired seeds 1–64 at 600 episodes as a prior sweep. No pilot SHALL
precede the panel: every value the arms run with is panel 1's registered pin. The per-seed ranked
metric SHALL be the committed plateau-tail full-clear success. The confirmatory family SHALL be
exactly four one-sided paired tests corrected together at α = 0.05: (P1) wild-type Hebbian over
rewired Hebbian under degree-scaled initialisation on seeds 1–16, the primary; (P2) the same
under count-scaled initialisation; (P3) count-scaled over degree-scaled on the wild-type Hebbian
arm; (P4) wild-type frozen over rewired frozen on seeds 1–64. The verdict SHALL be assigned from
P1 alone as `insufficient_seeds`, `wiring_specific`, `rewired_beats_wild_type`,
`degree_statistics` or `inconclusive`, in that order and by the same rules as the earlier panel's
map; P2–P4 SHALL annotate the verdict and never change it. The harness SHALL report, for every
arm, the distribution of plateau tails and the competent fraction — the share of seeds whose
plateau tail is at least 20% — and the learning gain of each Hebbian arm over its own frozen arm.
A launch record SHALL be committed before the campaigns run, and a Hebbian seed without a
plateau at 1000 episodes SHALL receive exactly one extension, a fresh run at 1500 replacing the
shorter log.

#### Scenario: Seed ranges are enforced per arm

- **WHEN** the panel-2 harness reads a campaign
- **THEN** a Hebbian-arm log with a seed outside 1–16, or a frozen-arm log with a seed outside 1–64,
  SHALL be rejected in confirmatory mode

#### Scenario: The family has exactly four members and the verdict follows P1

- **WHEN** the harness applies multiple-comparisons correction
- **THEN** exactly P1–P4 SHALL be corrected together
- **AND** the verdict SHALL be `wiring_specific` when P1 passes, `rewired_beats_wild_type` when its
  interval lies entirely below zero, `degree_statistics` when it spans zero, and `inconclusive`
  otherwise, with `insufficient_seeds` taking precedence
- **AND** P2, P3 and P4 SHALL be recorded as annotations that leave the verdict unchanged

#### Scenario: The prior sweep reports distributions and competent fractions

- **GIVEN** the four frozen arms over seeds 1–64
- **WHEN** the harness analyses them
- **THEN** it SHALL report each arm's per-seed plateau tails and the fraction at or above 20%
- **AND** P4 SHALL be the paired wild-type-over-rewired test on those seeds under degree-scaled
  initialisation

#### Scenario: Panel 1's floors are reproduced

- **WHEN** the Hebbian and frozen arms under degree-scaled initialisation run on seeds 1–8
- **THEN** their plateau tails SHALL equal panel 1's values for the same arms and seeds

#### Scenario: The launch precedes nothing and the extension is bounded

- **WHEN** the campaigns are launched
- **THEN** a launch record naming the commit, commands, seeds and budgets SHALL already be committed
- **AND** a Hebbian seed still without a plateau at 1000 SHALL be re-run once at 1500 and its log
  SHALL replace the shorter one
