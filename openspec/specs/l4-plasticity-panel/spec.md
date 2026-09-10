# l4-plasticity-panel Specification

## Purpose

This capability is the pre-registered protocol of the L4 panel: the experiment that asks whether the
wild-type *C. elegans* wiring is load-bearing under a biologically plausible three-factor learning
rule, by comparing the plastic wild-type connectome against its plastic degree-preserving rewired-null
on paired seeds, with frozen-weights and unmodulated-Hebbian sanity floors on both wirings and a
matched-rule MLP as the ranking yardstick.

Its value is the order of operations it fixes. The arms, seeds, ranked metric, confirmatory family,
band test and verdict map are registered before any panel data exist; a pilot on disjoint seeds pins
the recipe and the budget by rules stated in advance; a launch record is committed before the panel
runs; one extension and one sensitivity pass are defined and bounded. The harness that computes the
result reuses the project's committed plateau-tail metric and paired-seed statistics layer, and every
artefact the logbook cites is promoted to the supporting directory.

## Requirements

### Requirement: The panel's arms, seeds and metric are fixed in advance

The L4 panel SHALL compare exactly seven arms on the frozen C3 cell: the wild-type connectome and
its degree-preserving rewired-null, each under the frozen, unmodulated-Hebbian and three-factor
rules, plus the matched-rule MLP under the three-factor rule. Every arm SHALL run the same paired
seed list (seeds 1–8), with the rewired arms' `rewire_seed` derived from the run seed so wild-type
and rewired arms pair seed for seed. The per-seed ranked metric SHALL be the committed plateau-tail
(final-quarter) full-clear success rate, and the per-seed convergence verdict SHALL be the
level-agnostic plateau detector's, read from the run's experiment record. Every arm SHALL run with
both scaling switches of the three-factor rule on, so the rate means the same root-mean-square
step per unit modulator on every arm. The two rewired floors SHALL each be a one-key (`wiring`)
delta from their wild-type parent config, and SHALL share the plastic rewired-null arm's wiring
seed for seed, with `rewire_seed` derived from the run seed.

#### Scenario: Each rewired floor is a one-key delta from its wild-type parent

- **GIVEN** the rewired frozen floor and the rewired Hebbian floor configs
- **WHEN** each is compared to its wild-type parent
- **THEN** the only key that differs SHALL be `brain.config.wiring`
- **AND** each SHALL load with its parent's learning rule, freeze flag, strict chemical mask and
  activity traces, and with `rewire_seed` unset
- **AND** the parent's file name SHALL remain a prefix of the derived name

#### Scenario: The rewired floors share the rewired plastic arm's wiring

- **GIVEN** the rewired frozen floor and the plastic rewired-null arm built at one seed
- **WHEN** their chemical masks are compared
- **THEN** the masks SHALL be identical
- **AND** the same SHALL hold for the rewired Hebbian floor

#### Scenario: The harness identifies arms and seeds from a campaign directory

- **GIVEN** a campaign directory whose per-run logs are named by config stem and seed
- **WHEN** the panel harness reads it
- **THEN** every registered config stem SHALL map to its arm key and every seed SHALL be parsed from
  the label
- **AND** a log whose stem is not in the registry SHALL be skipped with a warning, never guessed
- **AND** in confirmatory mode a log whose seed is outside 1–8 SHALL be rejected

#### Scenario: A missing experiment record reports convergence unknown

- **GIVEN** a run whose log is present but whose experiment JSON cannot be located from the logged
  experiment id
- **WHEN** the harness scores that seed
- **THEN** the plateau-tail metric SHALL still be computed from the log
- **AND** the seed's convergence SHALL be reported as unknown, never as converged

### Requirement: The confirmatory family, the band test and the verdict map are pre-registered

The panel SHALL evaluate exactly four confirmatory one-sided paired-seed tests on the ranked metric,
corrected together as one Benjamini-Hochberg family at α = 0.05: (T1) plastic wild-type over plastic
rewired-null; (T2) plastic wild-type over frozen wild-type; (T3) plastic wild-type over Hebbian
wild-type; (T4) wild-type learning gain (plastic minus frozen) over rewired-null learning gain. A
test passes only when its q-value is below 0.05 and its mean delta is positive. The matched-rule MLP
band test SHALL pass when the 80% bootstrap confidence interval of the paired delta (plastic
wild-type minus plastic MLP) contains or lies above zero, and fail when it lies entirely below zero; the harness SHALL report the band delta's mean and
interval width beside the outcome.
A significant reverse result on T1 SHALL be detected by its interval lying entirely below zero and
reported as its own outcome. The verdict SHALL be one of `insufficient_seeds`, `sanity_floor_fail`,
`rewired_beats_wild_type`, `recovery`, `structure_only`, `robustness` or `inconclusive`, assigned in
that order from the family results as the design records: `insufficient_seeds` first, when any of
T1, T2 or T3 has fewer than two common seeds, or when the floors and T1 pass but the band test has
fewer than two common seeds and so cannot separate `recovery` from `structure_only`. All other pairwise deltas SHALL be
reported descriptively, uncorrected, and labelled as such. The harness SHALL report, for T1 and T4,
how many paired seeds have a positive delta.

#### Scenario: The family has exactly four members

- **WHEN** the harness applies multiple-comparisons correction
- **THEN** exactly the four registered p-values SHALL be corrected together
- **AND** the 21 pairwise descriptive deltas SHALL carry no q-value and SHALL be labelled descriptive
  in the output

#### Scenario: Floors decide first

- **GIVEN** panel results in which T2 or T3 does not pass
- **WHEN** the verdict is assigned
- **THEN** it SHALL be `sanity_floor_fail` regardless of T1 or the band test

#### Scenario: The primary contrast and the band test separate recovery from structure-only

- **GIVEN** panel results in which T2 and T3 pass and T1 passes
- **WHEN** the band test passes
- **THEN** the verdict SHALL be `recovery`
- **WHEN** the band test fails
- **THEN** the verdict SHALL be `structure_only`

#### Scenario: A null primary contrast is the robustness branch

- **GIVEN** panel results in which T2 and T3 pass, T1 does not pass, and T1's interval spans zero
- **WHEN** the verdict is assigned
- **THEN** it SHALL be `robustness`
- **AND** an interval lying entirely below zero SHALL instead give `rewired_beats_wild_type`
- **AND** an interval that neither spans zero nor lies entirely below it SHALL give `inconclusive`

#### Scenario: The gain contrast annotates but never overrides

- **GIVEN** any panel result
- **WHEN** T4 disagrees with T1
- **THEN** the output SHALL record the disagreement alongside the verdict
- **AND** the verdict SHALL be unchanged by T4

### Requirement: The pilot pins the shared recipe and the budget by pre-registered rules

Before the panel runs, a pilot SHALL run the rule-bearing arms at `plasticity_rate` values of 3e-4, 1e-3
and 3e-3 and the frozen arms once, on pilot seeds 101 and 102 only, at 3000 episodes, with one
extension to 6000 for any three-factor arm not converged on either seed at the selected rate; if that arm still has no plateau at 6000, the budget SHALL be pinned at 6000
and the arm flagged. Every extension, pilot or panel, SHALL be a fresh run at the longer budget at
the same seed, whose log replaces the shorter run's. The
selected rate SHALL be the one maximising the pooled mean plateau-tail success of the three
three-factor arms over the pilot seeds, ties to the default 1e-3, and SHALL be written explicitly into
every plastic-family panel config. The panel's uniform budget SHALL be the smallest multiple of 500
episodes at or above 1.25 times the latest convergence onset among converged pilot runs at the
selected rate, and never below 2000. Both values SHALL be recorded in the change's design by dated
amendment, with the pilot summary committed, before the panel launches. Nothing else in the
registered protocol SHALL change after the pilot.

#### Scenario: The pilot selects the rate on a pooled criterion

- **GIVEN** pilot plateau-tail results for the three three-factor arms at each grid rate
- **WHEN** the pilot summary selects the recipe
- **THEN** it SHALL choose the rate whose pooled mean across those three arms and both pilot seeds is
  highest
- **AND** on a tie it SHALL choose 1e-3

#### Scenario: The budget rule rounds up from the slowest converger

- **GIVEN** converged pilot runs at the selected rate whose latest convergence onset is N episodes
- **WHEN** the budget is computed
- **THEN** it SHALL be the smallest multiple of 500 that is at least 1.25 × N
- **AND** it SHALL be at least 2000 even when 1.25 × N is smaller

#### Scenario: Pilot seeds never enter the panel

- **WHEN** the panel campaign is planned
- **THEN** its seed list SHALL be exactly 1–8
- **AND** no pilot run SHALL contribute to any confirmatory test

### Requirement: The launch is recorded before results are read, and one sensitivity pass is defined

Before the panel command runs, the commit SHA, exact command, seed list, budget and recipe SHALL be
committed to the supporting directory. Any seed still climbing at the pinned budget SHALL receive
exactly one extension to 1.5 times the budget; a seed still without a plateau SHALL be flagged and
ranked on its plateau-tail. If the verdict is `robustness`, exactly one sensitivity pass SHALL be
run: the primary pair on the panel seeds at the two unselected grid rates, reported descriptively
and never altering the verdict. No other re-run SHALL occur under this protocol.

#### Scenario: The launch record precedes the panel

- **WHEN** the panel campaign is launched
- **THEN** a launch record naming the commit, command, seeds, budget and recipe SHALL already be
  committed

#### Scenario: The sensitivity pass cannot change the verdict

- **GIVEN** a `robustness` verdict and the sensitivity pass's recomputed T1 at the two other rates
- **WHEN** the harness reports
- **THEN** the verdict SHALL remain `robustness`
- **AND** the sensitivity results SHALL be labelled descriptive

### Requirement: Panel data persist beside the logbook that will cite them

The harness output, the per-seed table, the learning curves, the manifest, the pilot summary and the
launch record SHALL be promoted to `docs/experiments/logbooks/supporting/040-l4-panel/` so the
logbook that follows references only permanent repository paths.

#### Scenario: Every artefact the logbook will need is committed under supporting

- **WHEN** the panel analysis is complete
- **THEN** `panel.json`, `per-seed.csv`, `curves.csv`, the manifest, `pilot.json` and `launch.md`
  SHALL exist under the supporting directory
- **AND** none of them SHALL reference a path under a temporary directory

### Requirement: Panel 2 tests the Hebbian wiring contrast and measures the prior over policies

The second L4 panel SHALL compare eight arms on the frozen C3 cell under the recipe panel 1
pinned: wiring {wild-type, rewired-null} × initialisation {degree-scaled, count-scaled} ×
rule {frozen, unmodulated Hebbian}. The four Hebbian arms SHALL run on paired seeds 1–16 at a
uniform 1000 episodes, seeds 1–8 reproducing the first 1000 episodes of panel 1's streams; the four
frozen arms SHALL run on paired seeds 1–64 at 600 episodes as a prior sweep. No pilot SHALL
precede the panel: every value the arms run with is panel 1's registered pin. The per-seed ranked
metric SHALL be the committed plateau-tail full-clear success. The confirmatory family SHALL be
exactly four one-sided paired tests corrected together at α = 0.05: (P1) wild-type Hebbian over
rewired Hebbian under degree-scaled initialisation on seeds 1–16, the primary; (P2) the same
under count-scaled initialisation; (P3) count-scaled over degree-scaled on the wild-type Hebbian
arm; (P4) wild-type frozen over rewired frozen on seeds 1–64. The verdict SHALL be assigned from
P1 alone as `insufficient_seeds`, `specific_wiring`, `rewired_beats_wild_type`,
`degree_statistics` or `inconclusive`, in that order and by the rules of the rewired-null
control's map; P2–P4 SHALL annotate the verdict and never change it. The harness SHALL report, for every
arm, the distribution of plateau tails and the competent fraction — the share of seeds whose
plateau tail is at least 20% — and the learning gain of each Hebbian arm over its own frozen arm.
A launch record SHALL be committed before the campaigns run, and a Hebbian seed the committed
plateau detector marks non-converged at 1000 episodes SHALL receive exactly one extension, a fresh run at 1500 replacing the
shorter log.

#### Scenario: Seed ranges are enforced per arm

- **WHEN** the panel-2 harness reads a campaign
- **THEN** a Hebbian-arm log with a seed outside 1–16, or a frozen-arm log with a seed outside 1–64,
  SHALL be rejected

#### Scenario: The family has exactly four members and the verdict follows P1

- **WHEN** the harness applies multiple-comparisons correction
- **THEN** exactly P1–P4 SHALL be corrected together
- **AND** the verdict SHALL be `specific_wiring` when P1 passes, `rewired_beats_wild_type` when its
  interval lies entirely below zero, `degree_statistics` when it spans zero, and `inconclusive`
  otherwise, with `insufficient_seeds` taking precedence
- **AND** P2, P3 and P4 SHALL be recorded as annotations that leave the verdict unchanged

#### Scenario: The prior sweep reports distributions and competent fractions

- **GIVEN** the four frozen arms over seeds 1–64
- **WHEN** the harness analyses them
- **THEN** it SHALL report each arm's per-seed plateau tails and the fraction at or above 20%
- **AND** P4 SHALL be the paired wild-type-over-rewired test on those seeds under degree-scaled
  initialisation

#### Scenario: Panel 1's episode streams are reproduced

- **WHEN** the Hebbian and frozen arms under degree-scaled initialisation run on seeds 1–8
- **THEN** their per-episode outcomes over the first 1000 (Hebbian) and 600 (frozen) episodes SHALL
  equal the same episodes of panel 1's logs for the same arms and seeds

#### Scenario: The launch precedes nothing and the extension is bounded

- **WHEN** the campaigns are launched
- **THEN** a launch record naming the commit, commands, seeds and budgets SHALL already be committed
- **AND** a Hebbian seed still without a plateau at 1000 SHALL be re-run once at 1500 and its log
  SHALL replace the shorter one

### Requirement: Panel 3 replicates the Hebbian wiring contrast on fresh seeds

The third L4 panel SHALL run the two degree-scaled Hebbian arms of panel 2, unchanged, on paired
seeds 17–64 at 1000 episodes, with no pilot and no other arm. The frozen floors for those seeds
SHALL be panel 2's prior-sweep runs, read by seed from panel 2's committed per-seed table and not
re-run. The per-seed ranked metric SHALL
be the committed plateau-tail full-clear success. The confirmatory family SHALL be exactly two
one-sided paired tests corrected together at α = 0.05: (R1) wild-type Hebbian over rewired-null
Hebbian by the committed paired Wilcoxon, the primary; (R2) wild-type over rewired-null in
competent-fraction discordance, an exact binomial on the discordant pairs at the 20% competent
threshold. The verdict SHALL be assigned from R1 alone as `insufficient_seeds`, `specific_wiring`,
`rewired_beats_wild_type`, `degree_statistics` or `inconclusive`, in that order and by the same
rules as panel 2; R2 SHALL annotate the verdict and never change it. Seeds 1–16 SHALL enter only
a pooled descriptive summary read from panel 2's committed per-seed table, never a confirmatory
test. A launch record SHALL be committed before the campaign runs, and a seed without a detected
plateau at 1000 episodes SHALL receive exactly one extension, a fresh run at 1500 replacing the
shorter log.

#### Scenario: Only the replication seeds are confirmatory

- **WHEN** the panel-3 harness groups a campaign
- **THEN** a Hebbian-arm log with a seed outside 17–64 SHALL be rejected
- **AND** panel 2's seeds 1–16 SHALL appear only in the pooled descriptive summary

#### Scenario: The family has two members and the verdict follows R1

- **WHEN** the harness applies multiple-comparisons correction
- **THEN** exactly R1 and R2 SHALL be corrected together
- **AND** the verdict SHALL follow R1 by panel 2's map
- **AND** R2's outcome SHALL be recorded as an annotation, including the case where R2 passes and
  R1 does not

#### Scenario: The discordance test is exact

- **GIVEN** `b` seeds where only the wild-type arm is competent and `c` where only the rewired arm is
- **WHEN** R2 is computed
- **THEN** its p-value SHALL be the exact binomial `P(X ≥ b)` for `X ~ Bin(b + c, ½)`, and R2 SHALL
  pass only when its corrected q is below α and `b > c`
- **AND** with no discordant pairs R2 SHALL report p = 1.0 and fail

#### Scenario: The floors come from panel 2's sweep via its committed table

- **WHEN** the harness computes learning gains for seeds 17–64
- **THEN** each Hebbian arm's floor SHALL be panel 2's frozen value at the same seed, read from
  panel 2's committed per-seed table, and the record SHALL name that table and the campaign it
  came from
- **AND** each family test SHALL report whether its seed set equals the registered seeds 17–64

#### Scenario: The launch precedes the run and the extension is bounded

- **WHEN** the campaign is launched
- **THEN** a launch record naming the commit, command, seeds and budget SHALL already be committed
- **AND** a seed still without a plateau at 1000 SHALL be re-run once at 1500 and its log SHALL
  replace the shorter one

### Requirement: The warm-start panel starts every arm from a cloned competent policy

The warm-start panel SHALL train the MLP-PPO champion configuration on seeds 1–8 at 6000
episodes, select the seed with the highest committed plateau tail as the one teacher, record it
frozen for 300 episodes at seed 101 with its action means, and clone its policy into every
student at its run seed under two parameter sets — the plastic set (chemical weights behind the
anatomical readout, its student the plastic frozen arm) and the full set (every parameter PPO
trains except the noise, its student a low-noise PPO arm carrying `initial_log_std: -1.0` so the
clone's saved noise matches the plastic arms') — with
cloning hyperparameters fixed before any run (300 epochs, learning rate 1e-3, batch 256, holdout
0.2) and every clone's fit recorded. Twelve arms SHALL run on paired seeds 1–8: from the
plastic-set clone the frozen, unmodulated-Hebbian and three-factor arms on both wirings; from the
full-set clone the frozen arm and the PPO arm on both wirings; and low-noise PPO from random
weights on both
wirings. Frozen arms SHALL run 600 episodes, plastic and Hebbian arms 2000, PPO arms 3000, each
with the single registered extension (a fresh run at 1.5× replacing the shorter log) for a run
the plateau detector marks non-converged. No pilot SHALL precede the panel. The per-seed ranked
metric SHALL be the committed plateau-tail full-clear success. The confirmatory family SHALL be
exactly six one-sided paired tests corrected together at α = 0.05: (W1) the wild-type plastic-set
frozen clone over panel 2's random-initialisation wild-type frozen floor on the same seeds;
(W2) the wild-type over the rewired plastic-set frozen clone; (W3) the wild-type over the
rewired three-factor arm from the plastic-set clone, the primary; (W4) that arm over its frozen
clone; (W5) that arm over its Hebbian clone; (W6) the wild-type PPO arm from the full-set clone
over wild-type low-noise PPO from random weights, both starting at the same noise. The verdict SHALL be assigned in order as
`insufficient_seeds`, `clone_fail` (W1 fails), `sanity_floor_fail` (W4 or W5 fails),
`rewired_beats_wild_type` (W3's interval entirely below zero), `specific_wiring` (W3 passes),
`degree_statistics` (W3's interval spans zero) or `inconclusive`; W2 and W6 SHALL annotate and
never change it, and `rule_destroys_clone` SHALL be recorded when W4's interval lies entirely
below zero. A launch record SHALL be committed before any campaign runs.

#### Scenario: One teacher, selected by the committed metric

- **WHEN** the teacher campaign completes
- **THEN** the teacher SHALL be the seed with the highest plateau tail, its weights copied and its
  frozen plateau tail at the recording seed recorded as the ceiling

#### Scenario: Clones are made at the run seed with fixed hyperparameters

- **WHEN** the students are cloned
- **THEN** each clone SHALL be trained from the student's initial weights at its run seed with the
  registered hyperparameters, and its initial, final and held-out losses SHALL be recorded, with a
  flag on any clone whose held-out loss does not fall below half its initial

#### Scenario: The family has six members and the verdict follows the map

- **WHEN** the harness applies multiple-comparisons correction
- **THEN** exactly W1–W6 SHALL be corrected together
- **AND** the verdict SHALL follow the ordered map with W1 as the gate and W4, W5 as the floors
- **AND** W2, W6 and `rule_destroys_clone` SHALL be recorded as annotations that leave the verdict
  unchanged

#### Scenario: Seeds are enforced

- **WHEN** the harness reads a campaign
- **THEN** a log with a seed outside 1–8 SHALL be rejected

#### Scenario: The launch precedes the run and extensions are bounded

- **WHEN** the campaigns are launched
- **THEN** a launch record naming the commit, commands, seeds and budgets SHALL already be committed
- **AND** a run still without a plateau at its budget SHALL be re-run once at 1.5× and its log SHALL
  replace the shorter one

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
`substrate_fail` when the grounded frozen arms' competent fraction falls below half of the
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

- **GIVEN** grounded frozen arms whose competent fraction is below half of the committed
  random-sign value
- **WHEN** the verdict is assigned
- **THEN** it SHALL be `substrate_fail` and the record SHALL state that nothing downstream of it is
  interpretable

#### Scenario: Enforcement is reported against the flip rate it removes

- **WHEN** the test reports its descriptive layer
- **THEN** it SHALL include the fraction of synapses whose sign changed under the unenforced arms
  and confirm that it is zero for grounded synapses under the enforced arms

### Requirement: A consolidation variant is screened by the clone assay before any panel

Every candidate consolidation mechanism SHALL be screened by the clone assay before the 2×2 panel
is re-run under it. The assay is the protocol registered with the clone-destruction diagnostic and
is not restated with new values here: the wild-type plastic clone arm with the variant's rule keys
and nothing else changed, started from the warm-start panel's plastic-set wild-type clone for that
seed, seeds 1–8 paired, a 2000-episode budget with no extension, the committed plateau-tail
full-clear success metric, and the same seeds' published frozen-clone values as the comparator.
A variant **holds** when its mean is within 5 points of the frozen clone's mean and at least 6 of
8 seeds are no more than 10 points below their own frozen clone; it **improves** when its mean is
above the frozen clone's and at least 6 of 8 seeds are above their own; it **passes** when it
holds or improves.

The assay SHALL be reported as a **screen and not a confirmatory test**. It reuses seeds already
reported, so it SHALL NOT declare a multiple-comparisons family or a verdict map, and the record
SHALL state that a pass licenses running the registered panel and nothing more.

Each variant's screen SHALL report the eight per-seed values, the mean delta against the frozen
clone, the count of seeds at or above their own frozen clone, and the cosine of the endpoint
weights to the clone the run started from, since a variant can pass on behaviour while having
rewritten the policy.

#### Scenario: A variant is screened before the panel

- **GIVEN** a consolidation variant proposed for the panel
- **WHEN** the panel is scheduled
- **THEN** the variant SHALL have a recorded clone-assay result
- **AND** the panel SHALL NOT be run under a variant whose screen did not pass

#### Scenario: The screen is reported as a screen

- **WHEN** a screen result is written up
- **THEN** the record SHALL state that it reuses previously reported seeds and licenses the panel
  only
- **AND** it SHALL NOT assign a panel verdict

#### Scenario: Holding by consolidation is distinguished from holding by not moving

- **WHEN** a variant passes the screen
- **THEN** the record SHALL report the endpoint cosine to the clone and the effective rate
  multiplier alongside the metric

### Requirement: Consolidation hyperparameters are pinned by a pre-declared pilot

Where a mechanism has hyperparameters with no value to inherit, they SHALL be pinned by a pilot
declared before it runs: seeds 1–2 of the same clone arm at the same budget over a grid written
into the launch record, pinning the combination with the highest mean plateau tail across the two
seeds and breaking ties toward the weaker constraint. The pilot's grid, its criterion and its
results SHALL be recorded before the screen is run, and the pilot seeds SHALL be reported with the
screen so that a pin which only worked on its own seeds is visible.

A mechanism whose values are fixed by the comparator rather than chosen SHALL record that instead
of running a pilot.

#### Scenario: The pins are declared before they are used

- **WHEN** a screen is launched
- **THEN** the launch record SHALL already contain the pilot grid, the pinning criterion and the
  pinned values

#### Scenario: The oracle arm is declared as a bound, not a candidate

- **WHEN** the oracle variant is screened
- **THEN** the record SHALL state that it consumes the environment's episode-success flag, that it
  is not a mechanism the animal could host, and that it exists to bound what a quality-gated
  consolidation could achieve
- **AND** the launch record SHALL state, before the run, that with its reference pinned at the
  comparator the arm bounds holding only and cannot register as improving

### Requirement: The decorrelation test asks whether a decorrelating term recovers what grounding cost

The sign-grounding test found that grounding the substrate's synapse signs made reward-free
Hebbian learning substantially worse, and predicted that a rule with an anti-Hebbian or
decorrelating term would recover the loss. That prediction SHALL be tested by re-running the
sign-grounding test's own Hebbian protocol under each decorrelating variant: the wild-type and
rewired-null grounded Hebbian arms, paired seeds 1–16, a 1000-episode budget, the committed
plateau-tail full-clear metric, and the single registered extension of a fresh run at 1.5×
replacing a shorter log the plateau detector marks non-converged. The comparators SHALL be the
sign-grounding test's committed per-seed values on the same seeds 1–16, read from its published
table and paired seed for seed, and no grounded Hebbian arm SHALL be re-run.

The confirmatory family SHALL be exactly four one-sided paired tests corrected together under
BH-FDR: **D1** the wild-type anti-Hebbian arm over the committed wild-type grounded Hebbian
values, **D2** the wild-type Oja arm over the same, **D3** the wild-type over the rewired null
under the anti-Hebbian variant, and **D4** the same under the Oja variant.

The verdict SHALL be assigned in order as `insufficient_seeds`; then **`no_recovery`** when
neither D1 nor D2 confirms; then `recovery_specific` (D1 only), `recovery_general` (D2 only) or
`recovery_both`. D3 and D4 SHALL annotate the verdict and never change it. A launch record SHALL
be committed before any run.

Two annotations SHALL be computed and reported without changing the verdict: `full_recovery`,
true when the 80% bootstrap interval of a recovered arm's mean plateau tail over seeds 1–16
includes or exceeds the committed random-sign mean for the same wiring,
which separates a term that helps from one that restores what grounding cost; and the mean
decorrelation share from the rule's telemetry, so that a recovery whose share is near zero is
recorded as attributable to something other than the term.

#### Scenario: The prediction can fail

- **GIVEN** neither D1 nor D2 confirming
- **WHEN** the verdict is assigned
- **THEN** it SHALL be `no_recovery`
- **AND** the record SHALL state that the missing inhibitory brake was not what limited the rule

#### Scenario: Helping and restoring are distinguished

- **WHEN** a variant's arm confirms its recovery test
- **THEN** the record SHALL report whether the 80% bootstrap interval of its mean plateau tail
  includes or exceeds the committed random-sign mean for the same wiring

#### Scenario: A recovery is attributed to the term or not

- **WHEN** a variant's arm confirms its recovery test
- **THEN** the record SHALL report that arm's mean decorrelation share

#### Scenario: The wiring contrast annotates and never decides

- **WHEN** D3 or D4 confirms or fails
- **THEN** the verdict SHALL be unchanged by it

#### Scenario: The panel stays gated

- **WHEN** a variant recovers
- **THEN** the 2×2 panel SHALL still require a passing clone assay before it is re-run

### Requirement: The structured-instruction test asks whether routing the third factor matters

The routed third factor SHALL be tested against the global scalar on both wirings: four arms —
wild-type and rewired-null under each routing mode — on paired seeds 1–16, the panel's 3000-episode
plastic budget, the committed plateau-tail full-clear metric, and the panel's registered extension
of a fresh run at 1.5× replacing a shorter log the plateau detector marks non-converged.

The global-scalar arms SHALL be **re-run concurrently** rather than read from the first panel's
committed table, whose plastic arm carries eight seeds at this budget; the committed values SHALL
be reported beside the new global arm as a consistency check and SHALL NOT be used as a comparator.

The confirmatory family SHALL be exactly four one-sided paired tests corrected together under
BH-FDR: **S1** the wild-type pathway arm over the wild-type global arm, **S2** the rewired-null
pathway arm over the rewired-null global arm, **S3** wild-type over rewired null under pathway
routing, and **S4** the same under the global scalar.

The verdict SHALL be assigned in order as `insufficient_seeds`; then **`no_routing_effect`** when
neither S1 nor S2 confirms; then `routing_helps_both`, `routing_helps_wild_type_only` and
`routing_helps_rewired_only`. S3 and S4 SHALL annotate the verdict and never change it. A launch
record SHALL be committed before any run.

The record SHALL report **each arm's** instructed fraction — the rewired null derives its own
pathway from its own edges — and each arm's mean instructed share of the update, and SHALL state
that the pathway is a model of aminergic reach by synaptic connectivity — a lower bound, since
aminergic transmission in this animal is substantially extrasynaptic — so that a negative is
recorded as refuting this proxy rather than structured instruction itself.

#### Scenario: Routing changing nothing is a nameable outcome

- **GIVEN** neither S1 nor S2 confirming
- **WHEN** the verdict is assigned
- **THEN** it SHALL be `no_routing_effect`
- **AND** the record SHALL state that the global scalar was not the limitation, and that the
  pathway tested was the synaptic proxy rather than expressed-receptor reach

#### Scenario: Helping the scramble as much as the animal is not a win

- **GIVEN** both S1 and S2 confirming
- **WHEN** the result is written up
- **THEN** the record SHALL state that a routed third factor helping the rewired null as much as
  the wild type is a fact about running two learning regimes in one network and not about this
  animal's connectivity

#### Scenario: The proxy's coverage is reported with the result

- **WHEN** the test is written up
- **THEN** each arm's instructed fraction SHALL be reported
- **AND** each arm's mean instructed share of the update SHALL be reported

#### Scenario: The panel stays gated

- **WHEN** routing confirms
- **THEN** the 2×2 panel SHALL still require a passing clone assay before it is re-run

### Requirement: An eligibility variant clears the clone assay before any panel arm

A variant of the three-factor rule that has passed the rule's positive control SHALL be run
through the clone assay before any arm of it enters a registered panel. The assay SHALL be the one
registered with the clone-destruction diagnostic, unchanged in arms, comparator, budget, metric and
pass rule, so that the variant's result is directly comparable with the mechanisms already screened
by it.

The variant SHALL be run at the parameter value its positive control pinned. That value SHALL NOT
be re-tuned against the assay's outcome: a gate whose parameter is chosen by its own result is a
search, and the record SHALL state the value it was run at.

The result SHALL be reported with the endpoint cosine to the clone the run started from, since a
variant can hold the metric while having rewritten the policy underneath it.

A variant whose mechanism perturbs the substrate SHALL additionally be reported with **a trajectory
annotation** — its plateau tail over the final quarter of the run against the first quarter — and
against **a frozen control**: the same arm with updates frozen and the perturbation applied. A
perturbing mechanism costs a competent policy something before any question of retention arises,
and without these two a fail cannot be attributed to the rule rather than to the exploration.
Neither SHALL change the verdict, which remains whatever the registered pass rule gives.

#### Scenario: The assay is unchanged

- **WHEN** an eligibility variant is screened
- **THEN** the arms, comparator, budget, metric and pass rule SHALL be those registered with the
  clone-destruction diagnostic

#### Scenario: The parameter comes from the control

- **WHEN** the variant is run
- **THEN** it SHALL use the parameter value its positive control pinned
- **AND** the record SHALL state that value

#### Scenario: A perturbing variant is read against its own frozen control

- **GIVEN** a variant whose mechanism perturbs the substrate
- **WHEN** its assay result is recorded
- **THEN** the frozen control's result and the trajectory annotation SHALL be reported beside it
- **AND** neither SHALL change the verdict the registered pass rule gives

#### Scenario: Failing the assay does not stop at the metric

- **GIVEN** a variant that does not pass
- **WHEN** the result is recorded
- **THEN** the record SHALL state that the variant learns from random weights without holding a
  competent policy, and that a panel remains gated
