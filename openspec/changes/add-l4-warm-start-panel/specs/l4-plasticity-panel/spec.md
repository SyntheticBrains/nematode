## ADDED Requirements

### Requirement: The warm-start panel starts every arm from a cloned competent policy

The warm-start panel SHALL train the MLP-PPO champion configuration on seeds 1–8 at 6000
episodes, select the seed with the highest committed plateau tail as the one teacher, record it
frozen for 300 episodes at seed 101 with its action means, and clone its policy into every
student at its run seed under two parameter sets — the plastic set (chemical weights behind the
anatomical readout) and the full set (every parameter PPO trains except the noise) — with
cloning hyperparameters fixed before any run (300 epochs, learning rate 1e-3, batch 256, holdout
0.2) and every clone's fit recorded. Twelve arms SHALL run on paired seeds 1–8: from the
plastic-set clone the frozen, unmodulated-Hebbian and three-factor arms on both wirings; from the
full-set clone the frozen arm and the PPO arm on both wirings; and PPO from random weights on both
wirings. Frozen arms SHALL run 600 episodes, plastic and Hebbian arms 2000, PPO arms 3000, each
with the single registered extension (a fresh run at 1.5× replacing the shorter log) for a run
the plateau detector marks non-converged. No pilot SHALL precede the panel. The per-seed ranked
metric SHALL be the committed plateau-tail full-clear success. The confirmatory family SHALL be
exactly six one-sided paired tests corrected together at α = 0.05: (W1) the wild-type plastic-set
frozen clone over panel 2's random-initialisation wild-type frozen floor on the same seeds;
(W2) the wild-type over the rewired plastic-set frozen clone; (W3) the wild-type over the
rewired three-factor arm from the plastic-set clone, the primary; (W4) that arm over its frozen
clone; (W5) that arm over its Hebbian clone; (W6) the wild-type PPO arm from the full-set clone
over wild-type PPO from random weights. The verdict SHALL be assigned in order as
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
