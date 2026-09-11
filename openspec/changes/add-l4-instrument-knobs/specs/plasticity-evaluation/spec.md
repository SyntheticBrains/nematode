## ADDED Requirements

### Requirement: A pinned setting is examined where the rule is known to learn

A setting pinned into a registered recipe SHALL be examined on a platform where the rule under test
has been shown to learn, with the pinned value as the baseline arm and the registered pass rule
unchanged. Where the rule does not learn on a platform, that platform SHALL NOT be used to examine a
setting, since an arm that does not learn cannot show a setting holding it back.

Where a setting cannot act on the available platform, the examination SHALL NOT be recorded as a
null result. The platform SHALL be extended so the setting can act, or the setting SHALL be recorded
as unexamined with the reason.

#### Scenario: A platform that does not learn is not used

- **GIVEN** a platform on which the rule under test does not exceed its own frozen control
- **WHEN** a pinned setting is to be examined
- **THEN** that platform SHALL NOT be used, and the reason SHALL be recorded

#### Scenario: A setting that cannot act is not reported as having no effect

- **GIVEN** a control whose protocol prevents a setting from changing any outcome
- **WHEN** that setting is varied
- **THEN** the result SHALL NOT be recorded as the setting having no effect, and the record SHALL
  state that the setting was unmeasurable on that control

### Requirement: The control may delay the reward to make the eligibility horizon measurable

The positive control SHALL support a delay between the scored action and the reward, so that a rule
whose eligibility decays over time can be tested on its ability to credit an action taken earlier.
At a delay of `D`, the trial SHALL present the cue, take the scored action, run `D` further steps
against a neutral observation, and deliver the reward once at the end.

The delay SHALL NOT change what is scored: one action per trial, taken while the cue is visible,
scored by the registered reward. The cue SHALL NOT be visible during the intervening steps, and the
network SHALL NOT be required to retain it, so that the arm measures credit over time rather than
memory.

The closed-form bounds SHALL be unchanged by the delay, since they depend on the targets and the
action noise alone; the floor, the optimum, the gap and the registered pass rule SHALL therefore
apply to a delayed arm as they do to the committed one.

A delay of zero SHALL reproduce the committed one-step control exactly.

#### Scenario: Zero delay is the committed control

- **GIVEN** the control at a delay of zero
- **WHEN** an arm is run at a registered seed
- **THEN** its score SHALL equal the committed one-step arm's at that seed

#### Scenario: The bounds do not move with the delay

- **GIVEN** any delay
- **WHEN** the cue-blind floor and the optimum are computed
- **THEN** they SHALL equal the committed one-step values

#### Scenario: The delay tests credit rather than memory

- **GIVEN** a delayed trial
- **WHEN** the intervening steps are run
- **THEN** the observation SHALL carry no information about the cue, and the action scored SHALL be
  the one taken while the cue was visible

#### Scenario: The horizon is examined across the scale the setting implies

- **WHEN** the eligibility horizon is examined
- **THEN** the delays SHALL reach beyond the number of steps over which the pinned decay retains a
  substantial fraction of the trace, so that a horizon-limited rule is seen to fail within the grid
