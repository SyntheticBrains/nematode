# associative-memory-probe Specification

## Purpose

A naturalistic chemosensory associative-memory probe: the biological "remember-and-use" twin of the
bit-memory positive control. Two cues are conditioned early in a trial (one paired with a positive
outcome); with probability `reversal_prob` a reversal block re-presents them with flipped outcomes;
after a delay the agent gives a binary readout of the CURRENT rewarded cue. The observation carries no
external memory aid (no STAM, no gradients), so the cue↔outcome association must be held — and updated
on reversal — in internal recurrent state. Because at `reversal_prob` 0.5 a hold-only policy is at
chance, above-chance accuracy requires genuine working-memory update; a memoryless policy is pinned at
chance, so the task separates recurrent/attention architectures from memoryless ones on the
working-memory-update axis.

## Requirements

### Requirement: Delayed-associative-match task structure

The environment SHALL support a config-gated **delayed-associative-match** task, disabled by default,
in which each trial progresses through a **conditioning** phase, an **optional reversal** phase, a
**delay** phase, and a **response** phase in order. In the conditioning phase two cues SHALL be
presented in sequence, each with an outcome (valence), with exactly one cue rewarded; the rewarded cue
and the presentation order SHALL be sampled uniformly per trial. Episodes SHALL comprise multiple
trials, so the association cannot be encoded in the policy weights and must be retained (and, on
reversal trials, updated) across the delay.

#### Scenario: Phases progress in order within a trial

- **WHEN** a trial runs with the task enabled
- **THEN** the phase SHALL advance conditioning → (reversal, on reversal trials) → delay → response, spending the configured number of steps in each, then begin the next trial

#### Scenario: The rewarded cue is sampled per trial and independent

- **WHEN** successive trials run
- **THEN** each trial's rewarded cue, presentation order, and whether a reversal occurs SHALL be sampled uniformly and independently, so no fixed policy exceeds chance

### Requirement: Cue-identity, outcome, and go observation channels

The task observation SHALL expose a **cue-identity** channel (a signed identity for the cue shown at
the current conditioning step), an **outcome (valence)** channel (positive for the rewarded cue,
non-positive otherwise), and a **go-signal** channel marking the response phase. The cue-identity and
outcome channels SHALL be non-zero only during the conditioning phase; the go-signal SHALL be asserted
only during the response phase.

#### Scenario: Cue-identity and outcome are present only during conditioning

- **WHEN** the task is in the delay or response phase
- **THEN** the cue-identity and outcome channels SHALL be zero (the association is not re-presented)

#### Scenario: Each conditioning step exposes one cue with its outcome

- **WHEN** the conditioning phase presents a cue
- **THEN** the cue-identity channel SHALL carry that cue's identity and the outcome channel its valence, so the agent can bind identity to outcome

#### Scenario: Go-signal marks the response phase

- **WHEN** the task enters the response phase
- **THEN** the go-signal channel SHALL be asserted (and de-asserted in the conditioning and delay phases)

### Requirement: Within-trial association reversal (working-memory update)

On a config-gated fraction of trials the task SHALL present a **reversal** block after conditioning that
re-presents the two cues with **flipped** outcomes (reusing the cue-identity + outcome channels; there
SHALL be no separate reversal-signal channel). On a reversal trial the trial's **current** rewarded cue
SHALL be the post-reversal one; on a non-reversal trial it SHALL remain the conditioning-phase rewarded
cue. Because reversal is probabilistic and the initial rewarded cue randomised, a policy that only
retains the initial association without updating it SHALL be at chance on the reversal fraction.

#### Scenario: A reversal flips the current rewarded cue

- **WHEN** a trial includes a reversal block
- **THEN** the current rewarded cue SHALL be the cue carrying the positive outcome in the reversal block, and the response SHALL be scored against it

#### Scenario: Non-reversal trials keep the original association

- **WHEN** a trial does not include a reversal block
- **THEN** the current rewarded cue SHALL remain the conditioning-phase rewarded cue

#### Scenario: A hold-only policy cannot exceed chance on reversal trials

- **WHEN** a policy retains the initial association but does not update it on the reversal evidence
- **THEN** it SHALL be at chance across the reversal fraction of trials — only a policy that updates the held association on the reversal evidence can be correct on them

### Requirement: No external memory aids in the task observation

When the task is enabled, the assembled observation SHALL contain **only** the task channels
(cue-identity, outcome, go-signal) with **no** short-term-memory buffer (STAM) and **no** gradient /
field sensing, so that only a policy's internal recurrent state can carry the association across the
delay. A configuration that enables the task with any such aid present SHALL fail loudly at
config-resolve time.

#### Scenario: A memoryless policy cannot recover the association at response time

- **WHEN** the task is enabled and the observation is assembled at the response step
- **THEN** it SHALL contain no channel from which the conditioning-phase association can be re-derived (no STAM, no re-presented cue/outcome), so a policy without internal state is at chance

#### Scenario: A leaked memory aid fails loudly

- **WHEN** a configuration enables the task but resolves to include STAM or gradient sensing
- **THEN** config resolution SHALL raise rather than silently run an invalid control

### Requirement: Association-conditioned response reward

A reward SHALL be granted **only** on response steps, according to whether the agent's binary response
matches the trial's **current** rewarded cue (post-reversal if the trial reversed, else the original).
The initial rewarded cue and the reversal SHALL be sampled uniformly per trial so a chance policy
scores 50%. No task reward SHALL fire in the conditioning, reversal, or delay phases.

#### Scenario: Correct response is rewarded

- **WHEN** the agent's response matches the trial's current rewarded cue
- **THEN** the configured correct reward SHALL be granted for that trial

#### Scenario: Incorrect response is not rewarded

- **WHEN** the agent's response does not match the trial's current rewarded cue
- **THEN** the correct reward SHALL NOT be granted (the configured wrong-response outcome applies)

#### Scenario: No task reward outside the response phase

- **WHEN** the task is in the conditioning, reversal, or delay phase
- **THEN** no association reward SHALL be granted that step

### Requirement: Binary action readout of the remembered rewarded cue

On the response step the agent's action SHALL be interpreted as a **binary** readout of the remembered
rewarded cue-identity (the sign of the continuous turn output, or the discrete LEFT/RIGHT action), and
SHALL be consumed as the response only — not as locomotion.

#### Scenario: Continuous turn sign selects the response

- **WHEN** a continuous-action arm emits its action on a response step
- **THEN** the sign of the turn component SHALL select the responded cue-identity, compared against the rewarded cue

### Requirement: Config-gated, off by default

The task SHALL be disabled by default. When disabled, the environment, observation pipeline, reward
calculation, and configuration schema SHALL behave exactly as before (byte-identical). When enabled,
the task dynamics SHALL replace the foraging/predator/thermal/satiety dynamics for the episode.

#### Scenario: Disabled leaves existing behaviour unchanged

- **WHEN** the task is not enabled
- **THEN** the environment and observation pipeline SHALL be byte-identical to the pre-change behaviour

#### Scenario: Enabling activates only the task dynamics

- **WHEN** the task is enabled
- **THEN** the episode SHALL run the conditioning/delay/response machine with the association reward, and the foraging/predator/thermal/satiety handlers SHALL be inert

### Requirement: Architecture-separation evaluation

The task SHALL be evaluable as an architecture-separation instrument: an evaluation runs the arm panel
over paired seeds and reports each arm's plateau-tail response accuracy, testing whether the
recurrent/attention arms clear both chance and the memoryless MLP, using the committed paired-seed
statistics layer. A null result SHALL be reported as such.

#### Scenario: Separation is reported when memory arms beat a memoryless arm at chance

- **WHEN** the panel is evaluated over paired seeds
- **THEN** the evaluation SHALL report per-arm response accuracy and the pairwise BH-FDR-adjusted deltas, and a separation verdict when the recurrent/attention arms significantly exceed the memoryless MLP (which stays near chance)

#### Scenario: A null result is reported as such

- **WHEN** the recurrent/attention arms do not significantly exceed the memoryless MLP
- **THEN** the evaluation SHALL report a null verdict rather than a separation
