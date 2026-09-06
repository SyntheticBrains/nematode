## MODIFIED Requirements

### Requirement: Substrate-invariant scaling of the three-factor update

The three-factor rule SHALL offer two independent, default-off scaling modes shared by every
plastic brain through the plasticity configuration mixin. With **modulator normalisation** on,
the third factor SHALL be `tanh(δ / σ) − c`, where `σ` is a running root-mean-square of the raw
prediction error `δ` maintained by exponential moving average at a configurable scale rate,
bias-corrected so its first observation counts fully, used at its pre-update value for the
current step, and floored at a configurable positive floor before division; and `c` is a
bias-corrected running mean of `tanh(δ / σ)` at the same scale rate, zero before any
observation and used at its pre-update value for the current step, so that the modulator is
zero-mean under the agent's own policy as a prediction error must be. With **trace
normalisation** on, the Hebbian term for each plastic tensor SHALL be divided by `ρ`, a running
root-mean-square of that tensor's eligibility trace over its masked entries, maintained,
bias-corrected, used and floored the same way, with an all-zero trace neither updating it nor
counting toward its correction. The decay term and the magnitude clamp SHALL be unchanged by
either mode. The scales and the centre SHALL advance under a freeze and in unmodulated mode,
where the modulator SHALL remain `1.0`. With both modes off, the rule SHALL be bit-identical to
the rule without this requirement. The rule SHALL report the effective modulator, the modulator
scale, the modulator centre and the mean trace scale beside its existing telemetry, and the raw
prediction error SHALL still be reported.

#### Scenario: The modulator is bounded and scale-free

- **GIVEN** modulator normalisation on and a warmed scale `σ`
- **WHEN** the rule steps with prediction error `δ`
- **THEN** the modulator SHALL equal `tanh(δ / σ_prev) − c_prev`, with the scale and the centre
  from before this step
- **AND** it SHALL lie in `[−2, 2]` for any `δ`
- **AND** the centre SHALL be zero before any observation and the bias-corrected running mean
  of `tanh(δ / σ)` after
- **AND** the raw `δ` SHALL still be reported as the prediction error, and the centre SHALL be
  reported beside it

#### Scenario: The modulator is zero-mean under a skewed reward stream

- **GIVEN** modulator normalisation on and a deterministic periodic stream of prediction errors
  with many small values, frequent moderate positives and rare large negatives, whose raw values
  sum to zero over each period
- **WHEN** the rule steps through whole periods past a whole-period warm-up
- **THEN** the mean of the modulator over those steps SHALL be within `0.005` of zero
- **AND** the mean of the uncentred `tanh(δ / σ)` on the same steps SHALL exceed `0.005` in
  magnitude, so the centring is shown to be load-bearing

#### Scenario: The trace step is invariant to the trace's scale

- **GIVEN** trace normalisation on and two topologies whose traces differ by a constant factor
- **WHEN** both scales have warmed and the rule steps each with the same modulator
- **THEN** the Hebbian steps SHALL be equal within floating-point tolerance
- **AND** each plastic tensor SHALL carry its own scale, computed over its masked entries only

#### Scenario: Scales are bias-corrected from the first observation

- **GIVEN** a freshly constructed rule with a scaling mode on
- **WHEN** it takes its first step
- **THEN** the modulator scale SHALL equal the first `|δ|` (floored) and the trace scale the first
  non-zero trace's root-mean-square (floored)
- **AND** after `t` observations each scale SHALL equal its bias-corrected running mean square
- **AND** an all-zero trace SHALL leave the trace scale and its observation count unchanged

#### Scenario: Both modes off is bit-identical

- **GIVEN** a configuration with the default scaling fields
- **WHEN** the rule steps in any mode (modulated or not, frozen or not)
- **THEN** the weights and the existing telemetry SHALL be bit-identical to the frozen reference

#### Scenario: Freeze and unmodulated mode keep the scales comparable

- **GIVEN** a frozen arm and an unmodulated arm with a scaling mode on
- **WHEN** each steps
- **THEN** the scales and the centre SHALL advance exactly as on the plastic modulated arm
- **AND** the frozen arm SHALL write no weight
- **AND** the unmodulated arm's modulator SHALL be `1.0` while its trace step is normalised

#### Scenario: Matched rule across substrates under normalisation

- **GIVEN** the connectome and the MLP with both scaling modes on and the same hyperparameters
- **WHEN** each steps from traces whose magnitudes differ by orders of magnitude
- **THEN** the root-mean-square Hebbian step per unit modulator SHALL be the same on both within
  tolerance

#### Scenario: The scaling fields are shared and bounded

- **WHEN** a plastic brain configuration is loaded
- **THEN** the four scaling fields SHALL come from the shared mixin with identical defaults on
  every plastic brain
- **AND** a scale rate outside `(0, 1]` or a non-positive floor SHALL fail at load
