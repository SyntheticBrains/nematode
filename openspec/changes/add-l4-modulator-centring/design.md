# Design: centre the compressed modulator

## Context

Under modulator normalisation the rule computes `u = tanh(δ / σ)` with `δ = r − b` and `σ` a
bias-corrected running RMS of `δ`. `δ` is zero-mean under the agent's policy because `b` is
its running mean. `u` is not: `tanh` is bounded, so rare large negatives (a `−10` death) and
frequent moderate positives (a `+2` food) both saturate at `±1`, and the frequency asymmetry
that the raw mean balanced now sets the sign of `E[u]`. On the panel's cell `E[u] ≈ +0.012` per
step. A modulator with a non-zero mean multiplies the trace by a constant on average, which is
plain Hebbian drift: coherent across steps and episodes, blind to reward, compounding on any
substrate without an activation bound.

## Goals / Non-Goals

**Goals**

- Restore `E[modulator] = 0` under the agent's policy while keeping the compression's bound
  and scale-freedom.
- Keep the raw `δ`, the baseline, `σ` and the trace normalisation exactly as they are.
- Stay byte-identical with the switch off.

**Non-Goals**

- Any change to the trace normalisation, the decay, the clamp, or the substrates.
- A new switch: the mode is corrected, not forked (see D3).

## Decisions

### D1. `m = tanh(δ / σ) − c`

`c` is a bias-corrected exponential moving average of `u = tanh(δ / σ)` at the same
`plasticity_scale_rate` as the scales: `c ← (1 − r_s) c + r_s u`, corrected by
`1 − (1 − r_s)^t`. The current step is scored against `c` from **before** `u` is absorbed,
the convention every other estimator in the rule already follows.

Ratified with Chris over two alternatives. *Compress the reward, then predict it*
(`r̃ = tanh(r / σ_r)`, modulator `r̃ − EMA(r̃)`) has the same zero-mean property with one
baseline instead of two, but it changes what the existing baseline and prediction-error
telemetry mean whenever the switch is on. *Switching the modulator normalisation off* returns
the death penalty to two hundred times an ordinary step and leaves the MLP at the mercy of
one kick. Centring keeps every existing quantity's meaning and adds one scalar.

### D2. Zero prior, not first-observation warm start

The scales warm-start from their first observation because a scale of zero is meaningless.
The centre is different: a prediction error is zero-mean a priori, so before any observation
`c = 0` and the first modulated step is the uncentred compression. From the second step `c`
is the bias-corrected mean of the `u` values seen so far. Starting `c` at the first `u`
instead would zero the first modulated step for no reason.

### D3. One mode, redefined; no new switch

The uncentred compression has never been used in a pinned recipe or a panel run: the panel's
configs turned the switches on days ago and the first probe with them on is what found the
defect. Forking a second switch would preserve, as a selectable option, a modulator known to
drift. The mode's requirement is modified instead, every scenario keeping its name; the raw
rule (switch off) is untouched and the frozen-reference test keeps proving it.

### D4. Range, telemetry, freeze, unmodulated

- The modulator lies in `[−2, 2]` (`u ∈ [−1, 1]`, `c ∈ [−1, 1]`); in practice `|c|` is a few
  hundredths. The bound scenario is restated accordingly.
- `plasticity_modulator_centre` reports `c` as used for the step (the pre-update value), beside
  `plasticity_modulator` (now the centred value), `plasticity_modulator_scale` and the raw
  prediction error. NaN when the switch is off, like the scales.
- Under a freeze `c` advances like `σ`. In unmodulated mode the modulator stays `1.0` and `c`
  is still tracked and reported, mirroring how `σ` and the baseline are reported there.

### D5. What the zero-mean test proves

A synthetic stream mimicking the cell — many small steps around zero, frequent `+2`, rare
`−10`, all baselined so the raw `δ` is zero-mean — drives the rule for a few thousand steps.
The mean of the centred modulator over the post-warm-up steps SHALL be within `0.005` of zero,
and the mean of the uncentred `tanh(δ / σ)` on the same stream SHALL exceed `0.005` in
magnitude. The second assertion is what makes the first meaningful: it shows the test would
have caught the defect.

## Risks / Trade-offs

- **The centre lags a non-stationary stream.** As the agent learns, food becomes more frequent
  and `c` drifts up with a lag of about `1 / r_s` steps. During that lag the modulator carries a
  small positive mean — the same lag the baseline already has, and far smaller than the
  uncorrected `+0.012`.
- **The MLP may still grow.** A zero-mean modulator removes the coherent drive, not Hebbian
  positive feedback along reward-correlated directions. Whether a dense ReLU substrate stays
  bounded under the matched rule is now a fair question for the re-probe rather than an
  artefact; the trace-scale telemetry answers it.

## Open Questions

None.
