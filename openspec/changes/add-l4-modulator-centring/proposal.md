# Centre the compressed modulator

## Why

The substrate-invariant scaling shipped for the three-factor rule made the third factor
`tanh(δ / σ)`: bounded, sign-preserving, scale-free. The first probe with both switches on
showed it is also **wrong in the mean**, and the mean is the whole point of a prediction error.

On the panel's cell an episode carries about 3.4 food rewards of `+2`, one terminal penalty of
`−10`, and some two hundred small steps. Raw, the prediction error `δ = r − b` is zero-mean by
construction: the baseline absorbs the average. Compressed, a `+2` and a `−10` both map to `±1`,
so the asymmetry that made the raw stream zero-mean is erased and the frequent events win:
the compressed modulator averages about `+0.012` per step. That is a steady, reward-blind
Hebbian drive of order `2η` per episode along whatever the trace already correlates — exactly
the runaway the baseline exists to prevent, now reintroduced downstream of it.

The probe showed the consequence on both substrates. The connectome, whose units are bounded,
**declines** under the normalised rule at the rates where the raw rule learned (successes per
hundred episodes `3 1 0 0 0 0` at `1e-4`, `0 0 0 0 0 0` at `3e-4`, against `3 6 8 22 12 15`
raw). The MLP, whose ReLU units are not bounded, **explodes**: its trace scale climbs from
`0.01` at initialisation to `1e7–1e9` within six hundred episodes at every rate, the coherent
drive compounding across three layers. Neither is a property of the substrates; both are the
compression's non-zero mean at work.

The fix, ratified with Chris over compressing the reward before predicting it and over
dropping the modulator normalisation: **centre after compression**. The modulator becomes
`tanh(δ / σ) − c`, with `c` a bias-corrected running mean of `tanh(δ / σ)` used at its
pre-update value and starting from the zero prior a prediction error has a priori. The raw
`δ`, the baseline and `σ` keep their meanings; one running scalar and one telemetry key are
added. This is the smallest change that restores the invariant "a prediction error is
zero-mean under the agent's own policy" while keeping everything the scaling change bought.

## What Changes

- The modulator-normalisation mode of the three-factor rule centres its compressed value by a
  bias-corrected running mean (same scale rate, pre-update value, zero before any observation).
  The modulator's range becomes `[−2, 2]`; in practice the centre is small.
- One telemetry key and history field, `plasticity_modulator_centre`, beside the existing
  scaling telemetry.
- No new switch: the mode is redefined. It has never been used in a pinned recipe or a panel
  run, and the scaling switches are off by default, so the raw rule stays bit-identical.
- Tests: the centred value against an independent bias-corrected mean; the zero prior; the
  bound; the zero-mean property on a synthetic skewed stream where the uncentred compression
  is demonstrably biased; freeze and unmodulated behaviour; the key and the recorder. The
  scaling tests that pinned the uncentred value are updated.
- Docs: the architectures table's plasticity row, the CHANGELOG.

Out of scope: the trace normalisation (unchanged), any activation bound on the MLP (a
substrate change), the panel's registration (its grid follows once this lands).

## Capabilities

**Modified**: `learning-rules` — the modulator-normalisation mode of the substrate-invariant
scaling requirement centres its compressed value; every existing scenario keeps its name.

## Impact

- Edited: `learning_rules/three_factor.py`, `brain/arch/_brain.py` (one field), the scaling
  tests, `docs/architectures.md`, `CHANGELOG.md`.
- No config changes; trace-off, switch-off and PPO builds are byte-identical.
