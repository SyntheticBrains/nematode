# Probe 3 (centred modulator): the grid re-registration's evidence

Run 2026-09-06 from the panel branch on the rule as merged in PR #314 (modulator
`tanh(δ / σ) − c`, trace `E / ρ`, both switches on in every arm config): the three three-factor
arms at four rates, 600 episodes, seed 101. Diagnostic, not part of the registration; the same
seed and horizon as probes 1 and 2 so the columns compare.

## Full-clear successes per 100-episode block, seed 101

| arm | rate | blocks 1–6 | saturated (last 20) | trace scale ρ | centre c |
|---|---|---|---|---|---|
| wt_plastic | 1e-4 | 1 2 6 8 8 3 | 0.000 | 7.4 | −0.050 |
| wt_plastic | 3e-4 | 4 10 4 7 4 8 | 0.000 | 9.1 | −0.039 |
| wt_plastic | 1e-3 | 8 7 5 3 8 8 | 0.000 | 9.4 | −0.024 |
| wt_plastic | 3e-3 | 4 15 21 10 16 17 | 0.136 | 8.4 | −0.051 |
| rn_plastic | 1e-4 | 6 8 10 9 6 1 | 0.000 | 8.2 | −0.017 |
| rn_plastic | 3e-4 | 8 10 8 3 1 3 | 0.000 | 9.5 | −0.058 |
| rn_plastic | 1e-3 | 7 3 2 5 2 5 | 0.000 | 9.7 | −0.048 |
| rn_plastic | 3e-3 | 9 12 3 7 5 7 | 0.170 | 9.2 | −0.035 |
| mlp_plastic | 1e-4 | 0 0 0 0 0 0 | 0.000 | 6.9e1 | −0.019 |
| mlp_plastic | 3e-4 | 0 0 0 0 0 0 | 0.000 | 5.0e2 | −0.010 |
| mlp_plastic | 1e-3 | 0 0 0 0 0 0 | 0.000 | 1.3e5 | −0.017 |
| mlp_plastic | 3e-3 | 1 5 3 4 8 3 | 0.000 | 1.2e3 | −0.011 |
| wt_frozen (reference) | — | 4 3 4 3 1 2 | — | — | — |

## Reading

- **The decline is gone.** Centring removed the coherent drive: both connectome wirings now
  learn above the frozen floor at every rate, where probe 2 showed collapse below it. The centre
  settles a few hundredths *negative* on this cell, so the uncentred bias's sign depends on the
  stream and the correction is needed either way.
- **Saturation sets the ceiling of the grid.** At `3e-3` both wirings reach 14–17% of synapses on
  the bound by episode 600; at `1e-3` and below none do. `3e-3` also shows the strongest learning
  at this horizon (wild-type `4 15 21 10 16 17`), so whether it survives to a 3000-episode plateau
  or saturates into a constant policy is exactly what the pilot's plateau-tail selection decides.
- **The MLP yardstick does not learn at any rate**, and its trace scale still grows (from `0.01` at
  initialisation to `1e2–1e5`), far less than under the uncentred modulator but still growth: a
  dense ReLU stack under a local Hebbian rule has positive feedback along reward-correlated
  directions that a zero-mean modulator does not remove. This is now a fair property of the
  matched rule on that substrate rather than an artefact, and the band test's stated asymmetry
  (a non-learning yardstick makes test (ii) pass by construction) applies.
