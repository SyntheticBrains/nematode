# Probe 2 (uncentred normalisation): why the grid was not re-registered on it

Run 2026-09-06 from the panel branch with both scaling switches on in every arm config, on the
rule as merged in PR #313 (modulator `tanh(δ / σ)`, trace `E / ρ`): the wild-type plastic
connectome and the plastic MLP at four rates, 600 episodes, seed 101. Diagnostic, not part of the
registration; the same seed and horizon as the raw-rule probe in
[pilot-1-notes.md](pilot-1-notes.md).

## Full-clear successes per 100-episode block, seed 101

| arm | rate | blocks 1–6 | saturated (last 20) | trace scale ρ (last 20) |
|---|---|---|---|---|
| wt_plastic | 1e-4 | 3 1 0 0 0 0 | 0.000 | 8.9 |
| wt_plastic | 3e-4 | 0 0 0 0 0 0 | 0.000 | 9.5 |
| wt_plastic | 1e-3 | 4 4 4 2 4 4 | 0.000 | 9.4 |
| wt_plastic | 3e-3 | 1 12 0 3 2 3 | 0.028 | 9.1 |
| mlp_plastic | 1e-4 | 0 0 0 0 0 0 | 0.000 | 5.4e7 |
| mlp_plastic | 3e-4 | 0 0 0 0 0 0 | 0.000 | 1.0e9 |
| mlp_plastic | 1e-3 | 0 0 0 0 0 0 | 0.000 | 3.4e9 |
| mlp_plastic | 3e-3 | 0 0 0 0 0 0 | 0.000 | 5.8e9 |
| wt_frozen (reference) | — | 4 3 4 3 1 2 | — | — |
| wt_plastic, raw rule (pilot-1 notes) | 1e-4 | 3 6 8 22 12 15 | 0.002 | — |

The connectome **declines** under the normalised rule at the rates where the raw rule learned,
to below the frozen floor. The MLP's trace scale is `0.01–0.05` at initialisation with inputs of
order `0.3` (checked directly), so `1e7–1e9` is an explosion during training, not a property of
the inputs.

## Diagnosis

`tanh(δ / σ)` is bounded, so a `−10` death and a `+2` food both compress to `±1`; with about 3.4
foods and one death per 203-step episode, and small distance rewards that also compress
positive, the compressed modulator has a mean of order `+0.01` to `+0.04` per step. A
prediction error must be zero-mean under the agent's own policy; compressing it before centring
breaks that, leaving a steady reward-blind Hebbian drive of order `2η` per episode. Bounded
connectome units collapse under it; unbounded ReLU units compound it across layers.

## Consequence

Ratified with Chris: centre after compression (`tanh(δ / σ) − c`, `c` a bias-corrected running
mean of the compressed value from a zero prior), landed as the modulator-centring change
(PR #314). The grid is re-registered only on a probe with the centred modulator (probe 3).
