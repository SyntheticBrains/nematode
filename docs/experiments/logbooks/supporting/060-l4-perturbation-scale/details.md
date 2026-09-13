# R.1 — the perturbation dimension: the full records

Protocol, arms and pre-registered outcomes: [`launch.md`](launch.md). Machine record:
[`scale.json`](scale.json). Per-seed S1 scores: [`s1-per-seed.csv`](s1-per-seed.csv).

## S1 — the one-step control across the perturbation dimension

Seeds 1–8, 20 000 trials, σ 0.2, rate 1e-3, `trace_decay` 0.9, homeostasis on, action noise
`exp(-1)` — I.1's passing configuration, with the shape as the only axis. The task's closed-form
cue-blind floor is **−0.6909**, its optimum **−0.1353** (a gap of **0.5556**), and the registered pass
threshold — halfway between them — is **−0.4131**.

Gap fraction is `(mean − floor) / (optimum − floor)`. "Normalised" divides it by the analytic
reference's fraction at the same shape, which is what a frozen random readout at that shape can
actually reach. Trials-to-criterion is the first trial whose trailing 100-trial mean crosses the pass
threshold; "sustained" requires two consecutive blocks.

| shape | perturbed units | mean | gap fraction | reference | normalised | above floor | passes | median trials | range | sustained | censored |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `8x1` | 8 | −0.1962 | 0.8904 | 0.9986 | 0.8916 | 8/8 | **yes** | 4450 | 3000–11000 | 5450 | 0 |
| `16x1` | 16 | −0.1980 | 0.8872 | 0.9986 | 0.8885 | 8/8 | **yes** | 3200 | 2800–5500 | 4300 | 0 |
| `32x1` | 32 | −0.2124 | 0.8613 | 0.9986 | 0.8625 | 8/8 | **yes** | 3400 | 2300–5500 | 3950 | 0 |
| `64x1` | 64 | −0.2132 | 0.8598 | 0.9986 | 0.8610 | 8/8 | **yes** | 3650 | 1900–6400 | 4650 | 0 |
| `128x1` | 128 | −0.2203 | 0.8470 | 0.9986 | 0.8482 | 8/8 | **yes** | 2650 | 2300–4400 | 3850 | 0 |
| **`64x2`** *(amendment — the yardstick's own shape)* | **128** | **−0.2170** | **0.8530** | 0.9986 | 0.8542 | **8/8** | **yes** | **1250** | **700–1700** | 1700 | 0 |

### Four things this table settles

1. **The rule passes at every point.** 8 of 8 seeds above the floor at every shape, every mean above
   the registered halfway threshold. The **8-unit cell reproduces I.1's pass at 0.890**, which is the
   registered stop clause's condition and is why the rest is interpretable.
2. **Nothing is void and nothing is censored.** The analytic reference reaches 0.9986 of the gap at
   every shape, so reachability never limits a cell and the normalised column tracks the raw one to
   three decimals. Every seed crosses the criterion at every shape: **0 censored of 48**.
3. **Time-to-criterion does not grow with the dimension.** OLS of `log2(trials)` on `log2(N)` over the
   40 per-seed values of the width grid gives a slope of **−0.149**, bootstrap CI over seeds
   **[−0.239, −0.061]**, against the predicted **+1.0** and the registered bar of +0.5. The interval
   excludes zero **below**, so the dependence is real and runs opposite to the prediction. The spread
   narrows with N as well: 3000–11000 trials at 8 units against 2300–4400 at 128.
4. **The cost that does exist is in the level, and it is small.** The gap fraction declines
   monotonically across the width grid — 0.8904, 0.8872, 0.8613, 0.8598, 0.8470 — a perfect rank
   correlation (**rho −1.000**) spanning **0.043 of the gap across a 16-fold change in N**. Real, and
   two orders away from what would be needed to take a learner below its own frozen floor.

### The depth control

Registered as a dated amendment **after** the width grid returned flat, because that result left one
shape difference between the platform the rule passes and the platform it fails: 128 units as **one**
layer of 128 here against **two** of 64 in every failing yardstick arm.

At two layers of 64 — `Linear(K, 64) → tanh → Linear(64, 64) → tanh → Linear(64, 1)`, hidden-only
plasticity, **128 perturbed units, the yardstick's exact arrangement** — the rule reaches **0.853** of
the gap on 8 of 8 seeds, against **0.847** for the same dimension in one layer. The two shapes are
indistinguishable in level, and the two-layer arrangement reaches criterion **fastest of every cell in
the sweep**: a median of **1250 trials** (700–1700) against 2650 at `128x1` and 4450 at `8x1`.

So depth does not cost this rule anything either. It helps.

### The derived budgets, all flagged

| platform | fitted trials | a budget constraint? |
|---|---|---|
| the MLP yardstick (128 units) | 2 944 | **no** |
| the connectome, read as units (302) | 2 590 | **no** |
| the connectome, read as draws per decision (1208) | 2 106 | **no** |

Each carries `extrapolation: true` in the record. Each also carries
`is_a_budget_constraint: false`, because the flag reads the fitted **interval** rather than the point
estimate and this interval does not exclude zero from above. A fit with a negative slope predicts a
*smaller* requirement at larger N: the figures are what the fit says and they are **not** budgets, and
reporting them as budgets would invert the result.

### What the sweep cannot separate

- **Units from weights.** In a fully-connected layer the weight count is proportional to the width, so
  a slope in N would have been equally consistent with a per-weight law. With the slope flat this
  limitation costs nothing — there is no positive exponent to attribute — but it is recorded because it
  would have mattered had the result gone the other way.
- **The exponent, at five points and eight seeds.** The sweep was built to answer whether the
  dependence binds over this range, not to measure its exponent.
- **Anything about long episodes.** The control's task is one-step by construction. A pass at the
  yardstick's shape says the arrangement is not the problem; S2 and [Logbook 055](../055-l4-horizon-multistep/details.md)
  are where episode length is asked about.

## Reproduction

Both S1 runs — the registered grid, and the re-run that added the depth control — returned **identical**
values for the five width cells, which is the determinism check the re-run doubled as.

The committed one-step control was re-run at the default shape before the sweep and compared against
[`048-l4-rule-positive-control/control.json`](../048-l4-rule-positive-control/control.json):
**77 common leaves, 0 differing, none missing.** The 49 keys present only in the re-run are the
node-perturbation arms I.1 added after 048 was written. Threading width and depth through the control
therefore changed nothing that any committed value depends on.
