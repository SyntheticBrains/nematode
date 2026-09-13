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
| `8x1` | 8 | −0.1962 | 0.8904 | 0.9986 | 0.8916 | 8/8 | **yes** | 4068 | 2814–10711 | 5196 | 0 |
| `16x1` | 16 | −0.1980 | 0.8872 | 0.9986 | 0.8885 | 8/8 | **yes** | 2671 | 1773–4704 | 4328 | 0 |
| `32x1` | 32 | −0.2124 | 0.8613 | 0.9986 | 0.8625 | 8/8 | **yes** | 2590 | 1658–4423 | 4394 | 0 |
| `64x1` | 64 | −0.2132 | 0.8598 | 0.9986 | 0.8610 | 8/8 | **yes** | 2496 | 1608–5673 | 4806 | 0 |
| `128x1` | 128 | −0.2203 | 0.8470 | 0.9986 | 0.8482 | 8/8 | **yes** | 2478 | 931–3289 | 3321 | 0 |
| **`64x2`** *(amendment — the yardstick's own shape)* | **128** | **−0.2170** | **0.8530** | 0.9986 | 0.8542 | **8/8** | **yes** | **958** | **351–1696** | 1626 | 0 |

### Four things this table settles

1. **The rule passes at every point.** 8 of 8 seeds above the floor at every shape, every mean above
   the registered halfway threshold. The **8-unit cell reproduces I.1's pass at 0.890**, which is the
   registered stop clause's condition and is why the rest is interpretable.
2. **Nothing is void and nothing is censored.** The analytic reference reaches 0.9986 of the gap at
   every shape, so reachability never limits a cell and the normalised column tracks the raw one to
   three decimals. Every seed crosses the criterion at every shape: **0 censored of 48**.
3. **Time-to-criterion does not grow with the dimension.** OLS of `log2(trials)` on `log2(N)` over the
   40 per-seed values of the width grid gives a slope of **−0.216**, bootstrap CI over seeds
   **[−0.331, −0.092]**, against the predicted **+1.0** and the registered bar of +0.5. The interval
   excludes zero **below**, so the dependence is real and runs opposite to the prediction. The spread
   narrows with N as well: 2814–10711 trials at 8 units against 931–3289 at 128.
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
the sweep**: a median of **958 trials** (351–1696) against 2478 at `128x1` and 4068 at `8x1`.

So depth does not cost this rule anything either. It helps.

### The derived budgets, all flagged

| platform | fitted trials | a budget constraint? |
|---|---|---|
| the MLP yardstick (128 units) | 2 090 | **no** |
| the connectome, read as units (302) | 1 736 | **no** |
| the connectome, read as draws per decision (1208) | 1 286 | **no** |

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

## S2 — the width axis on the hard-food cell

Five widths × three arms × seeds 1–8, 3000 episodes, `--track-experiment` so the drift column has
weights to read. 120 runs, all succeeded, 5022 s wall clock at 15.5× parallelism. Scored on plateau-tail
mean foods through I.2's graded family, each width's learning arm against **its own** frozen control,
one-sided paired, BH-FDR across the five widths.

| width | perturbed units | learning foods | learning clear | frozen foods | PPO foods | PPO clear | shift | p | q | favouring | drift | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | **8** | **19.636** | **90.7%** | 1.692 | 19.687 | 87.6% | **+17.94** | 0.0039 | 0.0065 | **8/8** | 0.91 | `beats_control` |
| 8 | 16 | 18.834 | 79.2% | 1.308 | 19.937 | 96.8% | +17.53 | 0.0039 | 0.0065 | 8/8 | 1.07 | `beats_control` |
| 16 | 32 | 15.721 | 53.6% | 3.890 | 19.926 | 96.8% | +11.83 | 0.0039 | 0.0065 | 8/8 | 1.27 | `beats_control` |
| 32 | 64 | 8.338 | 12.9% | 3.536 | 19.924 | 97.0% | +4.80 | 0.0547 | 0.0547 | 5/8 | 1.38 | `no_improvement` |
| 64 | **128** | **4.226** | **3.0%** | 2.002 | 19.722 | 94.6% | +2.22 | 0.0195 | 0.0244 | 7/8 | 1.39 | `beats_control` |

Per-seed learning-arm foods, which is where the monotonicity is visible without the means:

| perturbed units | per-seed foods (seeds 1–8) |
|---|---|
| 8 | 19.8, 19.7, 19.5, 19.4, 19.6, 19.8, 19.5, 19.7 |
| 16 | 18.0, 19.3, 18.8, 18.7, 19.2, 18.9, 18.4, 19.5 |
| 32 | 17.0, 18.0, 16.8, 13.6, 18.7, 13.5, 17.0, 11.1 |
| 64 | 11.5, 5.2, 10.8, 10.1, 4.4, 6.1, 8.2, 10.4 |
| 128 | 6.2, 2.3, 4.1, 4.2, 6.6, 2.6, 4.1, 3.6 |

The eight-unit arm's *worst* seed (19.4) beats the 128-unit arm's best (6.6) by three-fold.

### The minima, and where they bound

Both registered minima were enforced together: **1.0 foods** of the cell's 20, and **10% of that
width's own PPO-minus-frozen gap**. The second is the binding one at every width — 1.80, 1.86, 1.60,
1.64 and 1.77 foods respectively — and every significant shift clears it comfortably. No width was
downgraded to `below_min_effect`; the only non-winner is the 64-unit cell, which fails on significance
rather than on size.

### The 64-unit cell

| seed | learning | frozen | favours |
|---|---|---|---|
| 1 | 11.48 | 1.04 | yes |
| 2 | 5.25 | 6.11 | no |
| 3 | 10.79 | 1.61 | yes |
| 4 | 10.08 | 3.09 | yes |
| 5 | 4.45 | 4.47 | no |
| 6 | 6.05 | 8.79 | no |
| 7 | 8.17 | 3.03 | yes |
| 8 | 10.44 | 0.15 | yes |

Five of eight, so p = 0.0547 — above the level while the shift (+4.80) is twice that of the 128-unit
cell, whose 7-of-8 gives q = 0.0244. The exact one-sided paired test counts sign agreements, and this is
what that costs. The three dissenting seeds are the ones whose *frozen* arm happened to start high
(6.11, 4.47, 8.79 against a width mean of 3.54), not seeds where the learning arm collapsed.

### The capability arm

Passes at **every** width: 19.69–19.94 foods and 87.6–97.0% full clear, against frozen controls at
1.31–3.89 foods and 0% clear. So no width is `uninterpretable`, the small-N end of the grid is not
capacity-limited — four hidden units per layer, below the input dimension, suffices for both optimisers
on this cell — and the **declared one-layer alternative was not needed**.

The comparator caveat stands as registered: the frozen arm perturbs and the PPO arm does not, so this is
a capability floor on the width rather than a matched pair, and it is not the committed calibrated MLP
arm for this cell.
