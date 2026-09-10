# I.1b — annealing the perturbation scale: the registered protocol

Registered in `openspec/changes/add-l4-sigma-annealing`, reviewed and committed **before** the run.
The values below were fixed in that change; nothing here was chosen after a result existed.

## The question

I.1 left a tension, not a verdict. Node perturbation learns at σ = 0.2 — 89% of the
floor-to-optimum gap, 8 of 8 seeds — and the frozen control shows that same σ takes a competent
policy from 38.7 to 8.9 with no weight ever written. **The σ that makes the rule learn is the σ
that makes a competent policy unrunnable.** That is a statement about a *constant* σ. Does a
schedule separate the two jobs?

## The schedule

| | |
|---|---|
| shape | geometric, `σ(e) = σ₀ · (σ_final/σ₀)^(e/E)`, constant at `σ_final` after `E` |
| σ₀ | 0.2 — the scale that passed this control |
| σ_final | 0.02 — an order of magnitude down; under the 0.05 that cost 7/8 seeds the bar, above the 0.01 at which the estimator was inert |
| E | half the budget (10,000 of 20,000 trials), so the score window lies entirely at the floor |
| rate regime | `normalise_trace: true` — the trace carries the perturbation, so without it the decay would cut the effective rate as well as the exploration |

## The gates, in order

1. **The positive control under the schedule.** Same task, same three validity arms, same seeds,
   same pass rule: 7 of 8 seeds above floor **and** a mean at or above half the floor-to-optimum
   gap. Alignment reported over the decay and over the floor **separately** — at the floor the
   estimator is nearly silent by design, so a low floor-phase alignment is what a *good* schedule
   looks like and is not the failure signature.
2. **The clone assay**, with a frozen control on the identical schedule — **only if 1 passes.**

**A failure at gate 1 stops the sequence** and is reported as a property of this schedule, not
resolved by re-tuning its bounds or its length.

## Reproduce

```bash
uv run python scripts/analysis/l4_rule_positive_control.py \
  --out docs/experiments/logbooks/supporting/051-l4-sigma-annealing/control.json \
  --csv docs/experiments/logbooks/supporting/051-l4-sigma-annealing/control-per-seed.csv
```
