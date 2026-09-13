# R.1c — a declarable perturbation set for the connectome: the registered protocol

Registered in `openspec/changes/add-l4-reduced-perturbation`, reviewed and committed **before** the run.

## The question

[R.1](../../060-l4-perturbation-scale.md) found the rule solving a multi-step foraging cell at **8
perturbed units** and collapsing to **3.0% full clear at 128**. The connectome perturbs **302 neurons at
each of four settling steps — 1208 draws per scored decision** — and until now there was no way to lower
it. So no connectome arm followed from R.1, and **R.1b, the wiring contrast 7b's gate asks for, is
blocked on this.**

Recon found something sharper than a missing knob. The readout mean-pools **only the 39 VB/DB/VA/DA
motor neurons**, and the eligibility is `E ← decay·E + M_chem ∘ (h_prev ⊗ perturbation)`, so a unit's
perturbation writes eligibility on every synapse onto it whether or not it can reach the readout. At
settling step `s` it can reach the readout only from within `depth − s` hops:

| settling step | hop budget | units that can still reach the readout |
|---|---|---|
| 1 | 3 | 277 of 302 (91.7%) |
| 2 | 2 | 247 of 302 (81.8%) |
| 3 | 1 | 109 of 302 (36.1%) |
| 4 | 0 | **39 of 302 (12.9%)** |

**672 of 1208 draws per decision are causally connected. The other 536 — 44.4% — cannot change the
action at all, and every one of them writes eligibility.** The largest budget is 3, so the **25** units
at four hops or more can never contribute at any step: 1 at four hops, 7 at five, 7 at six, 3 at seven or
more, and 7 unreachable in the directed chemical graph.

## The arms

One cell: the **calibrated hard-food connectome cell** (`max_steps: 350`,
`target_foods_to_collect: 20`) — block V's cell, so a result is comparable both to R.1's MLP sweep and to
the PPO reference already recorded. Seeds 1–8, 3000 episodes. **Ten arms, 80 runs.**

| arm | perturbed set | units | adaptable synapses | draws/decision | causally connected |
|---|---|---|---|---|---|
| `full` | every unit, every step — **today's behaviour** | 302 | 3709 | 1208 | 672 |
| `causal` | per-step reach mask | 277 | 3538 | **672** | 672 |
| `hop1` | within 1 hop of the readout pool | 109 | 1476 | 436 | 366 |
| `motor` | the readout pool itself | 39 | 323 | 156 | 156 |
| `motor_last` | the readout pool, **last settling step only** | 39 | 323 | **39** | 39 |

Every figure above is asserted by test against the built masks, not quoted from the recon.

**Each arm carries its own frozen control at the same σ and the same mask**, freezing only the update, so
the perturbation's cost to the policy is matched across each pair and the contrast measures the update
alone.

**`hop1` is registered un-intersected with the causal mask** — 70 of its 436 draws cannot reach the
readout from the last step — so that `causal` remains the arm testing the correctness fix and `hop1`
remains a pure set restriction.

**The noise is drawn and then masked**, so the random stream does not depend on the declared set: two
sets at one seed differ only in which draws are *used*, which is what makes them comparable.

### The capability reference is already on the record

Block V ran PPO on this exact cell and substrate over **32 seeds**
([Logbook 058](../../058-wiring-premise-difficulty.md)): wild type under PPO **19.31 foods of 20**, wild
type with frozen weights **3.82 foods and 0.0% full clear**. **No PPO arm is re-run.** The reachable gap
is therefore **15.49 foods**, fixing the relative minimum at **1.55 foods**. That frozen arm does not
perturb, so it is a reference and not the comparator — hence each arm's own frozen control.

## The reading

**Plateau-tail mean foods** through I.2's graded family, each arm against **its own** frozen control,
paired by seed, one-sided, BH-FDR across the five masks. Full-clear success recorded alongside.

**A shift counts only if significant *and* both** at least **1.0 foods** of the cell's 20 and at least
**10% of the reachable gap (1.55 foods)**, recomputed per arm where that arm's frozen mean differs.

**Drift** per arm, reported **separately for excluded and credited synapses** — see the dependency below.

**Every run records its perturbation set**: the set, its units, its adaptable synapses, its draws per
decision and how many of those can reach the readout. R.1 had to reconstruct those numbers afterwards.

## The homeostasis dependency, measured rather than argued

The rule's weight decay is **unconditional** — it writes every plastic weight, trace or no trace — so the
synapses a restricted set excludes shrink on every update. The homeostatic rescale returns each unit's
incoming norm to its construction target, and both operations are purely radial, so they cancel. Measured
on the recipe's own rates with every trace held at zero:

| updates | incoming-norm change | largest single weight's excursion | 1 − cosine of the weight vector |
|---|---|---|---|
| 1 000 | 0.0000% | 1.8e-05 | 0 |
| 10 000 | 0.0000% | 1.9e-04 | 6e-08 |
| 100 000 | 0.0000% | 1.9e-03 | 2.6e-06 |

The norm is held **exactly** and the direction to a cosine of **1 − 3e-05** extrapolated over a run's
~1.05M updates. What remains is float32 round-off — about 1.9e-02 on the largest single weight, ~9% of
these weights' rms — leaving the vector the unit computes with unchanged. **Without homeostasis it is
decay**: 2.0% of the norm per 20 000 updates, a collapse over a run. A config validator therefore refuses
a restricted set without homeostasis, every arm pins it, and the campaign reports excluded-synapse drift
separately so the bench claim is checked at real scale.

## Outcomes, fixed before the run

| verdict | test | what follows |
|---|---|---|
| `causal_mask_sufficient` | the `causal` arm beats its own frozen control by both minima | the strongest available result: the failure was **uninformative noise**, not scale. R.1b runs at the causal mask |
| `dimension_reducible` | some reduced arm beats its control by both minima, and `causal` does not | the dimension binds on the connectome as on the MLP. **R.1b is unblocked at the winning mask**, and the record states which synapses that mask gives up |
| `not_reducible` | no arm beats its own frozen control | the connectome's failure is **not** the perturbation dimension. R.1b stays blocked, R.2 (e-prop) becomes the live path, and R.1's result stays bounded to the MLP |

A partial reading — `motor_last` working where `motor` does not, or a non-monotone ordering — is recorded
as **partial with the ordering stated**, not resolved toward the nearest verdict.

**A deliverable that does not depend on the arms.** The causal-reach table is a property of the substrate
at a given depth. A `not_reducible` verdict still ships the measurement that 44.4% of this substrate's
perturbation is credited against an outcome it cannot influence.

### Stop clauses — void until found

- **`full` does not reproduce the known failure.** Every plastic connectome result to date sits at or
  below its floor; if it learns here, something changed since 055 and nothing is interpretable until
  that is found.
- **A mask does not mask.** Asserted by test at build time for all ten arms.
- **The frozen floor is at the ceiling.** Then there is nothing to add and the finding is about the cell.

## Honest prior

**`not_reducible` for `causal` alone, and better than even for `motor`.** The arithmetic, in advance:

- the causal mask takes draws per decision from 1208 to 672 — a **1.8×** reduction;
- R.1's MLP curve needed **16×** (128 → 8 units) to move from 3.0% to 90.7% full clear, and a 2×
  reduction (128 → 64) only reached 12.9%.

So 1.8× is very unlikely to suffice on its own, and the causal mask is argued for as **correctness**
rather than as a knob. `motor` at 156 draws is a **7.7×** reduction and 39 units, between the MLP's 32
(53.6% full clear) and 64 (12.9%).

Against it: only **323 synapses** are adaptable at `motor` against 3709 unmasked. There may be too little
left to express — though the frozen prior is far from competent here (3.82 of 20 foods, 0.0% full clear
over 32 seeds), so there is a large gap to move into. That is the risk `hop1` brackets, at 1476 synapses.

## Disclosure

The pilot runs on seeds **101–104**, disjoint from the registered 1–8. The per-run cost estimate in the
proposal (~20–37 min) was taken from the **C3 cell**, which carries predator and thermal modules this
cell does not, so it is conservative; the pilot measures it and the campaign is scheduled from the
measurement.

## Reproduce

```bash
P=configs/scenarios/foraging/connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_hard350_nodepert

# Pilot — the two ends of the grid, disjoint seeds
uv run python scripts/run_campaign.py \
  --config ${P}_full.yml --config ${P}_full_frozen.yml \
  --config ${P}_motor.yml --config ${P}_motor_frozen.yml \
  --seeds 101-104 --runs 3000 --output-dir campaigns/reduced-perturbation-pilot \
  -- --theme headless --track-experiment

# Campaign — 80 runs. `--track-experiment` is REQUIRED: drift reads final.pt through the
# experiment record's exports_path, and without it every drift figure comes back empty.
uv run python scripts/run_campaign.py \
  $(for S in full causal hop1 motor motor_last; do
      printf -- "--config %s_%s.yml --config %s_%s_frozen.yml " "$P" "$S" "$P" "$S"
    done) \
  --seeds 1-8 --runs 3000 --output-dir campaigns/reduced-perturbation \
  -- --theme headless --track-experiment
```

______________________________________________________________________

## Amendment, 2026-09-13 — calibrate σ on this substrate before the campaign

The pilot passed both registered gates: the platform has room (frozen 1.9 of 20 foods against PPO's
19.31) and `full` has not learned (2.55 of 20, **0.00% full clear**, all 3000 episodes FAILED). The
direction is as predicted — `motor` +1.11 foods on 4/4 seeds against `full` +0.61 on 4/4 — but at 7% of
the 15.49-foods reachable gap it is **below the registered 1.55-foods minimum**, so the campaign as
registered would most likely record `not_reducible`.

**It also surfaced an uncalibrated knob that may matter more than the mask.** These frozen controls
perturb; 058's did not. σ 0.2 takes the frozen prior from **3.82 foods to 1.9 — it halves it** before any
learning happens. σ 0.2 was selected on the **one-step control** and carried to the MLP; **no connectome
arm has ever had it calibrated**, and this is the tension 052 named: the σ that makes the rule learn is
the σ that makes a competent policy unrunnable.

**Added, before the campaign**: the `motor` arm at **σ ∈ {0.05, 0.1}**, learning and frozen, on the
pilot's disjoint seeds 101–104 — **16 runs**, about 30 minutes. σ 0.2 is not re-run; the pilot's eight
`motor` runs are that point of the grid.

**The decision rule, fixed before these run.** The campaign takes the σ that **maximises the learning
arm's plateau-tail mean foods**, with the learning-minus-frozen gap and the frozen arm's retention of the
unperturbed 3.82-food prior reported beside every point. Ties go to the **larger** σ, which carries more
signal. The absolute level is primary rather than the gap, because the question is whether the rule
**learns the cell** — not whether it beats a floor its own noise damaged.

**And it can stop the campaign.** If no σ lifts the learning arm above the **unperturbed frozen prior of
3.82 foods**, then at 39 perturbed units — inside the band where the MLP reached 15.7 to 18.8 foods — this
substrate is not learning this cell at any tested scale. That answers the campaign's question at pilot
cost, and `not_reducible` is recorded from 24 runs instead of 104.

**A measurement the pilot already delivers, independent of what follows.** The connectome badly
underperforms the MLP at matched dimension: R.1 reached 18.83 foods at 16 units, 15.72 at 32 and 8.34 at
64, while this substrate's 39-unit arm reaches **2.99**. Perturbation dimension does not transfer across
substrates — registered as a caveat, now a number.
