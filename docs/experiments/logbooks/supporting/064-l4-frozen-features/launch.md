# L.0 — the wiring as fixed features: the registered protocol

Registered in `openspec/changes/add-l4-frozen-features`, reviewed and committed **before** the arms
run. The pilot and the rate check below ran before any registered seed was spent.

## The question

Phase 7's flagship asked whether the wild-type connectome becomes load-bearing under a rule the
animal could host. For rules that **write** the wiring that is answered:
[R.1c](../../061-l4-reduced-perturbation.md) returned `not_reducible` at every perturbation dimension,
and [R.2](../../063-l4-eprop.md) found e-prop reaching competence with the chemical matrix **frozen**
— every arm that writes it doing worse, by 3.9 to 15.9 foods.

This asks the form left over. Under `readout_only` the chemical matrix is frozen and only the 2×4
motor readout learns, by its **own exact gradient** — so the connectome enters as a **fixed feature
map**, and a degree-preserving rewiring changes that map and nothing else:

> *Do the wild-type edges compute better four-dimensional features for this task than a
> degree-matched shuffle of the same edges?*

## Why this cell

`hard350` is the only cell where the comparator, the learner and the metric already meet.

| | on `hard350` | source |
|---|---|---|
| the same contrast under **PPO** | wild type **892** episodes against the null's **1165 — +23.5%**, 32 paired seeds | V.3 |
| this learner's level | `readout_only` at **17.570 foods**, **52.61% full clear**, 16/16 seeds | R.2 |
| the metric's viability | `episodes_to_30pct_success` well-posed — far above the 30% threshold, far below ceiling | R.2 + V.3 |

The last row is not a formality. V.3's calibration found the band **one step wide**: at
`max_steps: 250` learning was plainly happening while **no arm crossed the 30% threshold the primary
metric needs**, so the contrast would have been censored for every seed and read as a clean null.

## The arms

| arm | wiring | learner | runs |
|---|---|---|---|
| `wt_learning` | wild type | readout learns, `w_chem` frozen | 32 |
| `rn_learning` | degree-preserving rewired null | the same | 32 |
| `wt_frozen` | wild type | nothing learns | 32 |
| `rn_frozen` | rewired null | nothing learns | 32 |

Each rewired config differs from its wild-type partner in the **`wiring` key alone**. `rewire_seed` is
unset, so each seed's rewiring derives from its run seed and the arms pair — V.1 and V.3's discipline.

**The floors differ from the learning arms in three keys, not one, and two of them must.** A frozen arm
may not declare a plastic readout: with no update no tensor moves, so "a plastic-readout floor" is not
a thing and a guard refuses it. R.2's shared floor is therefore the behaviourally correct null here.

## The matched projection

`B` — the broadcast projection the learning signal arrives through — is **identical across wirings at
a seed** by construction: drawn from a `torch.Generator` seeded with the run seed, where the rewiring
draws from a separate numpy generator, and rewiring preserves `n_neurons`. Verified at seeds 1, 2 and
7 and asserted by test, because the alternative — each wiring exploring through a different random
projection — would confound the wiring with the feedback path invisibly.

Also asserted: the rewiring preserves the neuron set and ordering, **per-post fan-in**, edge count and
the **motor pool**, so the weight-init scale `1/√(in-degree)`, the strict mask's shape, the
gap-junction normalisation and the readout's inputs are the same on both sides — and it nonetheless
rewires.

## The stop clauses, both run before any registered seed

**The pilot** (disjoint seeds 101–104, 16/16 runs clean) confirmed what it was registered to:

- `w_chem` drift reads **0.00** on both wirings — the check that this is a fixed-features contrast;
- the untrained prior does **not** separate (−2.29 foods, q = 0.875);
- both learners move a long way off their floors, **+16.3** and **+14.4** foods, 4/4 seeds each.

Its **direction is not read**, and the harness withholds a verdict at that seed count: with 4 pairs the
smallest one-sided p an exact test can return is **2⁻⁴ = 0.0625**, above the gate, so no gate could
have passed whatever the arms did. R.1c's pilot is the cautionary case — its `full` arm went **+0.61
on 4/4 pilot seeds to −0.112 on 4/8 registered seeds**.

**The rate check** (same seeds, 8/8 runs clean) — the one R.2 waived and this campaign could not:

| rate | mean foods | per-seed |
|---|---|---|
| 1e-4 | 12.08 | 8.93, 11.10, 14.02, 14.29 |
| **1e-3 (committed)** | **17.41** | 16.57, 17.48, 17.80, 17.81 |
| 1e-2 | 12.81 | 5.37, 13.87, 15.00, 17.00 |

The committed rate is **+4.60 foods above the nearest alternative** and the only one tight across seeds
(spread 1.24 against 11.63 at the decade above), so it sits at an optimum rather than on a slope. That
removes the specific way a null here could be an artefact — the failure R.1c's σ calibration found on
this substrate, where a carried-over value was costing 31.5%.

## The reading

Block V's instrument, unchanged: `episodes_to_30pct_success` through the committed
`connectome_structure_efficiency.py`, four efficiency metrics, paired by seed, BH-FDR, against the
registered **≥ 20%** minimum on time-to-competence. **`wiring_premise.py` is not imported** — it
hard-codes its test family per cell and carries block V's committed verdicts, so the gates live in a
sibling module instead.

Three checks travel with the primary and two can void it: the **two learning gates** (each wiring
against its own floor), the **untrained prior** (measured here — V.1's −0.17 and V.3's −0.01 were
measured on PPO-configured floors at action std 1.0 where these run at 0.368, and carrying one
regime's figure into another as established is what this change's own design forbids), and
**credited drift**, which must read 0.00 on both wirings.

## The power arithmetic, registered because a null closes the phase

| | k needed for p ≤ 0.05 | at a 65.6% win rate | at 73.4% | at 81.3% |
|---|---|---|---|---|
| 16 pairs | 12/16 (75.0%) | 30.7% | **57.3%** | 83.4% |
| **32 pairs** | 22/32 (68.8%) | 43.4% | **79.2%** | 97.3% |

V.3's observed per-seed win rate on this contrast was **21–26 of 32**, so the midpoint is **73.4%**.
Sixteen pairs would have given **57.3%** power there — missing an effect of the comparator's size more
often than catching it — which is not an acceptable basis for a null that closes a phase.

## Outcomes

| verdict | test | what follows |
|---|---|---|
| `wiring_is_legible` | wild type faster by ≥ 20%, significant, gates pass, prior clean | The first wiring result under a **plausible** learner. Phase 7 closes with three citable results; **L.4 and L.5 open** |
| `wiring_is_inert_as_features` | no significant advantage at the bar | 034's degree-statistics verdict extends to a **second learning regime**. **L.1 is promoted to MUST** |
| `below_bar` | significant, under 20% | Suggestive with the bar unmet, as V.1 and V.3 would have been held |
| `void` | a learning gate fails, or the prior separates | Uninterpretable: the arms did not learn, or the rewiring changed the substrate before learning did |

### What this cannot be, whatever it returns

**Not evidence for D2's primary**, which requires *plastic* wild-type to beat *plastic* rewired-null;
nothing here makes the wiring plastic, and no result can convert Phase 7's SPLIT into a GO. Not a
dynamics claim. Not a result about the wiring in general — one cell, one readout width, one learner,
with the readout standing in for the entire motor periphery. And **V.3's +23.5% is context, never a
quantitative delta**: the two run under different learning regimes, which the project's own
commensurability rule forbids comparing quantitatively.

## Honest prior

**Genuinely open, and I would not bet.** For a positive: this is the first learner on this substrate
that can express a wiring difference at all — every earlier contrast ran under a rule that either
could not learn or actively destroyed what it was given, and reservoir-computing results generally do
find recurrent structure mattering to a linear readout. Against: 034 measured this wiring as
indistinguishable from its degree-matched null under PPO on the endpoint, V.2 found **no** graph
property predicting learning time across 64 rewirings, and the feature map here is squeezed through
four pooled means — eight numbers reaching the action. The pilot's direction is uninformative by
construction and is not being read.

## Reproduce

```bash
P=configs/scenarios/foraging/connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_hard350_eprop

# 1. pilot on DISJOINT seeds — 16 runs. Verdict withheld at this seed count by design.
uv run python scripts/run_campaign.py \
  --config ${P}_readout_only.yml --config ${P}_readout_only_rewired_null.yml \
  --config ${P}_frozen.yml --config ${P}_frozen_rewired_null.yml \
  --seeds 101-104 --runs 3000 --output-dir campaigns/frozen-features-pilot \
  -- --theme headless --track-experiment

# 2. the rate check, same seeds — the committed rate against one decade either side
uv run python scripts/run_campaign.py \
  --config ${P}_readout_only_r1e4.yml --config ${P}_readout_only_r1e2.yml \
  --seeds 101-104 --runs 3000 --output-dir campaigns/frozen-features-rate \
  -- --theme headless --track-experiment

# 3. the arms — 128 runs. `--track-experiment` is REQUIRED for the frozen-substrate check.
uv run python scripts/run_campaign.py \
  --config ${P}_readout_only.yml --config ${P}_readout_only_rewired_null.yml \
  --config ${P}_frozen.yml --config ${P}_frozen_rewired_null.yml \
  --seeds 1-32 --runs 3000 --output-dir campaigns/frozen-features \
  -- --theme headless --track-experiment

# 4. score — gates first, then the contrast
uv run python scripts/analysis/l4_frozen_features.py \
  --campaign campaigns/frozen-features --seeds 1-32 \
  --out docs/experiments/logbooks/supporting/064-l4-frozen-features/frozen_features.json
```
