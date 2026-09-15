# 064: The Wiring Is Not Legible as Fixed Features Either (7a-ii L.0 / Phase 7)

**Status**: completed — **`wiring_is_inert_as_features`**. The wild-type connectome shows **no
significant advantage** over its degree-preserving rewired null at the registered **≥ 20%** bar on
time-to-competence (q = 0.971 one-sided, block V's own instrument), under the one biologically
plausible learner on this substrate that reaches competence — `readout_only`, where the chemical
matrix is **frozen** and only the 2×4 motor readout learns, by its own exact gradient. The contrast is
fully interpretable and not voided: both wirings learn (**+14.801** and **+14.701** foods over their
own frozen floors, q = 0.000 each), the untrained prior does **not** separate (−0.975, q = 0.895), and
`w_chem` drift reads **0.00** on both sides, so the substrate really was frozen. **The point estimates
lean the other way** on three of four metrics — the null reaches competence on a median **405**
episodes against the wild type's **568**, ahead on **22 of 32** seeds — at **q = 0.058 reversed, short
of the gate and post-hoc in direction**, so it is recorded as a lean and not as a result. The most
concrete difference is not statistical: **five wild-type seeds never became competent** within 3000
episodes against **one** for the null. So [034](034-connectome-structure-controls.md)'s
degree-statistics verdict now extends to a **second learning regime** — the wiring is endpoint-inert
under gradient learning, actively harmful under local rules that write it
([063](063-l4-eprop.md)), and as fixed features indistinguishable from a degree-matched shuffle.
**No mechanism is available for the lean**, and none is offered: V.2 tested 64 rewirings against four
graph properties fixed in advance and found nothing predicting learning time.

**Branch**: `feat/l4-frozen-features`.

**Date**: 2026-09-16.

**OpenSpec change**: `add-l4-frozen-features` (extends `plasticity-evaluation`: a wiring contrast run
under a learner that does not write the wiring states what it is about; a campaign whose null carries
a registered consequence states its power in advance).

## Objective

Phase 7's flagship asked whether the wild-type connectome becomes load-bearing under a rule the animal
could host. For rules that **write** the wiring that is answered twice over —
[R.1c](061-l4-reduced-perturbation.md) returned `not_reducible` at every perturbation dimension, and
[R.2](063-l4-eprop.md) found e-prop reaching competence with `w_chem` **frozen**, every arm that writes
it doing worse by 3.9 to 15.9 foods.

This asks the form left over. The connectome enters as a **fixed feature map**: the settling dynamics
turn three klinotaxis inputs into 302 activations, those mean-pool into four motor-class means, and
eight parameters decode them into `(speed, turn)`. A degree-preserving rewiring changes that map and
nothing else.

> *Do the wild-type edges compute better four-dimensional features for this task than a degree-matched
> shuffle of the same edges?*

## Method

`hard350`, the only cell where the comparator, the learner and the metric already meet:
[V.3](058-wiring-premise-difficulty.md) ran this contrast there **under PPO** (892 episodes against
1165, **+23.5%**, 32 paired seeds), R.2 measured this learner there, and the metric is well-posed at
that level — V.3's calibration found the band one step wide, with no arm crossing the 30% threshold at
`max_steps: 250`.

Four arms, **32 paired seeds**, 3000 episodes, **128 of 128 runs succeeded**. Each rewired config
differs from its wild-type partner in the **`wiring` key alone**; `rewire_seed` is unset so each seed's
rewiring derives from its run seed. The feedback projection `B` is **identical across wirings at a
seed** by construction and by test — its generator is seeded by the run seed while the rewiring draws
from a separate one — so the pair differs by the wiring and not by the feedback path.

Block V's instrument, unchanged: four efficiency metrics through the committed
`connectome_structure_efficiency.py`, paired, BH-FDR, `episodes_to_30pct_success` primary.
`wiring_premise.py` was **not** imported — it hard-codes its test family per cell and carries block V's
committed verdicts, so the gates live in a sibling module.

**The seed count is the load-bearing protocol choice.** At 16 pairs a one-sided sign test needs 12/16
positive, and V.3's observed per-seed win rate was 21–26 of 32, so 16 seeds would have given **57.3%**
power at that range's midpoint of 73.4% — missing an effect of the comparator's size more often than
catching it. Thirty-two pairs give **79.2%**. A null here closes the phase, which is why the arithmetic
was registered before the run.

## Results

### The gates, read before the contrast

| test | effect | q | reading |
|---|---|---|---|
| `wt_learning − wt_frozen` | **+14.801** foods | 0.000 | the wild type learns this cell |
| `rn_learning − rn_frozen` | **+14.701** foods | 0.000 | so does the null |
| `wt_frozen − rn_frozen` | −0.975 foods | 0.895 | **the untrained prior is indistinguishable** |

`w_chem` drift is **0.00** on both learning arms — the check that this is a fixed-features contrast at
all. Nothing voids the primary.

The prior is reported beside V.1's −0.17 (q = 0.735) and V.3's −0.01 (q = 0.841) as **context only**:
those floors were PPO-configured at an action std of 1.0 where these run at 0.368.

### The primary, and the lean

Medians lead, because the means are censored — see below.

| metric | wild median | null median | wild mean | null mean | null better | q (reversed) |
|---|---|---|---|---|---|---|
| **`episodes_to_30pct_success`** | **568.0** | **405.0** | 1039.8 | 644.7 | **22/32** | 0.058 |
| `auc_success` | 0.46 | 0.56 | 0.43 | 0.54 | 22/32 | 0.058 |
| `auc_foods` | 16.90 | 18.10 | 16.55 | 17.32 | 21/32 | 0.058 |
| `episodes_to_90pct_foods_plateau` | 404.5 | 399.0 | 483.0 | 497.4 | 18/32 | 0.466 |

**Under the registered one-sided test — wild-better, block V's instrument — nothing is significant, at
q = 0.971 across the family.** That is the verdict.

**The lean is stated because it is not noise-shaped.** Reversed, the primary gives p = 0.0287
one-sided and 0.0573 two-sided, with the null ahead on **22 of 32** — exactly the *k* the registered
power arithmetic said is needed at 32 pairs. Under BH across the family the reversed q is **0.058** on
three of four metrics, in the same direction, which agrees with 034's PPO endpoint finding. It is
**short of the gate** and **post-hoc in direction**, and it is recorded as a lean.

### The censoring, which is a finding rather than a footnote

`episodes_to_30pct_success` is right-censored at the 3000-episode horizon. **Five wild-type seeds never
crossed the 30% threshold; one null seed did not.** That is the most concrete difference in the
campaign and it is not a statistical artefact — it is the wild type failing to become competent more
often. It is also why the means diverge so much further than the medians (1039.8 against 644.7, where
the medians are 568 against 405): five censored values at 3000 pull the wild mean up. The medians are
therefore the figures to quote, and the means are shown only beside them.

### No mechanism is available, and none is offered

V.2 regenerated all 64 rewirings and scored them on four graph properties fixed before looking, finding
**nothing** that predicts learning time. So this project's own evidence supplies no candidate mechanism
by which a degree-matched shuffle would compute *better* features, and the lean is recorded as
unexplained rather than attached to a story about spectral radius or path length.

## What this settles, and what it does not

- **034's verdict extends to a second learning regime.** Endpoint-inert under gradient learning
  ([034](034-connectome-structure-controls.md)), actively harmful under local rules that write it
  ([063](063-l4-eprop.md)), and indistinguishable from a degree-matched shuffle as fixed features.
  Three independent learning regimes, one answer.
- **L.1 is promoted to MUST**, by the conditional registered *before* this ran: a null at the 2×4
  pooling with no width test leaves the obvious question unanswered — whether the readout was simply
  too small to see the wiring's features. It is also the right test for the lean, since a bottleneck
  could produce both.
- **L.4 and L.5 stay shut.** Their gate was this result reading positive; against a null there is
  nothing for a feature ablation to have changed.

### What this may not be cited as

- **Evidence for D2's primary.** That requires *plastic* wild-type to beat *plastic* rewired-null.
  This learner leaves the wiring frozen, so no result here satisfies it or converts Phase 7's SPLIT
  into a GO. The harness carries that as a field rather than as prose.
- **"The shuffle beats the connectome."** The lean is sub-threshold at q = 0.058, post-hoc in
  direction, and inflated in the means by asymmetric censoring.
- **A result about the wiring in general.** One cell, one readout width, one feature dimensionality,
  one learner — and the readout stands in for the entire motor periphery, which the substrate ladder's
  body rung is what would replace.
- **A quantitative comparison with V.3's +23.5%.** Same cell, same metric, same bar, different
  learning regimes; the project's commensurability rule forbids treating cross-regime deltas
  quantitatively. Two answers to one question, not a difference.

## Corrections on the record

- **A discrepancy I reported three times did not exist.** Progress updates flagged this campaign's
  wild-type arm at 16.961 foods on seeds 1–16 against R.2's committed 17.570, as an unexplained gap
  needing reconciliation. It was a metric mismatch of my own: I was reading the logs' whole-run
  `Average foods collected per run` while the committed harness uses the **plateau-tail** (final
  quarter) mean. On the harness's metric every seed matches R.2 **exactly** to four decimals, and the
  two campaigns' seed-1 runs are **byte-identical across all 3000 episodes**. Task 4.3 passes. The
  systematic direction I read into it — 15 of 16 lower — is simply what a whole-run mean does against
  a plateau-tail mean on an arm that is still improving.
- **The registered power table was wrong before the run and was corrected in the change.** It quoted
  63% and 85% from a rounded 75% midpoint; V.3's observed 21/32–26/32 has a true midpoint of 73.4%,
  giving **57.3%** and **79.2%**. The 32-seed decision was unchanged and better supported. A test now
  pins both range endpoints.
- **The harness announced a gate failure the design made unavoidable.** On the four-seed pilot it
  printed `void — a learning gate failed` while both gates showed +16.3 and +14.4 foods on every seed:
  at 4 pairs the smallest reachable one-sided p is 2⁻⁴ = 0.0625, above the gate, so no gate *could*
  pass. It now withholds the verdict and reports the power floor — the same lesson R.2's harness
  learned in a more dangerous form.

## Next Steps

- [ ] **L.1 readout width**, now MUST: a readout over all 39 motor neurons against the pooled 2×4, on
  both wirings. Answers both the null and the lean.
- [ ] V.4, the fresh-rewiring panel — still the open caveat on block V's positive result, and now the
  only surviving wiring effect in the project.
- [ ] The Phase 7 synthesis, which closes on 7a + L.0 + V.4.
