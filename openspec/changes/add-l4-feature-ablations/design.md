# Design — L.4 + L.5, feature ablations at the per-neuron width

## An ablation is an interaction against a baseline, and the baseline is reused

L.1's positive is a difference: at the per-neuron width, `wt_wide − rn_wide = +0.1852` on
`auc_success`. An ablation asks whether that difference survives with one feature removed, so its
reading is a difference of differences:

```text
I_ablation = (wt_ablated − rn_ablated) − (wt_wide − rn_wide)      per seed, then the paired test
```

Negative: the feature carried some of the effect. Near zero: it did not. Positive: removing it helped
the wild type more, which is not predicted and is reported as itself.

**The baseline half is L.1's committed data, not a re-run**, for 384 runs' worth of reason — but only
under a condition. L.1's wide arms ran before the output controls existed; the ablated arms run with
them. The controls touch only what is written after a step, never the step, and a smoke test asserts
the console stream is intact — but "should be identical" is not "is". So **one seed per reused arm is
re-run under the new controls and compared to L.1's log field for field**. Identical: the baseline is
reused and the record says on what evidence. Different in any field: the baseline is re-run in full
and nothing is reused. There is no partial option.

## A minimum effect as a decision rule — the gap L.1 recorded

L.1 registered 0.1076 as its *sensitivity target* and not as a decision threshold, and the review
rightly asked why the verdict did not require it. It could not be retro-fitted there. It is a rule
here:

- **The effect an ablation can remove is the wide wiring effect, +0.1852.** Not L.1's +0.2818
  interaction: that figure includes the pooled cells, where the null led by 0.0966, and no ablation
  at the per-neuron width touches them. A feature carrying all of the effect gives an interaction of
  −0.185; the first draft registered "half of +0.2818 = 0.14", which is 76% of what can actually be
  removed while calling itself half.
- `carries_the_effect` requires the interaction to be significant **and** `abs(Δ) ≥ 0.123` —
  **two-thirds** of 0.1852: the feature is the *majority* carrier. Less than that is not credited.
- Sized from L.1's realised spread rather than assumed: sd 0.416 at n = 96 and ρ ≈ 0.08 between
  conditions, so the two halves are treated as independent. se = 0.0425; detectable at 80% is
  0.119; at the 0.123 minimum the power is **~80%** (z = 2.89). Half, 0.093, would sit at ~59% and
  need ~160 seeds — with a re-run baseline, since L.1's covers 1–96 — so the threshold is registered
  where the panel can resolve it and named for what it is.

## What each ablation removes, at mechanism level

**L.4 — atlas signs.** The readout pool receives **323 chemical synapses**, **311 of them
atlas-grounded: 275 excitatory, 36 inhibitory**. Under the random draw roughly half of those 311 carry
an inhibitory sign; under the atlas, 36 do. So this ablation rewrites the sign of about half the
readout's immediate chemical input, with every magnitude, norm and RNG draw untouched (asserted by
B.1's tests and re-checked at the per-neuron width: |w_chem| bitwise identical, 1,655 signs flipped
network-wide). B.1 found grounded signs left the untrained prior alone and made Hebbian learning worse
— under a rule that could not learn. Whether they change what the frozen substrate *computes as
features* is a different question, and L.1's learner is the instrument for it.

**L.5 — gap junctions.** `enable_gap_junctions: false` zeroes `g_gap` in the forward pass. Every
parameter is bitwise identical to the baseline — `g_gap` is a buffer built from the data with a
symmetric `1/√(d_i d_j)` scaling, no RNG — so this is the cleanest ablation available: the forward
pass and nothing else. It is not a peripheral one: **199 of 1,093 gap junctions touch the pool, 47 lie
within it, and all 39 pool neurons carry at least one**. The degree-preserving rewiring swaps gap
junctions as well as chemical synapses (`rewiring.py`), so both wirings lose theirs symmetrically.

Both ablations are flags that already exist; **no package code changes**. Both build at the
per-neuron width, verified before this was written.

## L.4 is not a clean removal, and that is registered rather than discovered

L.5 changes the forward pass and nothing else. L.4 changes the sign of about half the pool's
immediate input, taking it from ~50/50 to **275 E / 36 I** — a large shift in net drive into tanh
units, which can move their operating point independently of anything "features" means. So a
vanished effect under atlas has two live readings: the signs carried it, or the substrate saturated.

**The diagnostic, registered.** The two atlas frozen floors against the two wide frozen floors on
plateau-tail foods — arms in which nothing learns, so a difference is the operating point and not
learning. Tested two-sided at q ≤ 0.05, outside the ten-test family as a diagnostic. If it fires, a
`carries_the_effect` on L.4 is reported as **carries or saturates**, and a `survives_without_it` on
L.4 is the stronger result for having survived a moved operating point too. It cannot rescue an
L.4 reading; it can only qualify one.

## Amendment 2026-09-18, after the pilot and before the campaign — the gains diagnostic

The pilot (32 runs, seeds 101–104, all succeeded) passed its three stop-clause checks and exposed a
hole in the registration. The atlas learning arms gained **+2.2 and +2.8 foods** over their own
floors against the wide arms' **+13.7 and +7.8** at the same seeds — and **bimodally**: at seeds
101–102 both atlas arms finished *below* their own floors (−2.5, −1.5, −6.2, −0.7 foods) while at
103–104 they learned (+6.3 to +11.2), with both atlas arms censored on 2 of 4 seeds. The registered
floors diagnostic was **quiet** (q > 0.5): the frozen operating point did not move. What moved was
learnability on top of it, for **both** wirings.

That is a reading the rules as written get wrong. If it holds at 96 seeds the gates will likely pass,
the ablated wiring effect will sit near zero, the interaction near −0.2, and the harness will read
`carries_the_effect` — when what happened is that grounding the signs made the substrate nearly
unlearnable for wild type and shuffle alike, and the wild type simply had more to lose.

**The gains diagnostic**, registered now: each atlas arm's gain over its floor against the wide arm's
gain over its floor, paired per seed, per wiring, two-sided, BH over the pair, outside the family. If
**both** are significantly smaller, a `carries_the_effect` on L.4 is reported as **carries or
unlearnable**. It is the floors diagnostic's logic applied to gains rather than floors; it qualifies a
reading and never rescues one; and it applies to atlas only, since the nogap arms' gains (+17.1 and
+16.3) exceeded the baseline's. The nogap pattern — both wirings gaining, the null more — is **not
read** at four seeds.

## Amendment 2026-09-18 (second) — a registered rate check for the atlas arms, before the campaign

The pilot's atlas collapse has a mechanistic candidate: grounding makes the pool's inputs 275 E / 36 I,
so the 39 presynaptic activities the readout learns from sit nearer tanh saturation, and the readout's
own exact gradient `E[k,i] = score_k · h_i` becomes large and uninformative — the signature of a
learning rate that is too high for the substrate, and a plausible source of the bimodality (two seeds
destroying their policy, two learning). That is the one knob with both a reason and a precedent:
[L.0](../../064-l4-frozen-features.md) ran the committed rate against one decade either side on
disjoint seeds before its campaign. Nothing else is swept — a wider grid with no stopping rule, read
over seeds with bimodal outcomes, is how an effect gets manufactured.

**The check.** Both atlas learning arms at `plasticity_rate` **0.0001** and **0.01** on seeds 101–104
(0.001 is the pilot). Four configs, each differing from its atlas parent in `plasticity_rate` alone,
asserted by exact-key test. 16 runs. Read on **gain over own floor** (foods) per seed — the same
quantity the gains diagnostic reads — and on the count of seeds that finish **below** their own floor.

**The decision rule, fixed before the runs.** A decade is *better* than 0.001 if its mean gain over
floor is higher **on both wirings** and no more seeds sit below floor on either. A decade *learns
cleanly* if **no seed** sits below floor on either wiring **and** its mean gain reaches at least
**half** of the wide arms' gain at the same seeds (+13.7 wt, +7.8 rn).

| outcome | action |
|---|---|
| **A** — no decade is better than 0.001 | run the campaign **as registered**; the atlas collapse is a property of the substrate at this learner's operating point, and the gains diagnostic carries the reading |
| **B** — a decade learns cleanly on both wirings | run the atlas **learning** arms at that rate, and add the two wide **learning** arms at that rate (**+192 runs**, ~960 total) as a **rate-matched baseline**, so L.4's interaction compares learners at one rate. The floors are unaffected (nothing learns, so the rate is inert) and are reused. A registered two-key departure for L.4's learning arms, with this reason; L.5 is unchanged at 0.001 against L.1's baseline |
| **C** — a decade is better but does not learn cleanly | run **as registered**; the collapse is intrinsic to the substrate under this learner, and a `carries` reading on L.4 is expected to come back *carries or unlearnable* — which is then the finding |

No other setting is touched under any outcome. The check is calibration on disjoint seeds, read on
means and counts at four pairs, as V.3's `max_steps` calibration was; it decides how the campaign is
run and never what it reads.

### Rate-check outcome, 2026-09-18 — **B fired**

16 runs, all succeeded. Gain over own floor (foods), seeds 101–104, against the wide arms' gains at
the same seeds (+13.66 wt, +7.81 rn):

| wiring | rate | mean gain | seeds below floor | per-seed |
|---|---|---|---|---|
| wt | 0.0001 | **+15.02** | **0** | +9.44, +16.58, +16.47, +17.59 |
| wt | 0.001 | +2.24 | 2 | −2.52, −1.54, +6.73, +6.30 |
| wt | 0.01 | −1.61 | 4 | −2.62, −1.54, −0.38, −1.90 |
| rn | 0.0001 | **+16.05** | **0** | +11.66, +19.15, +17.47, +15.93 |
| rn | 0.001 | +2.81 | 2 | −6.16, −0.66, +11.19, +6.88 |
| rn | 0.01 | −0.64 | 3 | −6.16, −0.66, +6.73, −2.48 |

0.0001 is better than 0.001 on both wirings and **learns cleanly on both** — no seed below floor,
means above the wide arms' rather than merely half of them. 0.01 collapses everything. The atlas
collapse was a **rate mismatch**, and the mechanistic candidate — grounded inputs pushing the readout's
presynaptic activities toward saturation — was the right one. The honest prior above (A or C) was
wrong, and is left standing as written.

**Consequence, per the registered rule.** L.4's learning arms run at **0.0001**
(`..._readout_only_wide_atlas_r1e4{,_rewired_null}.yml`); two wide learning arms at 0.0001
(`..._readout_only_wide_r1e4{,_rewired_null}.yml`, one key from L.1's wide parents) are added as the
**rate-matched baseline**, +192 runs; the atlas floors and L.1's wide floors are reused, the rate
being inert under `freeze_updates`. L.5 is unchanged at 0.001 against L.1's baseline. The campaign is
**ten arms, 960 runs**. The harness takes a per-ablation baseline so L.4's interaction compares
learners at one rate. The rate check is the pilot for the swapped atlas arms — disjoint seeds, cleanly
learning, differing from their parents; the two new wide arms get their own 8-run pilot on the same
seeds before any registered seed.

**Not read at four seeds, and recorded so it cannot be read later as if it had been**: the atlas null
at 0.0001 gained as much as the atlas wild type. Whether the wide null does the same at 0.0001 is what
the rate-matched baseline exists to measure.

## The honest prior

**L.5: `survives_without_it`.** Gap junctions are symmetric and degree-scaled — the part of the wiring
most like the degree statistics the null preserves — and 034's verdict was that degree statistics are
what the wiring's endpoint contribution amounts to. The per-neuron effect is more likely to live in
the directed chemical graph. **L.4: uncertain, leaning `carries_the_effect` — but for the
operating-point reason as much as any feature reason**, which is why the diagnostic exists. B.1's
finding that grounded signs made a rule learn *worse* is not evidence either way about features.
Registered before any arm runs.

## The structural probe, registered before the correlation is computed

L.1 left one result unexplained: the per-neuron readout made the **null worse** (0.5106 → 0.3585)
while helping the wild type. The registered design predicted an interaction, not that direction of it.

**Hypothesis.** A per-neuron readout hurts a rewiring because degree-preserving rewiring
**decorrelates the inputs within each motor class**: the four class means average out
seed-specific input noise that 39 per-neuron weights fit instead. On the wild type, neurons within a
class share inputs, so per-neuron weights find structure rather than noise.

**Statistic.** For each wiring, the mean pairwise Jaccard of the presynaptic sets of the motor neurons
within a class (`m_chem[:, j]` for each pool neuron `j`), averaged over the four classes.

**What has been looked at.** Feasibility only: the wild type reads 0.068/0.140/0.175/0.231 across
VB/DB/VA/DA; three rewirings read 0.009–0.037. The statistic exists and discriminates. **The
correlation with L.1's outcome has not been computed**, and this document fixes the test before it is.

**Registered test.** Across the 96 rewirings, Spearman ρ between a seed's within-class Jaccard and
that seed's `rn_wide − rn_pooled` from L.1's committed `per-seed.csv`, **one-sided positive**, at
q = 0.05, with a registered minimum of **ρ ≥ 0.3**. A descriptive companion: the wild type's Jaccard
against the distribution of the 96 rewirings'. **A positive licenses the hypothesis for a follow-up
that manipulates within-class correlation directly; it is not a mechanism claim.** A null says the
puzzle is not this, and stays a puzzle.

## One BH-FDR family, ten tests

Per ablation: the interaction, the ablated wiring effect `wt_ablated − rn_ablated`, the two learning
gates against the ablated floors, and the untrained prior between the ablated floors — five. Two
ablations, **ten tests, one family**, corrected together as L.1's review required. The two
interactions share the baseline half, so they are positively dependent; BH-FDR holds under positive
dependence, and the shared half is why they are read separately rather than combined. The probe and
the L.4 floors diagnostic are single tests outside the family, because neither is a verdict input.

## Read per ablation, never pooled

L.4 and L.5 ask different questions. "The effect survives L.5 and not L.4" is the informative outcome
and must not be averaged into "the wiring half-carries the effect". Each gets its own reading in the
record.

## Read-only, and why

`l4_readout_width.py` produced the baseline this change reuses; editing it would let "the instrument
changed" compete with "the feature did not carry the effect". It, `connectome_structure_efficiency.py`
and `wiring_premise.py` are unmodified here, asserted by test against `origin/main` with the
shallow-clone skip guard from V.4. Read-only means **importable**: the new harness imports L.1's
`_two_sided`, `censoring`, reachability, orientation and family-adjustment helpers rather than copying
them, so the two cannot drift apart — copying is how the readout-width driver's first draft came to
disagree with the harness it was reading.
