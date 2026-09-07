# 041: L4 Panel 2 — The Hebbian Wiring Contrast, the Prior over Policies, and Count-Scaled Initialisation (7a-i / Phase 7)

**Status**: completed — **`inconclusive`** under the pre-registered verdict map. The one signal
[Logbook 040](040-l4-panel.md) left on the table — reward-free Hebbian alignment finds better
fixed points on the real wiring than on its degree-matched scramble — **held its size on fresh
seeds and lost its test**: wild-type Hebbian over rewired-null Hebbian is **+14.1** points
(CI[+2.7, +25.0], 10/16 seeds; +16.2 on panel 1's seeds 1–8, +11.9 on new seeds 9–16), but the
outcome is bimodal on both arms, so a paired Wilcoxon at n = 16 reaches only q = 0.28. A 64-seed
prior sweep puts the *untrained* wild-type only **+3.3** above its null (competent fractions 0.22
vs 0.16): descriptively, the wild-type's advantage is created by the Hebbian fixed point, not
present in the prior. And the panel's initialisation factor **reverses**: initialising chemical
weights in proportion to Cook 2019's synapse counts halves the wild-type Hebbian fixed point
(31.8% → 15.6%, CI entirely below zero) and erases the wiring contrast, while leaving every frozen
prior where it was. The anatomy's count structure is something the untrained network does not use
and this local rule uses badly.

**Branch**: `feat/l4-panel2` (PR #320).

**Date**: 2026-09-07.

**OpenSpec change**: `add-l4-panel2-hebbian-wiring` (archived; extends capability
`l4-plasticity-panel` and adds `weight_init` to `connectome-ppo-brain`).

## Objective

Three questions Logbook 040 raised and could not answer at n = 8: (1) is the descriptive
wild-type-over-rewired advantage under the unmodulated Hebbian rule real; (2) what is the *prior
over policies* each wiring imposes on random initialisations, against which every plasticity
result on this cell has to be read; (3) does the one piece of wiring data the substrate discards —
the per-edge synapse count — change either.

## Background

Panel 1 ([040](040-l4-panel.md)) resolved to `sanity_floor_fail` and showed that outcomes on the
C3 cell are fixed points seeded by the random initial weights: the frozen wild-type arm spans 0.4%
to 36.9% across seeds on identical wiring, and the reward-free Hebbian floor reaches 64–78% on
three seeds. Its descriptive wild-type Hebbian minus rewired Hebbian was +16.5 on 5/8 seeds while
the frozen floors tied. Cook 2019's chemical synapse counts (1–75, median 3, a third of edges
single-synapse) never reached a weight: every edge into a neuron with `k` inputs was drawn from
`N(0, 1/√k)`.

Ratified ordering after 040: this panel, then the imitation warm start (S.2), then 7a-ii with a
structured third factor.

## Hypothesis

Pre-registered before any run (the archived change; `supporting/041-l4-panel2/launch.md` committed
before the campaigns): four one-sided paired tests corrected together — **P1** wild-type Hebbian

> rewired Hebbian under degree-scaled initialisation (the primary, seeds 1–16); **P2** the same
> under count-scaled initialisation; **P3** count-scaled > degree-scaled on the wild-type Hebbian
> arm; **P4** wild-type frozen > rewired frozen over 64 sweep seeds (the prior). The verdict is
> assigned from P1 alone in the rewired-null control's vocabulary (`specific_wiring`,
> `rewired_beats_wild_type`, `degree_statistics`, `inconclusive`; `insufficient_seeds` first); P2–P4
> annotate and never change it. Every arm reports the distribution of plateau tails and the
> **competent fraction** (seeds at or above 20% with no learning). No pilot: every value the arms
> run with is panel 1's pin.

**The count-scaling law**, ratified over a logarithmic alternative: each chemical weight is a
standard-normal draw times `n / sqrt(Σ n²)` over the post-synaptic neuron's incoming counts —
magnitude linear in count (each contact adds conductance), sign still random (transmitter identity
is 7a-ii's), and every neuron's expected squared incoming norm exactly 1, the degree-scaled
expectation, so homeostatic targets and the bound are unchanged and only the structure *within* a
neuron's inputs differs. Under the rewiring each count travels with its edge, so the rewired-null
is a null of the count structure too.

## Method

Eight arms: wiring {wild-type, rewired-null} × initialisation {degree-scaled, count-scaled} × rule
{frozen, unmodulated Hebbian}; the reward-modulated arms and the MLP were left out until the prior
is known. Hebbian arms on seeds 1–16 at 1000 episodes (panel 1's Hebbian runs settled within their
first block); frozen arms on seeds 1–64 at 600 episodes as the prior sweep, seeds 1–16 doubling as
the learning-gain floors. Ranked metric and statistics: the committed plateau-tail full-clear
success, paired one-sided Wilcoxon, 80% bootstrap CI, BH-FDR — through panel 1's reader, so the
panels are measured identically. Harness `scripts/analysis/l4_panel2.py`, its family, verdict
map and sweep analysis fixed in code before the runs; 320 runs in 2 h 11 min on 16 workers.

**Reproduction check**: the degree-scaled arms run panel 1's configs at panel 1's seeds and
nothing in the brain or the rule reads the total budget, so their episode streams should coincide
with panel 1's. They do: 32 of 32 comparisons (four arms × seeds 1–8, the frozen arms' 600 and the
Hebbian arms' first 1000 episode lines) are identical.

Three Hebbian runs had no detected plateau at 1000 (wild-type 14, wild-type-count 3,
rewired-count 10) and received the single registered extension, a fresh run at 1500; all three
converged within four points of their shorter values, and the family was unchanged.

## Results

### The registered family (paired seeds, BH-FDR α = 0.05)

| test | contrast | mean Δ | 80% CI | q | +seeds | result |
|---|---|---|---|---|---|---|
| P1 | wt_hebbian − rn_hebbian (degree init) | +14.1 | +2.7 … +25.0 | 0.28 | 10/16 | fail |
| P2 | wt_hebbian − rn_hebbian (count init) | +2.0 | −7.6 … +11.8 | 0.47 | 8/16 | fail |
| P3 | wt_hebbian_count − wt_hebbian | −15.6 | −27.1 … −2.5 | 0.92 | 6/16 | **reverse** |
| P4 | wt_frozen − rn_frozen (64 seeds) | +3.3 | −0.0 … +6.8 | 0.28 | 36/64 | fail |

**Verdict: `inconclusive`** (P1's interval clear of zero, q above α). Annotations: count-scaled
initialisation does not preserve the contrast, does not improve the wild-type fixed point, and the
prior does not differ at the registered level.

### Per-arm plateau-tail full-clear success (%)

| arm | n | mean | sorted per seed | competent |
|---|---|---|---|---|
| wt_hebbian | 16 | 31.8 | 0 0 0 0 3 6 16 24 26 31 41 68 69 70 71 80 | 9/16 |
| rn_hebbian | 16 | 17.4 | 0 0 0 2 6 6 7 14 17 18 18 24 31 43 46 48 | 5/16 |
| wt_hebbian_count | 16 | 15.6 | 0 0 0 0 0 1 1 2 3 8 9 14 19 49 67 81 | 3/16 |
| rn_hebbian_count | 16 | 14.2 | 0 0 0 0 0 0 1 1 2 2 3 30 36 43 49 56 | 5/16 |
| wt_frozen | 64 | 11.3 | median 3.7, q75 14.2, max 74.0 | 0.22 |
| rn_frozen | 64 | 8.0 | median 2.3, q75 10.8, max 43.3 | 0.16 |
| wt_frozen_count | 64 | 10.9 | median 5.3, q75 13.0, max 52.0 | 0.16 |
| rn_frozen_count | 64 | 7.7 | median 2.7, q75 8.3, max 52.0 | 0.11 |

Learning gains (Hebbian minus own frozen, seeds 1–16): wild-type **+25.7** (11/16), rewired +9.6
(9/16), wild-type-count +9.3 (7/16), rewired-count +9.2 (8/16). Frozen count-minus-degree pairs:
−0.4 (wild-type) and −0.3 (rewired), both intervals spanning zero.

## Analysis

1. **The primary is an effect the test cannot carry.** Doubling the seeds did not shrink it
   (+16.2 on the original eight, +11.9 on the new eight, five positive in each half), and the
   80% interval excludes zero. But the per-seed deltas have a spread of 36.5 points — ten seeds
   between +6 and +70, six between −15 and −46 — because both arms are bimodal: alignment either
   finds a competent fixed point or a dead one, and which happens is set by the seed. A rank test
   on such an outcome needs several times this sample, or a different statistic (the competent
   fraction: 9/16 against 5/16, and the wild-type's upper mode at 68–80% against the null's
   43–48%, are the numbers that carry the difference).
2. **Where the wild-type's advantage lives.** The untrained prior differs by +3.3 with the
   interval touching zero; after Hebbian alignment the difference is +14.1, and the wild-type's
   learning gain is +25.7 against +9.6. Most of the advantage is *made* by reward-free alignment
   on the real wiring, not present before it. This is the design's flagged interesting case, and
   it is descriptive: neither P1 nor P4 confirmed.
3. **The prior over policies is broad, skewed, and slightly better on the real wiring.** On
   either wiring most random initialisations are dead (median 2–4%), a fifth are competent
   without any learning, and the wild-type's upper tail is heavier (74, 49, 47, 44 against 43,
   43, 41, 41). Every plasticity result on this cell is read against a floor whose seed-to-seed
   range is 0–74 points; a plastic arm's mean over eight seeds is mostly a statement about which
   seeds it drew.
4. **Synapse counts, used linearly, hurt the Hebbian fixed point and leave the prior alone.** The
   frozen arms are unchanged under count-scaled initialisation (the untrained network does not
   exploit the count structure either way), but the wild-type Hebbian arm falls from 9 competent
   seeds to 3 and its learning gain from +25.7 to +9.3, while the rewired arm barely moves. Linear
   scaling concentrates each neuron's input on its few high-count edges (the largest edge carries
   a median 29% of a neuron's input); Hebbian alignment then amplifies inputs that are already
   dominant, and on the wild-type wiring the dominant count structure is not the correlation
   structure the reward-free fixed point exploited under equal magnitudes. Two readings are open
   and neither is testable here: the counts carry information a *sign-aware* rule could use
   (7a-ii's atlas gives signs), or the linear law over-weights the tail and a compressive law
   would not.
5. **Method.** The reproduction check is worth keeping: it proved, at no cost, that nothing
   between the two panels perturbed the runs. The registered extension fired three times and
   changed nothing, which is what an extension rule should do.

## Conclusions

- The Hebbian wiring contrast is **not confirmed**: `inconclusive`, with an effect that held its
  size on fresh seeds and a test underpowered for a bimodal outcome. The claim type is performance
  and the result is descriptive.
- The prior over policies on this cell is broad and nearly the same on both wirings; the wiring's
  advantage under the unmodulated rule, where it exists, is created by alignment.
- Count-scaled initialisation as registered (linear, random signs) is **harmful** to the Hebbian
  fixed point and neutral to the prior. It is not a free improvement to carry into the plastic
  arms; the follow-ups noted in 040 (sparse random MLP; a rule with anti-Hebbian/decorrelating
  terms) stand, and count structure returns only with signs (7a-ii).

## Limitations

- n = 16 on a bimodal outcome; the registered statistic is a rank test on paired deltas whose
  spread exceeds the effect twice over. The competent fraction was descriptive, not a registered
  test.
- The prior sweep pairs seeds across wirings, but the two arms' draws land on different edges by
  construction; P4 is as much a comparison of two distributions as a paired test.
- One count law (linear) was run. The frozen priors say the law does not matter for the untrained
  network; the Hebbian arms say it matters for the rule, and a compressive law was not tried.
- Two hundred and fifty-six frozen runs at 600 episodes estimate each prior tail on 150 episodes.

## Next Steps

**S.2**, the imitation warm start — the good initial policy this cell's prior does not supply on
most seeds — then **7a-ii** with the third factor made *structured* (pathway-specific instruction
through the receptor atlas), where sign identity would let the count structure be revisited.
Queued behind them: a sparse random MLP arm; a rule variant with anti-Hebbian/decorrelating
terms; if the Hebbian wiring contrast is to be settled as a registered claim, the competent
fraction (or the upper-mode level) as the statistic, at a seed count set from this panel's spread.

## Data References

- Registration and design: `openspec/changes/archive/2026-09-07-add-l4-panel2-hebbian-wiring/`;
  capabilities `openspec/specs/l4-plasticity-panel/spec.md`,
  `openspec/specs/connectome-ppo-brain/spec.md` (count-scaled initialisation).
- Everything the panel produced:
  [supporting/041-l4-panel2/](supporting/041-l4-panel2/details.md) — `panel2.json`,
  `per-seed.csv`, `curves.csv`, `_manifest.txt` (with the three extensions), `launch.md`,
  `details.md`.
- Harness: `scripts/analysis/l4_panel2.py`; initialisation: `brain/arch/connectome_ppo.py`
  (`weight_init`); the four `_countinit` configs under
  `configs/scenarios/foraging_predator_thermal/`.
