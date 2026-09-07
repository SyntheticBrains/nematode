# 042: L4 Panel 3 — Replicating the Hebbian Wiring Contrast on Fresh Seeds (7a-i / Phase 7)

**Status**: completed — **`inconclusive`** under the pre-registered verdict map, and the Hebbian
wiring contrast is closed as a registered question. On 48 fresh paired seeds the wild-type
connectome under the unmodulated Hebbian rule beats its degree-preserving rewired-null by
**+8.1** points (80% CI[+2.3, +14.2]) with 24 positive and 22 negative paired deltas, so the
registered rank test reaches only q = 0.19; the registered secondary, competent-fraction
discordance (11 wild-type-only against 6 rewired-only), reaches q = 0.19 too. The estimate has
shrunk with every fresh look — +16.2 on panel 1's seeds, +11.9 on panel 2's fresh seeds, +8.1
here — and pooled over all 64 seeds sits at **+9.6** (CI[+4.3, +15.0]). What the replication
did settle is *where* the wiring's advantage lives: in the **level** of the good fixed points
(the wild-type's best eight seeds at 56–86% against the null's 33–56%), not in how often
reward-free alignment finds one (16 against 11 of 48). A paired rank test cannot see that
shape, and the registration forbids extending further on this hypothesis.

**Branch**: `feat/l4-panel3` (PR #322).

**Date**: 2026-09-07.

**OpenSpec change**: `add-l4-panel3-hebbian-replication` (archived; extends capability
`l4-plasticity-panel`).

## Objective

Close the one question [Logbook 041](041-l4-panel2.md) left open and cheap: is the reward-free
Hebbian wiring contrast — the only wiring-specific signal to survive since
[034](034-connectome-structure-controls.md) found the wiring inert under gradient learning —
real, at a sample size that can carry it, on seeds never used for it?

## Background

Panel 2 measured the contrast at +14.1 over seeds 1–16 with the interval clear of zero and
q = 0.28; both arms were bimodal (alignment finds a competent fixed point or a dead one, set by
the seed) and the paired deltas spread 36.5 points. Seeds 1–16 had been seen. Panel 2's prior
sweep had run the frozen arms on seeds 1–64, so floors for 17–64 already existed. Ratified
2026-09-07 as the item before S.2 because the recipe was pinned, the cost was an hour, and the
imitation warm start changes the initial policy on every seed, after which the reward-free
question cannot be revisited cheaply.

## Hypothesis

Pre-registered before any run (`supporting/042-l4-panel3/launch.md` committed first): two
one-sided paired tests corrected together — **R1** wild-type Hebbian > rewired-null Hebbian by
the committed paired Wilcoxon on seeds 17–64, the primary; **R2** the same direction in
competent-fraction discordance, an exact binomial on the discordant pairs at panel 2's 20%
threshold. Verdict from R1 alone in panel 2's vocabulary; R2 annotates. Seeds 1–16 enter only a
pooled descriptive summary. Registered power, stated in advance: roughly 75–85% for R1 at the
raw threshold and 63–75% at the BH-corrected one for an effect of +12 to +14; R2 projected from
panel 2's discordance (8 against 4) at roughly 55–65%.

## Method

Panel 2's two degree-scaled Hebbian arms, unchanged, on paired seeds 17–64 at 1000 episodes
(`rewire_seed` derived from the run seed); no pilot, no other arm. Frozen floors for 17–64 and
the Hebbian values for 1–16 read from panel 2's committed `per-seed.csv`, so the analysis
reproduces from the repository alone. Harness `scripts/analysis/l4_panel3.py`, fixed in code
before the run: seed range enforced, the family, the verdict, the pooling, a completeness flag
per test, the extension list. 96 runs in 62 minutes on 16 workers; no run needed the registered
extension.

## Results

### The registered family (paired seeds 17–64, BH-FDR α = 0.05)

| test | statistic | value | p | q | result |
|---|---|---|---|---|---|
| R1 | mean Δ, 80% CI, +seeds | +8.1, [+2.3, +14.2], 24/48 | 0.19 | 0.19 | fail |
| R2 | b wild-type-only / c rewired-only / both | 11 / 6 / 5 | 0.17 | 0.19 | fail |

**Verdict: `inconclusive`** (R1's interval clear of zero, q above α; R2 does not confirm).

### Per-arm plateau tails, seeds 17–64

| arm | mean | median | q75 | max | competent (≥ 20%) |
|---|---|---|---|---|---|
| wt_hebbian | 19.9 | 4.2 | 34.2 | 86.0 | 16/48 |
| rn_hebbian | 11.9 | 2.8 | 15.7 | 56.0 | 11/48 |

Paired deltas: 24 positive, 22 negative, 2 ties; spread 32.2. Learning gains over each arm's own
floor: wild-type +6.9 (21/48), rewired +3.9 (29/48).

### Pooled 1–64 (descriptive)

Mean Δ **+9.6**, CI[+4.3, +15.0], 34/64 positive; discordance 19 against 10; competent fractions
0.39 against 0.25; learning gains +11.6 against +5.3.

## Analysis

1. **The estimate shrank with each fresh look, then stabilised.** +16.2 (seeds 1–8, panel 1's
   runs), +11.9 (9–16), +8.1 (17–64). Panel 1's descriptive signal was the most favourable draw.
   The pooled 64-seed estimate of +9.6 with an interval clear of zero is the best statement of
   the effect's size; it is small.
2. **The null is the sign pattern, not a smaller mean.** The Wilcoxon signed-rank statistic
   uses only the signs and the ranks of |Δ|, so it is invariant to scaling every delta: a +8
   effect with panel 2's sign pattern would have passed exactly as a +14 would. Resampling panel
   2's sixteen deltas to n = 48 (20,000 draws) reproduces the registered power — 0.84 at the raw
   threshold, 0.74 at the corrected one. What differed on the fresh seeds was the balance of
   signs: 24 against 22 here, 10 against 6 in panel 2. The registered power was real; the
   outcome that fell outside it was the bimodality the design itself flagged.
3. **Where the wiring's advantage lives.** Alignment finds a competent fixed point about as
   often on either wiring (16 against 11 of 48; R2 cannot separate them). When it does, the
   wild-type's fixed points are better: its best eight seeds sit at 56–86% and the null's at
   33–56%. The mean delta is carried by that upper tail, and a paired rank test, seeing a
   near-even sign split, reports no shift. The wiring's mark is the *level* of the good fixed
   points, not their *frequency*.
4. **A statistic for that shape was not registered, and is not registered now.** A contrast on
   the upper mode (the mean among competent seeds, or an upper-quantile difference) is the
   natural test for what panels 2 and 3 show; it is also a choice made after three looks at the
   data. Registering it would need another 48 fresh seeds for a structure claim under a
   reward-free rule that the design labelled the weaker result from the start. The registration
   forbids extending on this hypothesis, and the lesson is carried forward instead: **on a
   bimodal outcome, register a statistic matched to the shape before the data exist.**
5. **Method.** Reading the floors and the pooling from a committed table, rather than a
   gitignored campaign directory, made the analysis reproducible from the repository; the
   harness refuses an incomplete table rather than dropping seeds silently. The registered
   extension rule fired zero times.

## Conclusions

- The Hebbian wiring contrast is **not confirmed** after 64 paired seeds and is **closed as a
  registered question**. The effect is real in the pooled interval and small (about +9 points),
  and it lives in the level of the wild-type's good fixed points rather than in their frequency.
  The claim type is performance; the result is descriptive.
- Nothing here changes the ordering. The dominant fact across panels 1–3 is that initialisation
  decides the outcome and most random initialisations are dead; the imitation warm start (S.2)
  supplies a competent policy on every seed and is on the path to the reward-modulated headline.

## Limitations

- Two registered tests, both rank- or count-based, on an outcome whose signal is in the upper
  tail; the mean's interval is reported but was not a registered test.
- Pooling seeds 1–16 mixes seen and fresh seeds; it is labelled descriptive and read as such.
- The floors are 600-episode runs from a different campaign; the Hebbian arms are 1000-episode
  runs. Learning gains compare plateau tails on different windows.

## Next Steps

**S.2**, the imitation warm start, then **7a-ii** with the third factor made structured. Queued
behind them: a sparse random MLP arm; a rule variant with anti-Hebbian/decorrelating terms. For
any future registration on this cell: choose the statistic for a bimodal outcome in advance.

## Data References

- Registration and design: `openspec/changes/archive/2026-09-07-add-l4-panel3-hebbian-replication/`;
  capability `openspec/specs/l4-plasticity-panel/spec.md`.
- Everything the panel produced:
  [supporting/042-l4-panel3/](supporting/042-l4-panel3/details.md) — `panel3.json`,
  `per-seed.csv`, `curves.csv`, `_manifest.txt`, `launch.md`, `details.md` (with the power
  computation's method).
- Inputs: `supporting/041-l4-panel2/per-seed.csv` (floors for 17–64; Hebbian values for 1–16).
- Harness: `scripts/analysis/l4_panel3.py`.
