# 110: Phase 10 — The Benchmark's Controls: Held-Out Worms Pass, Six Negatives Fail, and the Benchmark Is Valid

**Status**: completed. **B.4: the benchmark is VALID.** Held-out real worms pass, by the registered allowance, and all six negatives fail the wild type. This is the first reading of the held-out data. Every rule was fixed in a launch record committed before it ran.

> | Reading | Result | Rule |
> |---|---|---|
> | Positive controls (15 units) | **14 pass**; tph-1 fails on leaving | at most 2 may fail |
> | Wild type: small lawn (573 animals), uniform well (31), 10 López-Cruz batches | all pass; every z within ±1.29 | bar 2.84 |
> | Negatives (n = 200 each) | **all six fail**; D 4.49–8.02 | each must fail |
> | W's minimum | **1.42** (95th percentile; median 0.89) | B.3's rule |

**Date**: 2026-10-25.

**OpenSpec change**: `add-benchmark-controls`.

**Records** (in [`supporting/110-benchmark-controls/`](supporting/110-benchmark-controls/)):

- `launch.md`: the registration, committed in `d814eac8` before the read;
- `negatives_params.json`: the matched parameters, from band data;
- `pilot.json`: the negatives on pilot seeds, as description;
- `controls.json`: the reading.

## Objective

D34 makes the benchmark valid only if real held-out worms pass it and non-worms fail it. Otherwise it is void, and it is revised before anything is graded. B.4 reads the cohorts B.1 held out, for the first time, under rules fixed in advance.

The maintainer's decisions of 2026-10-25:

- a registered allowance across the controls;
- negatives matched to band data;
- the untrained policy as both arms, 8 networks × 25 animals each;
- random walks matched to the wild type's speed.

## What was built

- **Two batched negative controllers** (`evaluation/controllers.py`):

  - `BatchedStateHazards`, an open-loop two-state worm switching by a fixed chain every 10 s;
  - `BatchedCurvatureWalk`, runs turning by uniform or AR(1) curvature.

  Both draw from each animal's own generator, so an animal acts the same alone or in a batch.

- **`benchmark_negatives_params.py`**, the matched parameters from the OD 2 band cohorts.

  - Curvature and persistence are read only on seconds that moved at least 0.01 mm, since slower on-lawn headings are tracking noise.
  - Hazards are onsets per second outside every bout.

- **`benchmark_controls.py`**:

  - `pilot` runs the negatives on band data only;
  - `read` refuses to run until the launch record is committed and unchanged. It then grades the controls, runs the negatives, reads W's minimum and writes the verdict.

## The matched parameters

From 1,013 band animals:

| | Speed (mm/s) | Curvature SD (°/s) | Persistence | Reversal hazard (/s) | Omega hazard (/s) |
|---|---|---|---|---|---|
| dwelling | 0.022 | 61 | −0.06 | 0.088 | 0.0001 |
| roaming | 0.067 | 35 | −0.01 | 0.080 | 0.0010 |
| pooled | 0.027 | 57 | −0.05 | 0.087 | 0.0002 |

- The authors' HMM switches dwelling → roaming with probability 0.021 per 10 s, and back with 0.102. Its stationary roaming fraction is 0.17.
- **Persistence is near zero,** so the correlated walk is close to independent turns. It is recorded as measured, not tuned.
- **On-lawn reversals start about 5 times a minute in both states,** close to B.2's flag count.

## The positive controls

| Unit | n | Result |
|---|---|---|
| Small lawn, 24 held-out dates | 573 | passes; z from −1.29 (exits 0.53/h) to +1.29 (time on the lawn 0.984) |
| Uniform well, 1 held-out date | 31 | passes; roaming 0.247 (z 0.24) |
| López-Cruz, 10 held-out batches | 39–84 each | all 10 pass; γ 0.094–0.157 per min, early/late 2.97–6.17, presence bounds 2.33–4.89 |
| tph-1 (2 paired dates, against N2) | 35 | **fails**: Δ roaming +0.337 (z 0.62, "more", as resolved); **Δ exits +4.04/h (z 6.37)** |
| pdfr-1 (3 paired dates) | 74 | passes: Δ roaming −0.105, Δ exits +0.59 |
| tax-4 (6 paired dates) | 169 | passes: Δ roaming −0.123 ("less", as resolved), Δ exits +0.62 |

**One unit fails, within the allowance of 2.** Under the registered rates, P(more than 2 fail) is 0.025.

**tph-1's failure is about its reference, not the rule.** On its three band dates tph-1 left the lawn about as often as N2 (Δ +0.07/h). On its two held-out dates it left 4.0 times more per on-lawn hour. The roadmap expects tph-1 to leave more, and the held-out dates show it. Three band dates were too few to see it, and the reference's thin-arm inflation (√2 × the wild type's scale) could not cover a between-date difference that large. **The tph-1 leaving reference is unreliable for C's mutant grading,** and the tracker carries it.

## The negatives

| Negative | D | D_fit | Families passed | Local search present |
|---|---|---|---|---|
| open-loop HMM worm | 5.70 | 4.77 | none | no |
| Poisson worm | 7.68 | 7.91 | none | no |
| random walk | 8.02 | 8.41 | none | no |
| correlated walk | 6.96 | 8.39 | none | no |
| untrained MLP | 5.78 | 4.92 | uniform well | yes (bound 1.0015) |
| untrained connectome | 4.49 | 5.27 | local search | yes (bound 1.006) |

**All six fail the wild type.** Each sits more than three times above W's minimum of 1.42.

**What discriminates, as description:**

- **Staying on the small lawn decides every negative.** Their time on the lawn runs from 0.07 to 0.40, against 0.98, and their exits per on-lawn hour from 9.6 to 31, against 0.7. This is D34's closed-loop contingency: an open-loop worm cannot stay on a 3 mm lawn.
  - The HMM worm matches the wild type's roaming fraction (0.12) and dwelling bouts.
  - The untrained MLP matches its roaming and dwelling bouts on the uniform well.
- **Local search alone is weak.**
  - The untrained connectome passes it with γ = 0.21 per min (z 2.4) and early/late of 1.05 (z −1.9, against the wild type's 4.2).
  - The presence test passes it too, at a bound of 1.006: a population of 200 shows a decline that small with confidence.
  - The local-search band's width (early/late scale 1.62) is the López-Cruz batches' real spread, widened to a prediction interval. **D_fit leaves local search out, and its family should not be relied on alone.**

## W's minimum

The 24 held-out wild-type dates were split into halves 1,000 times, together with halves of the 10 held-out batches. The distance two real halves show by chance has a median of 0.89 and a 95th percentile of **1.42**. That is W's registered minimum effect on D (small lawn and local search, the uniform well left out). Every negative's D is 3–6 times larger.

## Conclusions

1. **The benchmark is VALID.** Held-out real worms pass it, 14 of 15 units with one allowed failure. Six negatives, matched and trivial, fail it.
2. **The closed-loop statistics carry it.** Staying on and leaving the lawn separate every negative from the animal. Local search alone does not.
3. **tph-1's leaving reference is unreliable.** Its three band dates miss an effect its two held-out dates show strongly.
4. **W's minimum is 1.42 on D,** the real between-batch floor.

## Next steps

- [ ] F.0: parameter recovery for both fitting methods.
- [ ] C's mutant table: tph-1's leaving reference rests on three band dates that disagree with its held-out dates. Grade it by magnitude with that stated, or leave it out. That is a decision at D37's registration.
- [ ] C.2's weights: `time_on_lawn` and exits dominate the discrimination. With `time_on_lawn`'s scale of 0.006, D_fit's weighting needs care (tracker).
