# I.1c step 0 — the endpoints, perturbation off

**Run 2026-09-11. Verdict: `fail` under the registered rule. Integrity check clean.**
**Licenses the low-σ programme.**

## The integrity check, read first

Every seed reproduced the cosine the assay recorded for it, to three decimals:

| seed | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| recorded | 0.688 | 0.725 | 0.722 | 0.678 | 0.691 | 0.748 | 0.679 | 0.649 |
| seen | 0.688 | 0.725 | 0.722 | 0.678 | 0.691 | 0.748 | 0.679 | 0.649 |

No void seeds. These are the endpoints, not the clones.

## The result

| | mean | vs comparator | within hold | at or above |
|---|---|---|---|---|
| endpoint, perturbation off | **20.6** | −18.1 | 2/8 | 2/8 |

The comparator is 38.7 and the rule requires a mean within 5 points with 6 of 8 seeds within 10.
**It fails.** The trajectory is flat (20.2 → 20.6 across the run), as a frozen policy's must be.

**Removing the perturbation is worth +8.6 points**: the same weights scored 12.0 while the arm's
own σ = 0.2 was running. So a substantial perturbation tax is real and was being paid — and
removing it does not come close to closing the 18.1-point shortfall.

## The finding the mean hides: the effect is bimodal

| seed | own clone | endpoint | delta |
|---|---|---|---|
| 1 | 39.3 | 8.4 | −30.9 |
| **2** | 44.0 | **73.4** | **+29.4** |
| 3 | 40.0 | 26.8 | −13.2 |
| 4 | 21.3 | 0.0 | −21.3 |
| 5 | 47.1 | 10.0 | −37.1 |
| 6 | 33.3 | 6.2 | −27.1 |
| 7 | 61.3 | 2.8 | −58.5 |
| **8** | 23.3 | **37.6** | **+14.3** |

Six seeds were degraded, two were **substantially improved**. Seed 2's 73.4% is the highest
full-clear rate this substrate has produced in the phase — above every committed frozen clone,
whose best is seed 7 at 61.3 — and it is stable, not a tail artefact: 71.0% over its first 500
episodes and 73.4% over its last, 1469 of 2000 overall, on a frozen policy whose process is
stationary by construction.

Mean 20.6, **median 9.2**. A paired one-sided Wilcoxon against each seed's own clone gives
**p = 0.074** — this panel does not even establish degradation at the 5% level, because two seeds
go the other way hard.

Nothing obvious predicts the split. Correlation of the per-seed delta with the cosine to the clone
is **+0.08** (moving less did not help), and with the clone's own quality **−0.47** (the best clone,
seed 7 at 61.3, was damaged worst; the weakest, seed 8 at 23.3, improved). The endpoint score does
track the under-perturbation score at **+0.84**, so the assay's per-seed ordering was informative
about the endpoint even though its level was not.

## What this establishes

- **The registered verdict is a fail, and the sequence it licenses is the low-σ programme.** The
  rule did not leave behind a policy that is competent once the noise is removed, on the panel as
  a whole. The I.1 assay fail is not merely an evaluation artefact.
- **But "the rule rewrites a competent policy into a worse one" is false as a general statement.**
  On 2 of 8 seeds it took a competent policy and made it substantially better, one of them to the
  best number this substrate has recorded. That is the first evidence in this phase of the rule
  *improving* a real policy on the real substrate rather than merely not destroying one.
- **I.2's premise is now demonstrated rather than argued.** Five panels applied a paired rank test
  to a bimodal outcome; here the bimodality is explicit, the mean and median differ by a factor of
  two, and the registered statistic returns p = 0.074 on a panel whose mean is 18 points down.
  A mixture-aware or competent-fraction contrast is not a refinement — on this panel it is the
  difference between "degrades policies" and "degrades six, transforms two".

## What it does not establish

- Not that the rule can build a policy. Every arm here started from a competent clone.
- Not *why* two seeds improved. Nothing measured here predicts it, and with n = 2 the honest
  statement is that it happened, not that it is understood.
- Not that the low-σ programme will help. It is licensed by the registered rule, not by evidence
  that a smaller scale is what those six seeds needed.

## Descriptive annotation

Endpoint against the same seed's score while its perturbation ran (mean 12.0 → 20.6):

| seed | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| under perturbation | 7.0 | 24.6 | 12.4 | 2.8 | 7.2 | 3.0 | 10.2 | 29.0 |
| perturbation off | 8.4 | 73.4 | 26.8 | 0.0 | 10.0 | 6.2 | 2.8 | 37.6 |
| tax recovered | +1.4 | +48.8 | +14.4 | −2.8 | +2.8 | +3.2 | −7.4 | +8.6 |
