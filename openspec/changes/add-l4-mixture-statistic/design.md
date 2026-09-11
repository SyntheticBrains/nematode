# Design: a statistic and a metric matched to the outcome's shape

## The shape, as it was described before the data that motivate acting on it

This design is constrained by a rule it is itself an instance of: choose the statistic before the
data exist. The statistic here is chosen from a description written on **2026-09-07**, in Logbook
042, four panels and five days before the endpoint evaluation that made acting on it urgent:

> Alignment finds a competent fixed point about as often on either wiring (16 against 11 of 48; R2
> cannot separate them). When it does, the wild-type's fixed points are better: its best eight
> seeds sit at 56–86% and the null's at 33–56%. The mean delta is carried by that upper tail, and a
> paired rank test, seeing a near-even sign split, reports no shift. **The wiring's mark is the
> *level* of the good fixed points, not their *frequency*.**

and, in the same logbook, the test that follows from it:

> A contrast on the upper mode (the mean among competent seeds, or an upper-quantile difference) is
> the natural test for what panels 2 and 3 show.

Both components — frequency and level — and the threshold that separates them
(`COMPETENT_THRESHOLD = 20.0`, committed in `scripts/analysis/l4_panel2.py`) predate 052. The
competent-fraction test was already *registered*, as panel 2's secondary, and already run. Nothing
in the family below was selected by looking at 052, and the threshold is not re-tuned here: a
threshold chosen now, with 052's per-seed values known, would be exactly the move this change
exists to prevent.

## The family

A mixture has two parameters a panel can move, and a test that reads only one of them will call the
other no effect. So the registered family has one member for each, plus the existing test:

| | member | statistic | reads |
|---|---|---|---|
| **F** | frequency | paired competent-fraction discordance at 20.0 — exact binomial on the discordant pairs | how *often* an arm lands competent |
| **L** | level | mean among seeds competent in **either** arm of the pair, paired, one-sided | how *good* it is when it does |
| **W** | all-seeds | the existing paired one-sided Wilcoxon + 80% bootstrap CI | the shift, if the outcome is unimodal after all |

BH-FDR across {F, L, W}. **W is retained deliberately**: it is what every committed table was
scored with, and dropping it would make the re-read incomparable with the record it re-reads.

**L's pairing rule is fixed here**: a pair enters L if *either* arm is competent. Restricting to
pairs where *both* are competent would condition on the outcome and discard exactly the pairs where
one arm found a good fixed point and the other did not — which is the effect. With no qualifying
pair, L is undefined and reported as such, not as a null.

n = 8 or 16 per panel gives a mixture model no usable power; F and L are two marginal readings of a
mixture, not an attempt to fit one. That limit is stated in the record rather than discovered.

## The metric

`plateau_tail` has always returned mean foods (0–10) beside the full-clear percentage, and every
committed per-seed CSV carries it. The cliff metric — ten of ten or nothing — cannot distinguish a
policy that reaches eight foods from one that reaches zero, and on a foraging task that is most of
the behaviour.

So the graded metric is read as a **parallel family** with the same three members, BH-FDR within
itself. The full-clear metric stays **primary**, because every committed verdict is in its units
and this change does not restate them. The outcome map names what a disagreement between the two
families means, so "graded improves, cliff does not" is a registered reading rather than a
consolation.

## The outcome map

Ordered; the first matching branch is the verdict.

| verdict | condition | licenses |
|---|---|---|
| `shift` | F and L both significant, same direction | the effect is unambiguous |
| `level_only` | L significant, F not | 042's finding: better fixed points, not more of them |
| `frequency_only` | F significant, L not | more competent seeds, no better |
| `mixed_response` | **neither** significant, and the arm both improves ≥ 1 seed above the comparator and degrades ≥ 1 below it by more than the hold band | **nothing** |
| `no_effect` | neither significant, no such split | the contrast is closed on this evidence |
| `degrades` | the reverse direction is significant on F or L | — |

`mixed_response` is the branch this change exists to create, and it is deliberately **sterile**: it
licenses no follow-on, gates nothing open, and requires its own registration to act on. A named
outcome that licenses nothing is an honest description; an unnamed one discovered after the fact is
a story. It is also not a way to avoid `no_effect` — it fires only on an explicit two-sided split,
not on any panel that happens to miss significance.

## The re-read, and its limits

Every committed table (040, 041, 042, 044, 045, 046, 047, 050, 052) is re-read under both families
from its committed per-seed CSV — not from a campaign directory, which 042 recorded as the practice
that made its analysis reproducible.

**The re-read is descriptive and cannot overturn a registered verdict.** Each stands exactly as
registered, in its own units, under the rule it was registered with. The re-read is a second,
pre-specified reading placed beside it, and it is an input to I.4's ladder re-read and to nothing
else. Where the two disagree, both are reported; the registered one is the verdict.

Two tables need their protocol stated rather than assumed. 045, 050 and 052 are *assays*, not
panels: their comparator is a committed per-seed table and their rule is a hold band, so F and L
are computed against each seed's own committed clone and W is the paired test against it. 044's
contrast is a prior sweep over frozen arms. The re-read reports each in its own protocol's terms
and does not pool across them.

## Alternatives considered

- **Fit a two-component mixture.** The honest model of the outcome, and unusable at n = 8: the
  component means and the mixing weight are three parameters, and 052's split is 6–2. F and L are
  the marginals a panel this size can actually estimate.
- **Replace the cliff metric.** Rejected: every committed verdict is in its units, and restating
  them is a different and much larger change. Beside, not instead.
- **Re-tune the competence threshold.** Rejected, and the reason is the change's own premise — 20.0
  is committed and was applied before 052 existed.
- **Drop W.** Rejected: it is the only member comparable with the committed record.
- **Do this after the low-σ programme**, which 052's verdict licenses. Rejected: that programme
  ends in an assay scored by the statistic under repair, so it would spend campaign time to produce
  a number this change would then have to re-read.
