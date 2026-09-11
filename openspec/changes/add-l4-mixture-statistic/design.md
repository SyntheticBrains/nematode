# Design: a statistic and a metric matched to the outcome's shape

## The shape, as it was described before the data that motivate acting on it

This design is constrained by a rule it is itself an instance of: choose the statistic before the
data exist. The statistic here is chosen from a description written on **2026-09-07**, in Logbook
042, four panels and four days before the endpoint evaluation that made acting on it urgent:

> Alignment finds a competent fixed point about as often on either wiring (16 against 11 of 48; R2
> cannot separate them). When it does, the wild-type's fixed points are better: its best eight
> seeds sit at 56–86% and the null's at 33–56%. The mean delta is carried by that upper tail, and a
> paired rank test, seeing a near-even sign split, reports no shift. **The wiring's mark is the
> *level* of the good fixed points, not their *frequency*.**

and, in the same logbook, the test that follows from it:

> A contrast on the upper mode (the mean among competent seeds, or an upper-quantile difference) is
> the natural test for what panels 2 and 3 show.

Both components — frequency and level — and the threshold that separates them
(`COMPETENT_THRESHOLD = 20.0`, committed in `scripts/analysis/l4_panel2.py`) predate 052. The competent-fraction test was already *registered*, as panel 3's secondary R2, and already run. Nothing
in the family below was selected by looking at 052, and the threshold is not re-tuned here: a
threshold chosen now, with 052's per-seed values known, would be exactly the move this change
exists to prevent.

## The family

A mixture has two parameters a panel can move, and a test that reads only one of them will call the
other no effect. So the registered family has one member for each, plus the existing test:

| | member | statistic | reads |
|---|---|---|---|
| **F** | frequency | paired competent-fraction discordance at 20.0 — exact binomial on the discordant pairs, panel 3's registered R2 generalised | how *often* an arm lands competent |
| **L** | level | difference in the mean level among competent seeds, each arm over its **own** competent subset; p from an exactly enumerated pooled-label permutation, interval from a bootstrap of the arms as observed | how *good* it is when it does |
| **W** | all-seeds | the existing paired one-sided Wilcoxon + 80% bootstrap CI | the shift, if the outcome is unimodal after all |

Every member is one-sided in the arm-improves direction at α = 0.05, corrected together by BH-FDR
across {F, L, W} — the sidedness and level 042's family used (`SIG_Q = 0.05`). F generalises the
registered `l4_panel3.discordance`, an exact binomial with `alternative="greater"`, rather than
re-deriving it. **W is retained deliberately**: it is what every committed table was scored with,
and dropping it would make the re-read incomparable with the record it re-reads.

**L is per arm, not paired, and that is the point.** 042's statement is about the location of each
arm's upper mode — "the wild-type's fixed points are better" — so L compares the mean of arm A's
competent seeds with the mean of arm B's competent seeds, each over its own subset. A paired
version over the union of competent seeds was considered and rejected: a pair in which one arm is
competent and the other dead would contribute the whole competent value to L, which is a
*frequency* event (the other arm found no fixed point) wearing level's name, and it would make L
significant on a panel where only frequency moved. Giving up the seed pairing is inherent to the
question and is stated here. A seed competent in one arm only contributes to that arm's level. With
no competent seed in either arm, L is undefined and reported as such, not as a null.

n = 8 or 16 per panel gives a mixture model no usable power; F and L are two marginal readings of a
mixture, not an attempt to fit one. That limit is stated in the record rather than discovered.

## The metric

`plateau_tail` has always returned mean foods (0–10) beside the full-clear percentage, and every
committed per-seed CSV carries it. The cliff metric — ten of ten or nothing — cannot distinguish a
policy that reaches eight foods from one that reaches zero, and on a foraging task that is most of
the behaviour.

So the graded metric is read as a **parallel family**, BH-FDR within itself. Competence is defined
**once**, on the primary metric at the committed 20.0 — a foods threshold chosen now, with 052's
per-seed values known, would be the post-hoc move this change forbids. F is therefore identical
across the two metrics and is not duplicated; the graded family is {L, W} on foods, L over the
same competent subsets the primary family defines. The full-clear metric stays **primary**, because every committed verdict is in its units
and this change does not restate them. The outcome map names what a disagreement between the two
families means, so "graded improves, cliff does not" is a registered reading rather than a
consolation.

## The outcome map

Each of F and L is **+** (significant, arm improves), **−** (significant, arm degrades) or **0**
(not significant), at the corrected level. Every cell is named; the table is the verdict.

| F | L | verdict | licenses |
|---|---|---|---|
| + | + | `shift` | the effect is unambiguous |
| − | − | `degrades` | — |
| + | − | `mixed_response` | **nothing** |
| − | + | `mixed_response` | **nothing** |
| 0 | + | `level_only` | 042's finding: better fixed points, not more of them |
| + | 0 | `frequency_only` | more competent seeds, no better |
| 0 | − | `degrades` | — |
| − | 0 | `degrades` | — |
| 0 | 0 | `mixed_response` if the arm improves ≥ 1 seed above the comparator **and** degrades ≥ 1 below it by more than the hold band; otherwise `no_effect` | nothing / the contrast is closed on this evidence |

L undefined is read as 0 for the table and the record says so.

`mixed_response` is the branch this change exists to create, and it is deliberately **sterile**: it
licenses no follow-on, gates nothing open, and requires its own registration to act on. A named
outcome that licenses nothing is an honest description; an unnamed one discovered after the fact is
a story. It fires on an explicit two-directional result — F and L significant against each other,
or an explicit improve-and-degrade split with neither significant — and not wherever significance
is merely missed. "Fewer seeds competent, but the competent ones better" is the strongest form of
it, not a degradation, and is the cell 052 most resembles.

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
- **Report the family from `l4_panel.py`.** Rejected for this change: `l4_panel2` imports
  `l4_panel`, and the new module takes its threshold from `l4_panel2`, so an import from
  `l4_panel` would be a cycle. The re-read is a standalone script over committed tables and
  `l4_panel.py` is not edited; a future panel registers with the family by calling the module.
- **Do this after the low-σ programme**, which 052's verdict licenses. Rejected: that programme
  ends in an assay scored by the statistic under repair, so it would spend campaign time to produce
  a number this change would then have to re-read.
