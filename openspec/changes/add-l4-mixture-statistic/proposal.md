# A statistic and a metric matched to the outcome's shape (7a-ii I.2)

## Why

Every panel in this phase has applied a paired rank test to an outcome that is not unimodal.
[Logbook 042](../../../docs/experiments/logbooks/042-l4-panel3.md) named the lesson on 2026-09-07
and stated the diagnosis precisely: alignment "finds a competent fixed point or a dead one", the
two wirings reach one about equally often, and **"the wiring's mark is the *level* of the good
fixed points, not their *frequency*"**. It then named the natural test — "a contrast on the upper
mode (the mean among competent seeds, or an upper-quantile difference)" — declined to adopt it
after three looks at that data, and carried the requirement forward verbatim: **"on a bimodal
outcome, register a statistic matched to the shape before the data exist."** It was never acted on.

Four panels later, the cost is no longer hypothetical. The endpoint evaluation
([052](../../../docs/experiments/logbooks/supporting/052-l4-endpoint-evaluation/details.md)) scored
a mean of 20.6 against a comparator of 38.7 — 18 points down — and the registered paired test
returned **p = 0.074**, because six seeds degraded and two improved, one of them to the best
full-clear rate this substrate has produced. Mean 20.6, median 9.2. The registered statistic could
not distinguish "degrades policies" from "degrades six, transforms two", and the registered metric
— a full clear or nothing — could not see any progress short of the cliff.

The machinery is largely already here and already registered. `COMPETENT_THRESHOLD = 20.0` and a
competent-fraction discordance test were registered as panel 2's **secondary** and used in 042 and
044\. The graded metric exists in every committed record: `plateau_tail` has always returned mean
foods beside the full-clear rate, and every committed per-seed CSV carries it. What is missing is a
registered *primary* family matched to the shape, the upper-mode contrast 042 named but never
registered, an outcome map with names for bimodal results, and the re-read.

## What Changes

- **A registered contrast family with two components, because the shape has two.** *Frequency*:
  paired competent-fraction discordance at the committed 20.0 threshold, an exact test on the
  discordant pairs — promoted from panel 2's secondary to a co-primary. *Level*: the mean among
  seeds competent in either arm, paired — the contrast 042 named and did not register. The existing
  all-seeds paired Wilcoxon is retained as the third member so every committed table stays
  comparable. BH-FDR across the family.
- **The graded metric beside the cliff.** Plateau-tail mean foods (0–10), already computed and
  already committed, read as a registered parallel family rather than a footnote, so learning short
  of a full clear is visible.
- **An outcome map that can name a bimodal result**, including one for "improves some seeds while
  degrading others". That outcome **licenses nothing on its own** and requires its own
  registration; naming it is what stops it being discovered after the fact.
- **A descriptive re-read of every committed table** (040, 041, 042, 044, 045, 046, 047, 050, 052)
  under both readings. **It cannot overturn a registered verdict**: each stands as registered, and
  the re-read is a second, pre-specified reading beside it, and the input to I.4.
- Records under `supporting/053-l4-mixture-statistic/`; tests; docs.

Out of scope: any new campaign — this change runs nothing. Any change to a committed verdict. Any
panel, which stays gated.

## Capabilities

**Modified**: `plasticity-evaluation` (the statistic and metric a panel is read with).

## Impact

- New: a statistics module and its tests, the supporting directory. Edited: `l4_panel.py` and the
  panel-family scripts to report the new family beside the existing one, `CHANGELOG.md`.
- No package code, no substrate, no rule changes. Every committed number stays as committed.
