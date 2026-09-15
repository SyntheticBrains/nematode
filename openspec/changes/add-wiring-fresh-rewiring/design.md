# Design: a replication, and why almost nothing new gets written

## What the caveat actually is

`rewire_seed` is unset in every wiring config, so each seed's rewired graph derives from its run seed.
That was a deliberate choice — it makes a seed's wild-type and rewired arms pair — and it has a
consequence 058 recorded: **the rewiring and the initialisation vary together, and every wiring result
so far has drawn its graphs from seeds 1–64.**

The sharper form, which recon established and which 058 states less directly: **V.3's rewirings at
seeds 1–32 are a subset of V.1's at 1–64.** They are the same graphs. So the project's two positive
wiring results do not corroborate each other on independent nulls — they share them. That is the gap
this closes, and after L.0 it is the last open question on the only surviving wiring effect.

Seeds 65–96 give graphs never used by any panel. What that **does not** do is separate rewiring from
initialisation: both move together, as they always have. Isolating the rewiring alone would need
`rewire_seed` pinned while the run seed varies, which is a different experiment and not the one 058
registered. Stating that distinction is part of the deliverable.

## Why no new harness

`wiring_premise.py` carries the registered test family — `thermal` as V5–V8, `hard_food` as V9–V12,
each with `wt_ppo`, `rn_ppo`, `wt_frozen`, `rn_frozen` — and reads a `<cell> <arm> <seed> <out>`
manifest. **It also imports and drives `connectome_structure_efficiency` itself**, so one entry point
covers the efficiency primary and the peak-axis gates that V.1's amendment split apart. And it already
owns everything a V.4 driver would otherwise re-implement:

| the harness already has | where |
|---|---|
| the registered ≥ 20% minimum | `MIN_EFFICIENCY_GAIN` |
| per-cell verdicts, gates read first | `verdict()` — `insufficient_seeds`, `no_learning`, `saturated`, `specific_wiring`, `below_min_effect`, `degree_statistics` |
| the censoring guard | `CROSSING_FLOOR = 0.8`, flagging a contrast as materially censored |
| the efficiency arms' mapping | `EFFICIENCY_ARMS` |

**Both score this campaign as they are**, and the only new code is a manifest builder plus a reporter
that maps V.1's prose branches onto the harness's verdict names. Recon shortened this change: an
earlier draft specified a driver that would have re-implemented the minimum, the branches and the
comparators, which is the duplication the next section forbids.

That is not a convenience, it is the design. A replication exists to vary the evidence and hold the
reading fixed. Writing a sibling module — the right call for L.0, whose arms the committed family did
not contain — would here reintroduce exactly the confound a replication removes: if the panel failed,
"the instrument changed" would compete with "the effect is not there", and the two are not separable
after the fact. So the only new code is a **thin driver**: build the manifest, call both committed
analyses, report each cell against the registered branches.

The corollary is worth stating too. If the committed harness turns out not to be able to score this,
that is a **limitation of the replication**, recorded as such — not a licence to write a new one.

## The branches, inherited rather than invented

V.1 registered its own replication branches before its panel 2 ran, and they are taken verbatim:

| harness verdict | V.1's prose branch | what follows |
|---|---|---|
| `specific_wiring` | **replicates** | the caveat closes; the positive is independent in rewiring |
| `below_min_effect` | **same direction, below the minimum** | a real but smaller effect, the shrinkage named. **Licenses the follow-up, not the claim** |
| `degree_statistics` | **does not replicate** | **the panel is reported as not holding and the first positive is withdrawn on the record rather than defended** |

The prose is kept because it is what V.1 registered and what a reader of 057 will look for; the
harness's names are what the record reports, because a parallel vocabulary is how two records come to
disagree about the same run.

**Three of the harness's verdicts have no prose branch, and they are registered here because they are
live rather than for completeness.** `saturated` is what the klinotaxis cell returned in V.1's pilot —
both wirings at 100%, a contrast of exactly zero — and it means the cell cannot answer on this axis,
which is **not** a replication failure. `no_learning` means a gate failed and the contrast is
uninterpretable. `insufficient_seeds` means too few paired seeds survived to score. And separately from
the verdict, the harness flags a contrast as **materially censored** when fewer than 80% of seeds in
either arm cross the threshold — the case L.0 has just met on `hard350`, where five wild-type seeds of
32 never became competent. A censored, saturated or gate-failed fresh panel is **not** evidence against
the original result, and fixing that in advance is what stops it being read as one.

Inheriting them matters for a reason beyond tidiness: branches written *after* seeing a replication's
result are not a registration. V.1 wrote these when it had no idea what a fresh panel would say, and
this campaign is the case they were written for.

**The disagreement case is fixed here because two panels and one caveat make it live.** Each cell is
read on its own against the three branches. A **split** — one cell replicating, the other not — is
reported as a split with the pooled reading withheld. Block V's claim is that the effect generalises
from "a foraging cell under thermal pressure" to "a foraging cell hard enough to discriminate"; a split
is therefore evidence about the **scope** of that generalisation, and reporting it as one is the honest
reading. What a split must not do is get resolved toward whichever cell supports the original.

## What the campaign inherits, and where its power sits

| panel | cell | committed result | prior seeds | fresh seeds |
|---|---|---|---|---|
| V.1 | thermal, `t20` | **+35.4%** off time-to-competence, pooled over 64 | 1–64 | 65–96 |
| V.3 | hard food-only, `max_steps: 350` | **+23.5%**, 32 paired seeds, three of four metrics significant | 1–32 | 65–96 |

Thirty-two paired seeds per cell, which is V.3's own count and half of V.1's. The power arithmetic is
the one L.0 registered and is repeated here because a non-replication carries a consequence for
committed records: at 32 pairs a one-sided sign test needs **22/32**, and against V.3's observed
per-seed win rate of 21–26 of 32 that is **43.4–97.3%** power, **79.2%** at the range's midpoint of
73.4%. These are **sign-test planning figures, not the registered procedure's power** — that is a
paired rank test under BH-FDR, which differs in both directions.

**V.1's panel is the one to watch on power.** Its pooled +35.4% came from 64 seeds and its individual
panels were weaker and not always significant (+46.4%, +32.6%, +31.8% across panels of 16, 16 and 32).
A 32-seed fresh panel is therefore better powered than V.1's own first panel but worse than its pooled
figure, and a near-miss on that cell should be read against that rather than as a clean failure.

## Honest prior

**Replication on both cells, with the thermal panel the likelier to come in under the bar.** For it:
V.2 found no graph property predicting learning time across all 64 rewirings, so there is no known
mechanism by which a particular set of graphs would carry the effect; and V.3 already reproduced V.1's
direction on a different cell with a different difficulty source. Against it: V.1's per-panel figures
varied by 15 points, the effect is on a censored time-to-threshold metric, and L.0 has just shown that
this substrate's wiring is inert in three other regimes — which lowers the prior that any wiring effect
is robust, including this one.

What would make me doubt a replication most is not the statistics but the pattern: three negatives and
one positive, where the positive's controls were never independent.
