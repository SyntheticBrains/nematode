# I.2 — a statistic and a metric matched to the outcome's shape

**Built 2026-09-11. No campaign was run. No committed verdict is changed by anything here.**

## The family

Logbook 042 stated the shape on 2026-09-07 — "the wiring's mark is the *level* of the good fixed
points, not their *frequency*" — named the contrast that follows from it, and carried the
requirement forward: *on a bimodal outcome, register a statistic matched to the shape before the
data exist.* The family has one member per component, plus the test the record was scored with:

| | member | statistic |
|---|---|---|
| **F** | frequency | competent-fraction discordance at the committed 20.0, exact binomial on the discordant pairs |
| **L** | level | difference in mean level among competent seeds, each arm over its **own** subset, seeded bootstrap |
| **W** | shift | the existing paired one-sided Wilcoxon + 80% bootstrap CI |

Both directions, each its own BH-FDR family at α = 0.05, so a member reads `+`, `−` or `0` and the
outcome map names every cell.

## Three things the build changed, each for a structural reason

**L is per arm, not paired.** Pairing over the union of competent seeds would let a pair with one
arm competent and the other dead contribute the whole competent value to L — a *frequency* event
wearing level's name — which would make L fire on a purely frequency-only panel and render the
map's central distinction unreachable. Tested directly.

**L's p-value comes from a permutation null.** Resampling each arm as observed is not a null: those
draws are centred on the observed difference, so the share of them below zero asks *where the
effect is*, not how often chance produces one this large. The null is built by pooling the two
arms' competent seeds and re-splitting at the observed sizes; the interval still comes from the
bootstrap of the arms as observed, which is what an interval is for.

The null is **enumerated exactly**, not sampled. Every committed panel is small enough — the
largest pool splits 12,870 ways — and exactness matters for more than precision: a sampled null
makes the p-value depend on the order values happen to arrive in, since a seeded generator applies
the same index permutation to whatever array it is handed, and seed-to-value assignment is
arbitrary. Enumerating makes the result a function of the two multisets and nothing else.

This correction changed the result. An earlier build used the uncentred draws as a p-value and
reported two contrasts at `level_only`; under the exact null they are **p = 0.1333 and 0.0536**,
and the rank test agrees closely (0.133, 0.071). **No contrast reaches `level_only`.** The finding
below is what survives a correct test.

**The registered split criterion was demoted to descriptive.** It was to decide the
doubly-non-significant cell: "improves ≥ 1 seed and degrades ≥ 1 beyond the band". Simulated
against a null where **both arms are drawn from the same bimodal law**, it fires on **81–100%** of
draws at every band from 5 to 25, and requiring three each way still fires on 77% at sixteen seeds.
The quantity it counts is what bimodality produces by itself, so no threshold on it is specific.
A doubly-non-significant panel is now `no_effect` whatever its spread, and `mixed_response` is
reserved for the two contrasts significant *against each other*. The counts are still recorded,
because the shape is worth seeing.

## What the members can detect

- **F** needs **six** discordant pairs all one way to reach q ≤ 0.05 (p = 0.0156); five reach only
  0.094. Six fits inside eight seeds — an arm competent on six where the other is competent on none
  — so `frequency_only` is reachable there, but only for a lopsided split. The registered test's
  strictness is left alone.
- **L** has a combinatorial floor shared by every distribution-free test here: at three competent
  seeds a side the pool has only twenty splits, so the smallest reachable q across three members is
  about **0.19** and `level_only` cannot fire. Four a side reaches 0.054, five a side 0.027. A
  limit of these panel sizes, not of the test.
- Neither is a mixture model. Two marginals are what eight or sixteen seeds support.

## The re-read

Seventeen contrasts across nine committed tables, each from its committed per-seed CSV. The
all-seeds member reproduces panel 3's committed **R1** and the frequency member its committed
**R2**, exactly — so the re-read is reading the numbers the record was scored on.

**Fourteen of seventeen contrasts read `no_effect`; three read `degrades`** (045's anchor, and
050's node-perturbation arm and its frozen control, all on F and W). No contrast reaches
`level_only`, `frequency_only`, `shift` or `mixed_response`.

| table | contrast | F/L/W | L | W | re-read |
|---|---|---|---|---|---|
| 040 | wiring_plastic | 000 | −4.8 | +6.6 | `no_effect` |
| 040 | wiring_hebbian | 000 | +25.0 | +16.5 | `no_effect` |
| 041 | wiring_hebbian (P1) | 000 | +14.8 | +14.1 | `no_effect` |
| 041 | wiring_hebbian_count (P2) | 000 | +22.7 | +2.0 | `no_effect` |
| 042 | wiring_hebbian (R1) | 000 | +12.9 | +8.1 | `no_effect` |
| 044 | atlas / dale | 000 | +10.7 / +8.8 | +4.9 / −2.5 | `no_effect` |
| 045 | anchor | −0− | −7.0 | −25.7 | `degrades` |
| 045 | rigidity / oracle | 000 | −1.5 / +11.1 | −9.4 / −9.8 | `no_effect` |
| 046 | antihebb / oja | 000 | +14.5 / +12.5 | +2.6 / +2.0 | `no_effect` |
| 047 | routing_wt | 000 | −17.9 | −3.2 | `no_effect` |
| 047 | routing_rn | 000 | −8.3 | −7.3 | `no_effect` |
| 050 | node_perturbation | −0− | −11.9 | −26.7 | `degrades` |
| 050 | perturbation_frozen | −0− | +3.9 | −29.8 | `degrades` |
| 052 | endpoint | 000 | +7.2 | −18.1 | `no_effect` |

## What it shows, and what it does not

**Nothing is promoted by the new statistic.** The descriptive level differences are large and
consistently positive on the Hebbian wiring comparison — panel 1 at **L = +25.0**, panel 2's
count-initialised arm at **+22.7**, panel 3 at **+12.9** — and not one of them is significant
against a pooled-label null. The reason is the floor above: these contrasts have two to five
competent seeds an arm, where no distribution-free test can resolve a difference at α = 0.05
after correction.

**So the wiring hypothesis stays closed, and the re-read adds no support for reopening it.** This
is the result a correct test gives; an earlier build of this change reported two of these as
`level_only` on a p-value that was not measuring a null, and that report was wrong.

**What the family did establish is the negative claim it was built to test honestly.** The
committed record's reading — that these contrasts show no confirmed effect — survives a statistic
matched to the outcome's shape. It was not an artefact of using the wrong test.

**052 re-reads as `no_effect`**, against its registered `fail`. Six seeds down and two up, with
L = +7.2 and W = −18.1, and neither contrast significant at n = 8. The honest reading is that this
panel establishes nothing either way — which is itself the case for the statistic, since the
registered assay rule returned a definite `fail` on evidence that cannot support one.

**The graded metric is available for panels only.** Eleven contrasts carry `foods`; the three
assay tables do not. Its level contrast is defined on only **five** of those eleven, since it needs
a seed competent on the primary metric in both arms. Where both are defined the two readings agree
in direction, with one exception: **047's `routing_wt`**, where the primary shift is **−3.2** and
the graded shift **+0.04** — an arm slightly worse on full clears and indistinguishable on food
collected. Neither is significant.

## Limits

- n = 8 or 16. Every "not significant" here is compatible with a real effect this size of panel
  cannot resolve, and `frequency_only` is effectively unreachable below sixteen seeds.
- The re-read is descriptive. It is an input to the ladder re-read (I.4) and to nothing else.
- The re-read corrects within each contrast's own family, as a panel does; it is not a
  seventeen-way test. No contrast reached a positive verdict, so nothing here would survive a
  further family-wise correction either.
- The p-value correction that removed the `level_only` results was found in review, after they had
  been reported. They are recorded here as what a wrong test produced, not omitted.
