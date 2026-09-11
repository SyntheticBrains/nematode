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

**L's test is a bootstrap, not Mann-Whitney.** With three competent seeds per arm — an ordinary
outcome on eight seeds — the smallest one-sided rank p is 1/20 = 0.05, so after correction across
three members the smallest reachable q is **0.15** and `level_only` could never fire. That is
combinatorics, not a result.

**The registered split criterion was demoted to descriptive.** It was to decide the
doubly-non-significant cell: "improves ≥ 1 seed and degrades ≥ 1 beyond the band". Simulated
against a null where **both arms are drawn from the same bimodal law**, it fires on **81–100%** of
draws at every band from 5 to 25, and requiring three each way still fires on 77% at sixteen seeds.
The quantity it counts is what bimodality produces by itself, so no threshold on it is specific.
A doubly-non-significant panel is now `no_effect` whatever its spread, and `mixed_response` is
reserved for the two contrasts significant *against each other*. The counts are still recorded,
because the shape is worth seeing.

## What the members can detect, recorded before the results are read

- **F** needs **six** discordant pairs all one way to reach q ≤ 0.05 (p = 0.0156); five reach only
  0.094. An eight-seed panel can show a real frequency difference F cannot call, so
  `frequency_only` is effectively a sixteen-seed verdict. This is the registered test and its
  strictness is left alone.
- **L** needs a competent seed in both arms, and its precision is set by how many.
- Neither is a mixture model. Two marginals are what eight or sixteen seeds support.

## The re-read

Seventeen contrasts across nine committed tables, each from its committed per-seed CSV. The
all-seeds member reproduces panel 3's committed **R1** and the frequency member its committed
**R2**, exactly — so the re-read is reading the numbers the record was scored on.

| table | contrast | F/L/W | L | W | re-read |
|---|---|---|---|---|---|
| 040 | wiring_plastic | 000 | −4.8 | +6.6 | `no_effect` |
| **040** | **wiring_hebbian** | **0+0** | **+25.0** | +16.5 | **`level_only`** |
| 041 | wiring_hebbian (P1) | 000 | +14.8 | +14.1 | `no_effect` |
| **041** | **wiring_hebbian_count (P2)** | **0+0** | **+22.7** | +2.0 | **`level_only`** |
| 042 | wiring_hebbian (R1) | 000 | +12.9 | +8.1 | `no_effect` |
| 044 | atlas / dale | 000 | +10.7 / +8.8 | +4.9 / −2.5 | `no_effect` |
| 045 | anchor | −0− | −7.0 | −25.7 | `degrades` |
| 045 | rigidity / oracle | 000 | −1.5 / +11.1 | −9.4 / −9.8 | `no_effect` |
| 046 | antihebb / oja | 000 | +14.5 / +12.5 | +2.6 / +2.0 | `no_effect` |
| 047 | routing_wt | 0−0 | −17.9 | −3.2 | `degrades` |
| 047 | routing_rn | 000 | −8.3 | −7.3 | `no_effect` |
| 050 | node_perturbation | −−− | −11.9 | −26.7 | `degrades` |
| 050 | perturbation_frozen | −0− | +3.9 | −29.8 | `degrades` |
| 052 | endpoint | 000 | +7.2 | −18.1 | `no_effect` |

## What it shows, and what it does not

**Two contrasts move from nothing to `level_only`**, both on the Hebbian wiring comparison: panel 1
at **L = +25.0** and panel 2's count-initialised arm at **+22.7**. That is precisely 042's
hypothesis — an advantage in the level of the good fixed points, invisible to a test of the shift —
now detected by a statistic matched to the shape.

**It does not resurrect the wiring hypothesis, and must not be read as doing so.** Panel 3 was the
registered replication on fresh seeds, and it re-reads as **`no_effect`**: L = +12.9, not
significant. An effect present in the first two panels and absent from the replication designed to
confirm it is exactly what 042 closed the hypothesis on. The committed verdicts stand as
registered; the re-read says the earlier panels' descriptive signal had a shape the old statistic
could not name, not that the contrast was real.

**052 re-reads as `no_effect`**, against its registered `fail`. Six seeds down and two up, with
L = +7.2 and W = −18.1, and neither contrast significant at n = 8. The honest reading is that this
panel establishes nothing either way — which is itself the case for the statistic, since the
registered assay rule returned a definite `fail` on evidence that cannot support one.

**The graded metric is available for panels only.** The six panel tables carry `foods`; the three
assay tables do not. Stated rather than quietly skipped. Where it is available it agrees with the
primary reading in direction on every contrast.

## Limits

- n = 8 or 16. Every "not significant" here is compatible with a real effect this size of panel
  cannot resolve, and `frequency_only` is effectively unreachable below sixteen seeds.
- The re-read is descriptive. It is an input to the ladder re-read (I.4) and to nothing else.
- Two contrasts reaching `level_only` out of seventeen, uncorrected across the re-read as a whole,
  is what a family-wise correction over seventeen contrasts would likely remove. The re-read
  corrects within each contrast's own family, as a panel does; it is not a seventeen-way test.
