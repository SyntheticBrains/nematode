# 095: The Gap-Junction Placement Advantage Is an Artefact of How the Model Scales Gap Junctions (Phase 9 G.0)

**Status**: completed. The registered verdict is **`artefact`**: once the electron-microscopy counts
are removed from the gap matrix, the wild type no longer leads the null whose gap junctions alone are
rewired.

On the thermal cell at target 35, on Emmons 2024, the wild type leads the gap-only null as Phase 8
found, on the panel's own 64 seeds. With every existing gap pair weighted 1 before the normalisation,
the lead is gone.

| registered reading, `wild type − null`, `auc_success` | seeds | mean | 80% CI | BH q | state |
|---|---|---|---|---|---|
| `gap(count)`, today's gap weights | 2401–2464 | **+0.104** | [+0.087, +0.123] | < 0.001 | `move_wt`: the effect is there |
| `gap(unit)`, every count set to 1 | 2401–2464 | **+0.003** | [−0.012, +0.018] | 0.82 | `no_move` → **fails** |
| `gap(outliers_scattered)`, only the 26 pairs of count ≥ 20 moved | 2401–2432 | +0.033 | [+0.005, +0.061] | 0.076 | `unresolved` → open |
| `gap(outliers_held)`, those 26 held, the rest rewired | 2401–2432 | +0.063 | [+0.041, +0.084] | 0.005 | `below` (described) |

The minimum is **0.0894**, two-thirds of the anchor's +0.1341: D23 re-fixes the minimum from the
anchor, and two-thirds is the launch record's rule for doing so. `gap(unit)` fails, so the
verdict is `artefact` whatever the outliers do (the registered map names every combination).

**Beside it, never read as verdicts:**

- **The lead follows the gap scale.** With the placement fixed and the whole gap matrix scaled, the
  lead is +0.004 at ×0.25, +0.062 at ×0.5, +0.104 at ×1 and +0.153 at ×2 (16 seeds; the ×1 value is
  the 64-seed reading). With each count replaced by `log1p(count)` it is +0.012 [−0.023, +0.046].
- **The null's count-weighted junctions cost it; no benefit to the wild type is detected.** With gap
  junctions off, the brain trails the wild type by +0.019 [−0.005, +0.044], a non-detection on 32
  seeds rather than an absence, and leads the count gap-only null by +0.090 [+0.064, +0.116]. The wild
  type's plateau is 69–71% with count, unit or log weights and 67% with no gap junctions at all; the
  null's is 52.7% with counts and 68.6% without them.
- **Not only the largest pairs.** Holding the 26 outliers in place while the other 1,069 pairs are
  rewired with their counts still leaves a +0.063 lead; moving only the outliers costs +0.033.
- **Depth.** At depth 2 the wild type barely learns (plateau 0.1%, above its floor on 7 of 16 seeds,
  which passes the gate) and the null leads, −0.034 [−0.050, −0.021]; at depth 3 the lead is +0.041,
  at depth 6 +0.080 with both plateaus lower (38% and 22%).
- **Episodes to 30% success**, registered beside: `count` +405 [+332, +484], `unit` +6 [−50, +65],
  `outliers_scattered` +94 [−12, +212], `outliers_held` +220 [+133, +303]. The censoring rule found
  the levels' crossing rates too different to compare (spread 1.0), so `auc_success` carries the reading,
  as registered.
- **The shortcut audit.** Every gap-only null, under every weighting, puts about four motor neurons one
  gap-hop from an injected sensor where the wild type has none. Under unit weights the null keeps that
  shortcut and does not trail, so a shortcut at unit coupling (at most 0.58) costs no lead; at count
  coupling this panel does not separate it (held, with the shortcut, trails by +0.063; scattered,
  without it, by +0.033).
- **The saturation predictor**, registered before it was computed, is `unresolved` on 082's Cook
  draws (ρ +0.28, p 0.026, under the 0.3 minimum) and on the anchor's 16 Emmons draws (ρ −0.31,
  p 0.24); on the panel's 64 draws, as description, `no_prediction` (ρ +0.14).

**Date**: 2026-10-17.

**OpenSpec change**: `add-gap-operating-point`.

**Pre-registration**: [supporting/095-gap-operating-point/launch.md](supporting/095-gap-operating-point/launch.md),
committed at cc8407c3 before any panel run. It followed the Emmons anchor pilot, a two-seed smoke of
every new level, a 64-run gate pilot for the deciding levels, the gate preflight and two spec reviews
(the second found and fixed a bias in the scattered null).

## The question

Phase 8's citable result is that gap-junction placement makes the wild-type connectome learn faster
than a null whose gap junctions alone are rewired: +0.096 `auc_success` on the point worm's thermal cell
([Logbook 082](082-plastic-gaps.md)) and +0.126 through the kinematic body
([Logbook 092](092-thermal-body.md)). The model couples gap junctions as `G_ij = count_ij / sqrt(d_i d_j)`,
`d` each neuron's number of gap partners, added to the settling map's tanh drive. That normaliser counts
partners, not counts, so a few high-count pairs between low-degree neurons couple far above 1: ALA to
CAN at 89.7, 98 pairs above 1 on Emmons. The rewiring moves those pairs, and the gap scale had never
been swept. **Is the lead a property of the wiring, or of how this model scales gap junctions?**

## Method

The thermal cell at target 35, on Emmons 2024 (D23), PPO at block V's point otherwise: edge-order draw,
pooled readout, depth 4 unless stated, settling dynamics, 3,000 episodes. Each level pairs a wild type
with a null, each learning and frozen; the deciding levels carry frozen floors on every seed. 896 runs,
every one exiting 0, in about 15 hours at 16 workers.

| level | null | gap seeds | role |
|---|---|---|---|
| `count` | gap-only | 64 | the reference |
| `unit` | gap-only, both wirings at unit weights | 64 | **deciding** |
| `outliers_scattered` | the 26 outliers each swapped once, the count landing on either endpoint | 32 | **deciding** |
| `outliers_held` | the 26 outliers held, the rest rewired | 32 | beside |
| gap-off | read against both `count` arms | 32 | beside |
| scale ×0.25, ×0.5, ×2; `log_count` | gap-only | 16 | beside |
| depth 2, 3, 6 | gap-only | 16 | beside |

Readings are paired `wild type − null` gaps on `auc_success`, the four in the table corrected together
(BH-FDR) and classified at the minimum. Every level passed its floor gate, and none saturated.

**The anchor** (seeds 9801–9816, 64 runs) re-established the effect on Emmons before the panel, in its
own pilot as D23 requires: +0.1341 [0.101, 0.168], both wirings readable. It sat above the
[0.058, 0.13] range in which Logbook 082's minimum would have been carried, so the minimum was re-fixed
at two-thirds of it.

## Results

### The registered reading

`gap(count)` reproduces on the panel's own seeds, +0.104 [+0.087, +0.123], so there is an effect to
screen. Under unit weights the two wirings learn alike: the wild type plateaus at 69.8% and the null at
68.6%, and the paired gap is +0.003 [−0.012, +0.018], q 0.82, inside ±*m* on both sides: `no_move`.
The registered map reads **`artefact`**.

The scattered outliers alone cost +0.033 [+0.005, +0.061], an interval that excludes zero but sits
well under the minimum, with q 0.076: `unresolved`. It does not decide the verdict, which `gap(unit)`
already fixed.

### What carries the lead

The beside levels agree with the verdict and say where the lead comes from.

| level | wild type plateau | null plateau | gap, `auc_success` | 80% CI |
|---|---|---|---|---|
| ×0.25 | 65.3% | 64.3% | +0.004 | [−0.022, +0.028] |
| ×0.5 | 69.6% | 59.8% | +0.062 | [+0.023, +0.103] |
| ×1 (`count`) | 69.5% | 52.7% | +0.104 | [+0.087, +0.123] |
| ×2 | 57.3% | 29.1% | +0.153 | [+0.098, +0.211] |
| `log_count` | 70.8% | 67.6% | +0.012 | [−0.023, +0.046] |
| `unit` | 69.8% | 68.6% | +0.003 | [−0.012, +0.018] |

The wild type learns about as well with counts, unit weights, log weights or no gap junctions at all
(67–71%). The null learns as well as the wild type whenever the large counts are removed or compressed,
and worse the larger they are. This is **consistent with** the null being hurt by its rewired
count-weighted gap junctions, scaled by how strongly the model lets them couple, rather than the wild
type being helped by its own placement; gap-off trails the wild type by +0.019 (not detected) and beats
the count null by +0.090. It does not say which property of the rewired junctions does the harm: the
swap moves where the counts land and also redistributes each neuron's total coupling (the count–degree
correlation falls from 0.19 to about 0), and separating those two is G.1's question.

The ×1 row is the 64-seed reading; on the profile's own seeds, 2401–2416, `count` reads +0.122 (71.0%
against 49.4%).

The harm is spread through the counts, not confined to the largest pairs. Holding the 26 pairs of count
≥ 20 in place and rewiring the other 1,069 still leaves +0.063; moving only those 26 leaves +0.033.

### Depth, and the shortcut audit

At depth 2 the wild type barely learns, 0.1% against the null's 12.6% (it passes its floor gate on 7 of
16 seeds), and the null leads, −0.034 [−0.050, −0.021], as Logbook 071 found on hard350 at the shallow
end. At depth 3 the lead is +0.041 and at depth 6 +0.080, with both wirings'
plateaus lower at depth 6.

The audit computed before launch ([table.json](supporting/095-gap-operating-point/table.json)) found that
every gap-only null creates about four sensor–motor gap pairs, putting four motor neurons one gap-hop
from an injected sensor where the wild type has none. Under unit weights the null keeps that shortcut,
coupling at most 0.58, and does not trail; at count coupling this panel does not separate the shortcut
from the rest of the rewiring.

## What the verdict licenses, and what it does not

**Registered consequence (G.0-A).** `artefact` re-aims block G. The substrate change G.0 names — count-capped or log-compressed gap weights, or ohmic coupling in the settling map — lands as its own change with an MLP positive control; **092's body contrast is re-run on the corrected model before anything else**; the decomposition (Logbooks 074, 075, 078, 079, 082, 092) is restated; and G.1–G.3b run on the corrected model or are deferred with a destination.

**Which substrate change.** This panel tested one of the three candidates directly: log-compressed
counts, under which both wirings learn alike at about 70% and the lead is +0.012. Count-capped weights
and ohmic coupling were not run. **The maintainer named the change on 2026-10-17**: ohmic gap coupling in
the settling map (current in proportion to the potential difference, which cannot latch at any
conductance and keeps the counts as conductances), chosen against log-compressed counts in a gated pilot
whose selection rule is fixed before it runs. G.0-A's change carries both.

**What it does not do.** The Phase 8 records are not re-read (D23): their registered readings stand as
what was measured on the count-weighted model, on Cook 2019. On Emmons at thermal t35, placement without
the counts carries no lead. What that means for Logbooks 078, 082 and 092 is G.0-A's registered
restatement, starting with the body's +0.126 re-run on the corrected model.

**Limits.**

- One cell (thermal t35) and one substrate (Emmons 2024); Phase 8's records are on Cook 2019.
- Unit weights remove the counts **and** cut the coupling (the largest from 89.7 to 1, the 99th
  percentile from 5.13 to 0.41). "Counts removed" and "coupling reduced" are separated only by the
  16-seed scale profile and the log-count level, which point the same way.
- `gap(unit)` compares against a null that keeps a sensor–motor shortcut, as the launch record states.
- The profile and depth levels read 16 seeds and borrow the count level's floors; the depth-2 wild type
  passes its gate at a 0.1% plateau.
- Achieved spreads: `count` σ 0.119 and `unit` σ 0.090 (detectable about 0.042 and 0.032 at 64 seeds),
  `outliers_scattered` σ 0.125 (about 0.062 at 32 seeds, above the registered 0.047–0.054 but below
  the minimum). Reported, never used to re-read.
- 3,000 episodes, below the horizon where the collapse statistic applies.

## Data

- [launch.md](supporting/095-gap-operating-point/launch.md): the registration.
- [anchor.json](supporting/095-gap-operating-point/anchor.json), [preflight.json](supporting/095-gap-operating-point/preflight.json): the anchor and the gate preflight.
- [panel.json](supporting/095-gap-operating-point/panel.json), [per-seed.csv](supporting/095-gap-operating-point/per-seed.csv): every level's gates, gaps and the reading.
- [table.json](supporting/095-gap-operating-point/table.json): the matched statistics and the shortcut audit.
- [predictor-cook.json](supporting/095-gap-operating-point/predictor-cook.json), [predictor-emmons.json](supporting/095-gap-operating-point/predictor-emmons.json), [predictor-emmons-panel.json](supporting/095-gap-operating-point/predictor-emmons-panel.json): the saturation predictor.

```bash
uv run python scripts/analysis/gap_operating_point.py score --logs campaigns/g0-panel/logs \
  --anchor docs/experiments/logbooks/supporting/095-gap-operating-point/anchor.json \
  --out-dir build/g0 --out panel.json --csv per-seed.csv
```

The campaigns' raw logs and session records are archived off-repo.
