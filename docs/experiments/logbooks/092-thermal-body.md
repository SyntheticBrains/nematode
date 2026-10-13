# 092: Through the Kinematic Body, the Wild Type's Gap-Junction Placement Advantage Survives (Phase 8b M.9)

**Status**: completed. The registered reading is **`move_wt` → the gap advantage survives
embodiment.**

On the thermal cell through the kinematic body, at target 20 over 6,000 episodes, the wild type
learns faster than a null whose gap junctions alone are rewired, by more than the registered minimum.

| registered reading, seeds 2301–2332 | mean `auc_success` | 80% CI | q | state |
|---|---|---|---|---|
| wild type − gap-only null | **+0.126** | [+0.067, +0.178] | 0.012 | `move_wt` → **the gap advantage survives embodiment** |

The interval's lower end clears the **0.058** minimum, two-thirds of the thermal split's committed
gap-junction effect. It is the first registered wiring advantage of this project to survive the
body: through the body on hard350, the chemical contrast was unresolved and bounded near 0.044
(Logbook 085).

**Beside it, never read as verdicts:**

- **Part of the lead is the null collapsing.** Six gap-only null learners learned and then lost it,
  against two wild-type learners (Fisher p = 0.26). On the 25 seeds where both learners'
  last-quarter plateau is at least 30%, which still include four collapsed learners, the lead is
  **+0.073** [+0.020, +0.122] (p = 0.15).
- **The two wirings crawl alike.** Undulation frequency, wavelength and speed match, trained and
  frozen, and untrained the two are identical, so the lead is not a gait difference. Trained, the
  null reverses more (4.3% of steps against 0.7%), a policy difference that fits its collapses.
- **Against the chemical-only null** the wild type leads by +0.086 [+0.019, +0.149], p = 0.14.
- **The body learns about five times more slowly than the point worm.** The doubled budget assumed
  twice. If a lead scales with learning time, the body's lead is about +0.048 on the point worm's
  3,000-episode scale, against the point worm's +0.096 at target 35, a comparison across cells.
- **The point-worm anchor at this cell saturates.** At target 20 both point-worm wirings plateau at
  97–99%, with no lead, so it cannot pair with the body reading.

**Date**: 2026-10-14.

**OpenSpec change**: `add-thermal-body-cell`.

**Pre-registration**: [supporting/092-thermal-body/launch.md](supporting/092-thermal-body/launch.md),
committed at 96a3e45d before any panel run. It followed a failed positive control, a retry
registered as a new decision, a connectome pilot, the gate preflight and two spec reviews.

## Objective

Phase 8b's body rung asks whether the wild type's wiring still matters once the worm has a body.
C.1e asked it on hard350 and left it unresolved: there, even the point worm's lead over the
chemical-only null was small (+0.022).

**The substrate's largest resolved wiring effect is on the thermal cell:** on the point worm, at
target 35, the wild type leads a null whose gap junctions alone are rewired by **+0.096
`auc_success`** (Logbook 082). The chemical wiring's lead there is about zero under every sign model
(Logbooks 078 and 091).

**M.9 asks the body rung's question where the point worm's answer is clear: does the wild type's
gap-junction placement advantage survive embodiment?**

**The gap-only null is primary.** A null must differ from the wild type in the layer under question,
so a gap-placement question reads the gap-only null (D21, scoped 2026-10-12, as M.8 did). The
chemical-only null is read beside it.

## Method

**The cell** is the thermal cell through the kinematic body:

- the connectome PPO brain on Cook 2019, settling to depth 4, `entropy_coef` 0.004;
- the thermotaxis block of Logbooks 078 and 082 (Tc 20, gradient 1.25, zone deltas and damage);
- 500 steps per episode, **target 20**, **6,000 episodes** per run;
- the kinematic body with reversal, driven through `body_drive`;
- temperature, its derivative and the zone are read at the body's head, and the left–right
  difference comes from the synthetic head-sweep (C.3 found weathervaning survives without it).

**Cook 2019, not C.1e's Emmons 2024.** The effect under test was measured on Cook, and Emmons differs
from Cook in exactly four gap-junction pairs, the layer under test. C.1e's body results are
therefore a loose comparison, not a pairing.

**The arms:** wiring {wild type, gap-only null, chemical-only null} × {PPO, frozen}. Seeds
2301–2332 (32), 192 runs, all of which completed 6,000 episodes, in 16.7 hours from a worktree pinned
at the registration commit.

**The reading:** `gap = wild type − gap-only null` on `auc_success` over all 6,000 episodes, paired
by seed, classified by `mc.classify` at 0.058 on the two-sided Wilcoxon q.

**Gates, per null:** each learner beats its frozen floor, and the two learners do not both reach the
90% bar.

### How the registration was arrived at

**Stage 1, the positive control, failed its fixed rule** ([stage1/](supporting/092-thermal-body/stage1/)).
MLP-PPO and its floor ran at targets 35 and 20 over 3,000 episodes (seeds 9701–9708). The rule:
target 35 if it passes, else 20, else M.9 stops. Both read `fallback`: each beats its floor, but not
every seed reaches 30%.

- **At 35:** the MLP collects 25–29 of 35 foods, with a plateau of 5% (at most 15% on a seed).
- **At 20:** six seeds plateau at 58–87%. Seeds 9704 and 9708 start learning only after about
  1,500–2,250 episodes, and are still climbing at 3,000.

**The retry, registered as a new decision, not a re-reading** (design Decision G;
[retry/](supporting/092-thermal-body/retry/)). With the user: target 20 only, 6,000 episodes at every
stage, new seeds 9721–9728, the same rule, and stop points before the pilot and the panel. It
**passed**: every seed competent, plateaus 57–95% (mean 81.6%) against 0% floors, +81.6 points over
the floor [77.2, 86.1].

**The connectome pilot** (seeds 9711–9714, configured as the panel;
[pilot/](supporting/092-thermal-body/pilot/)):

- every learner beat its 0% floor;
- the lead over the gap-only null was +0.147, on 4 of 4 seeds (sd 0.067);
- the C.3 instruments matched between the wild type and the null within about 1%.

**The gate preflight read `near_bar`** on both levels: the wild type's plateau, 86.1%, sat within the
5-point margin of the 90% bar. The user accepted it, on grounds recorded in the registration:

- the gate fails only if both learners reach the bar, and the null sat at 82.0%;
- the reading is `auc_success`, far from its ceiling;
- a ceiling compresses a lead toward `no_move`, so it biases against the hypothesis.

A level that saturated in the panel would have read unreadable.

**Sizing.** The pilot's spread reached the minimum at about 9 seeds, so the rule's floor was 16. With
the user, the panel ran **32**: the pilot's spread came from four seeds, and C.1e's spread through
the body was 2.6 times larger.

## Results

### Gates

Both levels are readable. Neither saturates; the near-bar concern did not materialise.

| level | wild type plateau | null plateau | floors | learners beating their floor |
|---|---|---|---|---|
| gap-only (primary) | 79.3% | 63.6% | 0.0% | 32 / 31 of 32 |
| chemical-only (beside) | 79.3% | 64.8% | 0.0% | 32 / 30 of 32 |

The connectome's wild type plateaus close to the MLP's 81.6% on the retry.

### The registered reading

**`move_wt`**: +0.126 [+0.067, +0.178], q = 0.012, positive on 21 of 32 seeds.

| | sd | MDE (`2.487 × sd / √n`) |
|---|---|---|
| pilot | 0.067 | — |
| panel | **0.244** | **0.107** |

The panel's spread was 3.6 times the pilot's and above C.1e's 0.175. The achieved MDE exceeds the
minimum. The reading is still `move_wt`, because its interval excludes zero and its mean clears the
minimum; the verdict is not re-read against the achieved MDE.

### Described beside

**Where the lead comes from** ([sensitivity.json](supporting/092-thermal-body/sensitivity.json)). The
per-seed spread is wide (−0.33 to +0.52), and the largest leads are seeds where the null collapsed.

**Collapse** is a learner that reaches a peak, then ends below half of it (blocks of 500 episodes):

| arm | collapsed | never reached 30% |
|---|---|---|
| wild type | 2 / 32 (peaks 75–82% → 7–9%) | 0 |
| gap-only null | **6 / 32** (peaks 63–85% → 0–20%) | 1 |
| chemical-only null | 5 / 32 (peaks 58–91% → 0–38%) | 2 |

The wild type collapses less often than the gap-only null, but not significantly (Fisher p = 0.26).
Every arm collapses sometimes.

**The lead on seeds where both learners' last-quarter plateau clears a threshold**, a post-hoc
description. The threshold is on the plateau alone, so a subset can still hold learners that peaked
higher and declined: the ≥ 30% subset includes four of the eight collapsed learners (wild type 2313
and 2328; gap-only null 2305 and 2307).

Every p value in this logbook is two-sided.

| seeds kept | n | lead | 80% CI | two-sided p |
|---|---|---|---|---|
| all (registered) | 32 | +0.126 | [+0.067, +0.178] | 0.012 |
| both last-quarter plateaus ≥ 5% | 29 | +0.093 | [+0.039, +0.148] | 0.062 |
| both last-quarter plateaus ≥ 30% | 25 | +0.073 | [+0.020, +0.122] | 0.15 |

So the lead shrinks as seeds with a low final plateau, most of them the null's collapses, are set
aside. On the 25 seeds where both learners end at 30% or more, the wild type still leads, though not
significantly by the Wilcoxon (p = 0.15).

**`episodes_to_30pct_success`:** the wild type reaches 30% **458 episodes sooner** [166, 707],
p = 0.023. Median episodes to 30%: the wild type 1,397, the gap-only null 2,007, the chemical-only
null 1,342.

**Against the chemical-only null:** +0.086 [+0.019, +0.149], p = 0.14, and 289 episodes sooner to
30% [−122, +678]. On hard350 through the body this contrast was +0.016 (Logbook 085).

**The frozen floors** are 0% for every wiring, so the floors' own gap is zero.

**The kinematics** ([kinematics.json](supporting/092-thermal-body/kinematics.json)):

| arm | frequency (Hz) | wavelength (body lengths) | speed (body lengths/s) | reversal fraction |
|---|---|---|---|---|
| wild type, trained | 0.300 | 0.650 | 0.114 | 0.007 |
| gap-only null, trained | 0.301 | 0.648 | 0.112 | **0.043** |
| wild type, frozen | 0.303 | 0.643 | 0.109 | 0.228 |
| gap-only null, frozen | 0.302 | 0.644 | 0.109 | 0.219 |

The gait is the same: frequency, wavelength and speed match within about 2%, trained and frozen, and
untrained the two wirings are identical in every instrument. **Trained, the null reverses six times
as often** (4.3% of steps against 0.7%). Untrained they reverse alike, so this is something the
null's policy does, not its body, and it fits the null's collapses (C.3 instruments, 32 seeds per
arm).

**The horizon check (Decision G).** The body's learners take a median **1,616** episodes to reach
30%, the point-worm anchor's **307**: a ratio of **5.3**, not the 2 the doubled budget assumed. If a
lead in episodes scales with learning time, a body lead in `auc_success` over 6,000 episodes is about
2.6 times the same relative lead over the point worm's 3,000. The panel's +0.126 then corresponds to
about **+0.048** on the point worm's scale, against the point worm's +0.096 at target 35. That
comparison crosses cells (targets 20 and 35) and rests on the scaling assumption, so it is
description only.

**The point-worm anchor** (pilot seeds, target 20, 6,000 episodes) saturates: both wirings plateau at
97–99%, and the wild type's lead is −0.046. At target 20 the point worm cannot show a learning-speed
advantage, so the body reading has no point-worm pairing on this cell.

## What this establishes, and what it does not

**Established (registered):**

- **The wild type's gap-junction placement makes the connectome learn the thermal cell faster through
  the kinematic body,** against a null whose gap junctions alone are rewired, by more than two-thirds
  of the point worm's committed gap-junction effect, under PPO.

**Described, with that standing:**

- **It is not the gait.** The two wirings crawl alike, and untrained they are identical, although
  the null also rewires the motor neurons' gap junctions. The trained null's extra reversals are
  behaviour its policy learned, not its body.
- **It is part speed, part stability.** On the 25 seeds where both learners' last-quarter plateau
  is at least 30%, the wild type's lead is smaller (+0.073): its 80% interval still excludes zero,
  but its Wilcoxon p is 0.15. That subset is defined by the final plateau, not by an absence of
  collapse, and still includes four collapsed learners. Much of the rest comes from the null's
  low-plateau seeds, most of them collapses.
- **The chemical wiring may matter here too.** Against the chemical-only null the lead is +0.086
  with p = 0.14, larger than on hard350. It was not registered, and is not read.
- **The size is uncertain.** The panel's spread was 3.6 times the pilot's, and the lead's size on the
  point worm's scale depends on an assumption.

**Not established:**

- **That the body preserves the point worm's effect at its size.** The comparison crosses cells, and
  the point worm saturates at target 20.
- **Why the null collapses more often,** or whether that difference holds on more seeds.
- **Anything about interior wiring.** The gap-only null rewires all gap junctions, sensory and motor
  included; no boundary-preserving gap null was run.

## Biological fidelity

- **Gap junctions are the layer under test.** Real *C. elegans* gap junctions are formed by
  innexins, and innexin mutants such as *unc-7* and *unc-9* have locomotion defects. Here the two
  wirings crawl alike, and identically untrained, so rewiring the gap junctions changes how the
  policy learns, not the gait this body produces. Whether this body could express a gap-junction
  gait defect at all was not tested.
- **The body is a 12-segment kinematic model**, not a muscle-driven one. It turns by undulating, and
  learns about five times more slowly than the point worm.
- **Thermal sensing is partly synthetic.** Temperature is read at the head, but its left–right
  difference comes from a head-sweep around the heading, not from the body's own head swing. C.3
  found the weathervane survives without the sweep.
- **The target was lowered for the body.** Target 35 was out of reach even for the MLP in 3,000
  episodes; target 20 is a lighter foraging demand.

## Next

- **M.9 is met.** Its registered reading is `move_wt`.
- **S8b, Phase 8's synthesis**, comes next. It can now state that one wiring effect survives the body,
  with these conditions.
- **Candidates, none scheduled:**
  - **The collapses.** Whether the wild type's lower collapse rate holds on more seeds, and what
    drives late PPO collapse at 6,000 episodes.
  - **A boundary-preserving gap null,** to ask whether the advantage is in the sensory and motor gap
    junctions or the interior.
  - **The chemical-only contrast through the body on this cell,** registered on its own.

## Artefacts

**The registration and the panel:**

- [launch.md](supporting/092-thermal-body/launch.md): the registration.
- [panel.json](supporting/092-thermal-body/panel.json): the gates, the reading and its verdict, and
  both nulls' gaps on both metrics.
- [per-seed.csv](supporting/092-thermal-body/per-seed.csv): one row per seed, every arm's plateau and
  floor and both gaps.
- [sensitivity.json](supporting/092-thermal-body/sensitivity.json): collapses, the lead on seeds where
  both learn, and the horizon check.
- [kinematics.json](supporting/092-thermal-body/kinematics.json): C.3's instruments for the wild type
  and the gap-only null, trained and frozen.

**The stages before it:**

- [stage1/](supporting/092-thermal-body/stage1/): the failed positive control at targets 35 and 20.
- [retry/](supporting/092-thermal-body/retry/): the retry at 6,000 episodes.
- [pilot/](supporting/092-thermal-body/pilot/): the connectome pilot, the point-worm anchor, the gate
  preflight and the pilot kinematics.

**To reproduce:**

```bash
uv run python scripts/analysis/thermal_body.py mlp-gate --logs campaigns/m9-retry/logs \
  --seeds $(seq 9721 9728) --targets 20 --out mlp_gate.json
uv run python scripts/analysis/thermal_body.py score --logs campaigns/m9-panel/logs \
  --seeds $(seq 2301 2332) --target 20 --out-dir /tmp/m9-score --out panel.json --csv per-seed.csv
uv run python scripts/analysis/thermal_body.py kinematics --logs campaigns/m9-panel/logs \
  --seeds $(seq 2301 2332) --target 20 --out kinematics.json
```

The campaign directories and session records are archived off-repo.
