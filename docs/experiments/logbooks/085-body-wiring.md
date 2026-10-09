# 085: Through the Body, the Wild Type's Lead Over the Chemical-Only Null Is Unresolved, Bounded Near 0.04 (Phase 8b C.1e)

**Status**: completed — **`unresolved_at_this_sensitivity`.** Through the kinematic body, under PPO, the
wild type learns the 500-step cell at most slightly faster than the chemical-only null, and plateaus at
the same level. The difference's interval reaches above the registered minimum, so it is neither a move
nor a non-move.

| registered reading, seeds 1801–1864 | mean | 80% CI | q | state |
|---|---|---|---|---|
| wild type − chemical-only null, `auc_success` | **+0.0163** | [−0.012, +0.044] | 0.26 | `unresolved` |

The reading's interval spans the **0.0367** minimum, so by the registered map the panel is unresolved
at this sensitivity. **The boundary stage does not run**: it was gated on `move_wt`.

Beside it:

- **Plateaus** are level: the wild type 58.6%, the null 58.8%, against 0% floors.
- **Episodes to 30% success**: +14 [−104, +129].
- **Competent seeds**: the wild type 56 of 64, the null 55 of 64 (exact McNemar p = 1.0).
- **MLP-PPO**, beside: 70.8% mean plateau, 60 of 64 seeds competent.
- **The achieved MDE is 0.054** (sd 0.175), against the pilot's projected 0.050.

**What the panel bounds.** On its 80% interval, the wild type's learning-speed lead through the body is
at most about **0.044 `auc_success`**, and its plateau lead is nil.

**Date**: 2026-10-10.

**OpenSpec change**: `add-body-wiring-contrast`.

**Pre-registration**: [supporting/085-body-wiring/launch.md](supporting/085-body-wiring/launch.md),
committed before any panel run, after the spec review and the gate preflight. The pilot that sized it is
[Logbook 084](084-body-wiring-pilot.md).

## Objective

Phase 8b's central reading: **does the wild-type wiring help a connectome forage through the body,
against a null that differs from it only in its chemical wiring?** It reads in a new reference frame, the
body cell, with nothing here a delta against 029, block V or a point-worm result.

## Method

**The cell.** hard350's food layout at 500 steps, through the frozen kinematic body (steering gain 2,
one-step reversals, wave floor 0.25), with the body-drive gain vector (D18), Emmons 2024 with reversal
on, settling dynamics at depth 4, 3,000 episodes, `entropy_coef` 0.004.

**The arms.** The wild type and the chemical-only null (D21's primary), each learning under PPO and
frozen, with MLP-PPO beside. **Seeds 1801–1864 (64), 320 runs**, all succeeded in 20.9 hours from a
separate worktree.

**The reading.** Wild type minus null on `auc_success`, paired by seed, its 80% bootstrap interval and
its two-sided Wilcoxon q, classified at the minimum the pilot fixed, 0.0367: `move_wt`, `move_null`,
`below`, `no_move` or `unresolved`. The boundary-preserving null was gated on `move_wt`. The current
null was left out, since it was read only beside and the 64-seed panel had no room for it. The
frozen-wiring learner left before the pilot, having failed its floor.

## Results

### Gates

Both learning arms beat their floors on every seed (floors 0.0% and at most 0.3%), and neither
approaches the 90% bar. The panel is readable.

### The registered reading

**+0.0163 `auc_success`** [−0.0123, +0.0442], q = 0.26, the wild type ahead on 36 of 64 seeds. The
interval includes zero and reaches past the minimum, so the state is `unresolved`. The panel can say the
lead is below about 0.044 on its 80% interval. It cannot say the lead is below the minimum, nor that it
is above zero.

### Achieved sensitivity, beside the registered one

| | pilot (16 seeds) | panel (64 seeds) |
|---|---|---|
| per-seed sd of the gap | 0.161 | 0.175 |
| MDE at the panel's n | 0.050 (projected) | 0.054 (achieved) |
| minimum | 0.0367 | 0.0367 |

The spread came in slightly above the pilot's. An effect between the minimum and the MDE was always
going to read `unresolved`; that was stated before launch. It is reported, and never used to re-read
the verdict.

### Described beside

| measure | wild type | chemical-only null | difference |
|---|---|---|---|
| mean plateau | 58.6% | 58.8% | −0.2 points |
| competent seeds (≥ 30%) | 56 / 64 | 55 / 64 | McNemar p = 1.0 |
| episodes to 30% success | | | +14 [−104, +129] |

MLP-PPO, at the same cell and entropy: 70.8% mean plateau, 60 of 64 seeds competent.

### For scale, not comparison

On the point worm's hard350 cell (350 steps, Cook 2019), the wild type led the chemical-only null by
+0.022 `auc_success` [+0.007, +0.036] ([Logbook 074](074-null-strength-control.md)). The body panel's
interval contains that value. Different cell, connectome release and step count make the two
incomparable as a delta, and the protocol forbids reading one against the other. They are set side by
side only to show the panel's bound is of the size the point worm's own lead had.

## What this establishes, and what it does not

- **Established**: the connectome learns the body cell under PPO on every seed, at the level of the
  null. With the anatomical readout, the drive gains and a dimension-matched entropy bonus, foraging
  through the body is learnable from this wiring.
- **Established, as a bound**: through the body, the wild type's lead over the chemical-only null in
  learning speed is at most about 0.044 `auc_success` on the 80% interval, and its plateau lead is nil.
- **Not established**: whether there is any lead. The interval includes zero and the minimum alike, so
  neither "the wiring helps through the body" nor "it does not" may be cited.
- **Not established**: anything about the interior wiring. The boundary stage was gated on a primary
  move and did not run.
- **Not established**: anything for a learner that reads the wiring without writing it. That learner
  failed its floor before the pilot (Logbook 084).

## Biological fidelity

The conditions recorded in Logbooks 083 and 084 carry over unchanged:

- only short reversals are possible;
- the wave floor, steering gain and drive gains are calibrations, not measurements;
- the body's speed sits at the band's floor;
- the entropy bonus is a learning setting matched to the action's width.

The readout is anatomy: which cells drive which muscles, signed by transmitter. Only its 25 gains are
learned, identically for every wiring.

## Next

- **C.1e's tracker item closes on this reading.** Its PPO half is `unresolved` with the bound above; its
  reading-learner half is unreachable-with-reason; its interior reading is gated and not run.
- **A sharper contrast needs lower spread, not more seeds.** Seeds differ widely in when they begin to
  learn, and that drives per-seed gaps between about −0.2 and +0.3. A design that measured learning
  onset separately, or trained longer so every seed reaches its plateau, might resolve it. Either is a
  new registration, not a re-read of this one.
- **The combined paper's body half** rests on this bound, beside the point worm's results.

## Artefacts

- [launch.md](supporting/085-body-wiring/launch.md), [preflight.json](supporting/085-body-wiring/preflight.json):
  the registration and its gate preflight on the pilot's runs.
- [panel.json](supporting/085-body-wiring/panel.json): gates, gaps, the reading and its verdict, the
  competence frequency, the MLP beside.
- [per-seed.csv](supporting/085-body-wiring/per-seed.csv): each seed's plateaus, floors and gaps.
- Reproduce: `scripts/analysis/body_wiring.py panel --logs campaigns/c1e-panel/logs --out-dir <dir> --out panel.json --csv per-seed.csv`. The panel's campaign directory and the worktree's session
  records are archived off-repo.
