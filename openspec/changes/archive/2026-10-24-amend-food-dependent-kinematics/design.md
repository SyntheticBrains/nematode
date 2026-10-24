## Context

- **The primitives** (`env/primitives.py`): `_start_bout` draws a reversal's duration from
  `kinematics.reversal_duration` and sets its speed to `-kinematics.reversal_speed_mm_per_s`.

  - The per-step environment calls it through `step_primitives`, from `move_agent_primitives`.
  - The batched evaluator calls `_start_bout` directly, from `_start_bouts`.
  - `GammaDuration.sample` makes one `rng.gamma` call, whatever its parameters. That call's
    Marsaglia–Tsang rejection loop uses a shape-dependent share of the stream. Checked: 200 draws at
    the off-food shape (2.36) and the on-food shape (5.18) leave `default_rng(1)` in different states.

- **"On a lawn" already has one meaning:**

  - Per step: `lawn_field.cell_at(position) is not None`. `lawn_field` is `None` in a phase without
    lawns. It is the same test as `Agent._food_contact`'s, and as the capture's `on_lawn`.
  - In the batched evaluator: `_cells_under(pos) >= 0`, which respects each animal's `lawn_on`.

- **B.1's record** (`supporting/105-reference-extraction/kinematics.json`) has each stratum's mean,
  interval and n, but no SD. The SD (0.899 s, CV 0.44) was computed from the same bouts, by the same
  rule, on 2026-10-24.

- **The on-food distribution.** It is right-skewed, with most mass at 1.3–2.3 s. Against a gamma of the
  same mean and SD, discretised to thirds of a second like the flag:

  | | 10th | 25th | 50th | 75th | 90th | 99th percentile |
  |---|---|---|---|---|---|---|
  | Scheer, on the lawn (s) | 1.33 | 1.33 | 1.67 | 2.33 | 3.33 | 5.33 |
  | gamma(2.047, 0.899) (s) | 1.33 | 1.67 | 2.00 | 2.67 | 3.33 | 5.00 |

  The fit is close in the tail and a third of a second high in the body. At the 1-s step, bouts are
  whole-step anyway (`Bout boundaries`).

## Goals / Non-Goals

**Goals**

- Reversals on a lawn as short and slow as the animal's, measured on cohorts the bands do not use.
- One switch that reproduces every record made before the amendment.
- Both runners drawing the same, from the same streams.

**Non-goals**

- Omega duration, forward speed, turning rate and pauses. Logbook 105 reported them as description:
  the omega flag is a posture-tracking span, not Broekmans's omega, and the turning-rate percentile is
  tracking noise.
- A shifted or empirical duration distribution. A gamma by mean and SD is how every other duration is
  given.
- Graded kinematics near the edge. The edge stratum (2.00 s, 0.076 mm/s) is close to the inside one,
  so a binary on/off test is enough.

## Decisions

1. **Key on the lawn's geometry, not its density.** Scheer's in-lawn flag is geometric, and B.1's strata
   were distances to the boundary. A depleted cell is still bacteria-covered agar to the animal's
   mechanosensors, and the classifier's on-lawn test is the same geometry. Keying on density would
   switch kinematics as a lawn is eaten.
2. **Fix a bout's kinematics at its start,** from the position at the start of the step it starts in.
   A reversal that carries the worm over the edge keeps its kinematics. It is one bout with one
   duration, and the start is the only point both runners already share.
3. **Default to food-dependent under primitives.** D36 says every arm and every negative runs on the
   same primitives. A default the generated configs had to opt into could be dropped by one config.
   The repository's convention of byte-identical defaults protects records. Here the records are kept by
   `reversal_kinematics: off_food`, which reproduces them, and free locomotion, the default everywhere
   outside Phase 10, is untouched.
4. **A new status, `measured`.** `verified` means read in a paper's text and `figure` read from a
   figure. A value computed from deposited tracks is neither. The loader accepts `measured` only with a
   source that names the script and its record.
5. **Speed with duration.** Both come from the same bouts. Changing duration alone would leave a
   reversal's distance 2.05 × 0.14 = 0.29 mm, against the animal's 0.14 mm.
6. **Held-out cohorts are the source.** The values come from the cohorts held out of the bands. The
   simulated worm's kinematics and the bands it is graded against come from disjoint recording dates.
   B.4's real-track positive control reads raw tracks, so the overlap with it cannot pass a worm by
   construction.

## Risks / Trade-offs

- **[Records before the amendment differ under the default]** → they name their setting, and
  `off_food` reproduces them exactly. The logbook lists them.
- **[The positive-control fixtures]** → E.2's positive control
  (`test_primitives_positive_control.py`) reads the first step on which the worm reaches the lawn.
  Every bout before that step starts off food, so its reading cannot move. It runs in the full suite as
  usual.
- **[Seeds shift after an on-lawn reversal]** → expected, and stated. No committed record compares runs
  across the two settings draw by draw.
- **[Gamma overstates the body by a third of a second]** → it is reported, and smaller than the 1-s
  step's own rounding.
- **[The on-food speed leans long]** → it is measured over a bout's whole seconds, so bouts that span no
  whole second (7,258 of 24,890) do not enter it. Tracking noise of a few µm per sample also lengthens the
  path at 0.07 mm/s. Both are reported in the logbook. Neither changes the speed's order of magnitude.
