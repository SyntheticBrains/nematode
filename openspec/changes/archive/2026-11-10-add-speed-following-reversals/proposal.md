## Why

D36 fixes the primitives' kinematics from data. Since the D36 amendment of 2026-10-24
([Logbook 108](../../../docs/experiments/logbooks/108-food-dependent-kinematics.md)), every reversal
that starts on a lawn backs at one speed, 0.069 mm/s. That value is pooled over roaming and dwelling
worms, measured on Scheer's held-out OD 2 cohorts.

[Logbook 131](../../../docs/experiments/logbooks/131-c2-clean-start.md) found that real on-lawn
reversals follow the worm's own speed just before them. On the held-out cohorts:

| Speed in the last 3 run seconds, deciles 1, 3, 5, 7 and 10 (mm/s) | 0.005 | 0.011 | 0.018 | 0.026 | 0.064 |
|---|---|---|---|---|---|
| reversal speed (mm/s) | 0.048 | 0.052 | 0.057 | 0.063 | 0.084 |

A dwelling worm reverses at about 0.059 mm/s, and a roaming one at about 0.087. With one pooled
speed, every simulated dwelling reversal carries the worm about 20% too far. That speeds up the 10-s
windows the classifier reads as dwelling, so the more on-lawn reversals a controller makes, the faster
its dwelling looks.

C.2's fit could not hold both the on-lawn onset rate and the dwelling windows' speed inside their
bars. Its best answers passed the wild type in 8 or 9 of 15 fresh populations. With the coupling
patched into the evaluator in scripts, and the network refitted, the same procedure passed all 24
statistics and every mutant in 15 of 15.

This change puts that coupling into the primitives, as an option, so the registered fit (Change B) and
C.1's refit before C.4 can run on it. The maintainer's order of 2026-11-10: this change, then C.2's
registered fit.

## What Changes

- **An on-food reversal that follows the run speed.** Under the new option, a reversal that starts on
  a lawn backs at a speed interpolated between measured knots at v̄, and flat beyond the outermost:
  0.048 mm/s for a still worm, up to 0.096 mm/s for the fastest.
  - v̄ is the worm's mean run speed over its last 3 s of run steps.
  - The knots are the mean speed before and the mean reversal speed in deciles 1 to 9 of the speed
    before, and in the top decile split at its 95th and 99th percentiles. The relation is concave
    at the fast end, where a line overshoots (*amended in the PR's review, 2026-11-10*; the line
    first proposed, clipped to 0.048–0.084 mm/s, understated the fastest 5% of reversals).
  - Until the worm has 3 s of run steps, the reversal takes the pooled 0.069 mm/s.
  - The duration keeps the pooled on-food gamma.
  - Off every lawn, nothing changes.
- **The values come from the held-out cohorts,** measured by a committed script with the simulator's
  own rule: every on-lawn bout, and its last 3 run seconds whenever they fell. The fit gives 0.0464
  [0.0451, 0.0478] + 0.606 [0.561, 0.655] × v_before on 83,697 bouts over 24 recording dates (95%
  bootstrap over dates), with deciles from 0.048 to 0.084 mm/s. The line is reported as description;
  the knots are what the primitives take. The band cohorts give 0.0473 + 0.578 × v_before, reported
  beside.
- **`primitives.json`** gains one `measured` entry, `on_food_reversal_speed_coupling`, naming the
  script and its record.
- **A config option,** `continuous.on_food_reversal_speed: pooled | follows_run_speed`.
  - It is `pooled` when unset, byte-identical to today.
  - Setting it needs `locomotion: primitives`, and `follows_run_speed` needs
    `reversal_kinematics: food_dependent`. Either is refused otherwise.
- **Both runners keep the same history,** from the same quantity in the same order: each run step's
  speed as the primitives move the worm. So the per-step runner and the batched evaluator still make
  the same RNG calls and draw the same bouts.
- **A confirmation on the package's option:** Logbook 131's passing answer graded on the same 10
  fresh populations under the package's option and under Logbook 131's patch, paired. The rule is
  fixed in advance: the option carries the probe's result if the wild type passes in at least 8 of
  10; otherwise the difference is diagnosed before C.2's fit relies on it. Description, on discarded
  seeds.
- **D36 amended, dated,** in the roadmap and the tracker.

## Capabilities

### Modified Capabilities

- `locomotor-primitives`: an optional on-food reversal speed that follows the worm's run speed, from
  a measured entry.

## Impact

- **Code:**
  - `env/primitives.py`: the coupling entry, the run-speed history in `PrimitiveState`, and the
    reversal's speed in `_start_bout`;
  - `env/continuous_2d.py`: the option, passed to `step_primitives`;
  - `evaluation/batched.py`: the same history as arrays;
  - `utils/config_loader.py`: the option and its refusals.
- **Data:** `data/locomotion/primitives.json` and its `PROVENANCE.md`.
- **Scripts:** `scripts/analysis/reversal_speed_coupling.py`, writing its record under
  `docs/experiments/logbooks/supporting/132-speed-following-reversals/`.
- **Default behaviour:** unchanged. With the option unset, every run is byte-identical to before it,
  RNG state included, and no committed record moves. With the option on, a reversal's speed changes
  but no draw does: the speed is computed, not sampled. So a seed's bout kinds, start steps and
  durations match the pooled setting's until positions diverge enough to change what the controller
  senses.
- **Downstream:** C.2's registered fit and C.1's refit before C.4 set the option on. Earlier C.1 and
  C.2 records name the pooled setting, which they reproduce.
- **Cost:** about a day. No campaign. The confirmation grade takes minutes.
- **Seeds:** tests on 9601–9632. The confirmation grade on 880,000,000 + 200·g + i, discarded.
