## Context

- **The primitives** (`env/primitives.py`):

  - `_start_bout` gives a reversal that starts on a lawn the speed `-on_food_reversal_speed_mm_per_s` (0.069), with its duration from the on-food gamma.
  - `step_primitives` calls it with `on_food` from `Continuous2DEnvironment._on_food_for_reversal`, the lawn's geometry at the start of the step.
  - A run step moves the worm at the controller's speed, `map_action(...).speed_mm_per_s`.
  - `PrimitiveState` holds the ventral side and the running bout.

- **The batched evaluator** (`evaluation/batched.py`):

  - `_move` computes each animal's run speed, `v_max·(x₀ + 1)/2`, for the step.
  - `_start_bouts` loops over the idle animals, calls `_start_bout(..., on_food=...)` and copies the bout's speed into `bout_speed`.
  - An idle animal that starts no bout runs at that speed.

- **Logbook 131's probe** patched the batched evaluator in each worker. It differed from this change in three ways:

  - it fed the line the realized displacement of the last 3 run steps, after any wall clamp;
  - it used the band cohorts' line;
  - it had no tracker floor.

  Its clip was [0.049, 0.084].

- **The record** (`supporting/132-speed-following-reversals/coupling.json`, from `scripts/analysis/reversal_speed_coupling.py`):

  | Set | Bouts (dates) | Line, 95% bootstrap over dates | Reversal speed, deciles 1 and 10 | Duration, deciles 1 and 10 |
  |---|---|---|---|---|
  | held out, every on-lawn start | 83,697 (24) | 0.0464 [0.0451, 0.0478] + 0.606 [0.561, 0.655] × v | 0.0484, 0.0839 | 2.00, 2.25 s |
  | held out, more than 0.5 mm inside | 21,940 (24) | 0.0458 + 0.574 × v | 0.0466, 0.0770 | 2.03, 2.30 s |
  | band, every on-lawn start | 148,557 (41) | 0.0473 + 0.578 × v | 0.0493, 0.0842 | 1.98, 2.26 s |

  A paused worm's 1-s displacement in the vendored still-worm library is 0.0075 mm (mean) and 0.0096 mm (RMS). The slowest decile's speed before is lower, 0.0046 mm/s.

## Goals / Non-Goals

**Goals**

- An on-lawn reversal's speed that follows the worm's run speed as a real worm's does, from cohorts the bands do not use.
- Off by default and byte-identical when off.
- Both runners computing the same speed from the same history.

**Non-goals**

- **A duration that follows the speed.** It rises from 2.00 to 2.25 s across the deciles, and is reported. Logbook 131's refit passed without it, and keeping the pooled gamma leaves every draw where it is.
- **A coupling off food.** Off-food reversals keep their published values, and the benchmark's off-food statistics are read from the detector, not from reversal speed.
- **Kinematics keyed on the roaming or dwelling label.** The body cannot see a state; it can see a speed.
- **A new default.** C.1's and C.2's fits are recorded on the pooled body.

## Decisions

01. **The run speed the primitives moved the worm at.** The history takes each run step's commanded speed, `map_action(...).speed_mm_per_s`, not the displacement after a wall clamp.

    - Both runners have that number at the same point, without position bookkeeping.
    - Neither runner's history depends on how it handles walls.
    - On the assays, a lawn lies far from the wall. Logbook 131's probe read realized displacements; the two differ only on steps at a wall.

02. **A 3-s window of run steps.** The window holds the last N = round(3 s / step) run steps' speeds, oldest first.

    - Bout steps neither enter it nor clear it. It persists across bouts and assay phases, as the worm's state does.
    - Until N run steps are seen, a reversal takes the pooled 0.069 mm/s.
    - This is the measurement's rule: the last 3 run seconds before the bout, wherever they fell.

03. **The line from every on-lawn start, not the inside stratum.** The primitives give the on-food kinematics to every reversal that starts on a lawn, the edge included, so the line comes from the bouts the rule governs.

    - The inside stratum is reported beside. Its line agrees at the slow end and runs lower at the fast end (0.077 against 0.084 mm/s), where roaming worms near the edge reverse fastest.
    - The pooled 0.069 stays the inside stratum's value, as Logbook 108 measured it.

04. **No tracker floor.** The line was fitted to tracked speeds, and the simulator applies it to true ones, so reading the true speed through the tracker's noise, √(v̄² + d²), looks natural. The record rules it out:

    - The slowest decile's speed before, 0.0046 mm/s, is below the still-worm library's paused-worm mean, 0.0075. The library's worms read faster than many real worms in these tracks, so its d overstates the tracker's noise here.
    - Through d = 0.0075, a still simulated worm would reverse at 0.0464 + 0.606 × 0.0075 = 0.0509 mm/s, faster than the slowest real decile's 0.0484. The bias would fall on dwelling reversals, the ones this change is for.
    - Without it, the slow end is set by the first knot (Decision 5).

05. **Knots, not a line** *(amended in the PR's review, 2026-11-10)*. A reversal's speed is interpolated linearly between knots and is flat beyond the outermost. The knots are the mean speed before and the mean reversal speed in deciles 1 to 9 of the speed before, and in the top decile split at its 95th and 99th percentiles (12 knots, 0.0484 to 0.0961 mm/s).

    - The relation is concave at the fast end. The line, clipped to the deciles' range (0.048 to 0.084 mm/s) as first designed, ran 0.002 to 0.004 mm/s low in the 80th to 95th percentiles and capped the fastest 5% of reversals at 0.084, where real worms reverse at 0.086 and 0.096.
    - At the slow end, the first knot holds a still worm at 0.0484, the slowest decile's real value, as the clip did.
    - The line stays in the record as description.

06. **Speed only.** The duration's gamma is one call either way, so the option changes no draw: a seed's bout kinds, start steps and durations stay those of the pooled setting until the worm's path changes what its controller senses.

07. **Off by default:** `continuous.on_food_reversal_speed: pooled | follows_run_speed`, `pooled` when unset.

    - Logbook 108's amendment changed the default before any fit depended on the primitives. Here C.1's and C.2's records do, and they keep reproducing.
    - The fits that follow set the option on.
    - It is refused without primitives, and `follows_run_speed` is refused with `reversal_kinematics: off_food`, where no reversal takes the on-food values.

08. **One `measured` entry, `on_food_reversal_speed_coupling`,** with its script and record.

    - It holds the knots and the window.
    - The loader refuses a missing entry by name, and checks that the knots are at least two, equal in length and finite, the speeds before non-negative and strictly increasing, the reversal speeds positive, and the window positive.
    - `PrimitiveKinematics` always carries the coupling. Only the option reads it.

09. **The same arithmetic in both runners, through one function.** Each runner keeps the window as a Python list of floats, oldest first. Both take its mean through the primitives' `mean_run_speed` (Python's `sum`, divided by N) and the interpolation through `SpeedCoupling.speed`, in pure Python. Their inputs are equal, since both compute the run speed as v_max·(x + 1)/2 in float64 in the same order, so the speeds match bit for bit. The parity test checks it.

10. **The speed reading is passed to `_start_bout`.** The batched evaluator builds a fresh `PrimitiveState` at each bout start, so the window cannot live in the state it passes. `_start_bout` takes the window's mean as an argument (None until the window fills). The per-step runner keeps the window in its `PrimitiveState`, and the batched evaluator keeps one Python list per animal.

11. **Without lawns the option cannot act, and cannot be set** *(amended in the PR's review)*. Primitives need a step of at most one second, and a configured step is accepted only on lawns, so a primitives config without lawns is refused before the option is read. A refusal of its own would be unreachable, so a test pins the guarantee instead.

## Risks / Trade-offs

- **[Fitted on tracked speeds, applied to true ones]** → tracking noise lifts the slowest tracked speeds, and the clip holds the slow end at the slowest decile's real reversal speed (Decision 4). Tracking noise also lengthens the reversal's own path a little, as Logbook 108 reported for the pooled value. That is unchanged here.
- **[Delta turns count differently]** → Scheer's data has no delta flag, so a real worm's deep dorsal coils count as run seconds in the measurement, at their low 1-s displacements. The simulator's delta bouts stay out of the window, as every bout does, consistent with what the primitives call a run. A fitted C.2 makes about 3 delta turns a minute on the lawn, so the mismatch is reported in Logbook 132. Where it matters, it makes a simulated worm's window read a little faster than a real worm's would.
- **[A controller now sets its on-lawn reversals' speed through its run speed]** → that is the coupling real worms show. Off food nothing changes, and with the option off nothing changes anywhere.
- **[The first parity test could not see the window]** *(found in the PR's review)*: its MLP's run speeds put every on-lawn reversal on the line's ceiling, and its slow controller ran at one speed. A controller whose speed changes every step inside the knots, and an exact comparison of every bout's kind, food, window mean, speed and duration in both runners, now catch each of five deliberately broken windows on both lawns (`supporting/132-speed-following-reversals/mutants.log.txt`).
- **[Logbook 131's refit ran on a slightly different body]** (the band line, realized displacements, a 0.049 clip) → a confirmation task, as description. Its answer is graded on the same 10 fresh populations under the package's option and under Logbook 131's patch, paired, with the rule fixed in advance: at least 8 of 10 under the option, or the difference is diagnosed before C.2's registered fit relies on it.
- **[The forced-bout test assumes one on-food speed]** → it runs with the option unset. A new test forces reversals after runs at known speeds.
- **[Shorter steps]** → the window counts run steps, N = round(3 s / step). A test at 0.5 s checks 6 run steps.

## Migration Plan

None: the option is off by default, and no record moves.

## Open Questions

- Whether C.2's registered fit and C.1's refit set the option. Proposed: both do, fixed in Change B's registration.
