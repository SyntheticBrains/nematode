## Why

D36 fixes the primitives' kinematics from data, and E.2 took them from published off-food sources.
B.1 checked them against Scheer & Bargmann 2023's on-food flags on the held-out OD 2 wild-type cohorts
([Logbook 105](../../../../docs/experiments/logbooks/105-reference-extraction.md)) and found **a
disagreement**:

| Stratum | Reversal duration (s) | Reversal speed (mm/s) |
|---|---|---|
| more than 0.5 mm inside the lawn | **2.05 [2.01, 2.08]**, n = 24,890 | **0.069 [0.067, 0.071]** |
| more than 0.5 mm outside | 3.30 [2.99, 3.67], n = 169 | 0.174 [0.148, 0.205] |
| published, off food (Zhao 2003) | 2.69 ± 1.75, span 2.10–3.71 | 0.14 |

On a lawn, a reversal moves about 0.14 mm. The primitives make it 0.38 mm.

That matters for what B grades on a lawn. The roaming/dwelling classifier reads speed and turning in
10-s windows. A slow worm whose reversals carry it nearly three times too far is read as roaming more
often than its runs warrant. On Flavell's uniform lawn, seeds 9001–9064, a slow controller (0.02 mm/s
runs, reversal hazard 0.05/s) was classified **23.3% roaming with the off-food reversals and 0% with
on-food ones**. That first pilot drew on-food durations with the off-food CV (SD 1.33 s), before the
SD was measured. Task 5.1 re-runs it on the vendored 0.899 s. Fast controllers barely moved. C fits controllers against these statistics, so the
bias would be fitted into every arm.

The maintainer chose the form on 2026-10-24: **food-dependent kinematics, as its own change landing
before B.4.** B.4's negatives run on the primitives, so the primitives must be final before any
negative is read.

## What Changes

- **Reversal kinematics keyed on the lawn.** A reversal that starts while the worm is on a lawn draws
  its duration from gamma(2.047 s, SD 0.899 s) and backs at 0.069 mm/s. Off every lawn it keeps the
  published 2.69 ± 1.75 s at 0.14 mm/s. The bout's kinematics are fixed when it starts.
- **"On a lawn"** is the test the capture and the classifier already use: the worm's position lies in a
  lawn's cell footprint, in a phase that has lawns. It is geometric, like Scheer's in-lawn flag, so a
  depleted cell still counts.
- **`primitives.json`** gains two on-food entries citing Scheer 2023 and Logbook 105's record, under a
  new status, `measured`: computed from deposited tracks by a committed script.
- **The kinematics check records SDs.** `primitive_kinematics_check.py` writes each stratum's SD beside
  its mean, so the vendored SD traces to a committed record. Every existing value stays byte-identical.
- **A config option**, `continuous.reversal_kinematics: food_dependent | off_food`, defaulting to
  `food_dependent` under primitives. `off_food` reproduces E.2–B.2's records. It is refused with free
  locomotion.
- **The batched evaluator** draws the same: the parity tests run on the new default.
- **D36 amended, dated**, in the roadmap and the tracker. The pilot and its seeds, 9001–9064, are added
  to the seed table as used.

Omega duration, forward speed, turning rate and the pause fraction stay as they are. Logbook 105
reported them as description, and its rule acted only on reversal duration. Reversal speed changes
with it: one bout's distance is its speed times its duration, and the on-food speed comes from the same
bouts.

## Capabilities

### Modified Capabilities

- `locomotor-primitives`: kinematics from data gain on-food entries and the `measured` status; reversal
  kinematics depend on the lawn.

## Impact

- **Code**:
  - `env/primitives.py`;
  - `env/continuous_2d.py`;
  - `evaluation/batched.py`;
  - `utils/config_loader.py`.
- **Data**: `data/locomotion/primitives.json` and its `PROVENANCE.md`.
- **Scripts**: `scripts/analysis/primitive_kinematics_check.py` (SDs added), and its record
  regenerated.
- **Default behaviour**:
  - Free locomotion is untouched and byte-identical.
  - Under primitives, a reversal on a lawn changes. Its gamma is still one call, but a gamma draw's
    rejection sampler uses a shape-dependent share of the generator's stream. So a seed's later draws
    shift once a reversal starts on a lawn, and runs under the new default are not comparable draw by
    draw with earlier ones.
  - Two guarantees stand: `off_food` is byte-identical to before the option, and the per-step runner
    and the batched evaluator make the same calls in the same order, so their parity holds.
- **Committed records**:
  - Logbook 104's cost pilot and calibration, and B.2's simulated detector validation, ran before the
    amendment. They stay records of `reversal_kinematics: off_food`, the setting that reproduces them.
  - B.2's simulated gate reads only López-Cruz's off-food phase, where the kinematics are unchanged, so
    it is not re-run.
- **Cost**: about a day. No campaign. The pilot is re-run on the vendored values (minutes).
- **Seeds**: the pilot, 9001–9064, recorded as used; smokes on 5991–5999.
- **Downstream**: B.3's bands and B.4's negatives run on these primitives.
