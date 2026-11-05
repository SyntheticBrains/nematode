## Why

Benchmark v4 ([Logbook 124](../../../docs/experiments/logbooks/124-benchmark-v4.md)) fails C.1's v3 answer in all 5 of its registered populations: it reverses too rarely on the lawn (0.78–0.85 onsets per on-lawn minute against real worms' 2.28; z −6.4 to −6.7). Before refitting, the maintainer asked whether C.1's model can reach v4 at all, and if not, why.

Discarded-seed probes (2026-11-05, description) answer it:

- **C.1's 25-parameter model cannot.** Real worms make most on-lawn onsets while dwelling (2.12 per minute in dwelling windows, which are 83% of on-lawn time). C.1 dwells at a near-constant 0.013 mm/s with its dwelling reversal rate at its floor, and raising that rate breaks the off-food assays, since the same rate sets reversals off food.
- **One mechanism closes it:** a dwelling reversal rate of its own on food. With it, a fit continued from C.1 (v3)'s answer passes the wild type in 4 of 5 populations (D 1.28–1.44).
- **From the box centre, no fit reaches it** (three runs, D 1.78–3.02), including one with the states ordered after a run swapped them.

The maintainer's decision (2026-11-05): **defer the refit** until C.4 approaches, to run once on the benchmark current then, and ship the diagnosis now. A warm start would conflict with D38 as amended (C.1's x₀ is set without search data), and C.2, F.0b-ii and W do not depend on C.1's refit.

## What Changes

- **`BatchedMechanistic` gains `r_dwell_on_food`:** a dwelling animal's reversal rate wherever food contact is above 0, in place of `r_dwell`, before the klinokinesis, edge and search terms. Off food, and while roaming, the state's rate applies. Off by default: the model is byte-identical without it (59 of 59 identity jobs). Two other candidates, a reversal–speed coupling and intermittent dwelling, were probed at commit `886e7bfc` and are not kept.
- **The probes,** every one with its record and script as run, in `docs/experiments/logbooks/supporting/125-c1-v4-fit/probes/`, with the drafted fit design parked beside them (`parked_fit_design.md`, not registered).
- **Logbook 125**, the diagnosis.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `mechanistic-baseline`: an optional on-food dwelling reversal rate.

## Impact

- **Code:** `quantumnematode/evaluation/controllers.py` (`BatchedMechanistic`); tests.
- **Records:** `supporting/125-c1-v4-fit/`; Logbook 125; the tracker (C.1, C.4, the seed table); AGENTS.md; the CHANGELOG.
- **No registered run.** The eight probes took 10.75 h on 16 workers.
