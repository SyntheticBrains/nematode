## Why

C.1's 20-parameter model cannot reach benchmark v3's wild type. Eight discarded-seed probes (2026-11-02, description) traced why to three things it lacks, each a mechanism rather than a tuning:

- **one run speed per state.** v2's own reachability worm needed a per-window speed spread to match the graded window-speed medians and 90th percentiles (launch record 115, amendments 2–3). Without one, C.1b's fits faked the spread by shuttling back and forth (reversal-dominated, caught by v3). On v3 they fake it with extreme tortuosity, and dwelling speeds miss at z +6 to +10;
- **the same run kinematics on and off food,** so slow dwelling also slows search and chemotaxis;
- **one weathervane gain for two cues.** The gain that keeps the worm on the lawn (time on the lawn falls to z −9.5 at half strength) also sets Chen's butanone weathervane, about 11× too strong.

With all three (25 parameters):

- a warm-started fit passes 4 of 5 discarded populations (D 0.85–1.04);
- a fit from the box centre passes 3 of 5 (D 1.08–1.33);
- of three box-centre restarts, one reaches the passing basin, one comes within one family (D 1.44–1.53) and one lands far off (D 2.5–2.8).

The maintainer's decisions (2026-11-02):

- refit C.1 on v3 with the three mechanisms;
- a reduced recovery on the 25-parameter model first;
- C.1b's design for the registered fit, with 12 restarts and 5 selection populations per plateaued restart (decided at approval, before registration, to raise the verdict's odds).

## What Changes

- **`BatchedMechanistic` gains two optional mechanisms,** each off by default, so the model is unchanged without them (55 of 55 identity jobs byte-identical):

  - `v_off_food` and `sigma_off_food`: the run speed and curvature noise wherever food contact is 0, whatever the state;
  - `speed_log_sd_dwell` and `speed_log_sd_roam`: each state's run speed varies across 10-s windows as exp(sd·z) about its median. The normal is the sine partner of that second's curvature normal, so no draw is added.

  `kappa_off_food` (the κ probe's, already tested) is the third.

- **C.1 (v3), the 25-parameter model:** C.1's 20 parameters on F.0a's box, plus the five above, on ranges taken from the state priors (speed, curvature, gain) and from v2's reachability box (spread 0–1.5). None comes from search data.

- **The mutant encodings extended:** tax-4 also removes `kappa_off_food`. The four kinematic parameters are not chemosensory, so every mutant keeps them.

- **A reduced recovery** (F.0b-i's rules, 2 truths × 2 restarts) on the 25-parameter model, before any registered fit. The truths are the two passing probe answers.

- **The registered fit:** `scripts/analysis/c1_v3_fit.py`, C.1b's design on v3:

  - 12 box-centre restarts;
  - the ranking score against v3's bands;
  - the doubly guarded plateau, cap 330;
  - selection on 5 separate populations per plateaued restart;
  - a verdict at 3 of 5 registered populations with presence.

- **The probes are the pilot.** They are disclosed, they set the cap, and they show reachability. No further pilot runs.

- **Launch record 121** (Part A before the recovery; Part B after it, before the fits) and **Logbook 121**.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `mechanistic-baseline`: optional off-food run kinematics and a per-state speed spread.
- `controller-fit`: C.1 refitted on benchmark v3 with its 25-parameter model, a reduced recovery first, the probes as the pilot, and the extended mutant encodings.

## Impact

- **Code:**
  - `quantumnematode/evaluation/controllers.py` (`BatchedMechanistic`);
  - `scripts/analysis/c1_v3_fit.py` (new);
  - tests.
- **Records:** `docs/experiments/logbooks/supporting/121-c1-v3-fit/` (the probes as disclosure, the launch record, the recovery, the fit); Logbook 121; the tracker (C.1, C.4, the seed table); AGENTS.md; the CHANGELOG.
- **Cost** at about 33 s per generation on 16 workers:
  - the recovery: about 6–8 h (4 fits, plus its preflight's about 250 simulations);
  - the fits: about 18 h typical, 36 h at most;
  - grading: under 10 minutes.
