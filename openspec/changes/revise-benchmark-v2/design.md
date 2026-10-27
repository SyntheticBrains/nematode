## Context

- **B v1** (Logbooks 107–110) grades 11 wild-type statistics in three equal families:

  - **small lawn** (6): roaming fraction, roaming and dwelling bout means, exits per on-lawn hour, pre-exit enrichment, time on lawn;
  - **uniform well** (3): roaming fraction and both bout means;
  - **López-Cruz local search** (2): γ and early/late.

  D is the weighted RMS of z capped at 10, and D_fit is small lawn plus uniform well. Bands come from band cohorts by a date bootstrap at n, with κ calibration. B.4 read the held-out cohorts once and found v1 VALID. W's minimum v1 is 1.42.

- **The investigation's planning probes**, uncommitted and re-read by this change's `probe` stage:

  - **at the classifier's 10-s scale:**

    | Statistic | Real (41 dates, SD) | C.1 | z |
    |---|---|---|---|
    | roaming window speed, median | 0.037 (0.002) mm/s | 0.098 | +31 |
    | dwelling window speed, median | 0.013 (0.0015) | 0.012 | −0.5 |
    | dwelling window speed, 90th percentile | 0.028 | 0.046 | +7 |

  - **real dwelling's angular speed** is 18.5 ± 0.1 °/s on every date, the value a heading randomised every 5-s step gives. It is tracking noise, so it is not graded;

  - **detected on-lawn reversals** (the off-food detector on lawn samples): real 4.06 ± 0.30 per min; B.4's matched open-loop worm at the real hazard, 3.20 (z −2.9); C.1, 0.15 (z −13);

  - **the tracker** (band cohorts):

    - paused 1-s steps have a median of 0.0064 mm, which implies a per-axis jitter SD of about 0.0047 mm (an upper bound: paused worms may twitch);
    - P(missing | omega second) = 0.42, P(missing | reversal or pause second) = 0, and P(missing | other) = 0.017;

  - **the blind worm** (C.1 with every sensing term pinned off, sep-CMA against v1's D) plateaued at generation 83. It fails all five populations (D 3.9–5.7), on exits, time on lawn and local search.

- **Unread data:**

  - N2 at OD 2 has 42 band dates and 28 held-out dates (537 animals). B.4 read only the 2 paired with tph-1, so **26 are unread**;
  - López-Cruz's 10 held-out batches were all read in B.4.

- **Coverage audit:** the subagent report of 2026-10-27, summarised in the logbook.

## Goals / Non-Goals

**Goals**

- Close the confirmed loopholes, with statistics a faithful point worm can reach.
- Measure real and simulated tracks the same way.
- Make resistance to the known exploit a condition of validity.
- Declare what B does not grade, with reasons, so a later gap is a stated limit rather than a reopening.

**Non-goals, declared, with reasons:**

- **Chemotaxis** (the quintile ratio, the weathervane regression): no real gradient tracks are vendored.
- **Head pokes and edge contact:** they need a body, which opens in Phase 11 (D41).
- **Depletion-driven leaving:** there are about 3.5 exits per date.
- **Food-density dependence:** OD 1 has 45 animals on 1 date and OD 4 has 83 on 3. It is reported only.
- **The on-food omega rate:** omegas cannot be read from positions (real omega seconds are 42% missing). The reversal rate grades reorientation frequency, and the measurement model carries the omegas' missingness.
- **Uniform-well kinematics:** with 2 band dates, they are reported only.
- **Re-reading PD1074's held-out cohorts.**

## Decisions

1. **Versions.**

   - **Files:** v1 (`bands.json`) is frozen, and v2 is `bands_v2.json`.
   - **Grading:** `grading` carries both rules. `benchmark_grade.py --bands v1|v2` defaults to v2 once Part B is committed.
   - **Past verdicts:** B.4, F.0b-i and C.1 stay on v1 and are never re-read on v2 except as description.

2. **The measurement model** (`benchmark/measurement.py`, parameters in `bands_v2.json`):

   - **What it does:** for each simulated animal with seed s, `rng = default_rng([s, 2033])`:
     - every 1-s position gains independent N(0, σ²) noise per axis;
     - each second is then dropped (made NaN) with p_omega if the animal's primitive that second is an omega, and with p_other otherwise.
   - **What it applies to:** simulated sources only (`batched` and `captures`), before `window`. Real and imported tracks are never touched.
   - **Its parameters** are measured on band cohorts by the `probe` stage:
     - **σ** from paused seconds: the per-axis SD implied by consecutive paused 1-s steps, `mean step / √π`;
     - **p_omega and p_other** from Scheer's per-frame flags, by second.
   - **Missing runs:** before registration, the `probe` stage compares the lengths of real and modelled missing runs. If real runs are longer than Bernoulli gives (median run over 1 s), missingness is drawn as runs with the real length distribution instead, and Part A says which.

3. **The kinematics family** (small lawn; fitted, so in D_fit, for every controller):

   - **the four speed statistics:** median and 90th-percentile classifier-window speed (the classifier's own 10-s speed, from the 5-s steps) over on-lawn dwelling windows, and the same over roaming windows. They are pooled over the population's windows through a fixed log-spaced histogram per state, so the tally stays additive for the date bootstrap;
   - **the on-lawn reversal rate:** onsets from `readouts.reversal_onsets` with the off-food detector, applied to every on-lawn sample run identically on real and simulated tracks, per minute of valid on-lawn seconds. It is a statistic of the trajectory, not a count of the authors' labels (D36);
   - **roaming-bout survival at 5 min:** Kaplan–Meier over on-lawn roaming bouts, a bout at the window's edge censored. Bout lists are tallied per animal for the bootstrap.

4. **Local search's rise** (in the local-search family; held out for C.2 and C.3, fitted for C.1 under D37): the mean reorientation rate over the 2–4 and 4–6 min bins, divided by the 0–2 min bin's.

   - **Real:** from López-Cruz's band batches.
   - **Simulated:** from the same 2-min bins of `simulated_search`'s curve.
   - **The limit:** no unread López-Cruz batch exists, so the rise has only the band batches' rule check, no fresh positive control. The record says so.

5. **D v2.**

   - **D:** five families (small lawn, uniform well, local search, kinematics, chemotaxis), equal weight, split equally within each.
   - **D_fit**, which C.2 and C.3 fit on: small lawn, uniform well and kinematics. Local search and chemotaxis are their held-out predictions. **C.1** fits D.
   - **The bar:** z\* is Bonferroni over every wild-type statistic, 18 plus chemotaxis's.
   - **The z cap** stays at 10.
   - **The pass rule:** a population passes when every family passes, local search only where its presence holds. This is v1's rule with the new family.

5b. **The chemotaxis family** (decided 2026-10-27, after the first draft).
\- **The data:** Chen et al. 2025, naive N2, butanone in linear gradients on 10 cm square plates, 30 min at 14 Hz, 50–100 worms per plate, 9–13 plates per condition (CC BY 4.0). Derived 1-s midbody tracks are vendored with a `PROVENANCE.md` entry.
\- **The assay:** `chemotaxis_chen`, a linear odour field on a square plate, off food, on the 1-s primitives worm. The worm senses it through the existing concentration, head-sweep and dC/dt senses. The plate geometry, gradient profile and placement match Chen's methods.
\- **The statistics,** from positions and the known gradient, computed identically on real and simulated tracks, and finalised on the data before Part A:
\- up-gradient drift, or the chemotaxis index;
\- the quintile ratio, the reorientation rate in the lowest against the highest dC/dt quintile, by the off-food detector. It is unit-free, so robust to the gradient's absolute scale;
\- the weathervane regression, the slope of heading change against the perpendicular gradient.
\- **The bands:** by the date (or plate) bootstrap, from a band side. **The held-out split** (a third of dates or plates, by a seeded rule) is fixed and committed before any statistic is read on the data.
\- **The measurement model** has chemotaxis-assay parameters (jitter and missingness) measured from Chen's own tracks, since it is another rig at another frame rate.
\- **The positive control:** the held-out plates, read once.
\- **The negatives and adversaries:** run on the chemotaxis assay too. The blind worm must fail it by construction.
\- **The fallback, decided before Part A:** if the gradient profile is not recoverable from the deposit, only scale-free statistics enter (bearing-based drift and the quintile ratio).

06. **Bands v2.** The v1 builder is extended to histograms, bout lists, the reversal tally and the rise:

    - from band cohorts only, by the date bootstrap at n;
    - each band on its own generator, `default_rng([2030, index, n])`, with the new statistics' indices appended after v1's, so v1's values are reproduced;
    - **κ calibration and the rule check** as in v1 (a split of band dates): the check's failure rate per family is the input to the oracle and the allowance.

07. **The reachability oracle** (registered; run on scratch seeds before Part B):

    - **the populations:** B.4's matched open-loop worm (`hazard_controller("hmm")`, at the real band-cohort speeds and hazards) passes through the measurement model, at 40 populations of 200;
    - **the gate:** for each of the four speed statistics and the reversal rate, the mean z lies in [−1, 1] and |z| ≤ z\* in at least 90% of populations;
    - **excluded:** bout survival and the rise, which that worm is not matched for (its bouts are exponential, and it has no search). Their oracle is the N2 control;
    - **on failure:** the measurement model or the statistic is wrong. The run stops before Part B and returns to the maintainer.

08. **The positive control: fresh N2** (`benchmark_v2_controls.py read`, guarded):

    - **the bands:** N2 bands for the small lawn and kinematics families, from N2's 42 band dates, at the size of its unread held-out set;
    - **the units:** pooled small lawn and pooled kinematics over the 26 unread held-out dates;
    - **the gate:** each unit passes its family;
    - **the allowance:** with two units, at most 0 may fail. Each unit's failure rate is the rule check's, and the joint P(spurious void) is stated;
    - **the read:** once, under the guard. PD1074's held-out cohorts are not read.

    The N2 control validates band construction and the new statistics' calibration on unseen data from the same lab and assay. It does not validate PD1074's bands, which B.4 did.

09. **The negatives and adversaries** (simulated, through the measurement model).

    - **B.4's six negatives,** on B.4's animal seeds, each fail the wild type on v2.
    - **The blind worm:**
      - C.1 with every sensing term pinned off: κ_w, β_r, β_o and edge 0; intake and satiety slopes 0; search amplitude 0;
      - fitted by sep-CMA against v2's D, from the box centre, population 16, with 2 restarts;
      - plateau rule with the σ ≤ σ₀ guard (C.1's stuck restarts stopped at σ 0.27–0.36; every converged fit at ≤ 0.142), cap 210;
      - the best restart by its own objective must fail the wild type at 3 of 5.
    - **C.1's restart-3 answer** (v1's chosen fit), graded on five populations under v2, must fail the wild type at 3 of 5.

10. **Validity v2:** **VALID** when all of the following hold:

    - the oracle passes;
    - both N2 units pass;
    - all six negatives fail;
    - both adversaries fail.

    Otherwise **VOID**, and the statistics are revised in a new change. Every reading is recorded either way.

11. **W's minimum v2:**

    - the N2 held-out dates (26) are split into halves by date, 1,000 times (`default_rng(2034)`);
    - D over the small-lawn and kinematics families between the halves, on PD1074's v2 band scales at each half's size;
    - the 95th percentile.

    W's registration reads its arms on the same two families, closing v1's mismatch between W's minimum (small lawn plus local search) and D_fit.

12. **The launch record, in two parts.**

    - **Part A**, committed before the oracle, the N2 read, the negatives and the adversary fits, holds decisions 2–11 and every threshold and seed.
    - **Part B** adds the probe's measured parameters, the bands, the rule check and the oracle's result, before the N2 read and the adversary fits. It changes no rule.

    `read` refuses unless Part B is committed and `bands_v2.json` is byte-identical to its version in that commit.

13. **Seeds:**

    - oracle populations at `150_000_000 + 200·p + i` (p < 40);
    - blind-worm fits on CMA seeds 9401–9402, with animals `151_000_000 + 200·k + i` (k = 10,000·r + 16·(n − 1) + m), and its grading at `152_000_000 + 200·g + i`;
    - C.1's answer graded at `153_000_000 + 200·g + i`;
    - the negatives on B.4's seeds (5101–5300; networks 5301–5316);
    - the measurement model's streams, `default_rng([seed, 2033])`;
    - all recorded in the seed table.

14. **Track import and documentation.**

    - The round trip covers the v2 statistics on real tracks, with no measurement model.
    - `IMPORT.md` lists v2 and the declared non-goals.
    - The tracker adds:
      - the non-goals;
      - W.1 and W.2's reconstruction and weight-noise checks (Yad & Nabila 2026; the wild type on Witvliet 2021's adult dataset 8 beside Cook 2019; a 1% weight-noise check on the fitted C.2);
      - H.1, closed;
      - the F.0b-i re-run and C.1b as the next two changes.

## Risks / Trade-offs

- **[The oracle fails]** → the measurement model under-represents the real tracker. It stops before Part B, by design.
- **[N2 differs from PD1074]** → the N2 control tests the construction on N2's own bands, not PD1074's. That is stated.
- **[A faithful controller fails more often]** with 18 statistics and four families. The bar is Bonferroni, and the 3-of-5 oracle is recomputed for v2 from the rule check's rates. The C.1b and C.2 registrations carry the v2 figure.
- **[The rise has no fresh positive control]** → stated, and it sits in the held-out family for C.2 and C.3.
- **[The adversaries pass]** → v2 is void, and the loophole is not closed. That is the purpose.
- **[Cost]** → about 5–7 h of compute for this change. The F.0b-i re-run (11–13 h) and C.1b (8–9 h) follow in their own changes.
