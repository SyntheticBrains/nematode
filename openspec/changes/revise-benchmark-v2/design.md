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

- **Chemotaxis beyond Chen's butanone assay** (other odours, salt, other labs): one deposit is fitted. The family's definitions are assay-agnostic, so a later source enters within it.
- **Local search's rise** (considered and dropped, the maintainer's decision of 2026-10-27). López-Cruz's recordings start about 4 min after removal, so the rise from the first recording bin to the 2–6 min bins falls 4–10 min after removal, while the worms recover from being picked and transferred. B.3 already starts the γ fit at the peak bin because the curve rises first, and Calhoun's authors discard their first 2 min. A simulated assay has no handling, so a faithful search worm is already decaying there, and no controller could reach the statistic.
- **Head pokes and edge contact:** they need a body, which opens in Phase 11 (D41).
- **Depletion-driven leaving:** there are about 3.5 exits per date.
- **Food-density dependence:** OD 1 has 45 animals on 1 date and OD 4 has 83 on 3. It is reported only.
- **The on-food omega rate:** omegas cannot be read from positions (real omega seconds are 42% missing). The reversal rate grades reorientation frequency, and the measurement model carries the omegas' missingness.
- **Uniform-well kinematics:** with 2 band dates, they are reported only.
- **Re-reading PD1074's held-out cohorts.**

## Decisions

01. **Versions.**

    - **Files:** v1 (`bands.json`) is frozen, and v2 is `bands_v2.json`.
    - **Grading:** `grading` carries both rules. `benchmark_grade.py --bands v1|v2` defaults to v2 once Part B is committed.
    - **Past verdicts:** B.4, F.0b-i and C.1 stay on v1 and are never re-read on v2 except as description.

02. **The measurement model** (`benchmark/measurement.py`, parameters in `bands_v2.json`):

    - **What it does:** for each simulated animal with seed s, `rng = default_rng([s, 2033])`:
      - every 1-s position gains independent N(0, σ²) noise per axis;
      - each second is then dropped (made NaN) with p_omega if the animal's primitive that second is an omega, and with p_other otherwise.
    - **What it applies to:** simulated sources only (`batched` and `captures`), before `window`. Real and imported tracks are never touched.
    - **Its parameters** are measured on band cohorts by the `probe` stage:
      - **σ** from paused seconds: the per-axis SD implied by consecutive paused 1-s steps, `mean step / √π`;
      - **p_omega and p_other** from Scheer's per-frame flags, by second.
    - **Missing runs:** before registration, the `probe` stage compares the real and modelled missing-run length distributions. If the real median run is longer than 1 s, or the real 90th percentile exceeds the Bernoulli model's by more than 2 s, missingness is drawn as runs with the real length distribution instead, and Part A says which.
    - **Its effect on v1's statistics:** noise and missing seconds feed the classifier, so roaming fraction, the bout means and the rest change for *simulated* controllers under v2. That is intended: real and simulated tracks are then cut the same way. v1's real-data validation still holds, because real tracks are untouched. The six negatives' v1 and v2 values are recorded side by side.

03. **The kinematics family** (small lawn; fitted, so in D_fit, for every controller):

    - **the four speed statistics:** median and 90th-percentile classifier-window speed (the classifier's own 10-s speed, from the 5-s steps) over on-lawn dwelling windows, and the same over roaming windows. They are pooled over the population's windows through a fixed log-spaced histogram per state, so the tally stays additive for the date bootstrap;
    - **the on-lawn reversal rate:** onsets from `readouts.reversal_onsets` with the off-food detector, applied to every on-lawn sample run identically on real and simulated tracks, per minute of valid on-lawn seconds. It is a statistic of the trajectory, not a count of the authors' labels (D36);
    - **roaming-bout survival at 5 min:** Kaplan–Meier over on-lawn roaming bouts, a bout at the window's edge censored. Bout lists are tallied per animal for the bootstrap.

04. **Local search is unchanged:** γ, early/late and the presence test, as in v1. The rise was considered and dropped (see the non-goals).

05. **D v2.**

    - **D:** five families (small lawn, uniform well, local search, kinematics, chemotaxis), equal weight, split equally within each.
    - **D_fit**, which C.2 and C.3 fit on: small lawn, uniform well and kinematics. Local search and chemotaxis are their held-out predictions. **C.1** fits D.
    - **The bar:** z\* is Bonferroni over the 20 wild-type statistics (11 from v1, 6 kinematics, 3 chemotaxis), Φ⁻¹(1 − 0.05/40) ≈ 3.02.
    - **The z cap** stays at 10.
    - **The pass rule:** a population passes when every family passes, local search only where its presence holds. This is v1's rule with the new family.

06. **The chemotaxis family** (decided 2026-10-27, after the first draft; settled on the deposit).

    - **The data.** The deposit is Chen et al. 2025's figshare article 24764403 (CC BY 4.0), downloaded by hand into the off-repo archive:

      - outer `24764403.zip`, SHA-256 `94b59ead…72da`;
      - inner `Chen_learning_2024.zip`, `6fee42b5…e363`;
      - `S1_Data.xlsx`, `e0894b03…b51d`.

      `data4plots/track_learn_all.mat` holds the tracks by condition and recording; condition 1 is naive, in `learn_folders4.mat`'s `folder_nai` order. **Its 8 naive N2 recordings, on 8 dates:**

      - **6 on the standard protocol** (`Data_learn/N2/*_11mM_35_400*`: 2022-08-30, 09-13, 09-22, 11-08, 2023-06-09, 06-13). These are the reference;
      - **2 on another flow rig** (`Data_odor_flow_equ`: 2022-08-04 "biased" and 2022-02-03 "6_32ml_400air"). These are reported only.

    - **The tracks:** 14 Hz `Path` in pixels, at 0.0301 mm per pixel (the median ratio of the authors' `SmoothSpeed` to pixel speed, IQR 0.0298–0.0302), so the imaged field is about 90 × 75 mm. Tracks are fragments, a median of 39 s, and are resampled to 1-s midbody positions.

    - **The field:** `data4plots/Landscape_low_0623_2.mat`'s `vq1`, 2500 × 3000 pixels (y, x), **stored rotated by 180°**. Sampled at `(W − 1 − x, H − 1 − y)`, it reproduces the concentration the authors record along their tracks (`Data_nai_test2.mat`'s `dc`): correlation 1.0000, median absolute difference 0.041. So its registration and units are the authors' own.

    - **The assay:** `chemotaxis_chen`, a 90 × 75 mm field, off food, 30 min, with the odour field `vq1` (bilinear, in mm, in the authors' units). The 1-s primitives worm senses it through its concentration, head-sweep and dC/dt senses. Worms are placed by the measured start distribution of fragments beginning in the first 30 s, near the field's centre.

    - **The arena's edges:**

      - the simulated arena is Chen's 100 × 100 mm plate, with the imaged 90 × 75 mm field placed centred on it (an assumption, stated);
      - outside the field, the concentration takes the nearest field value;
      - samples outside the imaged field are missing, as a real worm's track ends when it leaves the field. No statistic reads a wall.

    - **Population sizes differ:** a real chemotaxis population is one plate of fragments (50–100 worms), and a simulated one is 200 continuous worms. The band's prediction scale comes from plate-to-plate spread, so it is conservative for the simulated side. The record says so.

    - **The probe stage records Chen's facts** before Part A:

      - the landscape's 180° registration against the recorded concentrations;
      - the pixel scale;
      - the jitter;
      - the fragment-length distribution.

    - **The measurement model for this rig**, measured from Chen's tracks by the `probe` stage:

      - jitter, from the residual of `Path` against the authors' smoothed track (SD 0.0075 mm per frame);
      - fragmentation: simulated 30-min tracks are cut into fragments drawn from the real fragment-length distribution. This stands in for missingness.

    - **The statistics,** pooled per recording over valid seconds, and identical on real and simulated tracks:

      - **up-gradient drift:** the mean velocity component along ∇C/|∇C| at the worm's position, in mm/s;
      - **the quintile ratio:** reorientation onsets per minute (the off-food detector) in the lowest dC/dt quintile against the highest, with dC/dt taken from the field along the path. It is unit-free;
      - **the weathervane slope:** the regression slope, during runs (more than 2 s from a detected reorientation), of the heading's turning rate on the sine of the bearing to the gradient.

    - **The bands and their validation** (there is no held-out side, since 6 dates cannot spare one):

      - the band is the mean over the 6 recordings, with its scale the SD across recordings times the Student-t prediction factor for a new recording (df 5);
      - **leave-one-recording-out validation:** each recording is graded against bands built from the other five (t, df 4), every statistic at |z| ≤ z\*;
      - the allowance comes from the binomial over 6 units at the family's nominal rate, registered before anything is computed;
      - this is cross-validation, not a held-out read. The record says so, and states the limit: 6 dates, one lab, one odour.

    - **The off-protocol recordings** are reported beside the grade: their statistics, and their position against the bands.

    - **Presence** (added during implementation, before Part A). With 6 recordings and the bar at 20 statistics, the prediction factor is 2.0, so the bands are wide: drift's runs from about −0.02 to 0.07 mm/s. The constant-hazard worm, which ignores the odour, sat inside every band (drift z −1.6, log ratio −1.8, weathervane −0.2 at n = 64). The spec requires a worm that ignores the gradient to fail. So chemotaxis, like local search, passes only where it is present:

      - drift and the log quintile ratio each above 0 at the 2.5% quantile of 2,000 resamples (seed 2041) of the population's units;
      - the units are animals for a simulated population and fragments for a real recording, whose fragments cannot be joined into animals. A resample where a statistic is undefined counts as absent;
      - the weathervane slope takes no part, since 4 of the 6 standard recordings' own lower bounds on it fall below 0;
      - each standard recording must be present too, or its leave-one-out unit fails.

      Measured before it was registered: all 6 standard recordings are present (drift's lower bounds 0.013–0.028 mm/s, the log ratio's 0.65–1.49). Of the off-protocol two, 20220804 is present and 20220203 is not: its log ratio's bound sits at 0 (0.010 on a trial seed, exactly 0 on the registered one, a resample with ratio 1), as its point value, 0.13, already set it apart. It is reported, not graded. The constant-hazard worm is absent at n = 64 and 200 (bounds −0.004 and −0.001 mm/s; −0.11 and −0.07). The pirouette worm, which turns on dC/dt, is present (0.015 and 0.017 mm/s; 0.73 and 0.84).

    - **The negatives and adversaries** run the chemotaxis assay too. B.4's four open-loop negatives (`hmm`, `poisson`, `random_walk`, `correlated_walk`) ignore the odour, and each must fail the chemotaxis family: the spec's "a worm that ignores the gradient", tested. The blind worm, with no sensing, should fail it too; that is reported, not gated, since it fails the wild type through other families.

    - **The definitions are assay-agnostic** (drift along the local gradient, a dC/dt quintile ratio, a bearing regression), so a later source could enter as further bands within the family at the same weight. The literature watch's brief gains "public *C. elegans* chemotaxis track datasets".

07. **Bands v2.** The v1 builder is extended to histograms, bout lists and the reversal tally (the chemotaxis family's bands are decision 06's):

    - from band cohorts only, by the date bootstrap at n;
    - each band on its own generator, `default_rng([2030, index, n])`, with the new statistics' indices appended after v1's, so v1's values are reproduced;
    - **κ calibration and the rule check** as in v1 (a split of band dates): the check's failure rate per family is the input to the oracle and the allowance.

08. **The oracles** (registered; run before Part B).

    **First, the measurement-model round trip,** for both rigs. Real band-side tracks are denoised: Chen's own `SmoothX/Y`, and Scheer's 1-s tracks smoothed by a centred 5-s moving average (registered). They are then passed through the model, and every new statistic is recomputed. For each statistic and rig, the round-tripped population value must lie within |z| ≤ 1 of the raw tracks' value, on that statistic's band scale. If it fails, the model is wrong, and the run stops.

    **Amended 2026-10-27, after the round trip failed** (launch record, amendment 1). The 5-s mean deletes real reversals and slow dwelling motion (alone: reversals z −23.8), so the Scheer round trip failed on the denoiser, not the model. Its place is taken by two checks needing no smoothed truth:

    - **the losses** on gap-filled band-side tracks (σ = 0), every kinematics statistic within |z| ≤ 1 at n = 1,013;
    - **the jitter** on the authors' paused stretches (≥ 10 s), whose readings are the tracker's noise, against each stretch's mean position held still with the model's jitter. Onsets per paused minute and the 10-s window speed's median and 90th percentile must each differ by at most one n = 200 band scale of the statistic they bias (the reversal rate and the two dwelling speeds).

    Chen's round trip is unchanged.

    **Then the reachability oracle** (scratch seeds):

    - **the populations:** B.4's matched open-loop worm (`hazard_controller("hmm")`, at the real band-cohort speeds and hazards) passes through the measurement model, at 40 populations of 200;
    - **the gate:** for each of the four speed statistics and the reversal rate, the mean z lies in [−1, 1] and |z| ≤ z\* in at least 90% of populations;
    - **excluded:** bout survival, which that worm is not matched for (its bouts are exponential). Its oracle is the N2 control, and the chemotaxis statistics' is their leave-one-out validation;
    - **on failure:** if the round trip has passed, the measurement model is right, so the failure belongs to the matched worm or to the statistic. The record says which, and the run stops before Part B and returns to the maintainer.

09. **The positive control: fresh N2** (`benchmark_v2_controls.py read`, guarded):

    - **the bands:** N2 bands for the small lawn and kinematics families, from N2's 42 band dates, at the size of its unread held-out set;
    - **the units:** the 26 unread held-out dates split into two halves by date (`default_rng(2035)`, registered), times the two families: 4 units;
    - **the gate:** at most **1** of the 4 units fails its family. From the rule check's per-family rates, P(more than 1 fails) is about 0.02, computed and stated in Part B. A systematic misfit of a family fails both its halves, so it is still caught;
    - **the read:** once, under the guard. PD1074's held-out cohorts are not read.

    The N2 control validates band construction and the new statistics' calibration on unseen data from the same lab and assay. It does not validate PD1074's bands, which B.4 did.

10. **The negatives and adversaries** (simulated, through the measurement model).

    - **B.4's six negatives,** on B.4's animal seeds, each fail the wild type on v2, and the four open-loop ones each fail the chemotaxis family.
    - **The blind worm:**
      - C.1 with every sensing term pinned off: κ_w, β_r, β_o and edge 0; intake and satiety slopes 0; search amplitude 0;
      - fitted by sep-CMA against v2's D, from the box centre, population 16, with 2 restarts;
      - plateau rule with the σ ≤ σ₀ guard (C.1's stuck restarts stopped at σ 0.27–0.36; every converged fit at ≤ 0.142), cap 210;
      - the best restart by its own objective must fail the wild type at 3 of 5.
    - **C.1's restart-3 answer** (v1's chosen fit), graded on five populations under v2, must fail the wild type at 3 of 5.

11. **Validity v2:** **VALID** when all of the following hold:

    - the measurement-model checks pass (as amended: the losses and the jitter on Scheer's rig, the round trip on Chen's);
    - the reachability oracle passes;
    - the N2 control passes within its allowance (at most 1 of 4 units fails);
    - the chemotaxis leave-one-recording-out validation passes within its registered allowance, a recording failing its unit where it is not present;
    - all six negatives fail, and the four open-loop ones fail the chemotaxis family;
    - both adversaries fail.

    Otherwise **VOID**, and the statistics are revised in a new change. Every reading is recorded either way.

12. **W's minimum v2, on both family sets.** W reads on D40's held-out fidelity distance, but which families that is for C.2 belongs to W's registration. So v2 records both:

    - **the fitted families** (small lawn and kinematics): between halves of N2's 26 unread held-out dates, split by date 1,000 times (`default_rng(2034)`), on PD1074's v2 band scales at each half's size;
    - **the held-out families** (local search and chemotaxis):
      - local search between halves of López-Cruz's 17 *band* batches, 1,000 splits (`default_rng(2036)`); its held-out batches were read in B.4;
      - chemotaxis between every 3-against-3 split of Chen's 6 recordings, all 10 of them;
    - each set's minimum is the 95th percentile.

    W's registration must read its arms on the same family set as the minimum it cites. That closes v1's mismatch between W's minimum (small lawn plus local search) and D_fit.

13. **The launch record, in two parts.**

    - **Part A**, committed before the oracle, the N2 read, the negatives and the adversary fits, holds decisions 2–12 and every threshold and seed.
    - **Part B** adds the probe's measured parameters, the bands, the rule check and the oracle's result, before the N2 read and the adversary fits. It changes no rule.

    `read` refuses unless Part B is committed and `bands_v2.json` is byte-identical to its version in that commit.

14. **Seeds:**

    - oracle populations at `150_000_000 + 200·p + i` (p < 40);
    - blind-worm fits on CMA seeds 9401–9402, with animals `151_000_000 + 200·k + i` (k = 10,000·r + 16·(n − 1) + m), and its grading at `152_000_000 + 200·g + i`;
    - C.1's answer graded at `153_000_000 + 200·g + i`;
    - the negatives on B.4's seeds (5101–5300; networks 5301–5316);
    - the measurement model's streams, `default_rng([seed, 2033])` on the lawn assays and `default_rng([seed, 2037])` on the chemotaxis assay;
    - chemotaxis-assay populations: every simulated population on `chemotaxis_chen` uses the same animal seeds as its lawn populations, so a controller's families are read on the same animals;
    - all recorded in the seed table.

15. **Track import and documentation.**

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
- **[A faithful controller fails more often]** with 20 statistics and five families. The bar is Bonferroni, and the 3-of-5 oracle is recomputed for v2 from the rule check's rates. The C.1b and C.2 registrations carry the v2 figure.
- **[The adversaries pass]** → v2 is void, and the loophole is not closed. That is the purpose.
- **[Cost]** → about 5–7 h of compute for this change. The F.0b-i re-run (11–13 h) and C.1b (8–9 h) follow in their own changes.
