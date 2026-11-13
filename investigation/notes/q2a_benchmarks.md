# Q2, first half: what each benchmark measures, its pass rule, its controls, and its revisions

Scope: the older behaviour validations (Logbooks 035, 036, 086, 088, 089, 090, 093) and Phase 10's
benchmark versions v1 to v4 (Logbooks 109, 110, 115, 120, 124). It covers each version's
statistics, pass rule and controls, the negatives' failures, and every revision that came after a
controller's result. C.1 and C.2 are scored here only where a score triggered a revision. Their
full scores are out of scope.

Paths are relative to the worktree
`/Users/chriszaharia/projects/personal/nematode/.claude/worktrees/fit-first-investigation`. A
`NNN:a-b` reference means lines a to b of `docs/experiments/logbooks/NNN-*.md`.

______________________________________________________________________

## Part (i): the older validations (Phases 6 to 8)

All of these were graded on **PPO-trained** (reward-trained) controllers, not fitted ones. Their
references are **literature-coded signatures**, mostly sign-only, not real tracks. The exception is
088, which built its reference from Scheer's windows.

| Logbook | What it measures | Real-worm reference (file) | Controllers | Verdict and numbers |
|---|---|---|---|---|
| **035** chemotaxis (point worm, 5-s step) | Klinokinesis: the down/up-gradient turn-rate ratio (thresholded) and the turn-magnitude ratio (threshold-free). Weathervane: the slope of curving rate against bearing, thresholded and over all steps (035:83-87) | `data/chemotaxis/behavioural_bias_signatures.json`. Pierce-Shimomura 1999's turn-rate ratio has a magnitude range of 1.5–3.0; the other three are **sign-only** (the file's `magnitude_range: null`). Iino & Yoshida 2009 for the weathervane | MLP-PPO (the gating arm), connectome-PPO and a derivative-sensing MLP control; n = 8 seeds, 300 episodes, the last 100 graded (035:114-123) | **Both strategies "reproduced"** (035:3-11). Turn-rate ratio 1.30 (MLP) and 1.15 (connectome), below the 1.5–3.0 range. Only the sharpest reorientations reach 2.78, inside it (035:134-147). Weathervane +0.027/+0.091 (MLP) and sign-only, because the units are not comparable (035:149-154). The derivative control collapses the weathervane by 87–93% (035:156-173). Wall check 2026-10-04: the verdicts hold, and the control's residual falls to PARTIAL (035:13-37) |
| **036** thermotaxis (point worm) | The same four statistics on the setpoint drive −|T−Tc| (036:37-46) | `data/thermotaxis/behavioural_bias_signatures.json`, **sign-only** for all four (Luo 2014 primary; 036:44-46) | MLP-PPO only, n = **4** seeds, klinotaxis and derivative arms; one derivative seed failed to converge (036:55-64) | **PARTIAL**: weathervane weakly present (+0.020 [+0.010, +0.031] at 95%), **klinokinesis absent** (ratio 1.015, 0.959; 036:70-93). Attributed to a state-independent action std and a "migrate and park" policy (036:100-111) |
| **086** chemotaxis through the kinematic body (C.3) | Frequency, wavelength, the variance in four eigenworms, speed, forward bouts, omega postures, and 035's bias curves (086:62-72) | Fang-Yen 2010 ranges; speed band 0.12–0.30 body lengths/s, after Ramot 2008 via Wormlight (`supporting/086-body-validation/launch.md:51`); eigenworms and real postures from `data/posture/EigenWorms.csv` and `shapes.csv` (WormPose, Stephens 2008) | Wild-type connectome, chemical-only null and MLP-PPO, seeds 1801–1864, plus a derivative control on 16 seeds (086:47-60) | The body checks **pass by construction**: 0.300 Hz and 0.650 body lengths are the generator's own settings, and the eigenworm clause is easier for a sine (98% against the real 96.5%; 086:87-90, 244-246). Speed is **partial** on every arm (0.113–0.116; 086:94-96). **0 of about 807,000 turns have an omega's posture** (086:24-26, 193-206). Klinokinesis turn-rate ratio is 1.03–1.24 against 1.5–3.0 (086:28-29). The weathervane is learned in every arm (086:133) |
| **088** patch leaving (D.1b) | Whether exits come out of roaming: the roaming share 30–180 s before an exit minus the overall share (088:36-47) | `data/roaming_dwelling/scheer2023_leaving.json`: +0.288 enrichment with the project's classifier, +0.227 with the authors' labels; 212 exits from 184 animals (088:29-47) | MLP-PPO and its untrained floor, 16 seeds (088:57-61) | **Unreadable**: only 7 of 16 seeds take both states, against 12 required (088:3-16). Exits run 7.7 per on-lawn hour against the real 0.21 (088:83-89). Learners leave without roaming (088:101-106) |
| **089** connectome on lawns (B.3 stage 0) | The on-lawn dwelling share, learner minus untrained floor | **None graded against real data.** "Shares are compared across learners read by the same instrument, never against the authors' fractions" (089:141-143) | Connectome-PPO (Emmons 2024, depth 4, no modulators), 19 seeds | **"dwells"**: +0.514 [+0.398, +0.626] (089:3-13). Only 4 of 19 seeds complete bouts of both states (089:98-100) |
| **090** serotonin and PDF knockouts (B.3) | Mean on-lawn speed under acute *tph-1* and *pdfr-1*, oriented to the mutant's direction; the minimum is 0.009 mm/s (090:84-92) | The directions are Ji 2021 and Flavell 2013 (`data/roaming_dwelling/ji2021_readings.json`). The circuit is hand-placed from `data/neuromodulators/` (090:55-64) | Connectome-PPO with the modulator field and atlas-grounded signs, 32 seeds, 5 arms (090:94-100) | ***tph-1*: no effect at the minimum** (−0.0015 [−0.0078, +0.0049] mm/s). ***pdfr-1*: unresolved** (+0.0043; 090:11-14). Networks raised without a modulator move *against* the mutants (090:161-162, 226-229). PDF's inhibition of NSM is weakened on 29 of 32 seeds (090:191, 211-219) |
| **093** thermotaxis through the body | 036's four statistics on M.9's trained bodies (093:57-78) | `data/thermotaxis/behavioural_bias_signatures.json`, sign-only (093:67-69) | Wild-type connectome and gap-only null, trained and untrained floors, seeds 2301–2332 | Weathervane **PRESENT** (+0.0119 wild type, +0.0067 null); **klinokinesis ABSENT** (ratio 0.935–0.960; 093:5-10). There is no specificity control, and the cell is not 036's assay (093:80-87, 145-150) |

**What the older validations show together.** Across every substrate, reward-trained learners
reproduce the **direction** of the weathervane. None reproduces a worm-sized klinokinesis. On the
lawns they do not roam before leaving, and their knockouts do not phenocopy (roadmap
`docs/roadmap.md:1651-1656` cites 088 and 090 as the reasons for Phase 10).

**Weaknesses (candid):**

- The references are mostly sign-only. The single magnitude reference (turn-rate ratio 1.5–3.0) is
  failed at the broad threshold everywhere except 035's sharpest-turn subset (035:144-147;
  086:112-117).

- Several measurement rules were added after a controller's data were seen, each disclosed:

  - 035's curving-rate floor came after the connectome companion produced a "spurious ABSENT" from
    one seed's raw slope of −33 (035:97-103);
  - 036's floor fallback came after a parked worm read +12.8 (036:128-131);
  - 086's paired floor rule was "adopted after the cost pilot showed exactly this lean" (086:148-149);
  - 090's reading moved from the classified dwelling share to speed, and its substrate to grounded
    signs, after a random-sign pilot showed knockouts in the wrong direction (090:106-114).

  Each change was registered before its scored panel.

- Sample sizes are small in places: 036 has n = 4, and 088 had 7 readable seeds.

______________________________________________________________________

## Part (ii): the Phase 10 benchmark, v1 to v4

### Data under every version

The data are in `data/benchmark/`; provenance is in `data/benchmark/PROVENANCE.md`.

- **Scheer & Bargmann 2023** (CC0): 2,847 animals of 40 min at 3 frames/s (PROVENANCE.md:30-41).
  - Wild type PD1074 at OD 2: 1,586 animals on 65 dates, of which 41 are band dates (1,013 animals)
    and 24 are held out (573 animals).
  - Uniform lawn: **100 animals on 3 dates**, 2 band dates and **1 held out (31 animals)**.
  - tph-1: **139 animals on 5 dates**, 3 band and 2 held out.
  - pdfr-1: 395 animals on 16 dates. tax-4: 499 animals on 19 dates.
  - The split is by recording date (PROVENANCE.md:87-93).
  - **"Held out means held out of the bands, not unseen by the instrument."** The roaming/dwelling
    classifier was calibrated on an animal-level split across every wild-type date
    (PROVENANCE.md:90-93).
- **López-Cruz 2019** (CC BY): reorientation counts per 2-min bin from 27 wild-type batches, 17 in
  the bands and 10 held out (109:97). These are **counts, not tracks**, from a rig the measurement
  model does not cover.
- **Chen 2025** (CC BY), from v2: naive N2 butanone tracks, **6 standard recordings** plus 2 off
  protocol. They are fragments, so they cannot be joined into animals (PROVENANCE.md:194-244).
- **Roberts 2016** and **Calhoun 2014** (CC0): off-food tracks, used to tune the detector and for the
  reported reversal share (17.34%).

### The common machinery

- 1-s positions are read at 5 s, cut into 10-s windows and classified by the project's Gaussian HMM
  (`grading.py:1-22`; classifier κ = 0.625 on held-back band dates, 109:84-91).
- Each band is a cluster bootstrap over band dates at the population's size. Its scale is inflated by
  the centre's uncertainty and a calibration factor κ (109:104-114).
- **z** is (s − centre)/scale, capped at ±10. **An undefined statistic sits at the cap and fails**
  (`grading.py:56`, `803-808`, `811-831`).
- **The bar** z\* is two-sided Bonferroni at a 5% family-wise rate (`grading.py:121-123`).
- **D** is the weighted RMS of the capped z's, with each family weighted equally and split equally
  among its statistics (`grading.py:834-858`).
- **Mutants** are graded on their difference from their own wild type (tph-1 against N2; pdfr-1 and
  tax-4 against PD1074) on Δ roaming fraction and Δ exits per on-lawn hour. A resolved direction
  must also match (`grade(..., directions=True)`, `grading.py:811-831`).
  - Mutants are graded in the scripts (`scripts/analysis/benchmark_controls.py:302`,
    `scripts/analysis/c2_encoder_probe.py:416`), not inside
    `grade_populations_v2/v3/v4` (`scripts/analysis/benchmark_grade.py:605-688`).
  - So mutants never enter D or the wild-type pass. Their bar is 2.2414 in every bands file. Only
    tph-1's roaming (more) and tax-4's roaming (less) have resolved directions (bands_v4.json
    `mutants`; 109:227-238).

### v1: Logbooks 109 (the rule) and 110 (the controls)

- **Statistics: 11 wild-type statistics in 3 families** (`grading.py:75-95`;
  `data/benchmark/bands.json`):

  - small lawn (6): roaming fraction, roaming bout mean, dwelling bout mean, exits per on-lawn hour,
    pre-exit roaming enrichment, time on the lawn;
  - uniform well (3): roaming fraction and the two bout means;
  - López-Cruz local search (2): γ and the linear early/late ratio, both level-free.

- **Pass rule:** |z| ≤ **2.84** (Bonferroni over 11). Mutants are their own families at 2.24
  (109:65-70, 253-256).

  - **Local search must be present:** early/late above 1 at its one-sided 97.5% lower bound
    (`grading.py:114-118`, `625-644`; 109:259-265).
  - Presence does not enter D.

- **D_fit** leaves local search out (`FIT_FAMILIES`, `grading.py:95`).

- **A controller's verdict** passes when at least 3 of 5 registered 200-animal populations pass
  every family. This is C.1's registered rule (`supporting/113-c1-fit/launch.md:31-38`), carried to
  later fits. For a perfect controller it holds with probability 0.971.

- **The rule's own check failed first.** The small lawn passed 0.80 against a 0.90 gate, and local
  search 0.86. A retry registered before it ran (adding the centre's term, κ factors of 1.19–1.30,
  and a Student-t interval for López-Cruz) passed at 0.925 and 0.962 (109:95-114).

- **Hardening addendum** (109:223-269), all before any controller was graded:

  - directions resolved by a paired t-test over dates;
  - bands built per family at any size;
  - γ is undefined without a decay;
  - pdfr-1's local search reported, not graded;
  - the local-search presence test, added after **a population with no local search passed the
    family 12–13% of the time**.

- **Preflight** (n = 64; 109:143-164):

  - ConstantHazards: D 6.63;
  - PirouetteHazards: D 7.22;
  - both fail all three families, on roaming fraction, exits, time on the lawn, the uniform roaming
    fraction and γ;
  - **four statistics pass for a Poisson worm:** the dwelling bout mean, pre-exit enrichment, the
    uniform roaming bout mean and early/late.

- **Controls (110), VALID** (110:5-10, 67-76):

  - 14 of 15 positive units pass, against an allowance of 2:
    - the small lawn on 24 held-out dates (573 animals), with z from −1.29 to +1.29;
    - the uniform well (31 animals);
    - all 10 López-Cruz held-out batches;
    - pdfr-1 and tax-4.
  - **tph-1 fails on leaving**: Δ exits +4.04/h, z 6.37. Its three band dates showed no leaving
    effect, and its two held-out dates showed a 4× one. **The tph-1 leaving reference is
    unreliable** (110:80).

- **Negatives** (n = 200; 110:84-103; per statistic from
  `supporting/110-benchmark-controls/controls.json`). **Every negative fails exits and
  time-on-lawn.** The other failures, per negative:

  | Negative | D | Also fails | Passes |
  |---|---|---|---|
  | `BatchedStateHazards` open-loop HMM | 5.70 | small-lawn roaming bout, enrichment, uniform roaming bout, γ | roaming fraction and dwelling bout on both lawns; early/late |
  | Poisson primitives | 7.68 | roaming fraction, dwelling bout, enrichment, uniform bouts, γ |: |
  | random walk | 8.02 | almost everything |: |
  | correlated walk (`BatchedCurvatureWalk` AR(1)) | 6.96 | the lawn and uniform statistics | γ and early/late (local search) |
  | untrained MLP | 5.78 | roaming fraction and bout, γ | **the uniform well** |
  | untrained connectome | 4.49 | roaming bout, enrichment, uniform roaming bout | **local search** (γ z 2.4, early/late −1.9; presence bound 1.006) |

  "Staying on the small lawn decides every negative," and **"[local search's] family should not be
  relied on alone"** (110:97-103).

- **W's minimum** (D between two real halves, 95th percentile): **1.42** (110:105-107).

### v2: Logbook 115

- **Statistics: 19 in 5 equal families, z\* 3.0078** (`data/benchmark/bands_v2.json`; 115:63-71):
  - v1's three families;
  - **kinematics** (5): dwelling and roaming 10-s window speed medians and 90th percentiles, and
    roaming-bout survival at 5 min (`grading.py:258-274`);
  - **chemotaxis** on Chen's tracks (3): drift, the log quintile ratio and the weathervane slope
    (`chemotaxis.py:41`).
- **Reported, not graded:** the on-lawn reversal rate (`KINEMATIC_REPORTED`, `grading.py:274`).
- **D_fit** is the small lawn, the uniform well and kinematics. Local search and chemotaxis are held
  out for C.2 and C.3, and **chemotaxis is fitted for C.1** (115:56).
- **Chemotaxis must be present:** drift and the log ratio above 0 at the 2.5% quantile of 2,000
  resamples (`chemotaxis.py:55-58`, `221-244`; `benchmark_grade.py:800-802`).
- **The measurement model** applies to simulated tracks only.
  - On Scheer's rig: resampled real paused-worm noise (`scheer_tracker_noise.npz`) plus losses.
  - On Chen's rig: jitter plus fragments.
  - Local search is never measured (115:73-78; `benchmark_grade.py:328-331`).
- **Four registered stops and three amendments** came before Part B (115:84-95):
  - the denoised round trip failed;
  - the Gaussian jitter failed, replaced by an empirical still-worm library;
  - the detector floor was raised from 0.01 to 0.03 mm;
  - **no open-loop worm reached the on-lawn reversal rate**, so it became reported, by a fallback
    registered first;
  - reachability missed by 0.07, so the oracle worm was refitted.
- **Controls, VALID** (115:16-24):
  - fresh N2 on 26 unread dates passes, 0 of 4 units failing;
  - Chen leave-one-recording-out passes, 0 of 6 failing;
  - the six negatives fail;
  - the **blind worm** (C.1 with sensing off, fitted to v2's D) fails all 5 populations, but
    **passes kinematics in 4 of 5** (115:150-154).
- **Negatives on v2** (`supporting/115-benchmark-v2/read.json`):
  - every one fails exits, time on the lawn and at least one kinematics statistic;
  - **the four open-loop negatives fail chemotaxis only through the presence test.** None of their
    chemotaxis z's is outside the bar (verified in `read.json`);
  - the untrained MLP passes chemotaxis, since it senses the odour;
  - the untrained connectome passes the uniform well and local search.
- **Weak chemotaxis bands.** With 6 recordings and a prediction factor of 2.01, the weathervane band
  is centre 0.364 ± 0.883 °/s and drift 0.026 ± 0.016 mm/s (`bands_v4.json`, unchanged since v2).
  **The weathervane band includes zero and both signs.** In practice the presence test carries the
  chemotaxis family, as 115:71 discloses ("a worm that ignores the odour inside every band").
- **What v2 does not establish** (115:182-186):
  - bout survival is reached by no oracle worm (mean z −2.47);
  - the tracker noise model is a still worm's, applied while the worm moves;
  - about 45% of v1's on-lawn reversals were tracker noise (115:163).
- **W's minimum:** 2.97 on the fitted families and 2.02 on the held-out ones (115:160-162).

### v3: Logbook 120

- **Statistics: 21, z\* 3.0381** (`bands_v3.json`; 120:57-68):
  - local search's early/late is graded on a **log** scale (`LOCAL_SEARCH_STATISTICS_V3`,
    `grading.py:87`; band 1.393 ± 0.375);
  - chemotaxis adds the **onset rate** (1.115 ± 0.943 per min) and the **direction-flip fraction**
    (0.0234 ± 0.0204), giving 5 statistics (`chemotaxis.py:43-46`, `125-138`).
- **Reported, not graded:** the reversal share against Roberts' 17.3%.
- **D_fit** is unchanged.
- **Controls, VALID** (120:15-24, 83-124):
  - rule checks 0.945, 0.975 and 0.991;
  - the oracle (C.1's model fitted to the three new statistics alone) reaches them;
  - the 10 held-out López-Cruz batches pass;
  - Chen leave-one-out passes;
  - **C.1b's four reversal-dominated answers are each caught in 5 of 5 populations**;
  - every negative fails.
- **Negatives' new statistics** (120:111-122):
  - flip fraction: the HMM, Poisson, untrained MLP, untrained connectome, blind worm and C.1 v1 fail
    it (z +4.9 to +7.4);
  - log early/late: the HMM fails it (z −3.7, from `read.json`);
  - the random walk and correlated walk pass both new chemotaxis statistics (z about −0.8) and fail
    elsewhere.
- **No single statistic catches every adversary.** Restart 4 is caught only by log early/late, and
  restart 6 only by flips (120:76-81).
- **W's minimum:** 2.967 fitted and 2.047 held out. A faithful controller passes 3 of 5 with
  probability 0.977 (120:128-135).

### v4: Logbook 124 (the current default)

- **Statistics: 24 in 6 families, z\* 3.0781** (`bands_v4.json`; 124:57-67). The new **activity**
  family on Scheer's small lawn, over on-lawn seconds (`grading.py:403-466`):

  - **pause share**: 1-s displacement under 0.02 mm, a floor derived from the still-worm library;
    0.4819 ± 0.0322;
  - **on-lawn flip fraction**: 0.1321 ± 0.0125;
  - **on-lawn onset rate**: 2.284 ± 0.225 per min.

- **The onset rate is graded only because the oracle reached it**, after v2 had left it reported
  (124:98). `grade_populations_v4` refuses bands with no recorded onset decision
  (`benchmark_grade.py:668-688`).

- **D_fit** is the small lawn, the uniform well, kinematics and activity (`bands_v4.json`
  `fit_families`).

- **Controls, VALID** (124:16-23, 80-128):

  - activity rule check 0.985;
  - the CSV and WCON round trip is exact;
  - the oracle reaches all three statistics;
  - fresh N2 passes, 0 of 2 units failing;
  - **the adversary (C.2's stage-2 answer) is caught in 5 of 5**: flips 0.005 (z −10), onsets
    0.086 per min, pause 0.60;
  - every negative fails the wild type in 5 of 5.

- **Negatives' activity z's** (124:115-128):

  | Negative | pause | flips | onsets |
  |---|---|---|---|
  | HMM | −8.4 | −2.5 to −2.0 | −4.2 to −3.9 |
  | Poisson | −10 | +1.0 to +1.2 (passes) | −1.6 to −1.5 (passes) |
  | random and correlated walks | −10 | −10 | −10 to −9.2 |
  | untrained MLP |: |: | −3.9 to −2.9 |
  | untrained connectome | +4.8 to +5.0 |: | −4.1 to −3.7 |

  - The Poisson worm fails activity only on the pause share.
  - **The untrained MLP passes the whole activity family in 1 of 5 populations.**
  - On v4, the untrained connectome still passes the uniform well, the blind worm passes kinematics,
    and **C.1's v1 answer passes local search** (`supporting/124-benchmark-v4/read.json`, population
    0 `family_passes`).

- **W's minimum:** 2.597 fitted and 2.047 held out. A faithful controller passes 3 of 5 with
  probability 0.975 (124:147-154).

______________________________________________________________________

## Revisions that followed a controller's result

Earlier versions stay frozen and are re-checked byte-identical each time (`identity.json` in 120
and 124). Every revision was registered before its validating read.

| Revision | What came before it | What changed | Source |
|---|---|---|---|
| v1 retry and hardening | The rule's own check failed; a synthetic no-search population passed 12–13% of the time. **Not a controller result** | Inflated scales, κ, a Student-t interval, presence | 109:95-114, 223-269 |
| **v2** | **C.1 passed v1** at 3 of 5 (113) "partly through routes no real worm takes": it roamed at about 2.6× real speed, and on-lawn reversals and bout shape were ungraded | A measurement model, kinematics and chemotaxis. "The maintainer asked for B to be reopened once, covering every angle, so it would not need reopening again." **It was reopened twice more** | 115:44-57; 113 header |
| v2 internal | A grading smoke found an odour-blind worm inside every chemotaxis band | The chemotaxis presence test | 115:71, 170 |
| v2 internal | No open-loop oracle reached the on-lawn reversal rate | Demoted to reported, by a registered fallback | 115:93-94 |
| **v3** | **C.1b failed v2** (119). Its best restart passed all 19 statistics in 2 of 5 populations, failing only on chemotaxis presence, and its fits are reversal-dominated off food (42–56% against 17.3%) | Log early/late, onset rate, flip fraction | 120:40-55; 119 header |
| **v4** | **C.2's stage-2 answer does not run** (123), and **C.1's v3 answer, which passed v3** (121), rarely reverses (about 3% against 12–17%) | The activity family, inside D_fit; the onset rate promoted to graded. **C.1's v3 pass then fails v4 at 0 of 5** (onset z −6.4 to −6.7) | 124:12-14, 40-51, 132-143 |

**Simulator revisions after fits (not benchmark changes, but the same pattern):**

- **Food-dependent reversal kinematics** (Logbook 108). These followed a data check (B.1), not a
  controller.
- **Speed-following on-food reversals** (Logbook 132). C.2's clean-start answer passed 15 of 15
  populations with this reversal patched into the evaluator, against 8 or 9 of 15 on the pooled
  speed (AGENTS.md, C.2 paragraph; Logbooks 131 and 132).
- The `feat/fit-c2` branch does **not** change grading or bands. Its only diff to
  `benchmark_grade.py` adds an arm (`git diff main...feat/fit-c2`).

______________________________________________________________________

## Candid assessment of the benchmark

1. **The benchmark has been tightened after each controller came close.** v2 followed C.1's v1
   pass, v3 C.1b's near-miss, and v4 C.1's v3 pass and C.2's non-running fit. Each revision was
   registered, oracled and validated on held-out data, and earlier verdicts are kept. But the
   statistics were chosen by **discriminability probes run on the very controllers they then
   catch**, and those controllers serve as the validity "adversaries":

   - v3's probe read C.1b's 9 answers (120:72-81, 146);
   - v4's probe read C.2 stage 2 and C.1 v3 (124:70-78).

   That is adversarial benchmark design. Its standing as a fixed yardstick is weaker than
   "registered" suggests.

2. **Held-out data have been read repeatedly in planning.**

   - v3's brief read every López-Cruz held-out batch, all five Roberts cohorts, Calhoun and all 8 Chen
     recordings (`supporting/120-benchmark-v3/brief.md:5-15`).
   - v4's probe read PD1074's held-out cohorts (124:145, 169).
   - The classifier was never blind to held-out dates (PROVENANCE.md:90-93).

   Fresh N2 dates (26, 501 animals) are the cleanest remaining held-out set (115:123-128).

3. **Thin cells.**

   - The uniform well rests on 2 band dates. Its rule check cannot be run, and its scales borrow
     the small lawn's (109:189-190).
   - Chemotaxis rests on 6 recordings from one lab, with a prediction factor of 2.01. Its z-bands
     do not exclude an odour-blind worm, so presence does the work, and the weathervane band
     spans zero.
   - tph-1 rests on 3 band dates, and its leaving reference is unreliable (110:80).
   - pdfr-1 has no resolved direction (109:227-238).

4. **Wide bands on some statistics.**

   - On v1, four statistics passed for a Poisson worm (109:163).
   - Some families are passed by non-worms: the untrained connectome passes the uniform well (v1, v4)
     and local search (v1, v2); the blind worm passes kinematics (v2, v4); the untrained MLP passes
     activity in 1 of 5 (v4); C.1's v1 answer passes local search on v4.
   - Discrimination comes from the family rule, the distance and the closed-loop lawn statistics
     (time on the lawn has a scale of 0.006, exits 0.20 per hour; 110:97-99).

5. **Reported, not graded:**

   - the on-lawn reversal rate (v2–v3, until it was promoted in v4);
   - the reversal share (v3–v4);
   - density, the quintile ratio, Flavell's timescales and pdfr-1's local search (109:166-183,
     253-257).

6. **Measurement asymmetries.**

   - Local search is graded on level-free statistics only, because the detector counts 2–3× López-Cruz's
     reorientations (brief.md:19).
   - The off-food detector's on-food recall is 0.43 (brief.md:22).
   - The noise model is still-worm noise applied while moving (115:185).
   - Bout survival is reached by no oracle worm (115:121).

7. **Oracles are partial.** Each reachability oracle fits only the new statistics. Its answers
   "say nothing about the other families" (120:101; 124:100). Their population z's spread with an SD
   of 0.2–0.3, far below the bands' between-date width (124:102).

## Not verified here

- How the uniform well's scales "rest on the small lawn's" (109:190). Not traced in code.
- N2's count of 26 unread held-out dates against PROVENANCE's 28. I assume two were read in B.4's
  tph-1 control, but did not check.
- Per-population detail for v3's and v4's negatives beyond the logbook tables. Their read.json files
  store only the new statistics for each negative.
- Nothing was run; every number is from committed logbooks, JSON records and source.
