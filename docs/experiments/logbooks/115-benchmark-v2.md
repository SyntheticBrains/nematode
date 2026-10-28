# 115: Phase 10 — B v2: The Benchmark Revised Once, VALID, and C.1's v1 Pass Does Not Survive It

**Status**: completed. **B v2 is VALID.** Every registered check holds:

- the measurement checks on both rigs;
- the reachability oracle;
- a fresh N2 control on 26 dates no reading had touched;
- the chemotaxis leave-one-recording-out validation;
- B.4's six negatives (and the four open-loop ones on chemotaxis);
- the blind worm, and C.1's v1 answer.

**C.1's v1 pass does not survive v2.** Its fitted answer now fails every population, so C.1 is refitted on v2 (C.1b) after the F.0b-i re-run.

The registration was amended three times before the readings it governs, each after a registered stop. As a result, the on-lawn reversal rate is reported, not graded, and the tracker's noise is resampled from real still-worm readings.

> | Registered check | Result |
> |---|---|
> | The measurement model: the losses, the jitter (on held-out band dates), Chen's round trip | pass |
> | Reachability: a fitted open-loop worm on 40 populations, four speeds | pass (mean z −0.29 to +0.35; worst −0.74; shares ≥ 97.5%) |
> | Fresh N2: 4 units, allowance 1 | 0 fail |
> | Chemotaxis leave-one-out: 6 recordings, allowance 1 | 0 fail, all present |
> | B.4's six negatives fail the wild type; the four open-loop ones fail chemotaxis | all |
> | The blind worm, fitted against v2's D, fails at 3 of 5 | fails all 5 (D 3.6–4.6) |
> | C.1's v1 answer fails at 3 of 5 | fails all 5 (D 6.1–6.2) |

**Date**: 2026-10-27 to 2026-10-28.

**OpenSpec changes**: `revise-benchmark-v2`, and `benchmark-v2-empirical-noise` (the noise model, after the first stops).

**Records**: [`supporting/115-benchmark-v2/`](supporting/115-benchmark-v2/):

- `launch.md` (Part A, amendments 1–3, Part B);
- `probes.json`;
- `bands_v2_record.json`, with `bands_v2_record_partA.json`;
- `oracle.json`, `oracle_amendment1.json`, `oracle_amendment2.json`, `oracle_amendment3.json`;
- `noise_pilot.json`, `reachability_worm.json`, `reach_refit.json`;
- `read.json`, `blind_v2.json`, `controls_v2.json`;
- `import_roundtrip_v2.json`.

The bands are [`data/benchmark/bands_v2.json`](../../../data/benchmark/bands_v2.json), and the noise library is [`data/benchmark/scheer_tracker_noise.npz`](../../../data/benchmark/scheer_tracker_noise.npz).

## Objective

C.1 passed the wild type on v1 ([Logbook 113](113-c1-fit.md)) partly through routes no real worm takes.

- **What v1 never graded:**
  - how fast a worm moves within a state (C.1 roamed at about 2.6× the real speed at the classifier's scale);
  - how often it reverses on food;
  - the shape of its roaming bouts.
- **The measurement asymmetry:** real tracks carry a tracker's noise and losses, and simulated ones did not.

The maintainer asked for B to be reopened once, covering every angle, so it would not need reopening again. The design (change `revise-benchmark-v2`, approved 2026-10-27) adds:

- a measurement model for simulated tracks;
- a kinematics family, fitted for every controller;
- a chemotaxis family from Chen et al. 2025's naive N2 butanone tracks, held out for C.2 and C.3 and fitted for C.1;
- validity conditions that include the known exploits failing.

v1 stays frozen for every verdict read on it.

## Method

- **The families, D and the bar.** D v2 weights five families equally:

  - the small lawn (6 statistics), the uniform well (3) and local search (2), as in v1;
  - kinematics (5): the dwelling and roaming 10-s window speeds' medians and 90th percentiles, and roaming-bout survival at 5 min;
  - chemotaxis (3): up-gradient drift, the log quintile ratio and the weathervane slope.

  D_fit (for C.2 and C.3) leaves out local search and chemotaxis. The bar is Bonferroni over 19 statistics, z\* = 3.0078. The on-lawn reversal rate is computed and reported (see below).

- **Chemotaxis must be present**, as local search must: drift and the log ratio above 0 at the 2.5% quantile of 2,000 resamples. A grading smoke found a worm that ignores the odour inside every band, against the spec's requirement that such a worm fail.

- **The measurement model**, applied to simulated tracks only, and only on the rigs whose trackers were measured:

  - **Scheer's rig** (the small lawn, the uniform well, kinematics): resampled real noise and context-dependent losses as runs;
  - **Chen's rig:** Gaussian jitter and real fragment lengths.

  Local search stays exact, since López-Cruz's counts come from a rig no track measures.

- **The controls,** each registered before it was read: the measurement checks, the reachability oracle, the fresh N2 control, the leave-one-out, the negatives and two adversaries.

## Results

### The registered stops, and what each changed

The run stopped four times at a registered rule before Part B. Each stop's verdict stands, and each change was registered before the reading it affects.

1. **The round trip, as registered in Part A, failed on Scheer's rig** (`oracle.json`: reversals z −12.6, dwelling speeds −5.1 and −5.3). The denoiser was the cause. A centred 5-s mean alone removes about half the detected reversals (z −23.8), while the loss model on gap-filled tracks moves nothing past 0.3. Amendment 1 replaced the round trip with a loss check and a jitter check on the authors' paused stretches.
2. **The Gaussian jitter failed the jitter check** (`oracle_amendment1.json`). The real tracker reads a still worm with fewer false onsets (2.14 against 3.87 per paused minute) and more 10-s drift (window speed median 0.0038 against 0.0016 mm/s). The maintainer chose an empirical model, built in its own change: a library of real paused stretches from half the band dates, with the check on the other half.
3. **The new change's pilots stopped it twice** (`noise_pilot.json`, `reachability_worm.json`):
   - **The noise draw:** joining stretches end to end added false onsets in-sample (+1.19 per paused minute). An autoregressive model and nearest-offset splicing did no better on onsets, though splicing kept the speeds.
   - **The cause:** the detector's 0.01 mm floor sits inside the tracker's noise. Raising it to 0.03 mm, by a stated rule (the still worm under 1% of the on-lawn rate), removes the dependence.
   - **Reachability:** B.4's matched worm missed the kinematics with no noise at all. A fitted two-state worm reached every speed only after it gained a speed spread within each state, but no open-loop worm reached the on-lawn reversal rate (11 parameters: median RMS z 1.59).
   - **The fallback**, committed before that last fit, applied: **the reversal rate is reported, not graded.**
4. **Reachability missed by 0.07** under amendment 2 (`oracle_amendment2.json`): roaming p90 mean z −1.07, though every population passed every speed. The worm had been fitted with the dropped reversal rate in its objective. Amendment 3 refitted it on the four graded speeds and checked Part A's own gate on 10 discarded populations before registering.

### The measurement checks (`oracle_amendment2.json`)

- **The losses:** every kinematics |z| ≤ 0.15 between gap-filled band-side tracks through the loss model and the raw tracks.

- **The jitter, on the 20 check-half dates (7,014 paused stretches):** each model-minus-real difference lies within the central 95% of 1,000 random date halvings:

  - window speed median −0.000011 mm/s, against a range of [−0.000414, +0.000386];
  - p90 −0.000044, against [−0.000352, +0.000310].

  The onset rate, reported, differs by +0.010 per paused minute.

- **Chen's round trip:** the authors' smoothed fragments with the model's jitter give every chemotaxis statistic within |z| ≤ 0.11.

### The reachability oracle (`oracle_amendment3.json`)

The refitted worm, on 40 new populations at 159 M, against Part A's gate (mean z in [−1, 1], and |z| ≤ z\* in at least 90%):

| Statistic | Mean z | Share within z\* |
|---|---|---|
| dwelling speed median | −0.29 | 100% |
| dwelling speed p90 | +0.04 | 100% |
| roaming speed median | +0.35 | 97.5% |
| roaming speed p90 | −0.74 | 100% |

Bout survival, excluded from the gate, reads −2.47. The worm's bouts are not shaped like real ones.

### The fresh N2 control (`read.json`)

N2's 26 unread held-out dates (501 animals) were split into two halves and graded on N2's own bands, giving 4 units.

- **No unit fails** (allowance 1); every |z| ≤ 1.31.
- P(more than 1 fails) under the rule checks' rates was 0.014, stated in Part B.

### The chemotaxis leave-one-out (`bands_v2_record.json`)

- Each of the six standard recordings passes against bands from the other five, and each is present, so there are 0 failures (allowance 1).
- The off-protocol recordings are reported: 20220804 is present; 20220203 is not (its log-ratio bound is 0).

### The negatives and the adversaries (`read.json`, `blind_v2.json`)

**B.4's six negatives,** on B.4's seeds, through the measurement model and with the chemotaxis assay:

| Negative | D on v1 (Logbook 110) | D on v2 | Fails chemotaxis |
|---|---|---|---|
| matched HMM | 5.70 | 5.07 | yes |
| Poisson | 7.68 | 6.47 | yes |
| random walk | 8.02 | 7.43 | yes |
| correlated walk | 6.96 | 6.75 | yes |
| untrained MLP | 5.78 | 6.17 | no |
| untrained connectome | 4.49 | 4.03 | yes |

Every negative fails the wild type, and the four open-loop ones each fail chemotaxis as required. The untrained MLP passes chemotaxis, but it senses the odour; only the open-loop four were required to fail.

**The blind worm** (C.1 with every sensing term off, fitted by sep-CMA against v2's D):

- its two restarts plateaued at generations 109 and 97, with σ at or below σ₀;
- it **fails all five populations** (D 3.6–4.6): the small lawn, the uniform well, local search and chemotaxis;
- it **passes kinematics** in four of five. Speeds alone are reachable without sensing, which is why kinematics is a family and not the benchmark.

**C.1's v1 answer** (restart 3) **fails all five populations** (D 6.1–6.2), failing the small lawn, the uniform well, kinematics and chemotaxis. Read through the tracker, its roaming and lawn statistics no longer pass.

### Reported beside the verdict

- **W's minimum, both family sets:**
  - the fitted families (small lawn and kinematics): 95th percentile 2.97 (median 1.50);
  - the held-out families (local search and chemotaxis): 2.02 (median 1.33).
- **The on-lawn reversal rate,** reported, has a band centre of 2.28 per minute at the 0.03 floor. On the library half's dates it reads 4.11 at the 0.01 floor and 2.27 at 0.03 (`noise_pilot.json`), so about 45% of v1's detected on-lawn reversals were tracker noise or sub-0.03-mm wobble.
- **The tracker's own false-onset rate varies about threefold between recording dates** (1.08–3.26 per paused minute at the 0.01 floor). The bands' between-date scales absorb this, and so does the permutation gate that judged the noise model.
- **The track importer's round trip is exact on v2's kinematics** in both formats (`import_roundtrip_v2.json`). Imported tracks are never measured, and chemotaxis cannot be imported (`data/benchmark/IMPORT.md`).
- **The 3-of-5 oracle on v2:** a faithful population passes all five families with probability 0.818, so a faithful controller passes 3 of 5 with probability 0.955 (Part B).

## Corrections made in the open

- **The chemotaxis presence test** was added after a grading smoke showed a worm that ignores the odour inside every band. It was measured on the real recordings and registered in Part A before any registered reading, and disclosed there.
- **The registered round trip's denoiser failed, not the model.** That reading stands in `oracle.json`, and amendment 1 replaced it.
- **The noise model, the reversal floor and the scope of the lawn model** were each decided after a registered stop, on the library half or on discarded seeds. Each was disclosed in amendment 2 before the check half was read. The exploratory pilots' scripts (the autoregressive and splicing draws, and the floor grid) are kept off-repo with the run logs. Their readings are recorded in the change's design and in amendment 2.
- **A preflight lapse:** before amendment 2, only the reachability worm's RMS z was checked on its discarded populations, not Part A's per-statistic gate, and reachability then missed by 0.07. Amendment 3 records it, and checked the gate itself before registering.
- **Two code faults stopped stages before they read anything:**
  - the amended oracle first stopped in its loss check, on a reported statistic without a band;
  - a Part B table failed markdownlint after commit and was reworded in the next commit.

## What this establishes, and what it does not

**It establishes** that the benchmark, as revised, is VALID under its own registered rule. Real held-out worms pass, the known negatives and both adversaries fail, and its measurement model reproduces the real tracker on both rigs' checks. The v1 loophole C.1 used is closed: C.1's v1 answer fails every v2 population.

**It does not establish** that the kinematics family is fully reachable, or that the tracker's noise is right while the worm moves.

- **The reversal rate:** no open-loop point worm reaches the on-lawn reversal rate through the tracker, so it is reported, not graded. Whether a closed-loop controller can is C.1b's and C.2's to show.
- **Bout survival** is reached by no oracle worm (mean z −2.47). Its validation is the N2 control.
- **The noise** is the real tracker's readings of a still worm, applied at every second.

**It changes** what C.1's v1 pass means. That pass stands at its own setting (v1), but it does not carry to v2. C.1 is refitted on v2 (C.1b), after F.0b-i is re-run on v2 to confirm the fitting method recovers known parameters under the new benchmark.

## Registered consequences

- **B v2 VALID → v2 becomes the benchmark:** *met*. `benchmark_grade.py` defaults to v2, and v1 stays for past verdicts.
- **The F.0b-i re-run on v2, then C.1b on v2:** *deferred-with-destination*, as the next two changes (tracker § F.0 and § C).
- **The on-lawn reversal rate graded:** *unreachable-with-reason* for an open-loop point worm through the tracker. It is reported beside the grade.
- **W's minimum on both family sets:** *met*. W's registration cites the set it reads (tracker § W).
- **W.1's reconstruction and weight-noise checks** (Yad & Nabila 2026): *deferred-with-destination*, to W.1.

## Artefacts

- `scripts/analysis/benchmark_bands.py --version v2` builds the bands.
- `scripts/analysis/benchmark_grade.py --bands v1|v2` grades.
- `scripts/analysis/benchmark_v2_controls.py` runs every stage: `probe`, `oracle`, `noise-library`, `noise-pilot`, `oracle2`, `reach-refit`, `oracle3`, `read`, `blind` and `verdict`.
- `quantumnematode.benchmark.measurement` holds the tracker model and the noise library, and `quantumnematode.benchmark.chemotaxis` the chemotaxis statistics and presence.
- `quantumnematode.evaluation.odour_field` holds Chen's field for the batched evaluator.
- `BatchedStateHazards` gains an optional speed spread and a reversal coupling, both off by default.
