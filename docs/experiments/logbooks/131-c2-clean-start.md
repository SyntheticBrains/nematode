# 131: Phase 10 — C.2 From a Clean Start Passes the Wild Type and Every Mutant in 15 of 15 Populations, Once a Reversal Follows the Worm's Own Speed

**Status**: completed, as description. No fit is registered. **C.2's encoded connectome reaches the benchmark from a clean start, with no imitation of C.1 in its lineage.** On 15 fresh populations graded with presence, its last answer passes all 24 wild-type statistics in every one, with tph-1, pdfr-1 and tax-4 passing in every one.

Two things made the difference:

- **Scoring the search the way the verdict scores.** The fit's objective is a distance, an RMS over every statistic. A statistic a few tenths past its bar barely moves it, while the verdict fails the population. The last stages add each statistic's estimated chance of failing a population, from its measured noise between populations.
- **A reversal that follows the worm's own speed.** The locomotor primitives give every reversal that starts on food one speed, 0.069 mm/s, with roaming and dwelling worms pooled. Real on-lawn reversals follow the speed just before them, from 0.048 to 0.084 mm/s, and a dwelling worm's are the slow ones. With the pooled value, the fit could not hold both the on-lawn onset rate and the dwelling windows' speed inside their bars. With a reversal that follows the worm's speed, patched into the evaluator in scripts, a refit holds both with room.

> | Step | Generations | Wild type, fresh populations with presence | Mutants | What limits it |
> |---|---|---|---|---|
> | the curriculum, from the box centre | 600 | 0 of 5 | 5 of 5 each | the on-lawn onset rate (z −3.37 to −3.65) |
> | a hinge past each bar, 0.3 inside | 50 | 2 of 5 | 5 of 5 each | three statistics near their bars |
> | the hinge set from each statistic's noise | 50 | 3 of 5; 9 of 15 with 10 more | 15 of 15 each | the onset rate (mean −3.08) |
> | each statistic's chance of failing | 50 | 3 of 5; 8 of 15 with 10 more | 15 of 15 each | the dwelling 90th percentile (mean +2.93) |
> | **the same, refitted with a reversal that follows the worm's speed** | **50** | **5 of 5; 15 of 15 with 10 more** | **15 of 15 each** | **nothing; the closest is chemotaxis's flip fraction, +2.85, SD 0.05** |

**Date**: 2026-11-08 to 2026-11-10.

**Records**: [`supporting/131-c2-clean-start/probes/`](supporting/131-c2-clean-start/probes/), each record beside its script as run, with the order in the [README](supporting/131-c2-clean-start/probes/README.md). Every piece outside the package's procedure options was applied in scripts. No package code changed.

## Objective

Logbook 130 built Logbook 129's procedure into the encoder: the soft cap and pool centres, per-ligand tying, the graded box and inhibition floor, PDF's and ser-5's cited signs, slow speed noise, and the guards' forward observer. The registered fit is the next change, and it needed two things the probes had not shown.

- **A start that owes nothing to C.1.** Every earlier passing answer descended from Logbook 126's imitation of C.1. W gives each arm the same procedure, so a start from the teacher would carry C.1 into every arm. It would also leave C.4's contrast between C.2 and C.1 with one built on the other.
- **Evidence at the registered point:** a procedure that reaches a passing answer by fitting alone, with no step chosen by hand.

## Four short probes

`probes.py`, on the package's procedure options at 64 animals a candidate and population 24.

| Probe | Start | Result |
|---|---|---|
| **B02**: tax-4 left out of the objective, 40 generations | Logbook 129's passing answer | tax-4 stays right: −0.034 at the start, −0.037 at the end, passing throughout. tph-1 and pdfr-1 pass; D 1.68 to 1.51 |
| **B01**: a clean start, the 77 parameters carrying senses to outputs, 60 generations | the graded box's centre, where every sensory gain is 0 | stopped by its rule: the reversal output pinned (spread 0.00), the worm on the lawn 34% of the time |
| **B03**: B01 with the responsiveness guard graded, 300 × (1 − spread/0.5) | the same | responsive (spreads 3.8 and 2.8); on the lawn 56%, against the rule's 75% |
| **B04**: B03 with stage 1 at 150 generations, then all 137 parameters on all six families | the same | stage 1: responsive, on the lawn 81%. Stage 2 stalled at D 5.0–5.7 for 100 generations, against its rule of D below 3 |

- **tax-4 stays right with nothing holding it.** That is a proxy for prediction, not the held-out test itself.
- **A flat guard gives the search no gradient:** every candidate paid the same 300, so B01's reversal output stayed pinned. Graded, the guard lets a clean start find a responsive worm that holds the lawn.
- **Fitting every family at once from there stalls.** That set up the curriculum.

## Imitating real worms instead of C.1

A cheap feasibility probe (`realworm_predictability.py`) first. On Scheer's band-cohort wild type, gradient boosting predicted a second's run speed from geometric senses at R² 0.64, and the next second's reversal onset at AUC 0.59. Both exclude the worm's own speed history.

The imitation (`realworm_imitation.py`): 975 small-lawn and 44 uniform-well animals, each replayed through the evaluator's own sensing, eating and satiety along its real path, with a shadow C.2 carrying its modulators.

- **Training:** C.2 was trained by likelihood on what the worm did next. The targets were the run speed and curvature, and reversal and omega onsets (Logbook 105's bout rule: real worms reverse 4.2 times a minute on the lawn, and omegas come at 0.035 a minute). It ran three rounds of 10 epochs.
- **The fit on held-out animals:** only the marginals. Speed R² was about 0.0, and onset AUC 0.51. Closed loop, the answer scored D 6.48 and left the lawn.
- **Why** (`imitation_ceiling.py`): boosting on the same simulated senses along the same paths reaches R² 0.24 and AUC 0.55, or 0.34 and 0.59 with slow histories, so the information is there. But on the lawn the senses barely vary: concentration's SD is 0.007 and food contact's 0.03. Sensory gains bounded at ±2 cannot carry that, and most of what a real worm does next comes from its own state, roaming or dwelling, not from its senses.

The imitation was dropped as a start.

## The curriculum

`curriculum.py`, from B04's stage-1 answer with the speed noise's timescale at 1 s. It ran at 200 animals a candidate and population 48 on 16 workers, with the graded guard.

| Stage | Fits | σ₀ | Generations | Rule at its end | Its last readout |
|---|---|---|---|---|---|
| 2a | the four fitted families | 0.1 | 300 | D_fit at most 2.5 | **D_fit 1.77**; passes |
| 2b | all six families | 0.05 | 150 | D at most 2.5, at least four families | **D 1.69, five families**; passes |
| 3 | the six, plus tph-1's, pdfr-1's and tax-4's lawn differences at half the objective | 0.05 | 150 | the wild type in 3 of 5 fresh populations, tph-1 and tax-4 each in 3 | **the wild type 0 of 5;** fails |

- **2a's cap was raised from 150 to 300 at generation 33, by the maintainer's choice,** with the rule unchanged. Its median was then tracking Logbook 126's guarded run without imitation, which reached D_fit 2.15 only at generation 250. It resumed from its saved state.

- **2b passed once local search was defined.** Until generation 30 the worm showed no decay after food removal, so γ could not be fitted and most candidates paid the 1,000 for an undefined statistic.

- **Stage 3's tph-1 was inert for 50 generations,** at z about −4.8 (`serotonin_route_gen25.json`):

  | | Serotonin's pool | ser-5's strength | Serotonin removed: speed, reversal log-hazard |
  |---|---|---|---|
  | stage 3, generation 25 | 0.003 ± 0.0015 | 0.05 | −0.0007 mm/s, +0.020 |
  | Logbook 129's answer, which passed tph-1 | 0.024 ± 0.009 | 1.44 | +0.0014 mm/s, −0.024 |

  The stages fitted without the mutants had let serotonin's producers fall silent. Raising ser-5 to 1.44 alone left tph-1 where it was (`serotonin_nudge.json`: z −4.74 and −4.75). A kill rule was set, at the maintainer's choice: stop at generation 60 if tph-1's z was below −4 at both generations 50 and 60. It read −3.79 and −0.69, so the run went on.

  By generation 60 the pool was 0.013, and tph-1 passed (+0.235, z −0.69), and kept passing. ser-5 stayed weak (0.065). For one step, removing serotonin still slows the worm and raises its reversals; the closed-loop roaming effect arrives by a slower route (`serotonin_route_gen60.json`).

- **The verdict fails on one statistic.** Each of the 5 populations fails the on-lawn onset rate alone (z −3.37 to −3.65), and presence passes in all five (`verify_presence.json`). Every mutant passes in 5 of 5.

**The onset gap is mostly behaviour** (`onset_gap.json`, on the stage-2b answer). The fit's worms start 3.0 reversals per on-lawn minute against real worms' 4.3, by the flags. The detector catches 44% of the fit's reversals against 53% of real ones. So it reads 1.5 a minute against the band's centre of 2.28.

## Scoring the search as the verdict scores it

The objective is a distance, an RMS over the families of 10·asinh(z/10). With every other statistic at the truth, an onset rate at z −3.3 adds 0.54 to it. A candidate's score varies by about that much from noise between candidates, so the search cannot see the bar. Each stage below continues from the one before, at population 48 and 200 animals, for 50 generations.

**A hinge 0.3 inside each bar** (`hinge.py`). It adds 5 × the sum of max(0, |z| − target)², with z compressed as the ranking score compresses it, and the targets at 2.78 for a wild-type statistic and 1.94 for a mutant's.

- **The first launch was stopped after 2 generations.** The curriculum drops a stage's search state when the stage ends, so it restarted at a uniform σ₀ of 0.03. Stage 3 had adapted its per-parameter steps to 0.008–0.042. The median scores were 417 and 709, against stage 3's 1.6. Six draws at that step (`hinge_candidates.json`) each put several statistics past their targets, some as far as z 10 to 13, with hinges of 39 to 863.
- **Restarted from stage 3's own steps,** saved at generation 132 (σ 0.0224), generation 1's median was 17.7.
- **The verdict:** the wild type in 2 of 5, with every mutant in 5 of 5. Populations 0 and 1 pass all 24 statistics. The other three each fail one statistic, a different one in each: the dwelling 90th percentile (+3.28), the onset rate (−3.19) and pre-exit enrichment (−3.83). The onset rate's readouts went from −3.61 to −3.11.

**The noise between populations** (`noise_measure.json`: that answer on 15 populations, the verdict's five reproducing). It passes the wild type in 8 of 15.

| Statistic | Mean z | SD between populations | Passing |
|---|---|---|---|
| the on-lawn onset rate | −2.95 | 0.10 | 14 of 15 |
| the dwelling 90th percentile | +2.91 | 0.18 | 12 of 15 |
| pre-exit enrichment | −1.81 | **1.17** | 12 of 15 |
| the small lawn's roaming bout | +2.24 | 0.37 | 14 of 15 |
| chemotaxis's flip fraction | +2.82 | 0.03 | 15 of 15 |

The spread between populations is small for most statistics. A worm leaves the lawn a few times an hour, so pre-exit enrichment is the noisy one.

**Targets from each statistic's noise** (`hinge2.py`): the hinge's target set at bar − 1.65 × SD, so a statistic held at its target passes a population 95% of the time.

- **The verdict passes at the threshold:** the wild type in 3 of 5, and every mutant in 5 of 5.
- **On 10 more populations** (`robustness_check.json`), the wild type passes in 6, so 9 of 15 in all, with every mutant in 10 of 10.
- **Three of the four failures are the onset rate.** It drifted to a mean of −3.08, with an SD of 0.15, onto its bar.

**Why the hinges let statistics drift** (`penalty_noise.json`). Each candidate is scored on one population, so a hinge on a noisy statistic adds chance to the ranking. Per candidate, on the measured noise:

| Statistic (mean z, SD) | Quadratic hinge | Steeper hinge, 5 × (e + e²), targets 0.5 or more inside | Each statistic's chance of failing, 10 × Φ((abs(z) − 3.0781) / SD) |
|---|---|---|---|
| pre-exit enrichment (+0.36, 1.39) | 2.4 ± 6.6 | 6.6 ± 12.2 | 1.2 ± 1.4 |
| the onset rate at its bar (−3.08, 0.15) | 0.25 ± 0.30 | 3.9 ± 1.5 | 5.0 ± 2.9 |
| the onset rate moved to −2.70 | 0.00 ± 0.02 | 0.9 ± 0.9 | 0.4 ± 0.8 |

The planned steeper hinge would have given moving the onset rate in about 3 points, against ±12 from pre-exit enrichment's chance readings alone. The failure probability gives about 4.6 points against about ±4 from every statistic together. The maintainer chose it, and its sum is about 10 × the expected number of failing statistics in a population.

**Each statistic's chance of failing** (`hinge3.py`, SD the larger of the two measurements).

- **The verdict passes at the threshold:** the wild type in 3 of 5, and every mutant in 5 of 5.
- **On 10 more populations** (`robustness_check_hinge3.json`), the wild type passes in 5, so 8 of 15 in all.
- **The onset rate now passes 9 of 10** (mean −2.90), and the dwelling 90th percentile passes 6 of 10 (mean +2.93, SD 0.25). The best candidate's penalty stayed near 4.2 for 50 generations.

The failures moved from one statistic to the other: a frontier.

## The body model: a reversal that follows the worm's own speed

**The dwelling windows, split by reversals** (`dwelling_windows.json`, on that answer): the small lawn's on-lawn dwelling windows, classified as the grader classifies them.

| | Simulated | Real (band cohorts) |
|---|---|---|
| windows holding a reversal | 50.5% | 66.1% |
| window speed, median / 90th percentile, all (mm/s) | 0.0147 / 0.0317 | 0.0128 / 0.0282 |
| without a reversal | 0.0119 / 0.0304 | 0.0106 / 0.0277 |
| with a reversal | 0.0172 / 0.0326 | 0.0138 / 0.0285 |

- **A reversal speeds a simulated dwelling window more than a real one:** it raises the median by 45% against 30%, and the 90th percentile by 7% against 3%.
- **The fit reverses less while dwelling than real worms do,** so the reversals the onset rate needs would speed the windows further.
- **Without reversals, the simulated windows are about 10% fast.**

**Real reversals by state** (`reversal_by_state.json`; on-lawn bouts, by their 10-s bin's label):

| | Per minute in the state | Duration | Speed |
|---|---|---|---|
| dwelling | 4.37 | 1.99 s | 0.059 mm/s |
| roaming | 3.94 | 2.22 s | 0.087 mm/s |
| the primitives on food, both states pooled | — | 2.05 s | 0.069 mm/s |

The body cannot see the state, but it can see the speed. Each on-lawn reversal bout whose 3 preceding seconds are on the lawn and free of bouts gives a reversal speed and a speed before it (`reversal_coupling_fit.json`, `reversal_coupling_fit_held_out.json`):

| Cohorts | Bouts | Least-squares line | Reversal speed by decile of the speed before |
|---|---|---|---|
| band | 126,306 | v_rev = 0.0474 + 0.576 × v_before | 0.049 to 0.084 mm/s |
| **held out** | 71,719 | **v_rev = 0.0463 + 0.603 × v_before** | 0.048 to 0.084 mm/s |

The two lines agree to within about 2%. The probes used the band's line. The held-out line is the one the package should take, as Logbooks 105 and 108 took the on-food values from the held-out cohorts.

**The coupling on the fitted answer** (`reversal_coupling_probe.json`). The evaluator was patched in every worker: a reversal that starts on food moves at the line applied to the mean of the worm's last 3 run seconds, clipped to [0.049, 0.084]. No draw changes. It was graded on the same 10 populations as `robustness_check_hinge3.json`, animal for animal.

| Statistic | Pooled reversal | Following the worm's speed | Paired shift | Passing, of 10 |
|---|---|---|---|---|
| the dwelling 90th percentile | +2.93 | +1.82 | −1.11 | 6 → 10 |
| pre-exit enrichment | −0.71 | +2.61 | +3.33 | 10 → 6 |
| exits per on-lawn hour | −1.87 | −2.48 | −0.61 | 10 → 10 |
| the onset rate | −2.90 | −3.00 | −0.10 | 9 → 7 |
| **the wild type** | | | | **5 → 3** |

The dwelling windows' bias goes. But the network had been tuned to the pooled reversal: with dwelling windows no longer sped up, fewer read as roaming, and roaming before an exit stands out more. The answer needed refitting under the new body.

**The refit** (`hinge4.py`). The coupled body ran in every worker, with the failure probability's SD the largest of four measurements, including the coupled body's 10 populations.

| Readout | 0 | 10 | 20 | 30 | 40 | 50 |
|---|---|---|---|---|---|---|
| the onset rate | −3.12 | −2.77 | −2.67 | −2.55 | −2.68 | −2.64 |
| pre-exit enrichment | +3.32 | +0.63 | −0.10 | −0.55 | +0.13 | −0.74 |
| the dwelling 90th percentile | +1.61 | +2.06 | +2.02 | +2.32 | +2.04 | +1.96 |
| D / families passing | 1.54 / 4 | 1.48 / 6 | 1.45 / 6 | 1.39 / 6 | 1.41 / 6 | 1.40 / 6 |

- **The verdict:** the wild type in 5 of 5, and every mutant in 5 of 5.
- **On 10 more populations, the wild type passes in all 10,** and every mutant in 10 of 10, with D from 1.37 to 1.51.
- **The best candidate's penalty fell from 1.6 to 0.4,** against the pooled body's plateau near 4.2.

The closest statistics now have room:

| Statistic | Mean z, 10 populations | SD |
|---|---|---|
| chemotaxis's flip fraction | +2.85 | 0.05 |
| the on-lawn onset rate | −2.64 | 0.09 |
| local search's log early/late ratio | −2.27 | 0.10 |
| exits per on-lawn hour | −2.21 | 0.18 |
| the dwelling 90th percentile | +2.10 | 0.18 |

## What this establishes, and what it does not

- **A clean start reaches the benchmark.** The route was a graded guard, a staged curriculum, the mutants at half the objective, and each statistic scored by its chance of failing. Its lineage owes nothing to C.1.
- **The mutants can be fitted from that start.** The route to tph-1 was slow, though: the stages fitted without the mutants let serotonin's producers fall silent first.
- **The frontier between the onset rate and the dwelling windows' speed belonged to the body model, not the network.** With the pooled on-food reversal speed, two scorings ended at 8 or 9 of 15 populations. With the reversal following the worm's speed, the refit passes 15 of 15.
- **Its limits:**
  - **The route was adapted after each reading,** at the maintainer's choices: 2a's cap, the hinge and its restart, the noise targets, the penalty, the body. It is readiness evidence, not a verdict. The registered fit has to fix its procedure in advance and run it fresh.
  - **The body is a patch in a script.** The reversal's speed before is the simulated worm's true displacement, while real worms' is measured through the tracker, whose noise lifts the slowest decile. Fitted on the held-out cohorts, the line should enter the package's primitives, off by default, with the per-step runner matching.
  - **The mutants were fitted, not predicted.** B02's tax-4 left out is a proxy for prediction, not the held-out test.
  - **One restart, on discarded seeds.**
  - **A seed overlap in the curriculum:** stage 2a's candidates ran on 840,000,000 + 200·k + i. After about 5,000 candidates those blocks ran into its own readout animals (841,000,000 + i), and after 10,000 into stage 2b's base (842 M). Readouts are description only, and every verdict's populations are disjoint from every search.
- **C.1 was fitted under the pooled body.** C.4 compares the two under one body, so C.1's refit, already deferred to C.4, runs under the coupled one.

## Cost

| Step | Wall time on 16 workers |
|---|---|
| the four probes | 4.0 h (64 animals, population 24) |
| the real-worm imitation | 18 min training, beside the curriculum on 4 workers |
| the curriculum | 26.3 h: 2a 6.5 h, 2b 7.5 h, 3 12.3 h |
| each continuation of 50 generations | 4.0–4.3 h |
| each grade of 5 or 10 populations | about 2 min |

Every 200-animal fit runs about 75 s a generation on two assays, 180 s on all six, and 300 s with the three mutants. A fit by this route on the coupled body would be the curriculum and one penalty stage, about 31 h.

## Implications

Three changes, in order (the maintainer's choice, 2026-11-10):

1. **A reversal that follows the worm's own speed, as an option of the primitives.**
   - Its values come from the held-out cohorts: v_rev = 0.0463 + 0.603 × v_before over the last 3 run seconds, clipped to the deciles' range, for reversals that start on food. Their provenance goes beside `data/locomotion/primitives.json`'s.
   - It is off by default and byte-identical when off, with the batched and per-step runners matching.
   - Every controller would run on it, C.1's refit included.
2. **The registered C.2 fit (Change B),** its procedure fixed in advance:
   - the curriculum's stages, each passing its search state to the next;
   - the mutants entering while the search is broad, so that serotonin's producers are shaped before the wild type settles without them;
   - each statistic scored by its chance of failing, with noise measured inside the procedure;
   - readouts recording every statistic.
3. **The compute for W.** At about 31 h a fit, W's arms decide between this machine and rented capacity.

Still open:

- the held-out test: held-out cohorts in every fit, plus one tax-4 leave-one-out, as decided at Logbook 130;
- F.0b-ii, sized on this procedure;
- the D37 amendment for the fit split.

## Seeds

All discarded:

- **B01–B04:** 810–832 M (64·k + i), readouts 811, 816, 821 and 831 M, CMA 9490–9496.
- **The real-worm probes:** the replay 850 M, the closed-loop readout 851 M, the onset gap 852 M, the serotonin route 853 M, the serotonin nudge 854 M.
- **The curriculum:** stages 840, 842 and 844 M (200·k + i), readouts 841 M, CMA 9497–9499, the verdict 846 M (200·g + i).
- **The first continuation:** 860 M (200·k + i), readouts 861 M, CMA 9500, the verdict 862 M. The noise measurement re-read 862 M and added 863 M.
- **The second continuation:** 864 M, readouts 865 M, CMA 9501, the verdict 866 M, and its check 867 M.
- **The third continuation:** 868 M, readouts 869 M, CMA 9502, the verdict 870 M, and its check 871 M. The coupling probe re-read 871 M, its smoke test used 874 M, and the dwelling windows used 872 M.
- **The refit:** 875 M, readouts 876 M, CMA 9503, the verdict 877 M, and its check 878 M.
