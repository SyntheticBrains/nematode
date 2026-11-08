# 129: Phase 10 — C.2 Fitted to the Wild Type With Its Three Mutants: Every Mutant Passes in 5 of 5 Populations, and the Wild Type Misses by Hundredths

**Status**: completed, as a diagnosis. No fit is registered. **C.2 is fitted to the wild type together with Scheer's three mutants, with PDF's and ser-5's signs from receptor biology.** On five fresh populations graded with presence:

- **tph-1, pdfr-1 and tax-4 pass in all 5,** each in its real direction;
- **both presence tests pass in all 5;**
- **the wild type passes in none,** but three populations fail a single statistic each, the on-lawn reversal onset rate, at z −3.12 to −3.14 against a bar of 3.078. Every other miss is within 0.07 of the bar.

The verdict fixed before the run needed the wild type in 3 of 5, so it is not met.

> | Fit | Generations | Wild type, 5 populations with presence | Mutants passing | D | What fails |
> |---|---|---|---|---|---|
> | the wild type with its mutants | 150 | 0 of 5 | 5, 5, 5 of 5 | 1.50–1.58 | the small lawn's roaming bout (+3.6) and the roaming speed's 90th percentile (−3.1 to −3.2) |
> | the same, continued with slow speed noise | 50 | 0 of 5 | 5, 5, 5 of 5 | 1.58–1.71 | the on-lawn onset rate (−3.12 to −3.15), and in two populations the roaming bout (+3.09, +3.10) or its survival (−3.09) |

**Date**: 2026-11-07 to 2026-11-08.

**Records**: [`supporting/129-c2-wild-type-and-mutants/probes/`](supporting/129-c2-wild-type-and-mutants/probes/), each probe's record beside its script as run. The constraints were applied in the scripts; no package code changed.

## Objective

Logbook 128 found that wild-type data do not identify C.2's perturbation responses. With PDF and ser-5 signed from receptor biology, the right responses are representable, but a 60-generation fit of the wild type with its mutants traded one against the other. The maintainer chose one larger fit to tell a limit of the search's budget from a limit of the model. Its verdict, fixed before the run: in both of two populations, at least four families pass, and tph-1 and tax-4 pass.

## The fit

`overnight.py`: Logbook 128's model and bounds (the inhibitory scale at least 0.09; PDF's and ser-5's strengths excitatory, after PDFR-1's Gαs and SER-5's Gαq couplings), from its 60-generation answer.

- **The objective:** the six families, plus each mutant's two lawn differences on its own band, with the mutants weighted to half the objective. Both guards apply.
- **The search:** sep-CMA at population 48 and σ₀ 0.05, 200 animals a candidate, 150 generations on 16 workers, in 12.5 h.

| Readout | D | Families | tph-1 | tax-4 | pdfr-1 |
|---|---|---|---|---|---|
| 0 | 2.51 | 2 | +0.12, fails | +0.05, fails | passes |
| 20 | 2.17 | 2 | +0.22, passes | −0.03, passes | passes |
| 60 | 2.39 | 2 | +0.27, passes | −0.18, passes | −0.04, passes |
| 100 | 1.76 | 4 | +0.26, passes | −0.09, passes | −0.08, passes |
| 150 | 1.56 | 5 | +0.24, passes | −0.13, passes | −0.06, passes |
| 150, confirmation | 1.51 | 5 | +0.24, passes | −0.12, passes | −0.06, passes |

**Its verdict is met:** five families and both mutants pass in both populations. tph-1 and tax-4 passed at every readout from generation 20 on. Weighted to half the objective, the mutants were no longer traded away.

## The full grade

`overnight_grade.json`: the answer on 5 populations of 200, every family graded with presence, and the three mutants.

- **Both presence tests pass in all 5,** chemotaxis's included, which failed in every earlier C.2 fit.
- **All three mutants pass in all 5.**
- **The wild type passes in none.** Each population fails one or two statistics: the small lawn's roaming-bout length (+3.57 to +3.62, in 3) and the roaming speed's 10-s window 90th percentile (−3.08 to −3.21, in 4). D is 1.50–1.58.
- **Reported beside the grade:** off food the worm spends 5.5% of its seconds reversing, against Roberts' 17.3%.

The small-lawn roaming-bout length is the statistic C.1 failed on too (Logbooks 113 and 121): one roaming state serves both of Scheer's lawns.

## Slow speed noise

`speednoise.json`, on the answer with no fitting, 200 animals. The probe model gains `encoder_speed_noise: ou` (Logbook 123), its timescale appended as the vector's last entry.

| Timescale | Roaming speed p90 | Small-lawn roaming bout | What newly fails |
|---|---|---|---|
| 0.25 s (nearly independent) | −3.09, fails | +3.15, fails | — |
| 3 s | −1.86 | passes | the roaming speed's median (+3.25); the on-lawn onset rate (−3.84) |
| 10 s | −0.81 | passes | the median (+4.0); the onset rate (−4.7) |
| 30–100 s | passes | passes | the median (+5.6 to +6.6); the dwelling speed's median (−4.5 to −5.3); the onset rate (−5) |

Slow speed noise widens the 10-s window speeds, as it was built to. The answer, tuned without it, moves elsewhere. This answers Logbook 123's question, whether slow noise moves the window-speed spread on a fit whose speed responds, as description. Its registered control is re-run in C.2's change.

## The continuation with speed noise

`continuation.py` continues the fit with the noise on and its timescale fitted, from 3 s.

- **The search:** σ₀ 0.03, population 48, 200 animals, 50 generations, in 4.1 h. The timescale settled at 0.85 s.
- **The verdict, fixed before the run:** the wild type in at least 3 of 5 populations with presence, and tph-1 and tax-4 each in at least 3 of 5.

| Population | D | What fails |
|---|---|---|
| 0 | 1.64 | the on-lawn onset rate (−3.12) |
| 1 | 1.65 | the on-lawn onset rate (−3.13) |
| 2 | 1.58 | the on-lawn onset rate (−3.14) |
| 3 | 1.65 | the roaming bout (+3.10); the on-lawn onset rate (−3.15) |
| 4 | 1.71 | the roaming bout (+3.09); roaming survival (−3.09) |

- **Every mutant passes in all 5, and so does presence.** Off food the worm reverses 7% of its seconds.
- **The verdict is not met:** the wild type passes in 0 of 5. Three populations miss on one statistic by 0.04–0.06 of the bar.
- **The one systematic miss is too few on-lawn reversals.** C.1 had the same gap until it gained its own on-food dwelling reversal rate (Logbook 125). C.2's reversal hazard serves on food and off food through one readout.

## What this establishes, and what it does not

- **C.2 can be fitted to the wild type and its three mutants together,** with every mutant right in 5 of 5 populations and the wild type within hundredths of its bars. That is the closest any fit of C.2 has come, and the first in which its perturbation responses are right.
- **Two conditions it needed.** Signs held from receptor biology (PDF and ser-5, where the fit had reversed both), and the mutants in the objective at enough weight to hold them.
- **Its limits:**
  - **The mutants were fitted, not predicted.** A held-out perturbation is still missing. pdfr-1 is graded by magnitude alone, and mgl-1 is inert (Logbooks 127 and 128).
  - **The start descends from the imitation of C.1** (Logbook 126), through every later probe.
  - **Every reading is description:** one restart, on discarded seeds, under constraints applied in scripts.

## Cost

- **The fit:** 12.5 h for 150 generations (about 300 s a generation, at population 48, 200 animals, four assays and three mutants).
- **The full grade:** about 20 min.
- **The speed-noise check:** about 30 min.
- **The continuation:** 4.1 h.

A fit like this takes about 17 h from Logbook 128's answer. W's arms would each need one from their own start.

## Implications

C.2's next change builds this procedure into the encoder and the fit, each piece off by default and byte-identical:

- the soft-capped readout and pool centres;
- per-ligand tying;
- the graded box and the inhibition floor;
- the two receptor signs, from verified citations;
- slow speed noise;
- both guards;
- the mutants in the objective.

Its open questions go to the maintainer:

- **The on-lawn onset rate:** an on-food reversal term of C.2's own, as C.1 needed, or more search.
- **The start:** whether the procedure reaches this basin without the imitation of C.1, which equal treatment across W's arms needs.
- **The fit split:** the wild type with Scheer's three mutants, as a D37 amendment. A held-out test is still to be chosen: held-out cohorts, or another perturbation.
- **F.0b-ii,** sized on this procedure.
- **The compute:** at about 17 h a fit, W's arms decide between this machine and rented capacity.

## Seeds

All discarded:

- **The fit:** 770,000,000 + 200·k + i, readouts 772 M, confirmation 773 M, CMA 9487.
- **The full grade:** 780,000,000 + 200·g + i.
- **The speed-noise check:** 99,840,000 + i.
- **The continuation:** 785,000,000 + 200·k + i, readouts 786 M, grade 790,000,000 + 200·g + i, CMA 9488.
