# 128: Phase 10 — Wild-Type Data Do Not Identify C.2's Perturbation Responses; Signs From Receptor Biology Make Them Representable, but a Fit of the Wild Type With Its Mutants Trades One Against the Other

**Status**: completed, as a diagnosis. No fit is registered.

Logbook 127 found C.2's perturbations unreadable once it was fitted on every family. The maintainer chose to constrain the encoder with data, prototyped in scripts first.

- **Wild-type data do not identify C.2's perturbation responses.** In every fit, the knockouts' effects fade as the wild type's fit improves. That held with inhibition tied or floored and with PDF's sign fixed (V1, V2).
- **The fit had reversed two signs that receptor biology fixes.**
  - **PDF:** made excitatory, as PDFR-1's Gαs coupling says, pdfr-1's knockout lowers roaming by 0.21, the real direction.
  - **ser-5:** made excitatory, as its Gαq coupling says, tph-1's knockout raises roaming by 0.18 and passes.
- **A draft literature table** resolves 24 of the 54 receptors with free signs: 16 excitatory, 8 inhibitory.
- **pdfr-1 cannot be a held-out test.** It is graded by magnitude alone, so a knockout that does nothing passes.
- **Fitted with its three mutants for 60 generations,** the wild type improves (D 3.37 → 2.48, 3 families), but the mutants trade against it. tph-1 passes at the start and is lost; tax-4 passes at generations 20–40 and is lost. At the end only pdfr-1's uninformative pass remains.
- **By the branch fixed before that fit, the result is written up,** and C.2's design goes back to the maintainer.

**Date**: 2026-11-07.

**Records**: [`supporting/128-c2-identifiability/probes/`](supporting/128-c2-identifiability/probes/), each probe's record beside its script as run. They run as Logbook 127's do ([its README](supporting/127-c2-readability/probes/README.md)), from `build/c2-run-mode/`. The constraints were applied in the scripts; no package code changed.

## Objective

Logbook 127 found C.2 fitted on all six of v4's families with its perturbations unreadable. Every fit drove the inhibitory wiring scale near zero and left the modulator signs free. The maintainer chose to constrain the encoder with data and then fit, and to prototype the constraints in scripts before building them into the package. This logbook asks two questions:

1. Do the constraints keep the perturbations effective while the fit improves?
2. If not, what does?

## Two signs the fit reversed

**PDF** (`pdf_sign_check.json`, on Logbook 127's generation-150 answer, 200 animals). The fit set both PDF couplings inhibitory: pdf-1 at −1.07 and pdf-2 at −3.00, at the box's bound. PDFR-1, PDF's only receptor, couples to Gαs, which the layer's rule signs excitatory.

| PDF's strengths | Wild type's roaming | pdfr-1 roaming change |
|---|---|---|
| as fitted (inhibitory) | 0.23 | +0.09, the wrong way |
| the same magnitudes, excitatory | 0.53 | −0.21, the right way |

With the biological sign, PDF promotes roaming, as Flavell 2013 found, and its knockout lowers it.

**Serotonin** (`tph1_capacity.json`, on the leave-one-out screen's answer below). The fit set ser-5, serotonin's one receptor of unestablished sign, inhibitory (−1.58). It held serotonin's fixed-sign group at 0.64 of a box up to 3.0.

| Serotonin's strengths | Wild type's roaming | tph-1 roaming change |
|---|---|---|
| as fitted | 0.31 | +0.006 (z −5.2), fails |
| fixed group 3.0, ser-5 −3 | 0.26 | +0.06, fails |
| fixed group 3.0, ser-5 0 | 0.23 | +0.09, fails |
| fixed group 3.0, ser-5 +3 | 0.14 | **+0.18 (z −1.7), passes** |

With ser-5 excitatory, serotonin promotes dwelling and its loss raises roaming, as real tph-1's does. The literature pass below places ser-5 on Gαq in vivo, so excitatory.

## The receptors' couplings (a draft table)

`receptor_couplings_draft.json` is a subagent's literature pass over the 54 receptors behind the layer's 105 fitted couples. It uses `receptor_signs.json`'s rule and accepts a coupling only from an assay without a promiscuous G protein, or from in vivo genetics.

- **24 resolve:** 16 excitatory and 8 inhibitory.
  - PDFR-1 is Gαs (Janssen et al. 2008; Flavell et al. 2013).
  - NPR-1, NPR-3, NPR-4, NPR-11, NPR-17, EGL-6, DMSR-1 and DMSR-7 are Gαi/o.
  - AEX-2 is Gαs.
  - CKR-1, CKR-2, FRPR-3, FRPR-18, GNRR-1, TRHR-1, NPR-5, NPR-35 and SER-5 are Gαq.
- **Weaker entries, flagged in the table's notes:**
  - three rest on a PhD thesis (gnrr-3, nmur-1, tkr-2);
  - ntr-1 rests on a review, with a conflicting report;
  - npr-22's G protein is inferred;
  - ser-5 rests on a co-requirement with a reduction-of-function Gαq allele.
- **30 stay unresolved.** For most, the only evidence is Beets et al. 2023's screen, which used a promiscuous Gα16.
- **Status: a draft.** Two quotes were checked against their sources and are verbatim: PDFR-1's in Janssen et al. 2008, and EGL-6's in Ringstad & Horvitz 2008. Every signed entry is checked before the table enters the package.
- **A data issue it found:** the layer's flp-24 → npr-17 couple never loads, since flp-24 has no producer.

## Two constrained fits (V1, V2)

`prototypes.py`, `prototypes_V1.json` and `prototypes_V2.json`. Each fitted all six families at 64 animals a candidate, with both guards, on 8 workers each, with readouts every 10 generations on 200 animals:

- **V1:** the two chemical scales tied, PDF held excitatory; from the imitation answer, at population 48 and σ₀ 0.1.
- **V2:** the inhibitory scale floored at 0.09, PDF held excitatory; from the imitation's refinement, at population 24 and σ₀ 0.05.

| | Gen | D | Families | mgl-1 | pdfr-1 | tph-1 | tax-4 |
|---|---|---|---|---|---|---|---|
| V1 | 0 | 7.36 | 0 | 1.02 | −0.30 | +0.04 | −0.01 |
| | 20 | 5.35 | 0 | 0.37 | +0.04 | −0.01 | +0.01 |
| | 90 | 4.11 | 2 | 1.01 | +0.09 | −0.00 | −0.03 |
| V2 | 0 | 6.18 | 0 | 0.98 | +0.92 | −0.01 | +0.03 |
| | 10 | 5.51 | 2 | 0.81 | −0.27 | −0.06 | −0.23 |
| | 20 | 4.39 | 3 | 1.02 | −0.23 | +0.33 | −0.11 |
| | 60 | 2.59 | 3 | 0.99 | −0.04 | −0.00 | +0.05 |
| | 90 | 2.07 | 3 | 1.00 | −0.00 | +0.01 | −0.34 |

(mgl-1 is its early search against its wild type's; the mutants are their roaming change from their wild type.)

- **V2's fit improved, and its perturbation effects faded as it did.** pdfr-1 went from −0.27 to zero, mgl-1 from 0.81 to 1.00, and tph-1's +0.33 at generation 20 was gone by 60. Holding PDF's sign did not hold its effect: the fit shrank the pathway instead.
- **V1 fitted slowly** (D 4.1 at generation 90), and pdfr-1 went the wrong way even with PDF excitatory. Its one mgl-1 reading of 0.37, at generation 20, came during the search's transient.
- **Both were stopped by hand,** at their readouts' evidence (V2 at generation 91, V1 at 98). Their records end at generation 90.

## Leaving one mutant out

`leave_one_out.py` and `leave_one_out.json`. From V2's last answer, with V2's bounds, the objective added tph-1's and tax-4's lawn differences to the six families. pdfr-1 was held out. The fit ran at σ₀ 0.03, 64 animals, population 24, on 16 workers.

| Readout | D | tph-1 change (z) | tax-4 change (z) | Held-out pdfr-1 |
|---|---|---|---|---|
| Gen 0 | 2.08 | +0.003 (−5.2) | −0.35 (−5.3) | −0.01, passes |
| Gen 30 | 2.06 | +0.012 (−5.1) | −0.15 (−0.7), passes | −0.01, passes |

- **tax-4 was fitted. tph-1 did not move,** and the stop rule ended the run at generation 30. Reaching ser-5's excitatory side was beyond that step size, as the capacity check above shows.
- **pdfr-1 cannot be a held-out test.** The reference data do not resolve its direction, so it is graded by magnitude alone, and a knockout that does nothing passes. tph-1 and tax-4, whose directions are resolved, can.

## Fitting the wild type with its mutants

`fit_mutants.py` and `fit_mutants.json`.

- **The start:** the capacity check's point (serotonin's fixed group at 3.0, ser-5 at +3), with V2's bounds and ser-5 held excitatory.
- **The objective:** the six families plus each of the three mutants' lawn differences, nine equal families.
- **The search:** σ₀ 0.05, population 24, 64 animals a candidate, 60 generations on 16 workers, 1.26 h.

| Readout | D | Families passing | tph-1 change (z) | tax-4 change (z) | pdfr-1 (magnitude only) |
|---|---|---|---|---|---|
| Gen 0 | 3.37 | 2 | +0.19 (−1.5), passes | +0.11 (+5.0) | +0.05, fails |
| Gen 20 | 2.37 | 2 | −0.02 (−5.7) | −0.07 (+1.0), passes | passes |
| Gen 40 | 2.39 | 3 | −0.03 (−5.9) | −0.05 (+1.4), passes | passes |
| Gen 60 | 2.48 | 3 | +0.08 (−3.7) | +0.01 (+2.8) | passes |

- **The wild type improved, and the mutants traded against it.** Each mutant is one family of nine in the objective. The fit gained more on the wild type's families by weakening serotonin than it lost on tph-1.
- **The small lawn failed at every readout.** The families passing at the end are the uniform well, local search and kinematics.
- **The branch, fixed before the run,** needed four families and two mutants passing. It was not met.

## What this establishes, and what it does not

- **Wild-type data do not identify C.2's perturbation responses.** Across every fit here and in Logbook 127, the search settles where the knockouts barely matter, because the wild type does not need those pathways. Constraints on signs and inhibition do not change that: the fit can still shrink a pathway, or route it to the opposite behaviour.
- **With signs from receptor biology, the right responses are representable.** pdfr-1 reaches −0.21 with PDF excitatory, and tph-1 reaches +0.18 with ser-5 excitatory.
- **mgl-1 stays inert.** AIA barely reaches the motor pools in these networks (Logbook 127).
- **Every reading is description:** one restart per design, on discarded seeds, under constraints applied in scripts.

## Implications

The maintainer decides C.2's design from here. Three directions are open:

- **A larger fit of the wild type with its mutants:** population 48, 200 animals, a few hundred generations, about 10 h or more. It would tell a limit of the search budget from a limit of the model, and could weight the mutants more heavily.
- **Accept the limit for this model class:** report C.2's fitted families and its perturbation failures in the fidelity table, and read W on the fitted distance, with the perturbations reported as unidentified.
- **Change the model class:** give the network state that carries between seconds, and tonic sensory activity. That bears on mgl-1, and is a larger change, possibly Phase 11's.

Whatever is chosen, the draft receptor table is checked entry by entry before it enters the package.

## Seeds

All discarded:

- **The checks:** the PDF check 99,820,000 + i; the tph-1 capacity check 99,830,000 + i.
- **V1 and V2:** fits 740 and 741 M (64·k + i), readouts 742 M, CMA 9483 and 9484.
- **The leave-one-out screen:** fits 750 M, readouts 752 M, CMA 9485.
- **The mutant fit:** fits 760 M, readouts 762 M, CMA 9486.
- **The smokes:** k = 99,999 of each fit block.
