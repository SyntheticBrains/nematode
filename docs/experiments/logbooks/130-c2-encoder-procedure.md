# 130: Phase 10 — C.2's Fitting Procedure Built Into the Encoder: Byte-Identical When Off, the Probe's Passing Answer Reproduced

**Status**: completed, engineering. No fit is registered. Change `add-c2-encoder-procedure` builds Logbooks 126–129's probe procedure into the package, each piece off by default:

- **Byte-identical when off:** every existing connectome output (24 of 24 identity jobs) is unchanged.

- **The probe's passing answer reproduces:** the package's forward computes what the probe scripts' patched forward did, within 7×10⁻⁷ against a gate of 10⁻⁵, with the same primitive at every one of 7,800 steps.

- **The receptor table:** a literature pass, checked at source, signs 22 of the layer's 54 fitted-couple receptors. PDFR-1 and SER-5 are the two switched on.

- **For the next change:** a short continuation from the passing answer shows that fitting holds the on-food reversal route the answer relies on.

- **The grade reproduces too, reported as description:** on the probe's own five populations, the converted answer passes the wild type in the same 3 of 5, fails on the same statistics in the other two, and passes every mutant in 5 of 5, with D within 0.01 of the probe's.

**Date**: 2026-11-08.

**Records**: [`supporting/130-c2-encoder-procedure/`](supporting/130-c2-encoder-procedure/):

- `identity.json`: the identity check against main's baseline;
- `reproduce.json`: the forward parity and the five-population grade;
- `probes/`: the stability check, with its script as run.

## Objective

Logbook 129 passed the wild type in 3 of 5 populations with every mutant in 5 of 5, from a procedure applied in probe scripts:

- a monkeypatched forward;
- bounds on the CMA vector;
- PDF's and ser-5's signs held by bounds.

A registered fit, and every W arm, needs that procedure in the package: tested, byte-identical when off, and the same in both runners. The maintainer's decisions (2026-11-08), from the change plan:

- the inhibition floor at 0.09;
- PDFR-1 and SER-5 switched on, the table's other receptors written in but off;
- a clean, staged start for the registered fit;
- held-out cohorts plus one tax-4 leave-one-out.

The last two belong to the next change.

## What was built

| Option | What it does | Default |
|---|---|---|
| `encoder_readout: soft_cap` | each output's pre-squash value through 2·tanh(x/2) | `linear` |
| `encoder_pool_centres: true` | four fitted centres, subtracted from the pooled motor rates before the readout; appended after the speed noise's timescale | `false` |
| `encoder_box: graded` | wiring scales on [0, 0.75], sensory gains on [−2, 2] | `wide` |
| `encoder_inhibition_floor` | the inhibitory scale's lower bound | none |
| `modulator_receptor_signs` | named receptors' fitted couples take their cited sign; under per-ligand tying each is its own group | empty |
| `ConnectomeTopology.forward_observer` | called after each settling with the outputs and the hidden state, so a fit's guards need not replace the forward | none |

With the procedure's settings, the per-ligand encoder has 137 parameters: 132, plus the speed noise's timescale, plus four centres. ser-5's couple is its own group, `modulator.fixed.serotonin.ser-5`, beside serotonin's fixed group. The passing answer had their strengths at 1.44 and 2.97, which one shared magnitude could not hold.

## The receptor table

`data/neuromodulators/receptor_couplings.json` covers the 54 receptors behind the layer's 105 fitted couples (`PROVENANCE.md` has the rule).

- **The rule:** a coupling is accepted only from an assay without a promiscuous or chimeric G protein, or from in vivo genetics.
- **The draft** (Logbook 128) signed 24.
- **The check:** each signed entry's quote was checked verbatim against its source, two by hand and the rest by a delegated pass.
  - 22 hold: 14 excitatory, 8 inhibitory.
  - Two were withdrawn. npr-22's source names no G protein. ntr-1's quote came from a review, while an independent primary used Gα15.
- **Caveats in the entries' notes:**
  - three entries rest on a PhD thesis (gnrr-3, nmur-1, tkr-2);
  - PDFR-1's sign is its isoforms a and b's, since isoform c inhibits cAMP;
  - ser-5's rests on a co-requirement with a reduction-of-function Gαq allele.
- **The table changes nothing by itself.** A brain applies a sign only for receptors it names.

## The checks

- **Identity** (`identity.json`): `connectome_identity_check.py` against a baseline recorded on main before any change. All 24 jobs are byte-identical: every batched and per-step connectome job, the per-connection configs' built brains, and the encoded forward at the staged pilot's answers.
- **Runner parity:** the batched evaluator matches the per-step runner with every option on (seeds 5001–5032): states, action means and sampled actions within 1e-5, and the trajectories.
- **Unit tests:**
  - the defaults change nothing;
  - the cap's bound;
  - centres at zero leave the readout unchanged;
  - the graded centre and the floor;
  - out-of-box floors are refused;
  - the counts, the prefix property and the round trip;
  - the centres are learnable under PPO;
  - the observer changes nothing;
  - the receptor signs, their refusals, and their travel under the receptor-placement null;
  - every signed table entry is verified.
- **The forward parity** (`reproduce.json`):
  - **the conversion:** Logbook 129's passing answer, converted entry by entry by name (`c2_encoder_probe.package_vector`);
  - **the comparison:** the package brain against the probe scripts' patched brain, on one reference episode's recorded states on Scheer's small lawn (2,400 steps) and on food removal (5,400);
  - **the result:** the outputs differ by at most 7.2×10⁻⁷ and 4.8×10⁻⁷, against a gate of 10⁻⁵, and the chosen primitive is the same at every step.

**The five-population grade** (`reproduce.json`), reported as description beside the probe's. The probe's pass sits at the threshold, so this is not a gate.

| Population | Package: wild type, D | Probe: wild type, D | What fails, in both |
|---|---|---|---|
| 0 | passes, 1.684 | passes, 1.683 | nothing |
| 1 | fails, 1.678 | fails, 1.669 | chemotaxis's flip fraction (+3.11) |
| 2 | fails, 1.659 | fails, 1.659 | roaming survival (−3.30) |
| 3 | passes, 1.621 | passes, 1.628 | nothing |
| 4 | passes, 1.660 | passes, 1.661 | nothing |

Every mutant passes in all five, in both.

## Fitting holds CEP's route

`probes/stability.py` is a short continuation from Logbook 129's passing answer, run as description on discarded seeds. It asks whether fitting keeps the on-food reversal handle that the answer's last, hand-made step relied on: CEP's food-contact gain, raised from 0.41 to 1.21.

- **The search:** sep-CMA at σ₀ 0.02, on Logbook 129's objective, at population 48 and 200 animals, for 25 generations.
- **The rule, fixed before the run:** fitting holds the route if at generation 25 the on-lawn onset rate is inside its band and CEP's gain keeps at least half the nudge, so at least 0.81.

| Readout | D | CEP gain | What fails |
|---|---|---|---|
| 0 | 1.65 | 1.21 | nothing |
| 5 | 1.66 | 1.12 | the roaming speed p90 (−3.47); on-lawn onsets (−3.13) |
| 10 | 1.59 | 1.08 | p90 (−3.14); onsets (−3.12) |
| 15 | 1.59 | 1.10 | the small-lawn roaming bout (+3.26); onsets (−3.21) |
| 20 | 1.51 | 1.10 | p90 (−3.24) |
| 25 | 1.51 | 1.13 | the roaming bout (+3.47) |

- **The route holds:** at generation 25 onsets pass and CEP's gain is 1.13. Every mutant passed at every readout.
- **The answer sits at a frontier.** Three statistics hover around their bars, and the fit trades among them: the roaming speed p90, on-lawn onsets and the small-lawn roaming bout. A registered fit's pass will be borderline, so the next change's readiness pilot measures how often a fit lands inside.

## Implications

**The next change is the fit and its registration:**

- `c2_fit.py`, with the objective (the six families plus the three mutants at half the objective), both guards through the forward observer, readouts, stop rules, resumable state and the final grade;
- the clean, staged start from the graded box's centre, where every sensory gain is 0, so its first stage must fit the senses;
- the D37 amendment: the wild type plus Scheer's three mutants fitted, with held-out cohorts and one tax-4 leave-one-out;
- a readiness pilot first, then F.0b-ii, sized on this procedure;
- the compute for W.

## Seeds

All discarded:

- **The stability check:** fit animals 800,000,000 + 200·k + i, readouts 801,000,000 + i, CMA 9489.
- **The forward parity's reference episodes:** 99,860,000.
- **The five-population grade:** Logbook 129's own 795,000,000 + 200·g + i.
