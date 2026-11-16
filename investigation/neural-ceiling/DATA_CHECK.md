# Data check: controlled inputs with neural activity and behaviour

Drafted 2026-11-15 for launch record 135. It is desk work from `investigation/DATA_INVENTORY.md`:
every fact below comes from that document and its sources, and nothing was downloaded. It asks one
question: which datasets pair a **controlled input** (a stimulus or perturbation delivered at known
times or places) with **neural activity** and **behaviour**, in the same animals? Only those could
train or test a closed-loop learned worm later. Nothing is built now.

## Datasets with all three

| Dataset | Controlled input | Neural coverage | Behaviour | Worms | Access and licence |
|---|---|---|---|---|---|
| Atanas et al. 2023, Cell | a 1-s heat pulse, in 30 of 68 recordings | 105 to 163 neurons; NeuroPAL identities in 40 of 68 | velocity, head curvature, pumping | 68 (38 on DANDI) | DANDI 000776, CC-BY-4.0; 1.01 TB NWB, with byte-range reads possible (SESSION_0.md, G) |
| Kramer et al. 2026, Nat Neurosci | odour boundaries; tdc-1 against wild type | about 102 identified, NeuroPAL | freely moving | 49 (17 tdc-1, 32 wild type) | Dryad, CC0-1.0; 1.46 GB |
| Baskoylu et al. 2026, bioRxiv | a copper boundary | about 96, NeuroPAL | freely moving | 27 | Dryad, CC0-1.0; 320 MB |
| Dag et al. 2023, Cell | food encounter and re-feeding | 128 to 167; NeuroPAL in 8 of 15 | freely moving | 15 | Zenodo and WormWideWeb; licence not stated (a 2023 snapshot is MIT) |
| Ji et al. 2021, eLife | Chrimson in AVA and AVB; thermal; ablations, TeTx, five mutants | targeted only: AFD, AIY, AVA, RIM, RME, SMD | semi-constrained | 3 to 9 per condition | article CC-BY-4.0; source data only |

## Close, but missing one of the three

| Dataset | What it has | What it lacks |
|---|---|---|
| Casademunt et al. 2026, bioRxiv | odour gradients and pulses; 133 to 163 neurons; 41 moving worms | not released |
| Fieseler et al. 2025, bioRxiv | O₂; gcy-9;31;35 and muscle HisCl; 51 moving | not released |
| Pradhan et al. 2025, Nat Commun | moving whole-brain imaging; PA14 infection; flp-13 | an input with known timing (infection is a condition) |
| Kaplan et al. 2020, freely moving | AIB and RIA silencing; lgc-55, tdc-1, unc-25 | coverage: 1 to 3 targeted neurons |
| Venkatachalam et al. 2016 | a thermal sinusoid; about 80 neurons | a deposit |
| Randi et al. 2023; Dunn et al. 2025; Yemini et al. 2021; Seyedolmohadesin et al. 2026 | controlled stimulation or sensory input, with whole-brain activity | behaviour: all are immobilised |

## What this means for a closed-loop learned worm

- **Three licensed sets pair a timed or spatial input with whole-brain activity and free
  behaviour:** Atanas (heat pulse), Kramer (odour boundaries) and Baskoylu (copper boundary).
  Kramer and Baskoylu are small (1.46 GB and 320 MB) and CC0. Atanas is CC-BY but about 1 TB.
- **None of them stimulates single neurons in a moving animal.** The causal single-neuron data
  (Randi, Dunn) are all from immobilised worms. So a learned closed-loop worm would learn sensory
  input to activity to behaviour from the moving sets, and could be checked against single-neuron
  causal structure only from the immobilised ones.
- **Not yet checked:** identity coverage per recording, frame rates, whether the stimulus timing is
  in the files, and the Flavell strain's possible lite-1 and gur-3 nulls, which DATA_INVENTORY.md § 4
  flags for confirmation. A later step would start there.
