# Data inventory: every candidate dataset

Written 2026-11-14. A research agent inspected public data hosts and papers read-only (no files
downloaded); I re-checked the licences that matter most on the hosts. Its full notes, with every URL it
opened and a provenance tag per fact, are in [`notes/data_inventory_raw.md`](notes/data_inventory_raw.md).
This document is the curated version.

**Provenance column (src):** H, the data host's own page or API; P, the paper or its preprint; 2,
second-hand. **Checked** means I re-opened the host myself. "Not stated" means the host shows no
licence, which in law means all rights reserved.

## Summary

- **Perturbation data that are open:** Randi et al. 2023's signal-propagation atlas (113 immobilised
  animals, single-neuron optogenetic stimulation, whole-head nuclear GCaMP6s at 2 Hz) and Dunn et al.
  2025 (Kato lab; AWA, SMD, RIV and RIA stimulation). **Both are on DANDI under CC-BY-4.0**, which I
  checked on the hosts.
- **This corrects a project record.** `data/connectome/PROVENANCE.md` (checked 2026-09-23) found the
  Randi atlas unusable under the project's CC0-or-CC-BY rule because its OSF deposit states no licence
  and `wormneuroatlas` is GPL-3.0. It did not check DANDI 001075, which is CC-BY-4.0 (version
  0.240930.1859, 113 subjects, 223 NWB files, 4.07 TB). Whether the extracted traces sit in the small
  segmentation files (13 to 25 MB each) or only inside the 8 to 60 GB imaging files is not yet known.
- **Silencing and mutant data with whole-brain readout** exist from the Zimmer lab (HisCl silencing of
  AVA, hub neurons, SMD, RIS, motor neurons; npr-1, daf-2), the Flavell lab (tdc-1, flp-13) and the
  Samuel lab (unc-13 on a sensory panel). The Zimmer sets state no licence.
- **Controlled sensory drive:** Yemini 2021 and Seyedolmohadesin 2026 (both CC-BY-4.0 on DANDI), Lin
  2023, and O₂ steps in the Zimmer sets.
- **Freely moving whole-brain data:** Atanas 2023 (CC-BY-4.0 on DANDI), Dag 2023, Kramer 2026,
  Pradhan 2025, Hallinen 2021.
- **Main gaps:**
  1. No open freely moving whole-brain dataset with optogenetic perturbation (Fieseler 2025 is
     unreleased).
  2. Perturbation data are immobilised calcium at 1.6 to 4 volumes/s, under levamisole or tetramisole,
     which act on acetylcholine receptors.
  3. Neuromodulator contrasts with stimulation rest on one arm: Randi's 18 unc-31 animals, processed
     data only, no licence, shorter pulses. No stimulate-and-record data exist in tph-1, pdfr-1 or tax-4
     backgrounds, the project's three mutants.
  4. Open electrophysiology covers about 15 neuron classes, all from dissected preparations.
  5. Licences are missing on many key sets.

## 1. Stimulate one neuron, record the rest

| Dataset | Neurons per animal | Identified | Preparation | Perturbation | Worms | Licence | Size, format | src |
|---|---|---|---|---|---|---|---|---|
| Randi et al. 2023 atlas, wild type | 186 of 188 head neurons pooled; per animal not stated | NeuroPAL, by hand | immobilised (levamisole, beads) | two-photon GUR-3/PRDX-2 activation of one neuron, 0.5 s, every 30 s, up to 40 min | 113 | DANDI 001075: CC-BY-4.0 (**checked**). OSF e2syt exports: not stated | DANDI 4.07 TB, NWB; OSF 0.52 GB text, 1.14 GB full export | H+P |
| Randi 2023, unc-31 arm | not stated | NeuroPAL | immobilised | as above, 0.3-s pulses, no dense-core-vesicle release | 18 | OSF: not stated | 84.9 MB text | H+P |
| `wormneuroatlas` (Randi) | pairwise matrices | from the atlas | derived | per-pair mean and all-trial responses, q, equivalence q, trial counts, fitted kernels, wild type and unc-31 | 113 + 18 | GPL-3.0 | 18.3 MB HDF5 | H |
| Creamer, Leifer and Pillow, derived | 154 to 156 modelled | from the atlas | derived | fitted to 110 wild-type animals | 110 | repository MIT; their reformatted data on OSF qxhjd: not stated | 42 KB weights CSV (vendored); 3 × 20 MB model pickles | H+P |
| Dunn et al. 2025, Curr Biol (Kato lab) | 45 to 140 detected; 0 to 82 identified | NeuroPAL, by hand; many unlabelled | immobilised (tetramisole, chip) | closed-loop AWA drive timed to head-swing phase; single-neuron SMDD, SMDV, RIV, RIA stimulation | 95 NWB subjects | DANDI 001623: CC-BY-4.0 (**checked**); Zenodo CC-BY-4.0 | 774 GB NWB; 0.67 GB pickles | H+P |
| Kato lab SK1, SK2 (DANDI 000565, 000472) | 78 to 180 segmented, 30 to 82 identified | partial NeuroPAL | immobilised | DANDI says none; WormID calls SK1 "single cell optical stimulation" (unresolved) | 21 + 10 | CC-BY-4.0 | 46 + 18 GB NWB | H |

**Mapping to model variables.** The signal is nuclear calcium at 2 Hz, not voltage. A stimulation is
a brief depolarising current into one identified node. Each response is effective: it sums every
path, so a model must be simulated end to end (inject into j, read every i). The data constrain
chemical and gap weights only jointly, plus response timescales and an indicator filter. Randi's
equivalence test gives evidence of absent coupling. Dunn's AWA drive adds a sensory input with known
timing, which the atlas lacks.

**Quality issues** (from the Randi paper unless noted): some neurons are hard to identify (AVB, ADF,
SIA, SIB, RMF; AWCon against AWCoff); a pair counts only if the stimulated neuron responds and both
are identified unambiguously; pairs were sampled 1 to 59 times; the transgenic strain is
"significantly less active than WT"; GUR-3 is over-expressed in every neuron and is itself
light-sensitive (Gauthey et al. 2024, second-hand); Creamer drops each recording's first minute; the
data reproduce themselves at only 0.16 to 0.17 (Creamer).

## 2. Silencing, ablation or mutant backgrounds with population readout

| Dataset | Neurons per animal | Identified | Preparation | Perturbation or contrast | Worms | Licence | Size | src |
|---|---|---|---|---|---|---|---|---|
| Kato et al. 2015 (OSF 2395t) | 107 to 131 | by hand, most active neurons | immobilised (tetramisole, chip) | O₂ steps; AVA::HisCl | 5 + 13 + 5 recordings | not stated | 145 MB .mat | H+P |
| Nichols et al. 2017 (OSF kbf38) | 80 to 130; 20 to 41% identified | by hand | immobilised, L4 | O₂; npr-1 against N2; lethargus | 43 | not stated | 284 MB .mat | H+P |
| Skora et al. 2018 (OSF za3gt) | not stated | by hand, partial | immobilised | O₂; fasted against starved; daf-2 | 6 per condition | not stated | 41 MB .mat | H+P |
| Kaplan et al. 2020, whole brain (OSF 9nfhz) | 113 to 129 head; 103 to 129 whole nervous system | by hand | immobilised | HisCl in RIS, SMD, ventral-cord cholinergic motor neurons | 14 head + 5 whole nervous system | not stated | 252 MB | H+P |
| Kaplan 2020, freely moving (OSF eyj92) | 1 to 3, targeted | yes | moving | AIB and RIA silencing; lgc-55, tdc-1, unc-25 | about 10 to 15 per condition | not stated | 193 MB | H+P |
| Uzel et al. 2022 (OSF 3vkxn) | not verified | by hand | immobilised (inferred) | HisCl in AVA, AVE, AIB, RIB, RIM, PVC, AVB singly and in 3 combinations | 6 wild type; per line not verified | not stated | 365 MB .mat | H |
| Kramer et al. 2026, Nat Neurosci (Dryad) | about 102 identified | NeuroPAL | moving | odour boundaries; tdc-1 (17) against wild type (32) | 49 | CC0-1.0 | 1.46 GB | H+P |
| Pradhan et al. 2025, Nat Commun | 117 to 176; 40 to 121 identified | NeuroPAL | moving | PA14 infection; flp-13 | 8 | CC0-1.0 (Dryad); CC-BY-4.0 (Zenodo) | 38 MB | H+P |
| Lin et al. 2023, Sci Adv (Zenodo 7563053) | 22 sensory neurons | landmarks | immobilised (chip) | 23 odorants and 5 ascarosides; unc-13 | at least 10 per condition | CC-BY-4.0 | 88 MB | H+P |
| Ji et al. 2021, eLife 10:e68848 (Samuel lab) | targeted: AFD, AIY, AVA, RIM, RME, SMD | yes | semi-constrained | ablations, TeTx, five mutants, Chrimson in AVA and AVB, thermal | 3 to 9 per condition | article CC-BY-4.0 | source data | P |
| Susoy et al. 2021, Cell (Mendeley) | tail ganglia, 58 columns | 76 identifiable | moving male | natural partner; ablations | 8 + 14 partial | CC-BY-4.0 | 28 MB CSV | H+P |
| Fieseler et al. 2025, bioRxiv (Zimmer) | 118 to 161 | about 37% | both | gcy-9;31;35; muscle HisCl; O₂ | 51 moving, 28 immobilised | unreleased | | P |
| Madhusudhanan et al. 2026, bioRxiv (Zimmer) | 105 to 185 | 82 types across 9 animals | immobilised | O₂ × 16 trials; npr-1 | 9 | not found | | P |

**Mapping.** A HisCl line is an acute chloride shunt on named neurons; a mutant removes a transmitter,
receptor or release pathway; lethargus and starvation are global neuromodulatory states. These
constrain coupling to and from the silenced nodes, global gain and state parameters, but only through
correlational structure. Nothing here is a stimulus-locked causal response like the atlas's.
**Quality:** low identified fractions (20 to 67%); Kato reports that immobilisation itself changes the
dynamics; O₂ modulation of reversals was weaker under imaging than in behaviour (Skora).

## 3. Controlled sensory input, immobilised

| Dataset | Neurons per animal | Identified | Stimulus | Worms | Licence | Size | src |
|---|---|---|---|---|---|---|---|
| Yemini et al. 2021, Cell (DANDI 000541) | 166 to 188 segmented | NeuroPAL, about 86% automatic | NaCl, 2-butanone, 2,3-pentanedione pulses | 21 | CC-BY-4.0 | 30.5 GB NWB | H+P |
| Seyedolmohadesin et al. 2026, Curr Biol (DANDI 000981) | 196 sex-shared + 6 male-specific | NeuroPAL, by hand | 8 cues × 3 repeats; both sexes | 22 + 25 | CC-BY-4.0 | 82.3 GB NWB | H+P |
| Rusu et al. 2026, eLife | 50 to 100 | NeuroPAL for 4 neurons | random bacterial pulses | 7 | not stated | 51 GB raw TIFF | H+P |
| Pritz et al. 2023, eLife (OSF) | up to 16 classes | by hand | four conditioning protocols | 8 to 18 per condition | not stated | | H+P |
| Casademunt et al. 2026, bioRxiv (Samuel) | 133 to 163 | NeuroPAL, median 102 | odour gradients and pulses | 41 moving | not released | | P |

**Mapping.** Known sensory inputs at known times: these constrain sensory gains and the propagation
from sensory neurons inward, which the atlas cannot (it has no sensory stimulus).

## 4. Freely moving or spontaneous whole-brain imaging

| Dataset | Neurons per animal | Identified | Behaviour alongside | Worms | Licence | Size | src |
|---|---|---|---|---|---|---|---|
| Atanas et al. 2023, Cell (DANDI 000776; WormWideWeb) | 105 to 163 | NeuroPAL in 40 of 68 | velocity, head curvature, pumping; a 1-s heat pulse in 30 | 68 recordings (38 on DANDI) | DANDI CC-BY-4.0 | 1.01 TB NWB; 33 GB Zenodo | H+P |
| Dag et al. 2023, Cell (Zenodo; WormWideWeb) | 128 to 167 | NeuroPAL in 8 of 15 | food encounter, re-feeding; mutants behaviour-only | 15 | not stated (2023 snapshot MIT) | 75 MB | H+P |
| WormWideWeb portal (Flavell lab) | 105 to 176 | NeuroPAL in 56 of 91 | as its source papers | 91 recordings | not stated | 34 to 569 MB per paper | H |
| Baskoylu et al. 2026, bioRxiv | about 96 | NeuroPAL | copper boundary | 27 | CC0-1.0 | 320 MB | H+P |
| Kimura lab, DANDI 000692 | 149 to 163 | nearly all | none (spontaneous, semi-restricted) | 9 | CC-BY-4.0 | 80.5 GB NWB | H |
| Hallinen et al. 2021, eLife | median 121 | AVA only | locomotion | 11 + 11 GFP | node not stated | 4.4 GB | H+P |
| Nguyen et al. 2016, PNAS; 2017 tracking set | 56 to 156 | none | locomotion | 4; 2 | not stated; CC-BY-4.0 (IEEE, login) | about 0.5 TB; 328 GB | P+2 |
| Venkatachalam et al. 2016, PNAS | about 80 | 8 named | thermal sinusoid | 4 | no deposit found | | P |

**Mapping.** Spontaneous, closed-loop activity with behaviour as covariates. It constrains per-neuron
effective timescales (CePNEM ships its posteriors), the readout from neurons to motor variables, state
dependence, and correlational structure, but gives no causal constraint on individual weights.
**Quality:** mis-registration about 0.3% of frames and 2.1% false encoding in GFP controls (Atanas);
the Flavell imaging strain may carry lite-1 and gur-3 nulls (from Kramer's and Baskoylu's methods, to
confirm); head-angle sign flips per dataset in the raw HDF5.

## 5. Pooled corpora and identification data

- **WormID / Sprague et al. 2025, Cell Rep Methods (H+P).** 118 whole-brain datasets from five labs in
  NWB, with a cross-lab neuron-identification atlas and error estimates. CC-BY-4.0 per dandiset.
- **Simeon et al., Hugging Face (H+P).** About 919 worms from 12 studies, z-scored and resampled to a
  common rate (the sources disagree between versions on which), with no stimulus timing; the card
  says MIT, source terms not checked.
- **Atanas et al. 2026, eLife (H+P).** NeuroPAL labelling training data, 516 GB, CC0-1.0.
- Also: fDNC (Yu et al. 2021, OSF, not stated); the TWISP strain characterisation (Sharma et al. 2024,
  Figshare, CC-BY-4.0); light-evoked dynamics (Gauthey et al. 2024, OSF, not stated).

## 6. Electrophysiology

All are dissected single-cell recordings.

| Dataset | Cells | Licence | src |
|---|---|---|---|
| Liu et al. 2018, Cell (Mendeley tngf9w3pgd) | AWA (16 I-V), RIM (3), AIY (7), AFD (3) | CC-BY-4.0, 809 MB | H+P |
| ElectroPhysiomeGAN recordings (Kim et al., eLife reviewed preprint) | 9 classes | CC-BY-4.0, 38 MB | H+P |
| Naudin et al. 2022 (Mendeley) | RIM, AIY, AFD | CC-BY-4.0 | H+P |
| Jiang et al. 2022 (Mendeley) | AVL, DVB, RIS, DVC | CC-BY-4.0, 159 MB | H+P |
| Liu, Chen and Wang 2020, Nat Commun (source data) | AVA, AVB, VA5, VB6, VD5, DD3 | source data not stated | P |
| Liu et al. 2017 (AVA to VA5 rectifying gap junctions); Lindsay et al. 2011 (ASH to AVA) | paired or single | figures only | P |
| Nicoletti 2019 and 2024 models (ModelDB); OpenWorm ChannelWorm and c302 | models | code; licences vary | H |

**Mapping.** Membrane parameters (capacitance, input resistance), I-V curves and channel kinetics for
about 15 classes; edge-level facts (signs, rectification) for a few synapses, as figures. Nothing here
is measured in an intact network.

## 7. Expression and modulation priors

| Dataset | Coverage | Licence | src |
|---|---|---|---|
| CeNGEN L4 (Taylor et al. 2021, Cell) | 118 classes, 128 types; 100,955 cells; four calibrated thresholds | cengen.org not stated; R package GPL-3.0 | H+P |
| CeNGEN L1, adult hermaphrodite, adult male (2024 to 2026) | all neurons per stage | site not stated; L1 preprint CC-BY-4.0; files not found | H+P |
| Ripoll-Sánchez et al. 2023, Neuron, and NemaMod | 302 neurons, 92 peptide-receptor couples, three distance ranges | figshare CC-BY-4.0 (vendored by the project) | H+P |
| Beets et al. 2023, Cell Rep | 161 GPCRs × 344 peptides; 459 to 461 couples with EC50s | article CC-BY; full data "upon request" | P |
| Bentley et al. 2016, PLoS Comput Biol | monoamine 1,940 edges; peptide 7,035 | CC-BY (vendored) | P |
| Wang et al. 2024, eLife, transmitter atlas | all neurons, both sexes | CC-BY-4.0 (vendored) | P |
| Bhattacharya et al. 2019, Cell, innexin atlas | 118 classes × 14 innexins, L1 to adult and dauer | not verified | P |
| Fenyves et al. 2020, PLoS Comput Biol | polarity for 3,638 chemical connections | CC-BY (vendored) | P |
| Yemini et al. 2021, metabotropic receptor map; Hardege et al. 2023, ACh-gated anion channels | all neurons × 8 receptor genes; about 16 channel genes | not verified; CC-BY-4.0 | P |

**Mapping.** These set which channels, receptors and innexins a neuron has, and so which signs and
couplings are possible; they do not set magnitudes. The innexin atlas is the only expression prior
on gap junctions found, and the project does not yet use it.

## 8. Wiring

White 1986; Varshney 2011 (GitHub, no licence); Cook 2019 (wormwiring.org, no licence stated;
vendored through OpenWorm's MIT mirror) and Emmons 2024 (the same matrices under CC-BY-4.0, vendored);
Witvliet 2021 (8 animals, L1 to adult; nerve ring); Brittin 2021 contact maps (Zenodo, CC-BY-4.0 with
a conflicting README); Cook 2025 *P. pacificus* (head, CC-BY-4.0); Yim 2024 dauer (CC-BY-4.0); Choe
2026 dauer gap-junction preprint (not verified); OpenWorm `cect` readers (MIT). No new adult
hermaphrodite connectome from 2024 to 2026 was found, and Witvliet, Cook 2025 and Yim omit gap
junctions, so adult gap-junction counts still rest on White, Varshney and Cook 2019.

## 9. Behaviour datasets already in the project

None of these carries neural recordings. All are vendored with checksums under `data/`.

| Dataset | What was measured | Animals | Licence | Use in the project |
|---|---|---|---|---|
| Scheer and Bargmann 2023, eLife (Dryad, Zenodo) | positions and per-frame flags on small lawns, 40 min at 3 frames/s; wild type (PD1074, N2), uniform lawn, tph-1, pdfr-1, tax-4 | 2,847 across 8 arms (tph-1 139, pdfr-1 395, tax-4 499; uniform lawn 100) | CC0-1.0 | benchmark bands, held-out cohorts, mutants, on-food reversal kinematics, tracker noise, roaming-dwelling HMM |
| Calhoun, Chalasani and Sharpee 2014, eLife (Dryad) | off-food search tracks, 30 min at 3 frames/s, pixels with no scale | 47 | CC0-1.0 | reorientation detector |
| Roberts et al. 2016, eLife (Dryad) | off-food tracks at 29.97 frames/s with forward, reverse and pause states; ablation cohorts not vendored | 122 wild type | CC0-1.0 | detector calibration, reversal share |
| López-Cruz et al. 2019, Neuron (Mendeley) | reorientations per 2-min bin after food removal, by genotype | 184 rows; wild type n = 413 in the reference | CC-BY-4.0 | local-search reference (γ = 0.114 per min) |
| Chen et al. 2025, PLoS Biol (figshare) | naive N2 butanone chemotaxis, 30 min at 14 frames/s, with the measured odour landscape | 8 recordings (6 standard), 10,717 fragments | CC-BY-4.0 | chemotaxis family |
| Ji et al. 2021, eLife 10:e62889 (Flavell; Dryad) | fractions of time roaming by genotype | about one value per animal | CC0-1.0 | mutant directions only |
| Locomotor kinematics (Zhao 2003, Pierce-Shimomura 1999, Broekmans 2016, Gray 2005, Ben Arous 2009) | published off-food values | | literature | primitives |
| Posture: WormPose eigenworms; OIST shapes | eigenworm basis; real postures | | BSD-3; CC-BY-4.0 | body checks |
| Chemotaxis and thermotaxis signatures | literature indices and bias directions | | literature | Phase 6 validations |

The Flavell lab's Ji 2021 also recorded 10 neurons in freely moving animals; the project vendors only
the behaviour readings. Those recordings are a candidate bridge between the neural and behaviour
datasets (section 4's freely moving sets are the others).

## 10. How the data map to model variables: indicators and the voltage link

| Indicator | Datasets | Rate | Kinetics in worm neurons |
|---|---|---|---|
| nuclear GCaMP6s | Randi 2023, Dunn 2025, Yemini 2021 | 2 volumes/s (Randi) | no worm figure for 6s found; slower than 6f |
| nuclear GCaMP7f | Atanas 2023, Dag 2023, Kramer, Pradhan | about 1.7 Hz | Atanas states half-decay under 1 s |
| nuclear GCaMP5K | Kato 2015, Skora 2018 | 1.6 to 2.9 volumes/s | not found |
| nuclear GCaMP6f | Kaplan 2020 | 1.6 to 3.0 volumes/s | decay half-time 2.7 ± 2.0 s in ALM after mechanical stimulation (Ding et al. 2023) |
| GCaMP6m | Ji 2021 (Flavell) | about 2 Hz | not found |

- **Voltage to calcium.** Most neurons are graded. In AWA, the one neuron with simultaneous patch
  clamp and calcium, a graded depolarisation appears in fluorescence after about 1 s (Liu 2018). Brief
  voltage events in RIM were absent from calcium (Azimi Hashemi 2019).
- **Consequences for a fit.** The observation model needs a per-indicator kernel and per-neuron
  scale, offset and possibly saturation; sampling is about 2 Hz against the closed loop's 1-s step;
  nuclear calcium misses compartmental signals; immobilised dynamics differ from moving ones (Kato
  2015). Fast or voltage-level dynamics are unconstrained by any of these datasets.

## 11. Licences at a glance

- **CC-BY-4.0 or CC0, usable under the project's rule:** Randi 2023 raw (DANDI 001075), Dunn 2025
  (DANDI 001623), Yemini 2021, Seyedolmohadesin 2026, Atanas 2023 (DANDI), Kimura 000692, Kato lab
  000565 and 000472, Kramer 2026, Pradhan 2025, Baskoylu 2026, Lin 2023, Susoy 2021, the Mendeley
  electrophysiology sets, Ripoll-Sánchez, Bentley, Wang, Fenyves, Emmons 2024, and every behaviour
  dataset the project vendors.
- **Not stated (all rights reserved by default):** every Zimmer-lab OSF project (Kato 2015, Nichols,
  Skora, Kaplan, Uzel), Randi's OSF exports and the unc-31 arm, Creamer's reformatted data, Dag 2023's
  latest Zenodo version, WormWideWeb, Hallinen's node, Rusu, Pritz, cengen.org, wormwiring.org.
- **Copyleft:** `wormneuroatlas` (GPL-3.0), CeNGEN's R package (GPL-3.0).
- **On request only:** Beets 2023's full EC50 table.

A local, unvendored analysis of a "not stated" dataset is a different question from vendoring it; that
call is the maintainer's, and the spike plan avoids needing it.
