# C. elegans neural-activity, perturbation, electrophysiology, expression and wiring datasets: raw inventory

Compiled 2026-11-14 for the question of fitting a connectome-constrained model directly to neural recordings, especially perturbation data (stimulate one neuron, record the rest). Behaviour datasets the project already vendors are out of scope. Behaviour-only optogenetics from the Leifer lab is listed for completeness only.

## How this was compiled, and how to read it

- **Method.** Web search and web fetch only. Public data-host pages and their JSON APIs were inspected read-only: DANDI, OSF, Zenodo, Figshare, Dryad, Mendeley Data, Harvard Dataverse, Hugging Face, GitHub and PyPI. No data file was downloaded, and no local repository was touched. Only small README, readme-RTF and metadata files were opened.

  - The work ran in five parallel passes (sections A–E). Each pass listed every URL it opened in its entries. The merge then spot-checked these against the hosts again and found them consistent:
    - DANDI 001075, 001623 and 000776: name, licence, bytes, files and subjects.
    - OSF e2syt and 2395t: licence fields null.
    - PyPI and GitHub licences of `wormneuroatlas` and of the Creamer repository.
    - Zenodo 7563053 and 17353307: licences and file sizes.
    - The WormWideWeb home page.

- **Provenance tags,** used in every entry:

  - **[host]**: the data host's own page or API.
  - **[paper]**: the article, or its preprint where so marked: methods, legends, data-availability statement.
  - **[registry]**: Crossref or Europe PMC metadata.
  - **[2nd]**: second-hand, such as a search snippet, a citing paper, a portal describing someone else's data, or a third-party mirror.
  - "Not verified" means the page holding the fact was not opened, or could not be opened. "Not found" means the pages that were opened did not have it.

- **Rules applied.**

  - Licences are copied as the host shows them. "Not stated" means the host's licence field is empty or absent. Paper licences are recorded separately and are not the data licence.
  - Sizes are the host's byte counts.
  - Where the paper's worm count and the host's file or subject count differ, both are given.

- **Pages that blocked access**, and so left gaps:

  - cell.com and sciencedirect.com (HTTP 403). So the final Cell texts of Atanas 2023, Dag 2023 and Uzel 2022 were not read. Methods for Atanas, Dag, Yemini 2021, Dunn 2025 and Seyedolmohadesin 2026 come from their bioRxiv preprints.
  - pmc.ncbi.nlm.nih.gov, intermittently (browser check).
  - nature.com supplements.
  - royalsocietypublishing.org.
  - worm.peptide-gpcr.org.
  - wormatlas.org (TLS certificate errors).

- **Search budget.** The session's shared web-search allowance (200 calls) ran out near the end. Searches still pending at that point:

  - a data deposit for Venkatachalam 2016;
  - Uzel 2022's methods;
  - published versions of the 2025–2026 Zimmer preprints;
  - a final sweep for 2025–2026 adult connectomes and receptor atlases.

  Coverage of the newest items may therefore be incomplete. A follow-up session can close these. Every entry records what was and was not checked.

- **Layout.** Section 0 is the consolidated table, and section 0b is a reading guide for the fitting question. Then come the full entries:

  - A: Leifer lab and the Creamer/Leifer/Pillow model.
  - B: Flavell lab, WormWideWeb, NeuroPAL and DANDI.
  - C: Zimmer and Samuel labs.
  - D: other labs, 2024–2026, and electrophysiology.
  - E: expression, modulation and wiring.

  Datasets that two passes both covered (Randi 2023, Dunn 2025, Seyedolmohadesin 2026, Kimura DANDI 000692) have one full entry; the other place holds a pointer, and facts found only by the second pass were folded into the full entry and labelled as such.

______________________________________________________________________

## 0. Consolidated summary table

The "src" column gives the main provenance of the entry: H = data host, P = paper, p = preprint only, 2 = second-hand, R = registry. "seg." = segmented, "ID'd" = identified. Sizes are as the host lists them.

### 0.1 Targeted stimulation of identified neurons with population readout

| dataset (entry) | neurons/animal | identified | immobilised / moving | perturbation | worms | licence | size | format | access | src |
|---|---|---|---|---|---|---|---|---|---|---|
| Randi 2023 signal propagation atlas, WT (A1) | per-animal not stated; 186/188 head neurons pooled; 23,433 pairs measured | NeuroPAL, manual | immobilised (levamisole + beads) | 2-photon GUR-3/PRDX-2 single-neuron activation, 0.5 s, one every 30 s, up to 40 min | 113 | DANDI CC-BY-4.0; OSF not stated | DANDI 4.07 TB (223 NWB); OSF 0.52 GB txt / 1.14 GB full export | NWB; txt and Python exports (tar.gz) | open, no request | H+P |
| Randi 2023, unc-31 arm (A2) | not stated | NeuroPAL, manual | immobilised | as A1 but 0.3 s pulses; unc-31 (no dense-core-vesicle release) | 18 | OSF not stated | 84.9 MB | txt in tar.gz | open (processed only; raw on a host not found) | H+P |
| `wormneuroatlas` / `worm-functional-connectivity` (A3) | pairwise matrices (atlas IDs) | from A1 | derived | WT and unc-31 kernels, dFF, q, q_eq, trial counts | 113 + 18 (derived) | GPL-3.0 (wormneuroatlas; PyPI classifier GPLv3+); wfc not stated | funatlas.h5 18.3 MB; wheel 37.8 MB; pickles 2.1 / 0.6 MB | HDF5, pickle | open (pip, GitHub) | H |
| Creamer, Leifer & Pillow connectome-constrained LDS, fitted to Randi WT (A4) | 156 modelled (README: 154) | from A1 | derived | stimulus as one-hot input with learned 22.5-s filter | 110 (80 train / 30 test) | repo MIT; OSF data not stated; preprint CC-BY-NC 4.0 | 3 × 20.1 MB model pkl, 42 KB weights CSV; data zip 161.7 MB | pickle, CSV | open (GitHub + OSF) | H+P |
| Dunn 2025, Kato lab (D; DANDI 001623, Zenodo 17353307) | 45–140 detected (preprint); 43–139 seg., 0–82 ID'd (WormID) | NeuroPAL, manual; many recordings unlabelled | immobilised (tetramisole, chip); behaviour in separate free runs | closed-loop optogenetic AWA drive timed to head-swing phase; single-neuron SMDD, SMDV, RIV, RIA (nrV/nrD) stimulation | 95 NWB subjects; paper total not stated | CC-BY-4.0 (DANDI and Zenodo) | 774.2 GB (95 NWB); 0.67 GB zip + 95 pkl | NWB; pkl, CSV | open | H+p |
| Kato lab SK1 / SK2 (B; DANDI 000565, 000472) | 78–180 seg.; 30–82 ID'd | NeuroPAL, partial (30–50%) | immobilised | DANDI says none; WormID calls SK1 "single cell optical stimulation" (unresolved) | 21 + 10 | CC-BY-4.0 | 46.3 GB + 18.4 GB | NWB | open | H |

### 0.2 Whole-brain or multi-neuron imaging with silencing, ablation or mutant backgrounds

| dataset (entry) | neurons/animal | identified | immobilised / moving | perturbation | worms | licence | size | format | access | src |
|---|---|---|---|---|---|---|---|---|---|---|
| Kato 2015 whole-brain (C; OSF 2395t) | 107–131 detected (55–67% of expected) | manual; fraction not stated | immobilised (1 mM tetramisole, chip) | O2 steps 21↔4%; AVA::HisCl | 5 spont. + 13 O2 + 5 AVA::HisCl recordings | not stated (OSF) | 145.1 MB (4 files) | MATLAB .mat | open | H+P |
| Nichols 2017 sleep (C; OSF kbf38) | 80–130 | manual, 20–41% | immobilised (chip) | O2 10↔21%; npr-1 vs N2; lethargus vs prelethargus | 43 recordings | not stated (OSF) | 284.3 MB (5 files) | .mat | open | H+P |
| Skora 2018 starvation (C; OSF za3gt) | not stated | manual | immobilised (tetramisole) | O2 21↔10%; fasted vs starved; daf-2 | n = 6 per condition | not stated (OSF) | 40.6 MB (4 files) | .mat | open | H+P |
| Kaplan 2020 whole-brain (C; OSF 9nfhz) | 113–129 head; 103–129 whole nervous system | manual | immobilised (tetramisole + histamine) | HisCl: RIS (background), SMD, ventral-cord cholinergic MNs | 5 + 5 + 4 head; 5 whole-NS | not stated (OSF) | 252.4 MB (zip) | zip (.mat fields per readme) | open | H+P |
| Uzel 2022 hub silencing (C; OSF 3vkxn) | not verified | manual (IDs field) | immobilised (inferred, not verified) | HisCl in AVA, AVE, AIB, RIB, RIM, PVC, AVB singly + 3 combinations | 6 WT (code comment); per-line n not verified | not stated (OSF) | 364.9 MB (12 files) | .mat | open | H (methods not read) |
| Kaplan 2020 freely moving (C; OSF eyj92) | 1–3 targeted | yes (targeted promoters) | freely moving | AIB/RIA HisCl, RIA TeTx, lgc-55, tdc-1, unc-25 | ~10–15 per condition | not stated (OSF) | 193.2 MB | movies + .mat | open | H+P |
| Kramer 2026 tyramine turning (B; Dryad 8sf7m0cz2) | ~102 ID'd | NeuroPAL | freely moving | odour boundaries; tdc-1 mutant imaged (17) vs WT (32) | 49 | CC0-1.0 | 1.46 GB | HDF5, .mat | open | H+P |
| Pradhan 2025 sickness (B; Dryad nvx0k6f2t, Zenodo 19511989) | 117–176 | NeuroPAL, 40–121 ID'd | freely moving | PA14 infection; flp-13 background | 8 | Dryad CC0-1.0; Zenodo cc-by-4.0 | 37.7 MB | HDF5, JSON | open | H+P |
| Lin 2023 odour panel (C; Zenodo 7563053) | 22 sensory neurons (11 pairs) | yes (landmarks) | immobilised (chip) | 23 odorants + 5 ascarosides; unc-13 | ≥10 per condition; total not stated | CC BY 4.0 (data); MIT (code) | 87.8 MB (2 zips) | zip (inner format not verified) | open | H+P |
| Ji 2021 corollary discharge (C) | targeted: AFD, AIY, AVA, RIM, RME, SMD | yes (promoters) | semi-constrained (agarose + coverslip) | miniSOG ablations, TeTx, eat-4/tdc-1/cat-1/unc-25/unc-31, Chrimson in AVA/AVB, thermal | 3–9 per imaging condition | CC BY 4.0 (article) | not verified | eLife source data | with article | P |
| Susoy 2021 male mating (C; Mendeley Data) | ~all tail-ganglion neurons; 58 columns deposited | 76 identifiable across datasets | freely moving (male) | natural partner; laser ablations; let-23, ceh-14 | 8 main + 14 partial deposited (23 ablated in paper) | CC BY 4.0 | 28.1 MB (3 files) | CSV | open | H+P |
| Fieseler 2025 preprint (C) | ~118–161 tracked | ~37% (manual + NeuroPAL reference) | both | gcy-9;31;35; muscle HisCl; O2 (immobilised) | 36+8+7 moving; 28 immobilised | n/a (unreleased) | n/a | NWB planned (not verified) | not public ("upon publication") | p |
| Madhusudhanan 2026 preprint (C) | 105–185 | 82 types across 9 animals | immobilised (muscle HisCl) | O2 11↔21% × 16 trials; npr-1 background | 9 | not found | not found | not found | not found | p |

### 0.3 Sensory-stimulus panels and learning (immobilised)

| dataset (entry) | neurons/animal | identified | immobilised / moving | perturbation / stimulus | worms | licence | size | format | access | src |
|---|---|---|---|---|---|---|---|---|---|---|
| Yemini 2021 NeuroPAL activity (B; DANDI 000541, Zenodo 3906530) | 166–188 seg., 164–184 ID'd (WormID) | NeuroPAL (~86% automatic in head) | immobilised (tetramisole, chip) | NaCl, 2-butanone, 2,3-pentanedione pulses | 21 | DANDI CC-BY-4.0; Zenodo mit-license | 30.5 GB; 1.1 GB | NWB; zip | open | H+p |
| Seyedolmohadesin 2026 both sexes (D; DANDI 000981) | 196 sex-shared + 6 male-specific head neurons in the set | NeuroPAL, manual | immobilised (1 mM levamisole, chip; small movements remain) | 8 cues (food, ascr#3, osmotic, Cu²⁺, IAA ×2, NaCl ×2) × 3 repeats; sex | 22 herm + 25 male | CC-BY-4.0 | 82.3 GB (47 NWB) | NWB | open | H+p |
| Rusu 2026 random bacterial pulses (D) | 50–100 extracted | NeuroPAL for AVA/RME/SMDD/SMDV | immobilised (tetramisole, nose-only chip) | random on/off bacterial pulses | 7 stimulated | not stated | 51 GB tar | raw TIFF | open (HTTP, Salk server) | H+P |
| Pritz 2023 associative learning (D; OSF) | up to 16 classes | manual (anatomy + reporters) | immobilised (levamisole) | 4 conditioning protocols (short/long-term, appetitive/aversive) | ~8–18 per condition | not stated (OSF null) | not shown | OSF folders | open | H+P |
| Casademunt 2026 preprint (C) | 133–163 tracked (immobilised) | NeuroPAL: median 102 (immobilised), 64 (moving) | both | odour gradients (DA, IAA); DA pulses | 41 moving; immobilised n unresolved | not stated | n/a | n/a | viewer only; download "coming soon" | p+H |

### 0.4 Freely moving or spontaneous whole-brain imaging

| dataset (entry) | neurons/animal | identified | immobilised / moving | perturbation / stimulus | worms | licence | size | format | access | src |
|---|---|---|---|---|---|---|---|---|---|---|
| Atanas 2023 (B; DANDI 000776, Zenodo 8150514, WWW) | 105–163 (WWW); ~143 ± 12 (preprint) | NeuroPAL in 40/68 (WWW); ~96 ± 14 ID'd (preprint) | freely moving | 1-s heat pulse (30 recordings); GFP motion controls | 68 recordings (WWW); 38 on DANDI | DANDI CC-BY-4.0; Zenodo latest none shown (v1–3 cc-by-3.0-us) | DANDI 1.01 TB; Zenodo 33.0 GB | NWB; HDF5, JLD2; JSON | open | H+p |
| Dag 2023 serotonergic system (B; Zenodo 19414639, WWW) | 128–167 | NeuroPAL in 8/15; 87–110 labelled | freely moving | food-patch encounter, re-feeding; tph-1, 5-HT receptor mutants and NSM::Chrimson behaviour-only (per preprint) | 15 recordings | Zenodo not stated; 2023 snapshot mit-license | 74.6 MB | HDF5, JLD2, JSON | open | H+p |
| WormWideWeb portal (B) | 105–176 | NeuroPAL in 56/91 | freely moving | as source papers (Atanas, Dag, Pradhan) | 91 recordings, 3 Flavell papers | not stated | 34–569 MB per paper | HDF5, JSON | open, no registration | H |
| Baskoylu 2026 preprint (B; Dryad w9ghx3g4v) | ~96 | NeuroPAL | freely moving | copper boundary; perturbations behaviour-only | 27 | CC0-1.0 | 320 MB | not verified | open | H+p |
| Kimura lab DANDI 000692 (D) | 149–163 seg., ~all ID'd (WormID) | NeuroPAL Auto-ID + manual | semi-restricted (chip) | none (spontaneous) | 9 | CC-BY-4.0 | 80.5 GB (9 NWB) | NWB | open | H |
| Nguyen 2016 (A) | 56–77 | none (candidate IDs only) | freely moving | none | 4 (+1 GFP) | not stated | raw ~0.5 TB; supplement not verified | PNAS SI; raw on S3 | SI public; raw S3 requester-pays | P+2 |
| Hallinen 2021 (A) | median 121 ± 12 | AVA only | freely moving (+ immobilisation transitions) | none (levamisole transition) | 11 GCaMP + 11 GFP + 4 immobilisation | OSF node not stated; 2020 registration CC0 1.0 | 4.39 GB | tar.gz | open | H+P |
| Nguyen 2017 NeRVE tracking set (A10) | up to 156 tracked | none | freely moving | none | 2 | CC BY 4.0 (page link) | 328.01 GB | .dat/AVI + MAT | IEEE DataPort login; subscription wording ambiguous | H |
| Venkatachalam 2016 (C) | ~80 | 8 named | freely moving | thermal sinusoid 15–17.5 °C | 4 (+1 GFP) | not found | not found | not found | no deposit found | P |
| Kato 2015 freely moving (C; OSF cqmnv) | 1 per worm | yes (targeted) | freely moving | none | not stated | not stated (OSF) | 23.8 MB | .mat | open | H+P |

### 0.5 Identification, tracking and pooled corpora

| dataset (entry) | neurons/animal | identified | immobilised / moving | perturbation | worms | licence | size | format | access | src |
|---|---|---|---|---|---|---|---|---|---|---|
| WormID / Sprague 2025 NWB corpus (B) | 43–196 seg. | NeuroPAL, 0–193 ID'd | mixed | as source datasets | 118 (paper) + 95 (001623) | per dandiset CC-BY-4.0; paper CC BY-NC-ND | see DANDI table in B | NWB | open | H+P |
| DANDI 000714 / 000715, structural NeuroPAL (B) | n/a | NeuroPAL | n/a | none | 9 / 10 | CC-BY-4.0 | 0.48 GB / 0.84 GB | NWB | open | H |
| Atanas 2026 eLife cell-ID training data (B; Dataverse 8UE0H9) | n/a | NeuroPAL labels (98% reported) | mixed | none | 81 + 10 + 11 annotated images | CC0 1.0 | 516 GB compressed | .tar.zst | open | H+P |
| Yu 2021 fDNC (A7) | ~134 seg., 57.5 labelled | NeuroPAL, manual | immobilised | none | 11 (one volume each) | not stated (OSF) | ~0.86 GB | zip, model | open | H+P |
| Sharma 2024 TWISP strain (A8) | not stated | NeuroPAL | immobilised + plates | opsin cross-talk tests | not stated | CC BY 4.0 (Figshare) | 758.4 MB | zip by figure | open | H+P |
| Gauthey 2024 light-evoked dynamics (A9) | not verified | not verified | not verified | imaging light; gur-3 (2nd-hand) | not verified | not stated (OSF) | 0.50 GB | tar.gz | open | H+2 |
| Simeon et al. pooled corpus (D; Hugging Face) | 1–189 labelled (by source) | from sources | mixed | none encoded in visible columns | ~919, 12 sources | "mit" (HF card; source terms not checked) | 669.5 MB | parquet | open | H+P |

### 0.6 Behaviour-only optogenetics, Leifer lab (no neural recording; A10)

| dataset | neurons/animal | identified | immobilised / moving | perturbation | worms | licence | size | format | access | src |
|---|---|---|---|---|---|---|---|---|---|---|
| Liu 2022 closed-loop TRN optogenetics | 0 | n/a | moving | Chrimson in 6 TRNs | 95 + 47 plates | not verified | not verified | not verified | IEEE DataPort (not opened) | P |
| Kumar 2023 TRN/interneuron optogenetics | 0 | n/a | moving | Chrimson; gtACR2 in RIV/SMB/SAA | not stated | CC0 | 337.8 GB | tar | open | H+P |
| Li 2025 TRN subtypes | 0 | n/a | moving | intersectional opsins | see sample-size xlsx | CC BY 4.0 | 199.9 MB | .mat, .xlsx | open | H |
| He 2026 sensory-motor reflex | 0 | n/a | moving | whole-field optogenetics | not verified | CC BY 4.0 | 32.8 MB | zip | open | H |

### 0.7 Electrophysiology (section D, part 2; all dissected preparations)

| dataset | neurons/animal | identified | preparation | perturbation | worms / recordings | licence | size | format | access | src |
|---|---|---|---|---|---|---|---|---|---|---|
| Liu 2018 AWA action potentials (Mendeley tngf9w3pgd) | 1 (patch); I–V survey of RIM, AIY, AFD, AWA | promoter-GFP | dissected | channel mutants, drugs, ion substitution | AWA 16 (I–V); RIM 3; AIY 7; AFD 3 | CC BY 4.0 | 809.4 MB | 15 zip | open | H+p |
| EP-GAN recordings, 9 classes (Mendeley g5kcjp7jsk) | 1 | strain-labelled | dissected | none | not stated | CC BY 4.0 | 37.5 MB | zip | open | H+p |
| Naudin 2022 RIM/AIY/AFD (Mendeley 6pgdf3zrp3) | 1 | strain-labelled | dissected | none | I–V n = 3 / 7 / 3 | CC BY 4.0 | 37.5 MB | txt + xlsx | open | H+P |
| Jiang 2022 AVL/DVB/RIS/DVC (Mendeley 33mnszcbh8) | 1 | strain-labelled | dissected | mutants (voltage clamp) | not checked | CC BY 4.0 | 159.3 MB | 13 zip | open | H+P |
| Liu, Chen & Wang 2020 motor circuit (journal Source Data) | 1 (AVA, AVB, VA5, VB6, VD5, DD3) | strain-labelled | dissected | not checked | not checked | Source Data not stated; article CC BY 4.0 | 311.1 MB | zip (not opened) | open (journal) | P+R |
| Liu 2017 AVA–VA5 rectifying gap junctions | 2 (paired) | strain-labelled | dissected | ChR2 in command INs; unc-7/unc-9 | not checked | article CC BY 4.0 | n/a | figures / SI | figures only | P |
| Lindsay 2011 ASH→AVA | 1–2 | GFP/RFP | nearly intact | ChR2 in ASH; CNQX, MK-801; unc-13 | not checked | not recorded | n/a | figures | figures only | P |
| Nicoletti 2019/2024 HH models (ModelDB 267187, 2017403) | 8 classes modelled | n/a | n/a | in-silico knockouts | n/a | not stated | not shown | XPPAUT; NEURON .mod + Python | open (code) | H+P |
| OpenWorm ChannelWorm / ChannelWorm2 / c302 | n/a | n/a | n/a | n/a | n/a | MIT / not stated / MIT | not shown | spreadsheets, JSON, NeuroML2 | open (code) | H |

### 0.8 Expression and modulation priors (section E; "neurons/animal" = coverage, "worms" = samples)

| dataset | coverage | identified | n/a | perturbation | samples | licence | size | format | access | src |
|---|---|---|---|---|---|---|---|---|---|---|
| CeNGEN L4 scRNA-seq, Taylor 2021 (E1) | 118 classes / 128 types, L4 hermaphrodite | cluster annotation | n/a | n/a | 100,955 cells, 17 experiments | cengen.org not stated; R package GPL-3.0 | RDS 0.3–4 GB; threshold CSV sizes not listed | CSV, RDS, TSV | open (cengen.org/downloads) | H+P |
| CeNGEN L1 / adult hermaphrodite / adult male (E2) | all neurons per stage | class-level | n/a | n/a | 161,072 / 68,923 / 67,346 cells | site not stated; L1 preprint CC BY 4.0 | not stated | Shiny apps; download files not found | apps only | H+p |
| Ripoll-Sánchez 2023 neuropeptide connectome + NemaMod (E3) | 302 neurons, 92 peptide–receptor couples, 3 distance ranges | yes | n/a | n/a | derived (CeNGEN + EC50) | figshare CC BY 4.0; code MIT; NemaMod app not stated | ~78 MB CSV | CSV adjacency matrices | open (figshare c.6895870, GitHub) | H+P |
| Beets 2023 peptide–GPCR EC50 map (E4) | 161 GPCRs × 344 peptides; 459–461 couples | n/a | n/a | n/a | in vitro | PMC states both CC BY 4.0 and CC BY-NC-ND | Data S1 310.5 KB; PDFs 2.6 + 7 MB | ZIP, PDF | supplements; full data "upon request" | P+R |
| Bentley 2016 multilayer connectome (E5) | 302; monoamine 1,940 edges, neuropeptide 7,035 edges | yes | n/a | n/a | n/a | article CC BY | not shown | ZIP edge lists, DOCX | open (PLOS SI) | P |
| Wang 2024 neurotransmitter atlas (E6) | all neurons, hermaphrodite + male | NeuroPAL | n/a | n/a | not stated | article CC BY 4.0 | 5 xlsx, 10–99 KB | XLSX | open | P+R |
| Bhattacharya 2019 innexin atlas (E7) | 118 classes × 14 innexins; L1–adult + dauer | yes | n/a | n/a | not stated | not verified | figure PDFs up to 21 MB | figures, tables | open (PMC) | P |
| Fenyves 2020 synaptic polarity (E8) | 297 neurons, 3,638 chemical connections, 20,589 synapses | yes | n/a | n/a | n/a | article CC BY | not shown | XLSX, DOCX | open (PLOS SI) | P |
| Yemini 2021 metabotropic receptor map (E9) | all neurons × 8 receptor genes | NeuroPAL | n/a | n/a | not stated | not verified | not verified | Table S4 | PMC | P+2 |
| Hardege 2023 ACh-gated anion channels (E10) | ~16 LGC genes | NeuroPAL | n/a | n/a | n/a | CC BY 4.0 | not shown | DOCX, XLSX | open | P |

### 0.9 Wiring (section E)

| dataset | coverage | identified | n/a | perturbation | animals | licence | size | format | access | src |
|---|---|---|---|---|---|---|---|---|---|---|
| White 1986 (W1) | 302 neurons, composite series | yes | n/a | n/a | N2U, JSH, JSE series (2nd-hand) | not stated | not verified | xls, viewers | NemaNode, cect (WormAtlas not reached) | R+2 |
| Varshney 2011 (W2) | whole hermaphrodite; chemical + gap + NMJ | yes | n/a | n/a | composite of ≥3 worms | article CC BY; repo none | not shown | .mat, xls | GitHub lrvarshney/elegans; paper's URL dead | P+H |
| Cook 2019, both sexes (W3) | whole animal incl. end organs | yes | n/a | n/a | composite hermaphrodite + male | not stated (Emmons Lab ©2020) | not listed | XLSX | open (wormwiring.org) | H+2 |
| Witvliet 2021 development (W4) | nerve ring + ventral ganglion | yes | n/a | n/a | 8 hermaphrodites, L1 to adult | not stated | not listed | XLSX; BossDB | open | P+H |
| Brittin 2021 contact maps (W5) | nerve ring | yes | n/a | n/a | 2 (adult N2U, L4 JSH) | Zenodo cc-by-4.0, but README MIT-style (conflict) | ~25 MB analysis; ~17 GB total | SQL, GraphML, NumPy, TrakEM2 | open (Zenodo 10.5281/zenodo.4383276) | H |
| Cook 2025, P. pacificus (W6) | head; nerve-ring comparison | yes (homologues) | n/a | n/a | 2 P. pacificus adults | CC BY 4.0 | not verified | SI tables; BossDB | open | P+R |
| Yim 2024 dauer (W7) | dauer nerve ring | yes | n/a | n/a | 1 | CC BY 4.0 | not shown | XLSX; BossDB | open | P |
| Choe 2026 preprint, dauer vs adult gap junctions (W8) | nociceptive circuit | not verified | n/a | n/a | not verified | not verified | not verified | not verified | not verified | 2 |
| OpenWorm ConnectomeToolbox, `cect` (W9) | readers for ~15 datasets | harmonised names | n/a | n/a | n/a | code MIT; data per source | not shown | Python | open (GitHub, PyPI) | H |

______________________________________________________________________

## 0b. Reading guide for the fitting question

**Stimulate-one, record-the-rest data that are open.**

- **Randi 2023 atlas, wild type (A1).**
  - 113 immobilised animals, nuclear GCaMP6s at 2 volumes/s, NeuroPAL identities.
  - Brief two-photon activation of single neurons, giving 23,433 measured pairs.
  - Raw NWB on DANDI 001075 (CC-BY-4.0, 4.07 TB). Processed exports on OSF e2syt (no licence set).
  - A ready fitting target in `wormneuroatlas` (GPL-3.0): per-pair responses, q values, q_eq evidence of absence, and kernels.
- **Randi unc-31 arm (A2).** 18 animals, 0.3-s pulses, processed only and without a licence. It is the only open stimulate-record contrast for dense-core-vesicle (peptide and monoamine) signalling.
- **Dunn et al. 2025 (D; DANDI 001623, CC-BY-4.0, 774 GB).**
  - The only other open set with targeted single-neuron stimulation during whole-brain imaging: AWA (sensory) and SMDD, SMDV, RIV and RIA (head-swing circuit), in immobilised animals.
  - Which of its 95 recordings carry which stimulation has not been verified.
- **Creamer, Leifer & Pillow (A4).** Not new data, but published, MIT-licensed, signed effective weights on connectome edges fitted to 110 of Randi's animals. Chemical and gap junctions are merged into one weight, and the step is 0.5 s. Usable as an initialisation, prior or baseline.

**Perturbation by silencing, ablation or genotype, with population readout.**

- **Zimmer lab, immobilised (all open on OSF, no licence stated):**
  - Uzel 2022: HisCl silencing of seven hub neurons and three combinations.
  - Kaplan 2020: RIS, SMD and ventral-cord motor-neuron silencing.
  - Kato 2015: AVA::HisCl.
  - Nichols 2017: npr-1.
  - Skora 2018: daf-2.
- **Flavell lab, freely moving:** Kramer 2026, tdc-1 vs wild type (CC0); Pradhan 2025, flp-13 with infection (CC0 / cc-by-4.0).
- **Samuel lab:** Lin 2023, an unc-13 control on an identified sensory panel (CC BY 4.0); Susoy 2021, male ablations (CC BY 4.0). Ji 2021 carries ablations and Chrimson in AVA/AVB, but only targeted neurons, in eLife source data.
- **Counted in section B but not imaged whole-brain (per the preprints):** Dag 2023's tph-1 and receptor mutants and NSM::Chrimson, which were behaviour-only.

**Controlled sensory drive, immobilised (to fit sensory gains and propagation from known inputs).**

- Seyedolmohadesin 2026: 47 animals, 8 cues × 3 repeats, CC-BY-4.0.
- Yemini 2021: 21 animals, 3 chemicals, CC-BY-4.0.
- Kato 2015, Nichols 2017 and Skora 2018: O2 steps.
- Lin 2023: 28 odours and pheromones on 22 sensory neurons.
- Rusu 2026: random bacterial pulses (raw TIFF only, no licence).

**Spontaneous and freely moving, as held-out dynamics or behaviour-coupled targets.**

- Atanas 2023: DANDI CC-BY-4.0; also WormWideWeb.
- Kimura 000692: CC-BY-4.0.
- Hallinen 2021.
- The Simeon pooled corpus: z-scored and resampled to 3 Hz, with no stimulus timing.

**Main gaps.**

1. **No open freely-moving whole-brain dataset with optogenetic perturbation** was found in any lab. Fieseler 2025 (Zimmer; mutants and muscle silencing, moving and immobilised) is unreleased.

2. **Perturbation data are almost all immobilised calcium at about 1.6–4 volumes/s.** Where the entries state the indicator, the large sets use nuclear GCaMP:

   - GCaMP6s: Randi, Dunn, Yemini.
   - GCaMP7f: Flavell.
   - GCaMP5K: Kato 2015, Skora 2018.
   - GCaMP6f: Kaplan 2020.

   Seyedolmohadesin's GCaMP6s localisation is not verified. Paralysis by levamisole or tetramisole is a cholinergic manipulation.

   - So fits need a calcium/indicator forward model.
   - Fast or voltage-level dynamics are unconstrained.
   - Kato 2015 reports that immobilisation itself changes the dynamics.

3. **Neuromodulator contrasts with stimulation are thin.**

   - unc-31 has only 18 animals, a different pulse length, and no raw data on a verified host.
   - Mutant whole-brain imaging otherwise covers tdc-1, flp-13, npr-1, daf-2 and unc-13 (sensory panel).
   - No stimulate-and-record data in tph-1, pdfr-1 or tax-4 backgrounds were found.

4. **Electrophysiology is sparse.**

   - Open raw traces cover about 15 classes, all dissected single-cell recordings, mostly from the Qiang Liu lab, on Mendeley (CC BY 4.0).
   - Edge-level facts (ASH→AVA excitatory and glutamatergic; AVA–A-motor-neuron gap junctions rectifying) exist only in figures.

5. **Licences are missing on many key sets:** every Zimmer OSF project, Randi's OSF exports, the unc-31 arm, Creamer's OSF data zip, the Hallinen node, WormWideWeb, the latest Atanas and Dag Zenodo versions, Rusu, Pritz, cengen.org and wormwiring.org. For Beets 2023's EC50 table, the full data are only "upon request".

6. **Details not verified:**

   - Uzel 2022's methods.
   - Atanas 2023 and Dag 2023 final Cell texts (preprint methods used).
   - Randi's per-animal neuron counts and stimulation counts.
   - Dunn's volume rate and per-file stimulation.
   - Whether DANDI 000565 contains stimulation.
   - Whether Randi's unc-31 raw data exist anywhere.

7. **Wiring.** No new adult hermaphrodite EM connectome from 2024–2026 was found, but the search budget ran out before a final sweep. Witvliet 2021, Cook 2025 (P. pacificus) and Yim 2024 (dauer) exclude gap junctions, so adult gap-junction weights still rest on White 1986, Varshney 2011 and Cook 2019.

______________________________________________________________________

## A. Leifer lab and the Creamer/Leifer/Pillow model

Compiled 2026-11-14 with web tools only. Provenance tags: **[host]** means the data host's own page or API; **[paper]** means the article's methods or data-availability text; **[2nd]** means second-hand (a search snippet, citing paper or lab web page about someone else's host). The Nature article was read through its HTML page, its first ~200k characters. PMC article pages returned a browser check and could not be read. Each entry gives sizes in bytes as the host lists them.

Summary for the fitting question. The Randi 2023 atlas is the only Leifer dataset found in which one identified neuron is stimulated and the rest of the identified head is recorded. It exists in three forms: processed text and Python exports on OSF, raw NWB on DANDI, and a compact kernel/q-value file in the `wormneuroatlas` package. Its unc-31 arm is processed-only on OSF, and in the package. Creamer/Leifer/Pillow fit a linear model to 110 of these animals (calcium, 2 Hz, signed weights on connectome edges) and publish their fitted weights in their repo under MIT. Every other Leifer dataset found is in one of three groups: freely moving but without identities or perturbation (Nguyen 2016, Nguyen 2017, Hallinen 2021), neuron-position data (Yu 2021), or behaviour-only optogenetics. No extended functional atlas was found, and no newer neuropeptide signal-propagation dataset either (see A10).

______________________________________________________________________

### A1. Signal propagation atlas, wild type: Randi, Sharma, Dvali & Leifer, 2023, *Nature* 623:406–414

- **Verification:** Measurement details are [paper] (Nature HTML). The OSF file list, OSF licence field, DANDI metadata and DANDI licence are [host]. The DANDI link on the lab's publications page is [2nd].
  URLs opened:

  - <https://www.nature.com/articles/s41586-023-06683-4> (through its cookie redirect)
  - <https://api.osf.io/v2/nodes/e2syt/>
  - <https://api.osf.io/v2/nodes/e2syt/files/osfstorage/>
  - <https://api.osf.io/v2/nodes/e2syt/files/osfstorage/671a5286badd54a2128707e3/> (pages 1 and 68)
  - <https://api.osf.io/v2/nodes/e2syt/wikis/> and <https://api.osf.io/v2/wikis/qu643/content/>
  - <https://api.osf.io/v2/nodes/e2syt/children/>
  - <https://api.dandiarchive.org/api/dandisets/?search=Leifer>
  - <https://api.dandiarchive.org/api/dandisets/001075/>
  - <https://api.dandiarchive.org/api/dandisets/001075/versions/0.240930.1859/info/>
  - <https://api.dandiarchive.org/api/dandisets/001075/versions/0.240930.1859/assets/> (sorted by path; glob query)
  - <https://api.dandiarchive.org/api/assets/b5ec51cb-5763-4015-aaaa-fb104498c75e/>
  - <https://leiferlab.princeton.edu/publications.php>
  - <https://github.com/leiferlab/pumpprobe>
  - <https://zenodo.org/api/records/8312985>
  - <https://api.github.com/orgs/leiferlab/repos?per_page=100>

  Blocked: <https://pmc.ncbi.nlm.nih.gov/articles/PMC10632145/> (browser check).

- **Measured** \[paper\]:

  - **Neurons and identification:**
    - Whole-brain recording of the head. Pooled across animals, "activity from 186 of 188 neurons in the head, or 99% of all head neurons".
    - A per-animal count of segmented or identified neurons is not stated in the main text read; it may be in Extended Data, which was not read.
    - Identities were assigned by hand with NeuroPAL (mtagBFP2, CyOFP1.5, tagRFP-T, mNeptune2.5) "by comparing each neuron's colour, position and size to a known atlas".
    - Creamer et al., who used these data, report 156 neurons captured in total [2nd, their paper].
  - **Preparation:** Immobilised but awake, on 10% agarose pads with 100-nm polystyrene beads plus 2 µl levamisole (500 µM stock). "Pharyngeal pumping was visible during recordings."
  - **Stimulus and perturbation:**
    - Two-photon (850 nm, 500 kHz pulses, 1.2 mW at the sample) single-neuron activation of the purple-light-sensitive GUR-3/PRDX-2 system.
    - A 500-ms train of light pulses in wild type.
    - "Every 30 s, a random neuron was selected among the neurons found in the current volumetric image", for up to 40 min per animal.
    - The total number of stimulation events and the number of distinct stimulated neurons are not stated in the text read; Extended Data Fig. 3b plots events per neuron.
    - "Measured pairs were imaged at least once, and some as many as 59 times."
    - 23,433 pairs measured (about 66% of possible head pairs). 1,310 of them (6%) are functionally connected at q < 0.05.
    - The actuator is drug-inducible (QF/hGR, dexamethasone), to avoid developmental toxicity. Strain AML462 (TWISP); the GCaMP construct is rab-3::his-24::GCaMP6s.
  - **Behaviour:** Not recorded; the animals are immobilised.
  - **Worms:** 113 wild-type-background animals [paper]. DANDI lists 113 subjects [host]. The OSF `raw_extracted_data` folder holds 678 text files, numbered 0\_… to 112\_…, six per recording (`ds_name`, `gcamp`, `labels`, `stim_neurons`, `stim_volume_i`, `t`) [host]. 678 = 113 × 6 is my arithmetic.
  - **Recording length:** Up to 40 min per animal [paper]; Creamer et al. say 20–40 min [2nd].
  - **Rate:** Spinning-disk confocal at 2 volumes/s, with 505 nm excitation at 1.4 mW/mm² [paper].
  - **Indicator:** GCaMP6s, nuclear-localised [paper].

- **Access:**

  - **Processed data:** OSF project "Neural signal propagation atlas of C. elegans", <https://doi.org/10.17605/OSF.IO/E2SYT> (node e2syt), public. Downloadable without request [host]. The OSF wiki [host] describes four parts:
    - `exported_data_full.tar.gz` is for the lab's Python libraries; run `pumpprobe/scripts/adjust_paths.py` after extracting.
    - `exported_data` holds the wild-type basic results as plain text.
    - `exported_data_unc31` holds the same for the mutant (see A2).
    - A pointer to simplified kernels in `worm-functional-connectivity` and on funsim.princeton.edu (see A3).
  - **Raw data:** DANDI 001075, "Neural signal propagation atlas of Caenorhabditis elegans", version 0.240930.1859, DOI 10.48324/dandi.001075/0.240930.1859. Embargo status OPEN [host]. Built with github.com/catalystneuro/leifer_lab_to_nwb [host]. The lab page also links an example notebook at github.com/catalystneuro/example-notebooks/tree/leifer/001075, and a different DANDI version, 0.240920.1434 [2nd].
  - **Analysis code:** github.com/leiferlab/pumpprobe (Zenodo 10.5281/zenodo.8312985), wormdatamodel, wormneuronsegmentation-c, wormbrain. Acquisition code: pump-probe-acquisition [paper].

- **Licence:**

  - OSF e2syt: not stated. The node's `node_license` is null and it has no licence relationship [host].
  - DANDI 001075: "spdx:CC-BY-4.0" [host].
  - Paper: "This article is licensed under a Creative Commons Attribution 4.0 International License" [paper].
  - pumpprobe code: GPL-3.0 on the GitHub sidebar [host], but "gpl-2.0-or-later" on Zenodo record 8312985 [host]. The two disagree.

- **Size and format:**

  - **OSF** [host], in bytes:
    - `exported_data.tar.gz`: 523,093,816 (2023-05-10)
    - `exported_data_full.tar.gz`: 1,143,920,663
    - `exported_data_unc31.tar.gz`: 84,924,311
    - Pre-review versions (2022-11-21): `exported_data_pre_review.tar.gz` 241,016,857; `exported_data_full_pre_review.tar.gz` 621,084,938; `exported_data_unc31_pre_review.tar.gz` 84,924,579
    - `raw_extracted_data/`: 678 plain-text files, e.g. `0_gcamp.txt` is 10,108,632
    - `spont_data_fig6.zip`: 2,402,185 and `spont_data_fig6_raw.zip`: 1,114,732 (added 2025-10-10). These are the paper's spontaneous-activity comparison; the paper says spontaneous activity was "measured in immobilized worms lacking optogenetic actuators".
  - **DANDI** \[host\]: 223 NWB assets, 4,074,234,123,623 bytes (~4.07 TB), 113 subjects.
    - Each session has a `desc-imaging_ophys+ogen.nwb` file (about 8–56 GB) and a `desc-segmentation_ophys+ogen.nwb` file (about 3–25 MB).
    - Spot check of the sub-100 asset: genotype "WT", age "P1D".

- **Model mapping:**

  - **Signal:** Calcium (nuclear GCaMP6s ΔF/F), not voltage, sampled at 2 Hz.
  - **Stimulus:** A brief (0.5 s), depolarising drive into one identified node. The paper defers validation of the GUR-3/PRDX-2 mechanism to its Supplementary Information, which was not read.
  - **What a response is:** The paper's responses "represent the propagation of signals over all paths in the network". They are effective, multi-path responses, not monosynaptic ones. A model fitted to them has to be simulated end to end: inject into node j, read the calcium of every other node.
  - **Kernels:** Each response is fitted as a sum of convolved decaying exponentials (at most two terms).
  - **What it can constrain:**
    - The sign and magnitude of chemical and gap weights on connectome edges, but only jointly through network simulation. Gap and chemical contributions are not separated by the data.
    - Each neuron's response timescales, plus a calcium-indicator filter.
    - Sign of chemical synapses where the pathway is short.
  - **Negative evidence:** The q_eq equivalence test (ε = 1.2σ) gives evidence for absent coupling, usable as a penalty.
  - **What it cannot constrain:** Sensory gains, since there is no sensory stimulus. Behaviour readouts.

- **Known quality issues:**

  - **Hard identifications:** "Some neurons are particularly hard to identify in NeuroPAL", for example AVB, ADF, SIA, SIB and RMF; AWCon and AWCoff are hard to tell apart. RID has dim tagRFP-T and was targeted manually [paper].
  - **Inclusion criteria:** An observation required the stimulated neuron to show an autoresponse, and both neurons had to be "unambiguosly identified" (sic) [paper, Extended Data Fig. 4a].
  - **Off-target activation:** Checked by photobleaching and a z-shift test. "Only targeted neuron and not nearby neurons appear photobleached" [paper, Extended Data Fig. 2].
  - **Photobleaching:** Removed with a double-exponential fit to the baseline [paper].
  - **Lower bound:** The counts of functional connections are a lower bound [paper].
  - **Low activity:** The transgenic animals were "viable but still significantly less active than WT animals" [paper].
  - **Uneven sampling:** Pairs were sampled between 1 and 59 times. Creamer et al. report a train–test STAM correlation of only 0.16, from variable responses and "significant animal-to-animal variability" [2nd, Creamer v3].
  - **Recording start:** Creamer's data config drops the first 120 samples (1 min): "the calcium dynamics always look strange at the start of a recording, possibly due to the laser being turned on" [host, Creamer repo].
  - **Possible light confound:** GUR-3 is a light/H₂O₂-sensing receptor that this strain over-expresses in every neuron. Gauthey et al. 2024 report that high-intensity blue imaging light evokes global brain dynamics "partly mediated" by GUR-3, more at 488 nm than at 505 nm [2nd, search snippet; see A9]. The atlas images at 505 nm.
  - **Unsourced caveat:** Levamisole is a cholinergic agonist, so its effect on propagation along cholinergic edges deserves a check. This was not found stated on any page read.

- **Cross-check from section B's pass (DANDI asset listing) \[host\]:**

  - The 223 assets of DANDI 001075 are 113 `_desc-imaging_ophys+ogen.nwb` files (up to 60.5 GB each) and 110 `_desc-segmentation_ophys+ogen.nwb` files (from 1.9 MB). So three sessions have no segmentation file.
  - Creamer et al. fit "110 separately recorded animals". The match with the 110 segmentation files is my inference, not verified.
  - The DANDI description (re-read in the merge) does not mention unc-31.

- **Table row:** `| Randi 2023 signal propagation atlas (WT) | not stated per animal; 186/188 head neurons pooled | NeuroPAL, manual | immobilised (beads + levamisole) | 2-photon GUR-3/PRDX-2 single-neuron, 0.5 s, every 30 s | 113 | OSF: not stated; DANDI: CC-BY-4.0 | OSF 0.52 GB (txt) / 1.14 GB (full); DANDI 4.07 TB | txt, tar.gz (Python exports); NWB | public, no request |`

______________________________________________________________________

### A2. Signal propagation atlas, unc-31 mutant arm: Randi et al., 2023, *Nature* (same paper)

- **Verification:** Measurement details are [paper]. The files and licence are [host] (OSF e2syt). The URLs are those of A1.
- **Measured** \[paper\]:
  - Strain AML508, an unc-31 knockout derived from TWISP AML462, with the same imaging and NeuroPAL. 18 animals.
  - Stimulus: a **300-ms** pulse train, against 500 ms in wild type.
  - The paper says the mutation "disrupts dense-core-vesicle-mediated extrasynaptic signalling of peptides and monoamines". "unc-31 animals had a smaller proportion of functional connections than did WT animals"; no exact fraction is given in the text read.
  - Neurons per animal and stimulation counts are not stated in the text read.
- **Access:** OSF e2syt, `exported_data_unc31.tar.gz`, downloadable without request [host]. A DANDI copy of the unc-31 arm is **not verified**: DANDI lists 113 subjects, matching the wild-type count, and the one spot-checked asset is "WT". Also in `wormneuroatlas` (`strain="unc31"`) and in `worm-functional-connectivity` (`unc-31.pickle`) (A3).
- **Licence:** OSF: not stated (node licence null) [host].
- **Size and format:** `exported_data_unc31.tar.gz` is 84,924,311 bytes (pre-review version 84,924,579), plain-text basic results per the OSF wiki [host]. The raw imaging is not verified on any host.
- **Model mapping:** The wild-type vs unc-31 difference is the closest available handle on neuromodulatory coupling: removing dense-core-vesicle release, i.e. setting the peptide/monoamine couplings to zero while keeping chemical and gap wiring. Fitting both arms with shared wiring and a modulator term switched off in the mutant would constrain the modulator layer's contribution to the effective responses. The stimulus duration differs between arms (300 vs 500 ms), so the input has to be modelled explicitly rather than compared raw.
- **Known quality issues:**
  - Small n (18).
  - The shorter stimulus confounds a direct comparison of amplitudes.
  - unc-31 animals are less active than wild type [paper]. The first ~100k characters of the article give ~0.02–0.03 mm/s for unc-31 and transgenic animals against 0.15 mm/s for wild type; I saw these figures only in a summary of the page, not as a quote.
- **Table row:** `| Randi 2023 atlas, unc-31 arm | not stated | NeuroPAL, manual | immobilised | 2-photon GUR-3/PRDX-2, 0.3 s; loss of dense-core-vesicle release | 18 | not stated (OSF) | 84.9 MB | txt in tar.gz | public, no request |`

______________________________________________________________________

### A3. Distilled atlas packages: `wormneuroatlas`, `worm-functional-connectivity`, FunSim: Randi (code author) / Leifer lab, 2022–2025

- **Verification:** All [host].
  URLs opened:
  - <https://pypi.org/pypi/wormneuroatlas/json>
  - <https://api.github.com/repos/francescorandi/wormneuroatlas>
  - <https://github.com/francescorandi/wormneuroatlas>
  - <https://api.github.com/repos/francescorandi/wormneuroatlas/contents/wormneuroatlas>
  - <https://api.github.com/repos/francescorandi/wormneuroatlas/contents/wormneuroatlas/data>
  - <https://raw.githubusercontent.com/francescorandi/wormneuroatlas/main/wormneuroatlas/NeuroAtlas.py> (first 100k of 109.5k characters)
  - <https://github.com/leiferlab/worm-functional-connectivity>
  - <https://api.github.com/repos/leiferlab/worm-functional-connectivity/contents/atlas>
  - <https://funsim.princeton.edu>
- **Measured:** Derived from A1 and A2; no new recordings.
- **What `wormneuroatlas` ships:**
  - `wormneuroatlas/data/funatlas.h5` (18,266,176 bytes). The code reads these datasets per strain (`wt`, `unc31`):
    - `dFF`: response averaged over a time window and trials
    - `dFF_all`: every trial per pair
    - `q`
    - `q_eq`: the equivalence-test q
    - `occ1`: the occurrence count per pair
    - `kernels`: fitted kernel terms stored as rows of `g, factor, power_t, branch` and rebuilt with `ExponentialConvolution_min`. `get_kernel(i, j)` returns "the average kernel for pair i\<-j".
    - Also `neuron_ids` and the `time_compiled` attribute.
  - Accessors: `get_signal_propagation_map`, `get_signal_propagation_map_all`, `get_signal_propagation_q`, `get_signal_propagation_q_eq`, `get_signal_propagation_occurrence_matrix` and `get_kernel`.
  - Merging options (`merge_bilateral`, `merge_dorsoventral`, `merge_numbered`). The code warns that averaging q under merging "is not mathematically accurate from the statistical point of view".
  - Also bundled:
    - Anatomical connectomes: White 1986 A, L4 and whole; Witvliet 2020 datasets 7 and 8; `aconnectome.json`/`.h5`
    - Bentley 2016 monoamine and neuropeptide edge lists
    - CeNGEN expression at four thresholds, plus `cengen.h5`
    - Beets peptide–GPCR deorphanization
    - Fenyves 2020 synapse-sign spreadsheet
    - neuron positions, lineage
    - A builder for a Ripoll-Sánchez-style peptidergic connectome
- **Access:**
  - `pip install wormneuroatlas`. PyPI latest version 0.0.7.3, uploaded 2023-06-29; the wheel is 37,760,635 bytes and the sdist 37,451,656 bytes.
  - GitHub: github.com/francescorandi/wormneuroatlas, pushed 2025-08-24. Docs: francescorandi.github.io/wormneuroatlas (not opened).
  - `worm-functional-connectivity` ships `atlas/wild-type.pickle` (2,082,820 bytes), `atlas/unc-31.pickle` (608,632 bytes) and `mock_.pickle`. Its `FunctionalAtlas` class has `get_standard_stimulus()` and `get_responses()`.
  - FunSim (funsim.princeton.edu) is a web simulator over the same atlas, wild type and unc-31. It doesn't say the data can be downloaded.
  - All downloadable without request.
- **Licence:**
  - wormneuroatlas: GitHub "GNU General Public License v3.0" (GPL-3.0). PyPI has an empty licence field, with the classifier "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)".
  - worm-functional-connectivity: not stated (GitHub reports no licence).
  - FunSim: not stated.
  - The bundled third-party tables carry their source papers' terms. The README says "Please cite the code if you use it, along with the papers containing the datasets you use."
- **Size and format:** HDF5 (`funatlas.h5`) and CSV/JSON/H5 tables within a ~37.8 MB wheel. Python pickles for worm-functional-connectivity.
- **Model mapping:** The fastest route to fitting targets. Per pair (i←j) it gives the mean response amplitude, all trials, q, q_eq, the trial count and the parametric kernel. That is enough for a likelihood or a z-scored loss on the response of every measured pair, with q_eq as evidence of absence. Its comparison tables (CeNGEN, Bentley, Fenyves signs) line up with the project's own data layer.
- **Known quality issues:**
  - Merged-class q values are approximate (code warning).
  - PyPI has not been updated since 2023-06, while GitHub has been pushed in 2025-08. Whether the bundled `funatlas.h5` matches the final OSF export (2023-05-10) is not verified.
  - The q values and kernels inherit every issue of A1.
- **Table row:** `| wormneuroatlas / worm-functional-connectivity (Randi atlas distilled) | n/a (pairwise matrices over atlas IDs) | NeuroPAL IDs from A1 | immobilised (derived) | WT + unc-31 kernels, dFF, q, q_eq | 113 + 18 (derived) | GPL-3.0 (wormneuroatlas); not stated (wfc, FunSim) | funatlas.h5 18.3 MB; wheel 37.8 MB; pickles 2.1/0.6 MB | HDF5, pickle | public, pip/GitHub |`

______________________________________________________________________

### A4. Connectome-constrained linear dynamical system: Creamer, Leifer & Pillow, bioRxiv 10.1101/2024.09.22.614271 (v1 2024-09-23; v3 2025-09-26)

- **Verification:** The model and data used are [paper] (v1 and v3 full text, first 100k of ~110k characters of v3). The repo and OSF entries are [host].
  URLs opened:
  - <https://www.biorxiv.org/content/10.1101/2024.09.22.614271v1> and v1.full-text
  - <https://www.biorxiv.org/content/10.1101/2024.09.22.614271v3> and v3.full-text
  - <https://api.github.com/repos/Nondairy-Creamer/Creamer_LDS_2026>
  - <https://api.github.com/repos/Nondairy-Creamer/Creamer_LDS_2024> (returns `Creamer_LDS_2026`, same created_at 2024-09-21, so the repo was renamed)
  - <https://github.com/Nondairy-Creamer/Creamer_LDS_2026>
  - <https://api.github.com/repos/Nondairy-Creamer/Creamer_LDS_2026/git/trees/main?recursive=1>
  - <https://api.github.com/repos/Nondairy-Creamer/Creamer_LDS_2026/contents/trained_models>
  - Raw files: `quick_start_examples/generate_weights_spreadsheet.py`, `quick_start_examples/model_weights.csv`, `submission_params/create_data_set.yml`, `submission_params/exp_test.yml`, `create_data_set.py`
  - <https://api.osf.io/v2/nodes/qxhjd/> and <https://api.osf.io/v2/nodes/qxhjd/files/osfstorage/>
  - A search for a journal version found none.
- **Measured (data fitted)** \[paper\]:
  - "All data used is published", citing Randi et al. 2023. "Of the 110 separately recorded animals, we trained the model on an 80-animal training set and evaluated it on a 30-animal held-out test set." The supplement used five 55/55 splits.
  - 110 is fewer than Randi's 113 wild-type animals, and the text read gives no reason.
  - unc-31 data and additional recordings are not mentioned. So the fitted set is a **subset** of the Randi 2023 wild type, not a superset.
  - 156 head neurons are in the model (paper). The repo README says "~40 hours on a desktop to fit across 80 animals and 154 neurons"; the 154 vs 156 discrepancy is unexplained.
  - The repo README points to the data at <https://osf.io/qxhjd/> (Creamer's own OSF project, created 2025-01-14): one file, `2025_creamer_lds.zip`, 161,743,797 bytes [host]. `create_data_set.py` reads per-recording `francesco_green.npy`, `nan_mask.npy`, `labels.npy`, `stim_recording_cell_inds.npy` and `stim_volumes_inds.npy` from "the output of the Randi et al. 2023 preprocessing pipeline" [host]. It is a repackaged Randi wild-type export; whether its content is byte-identical to OSF e2syt is not verified.
- **Model** \[paper, repo\]:
  - **Latent state:** "a latent vector of the calcium activity of every neuron". The paper notes the recordings "only measure calcium activity", which it treats as a proxy for voltage. So the model maps **calcium to calcium**, with no voltage or indicator model.
  - **Dynamics:** First-order linear dynamics x_t = W x\_{t−1} + stimulation + Gaussian noise. W is non-zero only where neurons "share a chemical or electrical synapse".
  - **Connectome:** Synapse counts summed over "three adult connectomes" and one L4 connectome (White 1986, Witvliet 2021). Chemical and electrical counts are summed into one mask. The README says the connectome was downloaded from NemaNode on 1 November 2025 as `white_1986_jsh`, `white_1986_n2u`, `witvliet_2020_7` and `witvliet_2020_8`.
  - **Signs:** "Magnitude and sign" are learned; no sign constraint.
  - **Stimulus input:** One-hot for the stimulated cell. Diagonal B, so the stimulus enters only the targeted neuron, through a learned per-neuron filter "45 timepoints long (22.5 seconds at 2 Hz)".
  - **Observations:** Partial (unrecorded neurons are latent), with an offset d. Process noise Q and observation noise R are diagonal.
  - **Fitting:** EM with a Kalman filter-smoother.
  - **Configuration** \[host, `exp_test.yml`\]: all lags are 1; the emission weights are frozen; `hold_out: 'worm'`.
  - **Result** (v3 abstract): it captures causal interactions "at 92% of the reproducibility of the perturbation data". A shuffled connectome does much worse, and a fully connected model gives no gain overall (slightly better on 56 putative extrasynaptic pairs, p = 0.011).
- **Fitted parameters published:** Yes, in the GitHub repo \[host\]:
  - `models/connectome_constrained.pkl` (20,103,437 bytes), `models/fully_connected.pkl` (20,103,431) and `models/shuffled_constrained.pkl` (20,103,481).
  - `quick_start_examples/model_weights.csv` (41,959 bytes; columns `presynaptic cell, postsynaptic cell, weight`). It is generated from `connectome_constrained.pkl` over the off-diagonal masked entries. The first rows include ADAL→ADAR 0.092504 and ADAL→AVAL −0.011104. The row count was not verified.
  - `data/measured_stams.pkl`, `data/measured_corr.pkl`, `data/example_recording.pkl` and `data/initial_conditions.pkl`.
  - `anatomical_data/` (`chemical.pkl`, `gap.pkl`, `peptide.pkl`, `cook_synapse_size_connectome.csv`, `leifer_extrasynaptic_pairs.csv`, `neurotransmitters.csv`, `neuron_type.csv`).
  - `trained_models/` holds only `.gitkeep`.
  - v3 text: "We list the learned weights in a supplemental CSV and table as a resource that researchers can use" (the supplement was not opened).
- **Access:** Public GitHub; downloadable without request. Data from OSF qxhjd, also public.
- **Licence:**
  - Repo: "MIT License" (GitHub API, LICENSE file) [host].
  - Preprint: "It is made available under a CC-BY-NC 4.0 International license" [paper/bioRxiv].
  - OSF qxhjd: not stated (node licence null) [host].
- **Size and format:** Repo 7,048 KB (GitHub API size) of Python pickles and CSV. OSF zip 161,743,797 bytes.
- **Model mapping:**
  - Their W is a fitted, signed, calcium-domain effective weight per connectome edge, with chemical and gap junctions merged and a 0.5-s step. It is a ready initialisation or prior for the project's per-edge signed weights, but it does not separate chemical from gap, and it has no separate per-neuron time constant beyond W's diagonal.
  - Their learned 22.5-s stimulus filter absorbs opsin and GCaMP6s kinetics. A voltage-level model would need its own indicator filter to be comparable.
  - Their `measured_stams.pkl` and `measured_corr.pkl` are ready-made fitting targets: stimulus-triggered average matrices and correlations.
- **Known quality issues:**
  - Model–data STAM correlation 0.15 against a train–test ceiling of 0.16: noisy STAMs.
  - The linear model "often underestimates the magnitude of the neural responses".
  - Uneven pair sampling.
  - Missing neurons in every recording [paper].
  - 110 vs 113 animals, and 154 vs 156 neurons, are unexplained.
  - In `create_data_set.py` the bleaching correction, filtering and ID handling sit in commented-out code, and with `interpolate_nans: True` the script would raise a NameError [host]. This is a reproducibility risk only.
- **Table row:** `| Creamer/Leifer/Pillow LDS fit (on Randi WT) | 156 modelled (154 per README) | NeuroPAL IDs from A1 | immobilised | single-neuron stimulation as one-hot input + learned 45-step filter | 110 (80 train / 30 test) | repo MIT; preprint CC-BY-NC 4.0; OSF data not stated | 3 × 20.1 MB model pkl + 42 KB weights CSV; data zip 161.7 MB | pickle, CSV | public GitHub + OSF |`

______________________________________________________________________

### A5. Whole-brain imaging in freely behaving worms: Nguyen, Shipley, Linder, Plummer, Liu, Setru, Shaevitz & Leifer, 2016, *PNAS* 113:E1074–E1081

- **Verification:** The published version's details are [paper] (PMC page). The arXiv v1 details are [paper, preprint]. The S3 bucket is [2nd] (the lab's sharing page; the bucket was not opened).
  URLs opened:
  - <https://www.pnas.org/doi/10.1073/pnas.1507110112> (HTTP 403)
  - <https://pmc.ncbi.nlm.nih.gov/articles/PMC4776509/> (loaded)
  - <https://ar5iv.labs.arxiv.org/html/1501.03463>
  - <https://leiferlab.princeton.edu/sharing.php>
- **Measured** \[paper, published\]:
  - Strain AML14 (wtfEx4), with nuclear-localised GCaMP6s and tagRFP under rab-3.
  - 6 head-volumes/s, about 33 z-slices per volume. The arXiv v1 says 5 volumes/s with 40 slices.
  - Four freely moving worms, with 77, 68, 57 and 56 nuclei (mean 64.5 of an expected 181 in the field of view). Plus one GFP control worm.
  - Recorded for up to ~4–5 min; the abstract says "over 4 min" and the text "up to 5 min".
  - No reliable identification. Identities are only candidates, and the best cross-worm matching got 6 of 11 known neurons right.
  - Behaviour: centreline, velocity, and an ethogram of forward, backward and turning (arXiv).
  - No perturbation.
- **Access:**
  - Processed traces and behaviour are in the PNAS supplementary datasets S1–S16 [paper]. The PNAS host was not opened (403).
  - Raw data: the PMC text says "Raw imaging data (>200 GB) will be provided to any researcher on request" [paper]. The lab's sharing page later lists raw imaging (~0.5 TB) in the Amazon S3 bucket `s3://leiferlabnguyen2015`, under Requester Pays, with egress charged at $0.09/GB at the time of posting [2nd].
  - Downloadable without request: the supplement, yes. The raw data: pay-per-download on S3, or originally on request.
- **Licence:** Data: not stated on any page read. Article: "Freely available online through the PNAS open access option" (PMC); no CC licence was visible.
- **Size and format:** The supplement's sizes and formats were not verified. Raw data ~0.5 TB per the lab page, >200 GB per the paper.
- **Model mapping:** Calcium only, with no stimulus and no identities, so it cannot be mapped onto connectome nodes. It can constrain only population-level statistics and the neural-to-behaviour relation in moving animals. It is weak for connectome parameters.
- **Known quality issues:**
  - Motion artefacts (a GFP control was used; residual noise remained).
  - Only 64.5 of the 181 neurons expected in the field of view were observed on average (~36%, my arithmetic). Neurons also leave the field of view, leaving gaps.
  - Identity matching is poor [paper].
- **Table row:** `| Nguyen 2016 freely moving whole-brain | 56–77 | none (candidate IDs only) | moving | none | 4 (+1 GFP control) | not stated | supplement not verified; raw ~0.5 TB (lab page) | PNAS SI; raw on S3 | SI public; raw S3 requester-pays |`

______________________________________________________________________

### A6. Decoding locomotion from population activity: Hallinen, Dempsey, Scholz, Yu, Linder, Randi, Sharma, Shaevitz & Leifer, 2021, *eLife* 10:e66135

- **Verification:** Measurement details are [paper] (the eLife API full text, first 200k of 277k characters). The OSF entries are [host].
  URLs opened:
  - <https://elifesciences.org/articles/66135> and <https://elifesciences.org/articles/66135v1> (did not render)
  - <https://api.elifesciences.org/articles/66135>
  - <https://api.osf.io/v2/guids/dpr3h/>
  - <https://api.osf.io/v2/nodes/dpr3h/files/osfstorage/>
  - <https://api.osf.io/v2/guids/r5tb3/> and <https://api.osf.io/v2/registrations/r5tb3/>
  - <https://api.osf.io/v2/licenses/563c1cf88c5e4a3877f9e96c/>
  - <https://api.osf.io/v2/registrations/r5tb3/files/osfstorage/5fe9f609e9ccb90245aece7a/>
  - <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010421> (TMAC, for quality)
  - The leiferlab repos listing (for PredictionCode)
- **Measured** \[paper\]:
  - **Strains:** AML32 and AML310, both "P rab-3::NLS::GCaMP6s; P rab-3::NLS::tagRFP"; AML310 adds a rig-3::tagBFP marker for AVA. Controls are AML18, a GFP strain.
  - **Neurons:** "the majority of the 188 neurons in the head"; median 121 ± 12 per recording, 134 in the exemplar.
  - **Identification:** None, except AVAL/AVAR through the BFP marker.
  - **Moving recordings:** 11 GCaMP recordings (7 AML32, 4 AML310) and 11 GFP control recordings (AML18).
  - **Immobilised recordings:** Three moving-to-immobile transitions with levamisole, plus one bead-immobilised recording.
  - **Rate and length:** 6 head volumes/s. Recordings of 4–15 min (Table 3), at least 200 s each.
  - **Behaviour:** Velocity, body curvature and plate position.
  - **Perturbation:** None, apart from the levamisole transition.
- **Access:**
  - The paper cites <https://doi.org/10.17605/OSF.IO/DPR3H>, a public OSF node created 2020-12-21. Its generated-dataset entry instead cites OSF.IO/R5TB3, an OSF registration of the same node made 2020-12-28. Both are downloadable without request [host].
  - Code: github.com/leiferlab/PredictionCode, "GPL-2.0" (GitHub API) [host].
- **Licence:**
  - Node dpr3h: not stated (`node_license` null) [host].
  - Registration r5tb3: "CC0 1.0 Universal" [host].
  - Paper: CC-BY-4.0, holder Hallinen et al. [paper].
- **Size and format** \[host\]:
  - **Node dpr3h** (updated 2021-07-02), in bytes:
    - `AML310_moving.tar.gz`: 348,444,164
    - `AML310_immobile.tar.gz`: 613,428,180
    - `AML310_transition.tar.gz`: 486,820,852
    - `AML32_chip.tar.gz`: 314,689,799
    - `AML18_moving.tar.gz`: 1,409,801,111
    - `AML32_moving.tar.gz`: 1,218,075,251
    - Total 4,391,259,357 bytes (my sum).
  - **Registration r5tb3** (snapshot of 2020-12-22) holds the same six names at smaller sizes: 311,105,693; 321,144,749; 305,298,374; 308,848,700; 945,888,405; 1,079,400,039. The two versions differ.
  - The paper says the data comprise "calcium activity traces and animal pose and position"; the format inside the tarballs was not verified.
- **Model mapping:** Calcium, freely moving, unidentified, no stimulus. It cannot constrain connectome edges. It can constrain the readout from neural activity to locomotion and the dimensionality and timescales of spontaneous dynamics in moving animals. The GFP controls give an empirical motion-artefact floor.
- **Known quality issues:**
  - Motion artefacts are assumed common to GCaMP and RFP and corrected by subtracting a scaled RFP.
  - Recordings were discarded on photobleaching.
  - Tracking failures leave gaps, and the transition periods were tracked by hand [paper].
  - Creamer et al. 2022 (TMAC), which used these recordings, find the GFP controls still weakly decodable after correction, so some behaviour-correlated artefact remains [2nd, PLOS Comp Biol page].
- **Table row:** `| Hallinen 2021 moving whole-brain + GFP controls | median 121 ± 12 | none (AVA only, BFP) | moving (+ immobilised/transition) | none (levamisole transition) | 11 GCaMP + 11 GFP + 4 immobilisation recordings | node: not stated; registration: CC0 1.0 | 4.39 GB (node) | tar.gz (contents not verified) | public, no request |`

______________________________________________________________________

### A7. Fast deep neural correspondence (fDNC): Yu, Creamer, Randi, Sharma, Linderman & Leifer, 2021, *eLife* 10:e66410

- **Verification:** Details are [paper] (eLife API, first 100k characters); the OSF entries are [host].
  URLs opened:
  - <https://api.elifesciences.org/articles/66410>
  - <https://api.osf.io/v2/guids/t7dzu/>
  - <https://api.osf.io/v2/nodes/t7dzu/files/osfstorage/> and its folders `5ff8ad0f86541a018314eb01/` and `5ff8ad98e80d37018ca5b777/`
- **Measured** \[paper\]:
  - New data: 11 immobilised NeuroPAL worms (strain AML320, nuclear GCaMP6s), one volume each, used as identification ground truth. About 133.6 neurons segmented per volume, of which a mean of 57.5 (~43%) were labelled by humans.
  - Semi-synthetic training sets of point clouds.
  - No activity time series is new: the moving recording used (1,514 volumes at 6 volumes/s, AML32) is from Nguyen 2017.
- **Access:** OSF <https://doi.org/10.17605/OSF.IO/T7DZU> (node "fDLC_Neuron_ID_C.elegans"), public, downloadable without request [host].
- **Licence:** OSF: not stated (`node_license` null) [host]. Paper: CC-BY-4.0 [paper].
- **Size and format** [host], under `DataAndPretrainedModel/Data/`, in bytes:
  - `train_synthetic.zip`: 738,312,040
  - `validation_synthetic.zip`: 39,815,873
  - `test_synthetic.zip`: 19,842,018
  - `NERVE.zip`: 54,869,287
  - `test_tracking.zip`: 3,454,873
  - `test_neuropal_all_features.zip`: 244,671
  - `test_neuropal_our.zip`: 34,302
  - `test_neuropal_Chaudhary.zip`: 29,447
  - Plus a `pretrained_model/` folder (not listed) and `ReadMe.txt` (853 bytes).
- **Model mapping:** No activity, so no dynamical parameters. Its value here is calibration: an automated identifier scores 64.1% across animals on position alone and 74.7% with NeuroPAL colour (top-3: 92.4%). This bounds the ID error of any dataset that relies on automated matching. Randi 2023 used manual NeuroPAL assignment.
- **Known quality issues:** Only 43% of segmented neurons have human labels. The preprint's numbers differ from the published ones (published: 79.1% within-animal tracking, 64.1% across animals) [paper / 2nd].
- **Table row:** `| Yu 2021 fDNC NeuroPAL ID set | ~134 segmented, 57.5 labelled | NeuroPAL, manual (ground truth) | immobilised | none | 11 (one volume each) | not stated (OSF) | ~0.86 GB in zips (my sum of listed files) | zip (point sets), pretrained model | public, no request |`

______________________________________________________________________

### A8. TWISP strain characterisation: Sharma, Randi, Kumar, Dvali & Leifer, 2024, *Genetics* 227(3):iyae077

- **Verification:** Details are [paper] (Europe PMC full-text XML, first 100k characters); the Figshare entry is [host].
  URLs opened:
  - <https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11228852/fullTextXML>
  - <https://api.figshare.com/v2/articles/23868972>
  - <https://pmc.ncbi.nlm.nih.gov/articles/PMC11228852/> (blocked by a browser check)
- **Measured** \[paper\]:
  - Opsin cross-talk screen. Chrimson and ChR2 respond to 475 and 505 nm, so they were judged unsuitable. TsChR/eTsChR and GUR-3+PRDX-2 respond to 475 nm but weakly or not at all to 505 nm, the GCaMP6s imaging wavelength.
  - Behavioural assays on plates of ~30–40 animals.
  - Whole-brain imaging demonstrations: 0.5-s, 850-nm two-photon stimulation every 30 or 60 s, NeuroPAL identification.
  - TWISP (AML462) = AML456 (QF+hGR > GUR-3+PRDX-2) × AML320 (NeuroPAL + GCaMP6s), about 50 integrated constructs. Pan-neuronal actuator overexpression without induction was toxic or lethal.
  - The total worm count is not given.
- **Access:** Figshare <https://doi.org/10.6084/m9.figshare.23868972.v1>, downloadable without request [host]. Strains from the CGC and plasmids from Addgene [paper].
- **Licence:** Figshare: "CC BY 4.0" [host]. Paper: CC BY 4.0 [paper].
- **Size and format:** One file, `Sharma et al., 2023.zip`, 758,409,991 bytes, "organized into folders that correspond to figures and supplementary tables" [host]. Whether it includes whole-brain recordings beyond Randi 2023 is not verified.
- **Model mapping:** This is the evidence base for treating the atlas stimulus as an on-target depolarisation without imaging-light cross-talk at 505 nm. It constrains the stimulus model, not network parameters.
- **Known quality issues:** GUR-3 is itself a photosensory (H₂O₂/light) receptor, now expressed in all neurons; see A9.
- **Table row:** `| Sharma 2024 TWISP characterisation | not stated | NeuroPAL | immobilised (imaging) + plates (behaviour) | opsin cross-talk tests; 2-photon GUR-3/PRDX-2 | not stated | CC BY 4.0 | 758.4 MB | zip by figure | public, no request |`

______________________________________________________________________

### A9. Light-evoked global brain dynamics: Gauthey, Randi, Sharma, Kumar & Leifer, 2024, *Current Biology* 34(1):R14–R15

- **Verification:** The OSF entries are [host]. The findings are [2nd] (a search-result snippet): the article and its supplement PDF could not be read as text.
  URLs opened:
  - <https://github.com/leiferlab/spontdyn>
  - <https://api.osf.io/v2/nodes/yt8aj/> and <https://api.osf.io/v2/nodes/yt8aj/files/osfstorage/>
  - <https://leiferlab.princeton.edu/publications.php>
  - <https://ars.els-cdn.com/content/image/1-s2.0-S0960982223014446-mmc1.pdf> and <https://par.nsf.gov/servlets/purl/10573501> (binary PDFs; not readable)
- **Measured** \[2nd\]: Whole-brain imaging. High-intensity blue light (as used for calcium imaging) evokes stereotyped 0.007–0.033 Hz global oscillations. These are more common at 488 nm than at 505 nm and are "partly mediated" by GUR-3. Strains, worm counts, rates and identification are **not verified**.
- **Access:** OSF <https://osf.io/yt8aj/> ("Light evokes C. elegans' stereotyped global brain dynamics", "Data from Gauthey et al. 2023"), public [host]. Code: github.com/leiferlab/spontdyn; no licence stated [host].
- **Licence:** OSF: not stated (`node_license` null) [host].
- **Size and format:** `exported_data.tar.gz` 250,889,010 bytes; `exported_data2.tar.gz` 250,894,189 bytes; `spontdyn_list_sort.txt` 6,410 bytes [host]. The format is presumably the same wormdatamodel export as A1; the spontdyn README lists wormdatamodel as a dependency, but the format is not verified.
- **Model mapping:** Imaging light acts as a sensory stimulus: a GUR-3/ROS-mediated drive. Any fit to "spontaneous" whole-brain dynamics imaged at 488 nm, and any atlas strain over-expressing GUR-3, has to treat the imaging light as an input, or exclude its frequency band.
- **Known quality issues:** Not verified beyond the snippet.
- **Table row:** `| Gauthey 2024 light-evoked dynamics | not verified | not verified | immobilised (not verified) | imaging-light intensity/wavelength; gur-3 (2nd-hand) | not verified | not stated | 0.50 GB (2 tar.gz) | tar.gz | public, no request |`

______________________________________________________________________

### A10. Other Leifer datasets, 2021–2026 (and one older set used by A7)

- **Verification:** As marked in each item. The lab's publication list (<https://leiferlab.princeton.edu/publications.php>) and the GitHub org listing (<https://api.github.com/orgs/leiferlab/repos?per_page=100>) were opened to enumerate them. An extended search found no Leifer neuropeptide signal-propagation dataset, and no extended functional atlas or freely moving perturbation-plus-whole-brain dataset after 2021. The Transmitter reports that the lab *plans* to measure neuropeptide flow with optical sensors [2nd, search snippet].
- **Behaviour-only optogenetic perturbation sets (freely moving, no neural recording).** These constrain sensorimotor or behavioural readouts only.
  - **Liu, Kumar, Sharma & Leifer 2022, *PLoS Biol* e3001524.**
    - Closed-loop Chrimson stimulation of the 6 touch receptor neurons.
    - 43,418 valid stimulus events from 95 plates in the anterior–posterior experiments, and 9,776 events from 47 plates in the turn-triggered experiments [paper].
    - Data on IEEE DataPort, DOI 10.21227/t6b0-bc36 [paper]. The host page was not opened, so its licence and size are not verified. Paper: CC BY 4.0.
    - URL opened: <https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3001524>
  - **Kumar, Sharma, Tran, Liu & Leifer 2023, *PLoS Biol* e3002280.**
    - Chrimson in the TRNs and interneurons; gtACR2 in RIV/SMB/SAA. 97,268 behaviour responses to stimulation [paper].
    - Figshare+ 10.25452/figshare.plus.23903202.v2: licence "CC0", one tar of 337,770,250,240 bytes [host]. Code: kumar-sensorimotor-integration, GPL-3.0 [host].
    - URLs opened: <https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3002280> and <https://api.figshare.com/v2/articles/23903202>
  - **Li, Kumar, Poon, Leifer & Zheng 2025, *Cell Reports* 44(10).** Intersectional optogenetics in TRN subtypes.
    - Figshare 10.6084/m9.figshare.28217246.v3: "CC BY 4.0", 44 files (.mat, .xlsx), 199,853,876 bytes [host]. Code: kumar-touch-neuron-subtypes, no licence stated [host].
    - URLs opened: <https://api.figshare.com/v2/articles/28217246> and <https://github.com/leiferlab/kumar-touch-neuron-subtypes>
  - **He, …, Kumar, …, Leifer, Chalfie & Zheng 2026, *PNAS* e2531407123.** Whole-field optogenetics and behaviour.
    - Figshare 10.6084/m9.figshare.30069295.v1: "CC BY 4.0", 2 zips, 32,817,279 bytes [host].
    - The README's bioRxiv link text and its href disagree (2025.02.03.636179 vs 2025.09.18.677212) [host].
    - URLs opened: <https://api.figshare.com/v2/articles/30069295> and <https://github.com/leiferlab/celegans-sensory-motor-reflex>
  - **Kumar, Sharma & Leifer 2024, *iScience*:** data not checked.
- **Analysis-only Leifer papers (no new recordings found on the pages read):**
  - **Dvali, Seguin, Betzel & Leifer 2025, *PRX Life* 3:033021.** Modules and a rich club of the signal-propagation network. arXiv 2412.14498, listed under CC BY-NC-ND 4.0 [host arXiv]. No data link.
  - **Li, (Lin, Sharma,) Leifer & Wolpert, *PNAS* 2026 / arXiv 2504.10300.** Computation estimated from whole-brain GCaMP time series. The source dataset is not named on the abstract page, and no data link is given [host arXiv].
  - **Creamer, Chen, Leifer & Pillow 2022, *PLOS Comp Biol*.** TMAC motion correction, applied to the Hallinen data; code at github.com/Nondairy-Creamer/tmac; data availability "listed in supporting information file S1 Table" [paper].
  - **github.com/leiferlab/explore-sp-atlas (2025).** A rotation project re-analysing OSF e2syt (`exported_data`, `raw_extracted_data`); no licence [host].
  - **Haque et al. 2025, *Sci Adv*,** and **Kochersberger et al. (bioRxiv):** Leifer is a co-author. Not checked beyond the titles; the only Haque data seen is a GEO single-cell RNA-seq entry (GSE286418) in a search result [2nd].
- **Older set used by A7: Nguyen, Linder, Plummer, Shaevitz & Leifer 2017, *PLoS Comp Biol* (NeRVE).**
  - "Tracking Neurons in a Moving and Deforming Brain Dataset", IEEE DataPort, DOI 10.21227/H2901H.
  - Two freely moving worms with nuclear GCaMP6s and RFP. Worm 2 is ~8 min long with 156 neurons tracked; worm 1 is shorter, with hand annotations.
  - Licence: Creative Commons Attribution (CC BY 4.0) per the page's licence link.
  - Size: 328.01 GB. Raw `.dat`/AVI video plus MAT files (`heatData.mat`, `pointStats.mat`, etc.).
  - Access: the page says "Open Access dataset files are accessible to all logged in users" and also "This dataset requires an IEEE DataPort Subscription". Login is required; the subscription status is ambiguous [host].
  - No identities and no perturbation.
  - URL opened: <https://ieee-dataport.org/node/813>
- **Table rows:**
  - `| Liu 2022 closed-loop TRN optogenetics | 0 (behaviour only) | n/a | moving | Chrimson in 6 TRNs | not stated (~12–45 worms/plate, 95+47 plates) | not verified (IEEE) | not verified | not verified | IEEE DataPort (not opened) |`
  - `| Kumar 2023 TRN/interneuron optogenetics | 0 (behaviour only) | n/a | moving | Chrimson / gtACR2 | not stated (plates of ~30–40) | CC0 | 337.8 GB | tar | public, no request |`
  - `| Li 2025 TRN subtype optogenetics | 0 (behaviour only) | n/a | moving | intersectional Chrimson/ACR2 | see sample-size xlsx | CC BY 4.0 | 199.9 MB | .mat, .xlsx | public, no request |`
  - `| He 2026 sensory-motor reflex | 0 (behaviour only) | n/a | moving | whole-field optogenetics | not verified | CC BY 4.0 | 32.8 MB | zip | public, no request |`
  - `| Nguyen 2017 NeRVE tracking set | up to 156 tracked | none | moving | none | 2 | CC BY 4.0 (per page link) | 328.01 GB | raw .dat/AVI + MAT | IEEE login; subscription wording ambiguous |`

______________________________________________________________________

### Items not verified (section A)

- **Randi 2023 counts:** Neurons segmented or identified per animal; the total number of stimulation events; the number of distinct stimulated neurons. These are likely in Extended Data and the SI, which were not read.
- **Randi 2023 mechanism:** The GUR-3/PRDX-2 activation mechanism and its validation (Supplementary Information not read).
- **Randi 2023 unc-31 activity:** The exact unc-31 locomotion figures were seen only in a page summary, not as a quote.
- **DANDI 001075:** Whether it contains the unc-31 animals, and the meaning of its 223 assets for 113 subjects (some subject apparently lacks one of the two files).
- **Data formats:** The internal format of the OSF `exported_data*.tar.gz` and Hallinen tarballs (not downloaded).
- **wormneuroatlas vs OSF:** Whether the bundled `funatlas.h5` equals the final OSF export.
- **Creamer counts:** The reason for 110 rather than 113 animals, and for 154 vs 156 neurons; the row count of `model_weights.csv`; whether OSF qxhjd is byte-identical to the Randi export; the v3 supplemental weights CSV. No journal version of the Creamer preprint was found.
- **Nguyen 2016 supplement:** Sizes and formats; the PNAS host returned 403. The S3 bucket was not opened.
- **Gauthey 2024:** Every measurement detail; the article and supplement could not be read.
- **IEEE DataPort t6b0-bc36 (Liu 2022):** Licence and size. Kumar 2024 iScience data was not checked.

## B. Flavell lab, WormWideWeb, NeuroPAL and DANDI

Compiled 2026-11-14 from pages opened in this session. Provenance tags: **[host]** = the data host itself (DANDI API, Zenodo API, Dryad API, Harvard Dataverse API, GitHub, WormWideWeb, WormID); **[paper]** = the paper or its preprint (methods, data-availability statement), read as full text via bioRxiv or the Europe PMC REST full-text service; **[2nd]** = second-hand (Europe PMC index metadata, a search-result snippet, a citing paper, a portal describing someone else's data). Sizes are in bytes as the host shows them, with a decimal approximation in brackets. Where a preprint was the only full text I could open, I say so: the published version may differ.

Could not open: cell.com and sciencedirect full text (HTTP 403), pmc.ncbi.nlm.nih.gov (browser check), Europe PMC full-text XML for the two Flavell Cell papers (HTTP 500), the Atanas preprint v2/v3 (403), and the Nature Neuroscience page (login redirect). Two PDFs downloaded but were not machine-readable. The Kramer 2026 full text **was** readable through Europe PMC (PMC13246447).

**Headline for the perturbation question.** In this scope, only two datasets stimulate one neuron and record the rest: Randi et al. 2023 (DANDI 001075, Leifer lab, likely also covered in another section) and Dunn et al. 2025 (DANDI 001623, Kato lab: optogenetic AWA drive plus single-neuron targeting of SMDD, SMDV, RIV and RIA, in immobilised animals). None of the Flavell lab's whole-brain datasets includes optogenetic stimulation during whole-brain imaging. Their perturbations are genetic backgrounds imaged whole-brain: *tdc-1* (Kramer 2026, 17 animals) and *flp-13* + PA14 infection (Pradhan 2025, 8 animals). They also include natural stimuli: a heat pulse (Atanas 2023), a food patch and re-feeding (Dag 2023), and odour and copper boundaries (Kramer 2026, Baskoylu 2026). In Dag 2023, the *tph-1* and serotonin-receptor mutants were used for behaviour only, not whole-brain imaging, according to the preprint v1 methods I read. I could not check the final Cell text.

______________________________________________________________________

### Atanas & Kim et al. (freely moving, NeuroPAL): Atanas, Kim, …, Flavell, 2023, *Cell* 186:4134

- **Verification:** Counts, files, licences and sizes come from the hosts (DANDI, Zenodo, WormWideWeb, WormID). Imaging parameters come from the paper, **bioRxiv preprint v1 only** (the Cell full text was blocked). The paper licence is [2nd] (Europe PMC index). URLs opened:
  - <https://api.dandiarchive.org/api/dandisets/000776/versions/0.241009.1509/info/>
  - <https://api.dandiarchive.org/api/dandisets/000776/versions/0.241009.1509/assets/?page_size=100>
  - <https://zenodo.org/api/records/8185377>
  - <https://zenodo.org/api/records/8185377/versions>
  - <https://zenodo.org/api/records/19388374>
  - <https://github.com/flavell-lab/AtanasKim-Cell2023>
  - <https://www.biorxiv.org/content/10.1101/2022.11.11.516186v1.full> (two chunks)
  - <https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.1016/j.cell.2023.07.035&resultType=core&format=json>
  - <https://wormwideweb.org/about/datasets/>
  - <https://raw.githubusercontent.com/flavell-lab/WormWideWeb-data/main/activity/raw/atanas_kim_2023.csv>
  - <https://wormwideweb.org/activity/api/data/datasets/>
  - <https://www.wormid.org/datasets>
  - Failed: <https://www.cell.com/cell/fulltext/S0092-8674(23)00850-0> (403) and the v2/v3 preprints (403).
- **Measured:**
  - Neurons per animal: WormWideWeb's dataset API gives `n_neuron` 105–163 across its 68 recordings [host]. The preprint v1 gives about 143 ± 12 head neurons per GCaMP animal [paper]. WormID lists the 38 DANDI NWB files as 64–114 segmented and 25–90 identified [host, WormID], a lower count than WormWideWeb's.
  - Identification: NeuroPAL (otIs670, low-brightness integrant), in a subset of animals. Per animal, WormWideWeb gives `n_labeled` 0–115 (0 for animals without NeuroPAL) [host]. The preprint v1 reports about 96 ± 14 identified per NeuroPAL animal, and 77 of 80 head classes recorded across animals [paper, preprint v1]. Low-confidence labels were excluded [paper].
  - Freely moving, on a "sparse food environment" (preprint v1), mounted in concentrated OP50 [paper].
  - Stimuli: a heat pulse from a 1436-nm, 500 mW laser, giving a 10 °C rise for 1 s at 4 or 6 min into the recording (decay τ 0.39 s) [paper, preprint v1]. The rest is spontaneous behaviour.
  - Perturbations: none genetic or optogenetic. Pan-neuronal GFP strains (no GCaMP) serve as motion controls [paper].
  - Behaviour recorded alongside: velocity, head angle, angular velocity, and pharyngeal pumping (optional), with reversal events [host, WormWideWeb].
  - Number of worms: **68 recordings on WormWideWeb**, typed as 14 baseline, 21 baseline+NeuroPAL, 11 heat, 19 heat+NeuroPAL and 3 GFP (manifest CSV) [host]. WormWideWeb says 40 of 68 have NeuroPAL [host]. **DANDI holds 38 subjects, all strain SWF702 (NeuroPAL)** [host]. The preprint v1 introduction says 37 animals [paper], so the published count is larger than the preprint's.
  - Recording length: about 16 min, about 1,600 time points [paper]. WormWideWeb `max_t` is 1544–1615 [host].
  - Volume rate: 1.7 Hz (1.4 Hz with an alternative piezo), 80 z-planes at 0.54 µm [paper, preprint v1].
  - Indicator: NLS-GCaMP7f, nuclear-localised, with NLS-TagRFP-T and NLS-mNeptune2.5 as references. Strains are SWF415 (flvIs17; flvIs18) and SWF702 (flvIs17 + otIs670) [paper, preprint v1].
- **Access:**
  - DANDI 000776: <https://dandiarchive.org/dandiset/000776> (DOI 10.48324/dandi.000776/0.241009.1509). Open, no request needed [host].
  - Zenodo concept DOI 10.5281/zenodo.8150514, latest version <https://zenodo.org/records/19388374>. Open [host].
  - WormWideWeb: per-dataset JSON downloads, plus a paper-level archive at `/activity/api/data/download/paper/atanas_kim_2023/` [host].
- **Licence:**
  - DANDI: "spdx:CC-BY-4.0" (CC-BY-4.0) [host].
  - Zenodo: versions 1–3 show licence id "cc-by-3.0-us". The **latest version (19388374) shows no licence field**, only access "open" [host].
  - WormWideWeb: not stated [host].
  - Paper: "cc by-nc" per the Europe PMC index [2nd].
- **Size and format:**
  - DANDI: 1,011,743,300,552 bytes (~1.01 TB), 38 NWB files of 26.3–27.0 GB each (`_behavior+image+ophys.nwb`). Each file holds bright-field and GCaMP imaging, NeuroPAL structural images, ROI locations and IDs, labelled traces and behaviour [host].
  - Zenodo latest version: 17 files, 32,989,558,105 bytes (~33.0 GB) in total [host]. Among them, `processed_h5.tar.bz2` (568,776,589 bytes) holds the per-animal processed HDF5 traces. The rest are CePNEM fits (`fit_results.jld2.bz2`, 12.4 GB; `relative_encoding_strength.jld2.bz2`, 14.0 GB), UMAP, network weights, NeuroPAL labels (JLD2 and JSON), and scrambled control datasets [host].
  - WormWideWeb JSON: z-scored, raw, F/F20 and F/Fmean traces, behaviour, timing, labels with a confidence value, and encoding [host].
- **Model mapping:**
  - Signal: calcium (nuclear GCaMP7f, a slow low-pass of somatic Ca²⁺), not voltage.
  - Mostly spontaneous, closed-loop activity with behaviour as covariates (velocity, head curvature, pumping). The only exogenous drive is a 1-s head heat pulse (a nociceptive/thermal input; the dataset does not name the receiving neurons).
  - Could constrain:
    - Per-neuron effective timescales. CePNEM's τ posteriors are shipped (`sampled_tau_vals_median.h5`).
    - The readout from neurons to motor variables (forward/reverse, dorsal/ventral, feeding).
    - State-dependent gain changes. The paper reports encoding changes across behavioural states.
    - Correlational structure that a connectome model must reproduce.
  - Gives no causal constraint on individual chemical or gap weights or their signs.
  - Nominally wild type, but see the strain-background issue below.
- **Known quality issues:**
  - Mis-registration is estimated at about 0.3% of frames, from GFP controls [paper, preprint v1].
  - GFP animals show 2.1% false "encoding" against 58.6% in GCaMP animals. A motion correction was derived from the GFP data [paper].
  - Photobleaching was corrected by dividing by a one-parameter exponential, and datasets with very high bleaching were excluded [paper].
  - Head-angle sign: raw HDF5 needs a per-dataset `θh_pos_is_ventral` flip (already applied in the JSON) [host, WormWideWeb].
  - Identification: when several identities were possible, the most likely was used [paper].
  - Count discrepancy: DANDI holds 38 NeuroPAL animals, while WormWideWeb counts 40 NeuroPAL animals among its 68 recordings [host].
  - Strain background: the Kramer 2026 and Baskoylu 2026 methods describe SWF702 as carrying *lite-1* and *gur-3* null mutations [paper]. I did not see this in the Atanas preprint v1 strain list, so check the Cell key-resources table.
- **Table row:** `| Atanas 2023 (DANDI 000776 / Zenodo / WWW) | 105–163 (WWW); ~143±12 (preprint) | NeuroPAL in 40/68 (WWW), ~96±14 ID'd (preprint) | freely moving | 1-s heat pulse (30 recs); GFP controls; no genetic/opto | 68 recs (WWW); 38 on DANDI | DANDI CC-BY-4.0; Zenodo latest: none shown (v1–3 cc-by-3.0-us) | 1.01 TB DANDI; 33.0 GB Zenodo | NWB; HDF5/JLD2; JSON | open |`

### Dag, Nwabudike, Kang et al. (serotonergic system): Dag, …, Flavell, 2023, *Cell* 186:2574

- **Verification:** Hosts give the files, counts and licences (Zenodo, GitHub, WormWideWeb). Imaging and perturbation details come from the paper, **bioRxiv preprint v1 only** (10.1101/2023.01.15.524132). The Cell text, PMC and the lab's PDF were all unreadable. The paper licence and abstract are [2nd] (Europe PMC index). URLs opened:
  - <https://zenodo.org/api/records/19414639>
  - <https://zenodo.org/api/records/7830052>
  - <https://github.com/flavell-lab/NSM_5HTR_analysis>
  - <https://raw.githubusercontent.com/flavell-lab/WormWideWeb-data/main/activity/raw/dag_nwabudike_kang_2023.csv>
  - <https://wormwideweb.org/about/datasets/>
  - <https://wormwideweb.org/activity/api/data/datasets/>
  - <https://www.biorxiv.org/content/10.1101/2023.01.15.524132v1.full> (two chunks)
  - <https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.1016/j.cell.2023.04.023&resultType=core&format=json>
  - Failed: <https://pmc.ncbi.nlm.nih.gov/articles/PMC10484565/>, <https://www.sciencedirect.com/science/article/pii/S0092867423004191> (403), and <https://bpb-us-e1.wpmucdn.com/sites.mit.edu/dist/0/1795/files/2023/05/PIIS0092867423004191.pdf> (unreadable binary).
- **Measured:**
  - Neurons per animal: 128–167 (WormWideWeb `n_neuron`) [host]. The example dataset has 148 [paper].
  - Identification: NeuroPAL in 8 of 15 recordings, 87–110 labelled per labelled animal (WormWideWeb `n_labeled`) [host].
  - Freely moving [paper].
  - Stimuli:
    - 7 "patchEncounter" recordings: a starved worm starts off food and meets a food patch. Event times are in the manifest [host].
    - 8 "neuropal, reFed" recordings: a starved worm is placed on a dense food patch [host].
    - Fasting before mounting: 3–4 h for SWF415, 1.5–2 h for SWF702 [paper].
  - Perturbations:
    - Behaviour only (no whole-brain imaging described in preprint v1): NSM::Chrimson optogenetic activation, *tph-1*, the sextuple serotonin-receptor mutant, single *ser-1*, *ser-4*, *ser-5*, *ser-7*, *mod-1* and *lgc-50* mutants, the "only-one-receptor" quintuples, and all 64 receptor-combination genotypes. Promoter-swap and Cre-dependent *mod-1* rescues were also used [paper, preprint v1].
    - **Whole-brain imaging in a mutant: none described** in the preprint v1 [paper]. Not checked against the final Cell text.
    - A receptor-expression map: six T2A-mNeonGreen knock-in reporters, each annotated with NeuroPAL in 5–10 adults. Each receptor is expressed in 8–24 of 118 classes, and 52 classes express at least one [paper].
  - Behaviour recorded alongside: velocity, head angle, angular velocity and pumping. Pumping was added in the Zenodo v1 update [host].
  - Number of worms: 15 recordings on WormWideWeb [host].
  - Recording length: 16 min for food encounter [paper]. WormWideWeb `max_t` is 799–1600 [host].
  - Volume rate: 1.7 Hz, 80 planes at 0.54 µm [paper].
  - Indicator and strains: NLS-GCaMP7f (nuclear), ratioed to mNeptune2.5. Strains are SWF415 and SWF702 [paper].
- **Access:**
  - Zenodo <https://zenodo.org/records/19414639> (concept DOI 10.5281/zenodo.19414638). Open [host].
  - The original release, Zenodo 7830052, is a GitHub snapshot of flavell-lab/NSM_5HTR_analysis "publication_withdata" [host].
  - WormWideWeb archive at `/activity/api/data/download/paper/dag_nwabudike_kang_2023/` [host].
- **Licence:**
  - Zenodo 19414639: not stated (no licence field) [host].
  - Zenodo 7830052 (the code+data snapshot): "mit-license" [host].
  - GitHub NSM_5HTR_analysis: none shown [host].
  - WormWideWeb: not stated [host].
  - Paper: "cc by-nc" per the Europe PMC index [2nd].
- **Size and format:**
  - Zenodo 19414639: `processed_h5.tar.bz2` 74,555,688 bytes, `NeuroPAL_labels_dict.jld2.bz2` 259,577 bytes, and `neuropal_label.json.bz2` 4,980 bytes [host]. The archive holds HDF5 per animal.
  - Zenodo 7830052: one zip of 82,555,125 bytes [host].
- **Model mapping:**
  - Signal: calcium, not voltage.
  - Food onset is a natural chemosensory/mechanosensory drive. NSM activity can be read from the recording itself, which makes it a measured serotonin-release covariate. The authors report about 45% of the brain varying with NSM [2nd, search snippet of paper text].
  - Could constrain:
    - Where serotonergic coupling lands (which neurons co-vary with NSM).
    - Its sign and timescale. The paper distinguishes SER-4 (phasic) from MOD-1 (tonic) [2nd, Europe PMC abstract].
    - Combined with the receptor-expression map, the placement of serotonin couplings in a neuromodulator layer.
  - The *tph-1* and receptor mutants (global serotonin loss, receptor removal) constrain behaviour only, not neural dynamics, unless the Cell version added mutant imaging (not verified).
  - Gives no causal constraint on synaptic weights. The authors say it is "not clear that these effects all reflect the causal influence of NSM" [paper].
- **Known quality issues:**
  - Neurons below a noise threshold were excluded, under 15% of GCaMP neurons [paper].
  - GFP-based motion correction [paper].
  - Uncertain identities sometimes resolved to the most likely candidate [paper].
  - NSM's long autocorrelation forced shuffle controls [paper].
  - Zenodo v1 fixed a "trace_array" error in 2022-05-02-01-data.h5 [host].
  - Small sample: 7 + 8 animals [host].
- **Table row:** `| Dag 2023 (Zenodo 19414639 / WWW) | 128–167 | NeuroPAL in 8/15; 87–110 labelled | freely moving | food-patch encounter; re-feeding; mutants (tph-1, 5-HT receptors) & NSM::Chrimson behaviour-only (preprint) | 15 recs | Zenodo: not stated; old snapshot mit-license | 74.6 MB (+82.6 MB snapshot) | HDF5, JLD2, JSON | open |`

### WormWideWeb portal: Flavell lab (wormwideweb.org)

- **Verification:** Everything here is \[host\]: the site, its GitHub repositories and its JSON API. URLs opened:
  - <https://wormwideweb.org>
  - <https://wormwideweb.org/about/>
  - <https://wormwideweb.org/about/datasets/>
  - <https://wormwideweb.org/activity/dataset/> (rows render client-side, so none were visible)
  - <https://wormwideweb.org/activity/api/data/datasets/>
  - <https://github.com/flavell-lab/WormWideWeb>
  - <https://github.com/flavell-lab/WormWideWeb/blob/main/dataset_json_readme.md>
  - <https://github.com/flavell-lab/WormWideWeb-data>
  - <https://raw.githubusercontent.com/flavell-lab/WormWideWeb-data/main/activity/papers.json>
  - <https://raw.githubusercontent.com/flavell-lab/WormWideWeb-data/main/activity/dataset_types.json>
  - <https://github.com/flavell-lab/WormWideWeb-data/tree/main/activity/raw>
  - the three manifest CSVs
  - <https://api.github.com/orgs/flavell-lab/repos?per_page=100&sort=updated>
- **What it aggregates:** Freely moving whole-brain recordings from **one lab (Flavell, MIT) and three papers**, 91 recordings in all. The dataset API returns 147 records, because one recording can appear under several types.
  - Atanas & Kim 2023: 68 recordings (40 NeuroPAL; with CePNEM encoding results).
  - Dag, Nwabudike & Kang 2023: 15 recordings (8 NeuroPAL).
  - Pradhan & Madan 2025: 8 recordings (all NeuroPAL).
  - Kramer 2026 and Baskoylu 2026 are **not** on WormWideWeb. Their data are on Dryad.
  - The site also serves 15 connectome datasets: White 1986 (several series), Cook & Jarrell 2019 (hermaphrodite and male), Witvliet 2021 datasets 1–8, and Cook 2020 pharynx.
- **Download options and format:**
  - Paper-level HDF5 bundles (`processed_h5.tar.bz2`) on Zenodo: records 19388374, 19414639 and 19511989.
  - Paper-level JSON archives at `/activity/api/data/download/paper/<paper_id>/`, plus per-dataset JSON downloads.
  - The JSON holds `timing` (mean timestep, confocal timestamps, events), `behavior` (velocity in 0.1 mm/s, head angle in rad, angular velocity in rad/s, pumping in pumps/s, reversal intervals), `gcamp` (`trace_array` z-scored and `trace_array_original`), optional `label` (NeuroPAL label, class, L/R, D/V, region, ROI id, confidence), optional `encoding` (CePNEM), and `metadata` (with SHA-256/BLAKE3 checksums).
  - The dataset API reports `avg_timestep` ≈ 0.01. The JSON readme says the importer stores minutes, so this is about 0.6 s per volume, consistent with 1.7 Hz. That conversion is my inference.
- **API / package:**
  - The Django app's read endpoints include `/activity/api/data/datasets/`, `/activity/api/data/<dataset_id>/<idx_neuron>/`, `/activity/api/data/<dataset_id>/behavior/`, `/activity/api/data/<dataset_id>/encoding/` and `/activity/api/data/download/<dataset_id>/`.
  - Julia packages: `WormWideWebData.jl` (generates the encoding and NeuroPAL JSON files) and CePNEM.jl.
  - The lab also has a `pub_utils` repository (MIT License), described as extracting public datasets into standardised formats. I did not inspect it.
- **Licence / terms:**
  - The site states no licence or terms. Its About page says only to cite Atanas & Kim 2023 and "Please also cite the original datasets you use."
  - The GitHub repositories WormWideWeb, WormWideWeb-data and WormWideWebData.jl show no licence.
  - The underlying data licences are the Zenodo ones: Atanas latest version none shown, Dag none shown, Pradhan "cc-by-4.0". The Pradhan Dryad original is CC0-1.0.
- **Model mapping:** The most convenient single entry point for the Flavell recordings: compact JSON, identified traces and behaviour covariates. It covers only freely moving, natural-stimulus data, with no optogenetic perturbation.
- **Known quality issues:**
  - The raw HDF5 needs the per-dataset `θh_pos_is_ventral` sign flip; the JSON already has it applied.
  - `n_labeled` is 0 for many Atanas records.
  - The NeuroPAL count for Atanas differs from DANDI (40 against 38).
- **Table row:** `| WormWideWeb (aggregator) | 105–176 | NeuroPAL in 56/91 | freely moving | heat; food patch/refeed; PA14 infection in flp-13 bg | 91 recs, 3 Flavell papers | site: not stated; per-record Zenodo licences vary | per-paper tarballs 34 MB–569 MB | HDF5, JSON | open, no registration |`

### Pradhan, Madan et al. (sickness behaviour): Pradhan, Madan, …, Flavell, 2025, *Nat. Commun.* (10.1038/s41467-025-58478-y)

- **Verification:** Files and licences come from the hosts (Dryad, Zenodo, WormWideWeb). Methods come from the paper, through Europe PMC full text PMC11968842 (first chunk only). URLs opened:
  - <https://datadryad.org/api/v2/datasets/doi%3A10.5061%2Fdryad.nvx0k6f2t>
  - <https://zenodo.org/api/records/19511989>
  - <https://raw.githubusercontent.com/flavell-lab/WormWideWeb-data/main/activity/raw/pradhan_madan_2025.csv>
  - <https://wormwideweb.org/activity/api/data/datasets/>
  - <https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.1038/s41467-025-58478-y&resultType=core&format=json>
  - <https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11968842/fullTextXML>
- **Measured:**
  - Neurons and identification: 117–176 neurons recorded and 40–121 identified per animal [paper]. WormWideWeb `n_labeled` reads 82–127 [host]. The two sources disagree.
  - Freely moving [paper].
  - Stimulus and state: animals were on PA14 during imaging, about 20 h after infection [paper].
  - **Perturbation:** a ***flp-13* mutant background** with pathogen infection. Only infected animals already showing quiescence were selected [paper]. ALA, ASI::Caspase, *daf-7*, *aptf-1* and *nlp-22* were tested in behaviour only [paper].
  - Behaviour recorded alongside: locomotion, including active and quiescent bouts [host].
  - Number of worms: 8 [paper; host]. No uninfected whole-brain control [paper].
  - Recording length: 8-min sessions [paper]. WormWideWeb `max_t` is 200–800, with analysis windows in `t_range` [host].
  - Indicator: NLS-GCaMP7f plus NeuroPAL [paper]. Volume rate not stated in the chunk read.
- **Access:**
  - Dryad doi:10.5061/dryad.nvx0k6f2t, open [host].
  - Zenodo <https://zenodo.org/records/19511989>, restructured for WormWideWeb, open [host].
- **Licence:**
  - Dryad: "<https://spdx.org/licenses/CC0-1.0.html%22> (CC0 1.0) [host].
  - Zenodo: "cc-by-4.0" [host].
  - Paper: "CC BY" per the Europe PMC index [2nd].
- **Size and format:**
  - Dryad: 37,678,790 bytes [host].
  - Zenodo: `processed_h5.tar.bz2` 34,369,614 bytes, NeuroPAL labels in JLD2 and JSON, and a restructuring script (`.jl`) [host]. Data are HDF5.
- **Model mapping:**
  - Signal: calcium.
  - The perturbation is a loss of one neuropeptide (FLP-13) combined with a systemic illness state. In model terms, a peptide-producer knockout plus a slow state variable.
  - Could constrain state-dependent (quiescence) modulation, for example ASI's correlation with quiescence. Little constraint on wiring.
- **Known quality issues:**
  - Small n (8).
  - Selection bias: only quiescent animals were selected.
  - No control condition, and a mutant background throughout.
  - ASI activity was raised during quiescence in only 2 of 3 identifiable cases [paper].
- **Table row:** `| Pradhan 2025 (Dryad nvx0k6f2t / Zenodo 19511989 / WWW) | 117–176 | NeuroPAL, 40–121 ID'd (paper) | freely moving | PA14 infection in flp-13 mutant background | 8 | Dryad CC0-1.0; Zenodo cc-by-4.0 | 37.7 MB | HDF5, JSON | open |`

### Kramer et al. (directed turning, tyramine): Kramer, Wan, …, Flavell, 2026, *Nat. Neurosci.* 29:1408 (10.1038/s41593-026-02257-5)

- **Verification:** Data files and licence come from the host (Dryad). Methods and the data-availability statement come from the paper, via the Europe PMC full text PMC13246447 (three chunks). URLs opened:
  - <https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.1038/s41593-026-02257-5&resultType=core&format=json>
  - <https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13246447/fullTextXML> (offsets 0, 100000, 200000)
  - <https://datadryad.org/api/v2/datasets/doi%3A10.5061%2Fdryad.8sf7m0cz2>
  - <https://datadryad.org/api/v2/versions/433175/files>
  - <https://flavell.mit.edu/data/>
  - Failed: <https://www.nature.com/articles/s41593-026-02257-5> (redirect to login).
- **Measured:**
  - Neurons per animal: about 102 identified on average [paper].
  - Identification: NeuroPAL [paper].
  - Freely moving [paper].
  - Stimuli: a sharp octanol (aversive) boundary, and separately diacetyl (attractive) encounters [paper].
  - **Perturbation imaged whole-brain:** a ***tdc-1* mutant background**, 17 animals (strain SWF1002), against 32 wild-type recordings [paper].
  - Behaviour only (not imaged whole-brain): GtACR2 inhibition and CoChR activation of SMB, SAA, SMD, RIV, SIA, AUA and RME; SAA genetic ablation; *unc-103(gf)* silencing; tyramine-receptor single and quintuple mutants (*tyra-2*, *ser-2*, *lgc-55*, *lgc-39*, *tyra-3*); *tbh-1*; RIC and RIM manipulations [paper].
  - Behaviour recorded alongside: turning and locomotion [paper].
  - Recording length: 8–16 min [paper]. Volume rate not stated in the text read.
  - Indicator and strains: NLS-GCaMP7f with NLS-TagRFP. Strains SWF702 or SWF1088, described as carrying *lite-1* and *gur-3* null mutations [paper].
- **Access:** Dryad <https://doi.org/10.5061/dryad.8sf7m0cz2>, open [host]. Code: github.com/flavell-lab/SAAV_Decoding, preprocess_videos and head_detector_unet2d [paper].
- **Licence:**
  - Dryad: "<https://spdx.org/licenses/CC0-1.0.html%22> [host].
  - Paper: "Creative Commons Attribution 4.0 International License" [paper].
- **Size and format:** 1,455,335,690 bytes (~1.46 GB), 82 files [host]. They include per-recording `*-data.h5` (about 10–15 MB each) and multi-worm tracker `.chxTracks.mat` files [host; only the first 20 file names were inspected].
- **Model mapping:**
  - Signal: calcium.
  - A *tdc-1* knockout removes tyramine synthesis, and with it octopamine (background knowledge, not from the pages read). In model terms, a producer knockout of a monoamine pool, imaged brain-wide in freely moving animals. That makes this the main Flavell dataset that tests a neuromodulatory coupling against neural data, not just behaviour.
  - The odour boundaries give a spatially structured chemosensory drive.
  - Could constrain tyramine coupling (sign, placement, timescale) and sequence timing among the SAA, SMB, SMD and RIV turn-sequence neurons.
  - The opto perturbations constrain behaviour only.
- **Known quality issues:**
  - About 2–3% of time points were interpolated after head-detection failures.
  - Mis-registration under 1%.
  - Neurons with fewer than four recordings were excluded.
  - Wide confidence intervals for some wild-type comparisons.
  - The receptor quintuple's phenotype was weaker than *tdc-1*'s, suggesting an unknown receptor. Octopamine conversion could not be excluded [paper].
- **Table row:** `| Kramer 2026 (Dryad 8sf7m0cz2) | ~102 ID'd | NeuroPAL | freely moving | octanol/diacetyl boundaries; tdc-1 mutant (17) vs WT (32) | 49 | CC0-1.0 | 1.46 GB | HDF5 + .mat | open |`

### Baskoylu et al. (aversive-state integrators): Baskoylu, …, Flavell, 2026, bioRxiv (10.64898/2026.06.04.729906)

- **Verification:** Data comes from the host (Dryad). Methods come from the preprint full text (two chunks). The preprint licence "CC BY" is [2nd] (Europe PMC index). URLs opened:
  - <https://datadryad.org/api/v2/search?q=Flavell%20elegans&per_page=50>
  - <https://datadryad.org/api/v2/datasets/doi%3A10.5061%2Fdryad.w9ghx3g4v>
  - <https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.64898/2026.06.04.729906&resultType=core&format=json>
  - <https://www.biorxiv.org/content/10.64898/2026.06.04.729906v1.full> (two chunks)
- **Measured:**
  - Neurons per animal: about 96 recorded on average [paper].
  - Identification: NeuroPAL, imaged after each recording, with automated identification [paper].
  - Freely moving [paper].
  - Stimulus: copper-bordered arenas, with food uniform across the arena [paper].
  - Strain: SWF702 (*lite-1*; *gur-3* background) [paper].
  - The perturbations in the paper are behaviour-only or single-neuron imaging, not imaged whole-brain \[paper\]:
    - CoChR/Chrimson activation of ASH, ADA, AVH and PVQ.
    - TeTx and *unc-103(gf)* silencing.
    - *unc-13*, *nlp-10* with AVH/PVQ rescue, and PVQ-specific *eat-4* knockout.
  - Number of worms: 27 [paper]. Recording length: 16 min [paper]. Indicator: NLS-GCaMP7f normalised to TagRFP [paper].
- **Access:** Dryad <https://doi.org/10.5061/dryad.w9ghx3g4v>, open [host].
- **Licence:** Dryad "CC0-1.0" [host].
- **Size and format:** 320,032,093 bytes [host]. The file list was not inspected.
- **Model mapping:**
  - Signal: calcium.
  - Repeated aversive chemosensory drive (copper; ASH is among the paper's targets).
  - Could constrain slow integration timescales, from seconds to minutes, in identified neurons. The abstract says the integrators "signal through different neurotransmitters", and the perturbations include *nlp-10* (neuropeptide) and PVQ-specific *eat-4* (glutamate), relevant to slow modulator timescales.
- **Known quality issues:**
  - Preprint, not peer-reviewed.
  - The Dryad record has no methods or usage notes [host].
- **Table row:** `| Baskoylu 2026 (Dryad w9ghx3g4v) | ~96 | NeuroPAL | freely moving | copper boundary (aversive); opto/mutants behaviour-only | 27 | CC0-1.0 | 320 MB | not verified | open |`

### Atanas et al. (BrainAlignNet / AutoCellLabeler training data): Atanas, …, Flavell, 2026, *eLife* 14 (10.7554/eLife.108159)

- **Verification:** Data comes from the host (Harvard Dataverse API). The data-availability statement and counts come from the paper, via the Europe PMC full text PMC12912725. URLs opened:
  - <https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.7554/elife.108159&resultType=core&format=json>
  - <https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12912725/fullTextXML> (offsets 150000 and 250000)
  - <https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=doi:10.7910/DVN/8UE0H9>
- **Measured:** Not an activity dataset for modelling. It holds training, validation and test data for registration and cell identification: AutoCellLabeler was trained on 81 human-annotated NeuroPAL images, with 10 for validation and 11 for testing. It also holds GFP data and jellyfish data [paper; host]. The paper reports 99.6% tracking accuracy and 98% labelling accuracy for more than 100 types per animal [2nd, Europe PMC abstract].
- **Access:** <https://doi.org/10.7910/DVN/8UE0H9>, open [host].
- **Licence:** "CC0 1.0" [host]. Paper "CC BY" per the Europe PMC index [2nd].
- **Size and format:** 15 zstd-compressed tar files, 516,150,872,011 bytes (~516 GB) compressed, about 980 GB decompressed [host].
- **Model mapping:** None directly. It sets the identification-error budget for Flavell-pipeline labels, and the training data for re-labelling.
- **Known quality issues:** n/a.
- **Table row:** `| Atanas 2026 eLife (Dataverse 8UE0H9) | n/a (ID training) | NeuroPAL, 98% reported | mixed | none | 81+10+11 annotated images | CC0 1.0 | 516 GB compressed | .tar.zst | open |`

### NeuroPAL (Yemini et al.) whole-brain chemosensory activity: Yemini, …, Hobert, 2021, *Cell* 184:272

- **Verification:** Files and licences come from the hosts (Zenodo, DANDI, WormID). Methods come from the paper, **bioRxiv v2 only** (10.1101/676312v2, first chunk). The Cell full text was not opened. The abstract is [2nd] (Europe PMC). URLs opened:
  - <https://www.biorxiv.org/content/10.1101/676312v2.full>
  - <https://zenodo.org/api/records/3906530>
  - <https://api.dandiarchive.org/api/dandisets/000541/versions/0.241009.1457/info/>
  - <https://api.dandiarchive.org/api/dandisets/000541/versions/0.241009.1457/assets/?page_size=100>
  - <https://api.dandiarchive.org/api/dandisets/000715/versions/0.241009.1514/info/>
  - <https://www.wormid.org/datasets>
  - <https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.1016/j.cell.2020.12.012&resultType=core&format=json>
  - <https://openworm.org/ConnectomeToolbox/Yemini_2021/>
- **Measured:**
  - Coverage: 21 heads (189 neurons) and 21 tails (42 neurons) across animals, with a median of 18 animals per neuron [paper]. WormID lists DANDI 000541 as 166–188 segmented and 164–184 identified per worm [host, WormID].
  - Identification: NeuroPAL. Automated ID was about 86% accurate in the head and 94% in the tail, and above 90% in the head after manual annotation of 8 neurons [paper].
  - Immobilised: 1 mM tetramisole, in a microfluidic chip [paper].
  - Stimuli: 10-s pulses of 160 mM NaCl, 10⁻⁴ 2-butanone and 10⁻⁴ 2,3-pentanedione, in randomised order at 1-min intervals [paper]. A second strain (OH15500, N = 7 heads) was given 200 mM NaCl for 20 s [paper].
  - Perturbations: none.
  - Behaviour recorded alongside: none, since the animals were immobilised.
  - Recording length: 4 min at about 4 Hz, after a NeuroPAL colour volume and 2 min rest [paper].
  - Indicator: pan-neuronal GCaMP6s, nuclear-localised, supplemented with a pansensory (*arrd-4*) driver. Young-adult hermaphrodites [paper].
- **Access:**
  - Zenodo <https://zenodo.org/record/3906530> ("NeuroPAL Data"), named in the paper's data statement [paper; host].
  - DANDI 000541, "NeuroPAL Microfluidic Chip Images and GCaMP activity", described as data from the NeuroPAL paper and from Nejatbakhsh et al. 2020 [host].
  - DANDI 000715 holds the 10 structural reference worms [host].
  - Table S6 holds the primary response data [paper].
  - All open.
- **Licence:**
  - Zenodo 3906530: "mit-license" [host].
  - DANDI 000541 and 000715: CC-BY-4.0 [host].
  - Paper: not stated in the Europe PMC record [2nd].
- **Size and format:**
  - Zenodo: one `NeuroPAL data.zip` of 1,118,208,396 bytes (~1.1 GB) [host].
  - DANDI 000541: 30,525,162,874 bytes (~30.5 GB), 21 `_ophys.nwb` files of about 1.45 GB each, with sessions dated 2019-09 to 2019-11 [host].
- **Model mapping:**
  - Signal: calcium.
  - Brief chemosensory pulses (salt, two attractive odours) delivered to the nose, with every recorded head neuron identified. The paper text I read names ASER's NaCl response; I did not verify which other sensory classes respond. This gives sensory gains and first-order response kinetics, and downstream propagation under a known input.
  - Stimulus repeats per animal are few, and recordings are short (4 min).
  - The paper reports functional–connectome correlations of R² ≤ 2.8% [paper].
  - Used by Ripoll-Sánchez et al. 2023 for the neuropeptide connectome [2nd, OpenWorm Connectome Toolbox page].
- **Known quality issues:**
  - Automated ID error about 14% in the head, worst in the ventral ganglion [paper].
  - Four classes (AVL, RIM, RIS, PVW) have variable NeuroPAL colour [paper].
  - Some NaCl responses differ from earlier literature. The authors attribute this to the high concentration [paper].
  - I could not establish from the host pages whether DANDI 000541's 21 subjects are the same 21 heads as the paper's.
- **Table row:** `| Yemini 2021 NeuroPAL activity (DANDI 000541 / Zenodo 3906530) | 166–188 seg., 164–184 ID'd (WormID) | NeuroPAL (~86% auto head) | immobilised (tetramisole, microfluidic) | NaCl, 2-butanone, 2,3-pentanedione pulses | 21 | DANDI CC-BY-4.0; Zenodo mit-license | 30.5 GB DANDI; 1.1 GB Zenodo | NWB; zip | open |`

### WormID / NWB community effort: Sprague, …, Yemini, Kato, 2025, *Cell Reports Methods* 5:100964 (preprint bioRxiv 2024.04.28.591397)

- **Verification:** The dataset list and hosting come from the hosts (wormid.org, GitHub, DANDI). Dataset counts, accuracy figures and the licence come from the paper (Europe PMC full text PMC11840940, first chunk). URLs opened:

  - <https://wormid.org>
  - <https://www.wormid.org/datasets>
  - <https://github.com/focolab/NWBelegans>
  - <https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:39826553&resultType=core&format=json>
  - <https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11840940/fullTextXML>
  - The DANDI info pages listed below.

- **What was converted to NWB:** According to the paper's Table 1, 118 worms from 5 labs (the text elsewhere says "108 worms from six datasets", a stated inconsistency) [paper]. All are on DANDI:

  | WormID code | DANDI | worms (paper Table 1) | content |
  |---|---|---|---|
  | NP | 000715 | 10 | structural only |
  | EY | 000541 | 21 | structural + activity |
  | HL | 000714 | 9 | structural only |
  | KK | 000692 | 9 | structural + activity |
  | SF | 000776 | 38 | structural + activity |
  | SK1 | 000565 | 21 | structural + activity |
  | SK2 | 000472 | 10 | structural only |

  The wormid.org datasets page adds **RLD = 001623** (Kato lab, 95 worms, "Whole-brain neural dynamics with single cell optical stimulation"). It also lists HL as 10 worms, while the paper and DANDI give 9 [host].

- **Access:** DANDI (open). wormid.org links to the code (focolab/NWBelegans) and to GUI apps.

- **Licence:**

  - wormid.org: only "© 2024 WormID.org".
  - NWBelegans: no licence shown [host].
  - Each dataset carries its own DANDI licence (CC-BY-4.0).
  - Paper: "This is an open access article under the CC BY-NC-ND license" [paper].

- **Size and format:** NWB (HDF5). Sizes are per dandiset; see the sub-table below.

- **Model mapping:** A harmonisation layer, not new physiology. Its value is uniform NWB access to identified traces from 4–5 labs, and a cross-lab identification-error estimate to propagate into fits.

- **Known quality issues:**

  - Cross-lab automated ID accuracy (top-1, averaged): CPD 39%, StatAtlas 62% and CRF_ID 74% after multi-lab training. StatAtlas trained on the original 10 worms reached about 41% on other labs [paper].
  - Colour-space differences between rigs; histogram matching added about 7–9 points [paper].
  - 14 worms were excluded for deformation or artefacts [paper].
  - Ventral and retrovesicular ganglion neurons are labelled less often and are more variable in position [paper].
  - In datasets where few neurons are annotated, annotators labelled the easy ones, which inflates apparent accuracy [paper].

- **Table row:** `| WormID (NWB harmonisation; DANDI 000472/000541/000565/000692/000714/000715/000776 [+001623]) | 43–196 seg. per worm (WormID) | NeuroPAL, 0–193 ID'd | mixed | as per source | 118 (paper) + 95 (001623) | per dandiset CC-BY-4.0; paper CC BY-NC-ND | see DANDI table | NWB | open |`

### Pointers to entries kept in other sections

Section B's pass also wrote full entries for four datasets that another pass covered. To keep one entry per dataset, they live elsewhere. Facts found only by this pass were folded into those entries, under a "Cross-check from section B's pass" bullet.

- **Dunn et al. 2025 (Kato lab), DANDI 001623 / Zenodo 17353307:** see section D, "Dunn et al. (Kato lab)".
- **Seyedolmohadesin et al. 2026, DANDI 000981:** see section D, "Whole-brain chemosensory responses of both sexes".
- **Randi et al. 2023, DANDI 001075:** see section A, entries A1–A3. This pass could not confirm NeuroPAL by name in the text chunk it read. Section A did confirm it, from the Nature HTML: identities were "assigned by hand with NeuroPAL".
- **Kimura lab semi-restricted spontaneous activity, DANDI 000692:** see section D, "Whole-brain spontaneous activity of semi-restricted worms".

### Kato lab NeuroPAL + immobilised activity (SK1, SK2): Dunn, Sprague, (Borchardt, Bubnis), Kato, 2024 (DANDI 000565, 000472)

- **Verification:** All facts are [host]. URLs opened:
  - <https://api.dandiarchive.org/api/dandisets/000565/versions/0.241009.1504/info/>
  - <https://api.dandiarchive.org/api/dandisets/000565/versions/0.241009.1504/assets/?page_size=100>
  - <https://api.dandiarchive.org/api/dandisets/000472/versions/0.241009.1502/info/>
  - <https://www.wormid.org/datasets>
- **Measured:**
  - 000565: NeuroPAL structural images. A subset has calcium imaging of animals immobilised in a microfluidic chip "without stimulation". About 30–50% of neuron centres are labelled.
  - **Conflict:** WormID names SK1 "Whole-brain neural dynamics with single cell optical stimulation", with 20 worms, activity "Some", 78–139 segmented and 30–82 identified.
  - 000472: partially labelled NeuroPAL volumes (structural). WormID lists SK2 with activity "Some", 166–180 segmented and 36–63 identified.
- **Access:** DANDI, open.
- **Licence:** CC-BY-4.0 (both) [host].
- **Size and format:**
  - 000565: 46,250,221,065 bytes, 21 `_ophys.nwb` files of 0.25–13.8 GB.
  - 000472: 18,442,605,394 bytes, 10 files.
  - NWB, with a proposed multichannel-volume extension (000472) [host].
- **Model mapping:** Mainly identification training. Some spontaneous immobilised activity. Whether any 000565 recordings contain optogenetic stimulation is unresolved; inspect the NWB stimulus tables.
- **Known quality issues:**
  - Partial labelling.
  - Three 000565 sessions recur in 001623 [host].
- **Table row:** `| Kato SK1/SK2 (DANDI 000565, 000472) | 78–180 seg. | NeuroPAL, 30–82 ID'd | immobilised | none stated on DANDI (WormID says optical stimulation for SK1) | 21 + 10 | CC-BY-4.0 | 46.3 GB + 18.4 GB | NWB | open |`

______________________________________________________________________

### All C. elegans dandisets on DANDI (searched 2026-11-14)

Searches of `https://api.dandiarchive.org/api/dandisets/?search=<term>`, with hit counts: "elegans" 12, "worm" 7, "NeuroPAL" 11, "Caenorhabditis" 10, "nematode" 3, "whole-brain" 14 (6 worm), "GCaMP" 50 (4 worm), "Flavell" 1, "Leifer" 1, "Zimmer" 0, "Venkatachalam" 3, "Kato" 12 (2 worm) and "SCAPE" 6 (2 worm). The union is **13 C. elegans dandisets, 10 with data**. All values come from each dandiset's info and assets endpoints [host]. Licences are shown as DANDI records them (`spdx:CC-BY-4.0`, displayed as CC-BY-4.0).

| ID | Title | Contact / lab | Size (bytes) | Files | Subjects | Licence | Contents |
|---|---|---|---|---|---|---|---|
| 000449 | Moving_C_Elegans_Corrected_Voleti_2019 | Simko, Peter | 0 | 0 | – | none (empty list; draft "Invalid") | Empty draft. Describes reprocessed SCAPE 2.0 moving-worm volumes (Voleti 2019), but no files were uploaded. |
| 000472 | NeuroPAL volumetric images | Kato, Saul (UCSF) | 18,442,605,394 | 10 | 10 | CC-BY-4.0 | Partially labelled NeuroPAL volumes (WormID SK2) |
| 000541 | NeuroPAL Microfluidic Chip Images and GCaMP activity | Yemini, Ev (Hobert lab) | 30,525,162,874 | 21 | 21 | CC-BY-4.0 | NeuroPAL + immobilised GCaMP6s chemosensory activity (Yemini 2021; Nejatbakhsh 2020) |
| 000565 | C. elegans whole-brain neuroPAL and immobilized calcium imaging | Kato, Saul | 46,250,221,065 | 21 | 21 | CC-BY-4.0 | NeuroPAL + immobilised unstimulated activity in a subset; 30–50% labelled (WormID SK1) |
| 000680 | Whole brain spontaneous activity plus NeuroPAL images of semi-restricted worms | Kimura, Kotaro | 0 | 0 | – | CC-BY-4.0 (draft) | Empty draft ("not publishable"), with the same description as 000692 |
| 000692 | Whole-brain spontaneous GCaMP activity with NeuroPAL cell ID information of semi-restricted worms | Kimura, Koutarou D (Nagoya City Univ.) | 80,537,036,658 | 9 | 9 | CC-BY-4.0 | Spontaneous whole-brain activity, NeuroPAL ID, semi-restricted |
| 000714 | Segmented and labeled NeuroPAL structural images | Chaudhary, Shivesh (Lu lab, Georgia Tech) | 478,075,700 | 9 | 9 | CC-BY-4.0 | Structural only (CRF_ID paper) |
| 000715 | NeuroPAL: Atlas of C. elegans neuron locations and colors in NeuroPAL worm | Yemini, Eviatar | 835,049,434 | 10 | 10 | CC-BY-4.0 | The 10 original NeuroPAL atlas worms, structural only |
| 000776 | Brain-wide representations of behavior spanning multiple timescales and states in C. elegans | Flavell, Steve (MIT) | 1,011,743,300,552 | 38 | 38 | CC-BY-4.0 | Freely moving, NeuroPAL (SWF702), GCaMP7f + behaviour |
| 000981 | Whole-brain chemosensory responses of both C. elegans sexes | Yemini, Eviatar; Venkatachalam, Vivek | 82,267,067,651 | 47 | 47 | CC-BY-4.0 | Partially immobilised, 8 chemosensory cues × 3, both sexes |
| 001075 | Neural signal propagation atlas of Caenorhabditis elegans | Leifer, Andrew M. (Princeton) | 4,074,234,123,623 | 223 | 113 | CC-BY-4.0 | Immobilised, two-photon single-neuron optogenetic stimulation + whole-brain imaging |
| 001450 | Test NeuroPAL SCAPE | Cohn, Raphael | 0 | 0 | – | CC0-1.0 (draft) | Empty test upload ("Testing data upload with worm 29") |
| 001623 | C. elegans whole-brain imaging datasets (NeuroPAL and timeseries) | Dunn, Raymond (Kato lab, UCSF) | 774,224,141,637 | 95 | 95 | CC-BY-4.0 | Immobilised, AWA opto drive + single-neuron SMD/RIV/RIA stimulation (Dunn 2025) |

Full entries above cover the substantial activity sets: 000776, 000541, 000981, 001623, 001075, 000692 and 000565/000472. The structural sets (000714, 000715) constrain no dynamics and matter only for identification.

______________________________________________________________________

### Not verified / open items

- Final Cell texts of Atanas 2023 and Dag 2023, and the Atanas key-resources strain table. Only preprint v1 of each was readable, so the published animal counts, volume rates and any mutant whole-brain imaging in Dag (e.g. *tph-1*) are unconfirmed.
- Whether SWF702 in Atanas 2023 carries *lite-1*; *gur-3*. This is stated for SWF702 in the Kramer and Baskoylu methods.
- Volume rates for Pradhan, Kramer, Baskoylu and Dunn (not in the text read). Per-animal recording length for Kimura 000692.
- File lists for Baskoylu (Dryad) and the remaining 62 Kramer files.
- The Yemini 2021 Cell article licence. Whether DANDI 000541's 21 subjects are the paper's 21 heads.
- Whether any DANDI 000565 recordings contain optogenetic stimulation (the DANDI text says no stimulation; WormID says single-cell optical stimulation).
- No Flavell-lab dataset with optogenetic stimulation during freely moving whole-brain imaging was found, on flavell.mit.edu/data, Dryad, Zenodo, DANDI or WormWideWeb.

## C. Zimmer lab and Samuel lab

Compiled 2026-11-14 with WebSearch/WebFetch only; no data files were downloaded (only readme/description files of a few KB were opened). "Host" = the data repository itself (OSF API JSON, Mendeley Data API, Zenodo record, GitHub API); "paper" = the article's text (methods, legends, data statements) as fetched from PMC, Europe PMC, bioRxiv or a publisher PDF; "second-hand" = a search snippet or someone else's summary. Long full-text pages were read through WebFetch's summarising model, so numbers attributed to "paper" were extracted by that model from the page text. Where I could check them against text I extracted myself (Kato 2015, Nichols 2017 PDFs), they agree.

**Where the Zimmer lab hosts data (host, verified).** All of the lab's whole-brain datasets I found are public OSF projects owned by the single OSF user "Manuel Zimmer" (user id `a64uz`, <https://api.osf.io/v2/users/a64uz/nodes/> lists exactly 7 nodes): `2395t` Kato2015 whole brain imaging data, `cqmnv` Kato2015_Ca++\_Imaging_in_freely_moving_animals, `kbf38` Nichols2017 whole brain imaging data, `za3gt` Skora2018 whole brain imaging data, `9nfhz` Kaplan 2020 whole brain imaging data, `eyj92` Kaplan2020_Ca++\_Imaging_Freely_Moving_Animals, `3vkxn` Uzel_2022_whole_brain_imaging_data. All 7 have `"public": true` and **`node_license: null`, so no licence is stated on any of them**. Code lives at github.com/Zimmer-lab (10 repositories, listed through the GitHub API) and at github.com/uzelkerem/uzel2022. A DANDI search for "Zimmer" returned 0 dandisets. The IMP group page (<https://www.imp.ac.at/groups/manuel-zimmer>) lists no data links. The University of Vienna lab URL it points to (<https://neurodevbio.univie.ac.at/zimmer-research/>) returned HTTP 404.

**Is the Kato 2015 ("WBI") data public, and under what licence?** Yes, it is publicly downloadable without request from OSF `2395t`: I fetched its readme through the public download link with no authentication. **No licence is stated** on the host (OSF `node_license` is null). Re-use by later papers is known to me only second-hand: a search snippet from a 2024 bioRxiv dLDS paper says "Data from Kato et al. (2015) were obtained from Zimmer (2021)".

______________________________________________________________________

### Kato2015-WB: Kato, Kaplan, Schrödel, Skora, Lindsay, Yemini, et al., 2015, Cell 163(3):656–669 (author list as in the host readme)

- **Verification:** The files, sizes, variables, behavioural-state labels and licence come from the host. Neuron counts, rates, indicator, stimulus protocol and n come from the paper (a PDF copy on a University of Vienna course page, from which I extracted the text myself). The cell.com full text returned HTTP 403. URLs opened: <https://api.osf.io/v2/users/a64uz/nodes/> ; <https://api.osf.io/v2/nodes/2395t/files/osfstorage/> ; <https://files.de-1.osf.io/v1/resources/2395t/providers/osfstorage/5d24885da26b34001805f3ef> (readme_Kato2015.txt, via its redirect) ; <https://cognitivescience.univie.ac.at/fileadmin/user_upload/p_cogsci/Literature/Kato_etal_2015_Global_Brain_Dynamics_Embed_the_Motor_Command_Sequence_of_Caenorhabditis%5C_%5C_elegans.pdf> ; <https://www.cell.com/cell/fulltext/S0092-8674(15)01198-4> (403).
- **Measured (paper unless marked):**
  - **Neurons:** "In each recording, we detected 107–131 neurons, covering 55%–67% of expected neurons in the imaging area" (all head ganglia plus the most anterior ventral-cord motor neurons).
  - **Identification:** manual, by "anatomical positions, also in relation to surrounding neurons ... and their activity patterns", with red marker lines crossed in to confirm ambiguous identities. The text says "most of the active neurons" were identified; I found no exact identified fraction. Figure 1 shows ambiguous identities in parentheses, e.g. "(OLQVL)" and "(SABVL)".
  - **Preparation:** immobilised with "1 mM tetramisole in microfluidic devices", recordings started within 5 min of removal from food.
  - **Stimuli:** WT_NoStim held 18 min at constant 21% O2. WT_Stim imaged for 12 min: 6 min at 21%, then 6 min of "30 s consecutive shifts between 4% and 21% O 2". The paper says this protocol reliably activates BAG, URX and AQR.
  - **Perturbation:** AVA silenced acutely by a histamine-gated chloride channel (AVA::HisCl).
  - **Number of worms:** n = 5 WT unstimulated; "n = 13 recordings" with O2 stimulation; AVA::HisCl "n = 5 recordings".
  - **Recording length and rate:** 18 min (unstimulated) or 12 min (stimulated), at "≈2.85 volumes per second".
  - **Indicator:** pan-neuronal nuclear GCaMP5K (strain ZIM504), nuclear-localised.
  - **Behaviour:** none recorded during whole-brain imaging (the animals are immobilised). The host provides state vectors inferred from neural activity: 8 states for WT_NoStim (FWD, SLOW, DT, VT, REV1, REV2, REVSUS, NOSTATE) and 4 for WT_Stim and AVA_HisCl (FWD, REV, REVSUS, TURN).
- **Access:** <https://osf.io/2395t/> (OSF, public). Downloadable without request: yes.
- **Licence:** host: not stated (OSF `node_license: null`). Paper: © 2015 Elsevier Inc.; the paper's licence for re-use was not verified.
- **Size and format (host):** 4 files, 145,074,837 bytes in total. `WT_NoStim.mat` 56,390,975 B (modified 2022-10-05, i.e. re-uploaded after the 2019 deposit), `WT_Stim.mat` 42,652,798 B, `AVA_HisCl.mat` 46,029,658 B, `readme_Kato2015.txt` 1,406 B. MATLAB .mat files with the fields `traces_raw` (uncorrected), `traces` (bleach-corrected), `tracesDif`, `IDs`, `timeVectorSeconds`, `fps`, `dataset`, `stimulus` (identity, type, switchtimes, initialstate, conc, concunits) and `States`. The number of recordings inside each file is not stated in the readme.
- **Model mapping:**
  - **Signal:** calcium (nuclear GCaMP5K ΔF/F0), a slow, low-pass readout of somatic voltage. The paper notes that nuclear localisation omits subcellular Ca2+ signals.
  - **Stimulus in model terms:** the O2 up-shift is a step drive to URX/AQR (and PQR, which is outside the imaged volume). The down-shift drives BAG.
  - **Perturbation in model terms:** AVA::HisCl is an acute chloride shunt on AVA (clamp AVA hyperpolarised or remove its output).
  - **Constrains:**
    - The O2 entrainment of the global cycle: sensory gains of URX/BAG and the downstream weights (the paper reports BAG rises evoking FALL transitions in 30/54 instances).
    - The phase ordering of RISE/FALL across the command interneurons and motor neurons (AVA, AVE, RIM, AIB, RIB, SMD, RIV, VB/DB/VA): recurrent weights, signs and time constants.
    - The AVA-silenced state durations: AVA's contribution to holding the reversal state (gap and chemical coupling onto A-class motor neurons). The paper reports that HIGH-state prolongation depends in part on AVA.
  - **Limitations:** ~40–60 identified neurons per animal. The immobilised dynamics differ from freely moving ones (see below).
- **Known quality issues:**
  - Immobilisation alters dynamics. Prolonged HIGH phases of positive-PC1 neurons were absent in freely moving worms, and "physical immobilization alone also caused RIM high phases" (paper).
  - Only 55–67% of expected neurons were detected (paper).
  - Some identities are ambiguous (paper figure labels in parentheses).
  - Photobleaching is handled by a separate bleach-corrected trace (host readme).
  - Nuclear localisation omits subcellular signals (paper).
- **Table row:** `| Kato2015-WB | 107–131 detected (55–67% of expected) | manual (position + activity + marker lines); fraction not stated | immobilised (tetramisole, microfluidic) | O2 steps 21↔4%; AVA::HisCl | 5 spont + 13 stim + 5 AVA::HisCl recordings (paper) | not stated (OSF) | 145.1 MB (4 files) | MATLAB .mat | OSF, public, no request |`

### Kato2015-FM: Kato et al. 2015 (freely moving single-neuron imaging, same paper)

- **Verification:** The files, sizes and fields come from the host (readme RTF). The protocol comes from the paper PDF (as above). URLs opened: <https://api.osf.io/v2/nodes/cqmnv/files/osfstorage/> ; <https://files.de-1.osf.io/v1/resources/cqmnv/providers/osfstorage/600970b0ba010909a38935ab> (readme, via redirect).
- **Measured:**
  - **Neurons:** single identified neurons per worm (one variable per neuron class, e.g. RIM, AVA, AVE, AIB, SMDV), from Figs 2, 3 and S3 (host readme).
  - **Preparation:** freely crawling on agar in an airtight chamber with gas-flow inlet and outlet, with automatic re-centring tracking (paper).
  - **Indicator:** cytosolic GCaMP plus mCherry in the neuron of interest (paper: "expressed both mCherry and GCaMP in the neuron of interest").
  - **Behaviour (host):** reversals (manual), head position from the stage, speed, and head angles (SMDV only).
  - **Not stated on host:** worm count and frame rate. Time comes from image creation times.
- **Access:** <https://osf.io/cqmnv/> (OSF, public). Downloadable without request: yes.
- **Licence:** host: not stated (`node_license: null`).
- **Size and format (host):** `Freely moving imaging (Kato 2015).mat` 23,756,008 B and a 47,673 B RTF readme. One struct per neuron, one entry per worm. It holds the raw `gcampmcherryratio`, not ΔR/R0; users compute R0 themselves, as the mean of the whole trace for some neurons and the mean of the bottom 10% for others.
- **Model mapping:** ratiometric calcium in single neurons alongside behaviour. Constrains the mapping of neuron activity to the motor state (readout and premotor tuning) in moving animals. It cannot constrain inter-neuron weights.
- **Known quality issues:** the R0 convention differs between neurons (host readme). Some neurons have no head-angle data (host readme).
- **Table row:** `| Kato2015-FM | 1 per worm | yes (targeted promoter) | freely moving | none (O2 chamber available) | not stated | not stated (OSF) | 23.8 MB | MATLAB .mat | OSF, public, no request |`

### Nichols2017: Nichols, Eichler, Latham & Zimmer, 2017, Science 356:eaam6851

- **Verification:** The files, sizes, variables and licence come from the host. The neurons, n and protocol come from the paper (a PDF copy on the Penn NGG site, from which I extracted the text myself). The GCaMP variant, volume rate and immobilisation agent were not verified: they are in the Supplementary Materials, which I did not open. URLs opened: <https://api.osf.io/v2/nodes/kbf38/> ; <https://api.osf.io/v2/nodes/kbf38/files/osfstorage/> ; <https://api.osf.io/v2/nodes/kbf38/contributors/> ; <https://files.de-1.osf.io/v1/resources/kbf38/providers/osfstorage/5d24879645253a001b3994a8> (readme, via redirect) ; <https://www.med.upenn.edu/ngg/assets/user-content/documents/paper1-a-global-brain-state-underlies-c.-elegans-sleep-behavior.pdf>
- **Measured (paper):**
  - **Neurons:** "a typical recording encompassed 80 to 130 neurons, of which the cell class of 20 to 41% could be identified" (manual; ambiguous identities flagged).
  - **Preparation:** immobilised in microfluidic devices that control O2, with a nuclear-localised pan-neuronal Ca2+ indicator. The variant was not verified.
  - **Animals:** L4 animals at prelethargus or lethargus, assayed in the presence of bacterial food.
  - **Stimuli:** 6 min at 10% O2, 6 min at 21% O2, 6 min at 10% O2 (an 18-min protocol).
  - **Perturbation (genetic):** npr-1 (the neuropeptide receptor; the paper places the arousal circuit in RMG and the O2 sensors) versus N2. Behavioural experiments only: genetic ablation of AQR/PQR/URX and npr-1 rescues.
  - **Number of worms:** npr-1 prelethargus n = 10, lethargus n = 11; N2 prelethargus n = 10, lethargus n = 12 ("n = 10 to 12 different individuals for each condition").
  - **Behaviour:** none (immobilised). Brain states are inferred from neurons (AVA, SMDD, RIV, RIS, etc.).
- **Access:** <https://osf.io/kbf38/> (OSF, public; contributor: Manuel Zimmer). Downloadable without request: yes.
- **Licence:** host: not stated (`node_license: null`). The paper's licence was not verified.
- **Size and format (host):** 5 files, 284,293,139 B in total: `npr1_prelet.mat` 62,603,879, `npr1_let.mat` 81,053,212, `n2_let.mat` 79,195,250, `n2_prelet.mat` 61,439,723, readme 1,075. MATLAB fields: `traces` (ΔF/F0), `tracesDif`, `IDs`, `timeVectorSeconds`, `fps`, `trialname`, `stimulus` (as in Kato), and `fourStates`/`fourStatesNames` and `FiveStates` (turns split dorsal/ventral/undefined).
- **Model mapping:**
  - **Signal:** calcium.
  - **Stimulus in model terms:** the O2 step drives URX/AQR/PQR.
  - **Perturbations in model terms:** the lethargus vs prelethargus contrast is a global neuromodulatory state change (the paper reports active neurons dropping from ~40% to ~10% during quiescence, and RIS active during quiescence). npr-1 loss of function removes NPR-1 neuropeptide-receptor inhibition, chiefly on RMG.
  - **Constrains:** neuromodulatory coupling and state parameters (RIS/sleep drive, the global gain), the O2 sensory gain, and arousal pathways, contrasting the npr-1 and N2 sensory-motor transformation.
  - **Limitation:** few identified neurons per animal (20–41%).
- **Known quality issues:** low identified fraction (paper). Larval (L4) animals, so their neuron positions differ from the adult atlas (my inference, not from a source). Stage-dependent comparisons rest on n = 10–12 per arm (paper).
- **Table row:** `| Nichols2017 | 80–130 | manual, 20–41% | immobilised (microfluidic) | O2 10↔21%; npr-1 vs N2; lethargus vs prelethargus | 43 recordings (10/11/10/12) | not stated (OSF) | 284.3 MB (5 files) | MATLAB .mat | OSF, public, no request |`

### Skora2018: Skora, Mende & Zimmer, 2018, Cell Reports 22:953–966

- **Verification:** The files, sizes, variables and licence come from the host. The method details come from the paper (Europe PMC full-text XML of PMC5846868). The PMC HTML page served a reCAPTCHA. I found no data-availability statement or recording length in the fetched text. URLs opened: <https://api.osf.io/v2/nodes/za3gt/files/osfstorage/> ; <https://files.de-1.osf.io/v1/resources/za3gt/providers/osfstorage/5d248967114a42001702233a> (readme, via redirect) ; <https://www.ebi.ac.uk/europepmc/webservices/rest/PMC5846868/fullTextXML> (both halves) ; <https://pmc.ncbi.nlm.nih.gov/articles/PMC5846868/> (reCAPTCHA).
- **Measured (paper):**
  - **Neurons:** per-recording neuron count not stated in the text I read. Identification is manual; only some sensory identities were certain (ASK, AWC) and URA/FLP were unresolved.
  - **Preparation:** immobilised with 1 mM tetramisole in microfluidic devices.
  - **Stimulus:** "the same protocol as used for behavioral experiments to switch between 21% and 10% O2". Step durations were not stated for imaging.
  - **Perturbations:** energy state (WT fasted vs WT starved) and the insulin-receptor mutant daf-2(e1370) fasted. daf-16;daf-2, cell ablations and rescues were behavioural only.
  - **Number of worms:** "Number of recordings (n) = 6 each" (Fig. 5) and "n = 6" (Fig. 6).
  - **Indicator and rate:** NLS-GCaMP5K (nuclear-localised) at 1.6–2.6 volumes/s.
  - **Recording length:** not verified.
  - **Behaviour:** freely moving assays separately. Imaging-derived states on host: REV, FWD, QUI (quiescent), TURN.
- **Access:** <https://osf.io/za3gt/> (OSF, public). Downloadable without request: yes.
- **Licence:** host: not stated (`node_license: null`). Paper: "open access article under the CC BY-NC-ND license".
- **Size and format (host):** 4 files, 40,595,718 B: `daf2_fasted.mat` 12,355,416, `WT_fasted.mat` 14,858,890, `WT_starved.mat` 13,380,486, readme 926. Fields: `traces` (bleach-corrected), `IDs`, `timeVectorSeconds`, `fps`, `dataset`, `stimulus`, `FourStates`/`FourStateKey`.
- **Model mapping:** calcium. The starvation-induced brain-wide quiescence and its dependence on daf-2 are a neuromodulatory (insulin) state variable acting on global excitability. The O2 step drives URX/AQR/PQR. Constrains neuromodulatory gain and state terms and the O2 sensory gain; it does not resolve individual synaptic weights.
- **Known quality issues (paper):** O2 modulation of reversals was weaker under imaging than in behaviour. Some sensory identities are ambiguous. The paper also suggests daf-2 animals may be more resilient to immobilisation stress (taken from the summarised paper text).
- **Table row:** `| Skora2018 | not stated | manual; fraction not stated | immobilised (tetramisole, microfluidic) | O2 21↔10%; fasted vs starved; daf-2 | n = 6 per condition (paper) | not stated (OSF) | 40.6 MB (4 files) | MATLAB .mat | OSF, public, no request |`

### Kaplan2020-WB: Kaplan, Salazar Thula, Khoss & Zimmer, 2020, Neuron 105:562–576.e9 (immobilised whole-brain and whole-nervous-system)

- **Verification:** The files, sizes, dataset structure and variables come from the host (Readme.rtf). The method details, n and the data statement come from the paper (PMC7014571 HTML, read in two parts). URLs opened: <https://api.osf.io/v2/nodes/9nfhz/files/osfstorage/> ; <https://files.de-1.osf.io/v1/resources/9nfhz/providers/osfstorage/6233ac621372e807a5cc4c43> (readme, via redirect) ; <https://pmc.ncbi.nlm.nih.gov/articles/PMC7014571/> (offsets 0 and 100000).
- **Measured (paper):**
  - **Neurons, head ganglia:** 113–129 per recording (~50–66% of expected), spinning disk, 1.55–2.23 volumes/s.
  - **Neurons, whole nervous system (head + ventral cord + tail):** 103–129 per recording (~39–43% of expected), ~3.03 volumes/s, deconvolved.
  - **Identification:** manual, with ambiguous identities (URY, RMD, SIB). Unidentified neurons carry numbers as IDs (host).
  - **Preparation:** immobilised in a microfluidic device with tetramisole and histamine, 30 min at 21% O2. Young adults, lite-1(ce314) background.
  - **Perturbations:**
    - **RIS::HisCl** is present in the whole-brain lines, expressed in "the quiescence-promoting neuron RIS" "to increase forward command state occurrence".
    - **SMD::HisCl** (Cre-lox).
    - **VNC cholinergic motor neurons::HisCl** ("VNCACh::HisCl").
  - **Number of worms:** n = 5 control, n = 5 SMD::HisCl, n = 4 VNCACh::HisCl head-ganglia recordings (Fig. 4A), and n = 5 whole-nervous-system datasets.
  - **Recording length:** 30 min.
  - **Indicator:** pan-neuronal NLS-GCaMP6f (Punc-31; nuclear).
  - **Host dataset structure:** `SMDhisCl_RIShisCl` and `MNhisCl_RIShisCl` are whole-brain only. `RIShisCl` holds whole-nervous-system recordings in rows 1–5 and whole-brain recordings in rows 6–10.
- **Access:** <https://osf.io/9nfhz/> (OSF, public). Downloadable without request: yes. The paper's own statement reads "Data and custom code generated in this study are available from the Lead Contact without restriction"; the OSF node was created later (2022-03-17).
- **Licence:** host: not stated (`node_license: null`). Paper: CC BY 4.0 (© 2019 The Author(s)).
- **Size and format (host):** `Kaplan2021WBdata.zip` 252,376,497 B plus `Readme.rtf` 2,682 B. Fields:
  - `raw_traces`, `traces_bleach_corrected`, `traces_derivatives`, `traces_bleach_corrected_detrended`
  - `volumes_per_second`, `time_vector`, `neuron_ID`
  - `five_state_annotation` (FWD, REV, REVSUS, DT, VT, UT; 4 states for SMDhisCl)
  - `peaks`, derivative parameters, and the state reference neurons.
- **Model mapping:**
  - **Signal:** calcium.
  - **Perturbations in model terms:** each HisCl line is an acute chloride shunt (node silencing) on the named neurons. SMD silencing tests head-motor feedback into the global cycle. VNC cholinergic motor-neuron silencing tests feedback from the motor periphery; the paper studies nested timescales of head casts and body bends within the forward/reverse states. The RIS::HisCl background is a permanent modification to model as well.
  - **Constrains:** the whole-nervous-system recordings constrain head ↔ ventral cord ↔ tail coupling (AVA/AVB to A/B-class motor neurons, gap junctions). SMD and motor-neuron silencing constrain feedback weights and the time constants of nested oscillations.
- **Known quality issues (paper):**
  - About 50% of SMD::HisCl animals showed residual SMD activity and were excluded.
  - In VNCACh::HisCl animals DB01/DB02 were identified by activity only.
  - Partial coverage (~39–66%) and ambiguous identities.
  - Volume rates of 1–3 Hz may miss fast dynamics.
  - ~3.3% of traces were excluded for detrending distortion.
  - Frequencies are ~10-fold lower in immobilised than in moving animals.
- **Table row:** `| Kaplan2020-WB | 113–129 head; 103–129 whole-NS | manual; fraction not stated | immobilised (tetramisole + histamine) | HisCl: RIS (background), SMD, VNC cholinergic MNs | 5 ctrl + 5 SMD + 4 MN head; 5 whole-NS (paper) | not stated (OSF) | 252.4 MB (zip) | zipped (content format not listed; fields per readme) | OSF, public, no request |`

### Kaplan2020-FM: Kaplan et al. 2020 (freely moving targeted imaging, same paper)

- **Verification:** The files, sizes and field list come from the host (readme RTF). The rate, n and indicator come from the paper (PMC7014571). URLs opened: <https://api.osf.io/v2/nodes/eyj92/files/osfstorage/> ; <https://files.de-1.osf.io/v1/resources/eyj92/providers/osfstorage/6009726fe80d370a0ca58deb> (readme, via redirect) ; PMC7014571 as above.
- **Measured:**
  - **Neurons:** SMDD, SMDV and DB02, recorded per worm (targeted; ratiometric GCaMP/mCherry).
  - **Preparation:** freely moving under coverglass on agar.
  - **Rate and length:** ~30 Hz (host: "timevec" of roughly 33 ms per frame), ~8-min recordings after 5-min acclimation (paper).
  - **Number of worms:** 11 SMD and 10 DB02 animals; 14–15 SMD animals in AIB::HisCl (paper).
  - **Perturbations (host folder list):** AIB::HisCl (with histamine and control subfolders); mutants lgc-55, tdc-1 and unc-25; RIA::TeTx; RIA::HisCl.
  - **Behaviour (host):** body angles (24 points), reversals, propagated bends, head casts, XY path and speed.
- **Access:** <https://osf.io/eyj92/> (OSF, public). Downloadable without request: yes.
- **Licence:** host: not stated (`node_license: null`).
- **Size and format (host):** `Freely moving imaging (Kaplan 2020).zip` 193,093,534 B plus a 60,124 B RTF readme. It holds raw calcium-imaging and behaviour movie files plus one `wormstruct` .mat per worm (ΔR/R0 for smdd/smdv/db2). The readme says "no extracted data" in the raw folder.
- **Model mapping:** calcium in single neurons during behaviour. The mutants remove a tyramine-gated chloride channel (lgc-55), tyramine/octopamine synthesis (tdc-1) or GABA synthesis (unc-25). TeTx in RIA removes RIA's chemical output; the HisCl lines silence AIB or RIA. Constrains SMD/RIA head-motor circuit weights and signs and the monoamine/GABA couplings onto SMD.
- **Known quality issues:** XYpath and Speed are not available for all recordings (host readme). Timing is irregular, so use `timevec` (host readme).
- **Table row:** `| Kaplan2020-FM | 1–3 targeted | yes (targeted) | freely moving | AIB/RIA HisCl, RIA TeTx, lgc-55, tdc-1, unc-25 | ~10–15 per condition (paper) | not stated (OSF) | 193.2 MB (zip) | movies + MATLAB .mat | OSF, public, no request |`

### Uzel2022: Uzel, Kato & Zimmer, 2022, Current Biology 32:3443–3459.e8

- **Verification:**
  - **Host:** files, sizes and fields (OSF readme RTF).
  - **Code repository:** the WT dataset count and units come from comments in the analysis scripts on GitHub.
  - **Abstract:** via Europe PMC.
  - **Not obtained:** the paper's methods. cell.com, sciencedirect.com and the Phaidra PDF all blocked the fetch (403, or an Anubis bot check). Indicator, volume rate, recording length, n per HisCl line and whether recordings were made with histamine are therefore **not verified**.
  - **URLs opened:** <https://api.osf.io/v2/nodes/3vkxn/files/osfstorage/> (and ?page=2) ; <https://files.de-1.osf.io/v1/resources/3vkxn/providers/osfstorage/62a212c27bb4560b95452070> (readme, via redirect) ; <https://api.github.com/repos/uzelk88/uzel2022> (redirected to uzelkerem/uzel2022) ; <https://github.com/uzelkerem/uzel2022> ; <https://raw.githubusercontent.com/uzelkerem/uzel2022/main/Fig1D.m> ; <https://raw.githubusercontent.com/uzelkerem/uzel2022/main/Fig6A_B_E.m> ; <https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:35809568..>. ; <https://uscholar.univie.ac.at/detail/o:1642746> ; <https://api.semanticscholar.org/graph/v1/paper/DOI:10.1016/j.cub.2022.06.039> ; <https://uzelkerem.github.io/publications> ; <https://www.cell.com/current-biology/fulltext/S0960-9822(22)01001-6> (403) ; <https://www.sciencedirect.com/science/article/pii/S0960982222010016> (403) ; <https://phaidra.univie.ac.at/api/object/o:1642746/download> (bot check).
- **Measured:**
  - **Neurons:** whole-brain, immobilised (inferred from the Zimmer-lab pipeline and the host fields; the paper's methods were not read). Neurons per animal: not verified. Identification: manual (host field "identified neuron IDs"). The code loads a `NeuronOrder` list of identified neurons, and "InputTraces contain DF/F0 traces of identified neurons" (code comment).
  - **Wild type:** "6 WT datasets" (code comment in Fig1D.m).
  - **Perturbations (host file names):** HisCl lines in AVA, AVE, AIB, RIB, RIM, PVC and AVB, plus the combinations AVB+RIB+AIB, RIM+PVC and AVA+AVE+RIM+PVC. The abstract says "inhibition of multiple hub neurons specifically disrupts brain-wide correlations".
  - **Behaviour:** inferred states for WT only (states key 1 FWD, 2 REV, 3 REVSUS, 5 DT, 6 VT).
- **Access:** <https://osf.io/3vkxn/> (OSF, public). Downloadable without request: yes. Code: <https://github.com/uzelkerem/uzel2022> (MATLAB; it includes connectome matrices `Neuro279_EJ.mat` and `Neuro279_Syn.mat`).
- **Licence:** host: not stated (`node_license: null`). The GitHub repo has no licence (API `license: null`). Paper: CC BY 4.0 per the University of Vienna u:scholar record and "CC BY" per Europe PMC and Semantic Scholar.
- **Size and format (host):** 12 files, 364,879,915 B in total:
  - `Uzel_WT.mat` 47,923,875
  - Single-neuron lines: `AVA_His.mat` 30,635,471; `AIB_His.mat` 29,529,472; `PVC_His.mat` 40,171,285; `RIB_His.mat` 30,114,925; `RIM_His.mat` 32,578,796; `AVB_His.mat` 30,025,719; `AVE_His.mat` 22,482,353
  - Combinations: `AVBRIBAIB_His.mat` 36,380,546; `RIMPVC_His.mat` 29,024,496; `AVAAVERIMPVC_His.mat` 36,007,736
  - RTF readme 5,241
  - Fields: `traces` (bleach-corrected), `derivatives`, `options`, `IDs`, `tv`, `fps`, `states` (WT only), `statesKey`.
- **Model mapping:** calcium. The most directly "perturbational" Zimmer dataset: chronic-transgene, acute-histamine silencing of single hubs and hub sets, i.e. node silencing in the model. The paper compares connectome graph features with functional correlations, and its code ships the 279-neuron gap and chemical matrices. Constrains the effective coupling through hubs: which edges carry the correlation structure, and the relative weight of gap and chemical paths through AVA/AVE/AIB/RIB/RIM/PVC/AVB. It does not give stimulus-evoked responses.
- **Known quality issues:** not verified (methods not read). The readme spells "Whole_brain_imagin_datasets" (cosmetic).
- **Table row:** `| Uzel2022 | not verified | manual (IDs field) | immobilised (inferred; not verified) | HisCl silencing of 7 hubs singly + 3 combinations | 6 WT (code comment); per-line n not verified | not stated (OSF) | 364.9 MB (12 files) | MATLAB .mat | OSF, public, no request |`

### Fieseler2025: Fieseler, Lev, Rey, Hille, Brenner & Zimmer, bioRxiv 2025 (10.1101/2025.03.09.642241), "An intrinsic neuronal manifold underlies brain-wide hierarchical organization of behavior in C. elegans"

- **Verification:** All facts are from the preprint (bioRxiv v1 HTML, read in three parts) and the lab's GitHub (`wbfm` README and `docs/paper_datasets.md`). No data host exists. URLs opened: <https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=AUTH:%22Zimmer> M"... ; <https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.1101/2025.03.09.642241..>. ; <https://www.biorxiv.org/content/10.1101/2025.03.09.642241v1.full> (offsets 0, 100000, 165000) ; <https://europepmc.org/api/fulltextRepo?pprId=PPR987859..>. (403) ; <https://api.github.com/orgs/Zimmer-lab/repos?per_page=100> ; <https://github.com/Zimmer-lab/wbfm> ; <https://github.com/Zimmer-lab/wbfm/blob/main/docs/paper_datasets.md>
- **Measured (preprint):**
  - **Neurons:** whole-brain, about 118–161 tracked per freely moving example, with ~37% identified per dataset (61 neurons, 38 classes after left/right pooling). Identified by anatomy, position and activity, with NeuroPAL stacks from immobilised animals (n = 6) used to resolve ambiguities.
  - **Freely moving (paper_datasets.md):** GCaMP n = 36 (strain ZIM2165), GFP motion control n = 8 (ZIM2319), gcy-9;gcy-31;gcy-35 mutant n = 7. Isotropic agar arena; no stimulus.
  - **Immobilised (paper_datasets.md):** GCaMP n = 10 at constant O2 and n = 10 with O2 cycles (11%/21%), mutant + O2 n = 4, HisCl + O2 n = 4. Immobilised by tetramisole or by muscle HisCl.
  - **Indicator:** pan-neuronal nuclear GCaMP7b with nuclear mScarlet reference.
  - **Rate and length:** ~3.5 volumes/s freely moving (~2 Hz immobilised), with behaviour at 83.3 Hz. Most recordings 8 min; immobilised O2 recordings 20 min.
  - **Behaviour:** forward/backward, turns, pauses, velocity, curvature and eigenworms.
- **Access:** **not public.** The preprint says "Data will be available upon publication". The wbfm repo docs say raw data are needed but name no host. No DANDI, OSF or Zenodo deposit found.
- **Licence:** data: n/a (not released). Code `Zimmer-lab/wbfm`: MIT (GitHub API). Preprint licence: not verified.
- **Size and format:** not available. The repo documents an NWB format (`docs/nwb_format.md`, not opened).
- **Model mapping:** calcium (nuclear GCaMP7b) during free behaviour, with a GFP control for motion artefacts. The gcy-9/31/35 mutants remove O2/CO2 sensory input (sensory-gain knockout). The immobilised vs freely moving contrast separates intrinsic dynamics from reafference. Would constrain intrinsic recurrent dynamics versus proprioceptive and sensory feedback gains.
- **Known quality issues (preprint):** "considerably more motion artifacts in freely moving animals", including image tearing within a volume during fast movement. Tracking outlier detection "degrades rapidly" with many mistakes. An earlier GCaMP6f line was "insufficiently bright".
- **Table row:** `| Fieseler2025 | ~118–161 tracked | ~37% (manual + NeuroPAL reference) | freely moving + immobilised | gcy-9;31;35 mutant; muscle HisCl; O2 11↔21% (immobilised) | 36+8+7 moving; 10+10+4+4 immobilised | n/a (unreleased) | n/a | (NWB planned, not verified) | not public ("upon publication") |`

### Madhusudhanan2026: Madhusudhanan, Parinov, Fieseler & Zimmer, bioRxiv 2026 (10.64898/2026.09.03.749061), "Flexible modulation of neuronal population dynamics drives variable decision-making in C. elegans"

- **Verification:** All facts come from the preprint (bioRxiv v1 HTML, read in three parts) and its Europe PMC abstract. I found no data or code availability statement in the HTML. URLs opened: <https://www.biorxiv.org/content/10.64898/2026.09.03.749061v1.full> (offsets 0, 100000, 200000) ; Europe PMC search as above.
- **Measured (preprint):**
  - **Neurons:** 105–185 detected per recording. 82 neuron types identified in total, each type in 1 to 9 animals, using "previously reported heuristics" (NeuroPAL not mentioned).
  - **Preparation:** immobilised in a 2-layer microfluidic device, with pan-muscle HisCl plus histamine. npr-1(ad609) background.
  - **Stimulus:** 450 s at 11% O2, then 16 trials of 60 s at 11% / 30 s at 21% O2.
  - **Number of worms:** N = 9 imaged. Recording length 31.5 min at 3 volumes/s.
  - **Indicator:** nls-GCaMP6f.
  - **Perturbations (behaviour only, not during imaging):** HisCl in BMNs, RIB, DVA and RIS; Chrimson in RIB, DVA and RIS.
  - **Behaviour:** recorded separately in freely moving animals (N = 25, 322 trials).
- **Access:** not found (no repository named in the HTML I read).
- **Licence:** not stated / not found.
- **Size and format:** not found.
- **Model mapping:** calcium. Repeated O2 up-steps (URX/AQR/PQR drive) with trial-to-trial variable responses. Constrains sensory gain, the slow state variable that biases choice (neuromodulatory input per the abstract) and the interneuron/motor readout. The optogenetic and HisCl results are behavioural targets only.
- **Known quality issues:** no imaging-quality statements were extracted. The 82 identified types are a union across animals, not per animal.
- **Table row:** `| Madhusudhanan2026 | 105–185 | heuristics; 82 types across 9 animals | immobilised (muscle HisCl) | O2 11↔21% (16 trials); npr-1 background | 9 | not found | not found | not found | not found |`

### Venkatachalam2016: Venkatachalam, Ji, Wang, Clark, Mitchell, et al. (Samuel lab), 2016, PNAS (10.1073/pnas.1507109113), "Pan-neuronal imaging in roaming Caenorhabditis elegans"

- **Verification:** All facts come from the paper (PMC4776525 HTML). I found no data host. My web-search budget ran out before I could search further for a deposit. URLs opened: `https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=(AUTH:"Samuel AD"...)...` ; <https://pmc.ncbi.nlm.nih.gov/articles/PMC4776525/> ; <https://www.ebi.ac.uk/europepmc/webservices/rest/PMC4776525/fullTextXML> (HTTP 500) ; <https://api.github.com/orgs/samuellab/repos?per_page=100> (first 100 KB read; no matching repo seen).
- **Measured (paper):**
  - **Neurons:** about 80 per animal (84 nuclear regions in the main animal). 8 named identities (AFDL/R, AVB, RMEL, AIBR, AVAL, AVDL, RIML), assigned by atlas registration with manual proofreading and DIC overlay.
  - **Preparation:** freely moving on agar under a coverslip, with a microbead-immobilised comparison.
  - **Stimulus:** sinusoidal temperature oscillation, 15–17.5 °C (thermoelectric stage).
  - **Perturbations:** none.
  - **Number of worms and length:** 4 worms plus 1 GFP control worm, recorded 1–3 min each (60-s analysed segments; the 84-neuron analysis covers 216 s).
  - **Rate:** 10 volumes/s, 20 planes × 2 µm.
  - **Indicator:** cytosolic GCaMP6s (rgef-1 promoter) with nuclear-localised TagRFP (rab-3) as a calcium-insensitive reference. GCaMP is **not** nuclear.
  - **Behaviour:** forward/backward movement and head velocity, with posture reconstructed from the head trajectory.
- **Access:** no data-availability statement found. Code was to be released on GitHub "on publication", with no repository named. Downloadable: not found.
- **Licence:** data: not found. Paper: not stated in the fetched page.
- **Size and format:** not found.
- **Model mapping:** calcium. The thermal sinusoid is a drive to AFD (the paper reports only two neurons with clear temperature-locked responses). With ~80 neurons, 8 identified and n = 4, it constrains little beyond the AFD sensory gain and the motor-state correlates of AVA/AVB/AIB/RIM.
- **Known quality issues (paper):** motion artefacts only partly corrected, by regressing out the red channel. Signal attenuates away from the objective. Z-feedback runs at 2 Hz with ~0.5 s latency. Index-to-identity mapping can vary between animals. Most identified neurons are on the left side. Only somatic calcium is measured.
- **Table row:** `| Venkatachalam2016 | ~80 | 8 named (atlas + proofreading) | freely moving | thermal sinusoid 15–17.5 °C | 4 (+1 GFP) | not found | not found | not found | not found |`

### Susoy2021: Susoy, Hung, Witvliet, Whitener, Wu, et al. (Samuel lab), 2021, Cell (10.1016/j.cell.2021.08.024), "Natural sensory context drives diverse brain-wide activity during C. elegans mating"

- **Verification:** The licence, files, sizes and format come from the host (Mendeley Data landing page and public API; the data-description RTF was opened). The method details come from the paper (PMC8488019 HTML, first 100 KB). URLs opened: <https://pmc.ncbi.nlm.nih.gov/articles/PMC8488019/> ; <https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8488019/fullTextXML> (HTTP 500) ; <https://data.mendeley.com/datasets/922xsfxh2g/1> ; <https://data.mendeley.com/public-api/datasets/922xsfxh2g/files?folder_id=root&version=1> ; <https://data.mendeley.com/public-files/datasets/922xsfxh2g/files/0c5158d7-0b3a-4d58-af5b-36ad44c4e2bd/file_downloaded> (redirected to an S3 object, opened).
- **Measured (paper):**
  - **Neurons:** nearly every neuron of the male posterior (tail) brain, imaged without the head. 76 identifiable neurons across datasets (57 types after pairing L/R). Identified by position, morphology and markers, cross-checked against Sulston 1980 and EM.
  - **Preparation:** freely swimming and mating males, tail tracked by moving the stage.
  - **Stimulus:** natural hermaphrodite partners (2–5 per recording).
  - **Perturbations:**
    - **Laser ablations:** PVV, PDB, R2B, HOA, PCA, PCB/PCC, with mock-ablated controls.
    - **Mutants:** vulvaless let-23 hermaphrodites; ceh-14 males.
  - **Number of worms:** 8 comprehensive datasets (5–10 min continuous), 14 targeted datasets and 23 datasets from ablated males.
  - **Rate:** 10 volumes/s, 20 sections × ~1.75 µm.
  - **Indicator:** pan-neuronal nuclear GCaMP6s::3xNLS with mNeptune reference (ZM9624).
  - **Behaviour:** rich ethogram and continuous features (velocity, tail curvature, spicules, position relative to the vulva).
- **Access:** <https://doi.org/10.17632/922xsfxh2g.1> (Mendeley Data, published 2021-09-10). Downloadable without request: yes (the RTF fetched without authentication). Interactive browser: <http://mating-data.worm.world/index.html> (named in the paper; not opened). Code: "available from the Lead Contact upon request" (paper).
- **Licence:** host: "CC BY 4.0". Paper licence: not stated in the fetched page.
- **Size and format (host):** 3 files, 28,122,804 B: `mating_data_traces_27june2021.csv` 28,053,603, `mating_data_motif_onsets_27june2021.csv` 16,589, `Susoy_et_al_data_description_2021.rtf` 52,612. The traces CSV concatenates "eight main and 14 partial datasets" (Dataset column). Its first 58 columns are neuron traces "centered and scaled", followed by behavioural columns and an Ethogram (main datasets only). The host description does not mention the ablated-male datasets; whether they are in the CSV was not verified.
- **Model mapping:** calcium, but deposited **centred and scaled**, not ΔF/F, so absolute response amplitudes are lost. The subject is the male tail connectome (Cook 2019 male wiring), not the hermaphrodite, which limits use for a hermaphrodite model. The ablations amount to node removal, but their traces may not be in the deposit. Could constrain male-specific circuit couplings and timescales of motif-locked activity.
- **Known quality issues (paper):** head and tail are not imaged together. Segmentation and tracking were largely manual. Some neurons are missing from some datasets; traces were imputed only when absent from at most two datasets. Discrete-motif timestamp uncertainty is ≤ 0.2 s (median).
- **Table row:** `| Susoy2021 | ~all tail-ganglion neurons; 58 columns in deposit | 76 identifiable across datasets (manual) | freely moving (male, mating) | natural partner; laser ablations; let-23, ceh-14 (paper) | 8 main + 14 partial in deposit (23 ablated in paper) | CC BY 4.0 | 28.1 MB (3 files) | CSV (+RTF) | Mendeley Data, public, no request |`

### Lin2023: Lin, Qin, Casademunt, Wu, Hung, Cain, Tan, Valenzuela, Lesanpezeshki, Venkatachalam, Pehlevan, Zhen & Samuel, 2023, Science Advances (10.1126/sciadv.ade1249), "Functional imaging and quantification of multineuronal olfactory responses in C. elegans" (citation verified)

- **Verification:** The citation was verified against the Europe PMC listing and the Zenodo creators list. The licence, files and sizes come from the host (Zenodo API and landing page). The method details come from the paper (PMC9977185 HTML). The code licence comes from the GitHub API. URLs opened: <https://pmc.ncbi.nlm.nih.gov/articles/PMC9977185/> (offsets 0 and 100000) ; <https://zenodo.org/api/records/7563053> ; <https://zenodo.org/records/7563053> ; <https://api.github.com/repos/samuellab/celegans-sensory-code>
- **Measured (paper):**
  - **Neurons:** the 11 amphid chemosensory pairs (AWA, AWB, AWC, ASE, ASG, ASH, ASI, ASJ, ASK, ADL, ADF), not the whole brain. Identified by a sparse wCherry landmark (gpc-1 promoter) and neighbourhood tracking. Unidentifiable neurons were excluded; the identified fraction per animal is not stated. NeuroPAL was not used.
  - **Preparation:** immobilised in a microfluidic device with odour flowed past the nose.
  - **Stimuli:** 23 volatile odorants at several dilutions (10⁻⁴ to 10⁻⁸; >70 conditions) and 5 ascarosides (#1, #2, #3, #5, #8 at 200 nM), as 10-s pulses separated by 30-s blanks.
  - **Perturbation:** unc-13(s69) synaptic-transmission mutant (5 odorants; responses ~60% of wild type).
  - **Number of worms:** ≥ 10 animals per condition (e.g. 14 for 1-octanol); total not stated.
  - **Rate and length:** 2.5 volumes/s; recording length per animal not stated.
  - **Indicator:** nuclear GCaMP6s (ift-20 promoter, 3xNLS) in ciliated sensory neurons. Strain ZM10104; mutant line ADS707.
  - **Behaviour:** none.
- **Access:** <https://doi.org/10.5281/zenodo.7563053> (Zenodo). Files are listed publicly with checksums (not downloaded). Code: <https://github.com/samuellab/celegans-sensory-code>.
- **Licence:** host: "Creative Commons Attribution 4.0 International" (API id `cc-by-4.0`). Code: MIT License (GitHub API). Paper: CC BY-NC 4.0.
- **Size and format (host):** 2 files, 87,771,303 B: `CElegans Chemosensory Data.zip` 80,221,372 B and `ImageProcessingTracesCode.zip` 7,549,931 B. Inner file formats were not verified. The landing page shows publication date 2022-05-29 and "v2", while the API version relations show a single version (concept DOI 10.5281/zenodo.7490257).
- **Model mapping:** calcium (ΔF/F0 with a fifth-percentile baseline). This is the cleanest input-layer dataset in scope. Each odour pulse drives the identified sensory neurons, and the unc-13 comparison separates direct transduction from synaptic input. Constrains sensory gains and tuning (odour × neuron response matrix, concentration dependence), and sensory-to-sensory synaptic contributions via the unc-13 contrast. It says nothing about interneuron weights.
- **Known quality issues (paper):** mistracked or occluded neurons removed, so trial counts vary. Missing neurons in some recordings were imputed from the ensemble. Left/right pairs are pooled except ASE/AWC in some analyses. Recordings are bleach-corrected.
- **Table row:** `| Lin2023 | 22 sensory neurons (11 pairs) targeted | yes (landmark-based) | immobilised (microfluidic) | 23 odorants + 5 ascarosides; unc-13 | ≥10 per condition (total not stated) | CC BY 4.0 | 87.8 MB (2 zips) | zip (inner format not verified) | Zenodo, public |`

### Ji2021: Ji, Venkatachalam, Rodgers, Hung, Kawano, et al. (Samuel lab), 2021, eLife 10:e68848, "Corollary discharge promotes a sustained motor state in a neural circuit for navigation" (citation verified)

- **Verification:** The citation was verified against the Europe PMC listing. The method details, data statement and licence come from the paper (PMC8139836 HTML, first 200 KB). I opened no data host; the data are eLife figure source files. URLs opened: <https://pmc.ncbi.nlm.nih.gov/articles/PMC8139836/> (offsets 0 and 100000).
- **Measured (paper):**
  - **Neurons:** targeted, not whole-brain: AFD, AIY (soma and neurite), AVA, RIM, RME, SMDD, SMDV.
  - **Preparation:** semi-constrained on agarose under a coverslip.
  - **Stimuli:** oscillating or constant temperature (thermoelectric stage); Chrimson activation of AVA or AVB.
  - **Perturbations:**
    - **miniSOG ablations:** RIM, AIB, AVA, AVB and combinations.
    - **TeTx:** in AIY and in RIM/RIC.
    - **Mutants:** eat-4(ky5), tdc-1, cat-1(e1111), unc-25(e156), unc-31.
  - **Number of worms:** calcium imaging "typically N = 3–9 per condition"; behaviour N in the tens to hundreds.
  - **Rate:** ~1.2 volumes/s.
  - **Indicator:** GCaMP6s, ratioed to wCherry (Pglr-1); Pmod-1 for AIY and Pgcy-8 for AFD.
  - **Behaviour:** thermotaxis tracks (20-min assays at 2 Hz frames).
  - **Recording length:** imaging length not verified.
- **Access:** data statement: "All data generated or analysed during this study are included in the manuscript and supporting files. Source data files have been provided for Figures 1-6. Source code has been provided for Figure 7." No repository. The source-data files are reached through the eLife article (not opened).
- **Licence:** paper: CC BY 4.0. Source data: not separately stated (not checked).
- **Size and format:** not verified (eLife source-data files: per-figure xlsx/zip per the fetched text).
- **Model mapping:** calcium. A causal test of motor → sensory-interneuron feedback (corollary discharge onto AIY from the reversal/forward circuit; tyramine and GABA via tdc-1 and unc-25; neuropeptides via unc-31). Constrains feedback weights and signs from RIM/AVA/AVB onto AIY, the AFD thermal gain, and monoamine coupling. It is not whole-brain, so constraints are local to the thermotaxis circuit, which is a frozen scenario in this project.
- **Known quality issues:** none stated by the authors in the text read.
- **Table row:** `| Ji2021 | targeted (AFD, AIY, AVA, RIM, RME, SMD) | yes (promoters) | semi-constrained (agarose + coverslip) | miniSOG ablations, TeTx, eat-4/tdc-1/cat-1/unc-25/unc-31, Chrimson AVA/AVB, thermal | 3–9 per imaging condition | CC BY 4.0 (paper) | not verified | eLife source data (xlsx/zip) | with article |`

### Casademunt2026: Casademunt, Zhang, Laver, Park, Kim, et al. (Samuel lab), bioRxiv 2026 (10.64898/2026.10.29.762753), "Brain-wide representations of olfactory behavior in C. elegans"

- **Verification:** All facts come from the preprint (bioRxiv v1 HTML, read in three parts) and the viewer page. I found no data or code availability statement in the HTML. URLs opened: <https://www.biorxiv.org/content/10.64898/2026.10.29.762753v1.full> (offsets 0, 100000, 200000) ; <https://helenacasademunt.github.io/worm-viewer/> ; <https://api.github.com/repos/helenacasademunt/worm-viewer> (404).
- **Measured (preprint):**
  - **Freely moving:** 47–70 neurons identified per worm (median 64, all one side), using strain ADS1046 with a simplified NeuroPAL-like three-colour code. 5 volumes/s. Animals: 12 in diacetyl gradients, 16 in isoamyl alcohol gradients, 13 in no-odour controls. Recordings 5–30 min.
  - **Immobilised:** 133–163 neurons tracked (median 146) and 84–114 identified (median 102), using NeuroPAL (OH16230). 2.51 volumes/s. Stimulus: 10-s diacetyl pulses.
  - **Number of immobilised animals:** the summarised text gives both 8 animals and "63 animals across 153 six-pulse runs" (not resolved).
  - **Indicator:** nuclear GCaMP6s (3xNLS) with mNeptune.
  - **Perturbations:** none beyond the lite-1(ce314) background.
  - **Behaviour:** velocity, head angle, heading relative to the odour peak, estimated local concentration, reversals and pirouettes.
- **Access:** **not downloadable.** The viewer <https://helenacasademunt.github.io/worm-viewer/> shows traces, but its "Full data download (coming soon)" control is a placeholder (href `#`).
- **Licence:** not stated (viewer page). Preprint licence not verified.
- **Size and format:** not available.
- **Model mapping:** calcium. Odour gradients act as time-varying drive to AWA (diacetyl) and AWC (isoamyl alcohol) during free behaviour; immobilised NeuroPAL pulses give identified stimulus-response maps. Would constrain sensory gains and the sensory → interneuron → motor transformation, with high identification counts.
- **Known quality issues (preprint):** a residual imaging artefact (first PC) was removed from freely moving data. Motion was handled by a Bayesian red/green model. Outlier timepoints were removed. Laser power was capped below light avoidance. There was no GFP-only control.
- **Table row:** `| Casademunt2026 | 133–163 tracked (immobilised) | NeuroPAL: median 102 (immobilised); median 64 (moving) | both | odour gradients (DA, IAA); DA pulses | 12+16+13 moving; immobilised n unresolved | not stated | n/a | n/a | viewer only; download "coming soon" |`

______________________________________________________________________

### Other items checked (not full entries)

- **BarlowTrack (Fieseler, Lev, Madhusudhanan, Zhai, Schwartz, Zimmer; bioRxiv 2025, 10.1101/2025.11.10.687595).** A tracking method that says it re-releases benchmark freely moving whole-brain datasets from the Zimmer, Leifer and Flavell labs in NWB. Its statement says "Data will be made available upon publication". Code: <https://github.com/Zimmer-lab/barlowtrack> per the preprint; the GitHub API lists the repository as `Zimmer-lab/barlow_track`, MIT. Its README "Citation: Coming soon!" and has no data links. Not public.
- **Avila et al. 2025 PNAS (Makse lab with Zimmer; 10.1073/pnas.2417850122).** Listed by Europe PMC. Not opened; I did not check whether it contains new recordings or re-uses OSF data.
- **Wan et al. 2026 Nat Commun (10.1038/s41467-026-69392-2; Samuel co-supervising).** Male pheromone navigation. Pan-neuronal GCaMP, but the analysis covers a few identified neurons (PHD, AWA, ASI, AVA, AVE, presumed PVN and EF), not whole-brain. Data are on CaltechDATA (10.22002/dp6gb-ky687) and <https://github.com/edmondztt/pheromone-traj-analysis> per the paper. Article CC BY 4.0. Neither the CaltechDATA record nor its licence was opened.
- **Park et al. 2024 Nat Methods (targettrack; 10.1038/s41592-023-02096-3).** A tracking method from the lpbsscientist group with Samuel-lab co-authors. "Sample 4D datasets" at <https://github.com/lpbsscientist/targettrack> (not opened). Not a neural-activity dataset release as far as the statement says.
- **Susoy & Samuel 2023 Curr Biol (10.1016/j.cub.2023.09.003).** Not opened; whether it contains imaging was not checked.
- **Related but outside this section's scope (DANDI API listing, host):** dandiset 000541 "NeuroPAL Microfluidic Chip Images and GCaMP activity" (contact Yemini, Ev; 21 assets, 30,525,162,874 B). This is likely the Yemini et al. 2021 NeuroPAL paper (Samuel co-author); I did not verify that link. Dandisets 000472 and 000565 belong to Saul Kato's later UCSF lab, not the Zimmer lab. None of these were opened beyond the listing.
- **Web search budget.** The session's WebSearch allowance ran out near the end. I could not search further for a Venkatachalam 2016 data deposit or for published versions of the Zimmer 2025–2026 preprints.

## D. Other neural-activity datasets (2024-2026 sweep) and electrophysiology

Sweep date 2026-11-14. Method: WebSearch (standard and extended) and WebFetch of landing pages, public APIs
(DANDI, Zenodo, OSF, Mendeley Data, Hugging Face, Crossref, Europe PMC, eLife) and paper full texts. I downloaded
no data files. Source tags used below: **[host]** = the data host's own page or API; **[paper]** = the article
or preprint itself (methods or data statement); **[registry]** = Crossref/Europe PMC metadata; **[2nd]** =
second-hand (search-result snippet, news item, a citing paper). Anything not opened in this session is marked
"not verified".

A DANDI API search for "elegans" (`https://api.dandiarchive.org/api/dandisets/?search=elegans&page_size=100`)
returned 12 dandisets, all imaging. None is electrophysiology [host]. Dandisets from labs other agents cover are
listed under "seen, covered elsewhere".

### Part 1: neural-activity datasets (full entries)

### Dunn et al. (Kato lab): Dunn, Costello, Borchardt, Sprague, Chiu, Miller, L'Etoile, Kato, 2025, *Current Biology* 35(22):5582-5593.e4

- **Verification:** Publication from Crossref [registry]. Methods from bioRxiv 2024.08.11.607402 v2 ("Working memory by
  distributed neural oscillators in a simple nervous system") [paper]. Data from DANDI and Zenodo [host]. The journal
  version's title is "Short-term memory by distributed neural network oscillators in a simple nervous system". I did
  not open the journal full text, so methods facts are from the preprint. URLs opened:
  <https://www.biorxiv.org/content/10.1101/2024.08.11.607402v2.full> ;
  <https://api.dandiarchive.org/api/dandisets/001623/versions/draft/info/> ;
  <https://api.dandiarchive.org/api/dandisets/001623/versions/draft/assets/?page_size=100> ;
  <https://github.com/focolab/2025-dunn-et-al-curr-biol> ; <https://zenodo.org/api/records/17353307> ;
  <https://api.crossref.org/works/10.1016/j.cub.2025.10.018>
- **Measured \[paper\]:**
  - Neurons: whole-brain volumetric imaging, 45-140 neurons detected per recording. Identification was manual against
    NeuroPAL, by position, neighbours and activity; ambiguous neurons were left unlabelled.
  - Preparation: imaging worms paralysed with tetramisole in PDMS microfluidic chips. Behaviour-only experiments ran
    freely moving on unlidded agar, with closed-loop "virtual olfactory" AWA stimulation timed to head-swing posture.
  - Perturbations: optogenetic AWA stimulation (WOrMsChRmine under an AWA promoter, DMD at 640 nm, 4-8 s pulses timed
    to head-swing phase), single-neuron stimulation of SMDD and SMDV, RIV, and RIA sub-compartments nrV and nrD. For
    the head-swing neurons the actuator is not clearly stated (a Chrimson plasmid is acknowledged). No mutants.
  - Indicator: pan-neuronal nuclear GCaMP6s.
  - Timing: imaging recordings 16-24 min, each with 10-30 forward-reverse-turn sequences. Volume rate not stated (10-12
    z-planes at 2.5-3 um, or 4 planes for a faster half-brain mode; 20-30 ms exposures in an 80 ms duty cycle).
  - Behaviour: forward, reversal and turns, head-swing dynamics, latency from AWA stimulation to the next reversal,
    and turn direction.
  - Worm counts: the paper gives no total. Fig. 2H's regression has n = 132; whether that counts worms or trials is
    unclear.
- **Access \[host\]:**
  - DANDI 001623, "C. elegans whole-brain imaging datasets (NeuroPAL and timeseries)" (<https://dandiarchive.org/dandiset/001623>),
    open, created 2025-10-09. It holds 95 NWB files, one subject and session each, with session dates 2022-03-02 to
    2024-09-17.
  - Zenodo 10.5281/zenodo.17353307 ("Data for 2025-Dunn-et-al-curr-biol", 2025-10-14) holds intermediate pickles.
  - GitHub focolab/2025-dunn-et-al-curr-biol has the analysis notebooks. Its README says the raw DANDI data are "not
    used in these analyses" and that the analyses run from the Zenodo pickles.
- **Licence \[host\]:** DANDI "spdx:CC-BY-4.0"; Zenodo "cc-by-4.0"; GitHub repository: not stated.
- **Size and format \[host\]:**
  - DANDI: 774,224,141,637 bytes in 95 NWB files, about 2.1-13.0 GB each (46 files near 13 GB).
  - Zenodo: `intermediate_datafiles.zip` (665,788,261 bytes) plus 95 `.pkl` files (18.9-209.9 MB).
  - Worm count vs files: the paper states no worm total; the host lists 95 subjects. Whether every file is a distinct
    worm, and which files carry optogenetic stimulation, is not verified.
- **Model mapping:**
  - Signal: calcium, nuclear and slow, immobilised.
  - Perturbations in model terms: AWA stimulation is a step input to one sensory class at a controlled phase of the
    ongoing head-swing oscillation. SMD, RIV and RIA (compartment) stimulation are direct current injections into motor
    and interneurons.
  - Constrains: effective AWA-to-network weights and their phase dependence; coupling among SMD/RIA/RIV and the
    head-swing oscillator; time constants and phase dynamics of the oscillatory short-term memory; and how sensory
    inputs gate state transitions (the reversal-turn sequence).
  - This is one of the few 2024-2026 whole-brain datasets with targeted single-neuron perturbations from a lab outside
    the Leifer/Flavell/Zimmer/Samuel set.
- **Known quality issues \[paper\]:**
  - All-trans retinal concentration differs between sections: 100 uM in the behaviour methods, 1 uM in the imaging
    methods.
  - Some datasets were cropped where neurons drifted out of focus, and undetected cells were extrapolated from
    neighbouring ROIs.
  - Forward states under 4 s were dropped from the PCA.
  - Photobleaching and opsin/imaging crosstalk were minimised but not quantified.
  - The GitHub README warns that file paths in the code are broken and need fixing [host].
- **Cross-check from section B's pass \[host unless marked\]:**
  - WormID lists the 001623 recordings as 43–139 segmented and **0–82 identified**, noting "many unlabeled recordings".
  - Three sessions also appear in DANDI 000565: sub-20221106-21-00-09, -21-23-19 and -21-47-31. WormID notes a "minor overlap with SK1".
  - DANDI version 0.251015.0312.
  - Zenodo lists 96 files.
  - Freely crawling behaviour runs were 1–2 × 12 min [paper, preprint].
  - The journal licence is "CC BY-NC-ND" [2nd, Europe PMC index].
- **Table row:** `| Dunn 2025 (Kato) | 45-140 | NeuroPAL manual | immobilised imaging; free behaviour | opto AWA, SMDD/V, RIV, RIA nrV/nrD | not stated (95 NWB) | CC-BY-4.0 | 774.2 GB + 0.67 GB | NWB; pkl | yes |`

### Whole-brain chemosensory responses of both sexes: Seyedolmohadesin, Fu, Torkashvand, Rasouli, Lang, Li, Kalinski, Cook, Schroeder, Yemini, Venkatachalam, 2026, *Current Biology* 36(18):4803-4816.e4

- **Verification:** Publication from Crossref [registry]. Methods from bioRxiv 2025.05.15.654129 v1 [paper]; I read
  only the first 100k characters and did not open the journal version. Data from DANDI [host] and the lab's viewer
  site [host]. Publication month from a Northeastern news item [2nd]. URLs opened:
  <https://www.biorxiv.org/content/10.1101/2025.05.15.654129v1.full> ;
  <https://api.dandiarchive.org/api/dandisets/000981/versions/draft/info/> ;
  <https://api.dandiarchive.org/api/dandisets/000981/versions/> ;
  <https://api.dandiarchive.org/api/dandisets/000981/versions/0.260910.1747/info/> ;
  <https://api.dandiarchive.org/api/dandisets/000981/versions/draft/assets/?page_size=5> ;
  <https://chemosensory-data.worm.world> ; <https://api.crossref.org/works/10.1016/j.cub.2026.08.040> ;
  <https://news.northeastern.edu/?p=316226>
- **Measured \[paper\]:**
  - Animals: 22 hermaphrodites and 25 males, strain OH16230 (otIs670 [NeuroPAL]; otIs672 [panneuronal::GCaMP6s]).
    No mutants.
  - Neurons: 196 sex-shared plus 6 male-specific head neurons were recorded. The field of view holds about 230 of 302
    neurons (hermaphrodite) and 304 of 387 (male). Identification was manual from the NeuroPAL colour code; the
    per-animal identified count is not reported. 84 of 89 head neuron classes responded.
  - Preparation: immobilised with 1 mM levamisole in a U-shaped microfluidic trap.
  - Stimuli: OP50 supernatant; 1 uM ascr#3; 800 mM sorbitol; 10 mM CuSO4; isoamyl alcohol at 10^-6 and 10^-2; NaCl
    at 100 and 450 mM. Controls were a buffer-to-buffer switch and 10 uM fluorescein.
  - Protocol: each cue was 15 s on, 45 s buffer; the stimulus set ran 3 times in random permutation; 31-min total
    recording.
  - Imaging: 2.67 volumes/s, 25 frames per volume, spinning-disk confocal.
  - Indicator: GCaMP6s. Nuclear localisation is not stated in the text I read (not verified).
  - Behaviour: none (immobilised).
- **Access \[host\]:**
  - DANDI 000981 (<https://dandiarchive.org/dandiset/000981>): published version 0.260910.1747 (2026-09-10).
    Description: "Datasets for the Current Biology paper (DOI: 10.1016/j.cub.2026.08.040)".
  - The interactive viewer at <https://chemosensory-data.worm.world> shows only "Copyright © Maedeh Seyedolmohadesin
    2024"; its /downloads path returned 404.
  - The preprint names imaging code at <https://github.com/venkatachalamlab/lambda> (not opened).
- **Licence \[host\]:** "CC-BY-4.0 (SPDX: spdx:CC-BY-4.0)" on DANDI. The viewer site shows a copyright notice only.
- **Size and format \[host\]:** 82,267,067,651 bytes; 47 NWB files, one per subject. The 47 subjects match the paper's
  22 + 25 animals. Example files are 1.44-2.83 GB. variableMeasured is ProcessingModule and PlaneSegmentation, so
  traces and segmentation are present; whether raw volumes are included is not verified.
- **Model mapping:**
  - Signal: calcium, immobilised.
  - Perturbations: none acute. Stimuli are step inputs to identified sensory classes, repeated 3 times per animal.
  - Constrains: per-class sensory gains and sign (ON/OFF) for 8 cues; propagation from sensory neurons to identified
    downstream classes, as effective connectivity from stimulus-locked responses.
  - Sex is a natural "rewiring" test: the male connectome differs, so a connectome-constrained model fitted on one sex
    can be tested on the other.
  - Repeated trials per animal allow trial-to-trial variability estimates.
- **Known quality issues:**
  - Levamisole paralysis and an immobilised preparation [paper].
  - The fluorescein control exists to check seal and flow, so a leaky seal is a possible failure mode [paper].
  - No per-animal identification counts are given in what I read.
  - The viewer site's download page 404s, so use DANDI [host].
  - This is probably the "Venkatachalam2024" set (22 worms) in the Simeon pooled data; that is an inference from
    matching counts, not verified.
- **Cross-check from section B's pass:**
  - The preparation is only partially immobilised: small movements and pumping remain [paper, preprint].
  - DANDI file-name prefixes give 22 "h" and 25 "m"; files are 1.2–2.8 GB [host].
  - Further quality notes \[paper, preprint\]:
    - Tracking outliers (ZephIR) were removed by an IQR threshold.
    - Bleaching was handled by band-pass filtering, which also removes slow components.
    - Recordings with a fluorescein leak were discarded.
    - AWC-ON and AWC-OFF were merged.
    - No identification-accuracy figure is reported.
  - The preprint licence is "CC BY-NC-ND" [2nd].
  - Nuclear localisation of the GCaMP6s: section B notes that the Yemini 2021 methods describe the pan-neuronal GCaMP6s as nuclear-localised, but that is not verified for strain OH16230.
  - Conflict: section B saw a Downloads page on the viewer site and did not open it; this pass found `/downloads` returns 404. Use DANDI.
- **Table row:** `| Seyedolmohadesin 2026 | 196+6 head neurons recorded | NeuroPAL manual | immobilised | 8 chemical cues ×3, both sexes | 22 herm + 25 male (47 NWB) | CC-BY-4.0 | 82.3 GB | NWB | yes |`

### Homogenized C. elegans neural activity (pooled): Simeon, Kashyap, Kording, Boyden, arXiv 2411.12091 v4 (2025-05-19); companion to Simeon et al. 2024, "Scaling properties for artificial neural network models of a small nervous system" (bioRxiv 2024.02.13.580186)

- **Verification:** Composition from arXiv HTML v4, Table 1 [paper]. Files, licence and columns from the Hugging Face
  API and dataset viewer [host]. Code repositories from GitHub [host]. I did not open the bioRxiv scaling paper (seen
  in search results only [2nd]). URLs opened: <https://arxiv.org/abs/2411.12091> ; <https://arxiv.org/html/2411.12091v4> ;
  <https://huggingface.co/datasets/qsimeon/celegans_neural_data> ;
  <https://huggingface.co/api/datasets/qsimeon/celegans_neural_data> ;
  <https://huggingface.co/api/datasets/qsimeon/celegans_neural_data/tree/main?recursive=true> ;
  <https://huggingface.co/datasets/qsimeon/celegans_neural_data/raw/main/datasetcard.md> ;
  <https://github.com/qsimeon/worm-graph> ; <https://github.com/qsimeon/worm-data-preprocess>

- **Measured \[paper, Table 1\]:** 12 source datasets with worms and mean labelled/recorded neurons:

  - Kato2015: 12 worms, 42 labelled / 127 recorded.
  - Nichols2017: 44 worms, 34/108.
  - Skora2018: 12 worms, 46/129.
  - Kaplan2020: 19 worms, 36/114.
  - Nejatbakhsh2020: 21 worms, 173/175.
  - Yemini2021: 49 worms, 110/110.
  - Uzel2022: 6 worms, 50/138.
  - Dag2023: 7 worms, 100/143.
  - Leifer2023: 110 worms, 63/67. The worm-data-preprocess README lists 103 [host], a discrepancy.
  - Lin2023: 577 worms, 8/8 (1-22 labelled).
  - Flavell2023: 40 worms, 88/136.
  - Venkatachalam2024: 22 worms, 187/187.

  That sums to about 919 worms (the abstract says "about 900"). All traces are resampled by linear interpolation to
  dt of about 0.333 s, about 3 Hz [paper]. Each neuron is z-scored over its whole series, which the paper notes is
  non-causal. 300 slots; CANs excluded.

- **Access \[host\]:** <https://huggingface.co/datasets/qsimeon/celegans_neural_data> (not gated). Connectome companion:
  <https://huggingface.co/datasets/qsimeon/celegans_connectome_data> (not opened). The worm-graph code repo is archived
  (read-only since 2025-03-31).

- **Licence:**

  - Hugging Face card: "mit" [host]. The arXiv paper page carries CC BY-NC-SA 4.0, which is the paper's licence
    [paper page].
  - worm-graph code: "MIT license" [host]. worm-data-preprocess: not stated [host].
  - The MIT label is applied to re-processed data from sources with their own terms. Whether the sources' licences
    permit that is not verified.

- **Size and format \[host\]:** one file, `worm_data_short.parquet`, 669,511,381 bytes; viewer shows 42.8k rows. The
  visible columns are source_dataset, raw_data_file, worm, neuron, slot, is_labeled_neuron, smooth_method,
  interpolate_method, normalization_method, original_calcium_data, calcium_data, original_time_in_seconds,
  time_in_seconds, original_max_timesteps and max_timesteps. Created 2024-02-13, last modified 2025-05-08.

- **Model mapping:**

  - Signal: calcium, mixed immobilised and freely moving.
  - Useful as a cross-lab spontaneous-dynamics corpus with a common neuron ordering, for pretraining or for testing
    whether a connectome model's dynamics generalise across labs.
  - Not usable for perturbation fitting: the visible columns carry no stimulus, optogenetic-event or behaviour
    timing, so the Randi/Leifer stimulation events are not encoded in what I saw. z-scoring also removes absolute
    scale.
  - Every source is covered elsewhere except Nejatbakhsh2020 and Venkatachalam2024.

- **Known quality issues \[paper\]:** calcium is a low-pass, lagged measure; resampling to 3 Hz loses fast content for
  high-rate sources and inflates apparent resolution for slow ones. Unlabelled and unmeasured neurons are not
  distinguished (one binary mask). Labelled counts vary widely (Lin2023 can have 1). Connectome and activity come
  from different animals.

- **Table row:** `| Simeon pooled (HF) | 1-189 labelled | from sources | mixed | none encoded | ~919 (paper), 12 sources | "mit" (HF) | 669.5 MB | parquet | yes |`

### Global head activity under random bacterial stimulation: Rusu, Cecere, How, Quach, Yemini, Sharpee, Chalasani, 2026, *eLife* 15:e108234 (VOR, 2026-10-13)

- **Verification:** Article content from the eLife API [paper]. Data listing and README from the Salk server [host].
  The analysis GitHub (<https://github.com/shreklab/Rusu_et_al_2026>) was not opened. URLs opened:
  <https://api.elifesciences.org/articles/108234> (offsets 0, 100k, 200k) ;
  <http://www.snl.salk.edu/~schalasani/Rusuetal2026/> ; <http://www.snl.salk.edu/~schalasani/Rusuetal2026/README.md>
- **Measured \[paper\]:**
  - Strains: ZIM294, pan-neuronal nuclear GCaMP5K (mzmEx199). For identification, OH15500 (NeuroPAL otIs669 +
    pan-neuronal GCaMP6s otIs672), with cells identified using Yemini et al. 2021.
  - Neurons: 50-100 extracted per animal of 189 head neurons (NMF after piecewise rigid registration). Six functional
    clusters: ON and OFF sensory, AVA, RME, SMDD and SMDV.
  - Preparation: 1.5 mM tetramisole, nose-only microfluidic device. Zeiss LSM 880 Airyscan "Fast", about 1.5 volumes/s,
    2 um z-steps.
  - Stimulus: bacteria (OD600 0.4, 1:1 in M9) in random pulse blocks of about 15 s (faster protocol about 1.5 s), with
    p(on|off) = 0.2 and p(off|on) = 0.4. Buffer-switch data come from How et al. 2021.
  - Recordings: about 10 min typically (21 min for the stimulus set). 7 stimulated animals; the total per dataset is
    not given.
  - Perturbations: none. Behaviour: none (paralysed).
- **Access \[host\]:** <http://www.snl.salk.edu/~schalasani/Rusuetal2026/> is a plain directory listing, open HTTP, no
  request needed. The README describes raw TIFF archives: "Experimental_raw" (7 zip archives of zim294 worms on OP50
  under SFST, SFSTnull and SFSTrevS conditions) and "Buffer_Data" (about 49 tar.bz2 archives for 2-nonanone,
  benzaldehyde, diacetyl, IAA at several concentrations, and NaCl at 200 and 600 mM).
- **Licence:** data host: not stated. Article: CC-BY-4.0 [paper].
- **Size and format \[host\]:** `compressed.tar` 51 GB, plus CHECKSUM.txt (151 B) and README.md (3.4 KB). The README
  describes raw image TIFFs only; extracted traces are not mentioned (not verified).
- **Model mapping:**
  - Signal: calcium, immobilised.
  - The stimulus is a random telegraph (Markov) on/off sequence, which suits linear-nonlinear system identification of
    sensory filters and of delays down to command and head-motor neurons.
  - Constrains: sensory gain and sign of the ON/OFF classes; integration time constants (sensory peaks within about
    15 s, command/motor clusters tens of seconds later).
  - Only raw images appear to be released, so traces would have to be re-extracted.
- **Known quality issues \[paper, peer review\]:**
  - Interneuron and motor dynamics are much slower than in moving animals.
  - Cluster silhouette scores were low, with no clear optimal cluster number.
  - Not all RME-cluster neurons were identified.
  - Buffer responses differed between channel 1 and channel 2; the authors attribute this to a possible pipeline error.
  - Unparalysed worms could not be motion-corrected.
- **Table row:** `| Rusu 2026 eLife | 50-100 | NeuroPAL for 4 classes | immobilised | random bacterial pulses (+ odour/salt buffer set) | 7 stimulated | not stated | 51 GB | raw TIFF (tar) | yes |`

### Associative-memory coding in sensory neurons and interneurons: Pritz, Itskovits, Bokman, Ruach, Gritsenko, Nelken, Menasherof, Azulay, Zaslaver, 2023, *eLife* 12:e74434

Included although it predates 2024: it is the clearest open multi-neuron dataset across learning conditions that I
found.

- **Verification:** Article from the eLife API [paper]. Data from the OSF API [host]. URLs opened:
  <https://api.elifesciences.org/articles/74434> (offsets 0 and 100k) ; <https://api.osf.io/v2/nodes/5v4qu/> ;
  <https://api.osf.io/v2/nodes/5v4qu/files/osfstorage/>
- **Measured \[paper\]:**
  - Strains: pan-chemosensory strain ZAS280, cytoplasmic GCaMP3 (osm-6 promoter) with NLS-mCherry for segmentation.
    Separate strains for AIY (ZAS256), AIA (CX16561), RIA (PS6510) and AVA/AVE (HBR191, ZAS323). URX used cameleon
    YC2.60.
  - Neurons: in the butanone/diacetyl exchange analysis, 16 classes (AWA, AWC-ON, AWC-OFF, AWB, ASER, ASEL, ASJ, ASK,
    URX, ASH, ASI, AIY, AIA, RIA subclasses, AVA, AVE). Identification was manual against anatomical maps and reporter
    strains. AFD was excluded (too dim); ASE, ADL, ASG and ADF were excluded from the butanone-only analysis.
  - Preparation: levamisole-paralysed for multi-neuron imaging; interneurite imaging unparalysed. Nikon A1R+ confocal,
    0.9-2 volumes/s, interpolated to 2-5 Hz.
  - Stimuli: butanone at 33.5 or 3.35 mM; diacetyl at 11.6 uM; 90-s on/off runs or six 30-s exchanges.
  - Perturbation: training history, in 4 protocols: short-term appetitive and aversive, long-term appetitive and
    aversive, plus mock and naive groups.
  - Animals: about 8-18 per condition; no grand total.
- **Access \[host\]:** <https://osf.io/5v4qu/> (public). Top-level folders: "activity data butanone", "acitivity data
  butanone diacetyl exchange", "chemotaxis behavior", "choice behavior", "image mining". Code:
  <https://github.com/ChristianPritz/Principles-for-coding-associative-memories-in-a-compact-neural-network> (not
  opened).
- **Licence \[host\]:** none set on OSF (`node_license: null`). Article licence: not recorded in this session.
- **Size and format \[host\]:** folder sizes not returned by the OSF API; file formats not verified.
- **Model mapping:**
  - Signal: calcium, cytoplasmic.
  - The perturbation is a learned state rather than an acute input. It constrains which parameters must be plastic:
    sensory gains for short-term memory, interneuron (AIY, AIA, RIA) responses for both short- and long-term memory.
  - Stimulus-locked responses for 16 identified classes give sensory gain/sign and sensory-to-interneuron effective
    weights.
- **Known quality issues \[paper\]:**
  - An about 30-min starvation was needed for imaging, applied equally to trained and mock groups.
  - AVA/AVE signals may reflect fictive behaviour.
  - Baselines for interneurons were set visually.
  - Ground states were defined differently for short and long recordings.
- **Table row:** `| Pritz 2023 eLife | up to 16 classes | manual (anatomy + reporters) | immobilised | 4 conditioning protocols | ~8-18/condition | none set (OSF) | not shown | OSF folders | yes |`

### Whole-brain spontaneous activity of semi-restricted worms: Suzuki, Wen, Sprague, Onami, Kimura (Kimura lab, Nagoya City University), DANDI 000692 (version 0.240402.2118)

- **Verification:** All facts from DANDI [host]; no related publication is listed. URLs opened:
  <https://api.dandiarchive.org/api/dandisets/000692/versions/draft/info/> ;
  <https://api.dandiarchive.org/api/dandisets/000692/versions/> ;
  <https://api.dandiarchive.org/api/dandisets/000692/versions/draft/assets/?page_size=50> . Also opened the lab's
  earlier 3DeeCellTracker image collection at <https://ssbd.riken.jp/database/131>.
- **Measured \[host\]:**
  - "Spontaneous neuronal activity extracted from GCaMP and tagRFP images, covering nearly all neurons in the C.
    elegans brain".
  - Segmentation and tracking by 3DeeCellTracker; identities by NeuroPAL Auto-ID followed by manual correction.
  - Worms semi-restricted in a microfluidic device. No stimulus or perturbation described.
  - Recording length, rate and indicator variant: not stated on the host.
- **Access \[host\]:** <https://dandiarchive.org/dandiset/000692>, open. A sibling dandiset, 000680 (same title family,
  Kimura), is empty (0 bytes).
- **Licence \[host\]:** "spdx:CC-BY-4.0".
- **Size and format \[host\]:** 9 NWB files, 8.58-9.37 GB each, 80,537,036,658 bytes total (versions list and asset
  list). The draft info endpoint's assetsSummary is stale: it says 61,926,154,300 bytes, 7 files.
- **Model mapping:** calcium; spontaneous dynamics only. Useful as extra held-out spontaneous data with identities in
  a semi-restricted (not fully paralysed) preparation. No inputs to constrain weights causally.
- **Known quality issues \[host\]:** "Tracking occasionally fails transiently"; the authors recommend a moving median
  filter (for example ±3 frames).
- **Cross-check from section B's pass \[host, WormID\]:** WormID lists 149–163 segmented neurons per worm, all identified. The images are GCaMP with tagRFP.
- **Table row:** `| Kimura lab DANDI 000692 | "nearly all" head neurons | NeuroPAL Auto-ID + manual | semi-restricted | none | 9 NWB | CC-BY-4.0 | 80.5 GB | NWB | yes |`

### Part 1: compact list of the rest

- **Lee, Vallier, Lu 2025 (Hang Lu lab), bioRxiv 2025.11.03.686171**: head-fixed microfluidics with behaviour.
  - Sparse strain: 12 identified neurons (AVA, AVB, AVD, AVE, SMDD, SMDV, RIA, RIM, RMD, RMDD, RMDV, RME), cytosolic
    GCaMP6s.
  - Whole-brain strain GT408: about 110 neurons, not identified. Light-sheet at 20 vol/s; 90-s trials with bacterial
    supernatant or 0.5 M glycerol.
  - 3 worms × 5 trials (comparison) and 7 worms × 2 trials (glycerol) [paper].
  - The data-availability statement was not found in either part of the page I read: not verified.
- **Chaudhary, Lee, Li, Patel, Lu 2021 eLife (CRF_ID)**: GitHub shiveshc/CRF_Cell_ID [host] holds "31 datasets of
  animals used in multi-cell calcium imaging analysis" as raw GFP z-stacks, plus NeuroPAL and gene-expression image
  sets. No traces mentioned; licence not stated. DANDI 000714 (Chaudhary) is structural NeuroPAL images only (478 MB)
  [host].
- **Quan Wen lab (USTC)**: CeNDeR (Wu et al. 2022 PLoS Comput Biol): freely moving whole-brain at about 5 Hz,
  nuclear GCaMP6 + mNeptune, 3 annotated animals. The data statement points only to code
  (<https://github.com/Wenlab/CeNDeR>); no traces stated as released [paper]. No 2024-2026 Wen-lab neural dataset
  found.
- **Hallacy, Yonar, Ringstad, Ramanathan 2024, eLife RP 97340**: optogenetic silencing screen over 29 overlapping
  lines (compressed sensing) for learned PA14 avoidance, with calcium imaging of AIY, AVK and SIA. Seen via the eLife
  API summary only; data availability not checked.
- **Saad et al. (Bamber lab) 2025, eLife RP 104828**: 1-octanol activates attractant and repellent olfactory neurons.
  Summary only; data availability not checked.
- **"Enhanced whole-brain calcium imaging and cell identification in C. elegans reveal AWCOFF neuronal responses to
  2-nonanone"**, J. Biosci. 2025 (PubMed 40619772): page redirect blocked; authors, data and licence not verified.
- **Toyoshima et al. 2020 BMC Biol (Iino lab), neuron ID dataset**: head-neuron positions and expression in 311
  worms; materials on Figshare 10.6084/m9.figshare.8341088 [2nd, not opened]. Positions, not activity.
- **SSBD 131 (Wen C., Kimura lab, 3DeeCellTracker)**: 6 image sets and 1 quantitative set (9.2 GB image + 12.9 MB
  quantitative), "CC BY", strains AML14 and KDK54165, time steps 1.015-2.04 s [host]. Tracking benchmarks, not
  identified traces.
- **Sordillo & Bargmann 2021 eLife**: AVA-only imaging (18-31 traces per genotype) under RIM perturbations (HisCl,
  tdc-1, eat-4 excision in RIM, unc-1(n494), ReaChR); Dryad 10.5061/dryad.ht76hdrf6 (not opened) [paper]. Single
  neuron class, so low value for whole-network fitting.
- **Model reuse, no new worm data:** POCO (Duan et al., NeurIPS 2025, arXiv 2506.14957; code github.com/yuvenduan/POCO;
  worm datasets not named on the abstract page) and CAPT (Xu, Zhang, Zhang, arXiv 2607.23258, 2026; worm datasets not
  named on the abstract page) [paper pages].
- **Searched with nothing found (2024-2026, open, multi-neuron):** whole-brain imaging during habituation, sleep
  (beyond the Zimmer-lab sets), egg-laying (Collins lab: single-circuit freely-behaving imaging only, no deposit
  found), larval or dauer whole-brain development, Iino-lab salt-learning whole-brain, Mei Zhen lab (Meng et al. 2024
  Sci Adv on AVA, data statement not reached), Gordus, Haspel, Kennedy, Bhattacharya. "Tom Kuhn" did not resolve to a
  C. elegans lab. A Venkatachalam NIH project (RePORTER 11304599) plans whole-brain chemosensory responses across
  developmental stages [2nd]; no dataset yet.

**Seen, covered elsewhere (not entered here):**

- Fieseler, Lev, Rey, Hille, Brenner, Zimmer 2025 bioRxiv 2025.03.09.642241 (freely moving + immobilised, nuclear
  GCaMP7b, O2 steps, gcy-31;gcy-35;gcy-9).
- Baskoylu & Flavell 2026, Dryad 10.5061/dryad.w9ghx3g4v (27 freely moving NeuroPAL worms, copper boundary; 320.03 MB).
- Kramer ... Flavell 2026, Zenodo code 10.5281/zenodo.12611760 (MIT; data at Dryad 10.5061/dryad.8sf7m0cz2, not
  opened; tyramine/RIM perturbations during olfactory turning).
- Atanas et al. 2025 eLife RP 108159 (BrainAlignNet/AutoCellLabeler).
- DANDI 000776 (Flavell), 001075 (Leifer, Randi atlas), 000541/000715 (Yemini), 000565/000472 (Kato, WormID corpus),
  000449 (empty).
- WormID / Sprague et al. 2025 Cell Rep Methods and WormID-Bench.
- bioRxiv 2024.09.22.614271 "Bridging the gap between the connectome and whole-brain activity" (not opened; uses
  optogenetic-perturbation recordings, likely the Leifer atlas).

______________________________________________________________________

### Part 2: electrophysiology

Overall \[host + paper\]: no C. elegans electrophysiology is on DANDI. Open raw traces of identified neurons exist
mainly as per-paper Mendeley Data deposits from the Qiang Liu lab (Bargmann lab, then City University of Hong Kong),
plus journal "Source Data" files. Most classic recordings are in figures only: Goodman, Lockery, Zhao-Wen Wang and
Mei Zhen labs. Nicoletti et al. say they received raw AVA recordings from Zhao-Wen Wang and Ping Liu, which shows
that sharing on request happens [paper].

### AWA action potentials: Liu, Kidd, Dobosiewicz, Bargmann, 2018, *Cell* 175(1):57-70 (citation from Nicoletti 2024 reference list [2nd]; Cell page not opened)

- **Verification:** Methods from bioRxiv 359935 v1 [paper]. Data from the Mendeley Data public API and landing page
  [host]. URLs opened: <https://data.mendeley.com/datasets/tngf9w3pgd> ;
  <https://data.mendeley.com/public-api/datasets/tngf9w3pgd> ; <https://www.biorxiv.org/content/10.1101/359935v1.full>
- **Measured \[paper\]:**
  - Preparation: whole-cell patch clamp in dissected adults.
  - Cells: RIM (n=3), AIY (n=7), AFD (n=3) and AWA (n=16) for I-V surveys. Most analysis is of AWA (149 spikes
    averaged; 19 isolated single spikes).
  - Current clamp: steps -15 to +35 pA, often with a -10 pA pre-pulse; 0.1 Hz sine (±10 pA); ramps of 0.4-3 pA/s and
    25 pA/s; a 4-step protocol.
  - Voltage clamp: -100 to +50 mV in 10 mV steps from -60 mV holding.
  - Mutants: egl-19 (lf and gf), unc-2, cca-1, nca-1;nca-2, shk-1, shl-1, slo-1, slo-2 (11 K+ channel mutants in
    total), unc-13, osm-9.
  - Drugs and ions: nemadipine-A, 4-AP, TEA; Ca2+ at 0-8 mM, Ba2+ replacement, NMDG for Na+.
  - Acquisition: filtered at 2 kHz, sampled at 10 or 50 kHz. Patch plus GCaMP6f imaging of AWA at 50 fps.
  - Odour-evoked AWA calcium (diacetyl 1.15 uM, two 5-s pulses) in 40 N2 and 56 shk-1 videos at 100 fps. Odour was
    not delivered during patching.
- **Access \[host\]:** <https://data.mendeley.com/datasets/tngf9w3pgd> (DOI 10.17632/tngf9w3pgd.1, 2018-09-14).
  Description: "Raw electrophysiological recording traces and calcium imaging movies shown in this paper and
  accompanying analyzed data." Open download.
- **Licence \[host\]:** "CC BY 4.0".
- **Size and format \[host\]:** 15 zip files, 809,369,871 bytes in total: "AWA Calcium Imaging Raw Data.zip" (219.5 MB)
  and one zip per main and supplementary figure (Figure S6 is 390.4 MB). File formats inside the zips not verified.
- **Model mapping:** voltage.
  - AWA: regenerative CaV1 (EGL-19) spikes repolarised by SHK-1, so a single-compartment model needs these
    conductances for AWA. A graded tanh unit cannot reproduce all-or-none spiking.
  - RIM, AIY, AFD I-V curves constrain steady-state conductance and input resistance for those classes.
  - The mutant series pins which conductance does what.
- **Known quality issues \[paper\]:** dissected preparation; acceptance required seal >1 GΩ and series resistance \<100
  MΩ. Odour delivery during patching failed, so sensory input and voltage are never recorded together.
- **Table row:** `| Liu 2018 AWA | 1 (patch) | promoter-GFP | dissected | channel mutants, drugs, ions | AWA 16; RIM 3; AIY 7; AFD 3 | CC BY 4.0 | 809.4 MB | zip | yes |`

### ElectroPhysiomeGAN recordings: Kim, Peng, Chen, Liu, Shlizerman; eLife reviewed preprint 95607 (v3, 2025-08-21); data 2025

- **Verification:** Data from the Mendeley API [host]. Methods from bioRxiv 2023.12.19.572452 v3 [paper]. Status from
  the eLife API [paper]. URLs opened: <https://data.mendeley.com/public-api/datasets/g5kcjp7jsk> ;
  <https://www.biorxiv.org/content/10.1101/2023.12.19.572452v3.full> ;
  <https://api.elifesciences.org/reviewed-preprints/95607>
- **Measured:**
  - Nine neuron classes \[host\]: RIM, DVC, HSN, AIY, URX, RIS, AFD, AWB, AWC.
  - Six were newly recorded \[paper\]: AWB (CX7893), AWC (CX3695), URX (ZG611), RIS (EG1285), DVC (UL2650), HSN
    (CX4857). RIM, AFD and AIY were previously published [paper].
  - Recordings \[paper\]: membrane-potential responses and steady-state currents; whole-cell current and voltage clamp
    (HEKA EPC-10, filtered at 2 kHz, sampled at 10 kHz, 50% series-resistance compensation); dissected young adults
    at room temperature.
  - Recordings per neuron and whether traces are averages: not stated in what I read.
- **Access \[host\]:** <https://data.mendeley.com/datasets/g5kcjp7jsk> (DOI 10.17632/g5kcjp7jsk.1, 2025-09-05), open.
- **Licence \[host\]:** "CC BY 4.0".
- **Size and format \[host\]:** one `exp.zip`, 37,482,294 bytes (SHA-256 given), laid out for the EP-GAN notebooks
  ("/exp" folder).
- **Model mapping:** voltage. Per-class Hodgkin-Huxley parameter inference for non-spiking neurons: the leak, K and Ca
  conductances and the resulting input resistance and rectification. This is the most directly reusable open set for
  setting per-class membrane time constants and nonlinearity. It includes RIS (sleep) and HSN (egg-laying), which
  matter for modulator links.
- **Known quality issues:** the method is limited to non-spiking neurons [2nd: search summary of the paper]. Reviewers
  questioned network-level claims without synaptic-input tests [2nd: eLife reviews via search snippet].
- **Table row:** `| EP-GAN (Kim/Liu/Shlizerman) | 1 per recording, 9 classes | strain-labelled | dissected | none | not stated | CC BY 4.0 | 37.5 MB | zip | yes |`

### Non-spiking neuron models' recordings: Naudin, Jiménez Laredo, Liu, Corson, 2022, *PLoS ONE* 17(5):e0268380

- **Verification:** Article [paper]; Mendeley API [host]. URLs opened:
  <https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0268380> ;
  <https://data.mendeley.com/public-api/datasets/6pgdf3zrp3>
- **Measured \[paper\]:** RIM, AIY, AFD. Current clamp from -15 to +35 pA. Averaged steady-state I-V from voltage clamp
  (n=3 RIM, 7 AIY, 3 AFD). RIM and AIY are reproduced from Liu et al. with consent; AFD is new.
- **Access \[host\]:** <https://doi.org/10.17632/6pgdf3zrp3.1> (2022-05-03), open. Code is linked via a Zenodo badge
  (<https://zenodo.org/badge/latestdoi/388104503>, not opened).
- **Licence \[host\]:** "CC BY 4.0".
- **Size and format \[host\]:** 37,457,466 bytes in 4 files:
  - `Fig 1_IV curves_AFD_RIM_AIY.xlsx` (13,146 B)
  - `Fig 1A_AIY Current-Clamp Trace.txt` (6.2 MB)
  - `Fig 1A_RIM Current-Clamp Trace.txt` (6.2 MB)
  - `Fig 1B_AFD Current-Clamp Trace ... 11 steps.txt` (25.0 MB)
- **Model mapping:** voltage. Full current-clamp step families plus I-V for three graded neurons. Directly constrains
  membrane time constant, input resistance and outward rectification of RIM, AIY and AFD.
- **Known quality issues:** small n. It overlaps with Liu 2018 (RIM/AIY) and with EP-GAN (all three).
- **Table row:** `| Naudin 2022 | 1 (RIM/AIY/AFD) | strain-labelled | dissected | none | IV n=3/7/3 | CC BY 4.0 | 37.5 MB | txt + xlsx | yes |`

### Enteric motor neuron action potentials: Jiang, Su, Zhang, Li, Tao, Liu, 2022, *Nat Commun* 13:2783

- **Verification:** Article (PMC9120479) [paper]; Mendeley API [host]. URLs opened:
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC9120479/> ; <https://data.mendeley.com/public-api/datasets/33mnszcbh8> ;
  Europe PMC search for the DOI [registry]
- **Measured \[paper\]:**
  - Current clamp: action potentials in AVL and DVB, with RIS and DVC as non-spiking GABAergic controls.
  - Voltage clamp: AVL (wild type and mutants) and DVB currents.
  - Dissected young adults, EPC-10.
- **Access \[host\]:** <https://doi.org/10.17632/33mnszcbh8.1> (2022-05-19), open. The article also has a 149.5 MB Source
  Data zip [paper].
- **Licence \[host\]:** "CC BY 4.0". Article: CC BY 4.0 [paper].
- **Size and format \[host\]:** 13 zip files, one per figure, 159,295,810 bytes.
- **Model mapping:** voltage. Spiking in two GABAergic enteric motor neurons and their synchrony (the defecation motor
  program), with mutant currents. Matters only if the model includes the posterior enteric circuit.
- **Known quality issues:** not assessed.
- **Table row:** `| Jiang 2022 AVL/DVB | 1 | strain-labelled | dissected | mutants | not checked | CC BY 4.0 | 159.3 MB | zip | yes |`

### Motor-circuit whole-cell recordings: Liu, Chen, Wang (Zhao-Wen Wang lab), 2020, *Nat Commun* 11:5076, "GABAergic motor neurons bias locomotor decision-making in C. elegans"

- **Verification:** Article (PMC7544903) [paper]; Europe PMC [registry]. I did not open the Source Data zip. URLs
  opened: <https://pmc.ncbi.nlm.nih.gov/articles/PMC7544903/> (offsets 0 and 100k)
- **Measured \[paper\]:**
  - Whole-cell current and voltage clamp in dissected worms of D-MNs (VD5, DD3), A-MN VA5, B-type VB6, AVA (AVAL,
    AVAR) and AVB. Membrane potential recorded with no current injection.
  - Recording counts and protocols were not visible in what I read.
- **Access:** journal-hosted Source Data, a 311.1 MB zip [paper]. Data statement: "This work did not include any data
  which mandated deposition in public databases."
- **Licence:** article CC BY 4.0 [paper]; Source Data licence not separately stated.
- **Size and format:** 311.1 MB zip (contents not verified).
- **Model mapping:** voltage. Resting potentials and I-V of command interneurons and ventral-cord motor neurons:
  membrane properties for the motor layer. Nicoletti 2024 fitted AVA and VA5 models to Wang-lab recordings [paper].
- **Known quality issues:** whether Source Data hold full traces or only plotted values: not verified.
- **Table row:** `| Liu/Chen/Wang 2020 | 1 | strain-labelled | dissected | not checked | not checked | article CC BY 4.0 | 311.1 MB | zip | yes (journal) |`

### Rectifying gap junctions at AVA to A-motor neuron synapses: Liu, Chen, Mailler, Wang, 2017, *Nat Commun* 8:14818

- **Verification:** Article (PMC5364397) [paper]; Europe PMC [registry]. URL opened:
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC5364397/>
- **Measured \[paper\]:**
  - Paired voltage- and current-clamp recordings of AVA with VA5, and in supplementary pairs AVA with VA8, DA4 and VB6.
  - Junctional current only when VA5 is more depolarised than AVA, so current flows from A-MNs into AVA. Coupling
    coefficient is large VA5→AVA and small in reverse. Coupling is lost in unc-7 and unc-9 mutants.
  - ChR2 in command interneurons (glr-1p; 2-s pulses every 30 s, 470 nm, 6.7 mW/mm2) evoked PSC bursts in VA5, blocked
    by d-tubocurarine.
- **Access:** data statement: "All data generated or analysed during this study are included in this published
  article" and "its Supplementary Information files" [paper]. Figures/SI only; raw traces not deposited.
- **Licence:** article CC BY 4.0 [paper].
- **Size and format:** n/a.
- **Model mapping:** voltage. Direct evidence for gap-junction rectification direction and its innexin basis at a
  specific connectome edge class (AVA–A-MN). It argues for directed (asymmetric) gap conductances, and the same paper
  shows the chemical synapse there is cholinergic and excitatory.
- **Known quality issues:** none assessed.
- **Table row:** `| Liu 2017 AVA-VA5 | 2 (paired) | strain-labelled | dissected | ChR2 in command INs; unc-7/unc-9 | not checked | article CC BY 4.0 | n/a | figures/SI | figures only |`

### ASH to AVA synaptic transmission: Lindsay, Thiele, Lockery, 2011, *Nat Commun* 2:306

- **Verification:** Article (PMC3935721, author manuscript) [paper]. URL opened:
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC3935721/>
- **Measured \[paper\]:**
  - ChR2 under sra-6 (strong in ASH, weak in ASI); whole-cell recording of AVA in nearly intact worms, plus ASH.
  - Postsynaptic current with onset transient, sustained component and increased unitary event frequency; PSPs.
  - Reversal near 0 mV (-7.0 ± 3.2 mV), so glutamatergic and excitatory; reduced by CNQX and MK-801, abolished in
    unc-13. Onset latency falls with irradiance.
- **Access:** no data statement; Supplementary Fig. S1 only [paper]. Figures only.
- **Licence:** not recorded in this session.
- **Model mapping:** voltage. Sign (excitatory), kinetics and dose-response of one sensory-to-command edge: a rare
  direct edge-level measurement for checking a fitted ASH→AVA weight and synaptic time constant.
- **Table row:** `| Lindsay 2011 ASH→AVA | 1-2 | physiological + GFP/RFP | nearly intact | ChR2 in ASH; CNQX/MK-801; unc-13 | not checked | not recorded | n/a | figures | figures only |`

### Hodgkin-Huxley models fitted to recordings: Nicoletti et al. 2019 *PLoS ONE* 14(7):e0218738 (AWCon, RMD) and Nicoletti et al. 2024 *PLoS ONE* 19(3):e0298105 (AIY, RIM, AVA, VA5, VB6, VD5)

- **Verification:** 2024 article [paper]; GitHub and ModelDB pages [host]. The 2019 article itself was not opened (seen
  via search [2nd]). URLs opened: <https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0298105> (offsets 0
  and 100k) ; <https://github.com/martinanicoletti92/CelegansInterMotorNeuronsModels> ; <https://modeldb.science/2017403> ;
  <https://modeldb.science/267187>
- **Measured \[paper, 2024\]:** models fitted to published recordings:
  - AVAL/AVAR: current clamp from Liu P. et al. 2020; raw recordings received from Zhao-Wen Wang and Ping Liu.
  - AIY and RIM: from Liu Q. et al. 2018.
  - VA5: voltage clamp from Niu et al. 2020 and Liu P. et al. 2014.
  - VB6 and VD5: steady-state I-V only; the paper says no current-clamp recordings have been published for these.
  - Whether the other traces were digitised from figures is not stated.
- **Access \[host\]:** GitHub (NEURON .mod files for about 22 currents plus intracellular Ca, Python
  simulation/vclamp/iclamp scripts); ModelDB 2017403 (NEURON/Python) and 267187 (XPPAUT; 2019 AWCon/RMD). No
  experimental traces are visible in the GitHub listing.
- **Licence \[host\]:** GitHub: not stated; ModelDB pages: not stated. The article is CC BY 4.0 [paper].
- **Model mapping:** prior per-class conductance sets (and knockouts done by zeroing a conductance) to initialise or
  regularise membrane parameters; not new data.
- **Table row:** `| Nicoletti 2019/2024 models | 8 classes modelled | n/a | n/a | in-silico KOs | n/a | not stated | not shown | XPPAUT / NEURON | yes (code) |`

### OpenWorm ChannelWorm, ChannelWorm2 and c302

- **Verification:** GitHub pages [host]. URLs opened: <https://github.com/openworm/ChannelWorm> ;
  <https://github.com/openworm/ChannelWorm2> ; <https://github.com/openworm/c302>
- **Content \[host\]:**
  - ChannelWorm: archived 2018-08-27; MIT. A curated ion-channel spreadsheet (`data/`), NeuroML2 channel models from
    parameter fitting, and plot-digitising and fitting tools.
  - ChannelWorm2: licence not stated; last dated news August 2018. `raw_data/` holds curated spreadsheets,
    publication plots and digitised time series (JSON exported from "Chopen"), plus NML2 models and GA fitting code.
  - c302: MIT. Generates NeuroML2 networks from the connectome; parameter sets A-D. `parameters_C.py` entries are
    tagged "BlindGuess".
- **Access:** open code repositories.
- **Model mapping:** channel kinetics digitised from the literature. Useful as priors only; c302's network parameters
  are largely unconstrained guesses.
- **Known quality issues \[host\]:** unmaintained since 2018; counts of channels and papers not stated.
- **Table row:** `| OpenWorm ChannelWorm/2, c302 | n/a | n/a | n/a | n/a | n/a | MIT / not stated / MIT | not shown | spreadsheet, JSON, NeuroML2 | yes (code) |`

### Part 2: compact list (figures only, or not verified)

- **Katta, Sanzeni, Das, Vergassola, Goodman 2019 J Gen Physiol 151(10):1213**: in vivo whole-cell recording of
  mechanoreceptor currents in ALM touch neurons with a piezo-driven bead. Analysis code is on GitHub
  (wormsenseLab/Matlab-PatchMaster, Puncta_analysis). No raw-trace deposit was seen in the first 100k characters of
  the PMC text [paper]; a data statement later in the page was not checked. The Goodman lab's Stanford Digital
  Repository deposits seen in search are EM, not electrophysiology [2nd].
- **Goodman et al. 1998 Neuron (ASER and others), Mellem et al. 2008 Nat Neurosci (RMD action potentials), Ramot et
  al. 2008 Nat Neurosci (AFD), Narayan et al. 2011 PNAS (AFD→AIY graded, tonic, peptidergic transmission)**: known
  only from citations and snippets [2nd]. Data access not verified; presumed figures only.
- **Gao et al. 2018 eLife 7:e29915 (Zhen lab)**: whole-cell recordings are of body-wall muscle (rhythmic PSCs, action
  potentials) under optogenetic stimulation of A-MNs, B-MNs and AVA; neurons are calcium-imaged [paper]. Data
  statement not reached.
- **Shindou et al. 2019 Sci Rep (ASEL dendritic regenerative depolarisation)**: page redirect blocked; not verified.
- **On request:** Nicoletti et al. 2024 acknowledge receiving raw AVA recordings from Zhao-Wen Wang and Ping Liu
  [paper], so Wang-lab traces are obtainable on request.

### What Part 2 can and cannot constrain

The open voltage data (Liu 2018; Naudin 2022; EP-GAN 2025; Jiang 2022; Wang-lab Source Data 2020) cover about 15
identified classes: AWA, RIM, AIY, AFD, AWB, AWC, URX, RIS, DVC, HSN, AVL, DVB, AVA, AVB and several ventral-cord
motor neurons.

They constrain:

- per-class membrane time constant, input resistance and rectification, from step families and I-V curves;
- where a graded unit is inadequate (AWA, AVL, DVB spike);
- two edge-level facts: ASH→AVA is excitatory glutamatergic, and the AVA–A-MN gap junctions rectify toward AVA.

They do not constrain:

- most edge weights;
- the signs of chemical synapses beyond those few edges;
- neuromodulator coupling.

All recordings are dissected preparations at room temperature, so their relation to the calcium time constants in
the imaging datasets above needs a calcium/voltage forward model.

## E. Expression, modulation and wiring priors

Compiled 2026-11-14 with WebSearch and WebFetch only; no data file was downloaded. Provenance tags:
**host** = the page of the site that serves the data (cengen.org, figshare or Zenodo API, wormwiring.org,
nemanode.org, the GitHub repo); **paper** = the article itself (journal page, PMC or Europe PMC full text,
bioRxiv landing page); **second-hand** = a search-result snippet, or a third party's description (the OpenWorm
ConnectomeToolbox pages, the Netzschleuder mirror, Sciety).

Access limits met in this session, which leave gaps marked below: cell.com, sciencedirect.com,
royalsocietypublishing.org, worm.peptide-gpcr.org and discovery.ucl.ac.uk returned HTTP 403. PMC began
returning a browser-check page part-way through, so the full text of Cook et al. 2019 and Brittin et al. 2021
was never read. wormatlas.org and ftp.wormatlas.org failed TLS certificate checks ("unable to verify the first
certificate" and "certificate has expired"). The session's web-search budget ran out before the last search
(for Cook 2019's methods). The 2025–2026 coverage of new connectomes may therefore be incomplete.

______________________________________________________________________

### E1. CeNGEN L4 single-cell atlas: Taylor et al., 2021, *Cell* 184(16):4329–4347

- **Verification:**
  - Files, sizes, thresholds and app list: host.
  - Cell counts, class coverage, threshold accuracy and limitations: paper (PMC author manuscript PMC8710130).
  - The cengenDataSC R package: host (its GitHub page), after a search snippet pointed to it.
  - URLs opened: <https://www.cengen.org/> ; <https://www.cengen.org/downloads/> ; <https://www.cengen.org/single-cell-rna-seq/> ;
    <https://pmc.ncbi.nlm.nih.gov/articles/PMC8710130/> ; <https://github.com/cengenproject/CengenApp> ;
    <https://github.com/AlexWeinreb/cengenDataSC> ; <https://www.cell.com/cell/fulltext/S0092-8674(21)00758-3> (403).
- **Measured:**
  - Animals and method: L4 hermaphrodites; 10x Genomics scRNA-seq of FACS-isolated neurons (host).
  - Cells: 100,955 from 17 experiments (host). The neuron-only analysis has 70,296 cells; 27,427 non-neuronal cells were separated out (paper).
  - Coverage: all 118 anatomical classes, resolved as 128 transcriptionally distinct neuron types, with subtypes in 10 classes (paper).
  - Expression calls: per-gene percentile thresholding of the proportion of cells detecting a gene, against a ground truth of 160 genes with high-confidence patterns. Four stringency levels; threshold 2 is used in the paper's main analyses, with TPR 0.81 and FDR 0.14 (paper).
  - Bulk companion (host):
    - Bulk RNA-seq of sorted neurons: normalised integrated counts for 143 replicates and raw counts for 160 samples.
    - A comparison set of 29 samples from 8 classes (GEO GSE169137).
    - A splicing/bulk set covering 41 neuron types, with sequencing ongoing.
- **Access:**
  - Downloads: <https://www.cengen.org/downloads/> (host; direct links, downloadable without request).
  - Raw reads: GEO GSE136049 (single cell) and GSE169137 (bulk).
  - Apps: <https://cengen.shinyapps.io/L4app> (current); legacy CengenApp and CengenApp-legacy.
  - Code: <https://github.com/cengenproject>.
  - R package: `devtools::install_github("AlexWeinreb/cengenDataSC")` (host).
- **Licence:**
  - cengen.org: not stated. The downloads page footer only credits WordPress and the theme.
  - CengenApp repo: no licence file shown.
  - cengenDataSC: GitHub label "GPL-3.0 license" (host; the LICENSE.md text itself was not read).
  - Article: not verified. The PMC copy is an author manuscript and the retrieved text carried no licence line.
- **Size and format (host):**
  - Thresholded matrices: four "binary expression" CSVs, `021821_liberal_threshold1.csv`, `021821_medium_threshold2.csv`, `021821_conservative_threshold3.csv` and `021821_stringent_threshold4.csv`, sizes not listed. A value of 0 means not expressed; a value above 0 is the mean TPM per gene per neuron type.
  - All cells: Seurat RDS 500 MB, Seurat v5 RDS 500 MB, Monocle3 RDS 300 MB.
  - Neuron-only: Seurat RDS 4 GB, Monocle RDS 300 MB.
  - Bulk: TSV count tables, sizes not listed.
  - cengenDataSC: `cengen_sc_2` is a 13,669 genes × 128 cell types binarised matrix, used as a mask over `cengen_TPM`; which threshold it encodes is not stated.
- **Model mapping:**
  - Receptor placement: which neuron class expresses each receptor gene. That covers ionotropic and metabotropic ACh, Glu and GABA receptors, monoamine receptors, neuropeptide GPCRs and innexins, so it places couplings for receptor-specific modulator and synapse terms.
  - Transmitter identity: from eat-4, unc-17, unc-25 and unc-47, the presynaptic input to a sign rule (see E8).
  - Neuropeptide precursor expression: the source cells of each peptide pool.
  - TPM values: at most a weak prior on relative receptor gain.
  - Resolution is the neuron class or type (128), so left/right members and individual cells within a class share values.
- **Known quality issues:**
  - Thresholding sensitivity: four thresholds trade TPR against FDR (paper). Ripoll-Sánchez et al. 2023 found threshold 4 may undercount (E3, paper).
  - False negatives: the nine classes with the fewest detected genes and lowest TPR "are likely to have the highest rates of false negatives" (paper).
  - Some classes are under-represented, probably from dissociation bias (paper).
  - DD and VD could not be separated in the 2021 analysis (paper). The host says the updated L4 dataset now separates VD from DD and VA1 from VA, and removes cells with high unfolded-protein-response gene levels.
  - The 3′ bias means isoforms are not resolved (host), and non-polyadenylated ncRNAs are largely excluded (paper).
  - mRNA is not protein (stated as a general caveat in Wang et al. 2024, E6).
  - L4, not adult.
- **Table row:** `| CeNGEN L4 scRNA-seq (Taylor 2021) | all 118 classes / 128 types, L4 herm | class-level (cluster annotation) | n/a | n/a | 100,955 cells, 17 experiments | site: not stated; R pkg GPL-3.0 | RDS 300 MB–4 GB; CSVs size not listed | CSV, RDS, TSV, GEO FASTQ | cengen.org/downloads (open) |`

### E2. CeNGEN 2024–2026 extensions (L1, adult hermaphrodite, adult male): CeNGEN consortium; L1: Taylor et al., bioRxiv 2025

- **Verification:**

  - Dataset descriptions: host (cengen.org stage pages).
  - L1 preprint metadata: paper (bioRxiv landing page).
  - URLs opened: <https://www.cengen.org/l1/> ; <https://www.cengen.org/adult/> ; <https://www.cengen.org/male/> ;
    <https://www.biorxiv.org/content/10.1101/2025.11.21.689793v1> ; <https://www.cengen.org/single-cell-rna-seq/>.

- **Measured (host):**

  | Dataset | Cells (neurons) | Samples | Animals and method | Classes |
  |---|---|---|---|---|
  | L1 | 161,072 (89,073) | 16, from 10 marker strains | Mid–late L1; live FACS of fluorescent neurons; 10x Genomics | 91 embryonically born classes, 17 classes born in mid–late L1, and neuronal progenitors |
  | Adult hermaphrodite | 68,923 (38,958) | 8 | Day-1 N2; all diploid cells, methanol-fixed and DAPI-sorted 2N (germline excluded) | Class count not given; clusters annotated by top markers against L4 |
  | Adult male | 67,346 (36,117) | 6 | Day-1 *him-8(e1498)*, size-filtered to about 85% males; same diploid-sort method | Sex-shared clusters annotated against L4. Male-specific clusters annotated with reporters; unassigned ones are labelled "male.X" and "annotation of male-specific neurons is ongoing" |

  Each dataset has four dynamic thresholds, with these TPR/FDR values:

  | Threshold | L1 | Adult hermaphrodite | Adult male |
  |---|---|---|---|
  | 1 | 0.853 / 0.202 | 0.855 / 0.238 | 0.806 / 0.227 |
  | 2 | 0.807 / 0.170 | 0.813 / 0.200 | 0.755 / 0.190 |
  | 3 | 0.689 / 0.127 | 0.699 / 0.151 | 0.635 / 0.155 |
  | 4 | 0.577 / 0.103 | 0.579 / 0.117 | 0.499 / 0.129 |

  - Preprint (paper): Taylor, Olson, Ripoll-Sánchez … Miller, "A gene expression atlas of a juvenile nervous system", posted 2025-11-22. It reports remodelled neuropeptide signalling networks and a role for *lin-14*.

- **Access:**

  - Apps: <https://cengen.shinyapps.io/L1app> , <https://cengen.shinyapps.io/adult> , <https://cengen.shinyapps.io/male> (host).
  - Downloadable matrices: not found. None of the stage pages, nor the downloads page, lists L1, adult or male files; the downloads page lists only L4 files.

- **Licence:** host: not stated. L1 preprint: "CC BY 4.0 International" (bioRxiv page).

- **Size and format:** not stated (apps only, as far as verified).

- **Model mapping:**

  - The adult hermaphrodite set gives stage-matched receptor and transmitter placement for an adult on-plate model; L4 differs from adult.
  - The male set gives sex-specific placement.
  - L1 can be paired with Witvliet's L1 connectomes (W4) for developmental checks.

- **Known quality issues:**

  - Male TPR at threshold 4 is 0.499 (host).
  - Adult and male classes are annotated by transfer from L4 markers (host), and male-specific annotation is incomplete (host).
  - The adult datasets come from fixed whole-animal diploid sorts, a different capture from L4's live marker-based FACS (host), so matrices from different stages are not directly comparable.

- **Table row:** `| CeNGEN L1 / adult herm / adult male scRNA-seq | all neurons per stage (class counts: L1 91+17; adult, male not stated) | class-level | n/a | n/a | L1 161,072 cells/16 samples; adult 68,923/8; male 67,346/6 | site: not stated; L1 preprint CC BY 4.0 | not stated | Shiny apps (files not found) | cengen.shinyapps.io (open apps) |`

### E3. Neuropeptidergic connectome and NemaMod: Ripoll-Sánchez et al., 2023, *Neuron* 111:3570–3589 (pages from a search snippet)

- **Verification:**
  - Content, method and limitations: paper (PMC7615469).
  - Files, sizes and licences: host (figshare API and the GitHub repo).
  - App: host (the NemaMod Shiny page; nemanode.org redirects there).
  - URLs opened: <https://pmc.ncbi.nlm.nih.gov/articles/PMC7615469/> ; <https://api.figshare.com/v2/collections/6895870> ;
    <https://api.figshare.com/v2/collections/6895870/articles?page_size=50> ; <https://api.figshare.com/v2/articles/24517180> ;
    <https://api.figshare.com/v2/articles/24187119> ; <https://api.figshare.com/v2/articles/24187113> ;
    <https://api.figshare.com/v2/articles/24517165> ; <https://api.figshare.com/v2/articles/24187104> ;
    <https://github.com/LidiaRipollSanchez/Neuropeptide-Connectome> ; <https://www.nemamod.org> (301 redirect) ;
    <https://vwl8jb-lidiaripollsanchez.shinyapps.io/NemaMod/>.
- **Measured (paper):**
  - Neurons: all 302, hermaphrodite.
  - Couples: from Beets et al.'s 461 peptide–GPCR pairs (E4), 92 neuropeptide–receptor gene couples were kept at EC50 ≤ 500 nM, giving 51 GPCRs with at least one ligand. 100 nM was used as a sensitivity threshold.
  - Expression: CeNGEN threshold 4, chosen against 17 neuropeptide precursor and 9 GPCR single-copy knock-in reporters.
  - Distance models:
    - long range: any pair;
    - mid range: processes in the same body region (head including pharynx, midbody, tail);
    - short range: processes in the same bundle;
    - a nerve-ring-strata model was judged too restrictive.
  - Edge weights range 1–18; network density is 0.3437 (short range) and 0.4429 (mid range).
- **Access:**
  - figshare collection <https://doi.org/10.6084/m9.figshare.c.6895870>. The paper cites v1; the API shows v4, published 2023-11-07, with 16 items (host).
  - Code: GitHub (host), plus Zenodo 10.5281/zenodo.8387059 cited by the paper (not opened).
  - NemaMod (host): a Shiny app with a network-by-neuron view and a network-by-gene view, a diffusion-threshold slider, and download of an image or data matrix.
  - Downloadable without request: yes.
- **Licence:**
  - figshare datasets: "CC BY 4.0" (host, each dataset item checked).
  - GitHub repo: "MIT license" label (host; LICENSE text not read).
  - NemaMod: not stated.
  - Article: "CC BY license" (paper).
- **Size and format (host, figshare):**
  - Aggregate networks: 3 CSVs, short/mid/long range, 185,773, 185,874 and 186,152 bytes.
  - Individual networks: 276 CSVs (92 couples × 3 ranges), each 185,444 bytes, 51,182,544 bytes in total.
  - Networks grouped by neuropeptide: 147 CSVs (49 peptides × 3 ranges), each 185,444 bytes.
  - Ligand–receptor table at 500 nM: `20230602_Ligand_receptors_interactions_all_EC50_explicit_500nM.csv`, 1,265 bytes.
  - `neuron_information.csv`: 8,343 bytes.
  - GitHub: CSV adjacency matrices and edge lists, plus xlsx, txt, Cytoscape `.cys` and PDF files.
- **Model mapping:**
  - Neuropeptide couplings: a binary sender→receiver mask per peptide–receptor couple (or per peptide), so each couple can carry one fitted coupling strength.
  - Receptor placement: GPCR-expressing cells.
  - Source pools: precursor-expressing cells.
  - The distance model is a spatial mask on which couples are allowed.
  - No sign or valence; strengths are not given beyond inclusion at the EC50 cut-off.
- **Known quality issues:**
  - Stated in the paper:
    - Threshold 4 may undercount expression.
    - Many receptors are still orphans, so the true network is denser.
    - The diffusion ranges are assumptions; peptides probably are not confined to bundles.
    - Non-neuronal senders and receivers are excluded.
    - No post-transcriptional, developmental or environmental variation is modelled.
    - It is a draft built from expression plus in-vitro pharmacology.
  - Version drift (host): the GitHub README says the adjacency matrices were updated on 02/02/2024 with new scRNA-seq data for *dmsr-5* and *npr-34*. The figshare CSVs date from 2023, so the two copies differ.
- **Table row:** `| Neuropeptide connectome (Ripoll-Sánchez 2023) + NemaMod | 302 neurons, 92 NPP–GPCR couples, 3 distance models | yes (neuron names) | n/a | n/a | n/a (derived from CeNGEN L4 + Beets EC50) | figshare CC BY 4.0; code MIT; app not stated | ~51 MB individual + ~27 MB grouped + 0.56 MB aggregate (byte counts above) | CSV adjacency matrices | figshare c.6895870, GitHub, nemamod.org (open) |`

### E4. Peptide–GPCR deorphanisation (EC50 table): Beets et al., 2023, *Cell Reports* 42(9):113058 (volume from a search snippet)

- **Verification:**
  - Numbers, method, data statement and supplementary list: paper (Europe PMC / PMC author manuscript PMC7615250).
  - Licence: paper plus Europe PMC metadata.
  - The publisher's supplementary page was not reachable (403).
  - URLs opened: <https://pmc.ncbi.nlm.nih.gov/articles/PMC7615250> (two passes) ;
    <https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.1016/j.celrep.2023.113058&format=json&resultType=core> ;
    <https://openworm.org/ConnectomeToolbox/Beets_2023/> ; <https://doi.org/10.1016/j.celrep.2023.113058> (redirect) ;
    <https://www.sciencedirect.com/science/article/pii/S2211124723010690> (403) ; <https://worm.peptide-gpcr.org> (403) ;
    <https://doaj.org/article/443e8531c5bd43708644a2c6199b3654> (403).
- **Measured (paper):**
  - Receptors: 161 cloned, about 87% of at least 149 putative peptide GPCR genes.
  - Peptides: a library of 344 synthetic peptides.
  - Screen: 55,384 pairs screened, 416 putative hits for 114 GPCRs, and 776 pairs taken to concentration–response.
  - Validated couples: 459 in the Results, 461 in the Highlights and Discussion, on 66 or 68 receptors respectively.
  - EC50 range: 0.1 pM to 22 µM, with 73% of couples sub-micromolar.
  - Assay: aequorin Ca²⁺ luminescence in CHO cells with promiscuous Gα16. Follow-ups used *Xenopus* oocyte TEVC and a HEK293T cAMP-luciferase reporter.
  - The text cites Tables S1–S9; Table S6 lists the EC50 values.
- **Access:**
  - Supplementary files on PMC (paper host):
    - Data S1, ZIP, 310.5 KB: GPCR cDNA sequences.
    - Document S1, PDF, 2.6 MB: Figures S1–S12 and Table S1.
    - Document S2, PDF, 7 MB: contents not described.
  - Which file holds Table S6 is not stated, and whether the publisher offers Tables S2–S9 as spreadsheets is not verified.
  - Data statement: "All data generated in this study is reported in this paper and will be shared by the lead contact upon request."
  - A machine-readable 500 nM subset is on figshare (E3; CSV, 1,265 bytes, CC BY 4.0).
  - Downloadable: PDFs yes; the full EC50 table as data is on request or in a PDF (not verified).
- **Licence:**
  - The PMC page carries two statements: "This work is licensed under a BY 4.0 International license" and "This is an open access article under the CC BY-NC-ND license" (paper host). Europe PMC metadata says "cc by".
  - No separate data licence.
- **Size and format:** as listed above (ZIP and PDFs); EC50 values as a PDF table, as far as verified.
- **Model mapping:**
  - Which peptide activates which receptor, and at what potency. log EC50 is a prior on a coupling's sensitivity, and the inclusion cut-off sets which couplings exist.
  - It does not give the sign: the Gα16 readout forces a Ca²⁺ signal, so the native G-protein class (Gq, Gs or Gi/o) and the valence are not measured.
- **Known quality issues (paper):**
  - Not every receptor expressed functionally, and some may need co-factors.
  - Gα16 is less effective for Gi-coupled receptors.
  - EC50 in vivo may differ.
  - Constitutive or G-protein-independent signalling is missed.
  - The map is incomplete.
  - The text gives two counts (459 vs 461 couples; 66 vs 68 receptors).
- **Table row:** `| Peptide–GPCR EC50 map (Beets 2023) | 161 GPCRs × 344 peptides; 459–461 couples | n/a (molecular) | n/a | n/a | n/a (in vitro, CHO/oocyte/HEK) | article: CC BY 4.0 and CC BY-NC-ND both stated; data on request | Data S1 310.5 KB; PDFs 2.6 MB, 7 MB | ZIP, PDF tables; 500 nM subset CSV on figshare | PMC7615250 supplements; lead contact |`

### E5. Multilayer connectome (monoamine and neuropeptide layers): Bentley et al., 2016, *PLoS Comput Biol* e1005283 (published 2016-12-16)

- **Verification:**
  - All facts: paper (PLOS article page, three passes).
  - The S1 Dataset itself was not opened.
  - URL opened: <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005283>.
- **Measured (paper):**
  - Sources:
    - a literature search for monoamine receptor, transporter and synthesis genes;
    - cell-level expression from WormBase WS248 and WormWeb (2014-11-16);
    - new GFP reporters for *lgc-53*, *dop-5* and *dop-6*.
  - Receptors:
    - five serotonin, three octopamine, four tyramine and four dopamine receptors;
    - *lgc-53*;
    - putative *dop-5*/*dop-6* in an extended network.
  - Monoamine layer:
    - 18 of 302 neurons release monoamines, and 251 (83%) express at least one monoamine receptor.
    - The aggregate layer has 1,940 directed connections, about 96% of them not in the wired layers.
  - Neuropeptide layer: 15 neuropeptides and 12 receptors with plausible EC50, 239 neurons and 7,035 connections.
  - No distance constraint.
  - Wired layers: Varshney 2011 plus the Albertson & Thomson pharynx.
- **Access:**
  - S1 Dataset (ZIP, edge lists for the monoamine and neuropeptide networks): <https://doi.org/10.1371/journal.pcbi.1005283.s004> (paper host).
  - S1–S14 Tables (DOCX): expression per cell type, receptor expression, and peptide receptor–ligand binding.
  - Downloadable without request: yes.
- **Licence:**
  - Article: "© 2016 Bentley et al." under the Creative Commons Attribution License (paper).
  - SI: not separately stated.
- **Size and format:** ZIP of edge lists and DOCX tables; sizes not shown on the page.
- **Model mapping:**
  - Monoamine source cells and receptor placement per receptor gene: a sender→receiver mask per monoamine–receptor pair, so per-receptor coupling strengths can be fitted.
  - Receptor identity allows a sign to be assigned from receptor biology (not given in the paper).
  - The neuropeptide layer is superseded by E3.
- **Known quality issues (paper):**
  - Valence and strength of the links are "largely unknown".
  - The neuropeptide layer is partial.
  - Monoamine receptor expression may be under-reported (false negatives), and some reporters label only some cells.
  - The *dop-5*/*dop-6* assignments are provisional.
  - Gap junctions are treated as undirected.
  - Not mapped: insulin-like peptides, purines, and metabotropic ACh/GABA.
  - Pre-dates CeNGEN and the reporter-allele atlas (E6), which reports new monoamine expression sites.
- **Table row:** `| Multilayer connectome (Bentley 2016) | 302 neurons; monoamine layer 1,940 edges; peptide layer 7,035 edges | yes | n/a | n/a | n/a (literature + reporters) | article CC BY | not shown | ZIP edge lists, DOCX tables | PLOS SI (open) |`

### E6. Neurotransmitter atlas (reporter alleles): Wang et al., 2024, *eLife* 95402 (VOR 2024-10-18)

- **Verification:**
  - Content and files: paper (PMC11488851, two passes).
  - Licence and date: Europe PMC metadata.
  - The reviewer remark about *snf-11*: search snippet only (second-hand).
  - URLs opened: <https://elifesciences.org/articles/95402> (did not render) ;
    <https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.7554/eLife.95402&format=json&resultType=core> ;
    <https://pmc.ncbi.nlm.nih.gov/articles/PMC11488851/>.
- **Measured (paper):**
  - 16 CRISPR/Cas9 knock-in reporter alleles at 15 loci: *eat-4*, *unc-17*, *unc-25*, *unc-47*, *unc-46*, *tph-1*, *bas-1*, *cat-1*, *cat-2*, *tdc-1*, *tbh-1*, *snf-3*, *oct-1*, *mod-5* (an earlier non-nuclear allele) and *hdl-1* (no expression found).
  - Identity: NeuroPAL (otIs669, or otIs696 for *eat-4*), plus dye filling and landmark reporters.
  - Animals: young-adult hermaphrodites and adult males; L1–L4 shown for *tph-1*.
  - Table 1 covers monoamine-uptake neurons, and Table 2 gives per-class expression categories, including co-expression.
- **Access:** eLife and PMC supplementary files; downloadable without request: yes.
- **Licence:**
  - Article: "© 2024, Wang et al", distributed under the Creative Commons Attribution License (paper); Europe PMC metadata "CC BY".
  - Data statement: "All data generated or analysed during this study are included in the manuscript and supporting files."
- **Size and format (PMC host listing):**
  - supp1.xlsx, 37.7 KB: scRNA vs reporters, hermaphrodite.
  - supp2.xlsx, 71.2 KB: updated hermaphrodite expression.
  - supp3.xlsx, 99.1 KB: male-specific neurons.
  - supp4.xlsx, 10.2 KB: dimorphism in sex-shared neurons.
  - supp5.xlsx, 10 KB: summary of updates.
  - MDAR checklist, docx, 87.8 KB.
- **Model mapping:**
  - The presynaptic transmitter for each neuron (ACh, Glu, GABA, 5-HT, DA, TA, OA) is the transmitter half of a per-connection sign rule.
  - Co-transmission flags mark candidate mixed-sign outputs.
  - Monoamine source cells for the modulator pools, in both sexes.
  - The OpenWorm ConnectomeToolbox lists "Wang et al. 2024 Herm/Male" datasets (second-hand; see W9).
- **Known quality issues (paper):**
  - Dim and variable reporter expression, possibly below detection or present in only some animals.
  - Neurons with incomplete pathways: AVJ (*cha-1* transcripts but no *unc-17* reporter); AVL, PVX and PVY (*cat-1* only); IL2 (*tbh-1* without *tdc-1*).
  - *unc-47* is expressed without evidence of GABA synthesis or uptake, suggesting an unidentified transmitter.
  - The *unc-25* isoform b.1 reporter showed no expression.
  - Transmitter identity alone does not fix the postsynaptic sign.
  - A reviewer noted the *snf-11* knock-in did not improve its pattern (second-hand).
- **Table row:** `| Neurotransmitter atlas (Wang 2024) | all neurons, herm + male; 7 transmitter systems | yes (NeuroPAL) | n/a | n/a | not stated in what was read | article CC BY 4.0 | 5 xlsx, 10–99 KB | XLSX | eLife/PMC supplements (open) |`

### E7. Innexin (gap-junction gene) expression atlas: Bhattacharya et al., 2019, *Cell* 176(5):1174–1189.e16 (citation from a search snippet)

- **Verification:**
  - Content: paper (PMC author manuscript PMC10064801).
  - Citation and abstract: Europe PMC metadata and search snippets (second-hand).
  - URLs opened: <https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.1016/j.cell.2018.12.024&format=json&resultType=core> ;
    <https://pmc.ncbi.nlm.nih.gov/articles/PMC10064801/>.
- **Measured (paper):**
  - Innexins: 25 genes in the genome, 14 expressed in the nervous system; *inx-21*/*inx-22* are gonad-specific.
  - Reporters: fosmid SL2::NLS::YFP::H2B transcriptional reporters and GFP translational fusions, plus CRISPR knock-ins for *inx-2* and *inx-6*. Expression scored present or absent.
  - Identification: Nomarski optics, landmark reporters (*eat-4*, *cho-1*, *unc-47*) and DiD dye filling.
  - Stages: L1 diapause, L2 to adult, and dauer.
  - Results:
    - 98 of 118 classes have unique innexin combinations; ASK and RIM express up to 11, and I6 expresses one.
    - The mean number of electrical partners per class is 9.7.
    - 86 of 118 classes change in dauer.
  - The main neuron × innexin matrix is Fig. 1B; developmental and dauer changes are in Table S2.
- **Access:** PMC supplementary files; the publisher page was not opened. Downloadable without request: yes, as figures and tables.
- **Licence:** not verified; Europe PMC marks it "not open access" and the record lists no licence.
- **Size and format:** figure supplements S1–S3 are 485.6 KB, 21.2 MB and 14.6 MB PDFs (paper host); table formats are not stated.
- **Model mapping:**
  - A prior on gap-junction existence and composition per neuron pair, for example down-weighting EM-annotated gap edges between classes that share no innexin, or flagging candidate rectifying, heterotypic pairs.
  - Stage-specific (dauer vs adult) gap-strength changes.
  - It cannot fix strengths.
- **Known quality issues (paper):**
  - Reporters show transcription, not protein or junctions.
  - The rules for heterotypic pairing are unknown, so the partners of most junctions cannot be inferred from expression.
  - *inx-10b* was not reliably detected.
  - Dauer data are reporter-only.
  - The text says 11 of 14 innexins are plastic in the Results and 12 of 14 in the Discussion.
- **Table row:** `| Innexin atlas (Bhattacharya 2019) | 118 classes × 14 neuronal innexins; L1–adult + dauer | yes | n/a | n/a | not stated in what was read | not verified | fig. PDFs 0.5–21 MB; tables not stated | figures/tables | PMC10064801 supplements |`

### E8. Synaptic polarity prediction: Fenyves, Szilágyi, Vassy, Sőti & Csermely, 2020, *PLoS Comput Biol* 16(12):e1007974

- **Verification:**
  - Content: paper (PLOS page).
  - Web tool: host, which resolved but returned only the title "EleganSign".
  - URLs opened: <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007974> ; <http://EleganSign.linkgroup.hu>.
- **Measured (paper):**
  - Connectome: the WormWiring adult hermaphrodite chemical network, with 3,638 connections, 20,589 synapses and 297 neurons.
  - Transmitters: Glu, ACh and GABA identities from the literature.
  - Receptors: 62 ionotropic receptor genes classed by ligand and channel type (cation or anion), 42 of them expressed postsynaptically. Expression came from WormAtlas, WormBase and binarised CeNGEN.
  - Rule (NT+R):
    - excitatory if only cation channels match the presynaptic transmitter;
    - inhibitory if only anion channels match;
    - "complex" if both;
    - otherwise unpredicted.
  - Results: 9,070 excitatory, 2,580 inhibitory and 3,413 complex synapses, about 73% predicted.
    - The text calls the E:I ratio "close to 4:1"; the stated counts give about 3.5:1.
    - Alternative predictions are given from transmitter only, from receptor only, and for other reconstructions.
- **Access:**
  - S1–S7 Data (XLSX): the NT+R, NT-only and R-only predictions, subnetworks, alternative reconstructions, an RNA-seq-excluded variant, and the data sources.
  - S1–S6 Tables (DOCX), including channel type per receptor gene and manual curation edits.
  - Downloadable without request: yes.
  - Web tool: EleganSign.linkgroup.hu; its content could not be read.
- **Licence:** article "© 2020 Fenyves et al.", open access under the Creative Commons Attribution License (paper). SI: not separately stated. Tool: not stated.
- **Size and format:** XLSX, DOCX and PDF; sizes not shown.
- **Model mapping:** a direct per-connection sign prior (excitatory, inhibitory, mixed or unknown), plus a receptor gene → channel-type table that can be re-run on newer expression data (E1, E2, E6, E10).
- **Known quality issues:**
  - Stated in the paper:
    - Cation = excitatory and anion = inhibitory ignores reversal potentials.
    - Metabotropic, monoamine and peptide transmission are excluded.
    - Uncharacterised LGICs remain.
    - Predictions rest on subunit expression, not on functional receptors.
  - Later deorphanisations add ACh-gated anion channels (E10, paper), so the rule tables are dated. The transmitter identities pre-date Wang 2024 (E6).
- **Table row:** `| Synaptic polarity (Fenyves 2020) | 297 neurons, 3,638 chemical connections, 20,589 synapses | yes | n/a | n/a | n/a (WormWiring adult herm + expression) | article CC BY | not shown | XLSX, DOCX | PLOS SI (open) |`

### E9. Metabotropic ACh/Glu/GABA receptor map in NeuroPAL: Yemini et al., 2021, *Cell* 184(1):272–288.e11 (citation from a search snippet)

- **Verification:**
  - Content: paper (PMC10494711).
  - Citation: search snippet (second-hand).
  - Licence: Europe PMC metadata.
  - URLs opened: <https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.1016/j.cell.2020.12.012&format=json&resultType=core> ;
    <https://pmc.ncbi.nlm.nih.gov/articles/PMC10494711/>.
- **Measured (paper):**
  - Receptors: *gar-1*/*-2*/*-3*, *mgl-1*/*-2*/*-3* and *gbb-1*/*-2*.
  - Reporters: mostly fosmid GFP; *gar-3* and *mgl-2* used 5′ cytoplasmic transcriptional reporters.
  - Animals: L4 and young-adult hermaphrodites; identity by NeuroPAL.
  - 97% of neurons express at least one metabotropic receptor: GAR 70%, MGL 54%, GBB 89%.
  - The neuron × receptor matrix is Table S4.
- **Access:** PMC/Cell supplements (Table S4 file not opened). Imaging and calcium data: Zenodo 10.5281/zenodo.3906530, cited by the paper and not opened.
- **Licence:** not verified; none is listed in the Europe PMC record.
- **Size and format:** not verified.
- **Model mapping:** placement of the slow (G-protein-coupled) ACh, Glu and GABA receptors per neuron, a second, slower synaptic component whose sign follows the receptor's G-protein class (from receptor biology, not this paper).
- **Known quality issues (paper):** fosmid and promoter reporters, not endogenous alleles; *gar-3* and *mgl-2* were too dim or had no fosmid.
- **Table row:** `| Metabotropic receptor map (Yemini 2021, NeuroPAL) | all neurons × 8 receptor genes; L4/YA herm | yes (NeuroPAL) | n/a | n/a | not stated in what was read | not verified | not verified | Table S4 (format not verified) | PMC10494711 supplements |`

### E10. ACh-gated anion channels and connectome polarity: Hardege, Morud … Schafer, 2023, *J Neurosci* 43(7):1111–1124

- **Verification:**
  - Content: paper (Europe PMC full-text XML, PMC9962794).
  - It was found through a search snippet.
  - URL opened: <https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9962794/fullTextXML>.
- **Measured (paper):**
  - Deorphanised in *Xenopus* oocytes by TEVC:
    - LGC-40, -57 and -58: choline- and ACh-gated, anion-selective.
    - LGC-49: ACh-gated, anion-selective (EC50 about 19 µM).
    - LGC-46: anion-selective.
    - LGC-39: polymodal (ACh, octopamine, tyramine), anion-selective.
  - Still orphans: LGC-42, -44, -45, -47, -48, ACC-4 and LGC-32/33/34. LGC-43 and LGC-45 are predicted to be cation-selective but untested.
  - Expression: transcriptional reporters with NeuroPAL, CRISPR GFP knock-ins for four channels, and a comparison with CeNGEN. CeNGEN was used to infer polarity across the connectome.
- **Access:** Extended Data Table 5-1 (docx: ligand and ion selectivity of the LGICs) and Table 4-3 (xlsx: reporter vs CeNGEN overlap). Analysis scripts, including a connectome-polarity repository, are on GitHub; repository names were not verified. TEVC data on request.
- **Licence:** CC BY 4.0 (paper).
- **Size and format:** docx and xlsx; sizes not shown.
- **Model mapping:** ACh synapses onto neurons expressing these channels can be inhibitory, which updates the channel-type tables behind a sign rule (E8).
- **Known quality issues (paper):** heterologous expression; many LGCs remain orphans; two cation predictions are untested.
- **Table row:** `| ACh-gated anion LGCs (Hardege 2023) | ~16 LGC genes; reporter expression for subset | yes (NeuroPAL) | n/a | n/a | n/a | CC BY 4.0 | not shown | DOCX, XLSX | J Neurosci / PMC9962794 extended data |`

______________________________________________________________________

### W1. Original wiring diagram: White, Southgate, Thomson & Brenner, 1986, *Phil Trans R Soc Lond B* 314(1165):1–340

- **Verification:**
  - Citation and abstract: Europe PMC metadata.
  - Animal series: second-hand (NemaNode host page and the OpenWorm ConnectomeToolbox page).
  - The publisher page (403) and WormAtlas (TLS errors) were not reached.
  - URLs opened: <https://royalsocietypublishing.org/doi/10.1098/rstb.1986.0056> (403) ;
    <https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.1098/rstb.1986.0056&format=json&resultType=core> ;
    <https://openworm.org/ConnectomeToolbox/White_1986/> ; <https://nemanode.org/> ;
    <https://www.wormatlas.org/neuronalwiring.html> (certificate error) ; <https://wormwiring.org/>.
- **Measured:**
  - Animals: adult hermaphrodite serial-section EM; 302 neurons in 118 classes (abstract). Series named on the hosts: N2U (adult brain), JSH (L4 brain) and JSE (adult tail) (NemaNode; ConnectomeToolbox, second-hand).
  - Counts: about 5,000 chemical synapses, 2,000 NMJs and 600 gap junctions (abstract).
  - Chemical and gap junctions both scored.
  - The ConnectomeToolbox reads JSH and N2U from "neurodata.txt in Durbin thesis" (second-hand).
- **Access:**
  - The article is not open access (Europe PMC).
  - Data:
    - WormAtlas "Mind of the Worm" and `NeuronConnect.xls`, as linked from wormwiring.org; not opened.
    - NemaNode viewer.
    - ConnectomeToolbox.
    - Re-annotated within Cook 2019 (W3).
- **Licence:** not stated on any page reached.
- **Size and format:** not verified.
- **Model mapping:** chemical and gap edge existence and counts for the head (N2U, JSH) as a second and third animal beside the later sets; the historical base of Varshney and Cook.
- **Known quality issues:**
  - A composite of several animals plus an L4 (second-hand).
  - Reliability issues listed on the ConnectomeToolbox page: human error tracing long bundles, processes near the sectioning plane, and dirt or lost sections (second-hand).
  - Superseded for most uses by W2 and W3.
- **Table row:** `| White 1986 (N2U, JSH, JSE series) | 302 neurons (abstract); head/tail series | yes | n/a | n/a | composite; N2U adult, JSH L4, JSE adult tail (second-hand) | not stated | not verified | xls (WormAtlas), viewer | WormAtlas (not reached), NemaNode, cect |`

### W2. Updated hermaphrodite wiring: Varshney et al., 2011, *PLoS Comput Biol* e1001066 (volume/issue not verified)

- **Verification:**
  - Content: paper (PLOS page, two passes).
  - Repo: host (GitHub).
  - Counts on the ConnectomeToolbox page: second-hand.
  - URLs opened: <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1001066> ;
    <https://www.mit.edu/~lrv/www/elegans/> (404) ; <https://mit.edu/lrv/www/elegans/> (302 redirect) ;
    <https://web.mit.edu/lrv/www/elegans/> (404) ; <https://github.com/lrvarshney/elegans> ;
    <https://openworm.org/ConnectomeToolbox/Varshney_2011/>.
- **Measured:**
  - Sources (paper): White 1986 reanalysed, with additions from Hall & Russell, Durbin and unpublished Hobert/Hall data, plus new reconstruction. Partial imaging combined "three worms", with the posterior midbody from the male N2Y.
  - Networks: a directed chemical network and an undirected gap network (paper). The paper's neuron and synapse counts were images and could not be read.
  - Polyadic synapses: "We count all polyadic synaptic connections", and send_joint synapses were "treated … no differently than other synapses" (paper).
  - CANL/R make no synapses (paper).
  - ConnectomeToolbox notes (second-hand): wiring data for 64 neurons had gaps or were missing (61 ventral-cord motor neurons); the synapse split is about 49% chemical sends, 31% receives and 20% electrical.
- **Access:**
  - The paper's data URL (<http://mit.edu/lrv/www/elegans/>) is dead: it redirects to web.mit.edu, which returns 404.
  - GitHub `lrvarshney/elegans` (host): `ConnOrdered_040903.mat`, `NeuronTypeOrdered_040903.mat` and `A_sendjoint.mat`, plus 43 MATLAB files.
  - Also WormAtlas `NeuronConnect.xls` (not reached) and NemaNode / ConnectomeToolbox.
- **Licence:**
  - Article: "© 2011 Varshney et al.", Creative Commons Attribution License (paper).
  - GitHub repo: no licence label; the README asks for a citation (host).
- **Size and format:** `.mat` (sizes not shown) and xls (not verified).
- **Model mapping:** whole-animal chemical and gap edge lists with synapse counts, the most-used weight source in connectome-constrained models. Its sends/receives/send_joint codes let polyadic synapses be split or kept whole.
- **Known quality issues (paper):**
  - "Partially complete".
  - About a fifth of chemical synapses may be missing from sparsely sampled cords.
  - "Many gap junctions are likely missing due to the difficulty in identifying them."
  - The postsynaptic partner is ambiguous when a terminal contacts two adjacent processes.
- **Table row:** `| Varshney 2011 herm wiring | whole-animal herm (counts not readable on page); chemical + gap + NMJ | yes | n/a | n/a | composite of ≥3 worms incl. male N2Y midbody | article CC BY; repo none | not shown | .mat (GitHub), xls (WormAtlas) | GitHub lrvarshney/elegans (open); paper's URL dead |`

### W3. Whole-animal connectomes of both sexes: Cook et al., 2019, *Nature* 571:63–71 (pages from the ConnectomeToolbox page)

- **Verification:**
  - The paper's full text was not read: PMC showed a browser-check page and nature.com redirected to a login.
  - Files: host (wormwiring.org).
  - Weight semantics and node/edge counts: second-hand (the Netzschleuder mirror and the ConnectomeToolbox).
  - URLs opened: <https://wormwiring.org/> ; <https://wormwiring.org/pages/emmonslab.html> ;
    <https://wormwiring.org/pages/adjacency.html> ; <https://www.wormwiring.org/pages/software.html> ;
    <https://networks.skewed.de/net/celegans_2019> ; <https://openworm.org/ConnectomeToolbox/Cook_2019/> ;
    <https://pmc.ncbi.nlm.nih.gov/articles/PMC6889226/> (browser check, twice) ; <https://europepmc.org/articles/PMC6889226> (403) ;
    <https://www.nature.com/articles/s41586-019-1352-7> (login redirect) ;
    <https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.1038/s41586-019-1352-7&format=json&resultType=core>.
- **Measured:**
  - Animals: adult hermaphrodite and adult male (Europe PMC abstract). The reconstructions draw on new and previously published micrographs, with new male-head EM (abstract; host). The host describes them as a reanalysis of White 1986 (hermaphrodite) and Sulston 1980 (male tail) plus new male-head micrographs.
  - Content (host): "Complete chemical and gap junction connectivity between all neurons", and between neurons and end organs (muscle, gonad, gut, hypodermis). Also muscle–muscle gap junctions, skeleton diagrams, synapse lists, and whether each synapse is monadic or polyadic.
  - Weights: "connection weights based on synapse number and synapse size" (host). In the full matrices, the `connectivity` edge value is "the number of EM serial sections in which a connection is observed"; the `_synapse` matrices count synapses and include only EM-scored connections, no extrapolated ones (second-hand, Netzschleuder).
  - Nodes: 460 hermaphrodite (302 neurons, 132 muscles, 26 other end organs) and 579 male (385 neurons, 155 muscles, 39 end organs) (second-hand, ConnectomeToolbox).
  - Edges in Netzschleuder's corrected networks (second-hand): hermaphrodite chemical 454 nodes / 4,879 edges; hermaphrodite gap 469 / 1,450; male chemical 575 / 5,306; male gap 586 / 1,758.
  - The paper's handling of polyadic synapses (split or not) was not verified.
- **Access:**
  - <https://wormwiring.org/pages/adjacency.html> (host), downloadable without request. Nine .xlsx files:
    - `SI 5 Connectome adjacency matrices, corrected July 2020.xlsx`;
    - the original `SI 5 Connectome adjacency matrices.xlsx`;
    - `SI 2 Synapse adjacency matrices.xlsx`;
    - `SI 3 Synapse lists.xlsx`;
    - `SI 4 Cell lists.xlsx`;
    - `SI 6 Cell class lists.xlsx`;
    - `SI 7 Cell class connectome adjacency matrices, corrected July 2020.xlsx`;
    - the original `SI 7 Cell class adjacency matrices.xlsx`;
    - `Adult and L4 nerve ring neighbors.xlsx` (contact adjacency, Brittin 2018).
  - Release: WormWiring v1.1, 2020-04-18 (host).
  - MATLAB-ready SI 5: "Premaratne MATLAB-ready files .zip" (host, software page).
  - Annotation software: Elegance, <https://github.com/Emmonslab/Elegance> (host).
  - Mirror: Netzschleuder `celegans_2019` (12 networks, added 2020-08-11).
- **Licence:**
  - wormwiring.org: not stated; the footer reads "Emmons Lab Copyright (c) 2020" (host).
  - Netzschleuder: none for the data (an AGPLv3 badge refers to its software).
  - Article: not verified.
- **Size and format:** .xlsx; sizes not listed (host).
- **Model mapping:**
  - The primary whole-animal edge-weight source for both chemical (directed) and gap (undirected) edges, including NMJ and end-organ edges.
  - Section-count weights reflect synapse number and size; synapse-count matrices are the alternative.
  - The male matrices allow sex comparison.
  - The corrected July 2020 matrices supersede the originals.
- **Known quality issues:**
  - Connectomes are assembled from several animals, with some connections extrapolated where no EM existed (second-hand, Netzschleuder).
  - Several corrections were made after publication: DVB/PVT, PVCL/PVCR, reassigned classes, and PDB reclassified as AS12 (second-hand, ConnectomeToolbox).
  - Section-count weights mix synapse number and size (second-hand).
  - Gap-junction identification caveats as for W2 (paper W2).
- **Table row:** `| Cook 2019 both sexes (WormWiring SI 2–7) | whole animal; herm 302 neurons + 158 muscles/end organs, male 385 + 194 (second-hand counts) | yes | n/a | n/a | composite herm (White series) + male (incl. new head EM) | not stated (Emmons Lab ©2020) | not listed | XLSX (+ MATLAB zip) | wormwiring.org/pages/adjacency.html (open) |`

### W4. Connectomes across development: Witvliet et al., 2021, *Nature* (volume/pages not verified)

- **Verification:**
  - Content: paper (PMC8756380, read before the PMC browser check began).
  - Files: host (wormwiring.org; NemaNode).
  - BossDB content: not readable.
  - URLs opened: <https://wormwiring.org/pages/witvliet.html> ; <https://nemanode.org/> ;
    <https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.1038/s41586-021-03778-8&format=json&resultType=core> ;
    <https://pmc.ncbi.nlm.nih.gov/articles/PMC8756380/> ; <https://bossdb.org/project/witvliet2020> (title only).
- **Measured (paper):**
  - Animals: eight isogenic N2 hermaphrodites, at about 0, 5, 8, 16, 23 and 27 h and two at about 45 h (young adults). The host files are labelled four L1, one L2, one L3 and two adults; the paper's Methods tally differs.
  - Region: the "brain" (nerve ring and ventral ganglion), traced in CATMAID with volumetric segmentation of neuron and muscle processes. Glia were not segmented.
  - Chemical synapses: consensus of three annotators, kept if at least two agreed.
  - Gap junctions: partially annotated, not consensus-scored, and excluded from analyses (paper). NemaNode says they were "not thoroughly annotated" (host).
  - Each presynaptic active zone was volumetrically reconstructed.
  - Polyadic weights are split: active-zone volume × the proportion of simulated transmitter particles reaching each partner.
  - Contact area was measured by dilation (140–280 nm).
  - Variability: about 43% of cell–cell connections (16% of synapses) are not conserved between isogenic animals. In adults, variable connections are about 50% of connections; stable connections carry about 72% of synapses.
- **Access:**
  - wormwiring.org: `witvliet_2020_1 L1.xlsx` … `witvliet_2020_8 adult.xlsx` (8 files; host).
  - NemaNode viewer and download (host).
  - EM and volumes at bossdb.org/project/witvliet2020, plus the paper's Supplementary Tables (paper).
  - Code: <https://github.com/dwitvliet/nature2021> (paper; not opened).
  - Downloadable without request: yes.
- **Licence:**
  - wormwiring.org: not stated ("Copyright (c) 2020", Emmons Lab footer).
  - NemaNode: not stated; it asks for citation.
  - Article: not verified (Europe PMC "isOpenAccess N").
- **Size and format:** .xlsx, sizes not listed.
- **Model mapping:**
  - Synapse-size weights with split polyadic partners are the best-founded chemical edge weights for the head.
  - The two adults give a direct estimate of inter-animal edge variability, so edge priors can carry a variance, and stable edges can be fixed while variable ones get looser priors.
  - Contact area gives a structural null.
  - Not usable for gap strengths.
- **Known quality issues (paper):** gap junctions excluded; one animal per time point except adults; head only (no ventral cord or tail).
- **Table row:** `| Witvliet 2021 development series | nerve ring + ventral ganglion (head) | yes | n/a | n/a | 8 herm (L1×4, L2, L3, adult×2 per host labels) | not stated | not listed | XLSX; BossDB volumes | wormwiring.org, nemanode.org, bossdb (open) |`

### W5. Volumetric contact maps: Brittin, Cook, Hall, Emmons & Cohen, 2021, *Nature* 591:105–110 (pages from a search snippet)

- **Verification:**
  - Files, licence and sizes: host (Zenodo API and README).
  - Abstract: paper (White Rose author manuscript page).
  - The full text was not read: PMC browser check, Europe PMC full text HTTP 500, and the WormAtlas PDF certificate had expired.
  - The neighbour definition is second-hand (search snippet).
  - URLs opened: <https://zenodo.org/api/records/4383277> ; <https://zenodo.org/api/records?q=conceptrecid:4383276&allversions=true> ;
    <https://zenodo.org/records/4763083/files/README.txt> ; <https://eprints.whiterose.ac.uk/172908/> ;
    <https://pmc.ncbi.nlm.nih.gov/articles/PMC11648602/> (browser check) ; <https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11648602/fullTextXML> (500) ;
    <https://ftp.wormatlas.org/papers/Brittin> et al 2021.pdf (certificate expired) ;
    <https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.1038/s41586-021-03284-x&format=json&resultType=core>.
- **Measured:**
  - Animals and region: volumetric reconstructions of the nerve ring of two animals (abstract), from White et al.'s EM series: `n2u_*` adult and `jsh_*` L4 (host file descriptions).
  - Data: SQL databases of membrane contacts, which are the primary analysis input, plus contact adjacency extracted from TrakEM2 and GraphML reference graphs (host).
  - Neighbour definition: "immediate neighbours if the membranes … are physically adjacent in at least one EM" (second-hand).
  - Integrated with the synaptic and gap-junction connectomes, the model finds "a precisely wired core circuit" within more variable connectivity (abstract).
- **Access:** Zenodo concept DOI 10.5281/zenodo.4383276; v1.0.0 is 10.5281/zenodo.4383277 and v1.1.0 is record 4763083 (host). Downloadable without request: yes. Also wormwiring `Adult and L4 nerve ring neighbors.xlsx` (W3).
- **Licence:** Zenodo metadata "cc-by-4.0" (host). The README.txt in the same record carries an MIT-style permission notice, © 2020 C. Brittin, S. Emmons, N. Cohen, provided "AS IS" (host). The two statements differ.
- **Size and format (host, bytes):**
  - Analysis files:
    - `adjacency.tar.bz2`: 879,737.
    - `databases.tar.bz2`: 18,454,962.
    - `reference_graphs.tar.bz2`: 36,502.
    - `brainmap.tar.bz2`: 1,244,468.
    - `population_spatial_models.tar.bz2`: 3,836,352.
  - Segmentations: 128,189,805 (JSH) and 112,734,522 (N2U).
  - Process structures: 69,559,108 (JSH) and 65,297,318 (N2U).
  - Image mipmaps: 7,154,101,220 (JSH) and about 9.65 GB for N2U in ten parts.
  - v1.1.0 adds `adult_volumes.tar.bz2` (61,616,473) and `l4_volumes.tar.bz2` (43,309,004), as .obj/.mtl meshes.
  - Formats: SQL, GraphML, NumPy, TrakEM2, Cytoscape.
- **Model mapping:**
  - Contact area and adjacency give a physical prior on which pairs can form chemical or gap synapses. That prior can bound edges absent from EM synapse lists and set contact-matched null wirings.
  - The reproducible "core" serves as a prior on edge reliability.
- **Known quality issues:** two animals at different stages (L4 and adult), nerve ring only (host and abstract). Further caveats were not verified because the full text was not read.
- **Table row:** `| Brittin 2021 volumetric contacts | nerve ring | yes | n/a | n/a | 2 (adult N2U, L4 JSH) | Zenodo cc-by-4.0; README MIT-style (conflict) | analysis files ~25 MB; full record ~17 GB | SQL, GraphML, NumPy, TrakEM2, OBJ | Zenodo 10.5281/zenodo.4383276 (open) |`

### W6. *Pristionchus pacificus* comparative connectome: Cook et al., 2025, *Science* 389(6759):eadx2143 (journal verified)

- **Verification:**
  - Citation and licence: Europe PMC metadata.
  - Content: paper (Europe PMC full-text XML, PMC12330220).
  - Preprint: bioRxiv landing page.
  - URLs opened: <https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.1126/science.adx2143&format=json&resultType=core> ;
    <https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12330220/fullTextXML> ; <https://www.biorxiv.org/content/10.1101/2024.06.13.598904v2> ;
    <https://pubmed.ncbi.nlm.nih.gov/40743352/> (cookie page) ; <https://discovery.ucl.ac.uk/id/eprint/10213222> (403).
- **Measured (paper):**
  - Animals: two adult hermaphrodite *P. pacificus* (PS312, eurystomatous) heads, from the nose tip to the retrovesicular ganglion; comparisons restricted to the nerve ring.
  - Comparison: against 10 published *C. elegans* samples, L1 to adult (Cook 2019, Witvliet 2021, and an adjacency dataset).
  - Weight: the number of consecutive serial sections showing a synaptic specialisation.
  - Gap junctions: annotated but excluded "because of ultrastructural ambiguity".
- **Access:**
  - Data: "All neuronal adjacency and connectivity data … are included in supplementary materials."
  - Code: <https://github.com/stevenjcook/cook_et_al_2025_pristionchus>. The Methods name a different repository, `cook_et_al_2024_mipristionchus`.
  - Raw EM: <https://bossdb.org/project/bumbarger2013>.
  - Renderings: promised on Zenodo "upon publication", with no URL given.
  - Older *P. pacificus* pharynx (Bumbarger 2013) and amphid (Hong 2019) tables are on wormwiring.org (host).
- **Licence:** article "Creative Commons Attribution 4.0 International License" (paper); preprint "CC-BY 4.0 International" (bioRxiv); data in SI under the article's licence (not separately stated).
- **Size and format:** not verified.
- **Model mapping:** not a *C. elegans* parameter source. Its conserved nematode "core connectome" is a prior on which *C. elegans* head edges are evolutionarily robust and so deserve tighter priors. The 10-sample *C. elegans* comparison is a cross-animal variability reference.
- **Known quality issues (paper):** two animals; gap junctions excluded; head and nerve ring only; the code repository names do not match.
- **Table row:** `| P. pacificus connectome (Cook 2025) | head to retrovesicular ganglion; nerve-ring comparison | yes (homologues) | n/a | n/a | 2 adult herm P. pacificus (+10 C. elegans, published) | CC BY 4.0 | not verified | SI tables; BossDB EM | Science SI, GitHub, bossdb |`

### W7. Dauer connectome: Yim, Choe, Bae … Hall, Kim & Lee, 2024, *Nat Commun* 15:1546

- **Verification:**
  - Content and data statement: paper (Europe PMC full-text XML, PMC10899629, two passes).
  - Toolbox listing: second-hand.
  - URLs opened: <https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.1038/s41467-024-45943-3&format=json&resultType=core> ;
    <https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10899629/fullTextXML> ; <https://openworm.org/ConnectomeToolbox/>.
- **Measured (paper):**
  - Animal and region: one dauer hermaphrodite nerve ring, in 364 serial TEM sections at 50 nm.
  - Synapse detection: a CNN detects presynaptic active zones, and postsynaptic partners are assigned by simulated transmitter diffusion.
  - Weight: the sum of active-zone sizes.
  - Counts: 2,813 active zones, 6,371 synapses and 2,200 connections (1,995 neuronal).
  - Polyadic: 81% of active zones, with 2.26 partners on average.
  - Gap junctions "are not included"; the gap-junction data are "available upon request".
  - Contact area was also measured.
- **Access:**
  - EM and segmentation: <https://bossdb.org/project/yim_choe_bae2023> (10.60533/BOSS-2023-RTPH).
  - Connectivity and contact data: Supplementary Data 1–6 (`41467_2024_45943_MOESM4–9_ESM`, xlsx except one PDF) and Source Data (xlsx).
  - Code: <https://github.com/jabae/Yim-Choe-Bae-et_al-2024> (10.5281/zenodo.10516067) and <https://github.com/jabae/Cnapse> (10.5281/zenodo.10516075).
  - Included in the ConnectomeToolbox as "Yim et al. 2024", "(non norm.)" and "(contact.)" (host listing).
- **Licence:** "Creative Commons Attribution 4.0 International License" (paper).
- **Size and format:** XLSX supplements; sizes not shown.
- **Model mapping:** not an adult parameter source. A bound on stage-dependent rewiring, and a second split-polyadic weight measure that can be compared with Witvliet's.
- **Known quality issues (paper):** a single animal; no gap junctions; automated synapse detection.
- **Table row:** `| Dauer connectome (Yim 2024) | dauer nerve ring | yes | n/a | n/a | 1 dauer herm | CC BY 4.0 | not shown | XLSX; BossDB EM | Nat Commun SI, bossdb, GitHub (open) |`

### W8. Dauer vs adult nociceptive gap junctions (preprint): Choe, Hall, Nguyen, Choi, Bae & Lee, bioRxiv 2026 (v1 2026-05-29, v2 2026-11-11)

- **Verification:**
  - Sciety page only: a third-party preprint aggregator (second-hand).
  - The bioRxiv landing page did not render, the bioRxiv API returned "no posts found", and the PDF exceeded the fetch limit.
  - URLs opened: <https://sciety.org/articles/activity/10.64898/2026.05.28.728331> ;
    <https://www.biorxiv.org/content/10.1101/2026.05.28.728331v1> (no article content) ;
    <https://api.biorxiv.org/details/biorxiv/10.1101/2026.05.28.728331> ;
    <https://www.biorxiv.org/content/biorxiv/early/2026/05/29/2026.05.28.728331.full.pdf> (too large).
- **Measured (second-hand):**
  - A "substantial expansion of gap junctions within the dauer nociceptive circuit" relative to adults.
  - Adding synthetic dauer-specific gap junctions to adults reproduced dauer-like activity and shortened avoidance.
  - Animal counts and the annotation method were not stated on the page.
- **Access:** DOI 10.64898/2026.05.28.728331; no data links on the Sciety page.
- **Licence:** not verified.
- **Size and format:** not verified.
- **Model mapping:** one of the few EM-based gap-junction comparisons in the head circuit. It may bound adult gap strengths in the nociceptive circuit once the data are found (neurons involved not verified).
- **Known quality issues:** a preprint; the data were not located.
- **Table row:** `| Dauer/adult gap-junction connectomics (Choe 2026, preprint) | nociceptive circuit (head) | not verified | n/a | n/a | not verified | not verified | not verified | not verified | DOI 10.64898/2026.05.28.728331 |`

### W9. Aggregator: OpenWorm ConnectomeToolbox (`cect`): OpenWorm, ongoing (docs generated with cect v0.3.5)

- **Verification:**
  - Repo and docs: host.
  - The per-dataset notes in its pages are second-hand descriptions of the papers.
  - URLs opened: <https://github.com/openworm/ConnectomeToolbox> ; <https://openworm.org/ConnectomeToolbox/> ;
    <https://openworm.org/ConnectomeToolbox/Cook_2019/> ; <https://openworm.org/ConnectomeToolbox/White_1986/> ;
    <https://openworm.org/ConnectomeToolbox/Varshney_2011/> ; <https://openworm.org/ConnectomeToolbox/Beets_2023/>.
- **Measured:** readers for these datasets (host):
  - White 1986 (JSH, N2U, whole) and Varshney 2011.
  - Bentley 2016 (monoamine, peptide).
  - Cook 2019 (herm, male) and Cook 2020.
  - Brittin 2021.
  - Witvliet 2021 (sets 1–8).
  - WormNeuroAtlas anatomy and Randi 2023.
  - Ripoll-Sánchez 2023 (short, mid, long).
  - Yim 2024 (three variants).
  - Wang 2024 (herm, male).
  - Models: Gleeson 2018 and Olivares 2021.
  - No 2025–2026 dataset is listed.
- **Access:** GitHub; PyPI package `cect`.
- **Licence:** "MIT license" (GitHub label; host). The docs site states none. The licences of the underlying data are not restated.
- **Size and format:** Python package; sizes not shown.
- **Model mapping:** one harmonised neuron naming across wiring, modulator and transmitter layers, which is the practical route for joining E3, E5, E6 and W1–W7 into one parameter table.
- **Known quality issues:** third-party conversions. The descriptions on its pages are summaries, not source documentation.
- **Table row:** `| OpenWorm ConnectomeToolbox (cect) | readers for ~15 datasets (wiring, modulator, transmitter) | yes (harmonised names) | n/a | n/a | n/a | code MIT; data per source | not shown | Python package | github.com/openworm/ConnectomeToolbox, PyPI cect (open) |`

______________________________________________________________________

### Not found or not verified in this session

- **CeNGEN:**
  - No licence for any data on cengen.org.
  - No downloadable L1, adult or male matrices: apps only were found.
  - The sizes of the four threshold CSVs are not listed.
  - Which threshold `cengen_sc_2` encodes is not stated.
  - The licence of the 2021 Cell article was not verified.
- **Beets 2023:**
  - Whether the EC50 table (Table S6) is published as a spreadsheet: the publisher page returned 403, and in the PMC copy it may sit inside a 7 MB PDF.
  - worm.peptide-gpcr.org was unreachable (403).
  - The article's licence is stated two ways on PMC.
- **NemaMod:** no licence stated; no download format documented beyond "image or data matrix".
- **Bentley 2016:** S1 Dataset size not shown.
- **Bhattacharya 2019 and Yemini 2021:** licences and supplementary table formats not verified.
- **EleganSign (Fenyves 2020):** the web tool's content could not be read.
- **Cook 2019:**
  - The paper's methods were not read (PMC browser check, Nature login).
  - Polyadic handling and the animal list come only from host and third-party descriptions; the polyadic split is not verified.
  - No licence on wormwiring.org.
- **White 1986 and Varshney 2011:**
  - WormAtlas files (`NeuronConnect.xls`, Mind of the Worm) were not reached because of TLS certificate errors.
  - Varshney's data URL is dead (404).
  - Varshney's counts were images on the PLOS page.
- **Witvliet 2021:** article licence and BossDB content not verified.
- **Brittin 2021:** full text not read; the Zenodo licence fields conflict with the README.
- **Cook 2025:** supplementary data sizes not verified; the Zenodo renderings had no URL.
- **Choe et al. 2026:** a preprint whose data, animal count and licence were not verified.
- **Newer adult *C. elegans* hermaphrodite EM volumes (2024–2026), multi-adult variability sets, and a separate male-head release beyond Cook 2019:** none found. The search budget was used up before a targeted search, so this is not proof that none exist.
- **A 2024–2026 receptor atlas from endogenous reporter alleles:** none found. The closest are Wang 2024 (transmitter pathways, not receptors), Yemini 2021 (metabotropic receptors, fosmid reporters) and Hardege 2023 (LGCs).
