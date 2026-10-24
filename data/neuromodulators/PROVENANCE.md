# Neuromodulator circuit

`roaming_dwelling_circuit.json` records the serotonin and PDF roaming/dwelling circuit that the
connectome's modulator field models. Every entry was checked on 2026-10-11 against the full text and
figure legends (not the figure images) of its source:

- Flavell, S. W. et al. (2013). Serotonin and the neuropeptide PDF initiate and extend opposing
  behavioral states in *C. elegans*. *Cell* 154:1023–1035. PMC3942133.
- Ji, N. et al. (2021). A neural circuit for flexible control of persistent behavioral states.
  *eLife* 10:e62889. PMC8660023.
- Dag, U. et al. (2023). Dissecting the functional organization of the *C. elegans* serotonergic
  system at whole-brain scale. *Cell*. PMC10484565.
- Rhoades, J. L. et al. (2019). ASICs mediate food responses in an enteric serotonergic neuron that
  controls foraging behaviors. *Cell* 176:85–97. PMC6526957.
- Ben Arous, J., Laffont, S. & Chatenay, D. (2009). Molecular and sensory basis of a food related
  two-state behavior in *C. elegans*. *PLoS ONE* 4:e7584. PMC2762077.

**Exclusions and conflicts are recorded in the file:**

- **RIB is excluded as a PDFR-1 site.** Ji 2021 cites Rhoades 2019 for it, and Rhoades 2019 does not
  contain it.
- **AIY's role as a MOD-1 site is contested.** Dag 2023's acute-slowing assay does not support it;
  Flavell 2013's roaming/dwelling assay does. That is a different behaviour.
- **HSN** is required for serotonin's effect too (Flavell 2013), but is not modelled as a release
  site. NSM's food-ingestion drive is the verified mechanism.

**PDF's inhibition of NSM** (Ji 2021 Fig 4E–G) is recorded as a functional link: PDF signalling, acting
through the PDFR-1 neurons, keeps NSM inactive during roaming. In Emmons 2024, NSM is wired only to
pharyngeal neurons, so the model carries this link as an inhibitory current on NSM rather than
through the wiring.

## `layer.json`: the data-placed layer

The connectome's neuromodulator layer placed from two CC BY sources, read by
`quantumnematode.connectome.neuromodulator_layer` when a brain sets `modulator_layer: data`. Derived by
`scripts/analysis/extract_neuromodulator_layer.py` (`fetch`, `synonyms`, `build`); the source files
are downloaded into gitignored `build/neuromodulators/` and not vendored.

- **Bentley, B. et al. (2016).** The multilayer connectome of *Caenorhabditis elegans*. *PLoS Comput
  Biol* 12:e1005283. S1 Dataset (`journal.pcbi.1005283.s004`), 69,151 bytes, SHA-256
  `aeda5f49d966106943b574283f456142cb3bb0ccbf3b93d87c647e13db273ebb`, downloaded 2026-10-24 by script.
  **Licence:** CC BY 4.0, the article's (PLOS publishes supporting information under it).
- **Ripoll-Sánchez, L. et al. (2023).** The neuropeptidergic connectome of *C. elegans*. *Neuron*
  111:3570–3589. figshare collection doi:10.6084/m9.figshare.c.6895870: the 92 long-range individual
  networks (item 24187119), the ligand–receptor list at EC50 ≤ 500 nM (item 24517180) and
  `neuron_information.csv` (item 24187104), 94 files, each checked against figshare's published MD5
  and recorded with its SHA-256 in the file's `fetched` block, downloaded 2026-10-24 by script.
  **Licence:** CC BY 4.0, item by item, as is the article.
- **Not used:** the authors' corrected matrices of 2024-02-02 (updated single-cell data for dmsr-5 and
  npr-34) and their threshold-4 expression table exist only in their GitHub repository, under the MIT
  licence, which this project's rule (CC0 or CC BY only) excludes. The layer therefore carries the
  2023 placement of dmsr-5 and npr-34, the two receptors the authors' README says the update concerns.

**What it holds.**

- **56 ligands:** 4 monoamines and 52 peptide genes. flp-24 has no producer in either source, so it
  carries no pool and its one couple never loads; 55 ligands have pools.
- **Monoamine producers** are Bentley's synthesis-marker rows, excluded rows dropped: serotonin from
  tph-1 (ADF, HSN, NSM), dopamine from cat-2 (ADE, CEP, PDE), octopamine from tbh-1 (RIC) and tyramine
  from tdc-1 (RIM). The reuptake markers are not producers: mod-5 would add AIM.
- **121 couples:** 17 monoamine couples from Bentley's receptor rows (each couple's ligand from the
  row's `monoamine` column; excluded rows and the provisional dop-5 and dop-6 dropped), and 104 peptide
  couples, the union of Ripoll-Sánchez's 92 and Bentley's 22 under WormBase names.
  - Each long-range network was confirmed to be exactly its producer set times its receptor set (all
    92, self-edges included), so both sets come from the CC BY networks without the MIT table.
  - Every producer and receptor cell carries the sources that place it.
  - **No common potency cut:** each couple records its EC50 and source. `within_500nM` marks the
    couples the `ec50_500nM` sensitivity keeps: those Ripoll-Sánchez lists, and Bentley's whose
    reported EC50 range ends at or below 500 nM. Eight are outside, pdf-1 → pdfr-1 (0.4–5 µM) among
    them.
  - **Signs:** 16 monoamine couples carry a fixed sign from `receptor_signs.json`; ser-5's coupling is
    inferred only from phylogeny, so it and every peptide couple (105 in all) are fitted.
- **One drive:** food ingestion onto NSM (Rhoades et al. 2019), kept from the hand-placed field.
- **`areas`:** each neuron's body areas from Ripoll-Sánchez's `neuron_information.csv`, kept for a
  later mid-range model.
- Neuron names: Ripoll-Sánchez's zero-padded motor names (DA01) read as the project's (DA1). Every
  name is present in the connectome.

## `gene_synonyms.json`

Every gene name either source uses (170), mapped to its WormBase public name through UniProt's
*C. elegans* entries (organism 6239), whose gene names WormBase supplies; WormBase's own site refuses
scripted queries. A name that is a gene's public name keeps it; a name that is only a synonym must be
listed by exactly one gene.

- **Renamed:** NLP-37 → pdf-2, NLP-38 → mihp-1, NLP-44 → capa-1. Bentley's pdf-2 and Ripoll-Sánchez's
  NLP-37 are therefore one ligand with one pool.
- **Unverified** (UniProt has no entry naming them; kept as written): NLP-42, NLP-54 and NLP-72
  (Ripoll-Sánchez ligands), npr-17 (Bentley's flp-24 receptor), and pdf-3 to pdf-12, which appear only
  in Bentley's peptide-expression table with no couple and never enter the layer.

## `receptor_signs.json`

Each monoamine receptor's coupling and the layer's sign for it, with a citation and a quoted sentence,
verified 2026-10-24 against the cited full texts or, where those are not open access, their
abstracts. The rule: Gαi/o-coupled and ligand-gated Cl⁻ channels inhibitory, Gαq- or Gαs-coupled
excitatory. It describes the receptor's effect on the cell carrying it, and assumes E_Cl below the
resting potential for the Cl⁻ channels.

- **Fixed:** ser-1, ser-3, ser-7, dop-1, dop-4, tyra-3 (+1); ser-4, mod-1, dop-2, dop-3, lgc-53, octr-1,
  ser-2, tyra-2, lgc-55 (−1); ser-6 (+1, its coupling Gαs in vivo but Gq-like in oocytes, excitatory
  either way).
- **Fitted:** ser-5, whose Gαs coupling rests only on phylogenetic clustering.
