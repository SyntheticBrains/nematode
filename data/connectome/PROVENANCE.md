# Connectome Data Provenance

This directory contains *C. elegans* connectome data files used by the
`quantumnematode.connectome` subpackage (Phase 6 Tranche 1 / L0).

## Files

### `cook_2019_si5_connectome_adjacency.xlsx`

- **Description**: Whole-animal connectome adjacency matrices for *C. elegans*
  hermaphrodite and male. Sheets are organised by (sex × connection-type):
  `hermaphrodite chemical`, `hermaphrodite gap jn`, `male chemical`,
  `male gap jn`. Cell values are EM-derived synapse / junction counts.
- **Original filename in upstream mirror**: `SI 5 Connectome adjacency matrices.xlsx`
- **Size**: 4,367,796 bytes (~4.4 MB)
- **SHA256**: `559989daa02cd9a76e9266537a6b80bfd47d338cb5d397d11296288629df364a`
- **Source URL**: <https://raw.githubusercontent.com/openworm/ConnectomeToolbox/master/cect/data/SI%205%20Connectome%20adjacency%20matrices.xlsx>
- **Mirror licence**: MIT (OpenWorm ConnectomeToolbox `LICENSE` file declares
  MIT, Copyright 2024 OpenWorm; note that the same repo's `setup.cfg` declares
  LGPLv3 — the `LICENSE` file is treated as legal source of truth per
  convention)
- **Retrieval date**: 2026-05-23
- **Accompanying paper**:
  - **Title**: Whole-animal connectomes of both *Caenorhabditis elegans* sexes
  - **Authors**: Steven J. Cook, Travis A. Jarrell, Christopher A. Brittin,
    Yi Wang, Adam E. Bloniarz, Maksim A. Yakovlev, Ken C. Q. Nguyen,
    Leo T.-H. Tang, Emily A. Bayer, Janet S. Duerr, Hannes E. Bülow,
    Oliver Hobert, David H. Hall, Scott W. Emmons
  - **Journal**: *Nature* 571, 63–71 (2019)
  - **DOI**: <https://doi.org/10.1038/s41586-019-1352-7>
- **Redistribution rationale**: Academic research re-use of *Nature*
  Supplementary Information is standard practice. We cite Cook et al. 2019 in
  papers / logbooks that consume this data. The vendored copy is sourced from
  OpenWorm cect's MIT-licensed mirror, which itself redistributes the *Nature*
  SI. Any consumer of our codebase should also cite Cook et al. 2019 when
  publishing derived results.

### `emmons_2024_s1_connectome_adjacency.xlsx`

- **Description**: the Cook et al. 2019 whole-animal adjacency matrices for both sexes, as
  republished under CC BY 4.0 by Cook et al.'s senior author: the S1 File ("Connectome Adjacency
  Matrices") of Emmons 2024. The sheets and their layout match the 2019 file above, except that the
  hermaphrodite gap-junction sheets are named `hermaphrodite gap jn symmetric` and
  `hermaphrodite gap jn asymmetric` rather than `herm gap jn …`. Cell values are EM serial-section
  counts; the legend notes that "the data are assembled from multiple animals and include connections
  added by extrapolation in gaps where no data were available". It records two revisions since 2019:
  - **Corrections, July 2020**: "The attempt is made here to remove all inconsistencies and errors
    in the published tables", in particular so that each gap-junction table agrees across its
    diagonal and with the asymmetric table.
  - **Addition, 2023**: gap junctions between BDU and the touch cells ALM and PLM, citing Jarrell et
    al. 2012 (*Science* 337:437) and Zhang et al. 2013 (*PLoS Genetics* 9:e1003618).
- **Against the 2019 file**, among the 302 neurons:
  - the chemical matrix is identical: 3,709 connections, 20,965 sections, 38 autapses, and all 956
    entries onto the 95 body wall muscles (5,515 sections, from 162 neurons);
  - the gap junctions differ only by the 2023 addition. ALML–BDUL and ALMR–BDUR are new at 23
    sections each, and BDUL–PLML and BDUR–PLMR rise from 23 to 37, so the loader returns 1,095 pairs
    and 5,864 sections against 1,093 and 5,790. The 2020 corrections touch three neuron cells whose
    two directions disagreed in 2019 (PVCR–VA9, 3 against 2; PDB→DD6, 2 with no mirror entry) and set
    each pair to the larger value, which the loader's fold already took.
- **Consumed by** `load_emmons_2024_hermaphrodite()`, `load_emmons_2024_neuromuscular()` and
  `scripts/export_wormlight.py`. Every existing experiment reads the 2019 file above, unchanged.
- **Original filename at the publisher**: `pbio.3002939.s001.xlsx`
- **Size**: 4,176,688 bytes (~4.2 MB)
- **SHA256**: `e866b43f19ba5c70b773c94efd06aff6d6b2887cd24eed4412da80c06986418d`
- **Source URL**: <https://journals.plos.org/plosbiology/article/file?type=supplementary&id=10.1371/journal.pbio.3002939.s001>
- **Licence**: CC BY 4.0, the article's licence, which covers its supporting information
  (<https://creativecommons.org/licenses/by/4.0/>)
- **Retrieval date**: 2026-09-25
- **Accompanying paper**:
  - **Title**: Comprehensive analysis of the *C. elegans* connectome reveals novel circuits and
    functions of previously unstudied neurons
  - **Author**: Scott W. Emmons
  - **Journal**: *PLoS Biology* 22(12): e3002939 (2024), published 2024-12-17
  - **DOI**: <https://doi.org/10.1371/journal.pbio.3002939> (PMID 39689061, PMCID PMC11651592)
- **Redistribution rationale**: CC BY 4.0 permits redistribution with attribution. The 2019 file
  states no licence of its own, so this is the licensed route to the same data, from the same lab.
  Any consumer should cite both Cook et al. 2019 and Emmons 2024.

### `elife-95402-supp2-v1.xlsx`

- **Description**: Supplementary File 2 of the *C. elegans* neurotransmitter atlas — a
  per-neuron table of CRISPR/Cas9 knock-in reporter expression for the transmitter-related
  genes (`eat-4`, `unc-17`, `unc-25`, `unc-47`, `cat-1`, `tph-1`, `cat-2`, `bas-1`, `tdc-1`,
  `tbh-1`, `mod-5`, `snf-3`, `oct-1`), the prior-report staining columns, and a curated
  `Neurotransmitter(s)` column giving each of the 302 hermaphrodite neurons a release identity.
  Consumed to populate the `neurotransmitter` slot of the project's 302-entry classification
  table.
- **Original filename in upstream mirror**: `elife-95402-supp2-v1.xlsx`
- **Size**: 73,537 bytes (~74 KB)
- **SHA256**: `0013e4b5f366b82a6b0ec0d682c3bace4027545c957823841293de93feafc0e2`
- **Source URL**: <https://raw.githubusercontent.com/openworm/ConnectomeToolbox/main/cect/data/elife-95402-supp2-v1.xlsx>
- **Mirror licence**: MIT (the same OpenWorm ConnectomeToolbox mirror the Cook 2019 file above
  is vendored from; see that entry's note on the `LICENSE` / `setup.cfg` discrepancy)
- **Retrieval date**: 2026-09-08
- **Accompanying paper**:
  - **Title**: A neurotransmitter atlas of *C. elegans* males and hermaphrodites
  - **Authors**: Chen Wang, Berta Vidal, Surojit Sural, Curtis Loer, G. Robert Aguilar,
    Daniel M. Merritt, Itai Antoine Toker, Merly C. Vogt, Cyril Cros, Oliver Hobert
  - **Journal**: *eLife* 13:RP95402 (2024)
  - **DOI**: <https://doi.org/10.7554/eLife.95402>
- **Redistribution rationale**: as for the Cook 2019 file — academic re-use of open-access
  Supplementary Information, sourced from OpenWorm cect's MIT-licensed mirror. *eLife* publishes
  under CC BY. Any consumer of our codebase should cite Wang et al. 2024 when publishing results
  derived from these identities.

### `witvliet_2020_dataset8_adult.xlsx`

- **Description**: One of eight developmental connectomes from the Witvliet
  et al. 2021 series; dataset 8 is the adult hermaphrodite worm. Covers the
  *C. elegans* nerve ring (~150-200 neurons of the whole-animal 302).
- **Original filename in upstream mirror**: `witvliet_2020_8 adult.xlsx`
- **Size**: 53,143 bytes (~53 KB)
- **SHA256**: `fdead89606257c1b26e57069fbe1de14c7696633b75b59b79e74c0bcc3497e62`
- **Source URL**: <https://raw.githubusercontent.com/openworm/ConnectomeToolbox/master/cect/data/witvliet_2020_8%20adult.xlsx>
- **Mirror licence**: MIT (same OpenWorm cect mirror as Cook 2019 above)
- **Retrieval date**: 2026-05-23
- **Accompanying paper**:
  - **Title**: Connectomes across development reveal principles of brain
    maturation
  - **Authors**: Daniel Witvliet, Ben Mulcahy, James K. Mitchell,
    Yaron Meirovitch, Daniel R. Berger, Yuelong Wu, Yufang Liu,
    Wan Xian Koh, Rajeev Parvathala, Douglas Holmyard, Richard L. Schalek,
    Nir Shavit, Andrew D. Chisholm, Jeff W. Lichtman, Aravinthan D. T. Samuel,
    Mei Zhen
  - **Journal**: *Nature* 596, 257–261 (2021)
  - **DOI**: <https://doi.org/10.1038/s41586-021-03778-8>
- **Note on year naming**: cect's source filename uses `witvliet_2020_` because
  the dataset was first released as a preprint in 2020; the journal publication
  followed in 2021. The project refers to it as "Witvliet 2021" in artefacts
  consistent with the published-paper year; the original cect filename is
  preserved on disk for traceability against the upstream mirror.
- **Redistribution rationale**: Same as Cook 2019 above — academic re-use of
  *Nature* SI via the MIT-licensed cect mirror.

### `creamer_lds_2026_model_weights.csv`

- **Description**: fitted synaptic weights from Creamer, Leifer & Pillow's connectome-constrained
  linear dynamical system — a signed weight for each directed neuron pair the model's mask allows,
  as `presynaptic cell,postsynaptic cell,weight`. 2,011 off-diagonal entries over 125 neurons (the
  fitted model holds 154). The values are coefficients of a 2 Hz dynamics matrix fitted to calcium
  imaging, on a mask that unions chemical and gap-junction edges from White 1986 and Witvliet 2020 —
  not Cook 2019 and not typed by connection, so a consumer joins it by name and reports coverage.
  Consumed as a measured prior for the connectome brain's chemical weights.
- **Original filename in upstream repository**: `quick_start_examples/model_weights.csv`
- **Upstream commit**: `bba43302d50a4947804d98b01779856e648237cc` (the file last changed at
  `009c767973688dcc7ea6124c620bb14334695904`)
- **Size**: 41,959 bytes
- **SHA256**: `f452b88461aa90d414fd652e246e11302293d207510b9f4c94bf9a9a8098924c`
- **Line endings**: CRLF, as upstream. Marked `-text` in `.gitattributes` so git stores the bytes
  unchanged; `text=auto` would otherwise normalise them and break the digest.
- **Source URL**: <https://github.com/Nondairy-Creamer/Creamer_LDS_2026/blob/bba43302d50a4947804d98b01779856e648237cc/quick_start_examples/model_weights.csv>
- **Licence**: MIT — the repository's single `LICENSE`, "Copyright (c) 2026 Matthew S. Creamer",
  reproduced in `LICENSE-creamer-lds.txt` beside the file as the licence requires. The repository
  has no separate data licence; the file is covered because it ships inside the repository, which
  is an interpretation of a licence worded for "the Software", and is recorded as such.
- **Retrieval date**: 2026-09-23
- **Accompanying paper**:
  - **Title**: Bridging the gap between the connectome and whole-brain activity in *C. elegans*
  - **Authors**: Matthew S. Creamer, Andrew M. Leifer, Jonathan W. Pillow
  - **Status**: **preprint**, bioRxiv 2024.09.22.614271 v3 (PMID 41040343); no journal version
    found at retrieval
  - **DOI**: <https://doi.org/10.1101/2024.09.22.614271>
- **Named for** its source repository, because the preprint (2024), its v3 (2025) and the
  repository (2026) each carry a different year.
- **Redistribution rationale**: MIT permits redistribution with the notice retained. Any consumer
  should cite Creamer et al. and the Randi et al. 2023 atlas the model was fitted to (below).

## What is NOT vendored

Cook 2019 SI 1 is **not** vendored. Pre-implementation investigation found
that Cook et al. 2019 does NOT publish a discrete "cell list" XLSX in its
supplementary information — the 302-neuron classification is spread across
paper tables, figures, and WormAtlas references. The de facto machine-readable
codification of the paper's classification lives in cect's `cect/Cells.py` as
hand-curated Python constants.

The 302-neuron classification therefore ships as code in
`packages/quantum-nematode/quantumnematode/connectome/neurons.py`, derived
from cect's MIT-licensed `Cells.py` constants. The module docstring records
the full attribution chain:

> this project → cect.Cells.py → Cook et al. 2019 paper + WormAtlas

**cect source URL**: <https://github.com/openworm/ConnectomeToolbox>
**cect commit / version pinned at curation time**: v0.3.1 (March 2026)
**cect licence**: MIT (per its `LICENSE` file)

### The Randi et al. 2023 signal-propagation atlas, and other Creamer artefacts

The raw functional measurement the fitted weights come from — Randi, Sharma, Dvali & Leifer,
"Neural signal propagation atlas of *Caenorhabditis elegans*", *Nature* 623:406 (2023),
<https://doi.org/10.1038/s41586-023-06683-4> — is **cited, not vendored**. Checked on 2026-09-23:

- **OSF `e2syt`** (<https://osf.io/e2syt/>), the primary deposit: **no licence stated**. Without one
  the default is all rights reserved.
- **`funatlas.h5`** inside the `wormneuroatlas` package (<https://github.com/francescorandi/wormneuroatlas>):
  **GPL-3.0**. Not vendored into this Apache-2.0 repository.
- **`leiferlab/worm-functional-connectivity`**: no licence stated.

Also not vendored, from the Creamer repository and deposit:

- **Model pickles** (`models/*.pkl`, ~20 MB each; MIT): unpickling executes code, and the CSV above
  holds what the connectome brain reads.
- **OSF `qxhjd`** (<https://osf.io/qxhjd/>), Randi's recordings reformatted for the fitting code:
  **no licence stated**.

A licence such as CC BY on the two OSF deposits would remove the obstacle to vendoring either.

## Verification

To verify the vendored files match the upstream mirror after a fresh clone:

```bash
shasum -a 256 data/connectome/cook_2019_si5_connectome_adjacency.xlsx
# expected: 559989daa02cd9a76e9266537a6b80bfd47d338cb5d397d11296288629df364a

shasum -a 256 data/connectome/emmons_2024_s1_connectome_adjacency.xlsx
# expected: e866b43f19ba5c70b773c94efd06aff6d6b2887cd24eed4412da80c06986418d

shasum -a 256 data/connectome/witvliet_2020_dataset8_adult.xlsx
# expected: fdead89606257c1b26e57069fbe1de14c7696633b75b59b79e74c0bcc3497e62

shasum -a 256 data/connectome/elife-95402-supp2-v1.xlsx
# expected: 0013e4b5f366b82a6b0ec0d682c3bace4027545c957823841293de93feafc0e2

shasum -a 256 data/connectome/creamer_lds_2026_model_weights.csv
# expected: f452b88461aa90d414fd652e246e11302293d207510b9f4c94bf9a9a8098924c
```
