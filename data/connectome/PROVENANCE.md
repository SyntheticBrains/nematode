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

## Verification

To verify the vendored files match the upstream mirror after a fresh clone:

```bash
shasum -a 256 data/connectome/cook_2019_si5_connectome_adjacency.xlsx
# expected: 559989daa02cd9a76e9266537a6b80bfd47d338cb5d397d11296288629df364a

shasum -a 256 data/connectome/witvliet_2020_dataset8_adult.xlsx
# expected: fdead89606257c1b26e57069fbe1de14c7696633b75b59b79e74c0bcc3497e62
```
