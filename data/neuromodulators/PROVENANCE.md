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
