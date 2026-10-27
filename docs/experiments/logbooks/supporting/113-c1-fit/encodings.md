# The mutant encoding table (D37)

**Registered 2026-10-26**, change `add-c1-fit`, with [C.1's launch record](launch.md), before any controller is fitted. Every controller's encoding of the three mutants is fixed here. C.4 grades the predictions under this table. An encoding is a structural knockout, with no free magnitude. A mutant with no counterpart in a controller is not encodable, and that controller makes no prediction for it.

## What each mutant is graded on

Every mutant is graded on Scheer's small lawn, by its difference from the controller's own wild type, at |z| ≤ 2.24 (Bonferroni over two statistics; `data/benchmark/bands.json`).

| Mutant | Graded | Reported beside |
|---|---|---|
| tph-1 | roaming fraction, direction more | **leaving**, left out of the graded set. Its three band dates show Δ +0.07 per on-lawn hour, and B.4's single read of its two held-out dates showed 4.0× more ([Logbook 110](../../110-benchmark-controls.md)). It is reported against both values, and no held-out data are read again. The bar stays at 2.24. |
| pdfr-1 | roaming fraction and leaving, by magnitude (neither direction is resolved) | local search, which no held-out cohort can positive-control |
| tax-4 | roaming fraction, direction less; leaving by magnitude | — |

## The encodings

| Mutant | C.1 (`BatchedMechanistic`) | C.2 and C.3 (the connectome with the data-derived layer) |
|---|---|---|
| **tph-1** | a_intake = b_intake = 0. Switching loses its dependence on ingestion, which NSM's serotonin reports (Rhoades et al. 2019; Flavell et al. 2013). Satiety's dependence is kept, since it runs through other signals. | `modulator_knockout_genes: [tph-1]`: the synthesis gene's loss holds serotonin at zero (`neuromodulator_layer.MONOAMINE_GENES`). |
| **pdfr-1** | **Not encodable.** C.1 has no PDF signal, so it makes no pdfr-1 prediction, a stated limit of the mechanism. | `modulator_knockout_genes: [pdfr-1]`: the receptor gene's couples are removed. |
| **tax-4** | κ_w = β_r = β_o = 0. The head-sweep and dC/dt responses are removed, since tax-4 is the cGMP-gated channel of the chemosensory neurons (Komatsu et al. 1996). The edge response to falling food contact is kept: food contact is the lawn density under the worm, not a gradient. | The encoder's input to every tax-4-expressing sensory class is zeroed: AWC, AWB, ASE, ASG, ASI, ASJ, ASK and AFD (Komatsu et al. 1996), and URX, AQR, PQR and BAG (Gray et al. 2004; Hallem et al. 2011). The last four receive no input in these assays. The row binds whatever encoder F.0b-ii builds. |

## The `unidentified` rule

For a controller:

1. take its restarts whose answers pass the wild type, at least 3 of 5 registered populations;
2. if a mutant's verdict differs among them, the prediction is **`unidentified`**;
3. otherwise it is their shared verdict;
4. with one passing restart, its verdict;
5. with none, the chosen restart's verdict, reported with the wild-type failure stated.

`c1_fit.mutant_verdict` implements this rule.

## References

- Flavell SW, Pokala N, Macosko EZ, Albrecht DR, Larsch J, Bargmann CI (2013). Serotonin and the neuropeptide PDF initiate and extend opposing behavioral states in *C. elegans*. *Cell* 154:1023–1035.
- Gray JM, Karow DS, Lu H, Chang AJ, Chang JS, Ellis RE, Marletta MA, Bargmann CI (2004). Oxygen sensation and social feeding mediated by a *C. elegans* guanylate cyclase homologue. *Nature* 430:317–322.
- Hallem EA, Spencer WC, McWhirter RD, Zeller G, Henz SR, Rätsch G, Miller DM, Horvitz HR, Sternberg PW, Ringstad N (2011). Receptor-type guanylate cyclase is required for carbon dioxide sensation by *Caenorhabditis elegans*. *PNAS* 108:254–259.
- Komatsu H, Mori I, Rhee JS, Akaike N, Ohshima Y (1996). Mutations in a cyclic nucleotide-gated channel lead to abnormal thermosensation and chemosensation in *C. elegans*. *Neuron* 17:707–718.
- Rhoades JL, Nelson JC, Nwabudike I, Yu SK, McLachlan IG, Madan GK, Abebe E, Powers JR, Colón-Ramos DA, Flavell SW (2019). ASICs mediate food responses in an enteric serotonergic neuron that controls foraging behaviors. *Cell* 176:85–97.
