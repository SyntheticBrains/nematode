# 106: Phase 10 — The Neuromodulator Layer from Data: Bentley 2016 and Ripoll-Sánchez 2023 on the Connectome

**Status**: completed. **N ships.** The connectome brain can now carry a neuromodulator layer placed from data instead of the hand-placed serotonin and PDF circuit:

- 55 ligand pools from two CC BY sources;
- 121 couples, every cell tagged with the source that places it;
- gene knockouts;
- support in the batched evaluator;
- a receptor-placement null with its audit.

The hand-placed field stays the default and is unchanged. Its machinery is reproduced by the new layer, restricted to it.

> - **The layer:** 4 monoamines and 52 peptide genes (flp-24 has no producer, so 55 pools), 121 couples. 16 carry a published sign; the other 105 are fitted.
> - **The two sources each support half of the hand-placed circuit.** Bentley places PDF-1 in AVB, PVP and SIAV, and MOD-1 on AIY. Ripoll-Sánchez places PDFR-1 on AIY, RIM, RIA and NSM. Neither places MOD-1 on ASI or RIF.
> - **The slow variable:** a ligand given τ = 540 s holds its state for nine minutes, within one step.
> - **The null keeps every receptor's cell and class counts exactly, but it cannot change sensor-to-motor reach.** Under one pool per ligand, sensory-produced ligands reach all 39 motor neurons in two hops, in the data and in every draw. It also pins receptors on the four ventral-cord classes of unique size.

**Date**: 2026-10-24.

**OpenSpec change**: `add-neuromodulator-layer`.

**Records**:

- [`supporting/106-neuromodulator-layer/circuit_agreement.json`](supporting/106-neuromodulator-layer/circuit_agreement.json);
- [`supporting/106-neuromodulator-layer/null_audit.json`](supporting/106-neuromodulator-layer/null_audit.json);
- the layer and its provenance in `data/neuromodulators/`.

## Objective

Phase 10 needs a neuromodulator layer that is placed, not chosen.

- **For C.2:** the roadmap names the data-derived layer's fitted timescales as the slow state's source, if the settling network cannot hold a 9-minute decay.
- **For W (D40):** it asks whether the real wiring, including its neuromodulator layer, beats its nulls. A hand-placed field has no placement to null.

Block N was to:

- confirm the licences (N.1);
- build the layer behind a byte-identical default (N.2);
- give it a null family with D24's audit (N.3).

These were the maintainer's decisions of 2026-10-24:

- the union of both sources' peptide couples, every edge tagged;
- one pool per ligand (Ripoll-Sánchez's long-range model);
- monoamine receptor signs fixed from their published coupling, and peptide couples fitted;
- a receptor-placement null keeping each receptor's count, moved by class.

## Licences (N.1)

- **Bentley et al. 2016:** the S1 Dataset is CC BY 4.0, with its article.
- **Ripoll-Sánchez et al. 2023:** the article and every figshare item used are CC BY 4.0.
- **Left out:** the authors' 2024 corrections, for dmsr-5 and npr-34, and their per-neuron expression table exist only on GitHub, under MIT. The project's rule admits only CC0 and CC BY, so the layer keeps the 2023 placement of those two receptors.

**The MIT table is not needed.** Each of Ripoll-Sánchez's 92 long-range networks is exactly its producer set times its receptor set; the extraction checks this on every network. So both sets come from the CC BY files alone.

## What was built (N.2)

- **`data/neuromodulators/layer.json`:**
  - **Monoamine producers** come from synthesis markers: serotonin from tph-1 (ADF, HSN, NSM), dopamine from cat-2, octopamine from tbh-1, tyramine from tdc-1.
  - **17 monoamine couples** come from Bentley's receptor rows.
  - **104 peptide couples** are the union of the two sources under WormBase names. Bentley's pdf-2 and Ripoll-Sánchez's NLP-37 are one gene, so they share one pool.
  - **No common potency cut.** A named sensitivity, `ec50_500nM`, drops the eight couples outside Ripoll-Sánchez's rule, PDF-1 → PDFR-1 at 0.4–5 µM among them.
- **`data/neuromodulators/receptor_signs.json`:** 16 monoamine receptors with a cited coupling. ser-5 is fitted: its Gαs coupling rests on phylogeny alone.
- **`data/neuromodulators/gene_synonyms.json`:** 170 names checked through UniProt, whose *C. elegans* gene names WormBase supplies.
- **`modulator_layer: data`** on the connectome brain, beside the default `hand_placed`:
  - one concentration per ligand, moving toward its producers' mean positive settled rate with its own timescale (60 s unless `modulator_timescales_s` names one);
  - a current summed over couples. A fixed couple's softplus strength starts at 1; a fitted couple's free signed strength starts at 0.
- **Gene knockouts** (`modulator_knockout_genes`): a producer gene holds its ligand at zero, and a receptor gene removes its couples. Each knockout knob refuses the other layer.
- **Sensitivities:** `modulator_layer_sensitivity: ec50_500nM`, and `modulator_layer_sources` to keep one source's cells.
- **`BatchedConnectome`** carries K concentrations and gene knockouts.
- **A config and its smoke run:** `*_lawns_datalayer_s1_primitives.yml`, one key off the modulated primitives arm.

## Positive controls

| Control | Result |
|---|---|
| The restricted layer: the hand-placed circuit written as a spec (two fixed couples, a link, a drive) | Currents equal the hand-placed field's bitwise on random inputs, with no knockout and under each of `tph1` and `pdfr1`. Thirty steps of the brain match within 1e-5 in action means and 1e-6 in concentrations: release is a product with producer weights, not a mean, so only its rounding differs |
| The slow variable: τ = 540 s at a 1-s step | the concentration reaches 1/e within one step of 540, and the receptor current scales with it |
| Batched parity, data layer, fitted strengths drawn nonzero | features (which carry every concentration), action means and derivatives within 1e-5 over 50 steps on four seeds |
| The default | modulators without a named layer build the hand-placed field; the 34 existing modulator tests pass unchanged |

## The data against the verified circuit

Each link the hand-placed circuit verified (Flavell 2013; Ji 2021; Rhoades 2019), and the sources that place it:

| Link | Bentley 2016 | Ripoll-Sánchez 2023 |
|---|---|---|
| serotonin released by NSM | yes | — (monoamines from Bentley only) |
| MOD-1 on AIY | yes | — |
| MOD-1 on ASI, RIF | no | — |
| PDF-1 released by AVB, PVP, SIAV | yes | no PDF-1 couple |
| PDF-2 released by AVB, PVP, SIAV | no | no |
| PDFR-1 on AIY, RIM, RIA | no | yes |
| PDFR-1 on NSM (PDF's link to NSM) | no | yes |

**Neither source alone contains the circuit, and together they contain most of it.** Bentley's reporter-curated data place PDF-1 where Flavell's Cre rescue did, but put PDFR-1 elsewhere. Ripoll-Sánchez's CeNGEN data place PDFR-1 on the cells where Flavell and Ji found its effects, and on NSM, the route by which PDF is thought to keep NSM inactive. MOD-1 on ASI and RIF, which the hand-placed circuit verified against its sources, is in neither. As registered, each disagreement is recorded, not patched. A fit on the data layer cannot use those two MOD-1 sites.

## The null (N.3)

**How it draws.** For each receptor gene, its cells are grouped by anatomical class and each group moves to a class of the same size, drawn without replacement, landing on the same member positions. The class table, `quantumnematode.connectome.neuron_classes`, rebuilds the 118 classes of the 302 neurons from their names.

**The audit** compares the data layer with 64 draws (seeds 7501–7564), in the model's units:

| Statistic | Data | Null: mean [min, max] |
|---|---|---|
| receptor cells per gene | — | identical in every draw |
| receptor load per neuron, mean | 17.5 | 17.5 (held by construction) |
| receptor load, 99th percentile / maximum | 43 / 48 | 41.4 [41, 44] / 43.9 [43, 53] |
| fixed-couple load, 99th percentile / maximum | 5 / 5 | 4.1 [3, 5] / 4.8 [3, 7] |
| fixed-sign balance, mean absolute / maximum | 0.81 / 4 | 0.83 [0.72, 0.94] / 3.3 [3, 5] |
| couples on ALA / CAN / PVD / AVA / AVB | 22 / 52 / 54 / 40 / 28 | 19.5 / 35.6 / 34.2 / 35.1 / 35.3, each ranging widely |
| share of receptor cells that are the producers' synaptic partners | 0.606 | 0.584 [0.556, 0.607] |
| sensory-produced couples reaching motor neurons / motor neurons reached | 16 / 39 | 16 / 39 in every draw |

**What it does not preserve, stated:**

- which classes carry each receptor;
- per-neuron load and sign balance, which are reported, not held;
- the overlap with synaptic partners. The data sit near the top of the null's range (0.606 against at most 0.607): the real receptors lie near their ligands' synaptic partners more often than most random placements do.

**Two findings W must read.**

1. **Under one pool per ligand, receptor placement cannot change sensor-to-motor reach.** Sixteen couples whose ligands sensory neurons produce reach all 39 motor neurons in two hops, in the data and in every draw. A wiring reading through this layer is about which cells read which pools, not about whether signals arrive.
2. **The null pins receptors on the four ventral-cord classes whose sizes no other class shares** (DA 9, DB 7, VA 12, VD 13). npr-34 keeps 59% of its cells in place, seb-3 and gnrr-1 53%, and dop-3 49% (`pinned_share` in the record). For those receptors the null is weak. A W registration on this null should report the pinned shares beside its result, or add a size-relaxed variant.

## What this means for C.2 and W

- **C.2** can fit 105 signed strengths, 16 fixed-sign strengths and up to 55 timescales. Its registration chooses the tying, for example one timescale per ligand class, and the budget.
- **The mutant table (D37)** can encode tph-1 as `modulator_knockout_genes: [tph-1]` (serotonin from ADF, HSN and NSM held at zero) and pdfr-1 as `[pdfr-1]` (both PDF couples removed).
- **W** rewires the layer with `modulator_layer_null: receptor_placement` and a fresh seed per fit. It reads the two findings above beside its result.

## Conclusions

1. **N ships.** The layer is placed from licensed data, tagged by source, and reproduces the hand-placed machinery when restricted to it.
2. **The two sources disagree with each other and with the hand-placed circuit, and the union recovers most of it.** PDF-1 comes from Bentley, PDFR-1's sites from Ripoll-Sánchez. MOD-1 on ASI and RIF is in neither.
3. **The null is honest about its limits.** It holds what it says, and its audit shows where it cannot bite: sensor-to-motor reach, and the ventral-cord receptors.

## Next steps

- [ ] C.2's registration: the strengths' and timescales' tying, its budget, and whether `ec50_500nM` or one source's cells is a registered sensitivity.
- [ ] D37's mutant table, using the gene knockouts.
- [ ] W: the null with the pinned shares reported, and a size-relaxed variant considered.
