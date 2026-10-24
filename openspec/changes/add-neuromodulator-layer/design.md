## Context

**The current field** (`brain/arch/connectome_ppo.py`, `_build_modulators`;
`connectome/neuromodulators.py`):

- **State:** two concentrations, serotonin and PDF. Each moves toward its release cells' mean
  positive settled rate: `c ← d·c + (1 − d)·release`, with `d = exp(−Δt / 60 s)`.
- **Current:** four pathways, each a fixed sign on hand-placed cells times a learnable softplus
  strength starting at 1:
  - serotonin through MOD-1 (inhibitory);
  - PDF through PDFR-1 (excitatory);
  - PDF's inhibition of NSM;
  - food ingestion onto NSM.
- **Mechanics:** the current is held through each settling step. The concentrations are state
  outside the gradient.
- **Knockouts:** `tph1` zeroes serotonin's release; `pdfr1` zeroes the PDFR-1 and PDF-onto-NSM rows.
- **Batched evaluator:** carries it through `modulator_release_batched`.

**The data** (checked 2026-10-24):

- **Bentley et al. 2016**, PLoS Comput Biol 12:e1005283, S1 Dataset (`.s004`, 471 KB zip,
  scriptable; CC BY 4.0 with the article).
  - `monoamine_expression.tsv`: producers by marker gene, 40 rows. tph-1 marks ADF, HSN and NSM;
    the other markers are cat-2, dat-1, mod-5, tbh-1 and tdc-1.
  - `monoamine_receptor_expression.tsv`: 543 rows over 19 receptors, each with an `excluded` flag and
    its reason. dop-5 and dop-6 are provisional.
  - `neuropeptide_expression.tsv`, `neuropeptide_receptor_expression.tsv` and
    `neuropeptide_receptor_mapping.tsv`: 22 couples, including pdf-1 → pdfr-1 and pdf-2 → pdfr-1.
  - Rows are individual neurons, curated from reporters and WormBase.
- **Ripoll-Sánchez et al. 2023**, Neuron 111:3570, figshare collection 10.6084/m9.figshare.c.6895870
  (CC BY 4.0 per item, scriptable).
  - 92 NPP–GPCR couples at EC50 ≤ 500 nM (`20230602_Ligand_receptors_interactions_all_EC50_explicit_500nM.csv`).
  - Each couple has binary 302 × 302 individual networks for the short-, mid- and long-range models.
    Every long-range network is exactly the outer product of its producer and receptor sets, checked
    on all 92, self-edges included. So both sets are recovered exactly.
  - The couples span 49 ligand genes and 51 receptors.
  - The authors' 2024 corrections (dmsr-5, npr-34) and their threshold-4 expression table are only
    on GitHub, under MIT.

**Where the sources and the hand-placed circuit differ:**

- **PDF-1:** Ripoll-Sánchez's 92 couples carry PDF only as NLP-37 (PDF-2), from AIA, AIY, ADA, BDU,
  AVH, PVQ, PVR, PVT and RID. Bentley lists pdf-1 in 41 cells, including Flavell's AVB, PVP and SIAV.
- **PDFR-1:** Bentley's PDFR-1 cells do not include the hand-placed AIY, RIM and RIA.
  Ripoll-Sánchez's do include NSM, the route by which PDF is thought to keep NSM inactive.

## Goals / Non-Goals

**Goals**

- A neuromodulator layer placed from data, behind a byte-identical default.
- One pool per ligand, signs from data where published, and strengths and timescales that C.2 can
  fit.
- Positive controls: the machinery reproduces the hand-placed field, and the data's departures from
  it are tabled.
- A receptor-placement null with D24's audit, ready for W.

**Non-goals**

- Fitting the strengths or timescales: that is C.2.
- Choosing the mutant encodings: that is D37's table, registered before C.
- The short- and mid-range models, beyond recording how to add them.
- The MIT-only 2024 corrections.

## Decisions

### Decision A: The derived layer file

`scripts/analysis/extract_neuromodulator_layer.py` downloads both sources by URL, checks their
checksums, and writes `data/neuromodulators/layer.json`. Only the derived file is vendored. It holds:

- **`ligands`:** one per gene. The four monoamines are named by transmitter (serotonin, dopamine,
  tyramine, octopamine). Each peptide is named by gene, lowercased across sources. Each ligand records
  its producer cells, each cell tagged with its sources.
  - **Monoamine producers** are Bentley's rows for each transmitter's **synthesis** marker, which the
    file names on every row, with `excluded = 1` rows dropped:

    - serotonin: tph-1, in ADF, HSN and NSM;
    - dopamine: cat-2, in ADE, CEP and PDE;
    - octopamine: tbh-1, in RIC;
    - tyramine: tdc-1, in RIM.

    The reuptake markers are not producers: dat-1 marks the same cells as cat-2, and mod-5 adds AIM,
    which takes serotonin up but does not make it. AIM's exclusion is recorded.
- **`couples`:** one per (ligand, receptor gene), with its receptor cells, each cell tagged with its
  sources. Each couple also records its sign class (`fixed`, with a sign and citation, or `fitted`) and
  its EC50 where a source gives one.
  - Peptide couples are the union of Ripoll-Sánchez's 92 and Bentley's 22, keyed by **canonical gene
    names**. `data/neuromodulators/gene_synonyms.json` maps every source name to its WormBase public
    name, each entry checked against its WormBase gene page at implementation. Bentley's pdf-2 and
    Ripoll-Sánchez's NLP-37 are one gene, so they become one ligand. After canonicalisation, a couple
    that still matches under two names is refused.
  - **The union applies no common potency cut.** Ripoll-Sánchez admits couples at EC50 ≤ 500 nM.
    Bentley's 13 further couples include micromolar ones: flp-1 → npr-4 at 0.4–9 µM, nlp-1 → npr-11 at
    1–100 µM, and pdf-1 → pdfr-1 at 0.4–5 µM, against 114 nM for pdf-2 → pdfr-1. Each couple records
    its EC50 range and its source. **A named sensitivity, `ec50_500nM`, applies Ripoll-Sánchez's rule
    to every couple.** It removes PDF-1, which matters behaviourally (Flavell 2013) but is a weak agonist
    in heterologous assays.
  - Monoamine couples come from Bentley's receptor rows, excluded rows dropped. Each couple's ligand is
    the row's `monoamine` column: for example, ser-2 is a tyramine receptor and ser-3 an octopamine
    one. dop-5 and dop-6 are left out, as in Bentley's main network.
- **`drives`:** sensory inputs onto a producer, outside any couple. Food ingestion onto NSM is kept
  from the hand-placed field (Rhoades 2019, DEL-3/DEL-7). Drives are sensory only.
- **`links`:** a ligand acting on named cells with a fixed sign and a cited mechanism, but no
  receptor gene. Each is tagged to the receptor gene whose knockout removes it. The data layer has
  no links, since its sources place only receptor genes. The hand-placed spec uses one: PDF's
  inhibition of NSM, removed by pdfr-1's knockout.
- **`provenance`:** URLs, checksums, licences and the extraction date.

Names are matched to the brain's connectome. A cell the connectome lacks is reported and skipped,
never invented, as `load_circuit` does now.

### Decision B: Receptor signs

`data/neuromodulators/receptor_signs.json` lists each monoamine receptor's coupling and sign with a
citation. It is verified against the source's text at implementation, as the hand-placed circuit was.

The mapping below is the layer's rule, not a claim about every cell: cAMP and Gq signalling can
excite or inhibit depending on the cell.

| Coupling | Sign |
|---|---|
| Gαi/o-coupled | inhibitory |
| Gαq- or Gαs-coupled | excitatory |
| MOD-1 (serotonin-gated Cl⁻) | inhibitory |
| LGC-53 (dopamine-gated Cl⁻) | inhibitory |
| LGC-55 (tyramine-gated Cl⁻) | inhibitory |

A receptor whose coupling cannot be verified becomes `fitted` and is listed. Every peptide couple is
`fitted`: most peptide GPCRs' coupling is unpublished, and a partial table would be a second kind of
hand placement.

### Decision C: Dynamics and current

With `modulator_layer: data`, the brain keeps one concentration per ligand, K in all (about 55):

- **Release:** the mean positive settled rate of the ligand's producers present in the connectome.
- **Update:** `c_k ← d_k c_k + (1 − d_k) r_k` with `d_k = exp(−Δt / τ_k)`. The step is
  `modulator_step_seconds`. `τ_k` defaults to the hand-placed 60 s for every ligand, and
  `modulator_timescales_s`, a mapping from ligand to seconds, overrides it. These are C.2's encoder
  parameters; PPO does not train them, since the concentrations sit outside the gradient.
- **Current:** `I = (c[ligand(j)] · s_j)_j @ M`. M is a J × N map with a 1 on each couple's or link's
  cells (about 121 couples once pdf-2 and nlp-37 are merged: 104 peptide and 17 monoamine).
  - **Fixed couples:** `s_j = sign_j · softplus(ρ_j)`, starting at 1, as now.
  - **Fitted couples:** `s_j = w_j`, a free signed value starting at 0.
  - **Drives:** rows of their own, as the ingestion row is now.

`modulator_layer: hand_placed`, the default, builds today's four-pathway field unchanged.

### Decision D: Knockouts by gene

`modulator_knockout_genes`, a list, read only with the data layer:

- a **producer gene's** knockout holds its ligand at zero (tph-1 holds serotonin at zero; pdf-1 holds
  the pdf-1 pool at zero);
- a **receptor gene's** knockout zeroes its couples' rows (pdfr-1 removes both PDF couples).

The hand-placed `modulator_knockout` keeps its two values and meaning. Each knob refuses the other
layer: `modulator_knockout` is refused with `modulator_layer: data`, and `modulator_knockout_genes`
with `hand_placed`. Neither is a mutant encoding; D37's table chooses those.

### Decision E: Batched evaluation

`BatchedConnectome` carries K concentrations per animal and the generalised current. Parity is held as
T.1's layer-1 test holds it: features, means and the K concentrations within 1e-5 over 50 steps, on
the data layer with a fixed draw of nonzero fitted strengths. The evaluator's declared surface admits
`modulator_layer: data`.

### Decision F: The positive controls

1. **The machinery reproduces the hand-placed field.** A layer spec built from the hand-placed circuit
   reproduces `_build_modulators`' currents, concentrations and knockouts bitwise, over random states
   and both knockouts. Its parts are:

   - two fixed couples: serotonin through MOD-1, and PDF through PDFR-1;
   - one link: PDF onto NSM, tagged to pdfr-1;
   - one drive: ingestion onto NSM.

2. **The slow variable.** With one ligand's τ set to 540 s at a 1-s step, a stopped release leaves its
   concentration at 1/e of its start within one step of 540, and a fixed couple's receptor current
   follows it. This is the layer's job for C.2: holding state for minutes.

3. **The data against the verified circuit.** A table, `docs/experiments/logbooks/supporting/106-neuromodulator-layer/circuit_agreement.json`,
   lists each link the hand-placed circuit verified and whether each source has it:

   - NSM releasing serotonin;
   - MOD-1 on its cells;
   - PDF from AVB, PVP and SIAV;
   - PDFR-1 on AIY, RIM and RIA;
   - PDF's link to NSM.

   Disagreements are reported, never patched. A missing link is not a failure, since the layer is
   meant to come from data, but the logbook states each one beside its source.

### Decision G: N.3, the receptor-placement null

**Classes.** `quantumnematode.connectome.neuron_classes` assigns each of the 302 neurons its anatomical
class by C. elegans naming:

- the numbered ventral-cord classes (AS, DA, DB, DD, VA, VB, VC, VD) drop their digits;
- other names drop a side suffix (L, R, DL, DR, VL, VR, D, V) only where another name shares the
  stem. So RMED, RMEV, RMEL and RMER are RME, while AVL stays AVL.

A test requires the 118 classes WormAtlas gives for the hermaphrodite's 302 neurons, and checks named
examples (OLQ has 4 cells, IL1 6, VD 13, SAB 3).

**The move.** For each receptor gene, its cells are grouped by class, giving k cells of a class of
size n. Each group moves to a class of the same size n, drawn without replacement among the classes
of that size, and lands on the same member positions: the k cells take the target class's members in
the source's order of suffix or digit. So:

- each receptor's class count and cell count are kept exactly;
- left/right pairs move together;
- a receptor gene with several ligands keeps one placement across its couples.

A class whose size is unique (for example VD, 13) can only map to itself. The share of each
receptor's cells pinned this way is reported in the audit.

Producers, drives and signs are held. A config names the draw by seed, `modulator_layer_null: receptor_placement` with
`modulator_layer_null_seed`, beside the brain's existing `wiring` options and validated the same
way.

**What it does not preserve, stated:**

- which classes carry each receptor;
- the per-neuron receptor load and sign balance (reported, not held);
- receptors' overlap with their ligand's producers and with those producers' synaptic partners.

**The matched-statistics table** (D24, in the model's units),
`scripts/analysis/neuromodulator_null_audit.py`, over 64 draws (seeds 7501–7564) against the data
layer:

- each receptor's cell count (exact by construction), and the share of its cells pinned by a unique
  class size;
- the per-neuron receptor load: mean, 99th percentile and maximum, over all couples and over fixed
  couples;
- the per-neuron fixed-sign balance;
- for ALA, CAN, PVD, AVA and AVB, their couples;
- for each ligand, the fraction of its receptor cells that are its producers' synaptic partners;
- the count of couples from sensory producers to motor receptor cells, which is the shortcut audit's
  modulator term. `sensory_motor_hops.py` gains it: under the long-range model a pool joins every
  producer to every receptor cell in two hops.

## Risks / Trade-offs

- **About 55 timescales and 130 strengths widen C.2's search.** The layer exposes them all; C.2's
  registration chooses the tying, for example one timescale per class of ligand, and the budget.
- **The union mixes two expression methods.** Every cell carries its source tag, so a sensitivity can
  drop either source. The logbook reports how much each contributes.
- **Long range is the least local model.** Short and mid range would add pools per area, or per-edge
  terms. The file records each source's neuron areas, so mid range can be added as a sensitivity
  without re-extraction.
- **Fitted couples starting at zero leave the peptides inert until fitted.** That is the neutral
  start. Under PPO they learn from zero, which the batched parity test covers.
