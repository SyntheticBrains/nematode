## Why

The connectome's neuromodulator field is hand-placed. Two modulators (serotonin and PDF), four
pathways and a dozen cells were each checked against a source on 2026-10-11
(`data/neuromodulators/roaming_dwelling_circuit.json`). That was enough to test whether a modulated
connectome phenocopies tph-1 and pdfr-1 (Logbook 090). It is not enough for Phase 10, for two reasons:

- **C.2's slow variable.** The roadmap names the data-derived layer's fitted timescales as where the
  slow state comes from if the settling network cannot hold a 9-minute decay.
- **W's wiring question.** It asks whether the real wiring, *including its neuromodulator layer*,
  beats its nulls (D40). A hand-placed field has no placement to null.

Two public datasets give that placement neuron by neuron:

- **Bentley et al. 2016:** the monoamine layer (serotonin, dopamine, tyramine and octopamine
  producers and receptors) and 22 reporter-curated neuropeptide couples.
- **Ripoll-Sánchez et al. 2023:** the neuropeptide connectome, 92 couples at EC50 ≤ 500 nM from
  CeNGEN expression.

This change is tracker block N: N.1 (licences), N.2 (the layer) and N.3 (its null family).

## What Changes

The maintainer's decisions of 2026-10-24 are marked (decided).

- **N.1, licences, confirmed 2026-10-24:**
  - Bentley 2016's S1 Dataset is CC BY 4.0, as is the article.
  - Ripoll-Sánchez 2023's article and its figshare data (collection 10.6084/m9.figshare.c.6895870)
    are CC BY 4.0, item by item.
  - Their 2024 corrections (dmsr-5 and npr-34) and the per-neuron expression table exist only in the
    authors' GitHub repository, under MIT. They are not vendored, and the change records what they
    would alter.
- **The peptide data are the union of both sources, every edge tagged (decided).**
  - Ripoll-Sánchez's 92 couples come from the CC BY long-range individual networks. Each is an exact
    product of its producer and receptor sets (checked on all 92), so both sets are recovered without
    the MIT expression table.
  - Bentley's 22 couples are added; they include PDF-1, which the 92 do not.
  - Gene names are canonicalised to WormBase's first, since Bentley's pdf-2 is Ripoll-Sánchez's
    NLP-37.
  - No common potency cut applies. Each couple records its EC50: PDF-1's is micromolar. A named
    sensitivity applies Ripoll-Sánchez's 500 nM rule to every couple.
  - Each producer and receptor cell records its source, so either source can be dropped as a
    sensitivity.
  - Monoamines come from Bentley, with its exclusions kept.
- **One pool per ligand: the long-range model (decided).** Each ligand gene has one concentration,
  moving toward its producers' mean positive settled rate with its own timescale. Every cell carrying
  one of its receptors reads it. That is the current field's dynamics, generalised from two
  modulators to about 55.
- **Signs: monoamine receptors fixed, peptide couples fitted (decided).**
  - A monoamine receptor's sign comes from its published coupling (Gαi/o and the Cl⁻ channels MOD-1,
    LGC-53 and LGC-55 inhibitory; Gαq/s excitatory), each with a citation. A receptor whose coupling
    cannot be verified is fitted instead, and recorded.
  - Each peptide couple carries a signed strength starting at zero.
- **`modulator_layer: hand_placed | data`** on the connectome brain, read only with `modulators: true`.
  `hand_placed` is the default, so every existing config and result is byte-identical.
- **Knockouts by gene.** In the data layer, a producer gene's knockout holds its ligand at zero, and a
  receptor gene's knockout removes its couples. The existing `tph1` and `pdfr1` keep their meaning.
- **Batched evaluation.** `BatchedConnectome` carries the data layer, with parity tests against the
  per-step brain, since C.2 fits through it.
- **Positive controls for the new component:**
  - restricted to the hand-placed couples, link and drive, the data layer reproduces the hand-placed
    field's currents exactly;
  - a ligand given a 9-minute timescale holds its state for minutes, the job C.2 needs it for;
  - a table compares the data's cells with every link the hand-placed circuit verified, recording each
    disagreement rather than patching it.
- **N.3, the receptor-placement null (decided: count per receptor, by class):**
  - each receptor gene's cells, grouped by anatomical class, move to classes of the same size on the
    same member positions. Its class count and cell count are kept exactly, and left/right pairs
    move together. Classes come from a new table checked against WormAtlas's 118;
  - producers are held;
  - it carries D24's matched-statistics table in the model's units and the shortcut audit.

## Capabilities

### New Capabilities

- `neuromodulator-layer`: the data-derived layer, its provenance, its dynamics, knockouts, batched
  support, positive controls and receptor-placement null.

## Impact

- **Code:**
  - `quantumnematode/connectome/neuromodulator_layer.py` (new): the loader, the layer and the null;
  - `brain/arch/connectome_ppo.py`: `modulator_layer`, the generalised field and gene knockouts;
  - `evaluation/controllers.py`: `BatchedConnectome` on the data layer;
  - `scripts/analysis/extract_neuromodulator_layer.py` and
    `scripts/analysis/neuromodulator_null_audit.py`.
- **Data:** `data/neuromodulators/` gains the derived layer, the receptor-sign table and their
  provenance.
- **Default behaviour:** none. `modulator_layer` defaults to `hand_placed`.
- **Cost:** about 6–10 engineering days (the tracker's estimate). No campaign; the null audit draws
  run in seconds.
- **Seeds:** the null audit's draws use a new band, 7501–7564, added to the tracker's table.
- **Downstream:**
  - C.2 fits the strengths and timescales; its registration decides which are tied;
  - W rewires the layer with N.3's null;
  - D37's mutant table can encode tph-1 and pdfr-1 as gene knockouts.
