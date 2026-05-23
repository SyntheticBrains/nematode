## Overview

This change ships Phase 6 Tranche 1 (L0): the *C. elegans* connectome data model, the loader from vendored *Nature* Supplementary Information files, the validation suite (including cross-validation against Witvliet 2021), and a smoke-test forward pass that confirms the data model supports learned-weight computation. T1 ends before any plugin work (L1 / T2) or training run; the boundary is deliberate so the L0 Risk-mitigation pivot has its own evidence.

The strategic decisions live in [phase6-tracking/design.md](../phase6-tracking/design.md). This file records the per-tranche implementation decisions specific to T1.

## Goals / Non-Goals

**Goals:**

- Cook 2019 hermaphrodite (302 neurons) loadable from vendored Nature SI via `load_cook_2019_hermaphrodite()`.
- Witvliet 2021 adult (dataset 8) loadable via `load_witvliet_2021_adult()`.
- Chemical synapses and gap junctions exposed as separately-typed connection categories in the data model (phase6-tracking Decision 7).
- Cross-validation pipeline that diffs Cook 2019 nerve-ring against Witvliet 2021 dataset 8 and produces a documented `DivergenceReport`.
- Smoke-test forward pass with PPO-shaped weight tensors over chemical synapses (strict-mask, gap junctions fixed at Cook 2019 counts) returns finite output.
- T1↔T2 API sketch published in the T1 logbook for T2 plugin-design review.

**Non-Goals:**

- L1 plugin interface or `Brain` Protocol registration — that's T2.
- Wiring the connectome to the env, agents, or training loop — T2.
- PPO weight optimisation on the connectome — T2.6 / Gate 1's G1.c probe.
- Extra-synaptic / peptidergic / monoaminergic signalling — phase6-tracking Decision 7 reserves that for Phase 7 L4 plasticity.
- Position / morphology data for neurons — not needed for the strict-mask + fixed-gap-junction L2 substrate. Add later if a Phase 6+ use case needs it.
- C302 / NeuroML export path — phase6-tracking Decision 2 + RQ4 mark this as a Future Directions item if Sibernetic interop becomes relevant.

## Design Decisions

### Decision T1.1: Direct *Nature* SI parsing (no third-party connectome package)

phase6-tracking Decision 2 picked Cook 2019 as the data and tentatively named OpenWorm `cect` as the API layer, with the caveat that the API library could be swapped without affecting the data source choice. Pre-implementation investigation (full notes in T1 logbook) surfaced three concerns with `cect`:

- **Licence inconsistency**: cect's `LICENSE` file declares MIT (Copyright 2024 OpenWorm); cect's `setup.cfg` declares `license = LGPLv3`. The conflict is unresolved at the repo level. Even treating the `LICENSE` file as legal source of truth, the inconsistency itself is a flag for future review.
- **Project maturity**: cect is pre-1.0 (v0.3.1, March 2026), small contributor base, no formal release-branch discipline. The roadmap commits Phase 6 to a 6-10 month timeline; binding L0 to a pre-1.0 niche dependency carries asymmetric upside (API convenience) vs downside (the dependency drifts or breaks during Phase 6).
- **Transitive dependency cost**: cect drags in `pyneuroml`, `wormneuroatlas` (GPL-3.0), `hiveplotlib`, and others — none of which are needed for L0's data-loading job.

The alternative — direct *Nature* Supplementary Information parsing with `pandas` and `openpyxl` (neither currently in our dep tree, but both are standard, well-maintained packages with stable APIs) — costs ~150-300 lines of parser code we own plus two new dependencies. The cect-bundled data files we'd have used ARE the Cook 2019 + Witvliet 2021 *Nature* SI files; using them directly removes the wrapper layer that carries the risk. The two new dependencies are domain-neutral data-parsing libraries, not connectome-specific niche packages — they trade one pre-1.0 niche library (cect) for two production-grade standard libraries (pandas + openpyxl).

**What we keep from cect-as-API-layer**:

- The `syntype` distinction "Chemical" vs "GapJunction" (verified from `cect/Neurotransmitters.py`: `CHEMICAL_SYN_TYPE = "Chemical"`, `ELECTRICAL_SYN_TYPE = "GapJunction"`). This taxonomy carries forward into our `model.py` types directly.
- The neuron-classification structure (sensory / interneuron / motor / muscle / pharyngeal). Sourced from Cook 2019 SI 1 (the cell-list table is the upstream source cect uses for its `cect.Cells` module).

**What we drop**:

- The wrapper code itself. The 8-12 lines saved by `Cook2019HermReader().read_data()` are not worth the risk surface.
- Indirect support for other datasets (White 1986, Cook2020, Wang2024, Yim2024, etc.). L0 commits to Cook 2019 + Witvliet 2021 only; if a Phase 6+ tranche needs another dataset, a follow-up change adds a parser for it (~50-150 lines per dataset).

phase6-tracking Decision 2 framed the choice as Cook 2019 (data) primary; cect (API) was always interchangeable. This change records the API-layer choice as "direct pandas parsing of the same upstream SI files."

### Decision T1.2: Data-model shape with separately-typed connections (phase6-tracking Decision 7 made concrete)

The data model lives in `connectome/model.py` as Pydantic models. Match the project's existing pydantic convention (e.g. `BrainParams` in `packages/quantum-nematode/quantumnematode/brain/arch/_brain.py:110-342`).

```python
from typing import Literal
from pydantic import BaseModel, Field

CellClass = Literal["sensory", "interneuron", "motor", "muscle", "pharyngeal"]

class Neuron(BaseModel):
    name: str  # canonical C. elegans name, e.g. "ASEL", "AVA", "VB02"
    cell_class: CellClass
    neurotransmitter: str | None = None  # e.g. "Glutamate", "Acetylcholine", "GABA"

class ChemicalSynapse(BaseModel):
    pre: str   # presynaptic neuron name
    post: str  # postsynaptic neuron name
    weight: int = Field(gt=0)  # synapse count from Cook 2019 EM data

class GapJunction(BaseModel):
    # Undirected: convention is (neuron_a, neuron_b) sorted alphabetically.
    neuron_a: str
    neuron_b: str
    weight: int = Field(ge=0)  # junction count from Cook 2019 EM data

class Connectome(BaseModel):
    neurons: dict[str, Neuron]
    chemical_synapses: list[ChemicalSynapse]
    gap_junctions: list[GapJunction]
    source: str    # e.g. "cook_2019_hermaphrodite"
    version: str   # e.g. "Cook et al. 2019 Nature, SI 5 v2019-06-26"
```

**Critical dual-edge case** (phase6-tracking Decision 7): many *C. elegans* neuron pairs are connected by BOTH a chemical synapse AND a gap junction. AVA↔AVB is the canonical example. The data model represents these as two separate entries (one `ChemicalSynapse` AND one `GapJunction`), never as a single edge with two weight attributes. A test in `test_model.py` asserts AVA↔AVB exists as both entry types.

**Why pydantic**: matches existing project convention; provides field-level validation (`weight: int = Field(gt=0)` rejects zero-weight chemical synapses at parse time, which surfaces XLSX-parsing bugs early); serialises cleanly for the T1↔T2 API sketch.

### Decision T1.3: Loader follows the Cook 2019 XLSX sheet structure literally

Cook 2019 SI 5 ("Connectome adjacency matrices") has the following sheet layout per the published paper (verified via `cect/readers/Cook2019DataReader.py` source):

- One sheet per (sex × connection-type) combination — e.g. `hermaphrodite chemical`, `hermaphrodite gap jn`, `male chemical`, `male gap jn`
- Each sheet is a presynaptic-row × postsynaptic-column adjacency matrix
- Cell values are synapse counts (chemical) or gap-junction counts

The loader reads the relevant two sheets for hermaphrodite (chemical + gap jn), walks the matrix non-zero cells, and emits `ChemicalSynapse` / `GapJunction` entries. The neuron name list comes from row + column headers cross-referenced against the SI 1 cell-list (which provides the 302-name canonical roster + cell_class metadata).

**Why this matters as a design decision**: the loader does NOT depend on cect's `read_data()` return shape (which we don't fully control). It depends on the *Nature* SI file's documented sheet structure (which is in a peer-reviewed published paper, stable for the lifetime of the project).

### Decision T1.4: Hand-curated neuron classification table in `neurons.py`

The 302-neuron classification (sensory / interneuron / motor / muscle / pharyngeal) comes from Cook 2019 SI 1 cell-list table — the same upstream source `cect.Cells` uses internally. We extract it once at L0 implementation time and ship as a static dict in `connectome/neurons.py`. WormAtlas (wormatlas.org) is used as a cross-check during data extraction; any disagreement gets resolved by deferring to the Cook 2019 SI 1 canonical table and noting the WormAtlas variant in a code comment.

**Format**:

```python
NEURON_CLASSIFICATION: dict[str, tuple[CellClass, str | None]] = {
    "ASEL": ("sensory", "Glutamate"),
    "ASER": ("sensory", "Glutamate"),
    "AFDL": ("sensory", None),  # no canonical NT, glutamate suspected
    # ... 302 entries
}
```

**Why a static dict, not parsed from SI 1 at runtime**: the 302-entry classification is essentially a constant for *C. elegans* (the connectome wiring is fixed; cell identities are canonical). Parsing it from SI 1 at every load adds runtime cost and a fragile dependency on SI 1's exact sheet structure. Ship it as code; bump it in a follow-up change if the field updates the classification.

**Coverage assertion**: tests assert exactly 302 entries with valid `cell_class` values (module-import-time `assert len(NEURON_CLASSIFICATION) == 302`); the T1 logbook reports the observed counts per class. No band-checking assertion is added in the test suite because cell-class boundaries are convention-dependent — Cook 2019 SI 1, WormAtlas, and the project's `docs/nematode_biology.md` (which states sensory ~40%, interneuron ~30%, motor ~30%) use slightly different boundary rules for polymodal / pharyngeal-vs-non-pharyngeal cells. The logbook records counts for forensic review; tests stay loose.

### Decision T1.5: Smoke-test forward pass uses NumPy + chemical-synapse strict-mask

The smoke test in `connectome/smoke.py` validates that the data model can support a learned-weight forward computation. It does NOT instantiate a `Brain` Protocol implementation or touch any existing brain code (that's T2).

```python
def run_forward_pass(c: Connectome, *, seed: int = 0) -> np.ndarray:
    """Single PPO-shaped forward pass on the connectome with random weights.

    - Build chemical-synapse adjacency matrix (N×N), apply random weight per
      edge sampled from N(0, 1/sqrt(fan_in)) where fan_in is the in-degree of
      the post-synaptic neuron. Non-existent edges pinned to zero (strict-mask).
      The 1/sqrt(fan_in) scaling prevents the pre-tanh sum from saturating —
      Cook 2019 has up to ~50 chemical inputs per neuron, so an unnormalised
      N(0, 1) init would push tanh to ±1 across most outputs.
    - Build gap-junction adjacency matrix (N×N), use Cook 2019 junction count
      as fixed weight (Decision 7).
    - Forward pass: input synthetic vector through (chemical_W + gap_W) @ x,
      apply tanh, return output of motor-neuron rows.
    """
```

Output shape: number of motor neurons. Test asserts: no NaN/Inf in output; output has non-zero variance across the motor-neuron rows (catches a degenerate constant output AND a fully-saturated ±1 output); the chemical-synapse adjacency has at least one entry (sanity guard against load failure that silently returns an empty connectome).

**Why NumPy and not PyTorch**: smoke test is data-validation-grade, not gradient-grade. Plain NumPy is one less import in the smoke-test path and makes the test fast (< 1 second). PyTorch enters the picture in T2 when the connectome becomes a `Brain` Protocol implementation with real `learn()` semantics.

### Decision T1.6: Cross-validation strategy (T1.4 sub-task)

Witvliet 2021 dataset 8 (adult) covers the nerve-ring (~180 neurons of the 302). The cross-validation compares:

- Set of neurons present in both Cook 2019 and Witvliet 2021 dataset 8 (intersection)
- For each shared neuron-pair (`(pre, post)`), do both datasets report a chemical synapse? A gap junction? With weights within an order of magnitude?
- Document the divergence map: pairs that Cook 2019 reports but Witvliet 2021 doesn't (and vice versa), with hypotheses where they are obvious (e.g. developmental-stage difference, lineage-tracing-difference, sex-specific connections).

The cross-validation returns a `DivergenceReport`:

```python
class DivergenceReport(BaseModel):
    shared_neurons: int
    shared_pairs_agreement: int  # both report a chemical synapse OR both don't
    shared_pairs_disagreement: int
    cook_only_pairs: list[tuple[str, str]]   # chemical-synapse pairs in Cook 2019 not in Witvliet
    witvliet_only_pairs: list[tuple[str, str]]
    weight_divergence_summary: dict[str, float]  # mean/median weight ratio for shared pairs
```

This is the G1.a evidence for phase6-tracking Gate 1. The full Gate 1 decision lands at T2 close; T1 ships the report as input to that decision.

**Why ~50 lines of pandas**: the two datasets are already in compatible adjacency-matrix form once loaded. Computing intersection + per-pair agreement is straightforward pandas/numpy.

### Decision T1.7: Vendoring strategy + LFS

Vendor three XLSX files under `data/connectome/`:

- `cook_2019_si5_connectome_adjacency.xlsx` — Cook 2019 SI 5, sheet-per-(sex,connection-type)
- `cook_2019_si1_cell_list.xlsx` — Cook 2019 SI 1 cell list (canonical 302-neuron roster + cell class)
- `witvliet_2021_dataset8_adult.xlsx` — Witvliet 2021 dataset 8 (adult, nerve-ring)

All three are added to `.gitattributes` via `data/connectome/**/*.xlsx filter=lfs ...`. The existing `.gitattributes` only LFS-tracks `artifacts/`, `benchmarks/`, and `configs/`; the connectome data is the first reproducibility artefact under `data/`.

`data/connectome/PROVENANCE.md` records per file:

- Source URL (Nature SI direct link or PMC-equivalent)
- DOI of accompanying paper
- SHA256 of vendored file
- Retrieval date
- Citation (BibTeX-ready)
- Redistribution rationale (academic research re-use of *Nature* SI is standard; cite paper in publications)

**Why vendor and not fetch-on-build**: fetch-on-build adds a network failure mode to CI and a reproducibility hazard (URLs rot; *Nature* renames SI files when papers republish). The combined file size is ~5-10 MB across the three files — comfortably under GitHub's LFS bandwidth limits.

## What This Change Does Not Decide

Deferred to later tranches; recorded here so T2-T9 don't re-litigate:

- **Sensor-projection mapping** (e.g. ASE → food-chemotaxis observation, AFD → thermotaxis observation, ASH/ADL → nociception observation): this is T4's connectome-architecture ablation scope per phase6-tracking Decision 7. T1 ships the neuron-name-to-cell-class lookup; T4 decides how environmental observations route to named sensory neurons.
- **Motor-readout mapping** (e.g. how VB02-VB11 + DB01-DB07 motor-neuron outputs combine into a single action choice): also T4 scope per phase6-tracking Decision 7.
- **Hand-curated subset definition** for the L0 Risk-mitigation pivot ("~50-100 neurons" per phase6-tracking T1 risk-mitigation row): T1 ships the full 302; if Gate 1 G1.c (PPO-on-connectome learning probe in T2) shows the full connectome is too dense for stable training, T2 defines and ships the subset as an amended T1.
- **C302 / NeuroML export adapter**: phase6-tracking RQ4 trigger. T1 does not pre-build the exporter.

## Reuse: existing project patterns

- **Pydantic models**: match `BrainParams` style in `_brain.py:110-342` (typed fields, validators where useful, `BaseModel` not `dataclass`).
- **Tests**: match `tests/quantumnematode_tests/brain/test_ppo_features.py` class-based test organisation.
- **Logbook**: match Phase 5 logbook style (e.g. `docs/experiments/logbooks/013-lamarckian-inheritance-pilot.md`) — sectioned by question + result + verdict.
- **OpenSpec change scaffolding**: match `2026-02-06-add-qrc-brain` for length / tone (simpler scope) over the framework-level changes.

## Risks

1. **Cook 2019 SI 5's sheet structure varies from what the loader expects.** Mitigation: integration test loads the actual file and asserts neuron count = 302 and chemical-synapse count > 5000 (Cook 2019's published bound is ~7700 connections). If counts deviate, the loader fails loudly at T1 close rather than silently in T2.
2. **Witvliet 2021 dataset 8 has different neuron-name conventions than Cook 2019.** Mitigation: cross-validation step explicitly handles name aliasing (e.g. `RIAL` vs `RIA-L` — known variant); aliasing rules go in `connectome/neurons.py` as a canonical-name map.
3. **The hand-curated 302-neuron classification is brittle — one typo in a CellClass label breaks downstream lookups.** Mitigation: `test_neurons.py` asserts every entry is valid, count is exactly 302, coverage by class is in expected bands.
4. **Vendoring the *Nature* SI files raises a legal-review-style question for the team.** Mitigation: PROVENANCE.md cites the standard academic-research re-use rationale; if legal review surfaces an issue at PR-review time, fall back to fetch-on-first-use with the loader caching to `data/connectome/` (one-line code change).
5. **Pydantic v1/v2 import paths**: project is on Pydantic 2.11.4+ per AGENTS.md, so we use `pydantic.BaseModel` and `pydantic.Field` directly. No fallback shim needed.
6. **`openpyxl` version pin**: the loader uses `pd.read_excel(..., engine="openpyxl")`. `openpyxl` is in the existing dep tree but if the version is ancient enough to lack XLSX-format support for the *Nature* SI files (unlikely — *Nature* SIs are standard XLSX), bump `openpyxl` in the same change.
