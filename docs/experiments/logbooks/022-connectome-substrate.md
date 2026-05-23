# 022: Connectome Substrate L0 Ingestion

**Status**: implementation complete. The *C. elegans* connectome (Cook et al. 2019 hermaphrodite + Witvliet et al. 2021 adult) is loaded, validated, vendored, and forward-passable on synthetic input. The data-model API is settled and published below as the T1↔T2 handshake.

**Branch**: `feat/connectome-substrate`.

**OpenSpec change**: `add-connectome-substrate` (this tranche's change directory).

**Date Started**: 2026-05-23.

**Date Last Updated**: 2026-05-24.

This logbook records the L0 implementation (no plugin work, no training): what was loaded, what counts came out, what data-source decisions were made along the way, and what API the downstream plugin design (Tranche 2) will consume.

## Objective

L0 imports the real *C. elegans* connectome into the codebase as the substrate for every subsequent Phase 6 architecture comparison. The deliverable is *data in, validated, forward-passable* — strictly bounded to that. No plugin interface, no training, no env wiring.

## Data source decision

**Decision**: ingest Cook et al. 2019 *Nature* SI 5 (hermaphrodite + male whole-animal connectome adjacency matrices) and Witvliet et al. 2021 *Nature* dataset 8 (adult worm) via direct pandas parsing. **No third-party connectome-domain package** (cect, wormneuroatlas, etc.) is taken as a dependency.

### What was considered

| Source | Status | Notes |
|---|---|---|
| OpenWorm `cect` (ConnectomeToolbox) | Investigated, rejected as the API layer | Licence inconsistency between `LICENSE` (MIT) and `setup.cfg` (LGPLv3); pre-1.0 (v0.3.1, March 2026); transitive deps include `pyneuroml`, `wormneuroatlas` (GPL-3.0), `hiveplotlib` — none needed for L0. The risk of building Phase 6 on a pre-1.0 niche dependency outweighed the wrapper convenience (~8-12 lines saved). |
| `wormneuroatlas` (Randi 2022) | Investigated, rejected | GPL-3.0 (full copyleft). Smaller community (7 stars, no formal releases). Designed primarily for functional connectivity (Randi 2023), not the anatomical adjacency we need at L0. |
| Direct *Nature* SI parsing | **Chosen** | The data files cect bundles ARE the *Nature* SI files. Reading them directly with `pandas` + `openpyxl` (both production-grade standard libraries) removes the wrapper layer that carried the licence + maturity risk. Costs ~250 lines of parser code we own. The neuron-classification curation (302 entries with cell_class) is sourced once from cect's MIT-licensed `Cells.py` constants and shipped as a static Python dict in `connectome/neurons.py` (no runtime dep on cect). |

### Vendoring

Both vendored XLSX files live under `data/connectome/` with LFS-tracking via a new `.gitattributes` rule (`data/connectome/**/*.xlsx filter=lfs ...`). Per-file provenance (source URL, DOI, SHA256, retrieval date, paper citation, redistribution rationale) is recorded in `data/connectome/PROVENANCE.md`. The files are sourced from OpenWorm cect's MIT-licensed mirror of the *Nature* SI rather than directly from Nature.com URLs, partly because Nature SI URLs rot and partly because the cect mirror provides a stable hash-pinnable redistribution chain.

Cook 2019 does NOT publish a discrete "cell list" SI file (this surprised the planning artefacts and required revising them mid-implementation). The 302-neuron classification is a curation spread across the paper's tables, figures, and WormAtlas references; cect's `Cells.py` is the de facto machine-readable codification. Our `connectome/neurons.py` derives the classification from cect's MIT-licensed lists at curation time, then ships it as a static Python literal with no runtime cect dependency.

## Implementation summary

`packages/quantum-nematode/quantumnematode/connectome/` ships six modules:

- `model.py` — typed Pydantic types (`Neuron`, `ChemicalSynapse`, `GapJunction`, `Connectome`, plus the `CellClass` `Literal`). Field-level validation: chemical-synapse `weight: int = Field(gt=0)`, gap-junction `weight: int = Field(ge=0)`. Model-level validation: gap-junction canonical pair order (`neuron_a < neuron_b` lexicographically; self-loops rejected), no orphan edges in `Connectome`.
- `neurons.py` — `NEURON_CLASSIFICATION: dict[str, tuple[CellClass, str | None]]` covering all 302 hermaphrodite neurons with `assert len() == 302` at import time. Neurotransmitter slot is `None` for every entry in this initial table (enrichment is follow-up work).
- `loader.py` — `load_cook_2019_hermaphrodite()` and `load_witvliet_2021_adult()` reading the two vendored XLSX files. Cook 2019 loader normalises zero-padded suffixes (`VC01` → `VC1`, `VD07` → `VD7`, `AS01` → `AS1`) so it matches canonical naming, and filters out non-neuron cells (muscles `dBWML*`/`vm*`, glia `GLR*`/`CEPsh*`).
- `validate.py` — `validate_neuron_count`, `validate_known_pathways`, `cross_validate(primary, secondary) -> DivergenceReport`.
- `smoke.py` — `run_forward_pass(connectome, *, seed=0)` runs a single random-weight forward pass through the connectome topology with `tanh((W_chem + W_gap).T @ x)`, returning motor-neuron rows. Chemical weights are strict-masked + fan-in-normalised; gap junctions use Cook 2019 junction counts as fixed weights, also fan-in-normalised for scale comparability.
- `__init__.py` — public API re-exports.

`packages/quantum-nematode/tests/quantumnematode_tests/connectome/` ships five test files with **71 tests total**, all passing in the default tier.

## Cook 2019 import findings

After filtering to the 302-neuron subset (dropping muscles, glia, end-organs, and the male-only sex sheet):

| Metric | Observed | Note |
|---|---|---|
| Neurons | **302** | Matches expected hermaphrodite count exactly |
| Chemical synapses (neuron→neuron) | **3709** | Loose lower-bound check: > 3000. Project docs cite "~7000 chemical synapses" but that figure includes synapses onto muscles + glia + end-organs which our loader filters out |
| Gap junctions (canonical, undirected) | **1093** | After merging the `herm gap jn symmetric` and `herm gap jn asymmetric` sheets and reducing to canonical `(neuron_a < neuron_b)` form. Project docs cite "~900 gap junctions" |
| Chemical self-loops | 38 | The Cook 2019 hermaphrodite-chemical adjacency matrix has non-zero diagonal entries for 38 neurons. These represent biologically-real axodendritic self-synapses + data-collection artefacts; our model retains them as `ChemicalSynapse(pre=X, post=X)`. No filtering applied at L0 |

### Sheet structure observation worth recording

Cook 2019 SI 5 has SEVEN sheets, not four:

- `hermaphrodite chemical`
- `herm gap jn symmetric` (split from the original "hermaphrodite gap jn" naming the planning artefacts assumed)
- `herm gap jn asymmetric`
- `male chemical`
- `male gap jn symmetric`
- `male gap jn asymmetric`
- `TITLE AND LEGEND`

The hermaphrodite loader reads three sheets (chemical + sym gap + asym gap), merges the two gap-junction sheets in canonical form, and ignores the male sheets entirely at L0. Future male / cross-sex work would be a straightforward additional loader function.

## Witvliet 2021 import findings

| Metric | Observed | Note |
|---|---|---|
| Neurons | **180** | Within the expected nerve-ring range (150-200 of the 302 whole-animal neurons) |
| Chemical synapses | **1933** | Long-format edge list, not an adjacency matrix |
| Gap junctions (canonical) | **296** | After canonicalisation |

Witvliet uses a different file format (long-format edge list with `pre`/`post`/`type`/`weight` columns) from Cook 2019 (adjacency matrix). The loader uses case-insensitive column-name sniffing to absorb the dataset's exact header naming.

**File naming note**: the project refers to Witvliet 2021 (the *Nature* publication year), but cect's vendored filename uses `witvliet_2020_` (the preprint year). The vendored filename preserves cect's convention for traceability against the upstream mirror; PROVENANCE.md documents this.

## Cross-validation findings (Cook 2019 vs Witvliet 2021)

`cross_validate(cook_2019_hermaphrodite, witvliet_2021_adult)` produced:

| Field | Value |
|---|---|
| `shared_neurons` | 180 (the full Witvliet nerve-ring subset is contained in Cook 2019's 302) |
| `shared_pairs_agreement` | 1271 (both datasets report — or both don't report — a chemical synapse) |
| `shared_pairs_disagreement` | 1942 |
| `primary_only_pairs` | many (Cook covers the whole animal; Witvliet covers only the nerve ring) |
| `secondary_only_pairs` | many (developmental-stage / lineage-tracing differences) |
| `weight_divergence_summary` | `{n_pairs: 1271, mean: 0.86, median: 0.57}` (ratio of secondary-weight ÷ primary-weight for shared pairs) |

The mean-vs-median gap (0.86 mean, 0.57 median) tells us the weight-ratio distribution is right-skewed: most pairs that exist in both datasets have *lower* weight in Witvliet 2021 than in Cook 2019, which is biologically reasonable (Witvliet 2021's adult is one specific worm; Cook 2019 aggregates across multiple animals + supplements with extrapolation per the SI's own documentation).

These divergence statistics are NOT validation failures — they're the expected outcome of comparing a whole-animal aggregate connectome against a single-animal nerve-ring slice. The `DivergenceReport` is designed for forensic inspection and downstream-consumer interpretation, not for a binary pass/fail gate.

## Forward-pass smoke result

`uv run python -m quantumnematode.connectome.smoke` on the Cook 2019 hermaphrodite connectome (seed=0):

- Loaded: 302 neurons, 3709 chemical synapses, 1093 gap junctions.
- Output: shape `(116,)` (one entry per motor neuron — 116 of 302), range `[-1.0, 1.0]`, variance 0.687974.
- **PASS**: output is finite and non-degenerate.

The fan-in-normalised initialisation (chemical weights sampled from `N(0, 1/sqrt(fan_in))` per postsynaptic neuron) successfully avoids tanh saturation — the output spans the full `tanh` range with healthy variance rather than clumping at ±1. This matters because Cook 2019 has neurons with up to ~50 chemical inputs; an unnormalised `N(0, 1)` init would push every output to ±1.

## T1↔T2 API sketch (handshake for the plugin-design tranche)

The next tranche's plugin design must consume the connectome through this API surface. The signatures + dataclass shapes below are the contract; if the plugin design finds an API mismatch, this tranche amends rather than the plugin tranche working around it.

### Public method signatures

```python
from quantumnematode.connectome import (
    # Data model
    CellClass,           # Literal["sensory", "interneuron", "motor", "muscle", "pharyngeal"]
    Neuron,              # Pydantic BaseModel
    ChemicalSynapse,     # Pydantic BaseModel
    GapJunction,         # Pydantic BaseModel
    Connectome,          # Pydantic BaseModel

    # Loaders
    load_cook_2019_hermaphrodite,   # () -> Connectome  (302 neurons)
    load_witvliet_2021_adult,       # () -> Connectome  (180 neurons, nerve ring)

    # Validators
    ValidationResult,
    DivergenceReport,
    validate_neuron_count,          # (Connectome, *, expected=302) -> ValidationResult
    validate_known_pathways,        # (Connectome) -> ValidationResult
    cross_validate,                 # (primary, secondary, *, list_cap=50) -> DivergenceReport
)
```

### Dataclass shapes

```python
class Neuron:
    name: str                       # canonical name (e.g. "ASEL", "AVAL", "VB2")
    cell_class: CellClass
    neurotransmitter: str | None    # primary NT when known, None otherwise (sparse for now)

class ChemicalSynapse:
    pre: str                        # presynaptic neuron name
    post: str                       # postsynaptic neuron name
    weight: int                     # Cook 2019 EM-derived synapse count; gt=0

class GapJunction:
    neuron_a: str                   # canonical: neuron_a < neuron_b (lex)
    neuron_b: str
    weight: int                     # ge=0 (rarely zero, but model allows it)

class Connectome:
    neurons: dict[str, Neuron]      # 302 entries for Cook 2019 hermaphrodite
    chemical_synapses: list[ChemicalSynapse]   # sorted by (pre, post)
    gap_junctions: list[GapJunction]           # sorted by (neuron_a, neuron_b)
    source: str                     # e.g. "cook_2019_hermaphrodite"
    version: str                    # vendored snapshot identifier
```

### Iteration patterns for plugin consumers

A future plugin needs to consume both connection types separately:

```python
c = load_cook_2019_hermaphrodite()

# Build a chemical-synapse weight mask (strict-mask: non-existent edges = 0)
for syn in c.chemical_synapses:
    pre_idx = neuron_indices[syn.pre]
    post_idx = neuron_indices[syn.post]
    chemical_mask[pre_idx, post_idx] = 1.0
    chemical_initial_weight[pre_idx, post_idx] = ...   # plugin's init policy

# Build a gap-junction fixed-weight matrix (Cook 2019 counts, symmetric)
for gj in c.gap_junctions:
    a_idx = neuron_indices[gj.neuron_a]
    b_idx = neuron_indices[gj.neuron_b]
    gap_weight[a_idx, b_idx] = float(gj.weight)
    gap_weight[b_idx, a_idx] = float(gj.weight)

# Look up cell class for sensor / motor routing
for name, neuron in c.neurons.items():
    if neuron.cell_class == "sensory":
        ...
    elif neuron.cell_class == "motor":
        ...
```

The `smoke.run_forward_pass()` source is the worked example for both iteration patterns — plugin designers can read it as a reference implementation.

### Explicit non-scope (what the API does NOT provide)

- **Position / morphology data** for neurons. Cell-body coordinates aren't in the loaded `Neuron` model. The `Neuron` shape can be extended later if a use case needs it (positions are in WormAtlas).
- **Peptidergic / extra-synaptic edges**. Not represented; modelling diffusible neuromodulator concentrations requires receptor-class metadata and diffusion dynamics that aren't built.
- **Sensor-projection mapping** (e.g. ASE → food-chemotaxis observation, AFD → thermotaxis observation). The connectome surface exposes neuron-name-to-cell-class lookup; how environmental observations route to named sensory neurons is the plugin's design decision.
- **Motor-readout mapping** (e.g. how VB1-VB11 + DB1-DB7 outputs combine into a discrete action choice or continuous action vector). Same as above — plugin's call.

## Verdict

L0 ships clean. No STOP signals: substrate loads, validators pass, smoke forward pass is finite + non-degenerate, cross-validation produces meaningful divergence map.

Substrate integrity check (G1.a evidence for the eventual mid-phase gate decision):

- ✅ Connectome loaded via direct *Nature* SI parsing (decision: drop the originally-planned third-party API library)
- ✅ Neuron count = 302
- ✅ Chemical-synapse count 3709 (> 3000 lower bound)
- ✅ Gap-junction count 1093 (> 600 lower bound)
- ✅ Cross-validation against Witvliet 2021 nerve-ring shipped with DivergenceReport
- ✅ Three canonical klinotaxis / thermotaxis / nociception pathways trace successfully through the chemical adjacency
- ✅ Forward-pass smoke test passes with non-degenerate variance

T1.8 API sketch above is the handshake for the next tranche. The plugin design picks it up from here.
