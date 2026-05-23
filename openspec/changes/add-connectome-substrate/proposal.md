## Why

Phase 6 (Connectome Substrate & Architecture Comparison) opens with Tranche 1 (T1, L0 layer): import the real *C. elegans* connectome into the codebase, validate it against an independent reference, and produce the data-model API that every Phase 6 architecture plugin will consume. Without T1, the headline Phase 6 platform claim — *first closed-loop learning + evolution on the real C. elegans connectome with a pluggable architecture interface* — has no substrate to operate on.

This change is scoped *narrowly* to L0. No plugin work (L1 / T2), no PPO-on-connectome training (T2.6), no env wiring. The substrate is loaded, validated, vendored, and forward-passable on a synthetic input — and that's it.

The narrow scope is deliberate. The roadmap's documented L0 Risk-mitigation pivot ("if c302 import takes > 2 months → drop to hand-curated subset of Cook 2019, ~50-100 neurons") needs its own evidence and its own decision artefact. Bundling L0 with the L1 plugin refactor or with the first PPO-on-connectome training run muddies the pivot signal: a failure could be the substrate, the plugin interface, or the training loop. Keeping T1 to "data in, validated, forward-passable" surfaces substrate-import diagnoses cleanly.

Five Phase 6 design decisions from [phase6-tracking/design.md](../phase6-tracking/design.md) directly shape this change's scope:

- **Decision 2 (data source)**: Cook 2019 hermaphrodite is the L0 primary dataset. The roadmap mentioned OpenWorm `cect` (ConnectomeToolbox) as a candidate ingestion library; investigation during this change's planning surfaced that (a) cect's licence is inconsistent across `LICENSE` (MIT) and `setup.cfg` (LGPLv3), (b) it is a pre-1.0 (v0.3.1) project with a small maintainer base, and (c) its bundled data files are themselves the *Nature* Supplementary Information from Cook 2019 and Witvliet 2021. We therefore consume the *Nature* SI files **directly** with pandas rather than taking a niche third-party connectome-package dependency. This drops the integration risk and matches the spirit of phase6-tracking Decision 2 (Cook 2019 is the data; cect was always interchangeable as the API layer).
- **Decision 7 (connection-type taxonomy)**: chemical synapses (directed, weighted by synapse count) and gap junctions (undirected, fixed-weight = Cook 2019 count) MUST be **separately-typed connections** in the data model — including for the dual-edge case (e.g. AVA↔AVB has one chemical-synapse edge AND one gap-junction edge, not one edge with two weight attributes). Extra-synaptic / peptidergic signalling is explicitly out of scope (reserved for Phase 7 L4 plasticity).
- **Decision 1 (tranche ordering)**: T1 publishes a signature-level data-model API sketch as the T1↔T2 handshake; T2's plugin Protocol design consumes it.
- **Decision 6 (Gate 1)**: T1 produces G1.a evidence — neuron count loaded, cross-validation against Witvliet 2021 nerve-ring subset shipped. The full Gate 1 decision lands at T2 close.
- **What this change does NOT decide**: the L1 registry implementation pattern, the continuous-action policy parameterisation, the real-worm validation dataset selection — all deferred to their respective tranches.

## What Changes

### 1. New `quantumnematode.connectome` Subpackage

Six files under `packages/quantum-nematode/quantumnematode/connectome/`:

- `__init__.py` — public API surface for downstream consumers (T2 plugin design consumes this)
- `model.py` — typed data model: `Neuron`, `ChemicalSynapse`, `GapJunction`, `Connectome`. Per Decision 7, chemical synapses and gap junctions are **separately-typed connection categories**; the dual-edge case (AVA↔AVB) is represented as two distinct edge entries with type metadata. Neuron `cell_class` is `Literal["sensory","interneuron","motor","muscle","pharyngeal"]` derived from Cook 2019 SI 1 cell-list metadata
- `neurons.py` — hand-curated 302-neuron classification table sourced from Cook 2019 SI 1 cell-list + cross-checked against WormAtlas (wormatlas.org). Static dict mapping neuron name → `cell_class` + `neurotransmitter`. Becomes the canonical project reference for *C. elegans* neuron identities for the rest of Phase 6+
- `loader.py` — `load_cook_2019_hermaphrodite()` and `load_witvliet_2021_adult()` returning `Connectome` instances. Reads the vendored XLSX files directly via pandas + openpyxl; constructs `Connectome` instances using the neuron classification dict in `neurons.py`. Pure stdlib + pandas — no third-party connectome dependency
- `validate.py` — neuron-count check (302 hermaphrodite), known-pathway check (≥ 1 of three canonical sensory → interneuron → motor pathways from the Bargmann lab klinotaxis / thermotaxis / nociception literature: ASE → AIY → RIA → SMD, AFD → AIY → RIA → SMD, or ASH → AVA → VA/DA), cross-validation between Cook 2019 and Witvliet 2021 nerve-ring subset returning a `DivergenceReport`
- `smoke.py` — instantiate a trivial PPO weight tensor over the connectome's chemical synapses (strict-mask: weights pinned to zero outside the wild-type adjacency), run a single forward pass on synthetic input, verify output shape + no NaNs/Infs. Per Decision 7, gap junctions participate at their fixed Cook 2019 counts in the forward pass

The forward-pass smoke check uses PPO-shaped weight matrices but does NOT instantiate a `Brain` Protocol implementation, does NOT register a new `BrainType` enum value, and does NOT touch the env. It validates that the data model can support a learned-weight forward computation; the actual plugin wiring is T2's job.

### 2. Vendored Connectome Data (Cook 2019 + Witvliet 2021 Nature Supplementary Information)

- `data/connectome/cook_2019_si5_connectome_adjacency.xlsx` — Cook 2019 SI 5 "Connectome adjacency matrices" XLSX, source for chemical-synapse and gap-junction adjacencies
- `data/connectome/cook_2019_si1_cell_list.xlsx` — Cook 2019 SI 1 cell-list table, source for the neuron classification dict in `neurons.py`
- `data/connectome/witvliet_2021_dataset8_adult.xlsx` — adult worm (dataset 8) from the Witvliet 2021 developmental connectome series, used for T1.4 nerve-ring cross-validation
- `data/connectome/PROVENANCE.md` — per-file: source URL (Nature SI direct link), DOI of the paper the SI accompanies, SHA256, retrieval date, citation, redistribution rationale (academic research re-use of *Nature* SI is standard practice; we cite the paper)

### 3. Two New Standard Dependencies (No Third-Party Connectome Package)

Add `pandas>=2.2` and `openpyxl>=3.1` to `packages/quantum-nematode/pyproject.toml`. Neither is currently in the dependency tree — verified against both `pyproject.toml` files and `uv.lock`. These are standard, well-maintained packages (not niche connectome-domain libraries) needed to parse the vendored XLSX files. No third-party connectome-domain package (cect, wormneuroatlas, etc.) is added — see [design.md § Decision T1.1](design.md) for the rationale.

### 4. `.gitattributes` LFS Rules

Add `data/connectome/**/*.xlsx filter=lfs diff=lfs merge=lfs -text` to the existing LFS-tracking block. The current `.gitattributes` only covers `artifacts/`, `benchmarks/`, and `configs/`; the three Cook 2019 + Witvliet 2021 SI files are several MB each and need LFS.

### 5. Tests

New directory `packages/quantum-nematode/tests/quantumnematode_tests/connectome/`:

- `test_loader.py` — Cook 2019 + Witvliet 2021 load correctly; neuron count = 302; expected sensory neurons (ASE, AFD, ASH, ADL, AWA, AWC, URX, BAG) present with correct cell-class labels; expected motor-neuron classes (VB / DB / VA / DA / VC / DD) present
- `test_model.py` — data-model invariants: no orphan synapses (every edge's pre/post neuron exists in `neurons`); chemical synapses and gap junctions are separately-iterable; the AVA↔AVB dual-edge case is represented as two distinct entries (one `ChemicalSynapse`, one `GapJunction`), not one entry; chemical-synapse weights are positive integers; gap-junction weights are non-negative integers
- `test_neurons.py` — neuron-classification table has 302 entries; every entry has a valid `cell_class`. Coverage-by-class is NOT band-asserted in tests (boundaries are convention-dependent across Cook 2019 SI 1 / WormAtlas / project docs); the test prints class counts for forensic review in the T1 logbook
- `test_validate.py` — neuron-count validator flags an artificially-broken (e.g. 301-neuron) connectome; known-pathway validator passes if ≥ 1 of three Bargmann-lab canonical pathways traces successfully (klinotaxis ASE → AIY → RIA → SMD, thermotaxis AFD → AIY → RIA → SMD, or nociception ASH → AVA → VA/DA); `cross_validate(cook_2019, witvliet_2021_adult)` produces a `DivergenceReport` with non-empty agreement set and documented divergence map
- `test_smoke.py` — `smoke.run_forward_pass()` returns finite output of expected shape; output has non-zero variance across motor-neuron rows (catches degenerate constants AND fully-saturated outputs); raises if connectome's chemical-synapse adjacency is zero-dense (sanity guard)

### 6. T1 Logbook + T1↔T2 API Sketch

- `docs/experiments/logbooks/022-connectome-substrate.md` — T1 logbook: implementation summary, data-model decisions, cross-validation findings, any divergences from Cook 2019 documented neuron counts (G1.a evidence), T1.4 divergence map summary, and the T1↔T2 signature-level API sketch (per phase6-tracking T1.8). The sketch lists the public method signatures + key dataclass shapes that T2's plugin Protocol design will consume

### 7. Tracking + Roadmap Updates

- `openspec/changes/phase6-tracking/tasks.md` — tick T1.1 through T1.10; flip Tranche 1 status to `🟡 in progress` on first commit and `✅ complete` on PR merge
- `docs/roadmap.md` Phase 6 Tranche Tracker — flip T1 row to `🟡 in progress` / `✅ complete` accordingly

## Capabilities

**Added**: `connectome-substrate` (new) — five requirements covering: data loading from vendored *Nature* SI files, the connection-type taxonomy (chemical-synapse vs gap-junction separately-typed, AVA↔AVB-style dual-edge case represented as two entries), the neuron metadata surface (302-entry classification table sourced from Cook 2019 SI 1), cross-validation against Witvliet 2021 nerve-ring subset, and the smoke-test forward pass.

## Impact

**Code:**

- `packages/quantum-nematode/quantumnematode/connectome/{__init__,model,neurons,loader,validate,smoke}.py` — new subpackage

**Data:**

- `data/connectome/cook_2019_si5_connectome_adjacency.xlsx` — vendored Nature SI, LFS-tracked
- `data/connectome/cook_2019_si1_cell_list.xlsx` — vendored Nature SI, LFS-tracked
- `data/connectome/witvliet_2021_dataset8_adult.xlsx` — vendored Nature SI, LFS-tracked
- `data/connectome/PROVENANCE.md` — provenance record

**Configs:** None.

**Dependencies:** Two standard packages added to `packages/quantum-nematode/pyproject.toml`: `pandas>=2.2` and `openpyxl>=3.1`. No third-party connectome-domain package added.

**Tests:**

- `packages/quantum-nematode/tests/quantumnematode_tests/connectome/{test_loader,test_model,test_neurons,test_validate,test_smoke}.py` — new test directory

**Docs:**

- `docs/experiments/logbooks/022-connectome-substrate.md` — T1 logbook + T1↔T2 API sketch
- `openspec/changes/phase6-tracking/tasks.md` — T1.x sub-tasks ticked
- `docs/roadmap.md` Phase 6 Tranche Tracker — T1 row updated

**Git:**

- `.gitattributes` — LFS rule added for `data/connectome/**/*.xlsx`

## Untouched (Tranche 1 boundary — these are T2 territory)

- `packages/quantum-nematode/quantumnematode/brain/arch/_brain.py` — `Brain` Protocol surface
- `packages/quantum-nematode/quantumnematode/utils/brain_factory.py` — `setup_brain_model()` dispatcher
- `packages/quantum-nematode/quantumnematode/dtypes.py` — `BrainType` enum
- `packages/quantum-nematode/quantumnematode/env/env.py` — env / sensor / motor wiring

## Breaking Changes

None. The change is purely additive — a new subpackage, new tests, new vendored data. No existing module is modified beyond `.gitattributes` (one new LFS rule).

## Backward Compatibility

Full. No existing entry-point, config, or behaviour is altered. The subpackage's public API is consumed only by its own tests in this tranche; T2 will be the first non-test consumer.
