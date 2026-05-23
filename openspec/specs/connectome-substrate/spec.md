# connectome-substrate Specification

## Purpose

TBD - created by archiving change add-connectome-substrate. Update Purpose after archive.

## Requirements

### Requirement: Connectome Data Loading

The `quantumnematode.connectome` subpackage SHALL load the *C. elegans* connectome from vendored *Nature* Supplementary Information files via direct pandas parsing, without dependency on third-party connectome packages.

#### Scenario: Loading the Cook 2019 hermaphrodite connectome

- **GIVEN** the vendored `data/connectome/cook_2019_si5_connectome_adjacency.xlsx` file is present (LFS-fetched)
- **WHEN** `load_cook_2019_hermaphrodite()` is called
- **THEN** a `Connectome` instance SHALL be returned
- **AND** `len(connectome.neurons) == 302`
- **AND** `len(connectome.chemical_synapses) > 3000` (loose lower bound; the loader filters to the 302-neuron subset and excludes muscles/glia/end-organs that inflate the project-docs ~7000 figure at `docs/nematode_biology.md:644`. Implementation observed: 3709 neuron-to-neuron chemical synapses)
- **AND** `len(connectome.gap_junctions) > 600` (loose lower bound; project docs cite ~900 total. Implementation observed: 1093 canonical gap junctions after merging the symmetric + asymmetric sheets)
- **AND** every entry in `chemical_synapses` and `gap_junctions` references neurons present in the `neurons` dict
- **AND** the `source` field is `"cook_2019_hermaphrodite"` and `version` records the SI 5 publication metadata

#### Scenario: Loading the Witvliet 2021 adult connectome

- **GIVEN** the vendored `data/connectome/witvliet_2020_dataset8_adult.xlsx` file is present
- **WHEN** `load_witvliet_2021_adult()` is called
- **THEN** a `Connectome` instance SHALL be returned
- **AND** `len(connectome.neurons)` is in the nerve-ring range (~150-200, smaller than Cook 2019's whole-animal 302; implementation observed 180)
- **AND** the loader SHALL apply any cross-dataset name aliases via `CANONICAL_NAME_ALIASES` so the result matches Cook 2019's canonical naming. (In the initial vendored snapshot the Witvliet 2021 file uses Cook-compatible names directly, so `CANONICAL_NAME_ALIASES` is empty at L0; the hook is present for future alias-discovery in downstream tranches.)

#### Scenario: Loader produces deterministic output

- **GIVEN** the same vendored data files
- **WHEN** the loader is called twice in succession
- **THEN** both `Connectome` instances SHALL be byte-equivalent (chemical synapses sorted by `(pre, post)`; gap junctions sorted by `(neuron_a, neuron_b)`; neurons in canonical order)

### Requirement: Connection-Type Taxonomy (chemical synapses + gap junctions separately-typed)

Per `openspec/changes/phase6-tracking/design.md` § Decision 7, the connectome data model SHALL expose chemical synapses (directed, learnable weight) and gap junctions (undirected, fixed weight = Cook 2019 count) as separately-typed connection categories. Extra-synaptic / peptidergic signalling SHALL NOT be represented in this data model (reserved for Phase 7 L4 plasticity).

#### Scenario: Iterating chemical synapses

- **GIVEN** a loaded `Connectome` instance
- **WHEN** a downstream consumer iterates `connectome.chemical_synapses`
- **THEN** each entry SHALL be a `ChemicalSynapse` with directed `pre` and `post` fields and a positive-integer `weight` (EM-derived serial-section count from Cook 2019; representing total connectivity including synapse number and size per the SI's documentation). Cook 2019 reports a small number of `pre == post` self-loops on the chemical adjacency diagonal (38 cells in the hermaphrodite); the loader preserves these as `ChemicalSynapse(pre=X, post=X)` entries

#### Scenario: Iterating gap junctions

- **GIVEN** a loaded `Connectome` instance
- **WHEN** a downstream consumer iterates `connectome.gap_junctions`
- **THEN** each entry SHALL be a `GapJunction` with undirected `neuron_a` and `neuron_b` fields (alphabetically sorted: `neuron_a < neuron_b`) and a non-negative-integer `weight` (junction count from Cook 2019)
- **AND** no `(a, b)` and `(b, a)` duplicate pair SHALL exist

#### Scenario: Dual-edge case represented as two distinct entries

This scenario codifies the load-bearing edge case for the connection-type taxonomy: a neuron pair connected by both a chemical synapse AND a gap junction must appear as two distinct entries.

- **GIVEN** Cook 2019 reports BOTH a chemical synapse AND a gap junction between AVAL and AVDL (one of many such pairs in the *C. elegans* connectome; AVAL↔AVAR, AVAR↔AVDR, AVBL↔AVBR are further examples)
- **WHEN** the loaded `Connectome` is inspected
- **THEN** the AVAL↔AVDL pair SHALL appear as `ChemicalSynapse` entries (one per direction; the chemical synapse is directed) AND ONE `GapJunction(neuron_a="AVAL", neuron_b="AVDL", ...)` entry
- **AND** there SHALL NOT be a single edge representation combining both weights into one entry with two weight attributes

#### Scenario: No extra-synaptic / peptidergic edges

- **GIVEN** a loaded `Connectome` instance
- **WHEN** the data model is introspected
- **THEN** there SHALL be no `ExtraSynaptic`, `Peptidergic`, `Monoaminergic`, or similar third connection-type category
- **AND** if a future tranche requires extra-synaptic representation, that work SHALL land as an amendment to this capability and to `phase6-tracking/design.md` § Decision 7

### Requirement: Neuron Metadata (302-entry hand-curated classification table)

The connectome subpackage SHALL ship a 302-neuron classification table (`NEURON_CLASSIFICATION`) derived from OpenWorm cect's MIT-licensed `Cells.py` constants (which themselves attribute Cook 2019 paper + WormAtlas). Cell-class labels are drawn from `Literal["sensory", "interneuron", "motor", "muscle", "pharyngeal"]`. Neurotransmitter identity is included where known.

#### Scenario: Classification table coverage

- **GIVEN** the `connectome.neurons` module is imported
- **WHEN** `NEURON_CLASSIFICATION` is inspected
- **THEN** `len(NEURON_CLASSIFICATION) == 302` (asserted at module import time as a fail-fast guard)
- **AND** every entry's `cell_class` SHALL be a valid `CellClass` value
- **AND** coverage by class SHALL be reported in the T1 logbook as observed counts (no exact-band assertion in the test suite — class-boundary conventions differ between Cook 2019, WormAtlas, and project docs for polymodal / pharyngeal-vs-non-pharyngeal cells)

#### Scenario: Canonical sensory neurons are classified correctly

- **GIVEN** a loaded Cook 2019 `Connectome`
- **WHEN** sensory neurons (ASEL, ASER, AFDL, AFDR, ASHL, ASHR, ADLL, ADLR, AWAL, AWAR, AWCL, AWCR, URXL, URXR, BAGL, BAGR) are looked up via `connectome.neurons[name].cell_class`
- **THEN** every one SHALL be `"sensory"`

#### Scenario: Canonical motor neurons are classified correctly

- **GIVEN** a loaded Cook 2019 `Connectome`
- **WHEN** motor neurons matching the prefix patterns `VB`, `DB`, `VA`, `DA`, `VC`, `DD` are looked up
- **THEN** every one SHALL be `"motor"`

### Requirement: Structural Validators (neuron count + known pathways)

The connectome subpackage SHALL provide structural validators that confirm a loaded `Connectome` matches expected *C. elegans* biology before downstream consumers (T2 plugin design, T4 L2 PPO training) depend on it.

#### Scenario: Neuron count validator

- **GIVEN** a loaded Cook 2019 hermaphrodite `Connectome`
- **WHEN** `validate_neuron_count(c)` is called
- **THEN** the validator SHALL return a `ValidationResult` with `passed == True` if `len(c.neurons) == 302` and `passed == False` otherwise
- **AND** the result's `summary` SHALL describe the actual vs expected counts
- **AND** the result's `details` dict SHALL include the `actual` and `expected` count integers (so callers can react to the verdict without re-parsing the summary string)

#### Scenario: Known klinotaxis / thermotaxis / nociception pathway is present

- **GIVEN** a loaded Cook 2019 `Connectome`
- **WHEN** `validate_known_pathways(c)` is called
- **THEN** the validator SHALL pass if at least one of the following pathways traces successfully through the connectome's chemical synapses: ASE → AIY → RIA → SMD (klinotaxis; Gray et al. 2005, Iino & Yoshida 2009); AFD → AIY → RIA → SMD (thermotaxis); ASH → AVA → VA/DA (nociception)
- **AND** the validator's result SHALL document which pathway(s) were found, for forensic review in the T1 logbook

### Requirement: Cross-Validation Against Witvliet 2021

The connectome subpackage SHALL provide a `cross_validate(primary, secondary) -> DivergenceReport` function that diffs Cook 2019 hermaphrodite against Witvliet 2021 dataset 8 (adult) on the shared nerve-ring neuron subset, producing a divergence map for Gate 1 evidence.

#### Scenario: Cross-validation produces a non-empty agreement set

- **GIVEN** a loaded Cook 2019 hermaphrodite connectome and a loaded Witvliet 2021 dataset 8 connectome
- **WHEN** `cross_validate(cook, witvliet)` is called
- **THEN** a `DivergenceReport` SHALL be returned
- **AND** `report.shared_neurons` SHALL be > 100 (the nerve-ring overlap; Witvliet 2021 covers ~180 neurons mostly within Cook 2019's 302; implementation observed exactly 180)
- **AND** `report.shared_pairs_agreement` SHALL be > 0 (the two datasets agree on at least some shared pairs; implementation observed 1271 agreement / 1942 disagreement)

#### Scenario: Divergence map is documented

The report fields are named generically (`primary_*` / `secondary_*`) so the same type can compare any two `Connectome` instances, not just Cook-vs-Witvliet.

- **GIVEN** a `DivergenceReport` returned from `cross_validate(primary, secondary)`
- **WHEN** the report is inspected
- **THEN** `report.primary_source` and `report.secondary_source` SHALL hold the `source` field strings of the two input connectomes
- **AND** `report.primary_only_pairs` SHALL list chemical-synapse `(pre, post)` pairs that appear in the primary connectome but not in the secondary (capped at `list_cap` entries, default 50)
- **AND** `report.secondary_only_pairs` SHALL list pairs that appear in the secondary but not in the primary (also capped)
- **AND** `report.weight_divergence_summary` SHALL include `n_pairs` always, and `mean` + `median` weight-ratio statistics when `n_pairs > 0`
- **AND** the T1 logbook SHALL include a textual summary of these divergences (e.g. plausible developmental-stage or lineage-tracing causes; implementation observed weight ratio mean 0.86, median 0.57 — right-skewed, consistent with Witvliet's single-animal slice vs Cook's multi-animal aggregate)

### Requirement: Smoke-Test Forward Pass

The connectome subpackage SHALL provide a `run_forward_pass(c, *, seed=0) -> np.ndarray` function that validates the data model can support a learned-weight forward computation. The smoke-test SHALL NOT instantiate any `Brain` Protocol implementation, register a new `BrainType` enum value, or touch the env (those concerns belong to T2 / L1).

#### Scenario: Forward pass on the full Cook 2019 connectome

- **GIVEN** a loaded Cook 2019 hermaphrodite `Connectome`
- **WHEN** `run_forward_pass(c, seed=0)` is called
- **THEN** the function SHALL return a finite `np.ndarray` of shape `(n_motor_neurons,)`
- **AND** the output SHALL NOT contain NaN or Inf values
- **AND** the output SHALL have non-zero variance across the motor-neuron rows (catches both degenerate constant outputs AND fully-saturated ±1 outputs caused by inadequate weight scaling)

#### Scenario: Chemical-synapse strict-mask applied with fan-in-normalised initialisation

- **GIVEN** a loaded `Connectome` with the chemical-synapse adjacency built into a learnable weight matrix `W_chem`
- **WHEN** the forward pass is constructed
- **THEN** every (i, j) pair where Cook 2019 reports no chemical synapse SHALL have `W_chem[i, j] == 0`
- **AND** non-zero entries SHALL be sampled from a seeded random distribution `N(0, 1/sqrt(fan_in))` where `fan_in` is the in-degree of the postsynaptic neuron (deterministic by `seed`; fan-in normalisation prevents tanh saturation given Cook 2019 has up to ~50 chemical inputs per neuron)

#### Scenario: Gap junctions participate at fixed Cook 2019 weights

- **GIVEN** a loaded `Connectome`
- **WHEN** the forward pass is constructed
- **THEN** the gap-junction weight matrix SHALL use Cook 2019 junction counts directly (not learnable scalars, not unit weights — see `phase6-tracking/design.md` § Decision 7)
- **AND** the unnormalised gap-junction weight matrix SHALL be symmetric (gap junctions are undirected). After the same fan-in normalisation that the chemical-synapse matrix uses (per-postsynaptic-column `1/sqrt(in_degree)`), the final per-column-scaled matrix may differ between `W[i,j]` and `W[j,i]` because each column is normalised by its own in-degree; the underlying junction-count adjacency it derives from remains symmetric, and the normalisation is applied consistently with the chemical matrix to keep the two connection types on a comparable scale

#### Scenario: Sanity guard against silent load failure

- **GIVEN** a deliberately-emptied `Connectome` (zero chemical synapses, zero gap junctions)
- **WHEN** `run_forward_pass` is called
- **THEN** the function SHALL raise an exception with a message indicating the connectome is empty
- **AND** the exception SHALL NOT be silently swallowed by returning a degenerate output
