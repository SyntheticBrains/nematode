# Tasks: add-connectome-substrate

Phase 6 Tranche 1 (L0). Sub-task numbering matches `openspec/changes/phase6-tracking/tasks.md` § Tranche 1 (T1.1–T1.10). Each Tn.x in *this* file maps 1:1 to T1.x in the tracker; ticking here MUST be reflected by ticking the matching T1.x in `phase6-tracking/tasks.md` as part of the same PR (per `phase6-tracking/specs/phase6-tracking/spec.md` § Requirement 1, "Phase 6 Living Tranche Checklist" — milestone PR updates the checklist).

## Phase 1 — OpenSpec scaffold (this commit)

- [x] 1.1 Create `openspec/changes/add-connectome-substrate/proposal.md`
- [x] 1.2 Create `openspec/changes/add-connectome-substrate/design.md`
- [x] 1.3 Create `openspec/changes/add-connectome-substrate/tasks.md` (this file)
- [x] 1.4 Create `openspec/changes/add-connectome-substrate/specs/connectome-substrate/spec.md`
- [x] 1.5 `openspec validate add-connectome-substrate --strict` clean
- [x] 1.6 Add `pandas>=2.2` and `openpyxl>=3.1` to `packages/quantum-nematode/pyproject.toml` `dependencies` list (neither currently in the dep tree; verified against both pyproject.toml files and uv.lock). Run `uv sync` to regenerate `uv.lock`. Resolved versions: `pandas==3.0.3`, `openpyxl==3.1.5`

## Phase 2 — Data vendoring (T1.1 + T1.5)

Maps to `phase6-tracking/tasks.md` T1.1 (import-library + data-source decision, recording the chosen approach) + T1.5 (vendor + provenance — full scope, not partial).

- [ ] 2.1 Create `data/connectome/` directory
- [ ] 2.2 Vendor `data/connectome/cook_2019_si5_connectome_adjacency.xlsx` (Cook 2019 *Nature* SI 5)
- [ ] 2.3 Vendor `data/connectome/cook_2019_si1_cell_list.xlsx` (Cook 2019 *Nature* SI 1 cell list)
- [ ] 2.4 Vendor `data/connectome/witvliet_2021_dataset8_adult.xlsx` (Witvliet 2021 dataset 8 adult)
- [ ] 2.5 Update `.gitattributes` to add `data/connectome/**/*.xlsx filter=lfs diff=lfs merge=lfs -text`
- [ ] 2.6 Verify each vendored file is LFS-tracked (`git check-attr filter <path>` reports `lfs`); regenerate index if needed
- [ ] 2.7 Write `data/connectome/PROVENANCE.md` with per-file: source URL, DOI, SHA256, retrieval date, citation, redistribution rationale. Note the original cect filenames (e.g. `witvliet_2020_8 adult.xlsx`) for traceability against the vendored normalised names (e.g. `witvliet_2021_dataset8_adult.xlsx`)
- [ ] 2.8 Closes T1.1 + T1.5 — tick the matching T1.1 + T1.5 boxes in `phase6-tracking/tasks.md`

## Phase 3 — Data model (T1.2)

Maps to T1.2 in the tracker: connectome data model exposing chemical synapses + gap junctions as separately-typed connections per phase6-tracking Decision 7.

- [ ] 3.1 Create `packages/quantum-nematode/quantumnematode/connectome/__init__.py` with explicit named re-exports matching project convention (e.g. `from .model import Neuron, ChemicalSynapse, GapJunction, Connectome, CellClass`; no wildcard imports)
- [ ] 3.2 Implement `connectome/model.py` per design.md Decision T1.2: pydantic `Neuron`, `ChemicalSynapse`, `GapJunction`, `Connectome`; `CellClass = Literal["sensory", "interneuron", "motor", "muscle", "pharyngeal"]`
- [ ] 3.3 Add field-level validation: chemical-synapse `weight: int = Field(gt=0)`; gap-junction `weight: int = Field(ge=0)`; neuron-set integrity (every synapse `pre`/`post`/`neuron_a`/`neuron_b` exists in `neurons` dict)
- [ ] 3.4 Implement gap-junction canonical-form convention (alphabetically sorted `neuron_a < neuron_b`) to deduplicate `AVA-AVB` and `AVB-AVA` reports
- [ ] 3.5 Closes T1.2 — tick the matching T1.2 box in `phase6-tracking/tasks.md`

## Phase 4 — Neuron classification table (supports T1.2 + T1.3)

302-neuron classification extracted from Cook 2019 SI 1 cell-list table. Static dict shipped as code per design.md Decision T1.4.

- [ ] 4.1 Extract neuron classification from `cook_2019_si1_cell_list.xlsx` into a working CSV (one-time data-prep step; CSV is intermediate, not committed)
- [ ] 4.2 Cross-check classifications against WormAtlas; resolve any disagreement by deferring to Cook 2019 SI 1 with a comment noting WormAtlas variants
- [ ] 4.3 Implement `connectome/neurons.py` with `NEURON_CLASSIFICATION: dict[str, tuple[CellClass, str | None]]` covering all 302 hermaphrodite neurons
- [ ] 4.4 Add `CANONICAL_NAME_ALIASES: dict[str, str]` for cross-dataset name normalisation (e.g. `RIA-L` → `RIAL`) per design.md Risks #2
- [ ] 4.5 Coverage assertion in module: assert `len(NEURON_CLASSIFICATION) == 302` at import time (fail-fast if a future edit drops an entry)

## Phase 5 — Loader (T1.3)

Maps to T1.3 in the tracker: import the Cook 2019 hermaphrodite connectome; verify counts.

- [ ] 5.1 Implement `connectome/loader.py:load_cook_2019_hermaphrodite()` reading the vendored SI 5 XLSX via `pd.read_excel(..., engine="openpyxl")`; iterate the two relevant sheets (`hermaphrodite chemical` + `hermaphrodite gap jn`) per design.md Decision T1.3
- [ ] 5.2 Map XLSX row/column headers to neuron names; cross-reference against `NEURON_CLASSIFICATION` from Phase 4
- [ ] 5.3 Emit `ChemicalSynapse` and `GapJunction` entries from non-zero cells in each sheet; populate `Connectome(neurons=..., chemical_synapses=..., gap_junctions=..., source=..., version=...)`
- [ ] 5.4 Implement `connectome/loader.py:load_witvliet_2021_adult() -> Connectome` reading `witvliet_2021_dataset8_adult.xlsx` (single adult dataset is sufficient for T1.4 nerve-ring cross-validation); apply `CANONICAL_NAME_ALIASES` for cross-dataset name normalisation
- [ ] 5.5 Loader produces deterministic output: chemical synapses sorted by `(pre, post)`; gap junctions sorted by `(neuron_a, neuron_b)`; neurons in canonical order. Determinism matters for test stability + future caching
- [ ] 5.6 Closes T1.3 — tick the matching T1.3 box in `phase6-tracking/tasks.md`

## Phase 6 — Validation suite (T1.4)

Maps to T1.4 in the tracker: cross-validation against Witvliet 2021 nerve-ring subset.

- [ ] 6.1 Implement `connectome/validate.py:validate_neuron_count(c) -> ValidationResult` — expects 302 for hermaphrodite
- [ ] 6.2 Implement `validate_known_pathways(c) -> ValidationResult` — confirms at least one of three canonical sensory → interneuron → motor pathways from the Bargmann lab klinotaxis / thermotaxis / nociception literature is present: ASE → AIY → RIA → SMD (klinotaxis; Gray et al. 2005, Iino & Yoshida 2009), AFD → AIY → RIA → SMD (thermotaxis), or ASH → AVA → VA/DA (nociception). Validator passes if ≥ 1 of the three pathways traces successfully through the connectome
- [ ] 6.3 Implement `cross_validate(primary, secondary) -> DivergenceReport` per design.md Decision T1.6: intersection of neurons; agreement / disagreement per shared (pre, post) pair; cook-only-pairs and witvliet-only-pairs; weight-divergence summary
- [ ] 6.4 Implement `connectome/validate.py:DivergenceReport` pydantic model (fields per design.md Decision T1.6)
- [ ] 6.5 Closes T1.4 — tick the matching T1.4 box in `phase6-tracking/tasks.md`

## Phase 7 — Smoke-test forward pass (T1.6)

Maps to T1.6 in the tracker: PPO-shaped forward pass on the connectome topology.

- [ ] 7.1 Implement `connectome/smoke.py:run_forward_pass(c, *, seed=0) -> np.ndarray` per design.md Decision T1.5: build N×N chemical-synapse adjacency with random weights sampled from `N(0, 1/sqrt(fan_in))` where `fan_in` is the in-degree of each postsynaptic neuron (prevents tanh saturation given Cook 2019 has up to ~50 chemical inputs per neuron); non-existent edges pinned to zero (strict-mask). N×N gap-junction adjacency uses Cook 2019 counts as fixed weights
- [ ] 7.2 Forward pass: `output = tanh((chemical_W + gap_W) @ input_x)`; return `output[motor_neuron_rows]`
- [ ] 7.3 Sanity guard: raise if chemical-synapse adjacency has zero non-zero entries (catches silent load failures)
- [ ] 7.4 Module exposes a way for tests to assert output has non-zero variance across motor-neuron rows (catches both degenerate constants AND fully-saturated ±1 outputs); the test itself lives in Phase 8
- [ ] 7.5 Add `if __name__ == "__main__"` block so `uv run python -m quantumnematode.connectome.smoke` runs the forward pass and prints output shape + finite-value assertion result. This is the executable smoke-test from design.md Verification §
- [ ] 7.6 Closes T1.6 — tick the matching T1.6 box in `phase6-tracking/tasks.md`

## Phase 8 — Tests + CI integration (T1.7)

Maps to T1.7 in the tracker.

- [ ] 8.1 Create `packages/quantum-nematode/tests/quantumnematode_tests/connectome/__init__.py`
- [ ] 8.2 Implement `test_loader.py`: Cook 2019 loads with neuron count = 302; chemical-synapse count > 5000 (loose lower-bound sanity check — the Cook 2019 paper reports ~7000 chemical synapses per `docs/nematode_biology.md:644`; the exact count depends on edge-collation conventions); gap-junction count > 600 (loose bound vs ~900 documented); Witvliet 2021 adult loads with reduced neuron count (~150-200 nerve-ring); known sensory neurons (ASEL, ASER, AFDL, AFDR, ASHL, ASHR, ADLL, ADLR, AWAL, AWAR, AWCL, AWCR, URXL, URXR, BAGL, BAGR) present with `cell_class == "sensory"`; known motor neuron classes (VB, DB, VA, DA, VC, DD) present with `cell_class == "motor"`
- [ ] 8.3 Implement `test_model.py`: pydantic field validation rejects zero-weight chemical synapses; AVA↔AVB dual-edge case present as one `ChemicalSynapse` AND one `GapJunction` (not one combined entry — per phase6-tracking Decision 7); no orphan synapses (every edge's pre/post/neuron_a/neuron_b exists in `neurons`)
- [ ] 8.4 Implement `test_neurons.py`: `len(NEURON_CLASSIFICATION) == 302`; every entry has a valid `CellClass`. Coverage-by-class is NOT band-asserted (boundaries are convention-dependent — Cook 2019 SI 1, WormAtlas, and project docs use different rules for polymodal cells); test instead prints class counts for forensic review in the T1 logbook
- [ ] 8.5 Implement `test_validate.py`: `validate_neuron_count` flags 301-neuron broken connectome; `validate_known_pathways` passes (≥ 1 of klinotaxis ASE → AIY → RIA → SMD, thermotaxis AFD → AIY → RIA → SMD, or nociception ASH → AVA → VA/DA traces successfully); `cross_validate(cook_2019_hermaphrodite, witvliet_2021_dataset8_adult)` produces non-empty agreement set and documented divergence map
- [ ] 8.6 Implement `test_smoke.py`: `run_forward_pass` returns finite output of expected motor-neuron shape; output has non-zero variance across motor-neuron rows (catches degenerate constants AND fully-saturated outputs); raises on a deliberately-emptied connectome
- [ ] 8.7 Tests run in default pytest tier (`uv run pytest -m "not nightly"` includes them). LFS fetch happens before tests in CI via `git lfs pull` or implicit smudge — verify
- [ ] 8.8 Closes T1.7 — tick the matching T1.7 box in `phase6-tracking/tasks.md`

## Phase 9 — T1↔T2 API sketch (T1.8)

Per phase6-tracking T1.8: publish a signature-level API sketch in the T1 logbook for T2 plugin-design review.

- [ ] 9.1 In the T1 logbook (created in Phase 10), include an "API sketch for T2 consumers" section with:
  - Public method signatures: `load_cook_2019_hermaphrodite() -> Connectome`, `load_witvliet_2021_adult() -> Connectome`, `cross_validate(primary, secondary) -> DivergenceReport`, `run_forward_pass(c, *, seed=0) -> np.ndarray`
  - Dataclass shapes for `Neuron`, `ChemicalSynapse`, `GapJunction`, `Connectome`
  - Iteration patterns for T2 consumers: how to iterate chemical synapses vs gap junctions separately; how to look up neuron `cell_class`; how strict-mask is applied
  - Explicit non-scope: position/morphology, peptidergic edges, sensor-projection mapping, motor-readout mapping
- [ ] 9.2 Closes T1.8 — tick the matching T1.8 box in `phase6-tracking/tasks.md`

## Phase 10 — T1 logbook (T1.10)

Maps to T1.10 in the tracker: publish T1 logbook feeding Gate 1's evidence base.

- [ ] 10.1 Create `docs/experiments/logbooks/022-connectome-substrate.md` (logbook 022, immediately following 021 Phase 5 synthesis)
- [ ] 10.2 Sections: Implementation summary; Data-source decision (the cect / wormneuroatlas / direct-SI investigation); Cook 2019 import findings (neuron count, chemical-synapse count, gap-junction count vs published bounds); T1.4 cross-validation divergence summary; T1↔T2 API sketch (per Phase 9 above); Verdict — does T1 ship "real connectome loaded, validated, vendored, forward-passable" with no STOP signals
- [ ] 10.3 Logbook is Gate 1 G1.a evidence per phase6-tracking design.md § Decision 6 § Gate 1. Full Gate 1 decision lands at T2 close
- [ ] 10.4 Closes T1.10 — tick the matching T1.10 box in `phase6-tracking/tasks.md`

## Phase 11 — Tracker + roadmap updates (T1.9)

Maps to T1.9 in the tracker.

- [ ] 11.1 Update `openspec/changes/phase6-tracking/tasks.md` Tranche 1 section: tick all T1.x sub-tasks; flip Tranche 1 status header from `🔲 not started` to `✅ complete`; update the `OpenSpec change` line to point at this change
- [ ] 11.2 Update `docs/roadmap.md` Phase 6 Tranche Tracker — flip T1 row to `✅ complete` with a one-line summary referencing the T1 logbook from Phase 10
- [ ] 11.3 Closes T1.9 — tick the matching T1.9 box in `phase6-tracking/tasks.md`

## Phase 12 — Pre-PR verification

Standard Phase 5+ project convention.

- [ ] 12.1 `openspec validate add-connectome-substrate --strict` clean
- [ ] 12.2 `uv run pre-commit run --files <changed>` clean during iteration
- [ ] 12.3 `uv run pre-commit run -a` clean before push
- [ ] 12.4 `uv run pytest -m "not nightly"` green
- [ ] 12.5 Confirm no `/Users/`, `/home/`, `C:\Users\` paths in any committed file (per `feedback_no_absolute_home_paths_in_commits.md`)
- [ ] 12.6 Confirm vendored XLSX files are LFS-tracked: `git check-attr filter data/connectome/cook_2019_si5_connectome_adjacency.xlsx` reports `lfs` (per `feedback_logbook_artefact_stash_lfs_check.md`)
- [ ] 12.7 Confirm `git diff --stat main` shows only the expected file set; no surprise edits in `packages/quantum-nematode/quantumnematode/brain/arch/`, `utils/brain_factory.py`, `env/env.py`, or `dtypes.py` (these are T2 territory; touching them in T1 would be scope creep)

## Out-of-task actions

These happen after Phase 12 is fully ticked and the user has explicitly approved each step (per `feedback_ask_before_push_and_pr.md`):

- Ask user to push the branch
- Ask user before opening the PR (with conventional-commits title: `feat(connectome-substrate): Add Phase 6 L0 — Cook 2019 connectome ingestion + validation`)
- `openspec archive add-connectome-substrate -y` runs after PR merge — NOT part of this task list (archive happens post-merge on `main`, by convention)
