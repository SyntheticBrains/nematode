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

- [x] 2.1 Create `data/connectome/` directory
- [x] 2.2 Vendor `data/connectome/cook_2019_si5_connectome_adjacency.xlsx` (Cook 2019 *Nature* SI 5, ~4.4 MB; source: cect MIT-licensed mirror)
- [x] 2.3 Vendor `data/connectome/witvliet_2020_dataset8_adult.xlsx` (Witvliet 2020/2021 dataset 8 adult, ~53 KB; source: cect MIT-licensed mirror)
- [x] 2.4 Update `.gitattributes` to add `data/connectome/**/*.xlsx filter=lfs diff=lfs merge=lfs -text`
- [x] 2.5 Verify each vendored file is LFS-tracked (`git check-attr filter <path>` reports `lfs`); regenerate index if needed
- [x] 2.6 Write `data/connectome/PROVENANCE.md` with per-file: source URL (cect mirror), DOI of accompanying paper, SHA256, retrieval date, citation, redistribution rationale. Note the original cect filenames (e.g. `witvliet_2020_8 adult.xlsx`) for traceability against the vendored normalised names. Cook 2019 SI 1 is NOT vendored — there is no discrete cell-list XLSX in Cook 2019's published SIs; the 302-neuron classification ships as code in `connectome/neurons.py` derived from cect's MIT-licensed `Cells.py` constants (which themselves attribute Cook 2019 paper + WormAtlas)
- [x] 2.7 Closes T1.1 + T1.5 — tick the matching T1.1 + T1.5 boxes in `phase6-tracking/tasks.md`

## Phase 3 — Data model (T1.2)

Maps to T1.2 in the tracker: connectome data model exposing chemical synapses + gap junctions as separately-typed connections per phase6-tracking Decision 7.

- [x] 3.1 Create `packages/quantum-nematode/quantumnematode/connectome/__init__.py` with explicit named re-exports matching project convention (e.g. `from .model import Neuron, ChemicalSynapse, GapJunction, Connectome, CellClass`; no wildcard imports)
- [x] 3.2 Implement `connectome/model.py` per design.md Decision T1.2: pydantic `Neuron`, `ChemicalSynapse`, `GapJunction`, `Connectome`; `CellClass = Literal["sensory", "interneuron", "motor", "muscle", "pharyngeal"]`
- [x] 3.3 Add field-level validation: chemical-synapse `weight: int = Field(gt=0)`; gap-junction `weight: int = Field(ge=0)`; neuron-set integrity (every synapse `pre`/`post`/`neuron_a`/`neuron_b` exists in `neurons` dict)
- [x] 3.4 Implement gap-junction canonical-form convention (alphabetically sorted `neuron_a < neuron_b`) to deduplicate `AVA-AVB` and `AVB-AVA` reports
- [x] 3.5 Closes T1.2 — tick the matching T1.2 box in `phase6-tracking/tasks.md`

## Phase 4 — Neuron classification table (supports T1.2 + T1.3)

302-neuron classification derived from OpenWorm cect's MIT-licensed `Cells.py` constants (which themselves attribute Cook 2019 paper + WormAtlas). Static dict shipped as code per design.md Decision T1.4. Cook 2019 does NOT publish a discrete SI 1 cell-list XLSX — cect's `Cells.py` is the de facto authoritative curation of the paper's classification.

- [x] 4.1 Source cect's `Cells.py` constants (`SENSORY_NEURONS_COOK`, `INTERNEURONS_COOK`, `MOTORNEURONS_COOK`, pharyngeal lists, `PREFERRED_HERM_NEURON_NAMES`) from `https://raw.githubusercontent.com/openworm/ConnectomeToolbox/master/cect/Cells.py`. Source URL + version pinned at cect v0.3.1 (recorded in PROVENANCE.md)
- [x] 4.2 Manually merge the lists into a single `NEURON_CLASSIFICATION: dict[str, tuple[CellClass, str | None]]` covering all 302 hermaphrodite neurons. Cross-checked: total = 302, ASEL/ASER → sensory, AVAL/AVAR → interneuron, VB2 → motor. Per-class counts: sensory 83, interneuron 83, motor 116, pharyngeal 20
- [ ] 4.3 For each neuron, attach a neurotransmitter when known (Glutamate, GABA, Acetylcholine, Serotonin, Dopamine, etc.) from cect's neurotransmitter data and `docs/nematode_biology.md` § Neurotransmitter Systems. Default `None` where uncertain — DEFERRED: shipped as `None` for all 302 entries in initial table; NT enrichment is a follow-up commit (not blocking for L0 deliverable per design.md Decision T1.4)
- [x] 4.4 Implement `connectome/neurons.py` with `NEURON_CLASSIFICATION` + module docstring attributing the curation chain (this project → cect.Cells.py → Cook 2019 paper + WormAtlas)
- [x] 4.5 Add `CANONICAL_NAME_ALIASES: dict[str, str]` for cross-dataset name normalisation (e.g. `RIA-L` → `RIAL`) — empty initial map; populated as the loader encounters dataset-name quirks in Phase 5/6
- [x] 4.6 Coverage assertion in module: assert `len(NEURON_CLASSIFICATION) == 302` at import time (fail-fast if a future edit drops an entry)

## Phase 5 — Loader (T1.3)

Maps to T1.3 in the tracker: import the Cook 2019 hermaphrodite connectome; verify counts.

- [x] 5.1 Implement `connectome/loader.py:load_cook_2019_hermaphrodite()` reading the vendored SI 5 XLSX via `pd.read_excel(..., engine="openpyxl")`; iterate the three relevant sheets (`hermaphrodite chemical`, `herm gap jn symmetric`, `herm gap jn asymmetric` — the actual sheet layout splits gap junctions into symmetric + asymmetric, not a single `hermaphrodite gap jn` sheet)
- [x] 5.2 Map XLSX row/column headers to neuron names; cross-reference against `NEURON_CLASSIFICATION` from Phase 4. Non-neuron cells in the header (muscles `dBWML*`/`vm*`, glia `GLR*`/`CEPsh*`, etc.) are filtered out
- [x] 5.3 Emit `ChemicalSynapse` and `GapJunction` entries from non-zero cells in each sheet; populate `Connectome(neurons=..., chemical_synapses=..., gap_junctions=..., source=..., version=...)`. Observed: 3709 chemical, 1093 gap-junction (after merging sym+asym sheets in canonical form)
- [x] 5.4 Implement `connectome/loader.py:load_witvliet_2021_adult() -> Connectome` reading `witvliet_2020_dataset8_adult.xlsx`. Witvliet uses long-format edge lists (pre/post/type/weight columns), not adjacency matrices; loader sniffs column names case-insensitively. Observed: 180 neurons, 1933 chemical, 296 gap-junction
- [x] 5.5 Loader produces deterministic output: chemical synapses sorted by `(pre, post)`; gap junctions sorted by `(neuron_a, neuron_b)`; neurons in canonical order. Verified by calling twice and comparing entry-by-entry
- [x] 5.6 Closes T1.3 — tick the matching T1.3 box in `phase6-tracking/tasks.md`

## Phase 6 — Validation suite (T1.4)

Maps to T1.4 in the tracker: cross-validation against Witvliet 2021 nerve-ring subset.

- [x] 6.1 Implement `connectome/validate.py:validate_neuron_count(c) -> ValidationResult` — expects 302 for hermaphrodite (passes on Cook 2019)
- [x] 6.2 Implement `validate_known_pathways(c) -> ValidationResult` — confirms at least one of three canonical sensory → interneuron → motor pathways traces. Passes on Cook 2019 with all three pathways traced (klinotaxis ASEL→AIYL→RIAL→SMDDL, thermotaxis AFDL→AIYL→RIAL→SMDDL, nociception ASHL→AVAL→VA1)
- [x] 6.3 Implement `cross_validate(primary, secondary) -> DivergenceReport`: intersection of neurons; agreement / disagreement per shared (pre, post) pair; primary-only-pairs and secondary-only-pairs (capped at 50); weight-divergence summary (mean + median + n_pairs ratios). Observed Cook 2019 vs Witvliet 2021 ds8: 180 shared neurons, 1271 agreement / 1942 disagreement, weight ratio mean 0.86, median 0.57
- [x] 6.4 Implement `connectome/validate.py:DivergenceReport` pydantic model + `ValidationResult` pydantic model
- [x] 6.5 Closes T1.4 — tick the matching T1.4 box in `phase6-tracking/tasks.md`

## Phase 7 — Smoke-test forward pass (T1.6)

Maps to T1.6 in the tracker: PPO-shaped forward pass on the connectome topology.

- [x] 7.1 Implement `connectome/smoke.py:run_forward_pass(c, *, seed=0) -> np.ndarray` per design.md Decision T1.5: build N×N chemical-synapse adjacency with random weights sampled from `N(0, 1/sqrt(fan_in))` where `fan_in` is the in-degree of each postsynaptic neuron (prevents tanh saturation); non-existent edges pinned to zero (strict-mask). N×N gap-junction adjacency uses Cook 2019 counts as fixed weights, normalised by the same fan-in factor for scale comparability
- [x] 7.2 Forward pass: `output = tanh((chemical_W + gap_W).T @ input_x)`; return `output[motor_neuron_rows]`
- [x] 7.3 Sanity guard: raise if chemical-synapse adjacency has zero non-zero entries (catches silent load failures)
- [x] 7.4 Module exposes a way for tests to assert output has non-zero variance across motor-neuron rows (catches both degenerate constants AND fully-saturated ±1 outputs); the test itself lives in Phase 8. Module exposes `_DEGENERATE_VARIANCE_THRESHOLD` constant for test reuse
- [x] 7.5 Add `if __name__ == "__main__"` block so `uv run python -m quantumnematode.connectome.smoke` runs the forward pass and prints output shape + finite-value assertion result. Verified end-to-end: 116 motor outputs, variance 0.69, PASS
- [x] 7.6 Closes T1.6 — tick the matching T1.6 box in `phase6-tracking/tasks.md`

## Phase 8 — Tests + CI integration (T1.7)

Maps to T1.7 in the tracker.

- [x] 8.1 Create `packages/quantum-nematode/tests/quantumnematode_tests/connectome/__init__.py`
- [x] 8.2 Implement `test_loader.py`: Cook 2019 loads with neuron count = 302; chemical-synapse count > 3000 (loose lower-bound sanity check — observed 3709 neuron-to-neuron chemical synapses; the project-docs "~7000" includes muscles/glia/end-organs which the loader filters out); gap-junction count > 600 (loose bound vs ~900 documented; observed 1093); Witvliet 2021 adult loads with reduced neuron count (~150-200 nerve-ring; observed 180); known sensory neurons (ASEL, ASER, AFDL, AFDR, ASHL, ASHR, ADLL, ADLR, AWAL, AWAR, AWCL, AWCR, URXL, URXR, BAGL, BAGR) present with `cell_class == "sensory"`; known motor neuron classes (VB, DB, VA, DA, VC, DD) present with `cell_class == "motor"`
- [x] 8.3 Implement `test_model.py`: pydantic field validation rejects zero-weight chemical synapses; a dual-edge pair (e.g. AVAL↔AVDL, which Cook 2019 reports with both chemical synapses in both directions AND a gap junction) is present as separate `ChemicalSynapse` AND `GapJunction` entries (not one combined entry — separately-typed connection categories); no orphan synapses (every edge's pre/post/neuron_a/neuron_b exists in `neurons`)
- [x] 8.4 Implement `test_neurons.py`: `len(NEURON_CLASSIFICATION) == 302`; every entry has a valid `CellClass`. Coverage-by-class is NOT band-asserted (boundaries are convention-dependent — Cook 2019, WormAtlas, and project docs use different rules for polymodal cells); test instead prints class counts for forensic review in the T1 logbook
- [x] 8.5 Implement `test_validate.py`: `validate_neuron_count` flags 301-neuron broken connectome; `validate_known_pathways` passes (≥ 1 of klinotaxis ASE → AIY → RIA → SMD, thermotaxis AFD → AIY → RIA → SMD, or nociception ASH → AVA → VA/DA traces successfully); `cross_validate(cook_2019_hermaphrodite, witvliet_2020_dataset8_adult)` produces non-empty agreement set and documented divergence map
- [x] 8.6 Implement `test_smoke.py`: `run_forward_pass` returns finite output of expected motor-neuron shape; output has non-zero variance across motor-neuron rows (catches degenerate constants AND fully-saturated outputs); raises on a deliberately-emptied connectome
- [x] 8.7 Tests run in default pytest tier (`uv run pytest -m "not nightly"` includes them). LFS fetch happens before tests in CI via `git lfs pull` or implicit smudge — verify
- [x] 8.8 Closes T1.7 — tick the matching T1.7 box in `phase6-tracking/tasks.md`

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
