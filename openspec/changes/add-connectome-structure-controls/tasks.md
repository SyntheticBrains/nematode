# Tasks — connectome-structure controls

Scope: the **degree-preserving rewired-null** control (`T7.controls.rewired_null`). The
learnable-gap-junction control (`T7.controls.learnable_gj`) is a tracked **fast-follow**, deferred
from this change (secondary, lower signal, adds the only non-trivial correctness risk; revisit if the
rewired-null result or the 6a synthesis review raises the frozen-electrical-synapse question).

## 1. Degree-preserving rewiring utility

- [x] 1.1 Add `connectome/rewiring.py`: `rewire_degree_preserving(connectome, rng, swaps_per_edge=10)`
  returning a new `Connectome` with the same `neurons` and degree-preserving edge-swapped
  `chemical_synapses` (directed double-edge-swap) + `gap_junctions` (undirected double-edge-swap),
  rejecting self-loops / duplicates; `source` annotated with the seed. Weights travel with the edge.
- [x] 1.2 Tests (`connectome/test_rewiring.py`): in/out degree per neuron preserved (chemical) + gap
  degree preserved; no self-loops / duplicates; neuron set + order unchanged; deterministic under the
  same seed, differs across seeds; edge count preserved.

## 2. Wiring config option + brain injection seam

- [x] 2.1 Add `wiring: Literal["wild_type", "rewired_degree_preserving"] = "wild_type"` and
  `rewire_seed: int | None = None` to `ConnectomePPOBrainConfig`.
- [x] 2.2 In the connectome brain, between `connectome = load_cook_2019_hermaphrodite()` and the
  `ConnectomeTopology(...)` construction, apply `rewire_degree_preserving` when
  `wiring == "rewired_degree_preserving"`, using a **dedicated** `np.random.default_rng(rewire_seed if rewire_seed is not None else self.seed)` — independent of `self.rng` / the global seed — so the
  `w_chem` init draws are unperturbed (matched init vs wild-type for the same seed). Everything
  downstream unchanged.
- [x] 2.3 Byte-identical invariant test: with `wiring: wild_type` the rewiring branch is skipped, so
  the built `m_chem`, `w_chem` init, and `g_gap` equal those from the unmodified load path (construct a
  wild-type brain and a directly-loaded-connectome topology; assert tensor-equal). Guards the 029
  ranking cell against drift.
- [x] 2.4 Rewired-brain test: with `wiring: rewired_degree_preserving`, `m_chem` differs from wild-type
  but has the same per-column/row sums (degree preserved), and the brain trains a step without error.

## 3. Config (match the c3_integrated cell verbatim)

- [x] 3.1 `configs/scenarios/foraging_predator_thermal/connectomeppo_small_continuous2d_combined_klinotaxis_rewired_null.yml`
  — a verbatim copy of the wild-type combined cell with only `wiring: rewired_degree_preserving` added.

## 4. Control-analysis harness

- [ ] 4.1 `scripts/analysis/connectome_structure_controls.py` — load a `<arm> <seed> <out>` manifest
  (arms: `wild_type`, `rewired_null`), compute the C3 plateau-tail ranked success per seed **reusing
  the committed ranking metric** (`t7_continuous_ranking` / the level-agnostic ranked metric — no new
  metric), print the paired-seed deltas + 80% bootstrap CI + BH-FDR
  (`weight_search_architecture_ranking`), and the verdict (wild-type > rewired at q\<0.05 → "specific
  wiring matters"; CI spans 0 → "degree statistics, not wiring"). Write a summary JSON.
- [ ] 4.2 Tests: metric + paired-delta + verdict logic on synthetic `.out`s (mirror
  `test_associative_memory_separation.py`).

## 5. Calibration / smoke (before the panel)

- [ ] 5.1 Single-seed smoke: the rewired-null trains on the combined cell without collapse and the
  wild-type cell reproduces its 029 plateau (a recipe-match sanity check). Record the directed-swap
  acceptance rate + whether `swaps_per_edge` needs raising for mixing.
- [ ] 5.2 **PAUSE for user review of the smoke before the full panel.**

## 6. Evaluation + verdict

- [ ] 6.1 Panel: `wild_type` vs `rewired_degree_preserving` on the combined continuous integrated-C3
  cell, **both arms re-run in one fresh panel** (identical code version + exact seed pairing),
  **n ≥ 8 paired seeds**, headless, parallelised (`OMP_NUM_THREADS=1`, `xargs -P`), same PPO recipe /
  **budget as the 029 `T7.connectome.c3_integrated` cell** (6000ep uniform; 8000ep top-up if 029 used
  one).
- [ ] 6.2 Run the harness (§4); record the per-arm ranked success, the paired BH-FDR table, and the
  verdict (specific-wiring vs degree-statistics).
- [ ] 6.3 **PAUSE for user review of the evaluation + verdict before writing the logbook** (project
  convention).

## 7. Logbook + tracker

- [ ] 7.1 Write the logbook (objective / method / results / analysis / limitations) + committed
  supporting artefacts (no `tmp/` references); it feeds the 6a synthesis (T9a / T9.2).
- [ ] 7.2 Add the logbook row to `docs/experiments/README.md`.
- [ ] 7.3 Tick `T7.controls.rewired_null` + `T7.controls.logbook` in
  `openspec/changes/phase6-tracking/tasks.md` with the verdict; note `T7.controls.learnable_gj` remains
  open as a fast-follow.
- [ ] 7.4 Document the `_rewired_null` connectome config variant in `AGENTS.md`.

## 8. Pre-merge gates

- [ ] 8.1 Targeted `pre-commit` during iteration; full `pre-commit run -a` before push.
- [ ] 8.2 `openspec validate add-connectome-structure-controls --strict`.
- [ ] 8.3 Full `uv run pytest -m "not nightly"` green (byte-identical wild-type invariant holds).
- [ ] 8.4 Archive the change in-PR (`openspec archive add-connectome-structure-controls -y`).
