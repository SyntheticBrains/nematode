# Tasks: the variant through the clone assay

## 1. The arm

- [x] 1.1 One config: the wild-type plastic clone arm with `plasticity_eligibility: node_perturbation` at the control's pinned `σ_node`, and nothing else changed from its parent.
- [x] 1.2 A second config: the same arm with `freeze_updates: true`, the frozen-perturbation
  control.
- [x] 1.3 Add both to the clone-assay harness registry; update the harness docstring, which still
  says it screens consolidation mechanisms.
- [x] 1.4 Report the trajectory annotation — final-quarter against first-quarter plateau tail —
  per arm, from the curves already read. Not a verdict input.
- [x] 1.5 Tests: each config is a minimal delta from its parent and loads as intended; the registry
  covers both; the pass rule, comparator, budget and metric are unchanged; the trajectory annotation
  is computed per arm and cannot change the verdict.

## 2. The runs

- [x] 2.1 Commit the launch record — the arms, the pinned σ, the unchanged assay, the annotation and what each outcome licenses — then run both arms on seeds 1–8 at 2000 episodes.
- [x] 2.2 Records under `docs/experiments/logbooks/supporting/050-l4-perturbation-clone-assay/`:
  `launch.md`, `screen.json`, `per-seed.csv`, `_manifest.txt`, `details.md`.

## 3. Documentation

- [x] 3.1 `CHANGELOG.md`.
- [x] 3.2 Tracker and roadmap updated with the outcome at close-out; on a pass, a connectome **panel** arm becomes buildable and the panel still waits on I.2. This closes into I.1's logbook rather than taking one of its own.
