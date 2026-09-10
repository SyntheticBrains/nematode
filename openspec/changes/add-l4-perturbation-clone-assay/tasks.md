# Tasks: the variant through the clone assay

## 1. The arm

- [ ] 1.1 One config: the wild-type plastic clone arm with `plasticity_eligibility: node_perturbation` at the control's pinned `σ_node`, and nothing else changed from its parent.
- [ ] 1.2 Add it to the clone-assay harness registry; update the harness docstring, which still
  says it screens consolidation mechanisms.
- [ ] 1.3 Tests: the config is a single-key-block delta from its parent and loads with the variant
  selected; the harness's registry covers it and its pass rule is unchanged.

## 2. The runs

- [ ] 2.1 Commit the launch record — the arm, the pinned σ, the unchanged assay, and what each
  outcome licenses — then run seeds 1–8 at 2000 episodes.
- [ ] 2.2 Records under `docs/experiments/logbooks/supporting/050-l4-perturbation-clone-assay/`:
  `launch.md`, `screen.json`, `per-seed.csv`, `_manifest.txt`, `details.md`.

## 3. Documentation

- [ ] 3.1 `CHANGELOG.md`.
- [ ] 3.2 Tracker and roadmap updated with the outcome at close-out; on a pass, a connectome arm
  becomes buildable and the panel still waits on I.2.
