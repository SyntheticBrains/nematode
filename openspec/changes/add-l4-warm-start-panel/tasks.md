# Tasks: the warm-start panel

## 1. Per-seed weight paths

- [ ] 1.1 `{seed}` in `weights_path` resolved with the run seed by the entry point before loading;
  unresolved braces rejected.
- [ ] 1.2 Tests: resolution, rejection, and a config without the placeholder unchanged.

## 2. Configs

- [ ] 2.1 Ten warm-started configs, each one `weights_path` key off its parent (`_clone` and
  `_fullclone` suffixes); variant tests for the delta and the naming; excluded from the smoke
  list with the reason stated there.

## 3. The campaign script

- [ ] 3.1 `scripts/campaigns/l4_warm_start.py teacher`: select the best MLP seed by plateau tail,
  copy its weights, write the frozen recording config, run the recording; `clone`: the 32
  clones at the registered hyperparameters, `clones.json` with every fit and the flag.
- [ ] 3.2 Tests on synthetic inputs: selection, config derivation, clone naming, the flag.

## 4. The harness

- [ ] 4.1 `scripts/analysis/l4_warm_start.py`: twelve-arm registry, seeds 1–8 enforced, panel 2's
  table for W1, W1–W6 as one family, the verdict map, the annotations incl.
  `rule_destroys_clone`, descriptive pairs and teacher ceilings, clone fits, per-seed CSV and
  curves.
- [ ] 4.2 Tests: registry and range, directions, family size, every verdict row in order, each
  annotation, the CSV shape.

## 5. Launch and run

- [ ] 5.1 Launch record under `supporting/043-l4-warm-start/` committed before any campaign runs.
- [ ] 5.2 Teacher campaign (8 × 6000), selection, recording (300 episodes, seed 101), 32 clones.
- [ ] 5.3 The panel: frozen arms at 600, plastic and Hebbian arms at 2000, PPO arms at 3000; the
  single registered extension for any run the plateau detector marks non-converged.

## 6. Analysis and records

- [ ] 6.1 Analyse; promote `panel.json`, `per-seed.csv`, `curves.csv`, `clones.json`, the manifest,
  the teacher record and a `details.md` to the supporting directory.

## 7. Close-out

- [ ] 7.1 `docs/usage.md`, `configs/README.md`, `CHANGELOG.md`; tracker S.2 ticked with the verdict.
- [ ] 7.2 Pre-commit gate on all files exit 0; full suite green.
- [ ] 7.3 No implementation code or docstring references a planning document.
- [ ] 7.4 Re-review for drift, archive, review the branch, open the PR.
