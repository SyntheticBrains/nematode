# Tasks

## 1. The endpoints

- [x] 1.1 Copy the eight auto-saved final weights of the I.1 clone-assay learning arm to
  `campaigns/l4-perturbation-clone/endpoints/nodeperturbation_wt_seed{seed}.pt` (an ignored
  directory, as the clones' is); record each source export ID in the launch record.
- [x] 1.2 Confirm an endpoint loads through the runner into the comparator's configuration
  (done: seed 1, two episodes, disclosed in the design).

## 2. The arm

- [ ] 2.1 One config, a single-key delta from
  `connectomeppo_small_continuous2d_combined_klinotaxis_plastic_frozen_clone.yml`: `weights_path`
  to the staged endpoints. No other key.
- [ ] 2.2 Registry entry in `scripts/analysis/l4_consolidation_screen.py` (`ARMS`) and the
  `ARM_KEYS` test pin; the harness's annotation of endpoint-vs-under-perturbation per seed, reading
  the assay's committed per-seed table.
- [ ] 2.3 Test: the config is the comparator's plus one key; the arm assesses under the registered
  rule; the annotation reads the committed 050 values.

## 3. The run

- [ ] 3.1 Launch record written before the run: the arm, the rule, both outcomes and what each
  licenses, the endpoint sources, the two disclosed episodes.
- [ ] 3.2 Run: 8 seeds × 2000 episodes, `--track-experiment`, from `main`, no branch switch while
  it runs.
- [ ] 3.3 Records under `supporting/052-l4-endpoint-evaluation/`: `screen.json`, `per-seed.csv`,
  `_manifest.txt`, `details.md`.

## 4. Close-out

- [ ] 4.1 `CHANGELOG.md`; tracker (as I.1c step 0) and roadmap with the verdict and the
  registration it selects.
- [ ] 4.2 The next change is authored from the verdict: the assay amendment if the endpoints hold
  or improve; the low-σ programme, as fixed by review, if they fail.
