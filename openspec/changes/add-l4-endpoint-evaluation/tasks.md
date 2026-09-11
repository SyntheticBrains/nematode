# Tasks

## 1. The endpoints

- [x] 1.1 Copy the eight auto-saved final weights of the I.1 clone-assay learning arm to
  `campaigns/l4-perturbation-clone/endpoints/nodeperturbation_wt_seed{seed}.pt` (an ignored
  directory, as the clones' is); record each source export ID in the launch record.
- [x] 1.2 Confirm an endpoint loads through the runner into the comparator's configuration
  (done: seed 1, two episodes, disclosed in the design).

## 2. The arm

- [x] 2.1 One config, a single-key delta from
  `connectomeppo_small_continuous2d_combined_klinotaxis_plastic_frozen_clone.yml`: `weights_path`
  to the staged endpoints. No other key.
- [x] 2.2 Registry entry in `scripts/analysis/l4_consolidation_screen.py` (`ARMS`) and the
  `ARM_KEYS` test pin; the harness's annotation of endpoint-vs-under-perturbation per seed, reading
  the assay's committed per-seed table.
- [x] 2.3 Test: the config is the comparator's plus one key; the arm assesses under the registered
  rule; the annotation reads the committed 050 values.
- [x] 2.4 The load-integrity check: the harness compares this arm's per-seed cosine to the clone
  against the 050 record's committed values and voids a seed that departs by more than 0.01, with
  a test that a cosine near 1.00 (the clone loaded instead) voids the verdict.

## 3. The run

- [x] 3.1 Launch record written before the run: the arm, the rule, both outcomes and what each
  licenses, the endpoint sources, the two disclosed episodes.
- [x] 3.2 Run: 8 seeds × 2000 episodes, `--track-experiment`, from `main`, no branch switch while
  it runs.
- [x] 3.3 Read the integrity check before anything else; a voided seed stops the scoring and
  the cause is found and the run repeated.
- [x] 3.4 Records under `supporting/052-l4-endpoint-evaluation/`: `screen.json`, `per-seed.csv`,
  `_manifest.txt`, `details.md`.

## 4. Close-out

- [x] 4.1 `CHANGELOG.md`; tracker (as I.1c step 0) and roadmap with the verdict and the
  registration it selects.
- [ ] 4.2 The next change is authored from the verdict. **Ran 2026-09-11: `fail`** (mean 20.6
  against 38.7, 2/8 within hold; integrity clean), so the verdict licenses **the low-σ
  programme, as fixed by review**, and not the assay amendment. Authored as its own change,
  carrying this panel's bimodality — six seeds degraded, two improved, seed 2 to 73.4, the
  best this substrate has recorded — which I.2's statistic now has a demonstrated case for.
