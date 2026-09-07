# Tasks: panel 3 — replicating the Hebbian wiring contrast

## 1. The harness

- [ ] 1.1 `scripts/analysis/l4_panel3.py`: panel 2's registry and readers imported; replication seeds
  17–64 enforced for the Hebbian arms; R1 (paired Wilcoxon) and R2 (exact binomial on
  competent-fraction discordance at 20%) as one BH-FDR family; the verdict from R1 with the R2
  annotation; the frozen floors for 17–64 read from panel 2's sweep; seeds 1–16 pooled from
  panel 2's committed `per-seed.csv` as descriptive; learning gains, distributions, sign counts;
  per-seed CSV and curves.
- [ ] 1.2 Tests on synthetic values: seed-range enforcement; R2's p-value on a known `(b, c)`; R1/R2
  directions; every verdict row; the R2-passes-R1-fails annotation; the pooled seed set; the CSV
  shape.

## 2. Launch and run

- [ ] 2.1 Launch record under `supporting/042-l4-panel3/` (commit, command, seeds, budget, the
  floors' provenance) committed before the campaign runs.
- [ ] 2.2 The campaign: `wt_hebbian` and `rn_hebbian` × seeds 17–64 × 1000 episodes; the single
  registered extension (fresh run at 1500) for any seed the plateau detector marks
  non-converged.

## 3. Analysis and records

- [ ] 3.1 Analyse; promote `panel3.json`, `per-seed.csv`, `curves.csv`, the manifest and a
  `details.md` to the supporting directory.

## 4. Close-out

- [ ] 4.1 `CHANGELOG.md`; tracker entry ticked with the verdict.
- [ ] 4.2 Pre-commit gate on all files exit 0; full suite green.
- [ ] 4.3 No implementation code or docstring references a planning document.
- [ ] 4.4 Re-review for drift, archive, review the branch, open the PR.
