# Tasks: the L4 2×2 panel

## 1. The two rewired floors

- [x] 1.1 `…_plastic_frozen_rewired_null.yml`: the frozen floor plus the rewired-null `wiring` key, with
  a header comment stating the one-key delta and why `rewire_seed` stays unset.
- [x] 1.2 `…_plastic_hebbian_rewired_null.yml`: the same one key off the Hebbian floor.
- [x] 1.3 Variant tests: each new config differs from its wild-type parent by exactly
  `brain.config.wiring`; each loads with the parent's rule, freeze flag, strict mask and
  traces; parent name stays a prefix. The rewired frozen arm's chemical mask equals the rewired
  plastic arm's at one seed — the shared wiring T4 assumes.
- [x] 1.4 Smoke entries for both.

## 2. The panel harness

- [x] 2.1 `scripts/analysis/l4_panel.py`: arm registry (config stem → arm key), campaign-dir and
  manifest readers, seed parse, plateau-tail and convergence per seed via the committed helpers,
  experiment-JSON lookup from the logged id with convergence reported unknown when absent.
- [x] 2.2 The four confirmatory tests as one BH-FDR family, the CI-based band test, the
  reverse-result detection, the ensemble-invariance counts, the verdict map in the registered
  order, and the 21 descriptive pairs — labelled as such in the JSON and CSV.
- [x] 2.3 Per-seed CSV and per-seed learning-curve export (rolling full-clear per 250 episodes).
- [x] 2.4 `--pilot` mode: per-rate per-arm plateau-tail and onset over the pilot seeds, the pooled
  selection with default tie-break, the budget rule.
- [x] 2.5 Tests on synthetic logs and JSONs covering every branch of the verdict map, both band
  outcomes, the reverse case, the family size, the pooled tie, the budget rule's rounding and
  floor, and the seed guard (confirmatory mode refuses seeds outside 1–8).

## 3. The pilot runner

- [x] 3.1 `scripts/campaigns/l4_panel_pilot.py`: derive grid configs (parent + `plasticity_rate`)
  under `<out>/configs/`, frozen arms once, invoke `run_campaign.py` with `--track-experiment`
  and the headless theme passed through; `--dry-run` prints the plan.
- [x] 3.2 Tests: derived configs are one key off their parents; the plan has the registered arm ×
  rate × seed shape.

## 4. Pilot, then pin

- [x] 4.0 Pilot 1 ran as registered (2026-09-06) and pinned nothing: grid two orders too hot, MLP
  dead or frozen at every shared rate. Recorded under `supporting/040-l4-panel/pilot-1-*`.
  Unblocked 2026-09-06: the rule-scaling and modulator-centring changes landed (PRs #313, #314); probes
  2 and 3 recorded; the grid re-registered as {3e-4, 1e-3, 3e-3}, ties to 1e-3, by dated amendment.
- [x] 4.1 Re-run the pilot on the re-registered grid {3e-4, 1e-3, 3e-3} (seeds 101–102, 3000 episodes); extend any
  non-converged three-factor arm at the selected rate once to 6000 as a separate campaign
  invocation (a fresh run; episode counts are uniform per campaign). If it still has no plateau,
  pin the budget at 6000 and flag the arm in the summary.
- [x] 4.2 Summarise with `l4_panel.py --pilot`; commit `pilot.json` under
  `docs/experiments/logbooks/supporting/040-l4-panel/`.
- [x] 4.3 Pin the recipe and the budget in `design.md § Pinned values` by dated amendment; write the
  selected `plasticity_rate` explicitly into all seven plastic-family configs, the MLP
  included.
- [x] 4.4 Write `launch.md` (commit SHA, command, seeds, budget, recipe) and commit it **before**
  launching the panel.

## 5. The panel

- [ ] 5.1 Launch seven arms × seeds 1–8 through `run_campaign.py` at the pinned budget with
  `--track-experiment`; apply the single pre-registered extension to any still-climbing seed.
- [ ] 5.2 Analyse with `l4_panel.py`; promote `panel.json`, `per-seed.csv`, `curves.csv`, the
  manifest and a short `details.md` into the supporting directory.
- [ ] 5.3 If the verdict is `robustness`: run the one pre-registered sensitivity pass (primary pair,
  the two unselected rates, panel seeds) and report it descriptively; otherwise record that it
  was not triggered.

## 6. Close-out

- [ ] 6.1 `configs/README.md`, `docs/architectures.md`, `CHANGELOG.md`; tracker A.7 ticked with the
  verdict named and "Next: A.8 the logbook"; AGENTS.md gains the harness usage line.
- [ ] 6.2 Pre-commit gate on all files exit 0; full suite green.
- [ ] 6.3 No implementation code or docstring references a planning document.
- [ ] 6.4 Re-review for drift (every scenario maps to a test or a committed artefact), archive,
  review the branch, open the PR.
