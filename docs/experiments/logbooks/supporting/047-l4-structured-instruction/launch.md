# Structured-instruction test launch record

Written and committed before any run, per the registration.

- **Date**: 2026-09-09
- **Pinned state**: `feat/l4-structured-instruction` at the commit implementing the routed third
  factor, the pathway derivation, the two arm configs and the harness.
- **The question**: every panel so far broadcast one reward-prediction error to every plastic
  synapse. Does routing it through the wiring that carries it change the answer?

## The pathway, derived before the runs

From the vendored atlas and the Cook 2019 chemical edges. Reading every identity column — the
sign-grounding loader read only the first, so ADF and HSN looked purely cholinergic and RIM purely
glutamatergic — the aminergic release set is **18 neurons**:

| source | neurons | reach |
|---|---|---|
| dopamine | `ADEL/R`, `CEPDL/R`, `CEPVL/R`, `PDEL/R` | 104 |
| serotonin | `NSML/R`, `ADFL/R`, `HSNL/R` | 84 |
| octopamine | `RICL/R` | 31 |
| tyramine | `RIML/R` | 31 |
| **all four** | **18** | **169 of 302 neurons** |

Keyed on the post-synaptic neuron, that is **2,636 of 3,709 chemical synapses (71.1%)** on the
wild type. The rewired null derives its own pathway from its own edges and covers **75.5%**; both
fractions are reported with the result.

Excluded by a stated rule, and recorded here so the choice is visible: uptake-only annotations
(`AIM`, `RIH`), the atlas's own "alternative synthesis/uptake mechanism" hedges (`I5`, `VC4`,
`VC5`), its precursor entry (`MI`, "5-HTP") and its male-only note (`PVW`). Counting those five
would give 24 neurons and 74.1%.

## What the routed arm does

The modulator reaches instructed synapses; everywhere else the third factor is `1.0`, which is
the **unmodulated Hebbian floor's** update. The arm is therefore an interpolation between two arms
the panels already measured, and introduces **no hyperparameter**.

## The pathway is a proxy, and this is the claim being tested

Aminergic transmission in this animal is substantially extrasynaptic: dopamine, serotonin,
octopamine and tyramine are released by volume onto receptors expressed by cells that need not be
synaptic partners of the releasing neuron (Bentley et al., *PLoS Comput Biol*, 2016). The wired
reach above is a **lower bound and a modelling choice**. What this tests is the falsifiable
proposition that the *synaptic* reach of the aminergic neurons picks out a functionally meaningful
subset. A negative refutes that proxy, not structured instruction; the receptor layer (B.3) is
what would replace wired reach with expressed-receptor reach.

## Protocol

Four arms — `wt_pathway`, `rn_pathway`, `wt_global`, `rn_global` — seeds 1–16 paired, **3000
episodes** (the panel's plastic budget), plateau-tail full-clear success, one registered extension
of a fresh run at 1.5× for a run the plateau detector marks non-converged. 64 runs.

The global arms are **re-run concurrently** rather than read from panel 1's committed table, which
carries eight seeds at this budget; panel 1's values are reported beside the re-run arm as a
consistency check and are never a comparator.

## Family and verdict, fixed before any run

Four one-sided paired tests corrected together under BH-FDR at α = 0.05: **S1** wt routed over wt
global (the primary), **S2** rn routed over rn global, **S3** wt over rn under routing, **S4** the
same under the global scalar.

Verdict in order: `insufficient_seeds`; **`no_routing_effect`** when neither S1 nor S2 confirms —
the outcome in which the global scalar was not the limitation; then `routing_helps_both`,
`routing_helps_wild_type_only`, `routing_helps_rewired_only`. S3 and S4 annotate and never decide.

**`routing_helps_both` is not a win, and that reading is fixed here in advance**: a routed third
factor that helps the scramble as much as the animal is a fact about running two learning regimes
in one network, not about *C. elegans* connectivity.

## Commands (from the repository root)

```bash
P=configs/scenarios/foraging_predator_thermal/connectomeppo_small_continuous2d_combined_klinotaxis_plastic
uv run python scripts/run_campaign.py \
  --config ${P}_pathway.yml --config ${P}_rewired_null_pathway.yml \
  --config ${P}.yml --config ${P}_rewired_null.yml \
  --seeds 1-16 --runs 3000 --output-dir campaigns/l4-routing -- --theme headless --track-experiment
```

## Analysis

`uv run python scripts/analysis/l4_structured_instruction.py --campaign-dir campaigns/l4-routing --out docs/experiments/logbooks/supporting/047-l4-structured-instruction/panel.json --csv .../per-seed.csv --curves .../curves.csv`

## Results

*(written here after the runs)*
