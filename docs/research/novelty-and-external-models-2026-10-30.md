# Novelty search and external-model scoping (2026-10-30)

Two research passes made while F.0b-i's attempt 2 ran: a fresh novelty search on Phase 10's three claims, and a scoping of published *C. elegans* behaviour models whose output tracks the benchmark could grade (`benchmark_grade.py tracks`). Description only; nothing here is a registered reading. Items marked *verified* were checked against the source by the maintainer's session; the rest come from the search and are marked where they could not be confirmed.

## Novelty (searched 2023 to October 2026)

| Claim | Verdict | Nearest work |
|---|---|---|
| A graded benchmark for simulated worms, with between-date bands, held-out real cohorts, matched negatives, a measurement model and track import | **Unoccupied for *C. elegans*** | flybench (fly); OpenWorm's planned kinematics tier |
| The wiring question on fidelity: a closed-loop connectome with a data-derived neuromodulator layer, fitted to real statistics, against equal-budget nulls, graded on held-out behaviour and mutants | **Unoccupied** | flybench's wiring nulls; OpenWorm's planned neuropeptide layer |
| Fitting against reward on the same architecture, on a graded fidelity metric | **Partly occupied in other species; unoccupied for the worm** | Fujii et al. 2025; Kutscher, Keller et al. 2026 |

### New works, not yet in the repository or the literature watch

- **Cho B. (2026), flybench** (*verified*): github.com/brandoncho369/flybench, MIT, v1.0 frozen 2026-10-22; preprint Zenodo doi:10.5281/zenodo.23678526.
  - What it is: 37 pre-registered tasks for whole-brain FlyWire models, with shuffled-wiring controls (rewired, random, sign-flipped) on every task, synapse-count jitter, and external submissions that declare their free parameters and fit data.
  - Relevance: the design precedent for the benchmark and for W's nulls.
  - Differs: the fly; stimulus–response task checks with mostly convention thresholds ("Most of our numbers are conventions and say so"), not population statistics against real-track bands or held-out cohorts.
- **Margolis & Gordus (2025)**, "A stochastic explanation for observed local-to-global foraging states in *C. elegans*", eLife 104972, 2025-07-30 (*verified*).
  - What it does: fits López-Cruz 2019's reorientations (1,631 worms, 45 min) with a Gillespie model whose rate decays continuously, and finds it beats a two-state switch by Jensen–Shannon divergence. Code: github.com/GordusLab/Margolis_foraging (MIT, MATLAB).
  - Relevance: bears directly on the local-search family and on the open-loop two-state negative.
- **OpenWorm design documents** (2026, docs preview branch; the DD010 and DD006 pages returned 404, summaries only):
  - a planned validation tier of five kinematics at ±15% against the Schafer database, blocked pending its toolbox;
  - a planned Ripoll-Sánchez neuropeptide layer in c302, with dwelling and roaming as its behavioural target, but no fitting to behaviour and no nulls.
  - Watch these: the second is the nearest planned overlap with W's neuromodulator layer.
- **Brunton, Abe, Hu & Tuthill, "The digital sphinx"**, now an eLife reviewed preprint (2026-08-17, doi:10.7554/eLife.111516.1). It says "behavioral fidelity is achievable without biological fidelity". Cite this version.
- **Zhou & Hasler (2026)**, bioRxiv 10.64898/2026.06.10.731214: argues that behavioural benchmarks cannot separate real wiring from nulls, and proposes representational geometry instead. The counter-position the benchmark paper should answer.
- **Fujii et al. (2025)**, arXiv 2510.10451, and **Kutscher, Keller, Kirsch, Pei & Nayebi (2026)**, arXiv 2610.34638: fitting or training objectives ranked on behavioural fidelity, in flies, newts, moths and zebrafish.
- **Antonic et al. (2025)**, arXiv 2510.00086: a fitted agent-based worm compared with real tracks (minor).
- Unverified, not used: "NeuroSimWorm" (Neurocomputing 2025); the page returned 403.

### A correction to the literature watch

Watch issue #418 read Reimers et al. (arXiv 2609.30508) as suggesting that biological wiring helps. The full abstract says the opposite (*verified*): "the randomized null models are often outperforming the original connectomes on the chosen benchmarks". Its tasks are machine-learning ones (reservoir computing), not behavioural fidelity.

## External models the benchmark could grade

**No published generative model of on-lawn behaviour has open code and a lawn**, other than the two-state hidden Markov class (Flavell 2013; Scheer & Bargmann 2023). The benchmark already covers that class with its open-loop negative. Every practical external model is an **off-food search model**, so it can be graded on `local_search` only.

| Rank | Model | Code and licence | What grading it would show | Effort |
|---|---|---|---|---|
| 1 | **Roberts et al. 2016**, the stochastic switch model, reimplemented from the paper | No model code. Data on Dryad (CC0). Its inputs are held constant over its 10-min observation period (*verified*). | A published model with no decay. Expected to fail local search's presence. **It is graded outside its stated 10-min scope**, so a fail shows the benchmark detects a missing decay, not that the model is wrong. | 2–4 h |
| 2 | **Margolis & Gordus 2025**, its event times embedded in a run–reversal walk | MIT, MATLAB; event times only, no positions (*verified*) | The decaying counterpart, expected to pass γ and the early/late ratio. Fitted to López-Cruz, the reference's own data, so it is an oracle for the import and detector path, not independent validation. | 4–6 h |
| 3 | **Costa, Ahamed, Jordan & Stephens 2024**, posture Markov model → centroid by resistive force theory | Code MIT. The README gives no data licence; the Kaggle page reportedly says CC BY-NC-ND (unverified). | The strongest published generative path model. Its authors state it misses the slow fall in pirouette rate, so it is expected to fail on the decay as its authors describe. Needs re-simulation at ≥ 45 min. | 10–20 h, ~1–3 CPU-h |
| (4) | **Haley et al. 2025** real single-patch tracks (data, not a model) | CC BY 4.0, 15 GB, login needed | A cross-lab check of the small-lawn bands; its protocol differs (worms start off the patch). | 6–12 h |

Not gradeable, with the reason:

- chemotaxis-only models (Izquierdo & Beer; Hironaka & Sumi 2025);
- GPU bodies with no lawn (BAAIWorm; Lee 2026, which is also non-commercial);
- game worlds (WormWars);
- classifiers or decision models (Ji 2021; Haley 2025's GLM);
- no generative code (Flavell 2013; Ahamed 2021; Milward 2011; Dal Bello 2021);
- the wrong assay (Ding 2019, aggregation).
