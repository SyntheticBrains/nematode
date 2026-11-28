# Documentation

Start with the [project README](../README.md) — it states the research question, the results so far and how to run a first simulation. Everything else is indexed here.

## The science

| Document | What it is |
|---|---|
| [roadmap.md](roadmap.md) | The research programme: vision, phase history with exit criteria and decision gates, research questions, every phase's plan and close through Phase 10 (the Phase 11 plan review is pending), relationship to OpenWorm and other projects |
| [experiments/README.md](experiments/README.md) | Index of every numbered experiment logbook — the evidence base behind each claim. Phase syntheses: [021](experiments/logbooks/021-phase5-synthesis.md) (Phase 5), [037](experiments/logbooks/037-phase6a-synthesis.md) (Phase 6a), [069](experiments/logbooks/069-phase7-synthesis.md) (Phase 7), [094](experiments/logbooks/094-phase8-synthesis.md) (Phase 8), [103](experiments/logbooks/103-phase9-synthesis.md) (Phase 9), [138](experiments/logbooks/138-phase10-synthesis.md) (Phase 10) |
| [nematode_biology.md](nematode_biology.md) | *C. elegans* sensory systems, navigation strategies, predators, learning, social behaviour and evolution, with references — the biology each simulation channel is modelled on |
| [research/quantum-architectures.md](research/quantum-architectures.md) | Specifications and the strategic assessment from the Phase 2 quantum-architecture campaign (closed) |
| [research/policy-architecture-candidates.md](research/policy-architecture-candidates.md) | Survey of candidate policy architectures for the continuous substrate and the memory axis |
| [research/associative-memory-probe.md](research/associative-memory-probe.md) | Design note for the chemosensory associative-memory probe ([Logbook 033](experiments/logbooks/033-associative-memory-probe.md)) |
| [research/phase-protocol.md](research/phase-protocol.md) | How a phase is run: the pattern Phases 5–6 used, where Phase 7 departed from it and why, thirteen principles each with the cost of ignoring it |

## Using the platform

| Document | What it is |
|---|---|
| [usage.md](usage.md) | CLI reference for `run_simulation.py` and `run_evolution.py`, outputs, experiment tracking, evolution and inheritance, analysis and campaign scripts, quantum hardware, the GPU container |
| [architectures.md](architectures.md) | Catalogue of all 26 brain architectures, their role in the programme, and which optimiser to use |
| [visualization.md](visualization.md) | Renderers, sprites, overlays and keyboard controls |
| [../configs/README.md](../configs/README.md) | Scenario directories and the `{brain}_{size}[_{variant}]_{sensing}.yml` naming convention |
| [experiments/templates/experiment.md](experiments/templates/experiment.md) | Template for a new logbook |

## Developing

| Document | What it is |
|---|---|
| [../CONTRIBUTING.md](../CONTRIBUTING.md) | Development setup, test tiers, code style, pull-request process |
| [../CHANGELOG.md](../CHANGELOG.md) | Notable changes per release, breaking changes first |
| [../SECURITY.md](../SECURITY.md) | How to report a vulnerability privately, and what is unsafe to load from untrusted sources |
| [architecture/plugin-developer-guide.md](architecture/plugin-developer-guide.md) | How to add a brain architecture through the plugin registry |
| [../AGENTS.md](../AGENTS.md) | Instructions for AI coding assistants: commands, repository layout, conventions |
| [../openspec/](../openspec/) | Spec-driven development: `specs/` hold the current capability specs, `changes/` the in-flight proposals, `changes/archive/` the history |
| [STANDARDIZATION.md](STANDARDIZATION.md) | Recorded decisions: no Gymnasium wrapper, no Hydra, and the removal of the NematodeBench benchmark system |
| [../.github/workflows/README.md](../.github/workflows/README.md) | What each CI workflow does |
| [../artifacts/README.md](../artifacts/README.md) | Curated experiment outputs referenced by logbooks (Git LFS) |

## Data

| Document | What it is |
|---|---|
| [../data/connectome/PROVENANCE.md](../data/connectome/PROVENANCE.md) | Source, licence and parsing notes for the vendored Cook et al. 2019 and Witvliet et al. 2021 connectome files |
| [../data/chemotaxis/README.md](../data/chemotaxis/README.md) | Published chemotaxis-index values and behavioural bias signatures used by the real-worm validation |
| [../data/benchmark/PROVENANCE.md](../data/benchmark/PROVENANCE.md) | The benchmark's derived real-worm sets (Scheer & Bargmann 2023, Calhoun 2014, Roberts 2016, López-Cruz 2019, Chen 2025) and its bands: sources, licences, checksums and units |
| [../data/benchmark/IMPORT.md](../data/benchmark/IMPORT.md) | How to grade external tracks, real or simulated, through the benchmark: the CSV and WCON formats and what each version grades |
| [../packages/wormbench/README.md](../packages/wormbench/README.md) | The `wormbench` grader as a standalone package: install, a first grading, your own tracks, versions and data |
| [../data/roaming_dwelling/PROVENANCE.md](../data/roaming_dwelling/PROVENANCE.md) | Scheer & Bargmann 2023's roaming/dwelling model and windows, and the classifier's calibration |
| [../data/neuromodulators/PROVENANCE.md](../data/neuromodulators/PROVENANCE.md) | The serotonin and PDF circuit, and the neuromodulator layer placed from Bentley 2016 and Ripoll-Sánchez 2023 |
| [../data/locomotion/PROVENANCE.md](../data/locomotion/PROVENANCE.md) | The locomotor primitives' kinematics, from published values |
| [../data/posture/PROVENANCE.md](../data/posture/PROVENANCE.md) | The eigenworm basis and real postures behind the body's posture validation |
