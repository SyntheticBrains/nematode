# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow [Semantic Versioning](https://semver.org/): while the project is at 0.x, a minor release may contain breaking changes, and they are listed first under each release.

Releases before 0.5.0 are documented on [GitHub Releases](https://github.com/SyntheticBrains/nematode/releases); this file starts at 0.5.0.

## [Unreleased]

The Phase 9 and Phase 10 release, and the first release of the `wormbench` grader. Phase 9 took the connectome's one surviving wiring advantage, gap-junction placement, back to its gap model, and found that it was the model's degree normaliser: on corrected gap models the wild type learns no faster than its gap-only null, on the point worm or through the body, and the architecture ranking stopped there ([Phase 9 synthesis](docs/experiments/logbooks/103-phase9-synthesis.md)). Roadmap v5.0 then re-framed the project's question: how faithfully a simulated worm reproduces standard *C. elegans* behaviour, graded against real-worm data, and what the connectome contributes to that fidelity. Phase 10 built that benchmark from real worms' tracks and found it valid: at v4 it grades 24 statistics over Scheer & Bargmann's small lawn and uniform well, kinematics and activity on the lawn, López-Cruz et al.'s local search after food removal and Chen et al.'s butanone chemotaxis, and every negative fails it. The encoded connectome was fitted to pass the wild type on v4, and a mechanistic model passed on v3, but no controller predicts a mutant, and the wiring question is deferred ([Phase 10 synthesis](docs/experiments/logbooks/138-phase10-synthesis.md)). The grader ships on its own as [`wormbench`](packages/wormbench/README.md) 1.0.1 on PyPI, with the real-worm tracks on Zenodo.

*Entries are short; the logbooks are the record.*

### Breaking changes

- **A brain config its own class refuses now fails at load.** The loader validates `brain.config` against the class `brain.name` selects, so a refused config raises that class's error instead of loading as another class's fields copied onto defaults. Seven committed configs that could not build their brain now fail when loaded (`qrc_small_oracle`, `qsnnreinforce_small_oracle` and five `hybridquantumcortex_*` specials); every other config loads as before.
- **The per-connection sign table signs 16 glutamatergic edges inhibitory** ([Logbook 122](docs/experiments/logbooks/122-c2-encoder.md), addendum). After López-Cruz et al. 2019, the edges from the tax-4 chemosensory cells ASE, ASG, ASK and AWC onto AIA, and from the mec-3 mechanosensory cells FLP and AVM onto ADE, which the per-neuron rule signed excitatory. A brain built with `synapse_signs: per_connection` changes unless its config sets `sign_override_exclude: [lopezcruz2019]`, as the nine committed configs behind recorded per-connection runs do.

### Added

**Phase 10 results.** Each logbook carries its own method, registered readings and limits.

- **[Logbook 138](docs/experiments/logbooks/138-phase10-synthesis.md) — the Phase 10 synthesis: COMPLETE, GO to the Phase 11 plan review.** The benchmark is valid and a worm can be fitted to pass it, but no controller predicts a mutant, and what the wiring contributes is unread.
- **[Logbook 137](docs/experiments/logbooks/137-fidelity-table.md) — the fidelity table: no controller reproduces the worm with its mutants.** C.1 (v3)'s two predicted mutants fail, and C.2's passing mutants are all fitted.
- **[Logbook 136](docs/experiments/logbooks/136-c1-v4-fit.md) — C.1's warm-started refit on v4 stops at its pilot**: tph-1 is reachable, and tax-4 fails at every probed point.
- **[Logbook 135](docs/experiments/logbooks/135-neural-readout-ceiling.md) — the neural readout's ceiling is about 0.42**, and a linear model learned on the connectome does not beat the calibrated unfitted network under it.
- **[Logbook 134](docs/experiments/logbooks/134-fit-first-neural-readings.md) — the connectome's specific wiring predicts measured neural responses, unfitted**, beating all 19 degree-preserving rewirings.
- **[Logbook 133](docs/experiments/logbooks/133-c2-fit.md) — C.2's registered fit fails on tph-1 alone**: its wild type passes every registered population, and its serotonin mutant stays inert, as in its failed recovery control (F.0b-ii, gain 0.635).
- **[Logbooks 131](docs/experiments/logbooks/131-c2-clean-start.md) and [132](docs/experiments/logbooks/132-speed-following-reversals.md) — in probes, C.2 from a clean start passes the wild type and every mutant in 15 of 15 populations, once an on-food reversal follows the worm's own speed.** *(As a procedure fixed in advance and run fresh, it does not reliably reproduce that tph-1, [Logbook 133](docs/experiments/logbooks/133-c2-fit.md).)*
- **[Logbooks 129](docs/experiments/logbooks/129-c2-wild-type-and-mutants.md) and [130](docs/experiments/logbooks/130-c2-encoder-procedure.md) — C.2 fitted with its three mutants passes the wild type in 3 of 5 populations, with every mutant in 5 of 5**, and the procedure is built into the encoder.
- **[Logbook 128](docs/experiments/logbooks/128-c2-identifiability.md) — wild-type data do not identify C.2's perturbation responses.**
- **[Logbooks 126](docs/experiments/logbooks/126-c2-run-mode.md) and [127](docs/experiments/logbooks/127-c2-readability.md) — C.2's fit: seven designs stall, a responsiveness guard fits the lawn, and fitted on every family it reaches five of six**, with its perturbations unreadable.
- **[Logbook 125](docs/experiments/logbooks/125-c1-v4-diagnosis.md) — C.1 on v4 lacks an on-food dwelling reversal rate**; given one, a continued fit reaches v4.
- **[Logbook 124](docs/experiments/logbooks/124-benchmark-v4.md) — benchmark v4: on-lawn activity graded from positions, VALID**; C.1's v3 answer fails it on the onset rate.
- **[Logbooks 122](docs/experiments/logbooks/122-c2-encoder.md) and [123](docs/experiments/logbooks/123-c2-speed-noise.md) — C.2's compact encoder, built and fitted toward the wild type in two stages**; its slow speed noise's positive control fails on a fit that does not run.
- **[Logbook 121](docs/experiments/logbooks/121-c1-v3-fit.md) — C.1, given three missing mechanisms, passes the wild type on v3.**
- **[Logbook 120](docs/experiments/logbooks/120-benchmark-v3.md) — benchmark v3: reorientation graded from positions, VALID**; C.1b's reversal-dominated answers fail it.
- **[Logbook 119](docs/experiments/logbooks/119-c1b-fit.md) — C.1b, refitted on v2, fails the wild type**; its fits reverse for most of their time off food.
- **[Logbooks 116](docs/experiments/logbooks/116-cma-recovery-v2.md) and [117](docs/experiments/logbooks/117-cma-recovery-v2b.md) — F.0b-i on v2**: the gate fails on the fit's objective (gain 0.690), then passes with definedness ranked first (0.873).
- **[Logbook 115](docs/experiments/logbooks/115-benchmark-v2.md) — benchmark v2: VALID, and C.1's v1 pass does not survive it.**
- **[Logbook 113](docs/experiments/logbooks/113-c1-fit.md) — C.1, the mechanistic baseline, passes the wild type at the threshold** (v1).
- **[Logbook 112](docs/experiments/logbooks/112-cma-recovery.md) — sep-CMA recovers C.1's model near the wild type** (F.0b-i, gain 0.991).
- **[Logbook 111](docs/experiments/logbooks/111-npe-recovery.md) — NPE recovery fails its informative gate, which was out of reach**; C.1 is fitted by sep-CMA instead.
- **[Logbook 110](docs/experiments/logbooks/110-benchmark-controls.md) — the benchmark's controls: held-out worms pass, six negatives fail, and the benchmark is valid.**
- **Logbooks [104](docs/experiments/logbooks/104-batched-evaluator.md)–[109](docs/experiments/logbooks/109-benchmark-grading.md) and [114](docs/experiments/logbooks/114-track-import.md)** — the instrument: the batched evaluator, the reference data, the neuromodulator layer, the readouts, short slow reversals on a lawn, the grading rule and track import (below).

**Phase 9 results.**

- **[Logbook 103](docs/experiments/logbooks/103-phase9-synthesis.md) — the Phase 9 synthesis: STOP.** The connectome's wiring advantage was the gap model's normaliser, and the architecture ranking is halted.
- **[Logbook 102](docs/experiments/logbooks/102-decomposition-restated.md) — block V's decomposition restated**: Phase 8's gap-placement headline is withdrawn; the chemical wiring at the sensory–motor boundary on hard350 stands.
- **[Logbook 101](docs/experiments/logbooks/101-gap-model-question.md) — strength-normalised gap junctions show no wiring effect**; ohmic coupling leaves a small lead below the minimum, unresolved.
- **[Logbook 100](docs/experiments/logbooks/100-body-logcount.md) — through the body too, the corrected gap model shows no wiring effect**, and the STOP condition is met.
- **[Logbook 099](docs/experiments/logbooks/099-logcount-contrast.md) — on log-compressed gap counts the wild type learns no faster than its gap-only null** (+0.003 `auc_success`).
- **[Logbook 098](docs/experiments/logbooks/098-collapse-remedies.md) — no standard PPO remedy beats re-rolling the run**; late collapse follows the training stream, not the seed.
- **[Logbook 097](docs/experiments/logbooks/097-gap-substrate.md) — log-compressed counts, not ohmic coupling, carry the corrected gap model.**
- **[Logbook 096](docs/experiments/logbooks/096-collapse-diagnosis.md) — late collapse leaves no registered signature before it falls.**
- **[Logbook 095](docs/experiments/logbooks/095-gap-operating-point.md) — the gap-placement advantage is an artefact of how the model scales gap junctions.**
- **Roadmap v5.0** ([§ Phase 10](docs/roadmap.md)) — Phase 10, *fidelity on the plate*, and the project's question re-framed.

**The benchmark.**

- **Real-worm reference data** in `data/benchmark/`, each set with its licence, checksums and units in `PROVENANCE.md`: Scheer & Bargmann 2023 at 1 s with every animal's genotype, date and cohort; Calhoun 2014's and Roberts 2016's off-food search tracks; López-Cruz 2019's reorientations after food removal; and Chen et al. 2025's butanone tracks.
- **Readouts** from 1-s positions alone (`benchmark.readouts`): a reversal-onset detector for off-food tracks, López-Cruz's local-search decay, and the roaming/dwelling classifier at 1 s.
- **Grading** (`benchmark.grading`; `scripts/analysis/benchmark_grade.py batched|captures|tracks`): bands from the spread between real recording dates, at any population size; mutants (tph-1, pdfr-1, tax-4) graded on their difference from their own wild type; the distances D and D_fit. `--bands v1|v2|v3|v4` picks the version, and v4 is the default: 24 statistics at z\* 3.0781 in six families. Simulated tracks are first measured the way the real trackers report positions (`benchmark.measurement`).
- **Controls for each version** (`benchmark_controls.py`, `benchmark_v2_controls.py`, `benchmark_v3_controls.py`, `benchmark_v4_controls.py`), with the negative controllers `BatchedStateHazards` and `BatchedCurvatureWalk`.
- **Track import** (`benchmark.track_import`): CSV with a JSON sidecar, or WCON 1.0's core, graded through the same windows, classifier and bands; the contract is `data/benchmark/IMPORT.md`.

**`wormbench`.**

- **The grader as a standalone package** (`packages/wormbench/`, Python 3.10 or later, numpy and scipy only), released on its own: [`wormbench` 1.0.1](https://pypi.org/project/wormbench/1.0.1/) on PyPI (`pip install wormbench`; 1.0.0 went to TestPyPI only). `wormbench grade`, `example`, `wcon` (WCON export), `data fetch`, `data verify` and `info`. Every old import path is an alias of the same module. Its history is in its own [changelog](packages/wormbench/CHANGELOG.md).
- **The real-worm tracks on Zenodo** ([doi:10.5281/zenodo.24585652](https://doi.org/10.5281/zenodo.24585652)), fetched by `wormbench data fetch` and checked against their SHA-256s.
- **A release workflow**, `.github/workflows/release-wormbench.yml`, publishing by PyPI's trusted publishing, and a CI job testing the wheel alone on Python 3.10 at the declared minimums.

**The 1-second worm and its assays.**

- **One step's duration**: a top-level `step_seconds`; the loader converts every step-coupled constant to the configured step.
- **Locomotor primitives**: `continuous.locomotion: primitives` with `action_space: primitives` moves the point worm only by runs, reversal bouts and omega/delta turns, with kinematics from data (`data/locomotion/primitives.json`). A reversal that starts on a lawn is short and slow, as Scheer's held-out cohorts show (`continuous.reversal_kinematics: off_food` restores the off-food values), and `continuous.on_food_reversal_speed: follows_run_speed` makes it follow the run speed before it. Sensory modules `food_contact` and `locomotor_state`; non-learning `ConstantHazards` and `PirouetteHazards` controllers.
- **Assay protocols**: `environment.assay` runs an episode as phases of food and placement, with circular arenas, centred lawns and `satiety.starvation_ends_episode`; `scripts/campaigns/generate_assay_configs.py` writes Scheer's small lawn and uniform well, López-Cruz's and Calhoun's food removal, Flavell's uniform lawn and Chen's chemotaxis.
- **A batched evaluator** (`quantumnematode/evaluation/`): many frozen animals of one config in lockstep, 48–98× faster per core than the per-step runner, each animal with the per-step runner's random streams.

**Controllers and fitting.**

- **`BatchedMechanistic`**, C.1's 20-parameter closed-loop roaming/dwelling worm, with optional off-food kinematics, a per-state speed spread, an off-food κ and an on-food dwelling reversal rate.
- **C.1's fits and their recovery controls**: `scripts/analysis/c1_fit.py`, `c1b_fit.py`, `c1_v3_fit.py` and `c1_v4_fit.py`, one per benchmark version; `cma_recovery.py`, `cma_recovery_v2.py` and `npe_recovery.py` (`sbi` is in the `analysis` extra).
- **C.2's compact encoder**: `encoder: compact` on the connectome brain fixes the wiring from data and makes the brain a function of 196 named parameters (`quantumnematode.connectome.encoder`), with `encoder_speed_noise: ou`, `encoder_readout: soft_cap`, `encoder_pool_centres`, `encoder_box: graded`, `encoder_inhibition_floor` and `modulator_receptor_signs` (cited couplings in `data/neuromodulators/receptor_couplings.json`).
- **C.2's registered fit and probes** (`scripts/analysis/c2_fit.py`, `c2_encoder_probe.py`), and **C.4's fidelity table** (`scripts/analysis/fidelity_table.py`).
- **A neuromodulator layer placed from data**: `modulator_layer: data` reads `data/neuromodulators/layer.json`, placed from Bentley 2016 and Ripoll-Sánchez 2023 (55 ligand pools, 121 couples), with per-ligand timescales, gene knockouts and a receptor-placement null.

**The connectome brain and PPO (Phase 9).**

- **Gap-junction models**: `gap_weight_mode` and `gap_weight_scale`, `gap_coupling: ohmic` and `gap_normalisation: strength`, and the wirings `rewired_gap_outliers_held` and `rewired_gap_outliers_scattered`.
- **PPO stabilisers** on MLP-PPO and connectome PPO: `kl_stop_factor`, `weight_clip`, `adam_eps`/`adam_beta2` and `rollout_shuffle_stream`.

Every option is off by default and byte-identical when off. The one exception is new in this release: under locomotor primitives, a reversal on a lawn uses the on-food kinematics unless `reversal_kinematics: off_food` is set.

**Tooling.**

- **Byte-identity checks** for a change that claims to leave outputs unchanged: `scripts/analysis/speedup_identity_check.py`, `connectome_identity_check.py` and `wormbench_identity_check.py`.
- **Collapse telemetry**: `run_simulation.py --update-telemetry` and `--checkpoint-every N`, and `scripts/analysis/collapse.py`.
- **A Wormturn export**: `scripts/export_wormturn.py`, the signed connectome with the neuromodulator layer.

### Changed

- **Faster, with identical output**: a connectome brain builds in about 0.2 s after the first in a process, against 2.4 s; a v2 simulation is 1.76× faster where only statistics are read ([Logbook 116](docs/experiments/logbooks/116-cma-recovery-v2.md)), and C.1's model 1.54× faster on one core ([Logbook 118](docs/experiments/logbooks/118-v2-speedup.md)).
- **The grader imports without the simulator or PyTorch**: 0.14 s and 218 modules, against 1.5 s and 2,511. `BehaviourStep` and the benchmark's modules now live in `wormbench`, and every old import path still works.

### Security

- `certifi` 2026.7.22 → 2026.10.23 (GHSA-6q48-6c7w-9gw2, reachable only through the `qpu` extra), through a uv override of `qiskit-serverless`'s exact pin, which stays until a qiskit-serverless release pins a patched certifi ([#511](https://github.com/SyntheticBrains/nematode/pull/511)).

## [0.7.0] - 2026-10-14

The Phase 8 release, *ground, then embody*. Phase 8 took the connectome's learning-speed advantage over a degree-preserving null apart, then put the connectome in a body. On hard350 the advantage came from the null's rewired gap junctions and its rewired sensory–motor boundary; against a null that keeps both, no interior chemical-wiring advantage is detectable. What holds is **gap-junction placement**: on the thermal cell the wild type learns faster than a null whose gap junctions alone are rewired, not because of each junction's strength, and the lead survives a 12-segment kinematic body driven by the anatomical motor-to-muscle map. Measured weights and real synapse signs moved no wiring reading. Through the body the worm learns to weathervane but never klinokinesis, and on patchy lawns it learns to dwell. See the [Phase 8 synthesis](docs/experiments/logbooks/094-phase8-synthesis.md).

*Entries are short; the logbooks are the record.*

### Breaking changes

- **The chemotaxis reference set is rebuilt, and tracked runs no longer record a literature verdict** ([#435](https://github.com/SyntheticBrains/nematode/pull/435)). None of the five chemotaxis-index values in `data/chemotaxis/literature_ci_values.json` came from its cited paper; the file now holds only verified values, with what was assayed and how each was read, and its schema changes (`assay` and `read_from` replace `ci_range` and `conditions`). The tracker no longer writes `biological_ci_range`, `biological_ci_typical`, `matches_biology` or `literature_source` (kept in the schema as `None`, so older records load), and `ChemotaxisValidationBenchmark`, its fallback dataset and `ValidationResult` are removed.

### Added

**Phase 8 results.** Each logbook carries its own method, registered readings and limits.

- **[Logbook 094](docs/experiments/logbooks/094-phase8-synthesis.md) — the Phase 8 synthesis: COMPLETE, GO to a plan review.** Every criterion carries one of five statuses; the phase after 8 is designed from its inventory.
- **[Logbook 093](docs/experiments/logbooks/093-thermal-body-validation.md) — through the body, the weathervane toward the thermal setpoint is learned; klinokinesis is absent.**
- **[Logbook 092](docs/experiments/logbooks/092-thermal-body.md) — `move_wt`: the gap-junction advantage survives embodiment**, +0.126 `auc_success` over the gap-only null at target 20 and 6,000 episodes.
- **[Logbook 091](docs/experiments/logbooks/091-per-connection-signs.md) — per-connection signs: no lead, and no change from grounding or the receptor corrections.**
- **[Logbook 090](docs/experiments/logbooks/090-modulators.md) — acute *tph-1* no effect on on-lawn speed at the minimum; acute *pdfr-1* unresolved.** Both knockouts lower intake.
- **[Logbook 089](docs/experiments/logbooks/089-connectome-lawns.md) — the connectome dwells on lawns where its untrained policy never does.**
- **[Logbook 088](docs/experiments/logbooks/088-patch-leaving.md) — patch-leaving out of roaming: unreadable**, since most learners leave without ever roaming.
- **[Logbook 087](docs/experiments/logbooks/087-patchy-lawns.md) — on patchy lawns a learner that pays to move dwells.**
- **[Logbook 086](docs/experiments/logbooks/086-body-validation.md) — the body's crawl passes its body checks**; speed partial, no omega postures, the weathervane survives without the head-sweep.
- **[Logbooks 084](docs/experiments/logbooks/084-body-wiring-pilot.md) and [085](docs/experiments/logbooks/085-body-wiring.md) — the wiring contrast through the body is unresolved**, bounded near 0.044.
- **[Logbook 083](docs/experiments/logbooks/083-body-control.md) — MLP-PPO learns hard350 through the body**, at 500 steps.
- **[Logbook 082](docs/experiments/logbooks/082-plastic-gaps.md) — `placement`: the gap-junction advantage survives letting both wirings tune their gap strengths.**
- **[Logbook 081](docs/experiments/logbooks/081-body-prerequisites.md) — the body's prerequisites validate**; 8b freezes on Emmons 2024 with reversal on.
- **[Logbook 080](docs/experiments/logbooks/080-across-step-state.md) — leaky dynamics: `inferior` on both cells.**
- **[Logbook 079](docs/experiments/logbooks/079-boundary-null.md) — `boundary`, `no_lead`: against a boundary-preserving null, block V's remaining lead disappears.**
- **[Logbook 078](docs/experiments/logbooks/078-thermal-split.md) — on thermal, the null's gap junctions carry about 84% of the lead.**
- **[Logbook 077](docs/experiments/logbooks/077-thermal-null-strength.md) — the thermal null-strength panel is unreadable**: saturated, a registration error stated.
- **[Logbook 076](docs/experiments/logbooks/076-8a-synthesis.md) — the 8a synthesis: 8a complete, D20 gate GO.**
- **[Logbook 075](docs/experiments/logbooks/075-gap-only-split.md) — the null's rewired gap junctions carry A.6's move on both learners.**
- **[Logbook 074](docs/experiments/logbooks/074-null-strength-control.md) — holding the null's gap junctions and autapses halves block V's advantage**, below the registered minimum.
- **[Logbooks 072](docs/experiments/logbooks/072-measured-prior-pilot.md) and [073](docs/experiments/logbooks/073-measured-prior-contrast.md) — measured weights do not change what the wiring is worth under PPO.**
- **[Logbook 071](docs/experiments/logbooks/071-operating-point-surface.md) — the wiring advantage is depth-critical under PPO.**
- **[Logbook 070](docs/experiments/logbooks/070-init-sharing-control.md) — the wiring advantage survives a shared initialisation.**

**The connectome brain.**

- **More nulls**: `wiring: rewired_chemical_only`, `rewired_gap_junctions_held`, `rewired_boundary_held` (the boundary is the body's drive cells under `action_space: body_drive`) and `rewired_gap_junctions_only`.
- **Shared-initialisation draw modes**: `weight_draw` (`edge_order` | `dense_mask` | `per_neuron_fanin`), orthogonal to `weight_init`.
- **Measured weights**: the Creamer, Leifer & Pillow fitted weights (MIT) vendored, placed by `weight_prior` (`random` | `measured` | `measured_signs` | `measured_shuffled`) with `measured_weight_scale`.
- **Synapse signs**: `per_connection_signs()` from vendored sources (Fenyves et al. 2020 and Wormlight's physiology table), and `synapse_signs: per_connection` with `chemical_weight_scale` to calibrate a grounded sign model's operating point.
- **Plastic gap junctions**: `plastic_gaps: true`, a learnable multiplier per existing gap pair under PPO.
- **Across-step dynamics**: `dynamics: leaky`, with `membrane_tau_steps`, `input_gain` and `bptt_chunk_length`.
- **Serotonin and PDF**: `modulators: true`, a modulator field from the verified circuit in `data/neuromodulators/`, with `modulator_knockout: tph1 | pdfr1` and `modulator_satiety_link`.
- **A frozen-wiring learner**: `freeze_wiring: true`.
- **Emmons 2024 as a source**: `connectome_source: emmons_2024_hermaphrodite`, the CC BY 4.0 release, with its neuromuscular synapses (`load_emmons_2024_neuromuscular()`).

Every option is off by default and byte-identical when off.

**The body, lawns and internal state.**

- **A kinematic body**: `continuous.body_model: kinematic`, a 12-segment body with a head switch and a relay setting the rhythm, driven through `action_space: body_drive` (25 numbers). The connectome reaches it through the anatomical neuromuscular map with a learnable gain per output; `body_substeps`, `body_steering_gain` and an optional `posture_log` are configurable.
- **Signed speed**: `continuous.allow_reversal`, with `signed_speed` on every continuous brain.
- **The step's duration in worm time**: `env/worm_time.py`, 5 worm-seconds per full-speed step.
- **Patchy lawns**: `foraging.food_model: lawns`, depleting disc lawns with an odour field, intake by quality and an optional movement cost; the `internal_state` sensory module gives the brain its satiety, and run summaries gain an `intake` column.

**Validation and analysis.**

- **Body validation**: the WormPose eigenworm basis (BSD-3-Clause) and 6,655 real postures (CC BY 4.0) vendored, posture and kinematic instruments in `validation/`, and `scripts/analysis/body_validation.py` and `body_kinematics_eval.py` (which captures food or thermal behaviour with `capture_modality`).
- **Roaming, dwelling and patch-leaving**: `validation/roaming_dwelling.py`, calibrated on Scheer & Bargmann 2023's real worms, read by `scripts/analysis/lawn_states.py`.
- **Wall-proximity exclusion**: `behavioural_chemotaxis_validation.py --wall-margin-mm --arena-mm`.
- **Panel analyses**: one script per registered panel under `scripts/analysis/`, among them `thermal_body.py` (`mlp-gate`, `score`, `kinematics`, `thermotaxis`) and `sign_rung.py`.

**Rendering and tooling.**

- **A realistic replay**: `run_simulation.py --record-replay <dir>` and `scripts/replay_viewer.py`, a bright-field worm on an agar plate at 1×, 10× or 60× worm time, adapted from Wormlight (Apache-2.0; see NOTICE).
- **The continuous renderer draws the body, lawns and internal state**, and `--record-frames` saves PNG frames.
- **A pre-launch gate preflight**: `scripts/campaigns/gate_preflight.py`.
- **A campaign progress reader**: `scripts/campaigns/campaign_progress.py`; `run_campaign.py` writes a `<run>.exit` marker per run.
- **A Wormlight export**: `scripts/export_wormlight.py`.
- **A weekly literature watch**: a scheduled workflow that opens a digest issue ([docs/research/literature-watch/](docs/research/literature-watch/README.md)); it needs an `ANTHROPIC_API_KEY` repository secret.

### Fixed

- **`enforce_synapse_signs` now holds grounded signs under PPO.** It was read only by the plastic rules; no committed PPO config set it.

### Security

- `urllib3` 2.7.0 → 2.8.0 (GHSA-8988-9cw3-xx77, GHSA-gh4c-6fx4-qh6g, GHSA-vxq7-64xx-v4gw) ([#412](https://github.com/SyntheticBrains/nematode/pull/412)).
- `pyjwt` 2.13.0 → 2.15.0 and `tornado` 6.5.8 → 6.5.9, both security-fix updates ([#414](https://github.com/SyntheticBrains/nematode/pull/414), [#415](https://github.com/SyntheticBrains/nematode/pull/415)).

## [0.6.0] - 2026-09-19

The Phase 7 release. Phase 7 asked whether the wild-type *C. elegans* connectome becomes load-bearing under a biologically plausible learning rule, and **closed as a SPLIT**: no rule in that family writes this connectome to any benefit, so the pre-registered question has no interpretable answer here. What the phase established instead is that the wiring is **learning-speed-relevant under gradient descent**, and **legible to a small local readout only at a particular readout width and learning rate**. Thirty logbooks, 040 through 069, carry the evidence.

*Changelog entries in this release are deliberately short.* Earlier drafts restated each logbook's method and figures, duplicating the record they link to; the logbooks are the record, and each entry now says what changed and where to read it.

### Breaking changes

- **Resuming an evolution run now requires `--allow-unsafe-resume`** ([#392](https://github.com/SyntheticBrains/nematode/pull/392), closing the security half of [#16](https://github.com/SyntheticBrains/nematode/issues/16)). Checkpoints are Python pickles, so `--resume` on a file of unknown provenance is arbitrary code execution; the flag defaults off and the refusal fires before any file is read. The format is unchanged and #16 stays open for it — the payload carries CMA-ES and numpy generator state rather than tensors, so `weights_only=True` does not apply.

### Added

**Phase 7 results.** Each logbook carries its own method, registered readings, sensitivity and "may not be cited as" list.

- **[Logbook 069](docs/experiments/logbooks/069-phase7-synthesis.md) — the Phase 7 synthesis: closes as SPLIT.** The flagship MUST is unmet and was unmeetable in the closing scope. Every exit criterion carries one of five statuses with none unmarked, and the roadmap, README and Success Levels are flipped to terminal state.
- **[Logbook 068](docs/experiments/logbooks/068-l1b-rate-calibration.md) — `width_favours_the_shuffle_at_this_rate`.** L.1's interaction does not survive the learning rate: −0.0657 at 0.0001 against +0.2818 at 0.001, a three-way of +0.3475 on 81 of 96 seeds. At the lower rate capacity dominates (+0.6178 width main effect on 96/96) where L.1 detected none.
- **[Logbook 067](docs/experiments/logbooks/067-l4-feature-ablations.md) — both ablations read `carries_the_effect`**, L.4's qualified *carries or unlearnable*. Removing gap junctions helped **both** wirings with the null catching up; grounding the synapse signs cut learnability on both at a rate where the null was already ahead.
- **[Logbook 066](docs/experiments/logbooks/066-l4-readout-width.md) — `pooling_hid_structure`.** Widening the readout from four pooled motor classes to one weight per motor neuron gives an interaction of **+0.2818** on 96 paired seeds, with the wiring effect's sign flipping between widths and no width main effect detected.
- **[Logbook 065](docs/experiments/logbooks/065-wiring-fresh-rewiring.md) — `specific_wiring_efficiency` on both cells.** Block V replicates on rewirings fresh to both prior panels: +55.3% and +40.1%, closing a shared-nulls defect where V.3's rewirings were a subset of V.1's.
- **[Logbook 064](docs/experiments/logbooks/064-l4-frozen-features.md) — `wiring_is_inert_as_features`.** With only the motor readout learning, the wild type shows no advantage over its degree-preserving null at the registered bar, extending the degree-statistics verdict to a third learning regime.
- **[Logbook 063](docs/experiments/logbooks/063-l4-eprop.md) — `learns_without_the_substrate`.** e-prop reaches competence on the connectome (17.570 foods of 20, 52.61% full clear) on the arm whose chemical matrix is **frozen**; every arm that writes the wiring does worse by 3.9 to 15.9 foods. This is the result that closed the rule programme.
- **[Logbook 062](docs/experiments/logbooks/062-l4-frozen-readout.md) — `readout_helps_but_not_enough`.** Substituting the frozen motor readout more than doubles what the rule reaches, 3.751 to 9.639 foods, without any arm becoming competent; the readout's scale does the work, not its direction.
- **[Logbook 061](docs/experiments/logbooks/061-l4-reduced-perturbation.md) — `not_reducible`.** No perturbation set from 1208 draws down to 39 makes the rule learn the connectome, and the per-unit perturbation is now restrictable to a declared set.
- **[Logbook 060](docs/experiments/logbooks/060-l4-perturbation-scale.md) — `mixed`, and the mixture is the finding.** The rule solves a multi-step foraging cell at 8 perturbed units and collapses at 128, matching node perturbation's ~1/N scaling (Werfel, Xie & Seung 2005) against the connectome's 1208 draws per decision.
- **[Logbook 059](docs/experiments/logbooks/059-7a-shipment.md) — SPLIT-shipment.** 7a ships a systematic negative with a diagnosed cause plus block V's positive, with GO unreachable on its own clause and STOP overstating.
- **[Logbook 058](docs/experiments/logbooks/058-wiring-premise-difficulty.md) — `specific_wiring_efficiency`.** The wiring advantage survives removing the thermosensory pathway: +23.5% off time-to-competence on a hard food-only cell, so difficulty is sufficient.
- **[Logbook 057](docs/experiments/logbooks/057-wiring-premise-contrast.md) — `specific_wiring_efficiency`, the project's first positive wiring result.** Under PPO the wild-type wiring reaches competence ~35% sooner than its degree-preserving null on a cell matched to a behaviour the animal performs. A [companion probe](docs/experiments/logbooks/supporting/057-wiring-premise-contrast/probe-v2.md) scored all 64 rewirings on four graph properties fixed before looking and found none predicting learning time.
- **[Logbook 056](docs/experiments/logbooks/056-l4-ladder-reread.md) — no committed verdict changed.** Of 32 registered contrasts across eight logbooks, ten are instrument findings and the rest are not, so "the negatives are artefacts of a broken instrument" covers less than it appears to.
- **[Logbook 048](docs/experiments/logbooks/048-l4-rule-positive-control.md) — the rule fails its own positive control, control valid.** The three-factor rule does not learn a task whose analytic reference closes 99.9% of the gap, with gradient alignment at +0.009. It ran twelfth in the sequence, after seven registered panels.
- **The instrument block** that followed, recorded under [`logbooks/supporting/`](docs/experiments/logbooks/supporting/): a node-perturbation eligibility that passes the control, a σ schedule, endpoints re-evaluated with the perturbation off (bimodal), a contrast family matched to that bimodality with every committed table re-read under it, the eligibility horizon found limiting on the control and **not** transferring to a multi-step task, and a delayed control on which the three settings pinned since A.3 were finally examined.
- **[Logbook 047](docs/experiments/logbooks/047-l4-structured-instruction.md) — `no_routing_effect`**, [046](docs/experiments/logbooks/046-l4-decorrelation.md) — `no_recovery`, [045](docs/experiments/logbooks/045-l4-consolidation.md) — no mechanism passes the clone assay, [044](docs/experiments/logbooks/044-l4-atlas-signs.md) — `degree_statistics`, [043](docs/experiments/logbooks/043-l4-warm-start.md) — `sanity_floor_fail` + `rule_destroys_clone`, [042](docs/experiments/logbooks/042-l4-panel3.md) and [041](docs/experiments/logbooks/041-l4-panel2.md) — `inconclusive`, [040](docs/experiments/logbooks/040-l4-panel.md) — `sanity_floor_fail`, the D10 2×2 panel that opened the phase.

**Substrate and rule mechanisms**, each default-off or byte-identical when unused.

- A reward-modulated **three-factor learning rule** for the connectome brain (`learning_rule: three_factor`): `dw = eta * delta * E` over the topology's eligibility trace, with the **matched-rule MLP arm** (`mlpppo` with the same rule over a `PlasticTopology` seam) and the **plastic degree-preserving rewired-null arm** as the 2×2's other cells.
- Two **sanity-floor arms**: frozen weights (`freeze_updates`) and an unmodulated Hebbian rule, both configured under the plasticity rule so they share its readout.
- **Consolidation** (`plasticity_consolidation`), **decorrelating terms** (`plasticity_decorrelation`) and a **routed third factor** (`third_factor: pathway`) — brakes and instruction mechanisms for a rule shown to drift at near-constant speed through good policies.
- **Neurotransmitter identities** on the substrate (`synapse_signs: atlas`), grounding 3,176 of 3,709 chemical synapses in the Wang et al. 2024 atlas — the fidelity ladder's first rung.
- **`readout_width`** (`pooled` | `per_neuron`), the per-neuron readout initialised by expanding the pooled draw so the RNG stream is untouched and both widths compute the same policy at initialisation.
- **`weight_init`** (`degree_scaled` | `count_scaled`), **`plastic_layers`** on the MLP brain, a **centred compressed modulator**, substrate-invariant scaling switches, the **L4 trace substrate** (`enable_activity_traces`, `trace_decay`), and the **state-dependent continuous action std** (`continuous_std_mode`, roadmap D7).
- **Warm-start tooling** for the imitation arm: `ConnectomePPOBrain` implements `WeightPersistence` (closes [#308](https://github.com/SyntheticBrains/nematode/issues/308)), and `weights_path` accepts `{seed}` so one committed config addresses one clone file per seed.

**Tooling and infrastructure.**

- **`scripts/run_campaign.py`**: configs crossed with seeds under a bounded worker pool, with per-run logs, progress reporting, a status summary and `--dry-run`.
- **Two campaign output controls** on `scripts/run_simulation.py`: `--no-detailed-export` and `--no-file-log`. Without them a run writes ~0.7 GB outside the campaign directory and a 768-run campaign needs ~500 GB, which filled the volume mid-campaign; with them, ~17 MB, and everything an analysis reads is still written.
- **`DeviceType.MPS`** (`--device mps`) so Apple's Metal GPU is selectable and measurable, with `bench_device_backends.py` and `bench_campaign_parallelism.py` behind the device and worker-count guidance.
- **CI test shards rebalanced**, with a guard so the balance cannot rot silently: 1,376 of 5,520 tests had no recorded duration and `pytest-split` was budgeting each at the mean of the rest.

### Changed

- Roadmap decision D2 described the frozen-weights baseline as using "Cook-2019 synapse-count-derived initial weights". The implementation has never done that: the connectome supplies which edges exist, weights along them are drawn `N(0, 1/sqrt(chemical in-degree))`, and the EM synapse count never reaches the weight matrix. The wording is corrected rather than the initialisation — changing the latter would move a substrate that existing results are recorded against.

- Connectome configs now appear in the smoke-test set. None did previously, so the plasticity code path — per-step updates, no value head, an unused rollout buffer — had never been exercised end to end through the run entry point.

- The connectome's eligibility trace is now temporally causal: `E <- lambda*E + M(h_prev (x) h)`, taking the previous step's settled state as the pre-synaptic factor instead of a same-step symmetric outer product. The adjacency mask already separated non-reciprocal edges, but the symmetric form gave both directions of a reciprocal pair identical eligibility and encoded no temporal order, which a rule crediting synapses for causing activity cannot rest on. The first step of an episode now accrues no eligibility. Traces remain off by default and, while no rule consumes them, training remains bit-identical to traces-off.

- Device selection is validated before brain construction and fails with an actionable message. Previously `--device gpu` mapped unconditionally to CUDA and crashed on non-CUDA hosts with a raw `AssertionError: Torch not compiled with CUDA enabled`, despite being an advertised CLI choice. Selection is also checked against the brain family: a PyTorch-only accelerator is rejected for quantum brains, which pass the device to Qiskit — `AerSimulator(device="MPS")` is accepted *without raising*, so the bogus backend would otherwise have been recorded in experiment metadata as though it were real.

- Connectome-brain runs now record the mean policy loss per PPO update in the tracked `losses` telemetry (the house convention; the brain previously recorded no loss), so connectome session exports gain a `losses` column.

## [0.5.0] - 2026-08-23

The pre-Phase-7 housekeeping release. It takes the platform that closed Phase 6a, removes the research tooling the platform no longer needs, moves the toolchain to Python 3.13 and Qiskit 2, and prepares the repository for a wider audience.

### Breaking changes

- **NematodeBench removed** ([#274](https://github.com/SyntheticBrains/nematode/pull/274)). The curated benchmark submission workflow, validation, category and leaderboard generation, `BENCHMARKS.md` and `docs/nematodebench/` are gone: across Phases 5 and 6 the architecture-comparison protocol read `--track-experiment` output directly and never used them. The one live component, the convergence detector and composite score, moved to `quantumnematode.experiment.convergence`; the 72 tracked sessions behind the old submissions were migrated to `artifacts/experiments/`; the `composite_benchmark_score` key is unchanged so historical artifacts still load.
- **`qqlearning` brain retired** ([#286](https://github.com/SyntheticBrains/nematode/pull/286)). `QQLearningBrain` was the last architecture with no path to the Phase 7 plasticity work. Configs with `brain.name: qqlearning` no longer load (the loader rejects unregistered brain types); 26 architectures remain, and every count in the docs is now derived from the registry. No tracked artifact carries the retired name.
- **Python 3.13 only, and the `gpu` extra moves to CUDA 11** ([#289](https://github.com/SyntheticBrains/nematode/pull/289)). `requires-python` is `>=3.13,<3.14` (it was pinned to 3.12). The CUDA-12 `qiskit-aer-gpu` build was abandoned upstream after 0.15.1 and has no 3.13 wheel, so the `gpu` extra now installs `qiskit-aer-gpu-cu11` (0.17.2): the bundled CUDA runtime goes from 12 to 11, which needs an NVIDIA driver ≥ 450 rather than ≥ 525. Every dependency was refreshed at the same time (torch 2.13, numpy 2.5, pydantic 2.13, scipy 1.18, scikit-learn 1.9, matplotlib 3.11, rich 15, optuna 4.9).
- **Qiskit 2.x** ([#290](https://github.com/SyntheticBrains/nematode/pull/290)). `qiskit>=2.0,<3`. The platform needed no code changes (it never used the APIs Qiskit 2 removed), but anything of yours built on Qiskit 1.x removals — `BackendV1`, `execute()`, `bind_parameters()`, `opflow`, `qiskit.pulse` — must migrate. Test-suite deprecation warnings dropped from ~382,000 to ~200.

### Added

- A README and documentation set designed for the project's public baseline ([#292](https://github.com/SyntheticBrains/nematode/pull/292)): the README leads with the research question and evidence-linked results; new [usage guide](docs/usage.md), [architecture catalogue](docs/architectures.md), [visualisation reference](docs/visualization.md) and [docs index](docs/README.md); `CITATION.cff`.
- `.lfsconfig` so a fresh clone fetches only the 4 MB of connectome data the code reads, not the 620 MB of curated logbook artifacts; `git lfs pull --include='artifacts/**'` fetches those on demand and CI fetches everything ([#294](https://github.com/SyntheticBrains/nematode/pull/294)).
- `SECURITY.md`, issue forms, a pull-request template, Dependabot for the lock file and GitHub Actions, package authorship metadata and a `.python-version` pin ([#296](https://github.com/SyntheticBrains/nematode/pull/296)).
- This changelog.

### Changed

- The Docker image installs the `torch` extra and copies `configs/` and `data/`, so every brain, scenario and the connectome data are available inside it; it is x86_64-only because the CUDA-11 Aer wheel is ([#292](https://github.com/SyntheticBrains/nematode/pull/292)).
- The continuous-2D screenshot export renders a short walk so the worm's body trail is visible; both documentation screenshots regenerated ([#292](https://github.com/SyntheticBrains/nematode/pull/292)).
- The remaining PPO and REINFORCE brains share `_policy.py` ([#275](https://github.com/SyntheticBrains/nematode/pull/275)); the Phase 5 campaign aggregators share `_common.py` ([#278](https://github.com/SyntheticBrains/nematode/pull/278)).
- CI: the test job runs in about five minutes instead of eighteen, split across five `pytest-split` shards with one BLAS thread per worker ([#288](https://github.com/SyntheticBrains/nematode/pull/288)); the workflows use the current Node 24 action majors ([#297](https://github.com/SyntheticBrains/nematode/pull/297), [#298](https://github.com/SyntheticBrains/nematode/pull/298)); the `phase6-tracking` change passes strict OpenSpec validation ([#287](https://github.com/SyntheticBrains/nematode/pull/287)).
- Documentation drift corrected across the roadmap, README, CONTRIBUTING and AGENTS.md, with the brain counts re-derived from the registry ([#280](https://github.com/SyntheticBrains/nematode/pull/280)); the Claude Code skills carry measured rather than guessed machine numbers ([#291](https://github.com/SyntheticBrains/nematode/pull/291)).

### Fixed

- `spikingreinforce`'s policy gradient was identically zero, so the brain never learned; it trains now, which changes its results relative to earlier logbooks ([#283](https://github.com/SyntheticBrains/nematode/pull/283)).
- The M3/M4 campaign aggregators reported generation-to-target one generation too high; the affected logbooks carry a correction note ([#284](https://github.com/SyntheticBrains/nematode/pull/284)).
- A partial F0 baseline override (or a mistyped `--campaign-root`) now fails closed instead of silently mixing baselines ([#285](https://github.com/SyntheticBrains/nematode/pull/285)).
- A flaky optimizer-identity test that compared `id()`s across object lifetimes, and two warnings the suite emitted on every run ([#293](https://github.com/SyntheticBrains/nematode/pull/293)).

### Removed

- The dead random-predator sprite; `PredatorType` rendering is now exhaustive ([#281](https://github.com/SyntheticBrains/nematode/pull/281)).
- `docs/OPTIMIZATION_METHODS.md`, which still used pre-rename class names; its guidance lives in [docs/architectures.md](docs/architectures.md) ([#292](https://github.com/SyntheticBrains/nematode/pull/292)).

### Security

- `pyarrow` 21.0.0 (CVE-2026-25087, reachable only through the `qpu` extra) is gone from the lock file: `qiskit-ibm-catalog` 0.16 → 0.19 and `qiskit-serverless` 0.32 → 0.35 no longer depend on it, and the catalog floor is raised to 0.19 ([#295](https://github.com/SyntheticBrains/nematode/pull/295)).
- `SECURITY.md` documents private vulnerability reporting and what is unsafe to load from untrusted sources ([#296](https://github.com/SyntheticBrains/nematode/pull/296)).

[0.5.0]: https://github.com/SyntheticBrains/nematode/compare/v0.4.0...v0.5.0
[0.6.0]: https://github.com/SyntheticBrains/nematode/compare/v0.5.0...v0.6.0
[0.7.0]: https://github.com/SyntheticBrains/nematode/compare/v0.6.0...v0.7.0
[unreleased]: https://github.com/SyntheticBrains/nematode/compare/v0.7.0...HEAD
