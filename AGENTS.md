# Quantum Nematode — AI Assistant Instructions

## Project Overview

Quantum Nematode is a closed-loop sensory-motor simulation platform: a simulated C. elegans forages, evades predators and navigates thermal/oxygen gradients while a pluggable brain architecture (MLP, recurrent, spiking, reservoir, quantum, hybrid, GA-evolved, or constrained to the real 302-neuron connectome) is trained by learning, evolution or fitting. The research question (re-framed 2026-10-23, roadmap v5.0, D33) is **how faithfully a simulated worm reproduces standard C. elegans behaviour, graded against real-worm data, and what the connectome contributes to that fidelity**. Architecture families are the comparison's controllers, compared on fidelity; quantum circuits are one family, not the organising principle (see docs/roadmap.md). Until v5.0 the question was which architecture best learns nematode behaviours; Phase 9 closed that line when the connectome's learning-speed advantage over its nulls turned out to be an artefact of the gap model (Logbook 103). Phase 10 graded simulated worms on standard on-lawn assays and closed on 2026-11-27 (Logbook 138): the benchmark is valid, a connectome controller can be fitted to pass its wild type, no controller predicts a mutant, and the wiring question is deferred to a later phase. Its publication is the `wormbench` release (1.0.1 on PyPI, 2026-11-28; the tracks on Zenodo, doi:10.5281/zenodo.24585652); the body and posture open Phase 11, after its plan review. Thermal, predator, multi-agent and oxygen scenarios are frozen (kept and tested, not extended; thermotaxis is the second tier).

## Tech Stack

- Python 3.13 (strictly >=3.13,\<3.14); the `wormbench` package alone supports Python 3.10 or later, with numpy and scipy only
- Quantum: Qiskit 2.x, Qiskit-Aer 0.17+, Qiskit-IBMRuntime 0.40+ (QPU)
- Classical ML: PyTorch 2.7+
- Config: Pydantic 2.11.4+, PyYAML 6.0+
- Viz: Matplotlib 3.10+, Rich 13.0+
- Optimization: CMA 4.0+ (evolutionary), custom gradient methods (PSR)
- Tooling: uv, Ruff, Pyright, pytest, pre-commit
- Docker with NVIDIA Container Toolkit for GPU support

## Common Commands

- Install: `uv sync --extra cpu --extra torch --extra pixel --extra analysis --extra qpu` (all five extras: `uv sync` is exact, so without `pixel` the renderer tests skip silently, without `analysis` there is no `sbi`, and pyright needs `qpu`)
- Test (default, excludes nightly): `uv run pytest -m "not nightly"`
- Test (pre-commit subset, fast): `uv run pytest -m "not smoke and not nightly and not slow"`
- Test (slow integration only): `uv run pytest -m slow -v`
- Test (smoke only): `uv run pytest -m smoke -v`
- Test (nightly E2E only): `uv run pytest -m nightly -v`
- Test (all, including nightly): `uv run pytest`
- Lint/format: `uv run pre-commit run -a`
- Run simulation: `uv run ./scripts/run_simulation.py --config ./configs/scenarios/<scenario>/<config>.yml`
- Export the connectome for Wormlight: `uv run python scripts/export_wormlight.py --out <path>` (refuses uncommitted changes to tracked files unless `--allow-dirty`)
- Export the signed connectome and the neuromodulator layer for Wormturn: `uv run python scripts/export_wormturn.py --out <path>` (refuses uncommitted changes to tracked files unless `--allow-dirty`)
- Real-worm behavioural-chemotaxis validation: set `sensing.capture_behaviour: true` in a foraging config (default `false` — a byte-identical no-op when off) to log a per-run behavioural trajectory to `exports/<session>/session/data/behaviour_capture.json`, then grade the klinokinesis + weathervane bias curves against the *C. elegans* literature with `uv run python scripts/analysis/behavioural_chemotaxis_validation.py --manifest <manifest.txt, one "<seed> <behaviour_capture.json>" pair per line> --tail-runs 100 --out <behavioural_curves.json> [--figure-dir <dir>] [--theta-sharp 0.45] [--wall-margin-mm 1.0 --arena-mm <world_size_mm>]` (the `--out` path writes the graded summary JSON; the wall margin drops transitions near an arena edge, where the clamped worm slides along the wall, and C.3 and D.1 require it; see [Logbook 035](docs/experiments/logbooks/035-realworm-chemotaxis-validation.md)). For **thermotaxis** validation set `sensing.capture_behaviour_modality: thermotaxis` (default `food`; it makes the captured drive the homeostatic thermal setpoint error `−|T−Tc|` so the same bias curves apply) and pass `--modality thermotaxis` to the harness (grades against the thermal `data/thermotaxis/` reference set; [Logbook 036](docs/experiments/logbooks/036-realworm-thermotaxis-validation.md)).
- **Campaigns: always pass the output controls.** `uv run python scripts/run_campaign.py --config <cfg> [--config ...] --seeds 1-96 --runs 3000 --output-dir campaigns/<name> -- --theme headless --track-experiment --no-detailed-export --no-file-log`. **The two `--no-*` flags are not optional at campaign scale**: without them each run writes ~0.7 GB *outside* the campaign directory (~440 MB of step-level exports plus ~256 MB of verbose log), so a 768-run campaign needs ~500 GB and will fill the volume — which is exactly what happened at run 337 of the L.1 campaign. With them a run writes ~17 MB and everything any analysis reads is still produced: the per-run summary CSVs, `weights/final.pt`, and the tracked-experiment record. The campaign directory itself is small, which is why the cost is invisible until it is not.
- One step's duration: a top-level `step_seconds` (unset by default; byte-identical when unset) sets the point worm's step in worm-seconds, e.g. `step_seconds: 1.0` for the 1-s worm (`configs/scenarios/foraging/*_lawns_*_s1.yml`). The config's per-step values stay written for its reference step, a full-speed step of its `max_step_mm` at 0.2 mm/s (5 s at 1 mm), and the loader converts every step-coupled constant to the configured step (`quantumnematode/utils/step_duration.py`), printing a `Step:` line and each conversion to stdout. Accepted only for the continuous point worm on lawns with `connectomeppo` or `mlpppo` and no within-lifetime plasticity; refused at load elsewhere.
- Locomotor primitives: `environment.continuous.locomotion: primitives` with `brain.config.action_space: primitives` (connectome or MLP-PPO, `signed_speed: false`, `allow_reversal: false`, and `step_seconds: 1.0` or shorter) moves the point worm only by runs, reversal bouts and omega/delta turns. Their kinematics are fixed from data (`data/locomotion/primitives.json`): published off-food values, and for a reversal that starts on a lawn the shorter, slower on-food values measured on Scheer's held-out cohorts (`continuous.reversal_kinematics: off_food` restores the off-food values everywhere, reproducing runs before 2026-10-24; Logbook 108). `continuous.on_food_reversal_speed: follows_run_speed` (default `pooled`, byte-identical) makes an on-food reversal back at a speed interpolated between measured knots at v, the mean run speed over the last 3 s of runs, flat beyond the outermost (0.0484 mm/s for a still worm to 0.0961 for the fastest), as real on-lawn reversals follow the speed before them (held-out cohorts; `scripts/analysis/reversal_speed_coupling.py`; Logbook 132). Each second the brain emits run speed, run curvature and three start hazards, and the environment draws the bouts (`quantumnematode/env/primitives.py`). Sensory modules `food_contact` (lawn density under the worm) and `locomotor_state` (the running primitive) are optional. `quantumnematode/env/hazard_controllers.py` holds the non-learning `ConstantHazards` (Poisson) and `PirouetteHazards` controllers. Configs: `configs/scenarios/foraging/*_s1_primitives.yml`.
- Assay protocols: `environment.assay` runs an episode as phases of food (`lawns` or `none`) and placement (`keep`, `lawn_centre`, `off_lawn`, `arena_centre`). The worm's satiety and the controller's state carry across phases. One phase is the assay phase, with an `analysis_from_s`. Assays need `satiety.starvation_ends_episode: false`, `lawns.start_on_lawn: false`, and `max_steps` × the reference step equal to the phases' length. `continuous.arena_shape: circle` (with `lawns.placement: centre`) gives round wells and plates. `scripts/campaigns/generate_assay_configs.py` writes Scheer's small lawn and uniform well, López-Cruz's and Calhoun's food removal, and Flavell's uniform lawn on the 1-s primitives worm (`configs/scenarios/foraging/*_assay_*.yml`). Captures record each step's `phase` and the protocol's timing.
- Batched evaluation: `quantumnematode/evaluation/` runs N frozen animals of one config in lockstep. `BatchedEvaluator(config, controller, seeds).run()` returns compact per-step arrays, and `to_behaviour_steps(i)` gives one animal's `BehaviourStep` series. Controllers are `BatchedMLP`, `BatchedConnectome`, `BatchedConstantHazards` and `BatchedPirouetteHazards`. Each animal has the per-step runner's random streams for its seed, so it behaves the same alone or in any batch. It covers the 1-s primitives worm on centred lawns or assays (klinotaxis + STAM + fold-change, Fick kernel, satiety never ending the episode) and refuses anything else. 48–98× faster per core than the per-step runner (Logbook 104). Parity tests in `tests/quantumnematode_tests/evaluation/` run on every PR, except the full-assay statistic checks on Flavell's lawn, which run nightly. Re-time with `uv run python scripts/analysis/batched_cost_pilot.py --out <json>`. A speed-up of C.1's simulation path must leave every output byte-identical: run `uv run python scripts/analysis/speedup_identity_check.py baseline` on the code before the change and `... check` after (v2 with and without presence, v1, and stored records; Logbook 118).
- Neuromodulator layer from data: with `modulators: true`, `modulator_layer: data` on the connectome brain replaces the hand-placed serotonin/PDF field (the default, unchanged) with `data/neuromodulators/layer.json`, placed from Bentley 2016 and Ripoll-Sánchez 2023 (CC BY; provenance in its `PROVENANCE.md`): one pool per ligand (55), 121 couples with every cell tagged by source, 16 fixed monoamine signs (`receptor_signs.json`) and 105 fitted signed strengths starting at 0. Knobs: `modulator_timescales_s` (per ligand, default 60 s), `modulator_knockout_genes` (producer gene holds its ligand at zero, receptor gene removes its couples), `modulator_layer_sensitivity: ec50_500nM`, `modulator_layer_sources`, and the receptor-placement null (`modulator_layer_null: receptor_placement` + `modulator_layer_null_seed`). Rebuild with `scripts/analysis/extract_neuromodulator_layer.py fetch|synonyms|build`; audit the null with `scripts/analysis/neuromodulator_null_audit.py audit` (Logbook 106). Neuron classes: `quantumnematode.connectome.neuron_classes` (the 118).
- **Benchmark package `wormbench`** (`packages/wormbench/`, a uv workspace member: Python 3.10 or later, numpy and scipy only; `quantum-nematode` depends on it). It holds the grader: `readouts`, `grading`, `measurement`, `chemotaxis`, `track_import`, `track_export` (the CSV and WCON writers), `references`, `classifier` (was `validation.roaming_dwelling`), `records` (`BehaviourStep`), `odour_field`, `bands` (band loaders, bands at a size), `builder` (`BandBuilder`) and `grade` (every version's population grading). Every old path (`quantumnematode.benchmark.*`, `validation.roaming_dwelling`, `report.behaviour`, `evaluation.odour_field`) is an alias of the same module, typed through `.pyi` stubs, and `benchmark_grade.py`/`benchmark_bands.py` re-export every moved name. **Data resolve when used, never at import** (`wormbench.data`): the host root first (`quantumnematode/__init__.py` sets this repository's `data/`), then `WORMBENCH_DATA`, the package's copies, and the fetched cache; nothing downloads except `wormbench data fetch` (from the Zenodo record doi:10.5281/zenodo.24585652, the tracks' version 1, resuming a transfer the server cuts short), and a listed file from the last three sources is checked against the package's SHA-256 manifest. After changing a file under `data/benchmark/` or `data/roaming_dwelling/` that the package copies, or a controls record, run `uv run python scripts/analysis/wormbench_records.py build && ... sync`, then `... check` (a test fails on a stale copy, record or manifest); `scripts/analysis/wormbench_example.py build|check` rebuilds the shipped example. Commands: `uv run wormbench grade --manifest <file> --out <json> [--bands v1|v2|v3|v4] [--simulated]` (the same function `benchmark_grade.py tracks` calls, plus a `wormbench` provenance block; it defaults to v4 in every 1.x release, while the scripts keep their newest-VALID default), `wormbench example --out <dir>`, `wormbench wcon --tracks <csv> --out <wcon> [--simulated]` (a CSV's tracks as WCON 1.0, refused unless it reads back unchanged), `wormbench data fetch|verify`, `wormbench info`. A change that moves or edits grading code runs `scripts/analysis/wormbench_identity_check.py baseline|check --tree <tree> --build <shared dir>` (each stage in its own tree and environment, run with that tree's `.venv/bin/python -I`; it refuses modules outside its tree). **A fit manifest that digests the package source digests `packages/wormbench/wormbench` too**, since the grader no longer lives in `quantumnematode`.
- Benchmark reference data: `data/benchmark/` vendors derived real-worm sets (licences, checksums and units in its `PROVENANCE.md`): Scheer & Bargmann 2023 at 1 s with per-frame flags and every animal's genotype, date and cohort side (`scheer2023_split.json`, recording dates, a third held out of the bands); Calhoun 2014 and Roberts 2016 off-food search tracks; López-Cruz 2019's reorientations per 2-min bin. Read them with `quantumnematode.benchmark.references` (`load_scheer_tracks`, `load_search_tracks`, `load_lopezcruz`). Re-derive from the deposits (kept off-repo) with `scripts/analysis/extract_benchmark_references.py <source>`; the primitives' on-food check is `scripts/analysis/primitive_kinematics_check.py` (Logbook 105).
- Benchmark readouts: `quantumnematode.benchmark.readouts` reads 1-s positions alone, real or simulated: `reversal_onsets` (reversal-initiated reorientations; tuned on Roberts 2016 and registered for **off-food tracks only**, since it misses most on-lawn reversals), `reference_onsets`, `match_onsets`, `local_search_curve`, `fit_local_search` (alpha·exp(−gamma(t−t0))+beta) and `quintile_ratio`. The López-Cruz reference (gamma 0.114/min over 8–44 min after removal) is `data/benchmark/lopezcruz2019_reference.json`. Re-run the detector's calibration with `scripts/analysis/reorientation_detector_calibration.py --out <json>` and the reference with `scripts/analysis/local_search_reference.py --out <json>` (Logbook 107).
- Benchmark grading (v1, frozen; the default is v4, below): `quantumnematode.benchmark.grading` reads real and simulated 1-s tracks through one window rule, tallies each animal, and grades 11 wild-type statistics: six on Scheer's small lawn (roaming fraction, censoring-aware roaming and dwelling bout means, exits per on-lawn hour, pre-exit roaming enrichment, time on the lawn), three on Scheer's uniform well (protocol `uniform_lawn_scheer`: roaming fraction and both bout means) and two on López-Cruz's local search (γ and the early/late ratio). Mutants (tph-1, pdfr-1, tax-4) are graded on their difference from their own wild type, a mutant's direction resolved by a paired t-test across recording dates; local search must be present (early/late above 1 at its own lower bound). Bands are between-date (`data/benchmark/bands.json`, built from band cohorts only by `scripts/analysis/benchmark_bands.py`, with the classifier from `scripts/analysis/benchmark_classifier.py`). Each family is graded at its own size: on the vendored bands at 64 or 200, and otherwise on bands built on demand (each band on its own generator). v1 passes a wild-type statistic at |z| ≤ 2.84 (Bonferroni over 11) and a mutant's at |z| ≤ 2.24 (Bonferroni over its two); D is the weighted RMS of the capped z's, and D_fit leaves local search out (Logbook 109). Grade with `uv run python scripts/analysis/benchmark_grade.py batched --controller constant|pirouette --seeds <a-b> --out <json>`, `... captures --manifest <"<family> <behaviour_capture.json>" per line> --out <json>`, or external tracks with `... tracks --manifest <"<family> <tracks.csv|tracks.wcon>" per line> --out <json>` (`quantumnematode.benchmark.track_import`: CSV with a `<file>.json` sidecar, or WCON 1.0's core with an `@nematode_benchmark` object; contract in `data/benchmark/IMPORT.md`; round trip `scripts/analysis/track_import_check.py`, Logbook 114). The controls (`scripts/analysis/benchmark_controls.py pilot|read`; `read` refuses until its launch record is committed) found it VALID: held-out real worms pass and six negatives fail (Logbook 110). Negative controllers: `BatchedStateHazards` (open-loop two-state worm; optional per-state `speed_log_sd` and `reversal_speed_coupling`, both off by default) and `BatchedCurvatureWalk`.
- C.1, the mechanistic baseline: `BatchedMechanistic` (20 parameters, `MECHANISTIC_PARAMETERS`; optional `v_off_food`, `sigma_off_food`, `speed_log_sd_dwell`/`speed_log_sd_roam`, `kappa_off_food` and `r_dwell_on_food` (Logbook 125), each off by default and byte-identical when off). It is fitted by sep-CMA from the box centre (D38 as amended): NPE's recovery, `scripts/analysis/npe_recovery.py pilot|simulate|train|diagnose` (needs the `analysis` extra's `sbi`), failed its informative gate, and `npe_region_check.py` records why (Logbook 111). Its fits, one per benchmark version: `scripts/analysis/c1_fit.py run|grade|read|smoke` (v1, passes 3 of 5, Logbook 113), `c1b_fit.py pilot|probe-wide|run|select|grade|read|smoke|probe-kappa|diagnose-reversals` (v2, fails, Logbook 119), `c1_v3_fit.py recovery-preflight|recovery|run|select|grade|read|smoke` (v3, passes, Logbook 121) and `c1_v4_fit.py pilot|recovery-preflight|recovery|run|select|grade|read|loo|smoke` (v4, closed at its pilot's no-go, so every later stage refuses, Logbook 136). Its recovery controls: `scripts/analysis/cma_recovery.py preflight|pilot|run|read|smoke` (v1; F.0b-i passed, gain 0.991, Logbook 112) and `cma_recovery_v2.py gate-probe|objective-probe|preflight|pilot|run|read|diagnose|smoke`, with attempt 2's `probe-b|diagnose-b|probe-b2|run-b|read-b` (v2, on `predictive_distance_v2`; failed at 0.690, then passed at 0.873 ranking candidates by `ranking_score`, `Objective(ranking=True)`: 1,000 × statistics undefined in one record + the weighted RMS of 10·asinh(z/10), Logbooks 116–117). D37's mutant encoding table for every controller is `docs/experiments/logbooks/supporting/113-c1-fit/encodings.md`.
- Benchmark v2 and v3 (frozen; each found VALID): `benchmark_grade.py --bands v1|v2|v3|v4` reproduces every past verdict. **v2** (Logbook 115) grades 19 statistics at z\* 3.0078 in five equal families: v1's three, **kinematics** (dwelling and roaming 10-s window speed medians and 90th percentiles, roaming-bout survival at 5 min; the on-lawn reversal rate at a 0.03 mm floor is reported, not graded) and **chemotaxis** on Chen et al. 2025's naive N2 butanone tracks (`data/benchmark/chen2025_tracks.npz`, `load_chen_tracks`; assay `chemotaxis_chen` with `evaluation.odour_field.chen_field`; drift, log quintile ratio and weathervane slope, and chemotaxis must be present). Simulated tracks pass through `quantumnematode.benchmark.measurement` before windowing: on Scheer's rig resampled real paused-worm noise (`data/benchmark/scheer_tracker_noise.npz`) and losses as runs, and on Chen's rig jitter and fragments. Local search and imported tracks are never measured. **v3** (Logbook 120) grades 21 statistics at z\* 3.0381: local search's **log** early/late ratio (`LOCAL_SEARCH_STATISTICS_V3`, `grading.log_search_statistics`), and in chemotaxis the off-food detector's **onset rate** per minute and the **direction-flip fraction** (consecutive 1-s displacements, both at least 0.03 mm, turning more than 135°; `chemotaxis.CHEMOTAXIS_STATISTICS_V3`, `direction_flips`, `pooled_statistics_v3`). The share of seconds in a reversal bout is reported beside the grade where steps carry primitive labels (`grading.reversal_share`, against Roberts' 17.3%), never graded. Bands `data/benchmark/bands_v2.json` and `bands_v3.json` (`benchmark_bands.py --version v2|v3`); controls `scripts/analysis/benchmark_v2_controls.py` and `benchmark_v3_controls.py round-trip|oracle-fit|read|verdict|preflight`, with every reading in `docs/experiments/logbooks/supporting/115-benchmark-v2/` and `.../120-benchmark-v3/`. `grade_populations_v2(presence=False)` gives a fit's statistics and distances without the presence resamples, with identical output (registered gradings keep presence); `grade_populations_v3` grades v3; `simulate_v2(..., assays=)` runs a subset of the assays with the same animals.
- Benchmark v4 (the default; Logbook 124 found it VALID): 24 statistics at z\* 3.0781 in six equal families, v3's five and **activity** on Scheer's small lawn, read from 1-s positions over each worm's on-lawn seconds (both samples present and on the lawn): the **pause share** (displacements below 0.02 mm, the floor `grading.pause_floor` derives from the still-worm library), the on-lawn **direction-flip fraction** and the on-lawn **onset rate** per minute (`grading.activity_tally`, `activity_statistics`, `ACTIVITY_STATISTICS`). D_fit is the small lawn, the uniform well, kinematics and activity. Grade with `grade_populations_v4`, which refuses bands whose onset decision is not recorded (`activity_graded`); every v3 band is carried byte-identical. Bands: `data/benchmark/bands_v4.json` (`benchmark_bands.py --version v4`, then `--onset graded|reported`, which follows the oracle's check alone). Controls: `scripts/analysis/benchmark_v4_controls.py preflight|round-trip|oracle-fit|oracle-check|read|verdict`, with every reading in `docs/experiments/logbooks/supporting/124-benchmark-v4/`. Speed-following grading arms: `benchmark_grade.ARMS["mlp_following"]` (configs `mlpppo_..._internal_s1_primitives_following_assay_*.yml`, generated as the `internal_following` arm) and `ARMS["connectome_c2_following"]`. C.4's fidelity table is `scripts/analysis/fidelity_table.py grade|table|smoke` (Logbook 137): `table` writes `docs/experiments/logbooks/supporting/137-fidelity-table/fidelity_table.{json,md}` from bound records, checking every recount. The Phase 10 synthesis's two description probes, `docs/experiments/logbooks/supporting/138-phase10-synthesis/probes/identifiability.py.txt` and `likelihood_check.py.txt`, re-run from the repository root in seconds (Logbook 138).
- C.2's compact encoder: `encoder: compact` on the connectome brain (with `modulator_layer: data`, `modulator_satiety_link: true`, `synapse_signs: per_connection` and `gap_normalisation: strength`; every weight setting it replaces is refused away from its default, and the loader refuses a `modulator_step_seconds` that does not come to the configured step) fixes the wiring from data and makes the brain a function of 196 named parameters (132 under `encoder_modulator_tying: per_ligand`): the wiring scales and unsigned-group gains, class-tied sensory gains, the pooled readout with a bias, the log-SDs, the data layer's strengths and a monoamine and a neuropeptide timescale. `quantumnematode.connectome.encoder`: `encoder_entries`, `write_encoder(brain, z)`/`read_encoder`, `tax4_vector`, compositions with a candidate axis. It builds on every null wiring and under the receptor-placement null with the wild type's names. The encoded settling takes NumPy's float32 `tanh` where no gradient is taken. Options, each off by default and byte-identical when off: `encoder_speed_noise: ou` (slow Ornstein–Uhlenbeck speed noise; its timescale `timescale.speed_noise` is the next parameter, held under PPO, which scores the speed by its exact conditional density given the previous step, the buffer's `PreviousStep`), `encoder_readout: soft_cap` (2·tanh(x/2)), `encoder_pool_centres: true` (four fitted centres, appended after the speed noise's timescale), `encoder_box: graded` (wiring scales on [0, 0.75], sensory gains on ±2), `encoder_inhibition_floor`, and `modulator_receptor_signs: [pdfr-1, ser-5]` (cited couplings from `data/neuromodulators/receptor_couplings.json`); `ConnectomeTopology.forward_observer` serves a fit's guards. Configs `configs/scenarios/foraging/*_lawns_encoded_s1_primitives*.yml` (the speed-following ones `*_encoded_s1_primitives_following_assay_*.yml`); grading arms `benchmark_grade.ARMS["connectome_c2"]` and `["connectome_c2_following"]`. Probes: `scripts/analysis/c2_encoder_probe.py` (cost, feasibility, the pilots, `pilot-staged`, `stage2-extend`, `sizing`, `speed-noise` and `reproduce`; records in `docs/experiments/logbooks/supporting/122-c2-encoder/`; `reproduce` converts a probe vector with `package_vector` and checks the package's forward against the probe's). The registered fit, `scripts/analysis/c2_fit.py speed-noise-control|manifest|pilot|recovery-preflight|recovery|run|grade|read|loo|smoke`, fails its verdict on tph-1 alone, read against its failed recovery control F.0b-ii (Logbook 133); since the grader moved into `wormbench`, its manifest no longer matches, so its guarded stages refuse and its records are final. How the encoder and its procedure were reached is in Logbooks 122–133. Check a connectome-brain change with `scripts/analysis/connectome_identity_check.py baseline|check --build <dir> [--out <json>]` (baseline on the code before it). The per-connection sign table signs the glutamatergic tax-4 → AIA and mec-3 → ADE edges inhibitory, after López-Cruz et al. 2019 (16 overrides); `sign_override_exclude: [lopezcruz2019]` on the connectome brain restores the earlier table, as the configs behind recorded per-connection runs do.
- Phase 7's L4 plasticity panel (closed): the pilot `scripts/campaigns/l4_panel_pilot.py` and the analysis `scripts/analysis/l4_panel.py` (`--pilot` summarises a pilot; confirmatory mode accepts seeds 1–8 only, with the test family and verdict map fixed in the script; Logbook 040).

## Running Experiments

Read [docs/research/phase-protocol.md](docs/research/phase-protocol.md) before designing any panel; its principles bind every
registration. The non-negotiables, each learned from a lost run:

- **Workflow**: plan, then an OpenSpec change, then `/nematode-review-spec`, then implement, archive and PR. Never skip the
  review for a change that launches a campaign over about two hours.
- **Register before running**: a `launch.md` under `docs/experiments/logbooks/supporting/<NNN>-<slug>/` fixing the arms,
  metric, registered minimum (from committed data, on the same cell), sizing, verdict map and gates, committed before any
  scored run. A registered verdict is never re-read under a different rule after the data are seen; anything else is
  reported beside it as description.
- **Gate preflight before launch**: `uv run python scripts/campaigns/gate_preflight.py --panel <analysis module> [--half ppo] --logs <dir of runs at the registered point>` evaluates every level's floor and saturation gates on committed runs,
  or on a pilot, and exits nonzero if any level would be unreadable, sits within 5 points of the bar, or has no evidence.
  A.6t ran 16 hours into an unreadable panel that one minute of this would have caught
  ([Logbook 077](docs/experiments/logbooks/077-thermal-null-strength.md)).
- **Oracle before launch**: a positive control's success bar must be met by its known answer before it registers. Score
  the generating parameters, re-simulated with fresh noise, against the bar. A bar on a statistic compared between
  simulations is set against the simulator's own same-parameter noise. F.0a spent five hours on a gate its oracle could
  not pass ([Logbook 111](docs/experiments/logbooks/111-npe-recovery.md); protocol principles 4 and 6).
- **Pilots** run on seeds disjoint from the registered band, configured exactly as the campaign, and long enough to
  reach the machine's sustained rate; cost is estimated from them. A smoke run that only checks configs execute is not a
  pilot.
- **During a campaign** never switch branches: the runner reads configs and package code from the working tree per run.
- **Compute**: new runs stay within the machine's cores, at most 16 workers in total across everything running. Stopping
  a pool run's parent orphans its workers: find them by parent PID 1 and stop them by PID
  ([Logbook 122](docs/experiments/logbooks/122-c2-encoder.md): 34 idle workers, about 16 GB, for 15 hours).
- **Before registering a fit**: check that the model can represent the target, break the objective down term by term
  against the verdict, and show that the recovery gate can fail a fit that did not recover the truth (protocol
  principles 2, 4 and 10). Smokes and probes that reach held-out code pass band-cohort stand-ins, never the defaults.
- **Retention**: commit the per-seed CSV, analysis JSON and launch record; archive raw logs off-repo.

## Key Directories

- `packages/quantum-nematode/quantumnematode/` — Main source code
  - `benchmark/` — aliases of the grader's modules, which live in `wormbench`; `evaluation/` — the batched evaluator and its controllers; `connectome/` — the wiring, signs, encoder and neuron classes
  - `brain/arch/` — 26 brain architectures: qvarcircuit, qrc, qrh, qef, crh, qrhqlstm, crhqlstm, qsnnreinforce, qsnnppo, qliflstm, hybridquantum, hybridclassical, hybridquantumcortex, mlpreinforce, mlpdqn, mlpppo, lstmppo, spikingreinforce, connectomeppo, feedforwardga, cfcppo, spikingppo, equivariantquantum, transformerppo, mingruppo, minlstmppo. Plug-in registry: each Brain self-registers via `@register_brain`; see [docs/architecture/plugin-developer-guide.md](docs/architecture/plugin-developer-guide.md) for how to add a new one.
  - `env/` — Environment simulation
  - `agent/` — Agent orchestration, rewards, metrics
  - `experiment/` — Experiment tracking, metadata, and convergence analysis
  - `optimizers/` — Learning algorithms (PSR, CMA-ES)
- `packages/wormbench/` — the benchmark grader as a standalone package (Python 3.10+, numpy and scipy; released on PyPI)
- `scripts/` — CLI entry points (run_simulation.py, run_evolution.py, experiment_query.py); `scripts/analysis/` holds every logbook's analysis and fit scripts
- `data/` — the connectome, the benchmark's real-worm reference sets and the other vendored data, each namespace with its `PROVENANCE.md`
- `configs/scenarios/` — YAML config files organized by scenario (`{brain}_{size}[_{variant}]_{sensing}.yml`)
  - Scenarios: `foraging`, `pursuit`, `stationary`, `thermal_foraging`, `thermal_pursuit`, `thermal_stationary`, `oxygen_foraging`, `oxygen_pursuit`, `oxygen_stationary`, `oxygen_thermal_foraging`, `oxygen_thermal_pursuit`, `oxygen_thermal_stationary`, `multi_agent_foraging`, `multi_agent_pursuit`, `multi_agent_stationary`, `foraging_predator_thermal`, `bit_memory`, `associative_memory`
  - Sensing suffixes: `_oracle`, `_temporal`, `_derivative`, `_klinotaxis`
  - Variant suffixes: `_classical`, `_fair`, `_separable`, `_modality_paired`, `_pheromone`, `_no_pheromone`, `_social`, `_aggregation`, `_full_social`, `_scarcity`, `_propfood`, `_mixed_phenotype`, `_ars_depletion`, `_no_respawn_control`, `_rewired_null`, etc.
  - Task suffixes: `_bit_memory` (the `bit_memory` family — a non-spatial delayed-match-to-cue working-memory positive control; spatial/foraging/predator/thermal dynamics are disabled, so it is its own family rather than a variant of a spatial scenario); `_associative_memory` (the `associative_memory` family — a non-spatial chemosensory delayed-associative-match with probabilistic within-trial reversal, a working-memory *update* probe; observation = cue + outcome + go only, so like `bit_memory` it is its own family, not a spatial variant)
  - Example: `configs/scenarios/foraging/mlpppo_small_oracle.yml`, `configs/scenarios/thermal_pursuit/lstmppo_large_temporal.yml`
- `configs/evolution/` — Evolutionary optimization configs
- `configs/special/` — One-off experimental configs
- `tests/` — Four-tier testing (unit, slow, smoke, nightly)
- `openspec/` — Spec-driven development framework

## Code Conventions

- PascalCase classes, snake_case functions, UPPER_SNAKE_CASE constants
- Leading underscore for private modules (`_brain.py`)
- Comprehensive type annotations required
- NumPy-style docstrings
- Pydantic BaseModel for data structures
- Line length: 100 (Ruff)
- **No planning-doc references in implementation code or docstrings**: never cite roadmap
  sections, decision IDs (D1/D7/…), OpenSpec changes, review-finding labels (B1/S3/…), phase or
  milestone names, logbooks, or issue/PR numbers inside `packages/` source. State the technical
  constraint itself — the reader of the code gets the *why* as an intrinsic property, not a
  pointer into planning history. (Tests are the exception: test docstrings SHOULD name the
  OpenSpec spec scenarios they cover.) Planning provenance belongs in commit messages, OpenSpec
  changes, and logbooks.

## Testing

Four tiers:

1. **Unit/Integration** — Default pytest, runs on commits via pre-commit
2. **Slow** (`@pytest.mark.slow`) — Heavy in-process integration (real `EvolutionLoop` runs etc.), excluded from pre-commit, run before push
3. **Smoke** (`@pytest.mark.smoke`) — CLI end-to-end, runs on PRs
4. **Nightly** (`@pytest.mark.nightly`) — Full training benchmarks, runs daily

Pre-commit runs only the fast tier (`not smoke and not nightly and not slow`). Run `uv run pytest -m "not nightly"` (includes slow + smoke) after substantive changes, especially when touching `evolution/`, and `uv run pre-commit run -a` before committing.

### CI sharding and `.test_durations`

CI splits the suite across five parallel shards with `pytest-split`, balanced by
the committed `.test_durations` file and assigned with the `least_duration`
algorithm.

A test missing from that file still runs, but it is **not** placed without an
estimate: `pytest-split` budgets it at the **mean of the recorded durations**. So
drift is silent and it is not free — on 2026-09-13 a three-week-old file left 25%
of the suite unmeasured, including one 100.7s test budgeted at 0.096s, and the
shards' true loads spread 2.36x (135s to 319s) while `pytest-split` reported its
own split as balanced to 1.03x. It cannot see the problem, because it scores each
split with the same stale numbers that produced it.

The `test durations freshness` pre-commit hook is the guard: it fails when more
than 10% of the collected suite has no recorded duration.

**Refresh it from CI, not from a laptop.** The split has to balance CI's runners,
and tests do not keep their relative speeds between a Mac and a runner. On
2026-10-24 a file timed locally split into shards of 499s, 1026s, 1046s, 1082s
and 1194s of test time on CI, balanced to 1.00x by its own numbers. So every CI
shard times its own tests and uploads them as the artifact
`test-durations-shard-<n>` (14 days' retention). The union of one run's five
files is the suite timed on CI. Take a `main` run whose five shards all passed:

```bash
dir=$(mktemp -d)
gh run download <run-id> --pattern 'test-durations-shard-*' --dir "$dir"
uv run python scripts/merge_test_durations.py "$dir"
```

Download into a new, empty directory every time. `gh` puts each artifact in its
own `test-durations-shard-<n>/` folder there, which is what the script reads. A
reused directory can still hold an earlier run's folders. The script refuses
fewer than five files, and any test that appears in two of them. Then run `pre-commit`.

Refresh in a standalone PR, from a green `main` run, when shard times drift
apart or after PRs that add heavy tests. Never add it to a feature PR: the new
commit costs that PR another full CI run. Until the next refresh, new tests are
budgeted at the mean, and the freshness hook allows up to 10% of the suite to be
unmeasured.

**Locally**, for example to cover a new module before its first CI run, regenerate with:

```bash
uv run pytest -m "not nightly" --store-durations --clean-durations --durations-path .test_durations
```

These are laptop timings, so prefer the CI refresh as soon as a run exists.
`--clean-durations` drops entries for tests that no longer exist; without it the
file accumulates them. This takes ~2.5 minutes, since `-n logical` comes from
`addopts`. `pytest-split` writes the file without a trailing newline, so run
`pre-commit` afterwards — the `fix end of files` hook adds it, and CI's Code
Quality job fails without it.

Locally the suite is unsharded; `-n logical` in `addopts` uses every logical core.

## Pull Requests

PR titles MUST use [Conventional Commits](https://www.conventionalcommits.org/) prefixes. Common types in this repo: `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:`. Use `!` after the type for breaking changes (e.g. `feat!: remove legacy preprocessing mode`). Examples from project history: `feat: Add aerotaxis (oxygen sensing) system`, `fix: multi-agent sensing - use agent's own position in BrainParams`, `docs: Klinotaxis Era multi-agent pheromone evaluation (Logbook 011)`.

Commit messages do not require this prefix — only PR titles do.

User-facing changes add a line to `CHANGELOG.md` under *Unreleased* (Keep a Changelog format; breaking changes first). Releases follow the checklist in CONTRIBUTING.md § Releasing.

<!-- markdownlint-disable MD025 -->
