# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow [Semantic Versioning](https://semver.org/): while the project is at 0.x, a minor release may contain breaking changes, and they are listed first under each release.

Releases before 0.5.0 are documented on [GitHub Releases](https://github.com/SyntheticBrains/nematode/releases); this file starts at 0.5.0.

## [Unreleased]

### Added

- **Assay protocols** (Phase 10, E.3). An `environment.assay` block runs an episode as phases of food and placement. The worm's satiety and controller state carry across a transfer, and captures record each step's phase. New options: a circular arena, a centred lawn, and `satiety.starvation_ends_episode`. Eight generated configs reproduce four reference assays on the 1-s primitives worm, each with its arena, lawn, start, food history and analysis window: Scheer & Bargmann's small lawn, López-Cruz's and Calhoun's food removal, and Flavell's uniform lawn.
- **Locomotor primitives for the 1-second point worm** (Phase 10, E.2). `continuous.locomotion: primitives` with `action_space: primitives` (connectome and MLP-PPO) moves the worm only by runs, reversal bouts and omega/delta turns, with kinematics fixed from published data (`data/locomotion/primitives.json`). The brain drives them through five numbers per second: run speed, run curvature and three start hazards. New sensory modules `food_contact` and `locomotor_state`. Non-learning `ConstantHazards` and `PirouetteHazards` controllers: a pirouette rule keyed to dC/dt finds a lawn that constant hazards at its mean rate find less often. Captures record each step's primitive.
- **One step duration, and a 1-second point worm** (Phase 10, E.1). A new top-level config key `step_seconds` sets the continuous point worm's step in worm-seconds. Configs keep their per-step values written for the 5-s reference step, and the loader converts every step-coupled constant to the configured step, so each keeps its value per worm-second, and prints the conversions to stdout. Unset, nothing changes. It is accepted only for the point worm on lawns with `connectomeppo` or `mlpppo`, and refused elsewhere. Behaviour captures record a non-5-s step; the roaming/dwelling classifier reads them at 5 s, and the klinokinesis/weathervane harness refuses them. Two 1-s lawn configs, `*_lawns_modulated_s1.yml` and `*_lawns_internal_s1.yml`.
- **G.0: the gap-junction placement advantage is an artefact of the gap model's operating point** ([Logbook 095](docs/experiments/logbooks/095-gap-operating-point.md)). With every gap count set to 1 the wild type no longer leads the gap-only null (+0.003 against +0.104 with counts); the lead follows the gap scale. New brain options `gap_weight_mode` and `gap_weight_scale`, the wirings `rewired_gap_outliers_held` and `rewired_gap_outliers_scattered`, and `scripts/analysis/gap_operating_point.py`.
- **M.1(a): late collapse leaves no registered signature before it falls** ([Logbook 096](docs/experiments/logbooks/096-collapse-diagnosis.md)). Re-running Logbook 092's eight late body collapses with telemetry, no registered cause separates them from their intact twins; the MLP control collapses on none of 8 seeds. New `run_simulation.py` flags `--update-telemetry` (one JSON line per PPO update, read-only) and `--checkpoint-every N`, and `scripts/analysis/collapse.py`, the collapse statistic every panel over 3,000 episodes now reports (D25, amended so the onset is the first fall a run does not recover from).
- **G.0-A: the corrected gap model is log-compressed counts** ([Logbook 097](docs/experiments/logbooks/097-gap-substrate.md)). A new `gap_coupling: ohmic` option on the connectome brain passes gap current in proportion to the potential difference, through `(I + L)⁻¹` in the settling map. A pilot on both block-V cells chose between it and `gap_weight_mode: log_count` on learning speed alone: ohmic coupling, its conductance scale calibrated to ×0.03 on untrained brains, learned more slowly on both cells, so block G runs on log counts.
- **G.0-A: on the corrected gap model the point worm shows no wiring effect** ([Logbook 099](docs/experiments/logbooks/099-logcount-contrast.md)). With log-compressed gap counts, the wild type learns the thermal cell no faster than its gap-only null over 64 seeds (+0.003 `auc_success`, interval [−0.009, +0.016], against a minimum of 0.0577). The gap-placement lead carried since Logbook 082 belonged to the count-scaled gap model. Whether the phase's STOP branch is taken now rests on the body contrast on log counts.
- **G.0-A: through the body too, the corrected gap model shows no wiring effect, and the phase's STOP condition is met** ([Logbook 100](docs/experiments/logbooks/100-body-logcount.md)). On 092's body cell, on Emmons 2024 with log-compressed gap counts over 128 seeds, the wild type learns no faster than its gap-only null over episodes 1–3,000 (−0.022 [−0.045, +0.005]), nor over the late window. The project's one surviving wiring effect belonged to the count-scaled electrical-synapse model.
- **G.0-B: the gap-placement lead came from the gap model's degree normaliser** ([Logbook 101](docs/experiments/logbooks/101-gap-model-question.md)). A new `gap_normalisation: strength` option on the connectome brain divides each gap pair by √(s_i s_j), s the neuron's total gap weight, so every coupling is at most 1. Under it the wild type shows no lead over its gap-only null (−0.009 [−0.026, +0.009]); under ohmic coupling at its calibrated conductance a small lead below the minimum is unresolved. The phase's STOP statement stands.
- **Block V's decomposition restated on the corrected gap model** ([Logbook 102](docs/experiments/logbooks/102-decomposition-restated.md)). Phase 8's headline, that the connectome's wiring advantage is gap-junction placement and survives the body, is withdrawn. What stands is the chemical wiring at the sensory–motor boundary on hard350, with no interior-wiring advantage there, and on thermal a residual below the minimum, with chemical wiring and autapses not separated.
- **Phase 9 complete** ([Logbook 103](docs/experiments/logbooks/103-phase9-synthesis.md)). The phase closed on its STOP branch: the connectome's gap-placement advantage was the gap model's degree normaliser. The architecture ranking is halted, and the next phase turns to standard on-lawn assays graded against real-worm data.
- **Roadmap v5.0: Phase 10, *fidelity on the plate*, and the project's question re-framed** ([roadmap § Phase 10](docs/roadmap.md)). The project now asks how faithfully a simulated worm reproduces standard *C. elegans* behaviour, graded against real-worm data, and what the connectome contributes to that fidelity. Phase 10 plans a graded benchmark of standard on-lawn assays (roaming/dwelling, lawn leaving, local search, chemotaxis components) from public data, a 1-s point worm with locomotor primitives, a data-derived neuromodulator layer, controllers fitted to match the animal, and the wiring question asked on held-out fidelity. The literature watch is re-aimed at it.
- **M.1(b): no standard PPO remedy prevents late collapse, which follows the training stream, not the seed** ([Logbook 098](docs/experiments/logbooks/098-collapse-remedies.md)). New PPO stabiliser keys on MLP-PPO and connectome PPO — `kl_stop_factor`, `weight_clip`, `adam_eps`/`adam_beta2` and `rollout_shuffle_stream` — each off by default and byte-identical when off. Read against a placebo that re-rolls only the minibatch stream, none lowered the late drawdown on 092's body collapsers; the placebo alone collapsed 2 of the 16. The registered stop rule fires: R.2 and G.5 register at 3,000 episodes.

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
