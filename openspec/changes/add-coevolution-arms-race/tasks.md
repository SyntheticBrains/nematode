## 1. Predator MLPPPO Brain (PR 1)

- [ ] 1.1 Create `quantumnematode/env/mlpppo_predator_brain.py` with `MLPPPOPredatorBrain` implementing the `PredatorBrain` Protocol; mirror agent-side MLPPPO via composition using the existing `DEFAULT_ACTOR_HIDDEN_DIM`, `DEFAULT_CRITIC_HIDDEN_DIM`, and `DEFAULT_NUM_HIDDEN_LAYERS` constants from `quantumnematode.brain.arch.mlpppo` (no literal hardcoding) plus value head; implement `WeightPersistence` for encoder round-trip.
- [ ] 1.2 Encode `PredatorBrainParams` to the 11-float input vector per the spec (predator pos / k_nearest=2 agents with present_flags / radii / step_index, all normalised by `grid_size` or `max_steps`); pad with zeros + `present_flag=0` when fewer than k_nearest agents are alive.
- [ ] 1.3 Map the 5-way categorical output to `PredatorAction.{STAY, UP, DOWN, LEFT, RIGHT}` in the order `0=STAY, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT`.
- [ ] 1.4 Create `quantumnematode/env/_predator_brain_pretrain.py` with the 50-episode behavioural-cloning helper that imitates `HeuristicPredatorBrain` decisions.
- [ ] 1.5 Add `tests/quantumnematode_tests/env/test_mlpppo_predator_brain.py` (~15 cases): Protocol conformance via `isinstance(brain, PredatorBrain)`, input-encoding correctness, action-mapping correctness, deterministic-action under fixed seed, weight round-trip via `WeightPersistence`, `copy()` independence.
- [ ] 1.6 Add `tests/quantumnematode_tests/env/test_predator_brain_pretrain.py` (~6 cases): imitation loss decreases monotonically (windowed); final action distribution matches heuristic on >70% of held-out test states; pretrained weights round-trip through encoder unchanged.
- [ ] 1.7 `uv run pytest -m smoke -v` clean; full predator-brain test suite green.

## 2. Predator-Brain Dispatcher and Schema Extension (PR 1)

- [ ] 2.1 Extend `PredatorBrainConfig.kind` Literal in `quantumnematode/env/predator_brain.py:179` from `Literal["heuristic"]` to `Literal["heuristic", "mlpppo_predator"]`.
- [ ] 2.2 Extend `PredatorBrainConfigSchema.kind` Literal in `quantumnematode/utils/config_loader.py:325` to match.
- [ ] 2.3 Add `mlpppo_predator` branch to `_build_predator_brain` in `quantumnematode/env/env.py:1538` (~6 lines) constructing `MLPPPOPredatorBrain` via direct import from `quantumnematode/env/mlpppo_predator_brain.py` (NOT via `PREDATOR_ENCODER_REGISTRY` — keeps env free of evolution-package deps); optionally accept a weight-load path in `extra` so the brain can be initialised from a saved genome's checkpoint.
- [ ] 2.4 Extend `tests/quantumnematode_tests/env/test_predator_brain_config.py` with cases for the `mlpppo_predator` kind: dispatcher constructs `MLPPPOPredatorBrain`; unknown kinds rejected at YAML validation; default `kind="heuristic"` continues to construct `HeuristicPredatorBrain`.

## 3. Predator Brain Factory, Encoder, and Fitness (PR 2)

- [ ] 3.0 Create `quantumnematode/evolution/_predator_brain_factory.py` with `instantiate_predator_brain_from_sim_config(sim_config: SimulationConfig, *, seed: int | None = None) -> MLPPPOPredatorBrain` mirroring the agent-side factory at `evolution/brain_factory.py:38`. Reads predator config from `sim_config.environment.predator.brain_config` (NOT `sim_config.brain` — agent-side); seeds the brain's RNG sources at construction. **Required because `_ClassicalPPOEncoder.decode` calls `instantiate_brain_from_sim_config` which only knows agent brains.**
- [ ] 3.1 Create `quantumnematode/evolution/predator_encoders.py` with `MLPPPOPredatorEncoder(_ClassicalPPOEncoder)` and `PREDATOR_ENCODER_REGISTRY: dict[str, type[GenomeEncoder]] = {"mlpppo_predator": MLPPPOPredatorEncoder}`. **`MLPPPOPredatorEncoder` MUST override `initial_genome`, `decode`, and `genome_dim`** to call `instantiate_predator_brain_from_sim_config` instead of the agent-side `instantiate_brain_from_sim_config` (plain subclassing is insufficient — the parent's three methods all dispatch through agent-side `setup_brain_model` which only knows the 19 registered agent brains).
- [ ] 3.2 Verify `MLPPPOPredatorEncoder` reuses the `_ClassicalPPOEncoder` parent's flatten/unflatten round-trip (the WeightPersistence + shape_map round-trip is brain-agnostic); pin `brain_name = "mlpppo_predator"`.
- [ ] 3.3 Create `quantumnematode/evolution/predator_fitness.py` with `PredatorEpisodicKillRate` (primary kill-rate signal) and `PredatorLearnedPerformanceFitness` (learned-PPO secondary variant); both conform to the canonical `FitnessFunction` Protocol surface `evaluate(genome, sim_config, encoder, *, episodes, seed) -> float` (NOT a `(results, predator_id)` form). Internally, predator fitness decodes the genome → predator brain (via `MLPPPOPredatorEncoder` from task 3.1), drives the multi-agent runner against frozen prey opponents (configured via `sim_config` patching at the call site), with **all predator slots in the env using the same decoded brain**, and aggregates per-episode `sum(per_predator_kills.values())` then averages across episodes.
- [ ] 3.4 Implement secondary proximity signal: when `sum(per_predator_kills.values())` is zero across all `episodes` runs, fall back to per-episode mean of `sum(per_predator_prey_proximity_steps.values()) / (max_steps * num_predator_slots)` (cross-slot normalised proximity ratio in [0, 1]); bound the fallback strictly below `1/episodes` so any non-zero kill-rate beats the proximity-only fallback regardless of proximity magnitude.
- [ ] 3.5 Add `tests/quantumnematode_tests/evolution/test_predator_encoders.py` (~7 cases): round-trip via `WeightPersistence`; genome dim matches MLPPPO predator param count; `initial_genome(seed)` reproducibility under fixed seed; registry lookup correctness; **encoder override correctness** — `decode` returns an `MLPPPOPredatorBrain` (not an agent MLPPPO brain); `decode` works under `CMAESOptimizer(diagonal=True)` flat-vector samples (no bounds rejection).
- [ ] 3.6 Add `tests/quantumnematode_tests/evolution/test_predator_fitness.py` (~9 cases): the `evaluate(genome, sim_config, encoder, *, episodes, seed) -> float` Protocol surface; mock multi-agent runner returns synthetic `MultiAgentEpisodeResult` objects with multi-slot `per_predator_kills` dicts; verify per-episode mean of summed kills calculation across slots; verify secondary proximity signal triggered when summed kills=0; verify fallback bounded strictly below `1/episodes`; FitnessFunction Protocol conformance via `isinstance`; pickle round-trip (so fitness can flow through the worker); verify all predator slots use same decoded brain (no slot-0 special-casing).

## 4. Hall-of-Fame Buffer (PR 3)

- [ ] 4.1 Create `quantumnematode/evolution/hall_of_fame.py` with `HallOfFame` class: bounded `deque[Genome]` with `replacement: Literal["quality", "fifo"]` (default `"quality"`); methods `push`, `sample`, `mix_with_pop`, `__len__`, `to_dict`, `from_dict`.
- [ ] 4.2 Implement quality-based eviction: on `push` when at capacity, find the lowest-fitness existing entry and replace it iff the new fitness is strictly greater; FIFO ablation evicts oldest regardless.
- [ ] 4.3 Implement `mix_with_pop(rng, pop, frac_hof)`: sample with replacement, ~`frac_hof * |pop|` from HoF and remainder from `pop`; empty-HoF fallback returns all-from-pop.
- [ ] 4.4 Add `tests/quantumnematode_tests/evolution/test_hall_of_fame.py` (~10 cases): quality eviction rules, FIFO ablation, mix_with_pop fraction correctness, reproducible sampling under seeded RNG, checkpoint round-trip.

## 5. Red Queen Metrics Module (PR 3)

- [ ] 5.1 Create `quantumnematode/evolution/redqueen_metrics.py` with the five pure functions: `phenotypic_cycling`, `trait_escalation`, `fitness_lag`, `coupled_rate`, `generality`; all operate on numpy arrays / floats only (no Genome / FitnessFunction dependencies).
- [ ] 5.2 Implement `phenotypic_cycling`: Lomb-Scargle periodogram + autocorrelation peak detection in lag range; return dict with `cycling_detected`, `dominant_period`, `p_value`.
- [ ] 5.3 Implement `trait_escalation`: linear regression over `gen_window=(5, 30)`; return dict with `escalation_detected`, `slope`, `slope_sign`, `slope_se`, `p_value`.
- [ ] 5.4 Implement `fitness_lag`: cross-correlation between two series with `max_lag` parameter; return best-lag scalar or NaN for zero-variance inputs.
- [ ] 5.5 Implement `coupled_rate`: per-generation delta computation followed by Pearson correlation of deltas; return scalar in [-1, 1].
- [ ] 5.6 Implement `generality`: summarise held-out opponent fitness curve; return scalar near +1 for uniform improvement, near 0 for flat, negative for decline.
- [ ] 5.7 Add `tests/quantumnematode_tests/evolution/test_redqueen_metrics.py` (~12 cases): synthetic series with known answers — pure sine cycling, monotone ramp escalation, anti-phase fitness lag, perfect-coupling coupled rate, uniform-improvement generality, plus negative cases (flat series, zero-variance series, random noise).

## 6. CoevolutionLoop Core (PR 3)

- [ ] 6.0 In `CoevolutionLoop.__init__`, implement gen-0 initialisation per design.md D12 + D7: prey side loads warmstart genome from `prey_gen0_seed_path` config field (default `configs/evolution/coevolution_warmstart_prey/seed_<run_seed>.json`) and passes its decoded weight vector as `CMAESOptimizer(x0=...)`; predator side either pretrains via `pretrain_against_heuristic` (D7 arm A, when `predator_gen0_bootstrap="heuristic_imitation_pretrain"`) or uses `x0=zeros` (D7 arm B, when `predator_gen0_bootstrap="cold_start"`).
- [ ] 6.1 Create `quantumnematode/evolution/coevolution.py` with `CoevolutionLoop` class composing two side-state objects (each carrying encoder, fitness, optimizer, inheritance, hof, population, champion_history); does NOT subclass `EvolutionLoop`.
- [ ] 6.2 Implement alternating-schedule controller: `run(*, generation_pairs, K_per_block=10, generality_probe_every=10, start_side="prey")`; for each K-block in alternating order, train one side while the opposing side's population/optimizer/hof are frozen.
- [ ] 6.3 Implement per-K-block fresh `CMAESOptimizer` construction with `diagonal=True` (sep-CMA-ES): at each K-block transition, re-construct the just-flipped side's optimizer as a new instance (canonical import `quantumnematode.optimizers.evolutionary.CMAESOptimizer`) with a deterministic seed derived from the run's master seed and K-block index. **Use `diagonal=True` for both sides** — predator MLPPPO weight count (~5k) and prey LSTMPPO weight count (~30k+) are above the n>~100 tractability threshold for full-covariance CMA-ES. The existing optimizer has no public reset method; re-construction is the equivalent operation. (TPE is NOT viable here — `OptunaTPEOptimizer` requires bounded parameters but weight encoders return `genome_bounds = None`; see [evolutionary.py:519-577](packages/quantum-nematode/quantumnematode/optimizers/evolutionary.py#L519-L577) and [encoders.py:330-342](packages/quantum-nematode/quantumnematode/evolution/encoders.py#L330-L342).)
- [ ] 6.4 Implement HoF push at K-block end: training-side block elite pushed to its HoF with the configured eviction policy.
- [ ] 6.5 Implement HoF-mixed opposition sampling: when evaluating a candidate on side X, draw 70% of opponents from Y's current pop and 30% from Y's HoF (deterministic given seeded RNG); empty-HoF fallback to all-from-pop.
- [ ] 6.6 Implement generality probe: every `generality_probe_every` generations, evaluate each side's elite against the held-out opponent set; write `generality_probe.csv` with `(generation, side, opponent_index, fitness)`; probe SHALL NOT mutate population/optimizer/hof state and SHALL NOT advance the generation counter.
- [ ] 6.7 Implement held-out opponent construction: prey side loads from a committed in-repo bundle at `configs/evolution/coevolution_held_out_prey/*.json` (one genome per file); predator side draws from heuristic-radius variants spanning the default `detection_radius ∈ {4, 6, 8, 10} × damage_radius ∈ {0, 1}` grid (8 combos at default `held_out_size=8`); when `held_out_size` differs from the natural grid count, deterministically widen-or-sub-sample via `held_out_rng.choice` with a fixed seed so the held-out set count always matches `held_out_size`. Commit the prey bundle to the repo (NOT to `artifacts/`) so a fresh checkout can run the campaign reproducibly.
- [ ] 6.8 Implement checkpoint to JSON every K-block (include both side states + both HoFs + generation counter); resume support reconstructs all state.
- [ ] 6.9 Reuse `EvolutionLoop._evaluate_in_worker` (or an equivalent reusable worker) for per-side evaluations to preserve the existing 11-tuple worker pattern.
- [ ] 6.10 Add `tests/quantumnematode_tests/evolution/test_coevolution.py` (~16 cases): K-block boundary correctness, opposing pop frozen during off-block, CMA-ES re-construction at transition (assert fresh instance + new seed), HoF push timing, 70/30 mixed sampling fraction, probe cadence and non-mutation invariant, checkpoint round-trip resume produces identical run **for both prey-side AND predator-side state independently** (HoF, optimizer, population, champion_history must all survive resume on each side; per S18).
- [ ] 6.11 Add a `rebalance_threshold: float | None = None` config field to `CoevolutionLoop` (per design.md Open Question 1 + Risk register row "One side dominates"): when set, if a side's K-block-mean fitness drops below `rebalance_threshold * opposing_side_mean` for ≥3 consecutive K-blocks, freeze the dominant side for an extra K-block. Disabled by default; pilot may enable if domination is observed. ~50 LoC + 2 unit tests.
- [ ] 6.12 Document `champion_history.json` schema in `evolution/coevolution.py` module docstring (per S19): top-level dict with `prey` and `predator` keys, each mapping to a list of dicts with `(genome_id: str, generation: int, k_block_index: int, fitness: float, params: list[float])`. The aggregator (task 8.1) reads this schema; the loop (task 6.8) writes it via `json.dump`. Add 1 unit test asserting schema round-trip via `json.dump` → `json.load`.

## 7. Configs and Campaign Driver (PR 4)

- [ ] 7.0a Curate the prey **held-out** bundle: sample `held_out_size` (default 8) random elite genomes across all 4 M3 lamarckian-LSTMPPO full-run seeds (e.g. 2 from each seed) from `artifacts/logbooks/<M3-lamarckian-logbook>/`; copy the JSONs to `configs/evolution/coevolution_held_out_prey/`; write a `README.md` next to them documenting provenance (source run, seed, generation, fitness, brain config). Required for task 6.7 to load.
- [ ] 7.0b Curate the prey **warmstart** bundle (per design.md D12 — prey gen-0 init): take ONE elite genome per M3 lamarckian-LSTMPPO full-run seed (4 genomes total, one per seed, distinct from the 8 held-out genomes — different roles, same provenance file format); copy to `configs/evolution/coevolution_warmstart_prey/seed_42.json` (etc.); document provenance in a `README.md` next to them. Required for `CoevolutionLoop.__init__` to load (task 6.X below) when `prey_gen0_seed_path` is set.
- [ ] 7.1 Create `configs/evolution/coevolution_pilot.yml`: 30 gens × 2 seeds × prey-pop 24 × predator-pop 16 × K=10 × HoF size 8 × probe every 10 gens × pretrain on for seed 42 / off for seed 43.
- [ ] 7.2 Create `configs/evolution/coevolution_full.yml`: 50–70 gens × 4 seeds × prey-pop 24 × predator-pop 16 × K=10 × HoF size 8 × probe every 10 gens × pretrain config locked from pilot result.
- [ ] 7.3 Create `scripts/campaigns/run_coevolution.py`: per-seed driver invoking `CoevolutionLoop.run`; argparse + multiprocessing + JSON+CSV outputs; mirror the Python argparse + `multiprocessing.Pool` pattern from `scripts/campaigns/baldwin_f1_postpilot_eval.py` (the bash wrappers in tasks 7.4–7.5 then drive this Python entrypoint).
- [ ] 7.4 Create `scripts/campaigns/phase5_m5_coevolution_pilot.sh`: bash wrapper, 2 seeds (42, 43), pretrain bootstrap arms.
- [ ] 7.5 Create `scripts/campaigns/phase5_m5_coevolution_full.sh`: bash wrapper, 4 seeds (42–45).
- [ ] 7.6 Smoke pilot: 5 gens × 1 seed × tiny pops (4/4) × K=2 × K=5 × HoF size 2 — completes in ~60 sec; validates loop runs end-to-end, lineage CSVs written for both sides, HoF accumulates, probe fires once.

## 8. Aggregator and Plots (PR 5)

- [ ] 8.1 Create `scripts/campaigns/aggregate_m5_pilot.py`: reads per-seed `lineage_prey.csv` + `lineage_predator.csv` + `generality_probe.csv` + `champion_history.json`; emits `summary.md` + `cycling.png` + `escalation.png` + `generality.png` + `verdict.csv`. Mirror `aggregate_baldwin_retry_pilot.py` (782 LoC reference).
- [ ] 8.2 Implement verdict logic: per-seed cycling + escalation evaluation via `redqueen_metrics`; aggregate verdict = GO if criterion fires in ≥2 of 4 seeds (per-seed firing = cycling OR escalation), STOP if zero seeds, PIVOT if exactly 1.
- [ ] 8.3 Implement `summary.md` generation: overall verdict + seeds-firing count + per-seed results + generality trajectory + plot references + (when M5.7 lands) Baldwin readout block clearly labelled "secondary observation, not a verdict input".
- [ ] 8.4 Implement `verdict.csv` schema per spec: columns `seed`, `cycling_detected`, `escalation_detected`, `escalation_slope`, `escalation_p_value`, `cycling_period`, `cycling_p_value`, `generality_scalar`, `gate_fires`.
- [ ] 8.5 Smoke-run aggregator on smoke-pilot output (from task 7.6) to validate end-to-end pipeline.

## 9. Pilot Run and Logbook (PR 6)

- [ ] 9.1 Run pilot: `phase5_m5_coevolution_pilot.sh` with 2 seeds × 30 gens × pop 24/16 × K=10 alternating; expect ~7–14 wall-hours total at parallel_workers=4, sequential 2 seeds (per design.md D4 compute envelope; ±50% pending pilot calibration).
- [ ] 9.2 Run aggregator on pilot output; capture summary.md + verdict.csv + plots.
- [ ] 9.3 Pilot decision-gate evaluation: did cycling OR escalation fire in ≥1 of 2 pilot seeds? If yes → calibrate full-run thresholds + lock pretrain on/off based on per-seed result; proceed to full run. If ambiguous (zero seeds firing) → run +1 seed before commit. If no signal after extra seed → STOP M5 / pivot config.
- [ ] 9.4 Compare pretrain vs cold-start arms: was pretrain necessary, or did cold-start also produce non-trivial signal? Lock the choice for the full run config.
- [ ] 9.5 Create `docs/experiments/logbooks/017-coevolution-arms-race.md` with pilot section: design recap, pilot results, threshold calibration, pretrain decision, full-run go/no-go, **and a wall-time-reconciliation row** comparing actual per-episode wall vs design.md D4's ~0.75–1.5 sec estimate (used to lock full-run wall budget before task 10.2 launches).
- [ ] 9.6 Stash pilot artefacts under `artifacts/logbooks/017-coevolution-arms-race/pilot/`.

## 10. Full Run and Verdict (PR 7)

- [ ] 10.1 Lock full-run thresholds into the OpenSpec change `co-evolution` capability spec (cycling lag-range, FFT power threshold, escalation slope/SE threshold, gen-window) before launching full run, per the threshold-revision protocol in design.md D6. If pilot evidence motivates revised thresholds, amend the `co-evolution` spec deltas + re-run `openspec validate add-coevolution-arms-race --strict` before launching the full run.
- [ ] 10.2 Run full: `phase5_m5_coevolution_full.sh` with 4 seeds (42–45) × 50-70 gens × pop 24/16 × K=10 × HoF + probe; expect ~30–60 wall-hours total at parallel_workers=4 per seed × 4 sequential seeds (per design.md D4 compute envelope; lock the precise estimate from the pilot's actual per-episode wall before launching).
- [ ] 10.3 Run aggregator on full-run output; capture verdict.
- [ ] 10.4 Verdict per the gate: GO if cycling OR escalation fires in ≥2 of 4 seeds; STOP if zero seeds; PIVOT if exactly 1.
- [ ] 10.5 Append verdict section to `docs/experiments/logbooks/017-coevolution-arms-race.md`: full-run results, per-seed metrics, generality trajectory, verdict + rationale, follow-up triggers (M4.7 if M5.7 readout fires).
- [ ] 10.6 Stash full-run artefacts under `artifacts/logbooks/017-coevolution-arms-race/full/`.

## 11. M5.7 Secondary Baldwin Instrumentation — REDUCED under CMA-ES weight evolution (PR 8)

**Scope reduction (vs original tracker M5.7 entry):** under D2's CMA-ES weight evolution, prey have a single fixed hyperparameter set across all genomes (only weights co-vary), so the original two-condition readout (signal-delta AND hyperparam spread tightening) cannot apply — there is no per-generation hyperparam spread to measure. Only the signal-delta condition remains observable. M5.7 is reduced to a single suggestive readout, not a definitive Baldwin closure. See design.md D11 for full rationale.

- [ ] 11.1 Create `scripts/campaigns/baldwin_m5_secondary_eval.py`: invoke `baldwin_f1_postpilot_eval.py` with per-gen prey elite snapshots from full-run lineage CSV, current-gen predator pop as the "task" axis, K′ ∈ {10, 25} paired-train comparison.
- [ ] 11.2 ~~Compute prey hyperparam spread~~ **DROPPED — not observable under weight evolution** (no per-genome hyperparameter variation; all genomes share fixed hyperparameters). Skip this sub-task; document the drop in the M5.7 logbook section per task 11.4.
- [ ] 11.3 Compute F1-style elite-vs-prior signal-delta against current predator pop at K′=10 and K′=25; check whether signal-delta > +0.05 at K′=10 across ≥2 of 4 seeds. **This is the sole remaining M5.7 readout** under the reduced scope.
- [ ] 11.4 Append M5.7 readout section to `docs/experiments/logbooks/017-coevolution-arms-race.md`: instrumentation results clearly labelled "secondary observation, not a verdict input"; if signal-delta fires across ≥2 of 4 seeds → flag M4.7 as armed-and-motivated (suggestive Baldwin signal under weight evolution); if signal-delta stays silent → flag as observation-only outcome (M4.7 stays armed but unmotivated by M5 evidence). **Runs regardless of M5 verdict:** M5.7 SHALL execute even if M5 closes STOP, since the readout is informative even on a STOP M5. **Document the dropped hyperparam-spread condition** with a clear note that definitive Baldwin closure requires a dedicated hyperparameter-evolution milestone (M4.7).
- [ ] 11.5 Stash M5.7 analysis artefacts under `artifacts/logbooks/017-coevolution-arms-race/baldwin_secondary/`.

## 12. Tracker, Roadmap, and Spec Sync (PR 8 or PR 9)

- [ ] 12.1 Tick M5.0–M5.9 sub-tasks in `openspec/changes/2026-04-26-phase5-tracking/tasks.md` as PRs land; add an M5.0 row above M5.1 for the predator stack (brain + encoder + fitness) if not already present.
- [ ] 12.2 On full-run verdict, flip Phase 5 milestone tracker M5 row in `docs/roadmap.md` to ✅ complete (GO) or ❌ stop / 🔁 pivot per the verdict.
- [ ] 12.3 If verdict is GO, sync OpenSpec deltas: run the `openspec-sync-specs` skill on `add-coevolution-arms-race` to merge ADDED requirements into `openspec/specs/co-evolution/spec.md`, `openspec/specs/red-queen-analysis/spec.md`, `openspec/specs/environment-simulation/spec.md`, `openspec/specs/evolution-framework/spec.md`.
- [ ] 12.4 Archive the OpenSpec change via `openspec-archive-change` skill (date prefix added automatically).
- [ ] 12.5 Update memory with any new conventions worth preserving across sessions (e.g. co-evolution-specific aggregator paths, naming conventions).

## PR Splitting (Reference)

The 12 task groups above split across **9 PRs** for review tractability:

- **PR 1** (Predator brain + dispatcher): tasks 1.1–1.7 + 2.1–2.4. ~600 LoC + ~400 LoC tests.
- **PR 2** (Predator brain factory + encoder + fitness): tasks 3.0–3.6. ~430 LoC + ~250 LoC tests (includes the new `_predator_brain_factory.py` module).
- **PR 3** (HoF + Red Queen metrics + CoevolutionLoop): tasks 4.1–4.4 + 5.1–5.7 + 6.0–6.12. ~1200 LoC + ~520 LoC tests (CoevolutionLoop carries ~100 LoC of replicated inheritance bookkeeping per design D10, ~50 LoC for the rebalancing knob, ~30 LoC for gen-0 init + JSON schema docstring). Largest PR; consider further splitting if review feedback suggests.
- **PR 4** (Configs + held-out + warmstart bundles + driver + smoke): tasks 7.0a–7.6. ~500 LoC + held-out bundle (~80-200 KB JSONs) + warmstart bundle (4 genomes × ~tens-of-KB each); smoke validates the substrate before pilot.
- **PR 5** (Aggregator + plots): tasks 8.1–8.5. ~700 LoC. **Can develop in parallel with PR 4** once PR 3 pins the lineage CSV / probe CSV / champion_history JSON schemas.
- **PR 6** (Pilot run + logbook): tasks 9.1–9.6. Run-only PR; no new code.
- **PR 7** (Full run + verdict): tasks 10.1–10.6. Run-only PR.
- **PR 8** (M5.7 Baldwin readout — reduced): tasks 11.1–11.5. ~200 LoC delta + logbook (signal-delta only; hyperparam-spread condition dropped per design D11).
- **PR 9** (Tracker + roadmap + spec sync + archive): tasks 12.1–12.5. Closure PR.

PRs 1→2→3 are sequentially dependent (substrate stack); PR 4 depends on PR 3; PR 5 can develop in parallel with PR 4. PRs 6/7/9 are gated on results, not just code-review.

**On STOP at PR 6 (pilot) or PR 7 (full run):** the run-only PR still lands regardless of verdict — the logbook + artefacts are the deliverable, and a STOP outcome is still a research result worth recording. Subsequent PRs may be skipped or scoped down: if PR 6 closes STOP, PR 7 is skipped; PR 8 (M5.7) still runs (per task 11.4); PR 9 closes the milestone with a STOP verdict in roadmap.md and the OpenSpec change is archived without spec sync (since the capability didn't deliver under this attempt — the spec deltas would be deferred to a follow-up change like `add-coevolution-arms-race-v2`).
