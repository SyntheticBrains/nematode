# Tasks: M0 — Brain-Agnostic Evolution Framework

## Phase 1: New Module Skeleton

**Dependencies**: None
**Parallelizable**: No (foundational)

- [ ] 1.1 Create `packages/quantum-nematode/quantumnematode/evolution/` directory with `__init__.py` exporting public API
- [ ] 1.2 Create `genome.py`: `Genome` dataclass (`params: np.ndarray`, `genome_id: str`, `parent_ids: list[str]`, `generation: int`, `birth_metadata: dict[str, Any]`)
- [ ] 1.3 Add `genome_id_for(generation: int, index: int, parent_ids: list[str]) -> str` helper (deterministic UUID derived from inputs)
- [ ] 1.4 Unit tests: `test_genome_id_deterministic`, `test_genome_id_distinct_for_distinct_inputs`

## Phase 2: Encoder Protocol and Concrete Encoders

**Dependencies**: Phase 1
**Parallelizable**: No

- [ ] 2.1 Create `encoders.py` with `GenomeEncoder` protocol (methods: `initial_genome`, `decode`, `genome_dim`)
- [ ] 2.2 Implement private `_flatten_components(components: dict[str, WeightComponent]) -> tuple[np.ndarray, dict]` (returns flat array + shape map for unflatten)
- [ ] 2.3 Implement private `_unflatten_components(params: np.ndarray, shape_map: dict) -> dict[str, WeightComponent]`
- [ ] 2.4 Implement `MLPPPOEncoder`: `brain_name = "mlpppo"`; uses components `{"policy", "value"}`; resets `brain._episode_count = 0` after `load_weight_components`
- [ ] 2.5 Implement `LSTMPPOEncoder`: `brain_name = "lstmppo"`; uses components `{"lstm", "actor", "critic"}`; resets `brain._episode_count = 0` after load
- [ ] 2.6 Define `ENCODER_REGISTRY: dict[str, type[GenomeEncoder]] = {"mlpppo": MLPPPOEncoder, "lstmppo": LSTMPPOEncoder}`
- [ ] 2.7 Unit tests: `test_mlpppo_encoder_roundtrip` (encode → decode → identical first-step action on seeded input)
- [ ] 2.8 Unit tests: `test_lstmppo_encoder_roundtrip` (same with recurrent brain)
- [ ] 2.9 Unit tests: `test_genome_dim_matches_flattened_state` (for both encoders)
- [ ] 2.10 Unit tests: `test_episode_count_resets_on_decode` (verify the determinism guard)

## Phase 3: Fitness Function

**Dependencies**: Phase 2
**Parallelizable**: Can start in parallel with Phase 4

- [ ] 3.1 Create `fitness.py` with `FitnessFunction` protocol (single `evaluate(genome, brain_config, sim_config, encoder, *, episodes, seed) -> float` method)
- [ ] 3.2 Implement `EpisodicSuccessRate`: decodes genome → instantiates env via existing `create_env_from_config` → runs `episodes` complete episodes → returns mean foods_collected / target ratio
- [ ] 3.3 **Frozen weights only**: do NOT call `brain.learn()` or `update_memory()`. M0 scope. (LearnedPerformanceFitness is M2.)
- [ ] 3.4 Reuse the per-step action loop pattern from `scripts/run_simulation.py` lines 240-1200 — strip rendering, CSV export, and learning calls
- [ ] 3.5 Unit test: `test_episodic_success_rate_returns_zero_for_random_brain` (sanity: a brain initialised to zero weights should fail to forage)
- [ ] 3.6 Unit test: `test_episodic_success_rate_deterministic_for_seeded_genome` (same genome + seed → same fitness)

## Phase 4: Lineage Tracker

**Dependencies**: Phase 1
**Parallelizable**: Yes

- [ ] 4.1 Create `lineage.py` with `LineageTracker(output_path: Path)` class
- [ ] 4.2 `record(genome: Genome, fitness: float, brain_type: str)` appends a CSV row
- [ ] 4.3 CSV columns: `generation, child_id, parent_ids, fitness, brain_type` (parent_ids `;`-joined for CSV-safety)
- [ ] 4.4 Append mode (not write mode) so resume works without recreating
- [ ] 4.5 Unit test: `test_lineage_csv_appends_across_generations` (record gens 0-5, verify row count and parent_ids)
- [ ] 4.6 Unit test: `test_lineage_csv_header_only_written_once` (resume scenario)

## Phase 5: Evolution Loop

**Dependencies**: Phases 2, 3, 4
**Parallelizable**: No

- [ ] 5.1 Create `loop.py` with `EvolutionLoop` class taking `optimizer`, `encoder`, `fitness`, `sim_config`, `evolution_config`, `output_dir`, `rng`
- [ ] 5.2 Implement `run(*, resume_from: Path | None = None) -> EvolutionResult`
- [ ] 5.3 Generation loop: `optimizer.ask()` → wrap each candidate as a `Genome` with proper parent_ids → parallel fitness eval → `optimizer.tell()` → record lineage → checkpoint every N gens
- [ ] 5.4 Multiprocessing: reuse the worker pattern from legacy `run_evolution.py:452-470` (SIGINT-ignore, per-worker logger config)
- [ ] 5.5 Worker function takes picklable args only (params array, sim_config dict, brain_config dict, episodes, seed) and reconstructs brain inside worker via the encoder
- [ ] 5.6 Pickle checkpoint: dump `{optimizer, generation, rng_state, lineage_path, checkpoint_version: 1}` to `output_dir/checkpoint.pkl`
- [ ] 5.7 Resume: load pickle, validate `checkpoint_version`, restore optimizer state, continue from saved generation
- [ ] 5.8 On completion: write `best_params.json` (compatible with legacy artifact contract) and `history.csv` to `output_dir`
- [ ] 5.9 Unit test: `test_loop_runs_3_generations_mlpppo` (minimal config, 3 gens, pop 4, 1 episode each — completes and produces best_params.json)
- [ ] 5.10 Unit test: `test_loop_resume_from_checkpoint` (run 2 gens → kill → resume → run 3 more — total 5 gens in lineage CSV)
- [ ] 5.11 Unit test: `test_loop_rejects_incompatible_checkpoint_version`

## Phase 6: Configuration Schema Extension

**Dependencies**: None (can be done in parallel with Phases 1-5)
**Parallelizable**: Yes

- [ ] 6.1 In `packages/quantum-nematode/quantumnematode/utils/config_loader.py`, add `EvolutionConfig` Pydantic model
- [ ] 6.2 Fields: `algorithm: Literal["cmaes", "ga"] = "cmaes"`, `population_size: int = 20`, `generations: int = 50`, `episodes_per_eval: int = 15`, `sigma0: float = math.pi/2`, `elite_fraction: float = 0.2`, `mutation_rate: float = 0.1`, `crossover_rate: float = 0.8`, `parallel_workers: int = 1`, `checkpoint_every: int = 10`
- [ ] 6.3 Add `evolution: EvolutionConfig | None = None` field to `SimulationConfig`
- [ ] 6.4 Verify (test): existing scenario configs without `evolution:` block still load cleanly
- [ ] 6.5 Verify (test): a config with `evolution:` block parses into a populated `EvolutionConfig`

## Phase 7: New CLI Script

**Dependencies**: Phases 5, 6
**Parallelizable**: No

- [ ] 7.1 First: `git mv scripts/run_evolution.py scripts/legacy/run_evolution_qvarcircuit.py` (preserve legacy)
- [ ] 7.2 Create new `scripts/run_evolution.py` (~150 LOC): argparse → load config → instantiate encoder via registry → instantiate optimiser → instantiate `EvolutionLoop` → run
- [ ] 7.3 CLI flags: `--config`, `--generations`, `--population`, `--episodes`, `--algorithm`, `--sigma`, `--parallel`, `--seed`, `--resume`, `--output-dir`, `--log-level`
- [ ] 7.4 CLI flags default to `None` and only override YAML when explicitly passed (single-source-of-truth = `EvolutionConfig` defaults)
- [ ] 7.5 On startup, log prominently: `Brain type: <name>`, `Algorithm: <cmaes|ga>`, `Population: <N>`, `Generations: <M>`
- [ ] 7.6 Error message if `brain.name` is not in `ENCODER_REGISTRY`: `f"No encoder for brain '{name}'. Registered: {list(ENCODER_REGISTRY)}. For QVarCircuit support use scripts/legacy/run_evolution_qvarcircuit.py"`

## Phase 8: Pilot Configs

**Dependencies**: Phase 6
**Parallelizable**: Yes

- [ ] 8.1 Create `configs/evolution/mlpppo_foraging_small.yml`: `mlpppo` brain, 20×20 grid, oracle sensing, target_foods_to_collect ≈ 5, `evolution: {generations: 10, population_size: 8, episodes_per_eval: 3}`
- [ ] 8.2 Create `configs/evolution/lstmppo_foraging_small_klinotaxis.yml`: copy brain block from `configs/scenarios/foraging/lstmppo_small_klinotaxis.yml` (gru, lstm_hidden_dim=64, klinotaxis sensing, STAM enabled), same minimal `evolution:` block

## Phase 9: End-to-End Smoke Verification

**Dependencies**: Phases 7, 8
**Parallelizable**: No

- [ ] 9.1 Run: `uv run python scripts/run_evolution.py --config configs/evolution/mlpppo_foraging_small.yml` — completes without error
- [ ] 9.2 Verify `evolution_results/<session>/best_params.json` exists; load it and decode back into a working `MLPPPOBrain` (test or one-off script)
- [ ] 9.3 Verify `evolution_results/<session>/lineage.csv` has 80 rows + header (10 gens × pop 8) with parent_ids populated from gen 1 onwards
- [ ] 9.4 Run: `uv run python scripts/run_evolution.py --config configs/evolution/lstmppo_foraging_small_klinotaxis.yml` — completes without error
- [ ] 9.5 Same artifact verification for the LSTMPPO smoke
- [ ] 9.6 Resume test: run smoke #1, kill at gen ~5, resume with `--resume evolution_results/<session>/checkpoint.pkl`, verify completes 10 gens total

## Phase 10: M-1 Invariant — Cross-Phase Tracking Updates

**Dependencies**: Phases 1-9 complete
**Parallelizable**: No (final step before PR)

- [ ] 10.1 Update `openspec/changes/2026-04-26-phase5-tracking/tasks.md`: mark M0.1 → M0.14 complete (`[x]`); update M0 status header to `complete`
- [ ] 10.2 Update `docs/roadmap.md` Phase 5 Milestone Tracker table: change M0 row status from `🔲 not started` to `✅ complete` (with one-line summary)
- [ ] 10.3 Run `openspec validate --changes 2026-04-28-add-evolution-framework --strict` — passes
- [ ] 10.4 Run `uv run pre-commit run -a` — clean
- [ ] 10.5 Run `uv run pytest -m "not nightly"` — all green
- [ ] 10.6 Open PR with Conventional Commits prefix (per AGENTS.md): `feat: add brain-agnostic evolution framework (M0)`
