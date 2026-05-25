# Tasks: Add NEAT-Weights Brain (`FeedforwardGABrain`)

## 1. Audit existing infrastructure (read-only; no code changes)

- [x] 1.1 Read `packages/quantum-nematode/quantumnematode/brain/arch/mlpppo.py` to confirm the matched-capacity reference (hidden width, layer count, sensory-module consumption, forward-pass shape).
- [x] 1.2 Read `packages/quantum-nematode/quantumnematode/brain/arch/_brain.py:411-434` to confirm the current Brain Protocol surface.
- [x] 1.3 Read `packages/quantum-nematode/quantumnematode/brain/arch/_registry.py` + `packages/quantum-nematode/quantumnematode/brain/arch/connectome_ppo.py:428-433` (the canonical recent `@register_brain` decorator call) + `docs/architecture/plugin-developer-guide.md` to confirm the decorator signature and the registration walkthrough.
- [x] 1.4 Read `packages/quantum-nematode/quantumnematode/optimizers/evolutionary.py:270-429` to confirm `GeneticAlgorithmOptimizer`'s public API: `__init__(num_params, x0, population_size, sigma0, elite_fraction, mutation_rate, crossover_rate, seed)`. Tournament size is hard-coded to 3 inside `_tournament_select` (not configurable); crossover is uniform (per-gene Bernoulli mask), not two-point.
- [x] 1.5 Read `packages/quantum-nematode/quantumnematode/evolution/encoders.py:246-403` (the `_ClassicalPPOEncoder` base + `MLPPPOEncoder` / `LSTMPPOEncoder` subclasses + `ENCODER_REGISTRY` registration pattern). Note that `_ClassicalPPOEncoder.decode()` at lines 305-306 reaches for `brain._episode_count` and `brain._update_learning_rate()` — both PPO-specific; the new brain MUST provide shims.
- [x] 1.6 Read `packages/quantum-nematode/quantumnematode/brain/weights.py:55` to confirm the `WeightPersistence` Protocol surface (`get_weight_components()`, `load_weight_components()`).
- [x] 1.7 Read `packages/quantum-nematode/quantumnematode/evolution/fitness.py:80-128` to confirm that `FrozenEvalRunner` monkey-patches `agent.brain.learn` and `agent.brain.update_memory` to no-ops during GA evaluation (load-bearing for the brain's no-op `update_memory` design).
- [x] 1.8 Read `packages/quantum-nematode/quantumnematode/utils/brain_factory.py` lines 54-160 to confirm the `_build_infra_kwargs` dispatch: the default-shape fallthrough at the end returns `{"num_actions": 4, "device": device}` — the new brain taking only `device` (and config) fits this default without a new branch.
- [x] 1.9 Read `scripts/run_evolution.py` argparser to confirm the launcher's flag set (no `--theme` flag exists; evolution always runs headless).
- [x] 1.10 Read `configs/evolution/mlpppo_foraging_small.yml` to confirm the smoke-config shape: `brain:` + `reward:` + `satiety:` + `environment:` + `evolution:` blocks. The `evolution:` block carries GA hyperparameters (NOT the brain config).

## 2. Implement `FeedforwardGABrain`

- [x] 2.1 Create `packages/quantum-nematode/quantumnematode/brain/arch/feedforward_ga.py` with:
  - `FeedforwardGABrainConfig(BrainConfig)` Pydantic model exposing the topology fields per the spec's `FeedforwardGABrainConfig` requirement (`hidden_dim: int = 64`, `num_hidden_layers: int = 2`, `sensory_modules: list[ModuleName]` — match the `MLPPPOBrainConfig` type at `packages/quantum-nematode/quantumnematode/brain/arch/mlpppo.py:116`, NOT `list[str]`). NO GA hyperparameters in this config.
  - `FeedforwardGABrain(ClassicalBrain)` class implementing the Brain Protocol surface (`run_brain` / `update_memory` / `prepare_episode` / `post_process_episode` / `copy` + `history_data` / `latest_data`) AND the `WeightPersistence` Protocol (`get_weight_components` / `load_weight_components`).
  - Topology: `input_dim → hidden_dim → hidden_dim → 4` feed-forward with ReLU activations. Input dimensionality inferred from `sensory_modules` at construction time (mirror `MLPPPOBrain`'s pattern).
  - `run_brain()`: single forward pass → softmax → categorical sampling over `DEFAULT_ACTIONS`.
  - `update_memory()`: no-op (does NOT accumulate reward; does NOT mutate weights). `FrozenEvalRunner` already monkey-patches this to a no-op during evaluation; the brain's own implementation is the belt-and-braces guard.
  - `post_process_episode()`: no-op (does NOT compute fitness; does NOT mutate weights).
  - **Encoder shim attributes**: `self._episode_count: int = 0` (typed; written by encoder after decode; not read by GA brain logic) AND a `_update_learning_rate(self) -> None` method that is a no-op (GA brain has no LR scheduler; encoder contract requires the method to exist).
  - `@register_brain(name="feedforwardga", config_cls=FeedforwardGABrainConfig, brain_type=BrainType.FEEDFORWARDGA, families=("classical",))` decorator.
- [x] 2.2 Add `BrainType.FEEDFORWARDGA = "feedforwardga"` to `packages/quantum-nematode/quantumnematode/brain/arch/dtypes.py` (after `BrainType.CONNECTOMEPPO`).
- [x] 2.3 Add `BrainType.FEEDFORWARDGA` to the `BRAIN_TYPES` `Literal` in the same `dtypes.py` file (after `BrainType.CONNECTOMEPPO`). Both the enum entry AND the Literal entry are needed.
- [x] 2.4 Import the new module from `packages/quantum-nematode/quantumnematode/brain/arch/__init__.py` so its `@register_brain` decorator fires at startup. (Implementation note: also added `FeedforwardGABrain` + `FeedforwardGABrainConfig` to the module's `__all__` for explicit re-export alongside the other registered brains.)
- [x] 2.5 **`brain_factory.py` requires NO new code.** The default-shape fallthrough at the end of `_build_infra_kwargs` returns `{"num_actions": 4, "device": device}` which matches `FeedforwardGABrain`'s `__init__` signature. Verified by inspection; no edit needed.
- [x] 2.6 **`config_loader.py`: `BRAIN_CONFIG_MAP` requires NO new code** (post-T2 it derives from the registry automatically). However, the `BrainConfigType` discriminated union at `config_loader.py:105-126` DID require a two-line additive edit (import `FeedforwardGABrainConfig` + add `| FeedforwardGABrainConfig` to the union) so Pydantic accepts the config when `SimulationConfig` is constructed programmatically. This was not predicted by the original task wording but is correctly recorded in commit `8678fea7`'s body.

## 3. Encoder integration

- [x] 3.1 Add `FeedforwardGAEncoder(_ClassicalPPOEncoder)` to `packages/quantum-nematode/quantumnematode/evolution/encoders.py` with `brain_name = "feedforwardga"`. Subclassing `_ClassicalPPOEncoder` is correct per design.md Decision 4; the encoder shim attributes on the brain (Task 2.1's `_episode_count` + `_update_learning_rate`) make this work.
- [x] 3.2 Register the encoder in `ENCODER_REGISTRY` at the bottom of `encoders.py:370-373` (extend the existing dict literal with `FeedforwardGAEncoder.brain_name: FeedforwardGAEncoder`). (Implementation note: also updated `tests/quantumnematode_tests/evolution/test_hyperparam_encoder.py` which snapshots the `ENCODER_REGISTRY` membership set — the hard-coded `{"mlpppo", "lstmppo"}` expectation gained `"feedforwardga"`.)

## 4. Smoke config

- [x] 4.1 Author `configs/evolution/feedforwardga_foraging_small.yml` mirroring `configs/evolution/mlpppo_foraging_small.yml`'s structure: `brain:` block with `name: feedforwardga` and `config:` (topology fields only — `hidden_dim`, `num_hidden_layers`, `sensory_modules`), plus `reward:`, `satiety:`, `environment:`, and a required `evolution:` block with `algorithm: ga`, `generations: 10`, `population_size: 8`, `episodes_per_eval: 3`, `parallel_workers: 1`, `checkpoint_every: 5`, AND the GA-specific hyperparameters `sigma0: 0.5`, `elite_fraction: 0.2`, `mutation_rate: 0.1`, `crossover_rate: 0.8` (defaults from `GeneticAlgorithmOptimizer.__init__`; including them explicitly demonstrates the brain-config-doesn't-carry-GA-knobs separation per the spec's `FeedforwardGABrainConfig` requirement). (Implementation note: foraging env was tuned slightly easier than MLPPPO's equivalent — `foods_on_grid: 10`, `target_foods_to_collect: 3` vs MLPPPO's 5/10 — so the tiny smoke budget produces a non-degenerate fitness trajectory at seed 2026. Rationale + tuning is documented inline in the YAML header.)
- [x] 4.2 Smoke-run via `uv run python scripts/run_evolution.py --config configs/evolution/feedforwardga_foraging_small.yml --seed 2026`. Verified: completes without exception; `evolution_results/<session-id>/{history.csv,lineage.csv,checkpoint.pkl,best_params.json,per_gen_elites.jsonl}` artefact set produced; best-fitness 0.33 → 1.0 by gen 3; mean-fitness varies 0.04 → 0.58; std-fitness varies 0.11 → 0.47 — non-degenerate variance confirmed.

## 5. Tests

- [x] 5.1 Create `packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/test_feedforward_ga.py` (under `brain/arch/` per the existing per-brain test layout; the original task wording omitted the `arch/` subdirectory) mirroring the test structure of `packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/test_connectome_ppo.py`:
  - Test: `FeedforwardGABrain` satisfies the Brain Protocol (callable surface + attribute shape).
  - Test: `FeedforwardGABrain` satisfies the `WeightPersistence` Protocol (`isinstance(brain, WeightPersistence)` is True; `get_weight_components()` returns at least one entry; `load_weight_components()` round-trips within float32 ulp tolerance on the same instance).
  - Test: forward-pass output is a finite 4-vector matching `DEFAULT_ACTIONS` order.
  - Test: forward-pass variance across ≥ 100 samples on the smoke env is strictly positive AND categorical sampling produces ≥ 2 distinct actions across the sample (the spec's "constant action" wording interpreted as the sampled-action surface, not deterministic argmax).
  - Test: encoder round-trip — instantiate brain A, capture its weights via the encoder into a `Genome`, decode the `Genome` into a fresh brain B, assert B's network parameters match A's element-for-element within float32 ulp tolerance and that `run_brain()` produces identical pre-sampling action logits for the same `BrainParams` input.
  - Test: `learn()` / `update_memory()` / `post_process_episode()` / `prepare_episode()` are no-ops — call each with a variety of arguments and assert weights are byte-identical before vs after.
- [x] 5.2 Run `uv run pytest -m "not nightly" packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/test_feedforward_ga.py` and verify all tests pass. (26/26 pass; broader brain/arch + evolution test sweep also green — 376 tests total.)

## 6. Documentation + pre-merge verification

- [x] 6.1 Update `.claude/skills/nematode-run-evolution/skill.md` Constraints section to add `feedforwardga` to the registered-brains list (the doc currently names only `mlpppo` + `lstmppo`). **NOTE**: `.claude/` is project-wide gitignored, so this update lives in the local working tree only and cannot ship in a PR — the skill doc is a per-user copy. Recorded here for completeness; the local update has been applied.
- [x] 6.2 Run `openspec validate add-neat-weights-brain --strict` and verify clean.
- [x] 6.3 Run targeted `uv run pre-commit run --files <changed-files>` and verify clean.
- [x] 6.4 Audit staged files for `/Users/`, `/home/`, `C:\\Users\\` absolute path leakage (per project memory feedback). No leaks found.
- [x] 6.5 Audit any > 100 KB artefacts against `.gitattributes` LFS rules (per project memory feedback). Largest new file is 19.6 KB (the test file); well under the 100 KB threshold.
- [x] 6.6 Ask user before pushing the branch or opening a PR.
