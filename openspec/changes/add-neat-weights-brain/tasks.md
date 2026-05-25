# Tasks: Add NEAT-Weights Brain (`FeedforwardGABrain`)

## 1. Audit existing infrastructure (read-only; no code changes)

- [ ] 1.1 Read `packages/quantum-nematode/quantumnematode/brain/arch/mlpppo.py` to confirm the matched-capacity reference (hidden width, layer count, sensory-module consumption, forward-pass shape).
- [ ] 1.2 Read `packages/quantum-nematode/quantumnematode/brain/arch/_brain.py:411-434` to confirm the current Brain Protocol surface.
- [ ] 1.3 Read `packages/quantum-nematode/quantumnematode/brain/arch/_registry.py` + `packages/quantum-nematode/quantumnematode/brain/arch/connectome_ppo.py:428-433` (the canonical recent `@register_brain` decorator call) + `docs/architecture/plugin-developer-guide.md` to confirm the decorator signature and the registration walkthrough.
- [ ] 1.4 Read `packages/quantum-nematode/quantumnematode/optimizers/evolutionary.py:270-429` to confirm `GeneticAlgorithmOptimizer`'s public API: `__init__(num_params, x0, population_size, sigma0, elite_fraction, mutation_rate, crossover_rate, seed)`. Tournament size is hard-coded to 3 inside `_tournament_select` (not configurable); crossover is uniform (per-gene Bernoulli mask), not two-point.
- [ ] 1.5 Read `packages/quantum-nematode/quantumnematode/evolution/encoders.py:246-403` (the `_ClassicalPPOEncoder` base + `MLPPPOEncoder` / `LSTMPPOEncoder` subclasses + `ENCODER_REGISTRY` registration pattern). Note that `_ClassicalPPOEncoder.decode()` at lines 305-306 reaches for `brain._episode_count` and `brain._update_learning_rate()` — both PPO-specific; the new brain MUST provide shims.
- [ ] 1.6 Read `packages/quantum-nematode/quantumnematode/brain/weights.py:55` to confirm the `WeightPersistence` Protocol surface (`get_weight_components()`, `load_weight_components()`).
- [ ] 1.7 Read `packages/quantum-nematode/quantumnematode/evolution/fitness.py:80-128` to confirm that `FrozenEvalRunner` monkey-patches `agent.brain.learn` and `agent.brain.update_memory` to no-ops during GA evaluation (load-bearing for the brain's no-op `update_memory` design).
- [ ] 1.8 Read `packages/quantum-nematode/quantumnematode/utils/brain_factory.py` lines 54-160 to confirm the `_build_infra_kwargs` dispatch: the default-shape fallthrough at the end returns `{"num_actions": 4, "device": device}` — the new brain taking only `device` (and config) fits this default without a new branch.
- [ ] 1.9 Read `scripts/run_evolution.py` argparser to confirm the launcher's flag set (no `--theme` flag exists; evolution always runs headless).
- [ ] 1.10 Read `configs/evolution/mlpppo_foraging_small.yml` to confirm the smoke-config shape: `brain:` + `reward:` + `satiety:` + `environment:` + `evolution:` blocks. The `evolution:` block carries GA hyperparameters (NOT the brain config).

## 2. Implement `FeedforwardGABrain`

- [ ] 2.1 Create `packages/quantum-nematode/quantumnematode/brain/arch/feedforward_ga.py` with:
  - `FeedforwardGABrainConfig(BrainConfig)` Pydantic model exposing the topology fields per the spec's `FeedforwardGABrainConfig` requirement (`hidden_dim: int = 64`, `num_hidden_layers: int = 2`, `sensory_modules: list[str]`). NO GA hyperparameters in this config.
  - `FeedforwardGABrain(ClassicalBrain)` class implementing the Brain Protocol surface (`run_brain` / `update_memory` / `prepare_episode` / `post_process_episode` / `copy` + `history_data` / `latest_data`) AND the `WeightPersistence` Protocol (`get_weight_components` / `load_weight_components`).
  - Topology: `input_dim → hidden_dim → hidden_dim → 4` feed-forward with ReLU activations. Input dimensionality inferred from `sensory_modules` at construction time (mirror `MLPPPOBrain`'s pattern).
  - `run_brain()`: single forward pass → softmax → categorical sampling over `DEFAULT_ACTIONS`.
  - `update_memory()`: no-op (does NOT accumulate reward; does NOT mutate weights). `FrozenEvalRunner` already monkey-patches this to a no-op during evaluation; the brain's own implementation is the belt-and-braces guard.
  - `post_process_episode()`: no-op (does NOT compute fitness; does NOT mutate weights).
  - **Encoder shim attributes**: `self._episode_count: int = 0` (typed; written by encoder after decode; not read by GA brain logic) AND a `_update_learning_rate(self) -> None` method that is a no-op (GA brain has no LR scheduler; encoder contract requires the method to exist).
  - `@register_brain(name="feedforwardga", config_cls=FeedforwardGABrainConfig, brain_type=BrainType.FEEDFORWARDGA, families=("classical",))` decorator.
- [ ] 2.2 Add `BrainType.FEEDFORWARDGA = "feedforwardga"` to `packages/quantum-nematode/quantumnematode/brain/arch/dtypes.py` (after `BrainType.CONNECTOMEPPO`).
- [ ] 2.3 Add `BrainType.FEEDFORWARDGA` to the `BRAIN_TYPES` `Literal` in the same `dtypes.py` file (after `BrainType.CONNECTOMEPPO`). Both the enum entry AND the Literal entry are needed.
- [ ] 2.4 Import the new module from `packages/quantum-nematode/quantumnematode/brain/arch/__init__.py` so its `@register_brain` decorator fires at startup.
- [ ] 2.5 **`brain_factory.py` requires NO new code.** The default-shape fallthrough at the end of `_build_infra_kwargs` returns `{"num_actions": 4, "device": device}` which matches `FeedforwardGABrain`'s `__init__` signature. Verify by inspection — do not add a branch.
- [ ] 2.6 **`config_loader.py` `BRAIN_CONFIG_MAP` requires NO new code.** Post-T2 it derives from the registry automatically. Verify by inspection.

## 3. Encoder integration

- [ ] 3.1 Add `FeedforwardGAEncoder(_ClassicalPPOEncoder)` to `packages/quantum-nematode/quantumnematode/evolution/encoders.py` with `brain_name = "feedforwardga"`. Subclassing `_ClassicalPPOEncoder` is correct per design.md Decision 4; the encoder shim attributes on the brain (Task 2.1's `_episode_count` + `_update_learning_rate`) make this work.
- [ ] 3.2 Register the encoder in `ENCODER_REGISTRY` at the bottom of `encoders.py:370-373` (extend the existing dict literal with `FeedforwardGAEncoder.brain_name: FeedforwardGAEncoder`).

## 4. Smoke config

- [ ] 4.1 Author `configs/evolution/feedforwardga_foraging_small.yml` mirroring `configs/evolution/mlpppo_foraging_small.yml`'s structure: `brain:` block with `name: feedforwardga` and `config:` (topology fields only — `hidden_dim`, `num_hidden_layers`, `sensory_modules`), plus `reward:`, `satiety:`, `environment:`, and a required `evolution:` block with `algorithm: ga`, `generations: 10`, `population_size: 8`, `episodes_per_eval: 3`, `parallel_workers: 1`, `checkpoint_every: 5`.
- [ ] 4.2 Smoke-run via `uv run python scripts/run_evolution.py --config configs/evolution/feedforwardga_foraging_small.yml --seed 2026`. Verify: completes without exception, per-generation fitness CSV produced, best-fitness shows non-degenerate variance across generations.

## 5. Tests

- [ ] 5.1 Create `packages/quantum-nematode/tests/quantumnematode_tests/brain/test_feedforward_ga.py` mirroring the test structure of `packages/quantum-nematode/tests/quantumnematode_tests/brain/test_connectome_ppo.py`:
  - Test: `FeedforwardGABrain` satisfies the Brain Protocol (callable surface + attribute shape).
  - Test: `FeedforwardGABrain` satisfies the `WeightPersistence` Protocol (`isinstance(brain, WeightPersistence)` is True; `get_weight_components()` returns at least one entry; `load_weight_components()` round-trips byte-for-byte on the same instance).
  - Test: forward-pass output is a finite 4-vector matching `DEFAULT_ACTIONS` order.
  - Test: forward-pass variance across ≥ 100 samples on the smoke env is strictly positive.
  - Test: encoder round-trip — instantiate brain A, capture its weights via the encoder into a `Genome`, decode the `Genome` into a fresh brain B, assert B's network parameters match A's element-for-element within float32 ulp tolerance and that `run_brain()` produces identical action logits for the same `BrainParams` input (using a seeded RNG to control the sampling step).
  - Test: `update_memory()` and `post_process_episode()` are no-ops — call each with a variety of arguments and assert weights are byte-identical before vs after.
- [ ] 5.2 Run `uv run pytest -m "not nightly" packages/quantum-nematode/tests/quantumnematode_tests/brain/test_feedforward_ga.py` and verify all tests pass.

## 6. Pre-merge verification

- [ ] 6.1 Run `openspec validate add-neat-weights-brain --strict` and verify clean.
- [ ] 6.2 Run targeted `uv run pre-commit run --files <changed-files>` and verify clean.
- [ ] 6.3 Audit staged files for `/Users/`, `/home/`, `C:\\Users\\` absolute path leakage (per project memory feedback). Sanitise any found.
- [ ] 6.4 Audit any > 100 KB artefacts against `.gitattributes` LFS rules (per project memory feedback).
- [ ] 6.5 Ask user before pushing the branch or opening a PR.
