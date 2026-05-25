# Tasks: Add NEAT-Weights Brain (`FeedforwardGABrain`)

## 1. Audit existing infrastructure (read-only; no code changes)

- [ ] 1.1 Read `packages/quantum-nematode/quantumnematode/brain/arch/mlpppo.py` to confirm the matched-capacity reference (hidden width, layer count, sensory-module consumption, forward-pass shape).
- [ ] 1.2 Read `packages/quantum-nematode/quantumnematode/brain/arch/_brain.py:346-368` to confirm the current Brain Protocol surface.
- [ ] 1.3 Read `packages/quantum-nematode/quantumnematode/brain/arch/_registry.py` + `docs/architecture/plugin-developer-guide.md` to confirm the decorator-registration signature and the "5-file walkthrough" still describes the workflow accurately.
- [ ] 1.4 Read `packages/quantum-nematode/quantumnematode/optimizers/evolutionary.py:GeneticAlgorithmOptimizer` to confirm its public API and the hyperparameters it exposes (population_size, mutation_sigma, tournament_size, crossover_rate, elite_fraction).
- [ ] 1.5 Read `packages/quantum-nematode/quantumnematode/evolution/loop.py`, `evolution/genome.py`, and `evolution/encoders.py` to confirm how an existing GA-evolved brain (if any) hands its parameters to the evolution loop. If no prior GA-evolved Brain Protocol implementation exists, document this in the design.md Open Questions section before writing code.
- [ ] 1.6 Read `scripts/run_evolution.py` to confirm the launcher's brain-instantiation path consumes the L1 registry (no per-architecture branches expected post-T2).

## 2. Implement `FeedforwardGABrain`

- [ ] 2.1 Create `packages/quantum-nematode/quantumnematode/brain/arch/feedforward_ga.py` with:
  - `FeedforwardGABrainConfig(BaseBrainConfig)` Pydantic model exposing the fields enumerated in the spec's `FeedforwardGABrainConfig` requirement (hidden_dim, num_hidden_layers, population_size, mutation_sigma, tournament_size, crossover_rate, elite_fraction, sensory_modules).
  - `FeedforwardGABrain` class implementing the Brain Protocol surface (`run_brain` / `update_memory` / `prepare_episode` / `post_process_episode` / `copy` + `history_data` / `latest_data`).
  - Topology: `input_dim → hidden_dim → hidden_dim → 4` feed-forward with ReLU activations. Input dimensionality inferred from `sensory_modules` at construction time (mirror `MLPPPOBrain`'s pattern).
  - `run_brain()`: single forward pass → softmax → categorical sampling over `DEFAULT_ACTIONS`.
  - `update_memory()`: accumulate per-step reward into the per-episode fitness accumulator (no gradient computation).
  - `post_process_episode()`: finalise episode fitness (cumulative reward) and expose it for GA selection (no gradient computation).
  - `@register_brain(name="feedforwardga", config_cls=FeedforwardGABrainConfig, brain_type=BrainType.FEEDFORWARDGA, families=("classical",))` decorator.
- [ ] 2.2 Add `BrainType.FEEDFORWARDGA = "feedforwardga"` to `packages/quantum-nematode/quantumnematode/brain/arch/dtypes.py` (the enum is registry-derived post-T2 but new members still need declaration; verify against T2's pattern).
- [ ] 2.3 Import the new module from `packages/quantum-nematode/quantumnematode/brain/arch/__init__.py` so its `@register_brain` decorator fires at startup.
- [ ] 2.4 Add the loader branch for `feedforwardga` in `packages/quantum-nematode/quantumnematode/utils/brain_factory.py` (registry-consume entry per the post-T2 convention; ≤ 5 LOC).
- [ ] 2.5 Add the YAML-name → config-class mapping in `packages/quantum-nematode/quantumnematode/utils/config_loader.py` `BRAIN_CONFIG_MAP` if T2's registry doesn't auto-derive it (verify against the T2 audit; the post-T2 docs/architecture/plugin-developer-guide.md is authoritative).

## 3. Wire to evolution-framework

- [ ] 3.1 Verify (don't modify) that `EvolutionLoop` consumes the new brain through the L1 registry without per-architecture branches. If branches are required, stop and revisit Decision 2 in design.md.
- [ ] 3.2 Confirm `BrainGenome` encoder (`evolution/encoders.py`) can serialise + deserialise `FeedforwardGABrain`'s weight tensors via the existing `state_dict()` pattern. If a new encoder is needed, document the gap in design.md and scope it explicitly.

## 4. Smoke config

- [ ] 4.1 Author `configs/scenarios/foraging/feedforwardga_small_klinotaxis.yml` mirroring `configs/scenarios/foraging/connectomeppo_small_klinotaxis.yml`'s structure (klinotaxis sensing, env grid size, foraging foods, headless-runnable). Use the GA defaults from the spec's `FeedforwardGABrainConfig` requirement.
- [ ] 4.2 Smoke-run via `uv run python scripts/run_evolution.py --config configs/scenarios/foraging/feedforwardga_small_klinotaxis.yml --theme headless --generations 10 --seed 2026`. Verify: completes without exception, per-generation fitness CSV produced, best-fitness shows non-degenerate variance across generations.

## 5. Tests

- [ ] 5.1 Create `packages/quantum-nematode/tests/quantumnematode_tests/brain/test_feedforward_ga.py`:
  - Test: `FeedforwardGABrain` satisfies the Brain Protocol (callable surface + attribute shape).
  - Test: forward-pass output is a finite 4-vector matching `DEFAULT_ACTIONS` order.
  - Test: forward-pass variance across ≥ 100 samples on the smoke env is strictly positive.
  - Test: GA-update determinism — two evolution runs with the same seed produce byte-identical per-generation best-fitness trajectories and byte-identical final-generation best-genome weight tensors.
- [ ] 5.2 Run `uv run pytest -m "not nightly" packages/quantum-nematode/tests/quantumnematode_tests/brain/test_feedforward_ga.py` and verify all tests pass.

## 6. Pre-merge verification

- [ ] 6.1 Run `openspec validate add-neat-weights-brain --strict` and verify clean.
- [ ] 6.2 Run `uv run pre-commit run --files <changed-files>` and verify clean.
- [ ] 6.3 Audit staged files for `/Users/`, `/home/`, `C:\\Users\\` absolute path leakage (per project memory feedback). Sanitise any found.
- [ ] 6.4 Audit any > 100 KB artefacts against `.gitattributes` LFS rules (per project memory feedback).
- [ ] 6.5 Ask user before pushing the branch or opening a PR.
