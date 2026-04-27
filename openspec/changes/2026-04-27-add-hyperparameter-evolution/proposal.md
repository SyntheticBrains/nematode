## Why

Phase 5's M0 (PR #132) shipped a brain-agnostic evolution framework that evolves brain weights via `MLPPPOEncoder` / `LSTMPPOEncoder`. It deliberately ships only `EpisodicSuccessRate` (frozen-weight fitness) and only weight encoders. M2 — the Hyperparameter Evolution Pilot — needs two new building blocks the framework was designed to support but doesn't yet implement:

1. **`HyperparameterEncoder`** — encodes brain hyperparameters (e.g. `learning_rate`, `actor_hidden_dim`, `feature_gating`, `rnn_type`) as a flat float vector with a per-slot schema, instead of weights. Each evaluation builds a *fresh* brain from the evolved hyperparameters and trains it from scratch.
2. **`LearnedPerformanceFitness`** — runs K training episodes (where `brain.learn()` IS called and weights mutate) followed by L frozen eval episodes (the existing `FrozenEvalRunner` for the eval phase). Score = eval-phase success rate.

These two slot into the existing `GenomeEncoder` / `FitnessFunction` protocols without changing them — the framework was designed for this. M2 also delivers the **MLPPPO pilot**: a real campaign that exercises the new encoder + fitness end-to-end, gives us a GO/PIVOT/STOP decision for hyperparameter evolution as a Phase 5 strategy, and ships before we sign up for the slower LSTMPPO arm in PR 3.

This change is **PR 2 of three** in the post-M0 evolution work split:

- **PR 1** (perf, merged as #133): cut per-step dead work + opt-in CMA-ES diagonal mode. Unblocks LSTMPPO weight evolution and is a hard prerequisite for the LSTMPPO arm of M2.
- **PR 2** (this change): M2 framework + MLPPPO pilot.
- **PR 3** (separate change, post-merge): LSTMPPO+klinotaxis pilot. Pure config + logbook + checklist tick. Ships only if PR 2's MLPPPO decision is GO or PIVOT.

User-confirmed design choices (from the approved plan):

- **Schema location**: top-level YAML key `hyperparam_schema:` (peer of `brain:`, `environment:`, `evolution:`). Not nested under `evolution:`.
- **Capability**: extends the existing `evolution-framework` capability (deltas only — no new capability spec).
- **Archive in-branch** following the M0 pattern: same PR ships impl + archive move + roadmap row update.

## What Changes

### 1. `HyperparameterEncoder` in `evolution/encoders.py`

Adds a third concrete encoder peer of `MLPPPOEncoder` and `LSTMPPOEncoder`. Conforms to the existing `GenomeEncoder` protocol unchanged. Stores a `param_schema` (list of typed entries) in `Genome.birth_metadata` so workers can decode without a side channel.

Per-slot supported types: `float` (with optional `log_scale`), `int`, `bool`, `categorical`. `decode()` reads each schema entry, slices the corresponding scalar from `genome.params`, applies the type-appropriate transform (clip-and-round for int, `> 0` for bool, nearest-bin for categorical, optional log-space for float), and applies the result to `sim_config.brain.config` via `model_copy(update={...})`. Then dispatches to `instantiate_brain_from_sim_config` to build a fresh brain.

**The hyperparameter genome does NOT include weights.** Weights are freshly initialised at every fitness evaluation and adapted during the train phase of `LearnedPerformanceFitness`. This is the central distinction from `MLPPPOEncoder` / `LSTMPPOEncoder`.

`HyperparameterEncoder` is **not** registered in `ENCODER_REGISTRY` — that registry is keyed by brain name (`"mlpppo"`, `"lstmppo"`) and is reserved for brain-specific encoders. `HyperparameterEncoder` is brain-agnostic; the dispatch layer (`evolution.encoders.select_encoder`) constructs it directly when `sim_config.hyperparam_schema is not None` and falls back to the M0 `get_encoder(brain.name)` path otherwise. See Decision 0 in `design.md` for the rationale.

### 2. `LearnedPerformanceFitness` in `evolution/fitness.py`

Adds a second concrete fitness peer of `EpisodicSuccessRate`. Conforms to the existing `FitnessFunction` protocol unchanged.

Per-evaluation flow:

1. `brain = encoder.decode(genome, sim_config, seed=seed)` — fresh brain with the genome's hyperparameters.
2. `env = create_env_from_config(sim_config.environment, seed=seed, theme=Theme.HEADLESS)` — same as `EpisodicSuccessRate`.
3. **Train phase**: `runner = StandardEpisodeRunner()`; loop `K = evolution_config.learn_episodes_per_eval` episodes. Brain's `learn()` fires per-step (the standard contract).
4. **Eval phase**: `runner = FrozenEvalRunner()` (M0's existing class); loop `L = evolution_config.eval_episodes_per_eval or evolution_config.episodes_per_eval` episodes. Count successes via `result.termination_reason == TerminationReason.COMPLETED_ALL_FOOD`.
5. Return `successes / L`.

`K = 0` is rejected with a clear `ValueError` (caller should use `EpisodicSuccessRate` instead). When `eval_episodes_per_eval` is None, falls back to `evolution_config.episodes_per_eval` for back-compat with M0-shaped configs.

### 3. `EvolutionConfig` extensions

Two new fields on `quantumnematode.utils.config_loader.EvolutionConfig`:

- `learn_episodes_per_eval: int = Field(default=0, ge=0)` — when 0, `LearnedPerformanceFitness.evaluate` raises (callers must opt in explicitly).
- `eval_episodes_per_eval: int | None = Field(default=None, ge=1)` — None falls back to `episodes_per_eval`.

Existing M0 configs are unaffected: `learn_episodes_per_eval=0` is the default and `EpisodicSuccessRate` ignores both fields.

### 4. `hyperparam_schema:` YAML key on `SimulationConfig`

New optional top-level field `hyperparam_schema: list[ParamSchemaEntry] | None = None` on `SimulationConfig`. Each entry is a Pydantic model with `name: str`, `type: Literal["float", "int", "bool", "categorical"]`, plus type-specific fields (`bounds`, `values`, `log_scale`).

Schema authoring footgun mitigation: load-time validation cross-checks every `name` field against the resolved brain config Pydantic model and rejects misspelled or non-existent fields. Without this, `model_copy(update={...})` would silently no-op typos and produce unevolved genomes (a known Pydantic v2 footgun).

When `hyperparam_schema` is present, the run_evolution CLI selects `HyperparameterEncoder` automatically (no separate `--encoder` flag). When absent, the CLI keeps M0 behaviour: dispatch by `brain.name`.

### 5. CLI flag: `--fitness {success_rate,learned_performance}`

`scripts/run_evolution.py` gains a `--fitness` flag with `success_rate` as the default (preserves M0 behaviour). When `learned_performance` is selected, the CLI validates BOTH that (a) `sim_config.hyperparam_schema is not None` and (b) `evolution.learn_episodes_per_eval > 0`, rejecting with a clear error otherwise. The first guard prevents accidentally combining a weight-encoder with `LearnedPerformanceFitness` — which would amount to Lamarckian inheritance (M3 scope, not this PR). See Decision 0 in `design.md` for the rationale.

### 6. MLPPPO pilot config

New `configs/evolution/hyperparam_mlpppo_pilot.yml` — 7 evolved hyperparameters (`actor_hidden_dim`, `critic_hidden_dim`, `num_hidden_layers`, `learning_rate` log-scale, `gamma`, `entropy_coef` log-scale, `num_epochs`). Pilot sizing: `generations: 20`, `population_size: 12`, `learn_episodes_per_eval: 30`, `eval_episodes_per_eval: 5`, `parallel_workers: 4`, `checkpoint_every: 2`, `cma_diagonal: false` (n=7 is well within full-cov tractability).

### 7. Campaign script

New `scripts/campaigns/phase5_m2_hyperparam_mlpppo.sh` — orchestrates the 4-seed pilot, captures per-seed `evolution_results/<session>/` directories, and feeds into the logbook write-up.

### 8. MLPPPO pilot run + logbook

Pilot is run during the spec-review/implementation window so the PR body includes the actual GO/PIVOT/STOP decision and the convergence curve plot. Logbook lands at `artifacts/logbooks/012/hyperparam_pilot_mlpppo.md`. Decision gate from the Phase 5 milestone scaffold (recorded in [`2026-04-26-phase5-tracking/tasks.md`](../2026-04-26-phase5-tracking/tasks.md), M2 section): GO if either brain ≥3pp over hand-tuned baseline AND fitness still rising at gen 20; PIVOT if marginal; STOP if no separation. This PR ships only the MLPPPO arm of the gate; the LSTMPPO arm lands in PR 3 if applicable.

### 9. Spec deltas

Three new requirements added to the existing `evolution-framework` capability (deltas only — capability is extended, not replaced):

- Hyperparameter Encoding (encoder protocol scenario for HyperparameterEncoder)
- Learned-Performance Fitness (K-train then L-eval)
- Hyperparameter Schema YAML (top-level YAML field + name validation)

### 10. M-1 invariant updates

Per the Phase 5 tracking change's invariant ([`2026-04-26-phase5-tracking/specs/phase5-tracking/spec.md`](../2026-04-26-phase5-tracking/specs/phase5-tracking/spec.md)), the same PR diff updates:

- `openspec/changes/2026-04-26-phase5-tracking/tasks.md` — tick `M2.1`-`M2.5` and `M2.7` (MLPPPO arm of the milestone).
- `docs/roadmap.md` Phase 5 Milestone Tracker — set M2 row to `🟡 in progress` (LSTMPPO arm pending PR 3).

### 11. Tests (new)

Under `packages/quantum-nematode/tests/quantumnematode_tests/evolution/` and `tests/quantumnematode_tests/`:

- `test_hyperparam_encoder.py` — round-trip determinism per type; categorical bin-boundary; log-scale round-trip; schema name validation rejects bogus fields; encoder is **not** in `ENCODER_REGISTRY` (per Decision 0 + spec scenario "Hyperparameter encoder is NOT in the brain-keyed registry"); encoder produces a brain whose `BrainConfig` matches the decoded values within float tolerance.
- `test_learned_fitness.py` — K=2/L=1 smoke; K=0 raises; missing `evolution`/`environment`/`reward` blocks raise; `eval_episodes_per_eval=None` falls back to `episodes_per_eval`; train phase actually mutates weights (anti-regression); eval phase doesn't (uses FrozenEvalRunner).
- `test_config.py` extension — `ParamSchemaEntry` type-conditional metadata validation; `hyperparam_schema:` YAML parses; name validation catches typos; schema requires brain block; unknown brain name fails clearly; `EvolutionConfig.learn_episodes_per_eval`/`eval_episodes_per_eval` defaults and bounds.
- `test_encoders.py` extension (existing M0 file) — `select_encoder(sim_config)` dispatch tests including the brain-agnostic case for brains in `BRAIN_CONFIG_MAP` but NOT in `ENCODER_REGISTRY` (e.g., `qvarcircuit`); `build_birth_metadata` helper contract.
- `test_loop_smoke.py` extension (existing M0 file) — Phase 4.5 wiring tests covering `birth_metadata["param_schema"]` population at both `Genome` construction sites and the `brain_type` fallback for hyperparameter runs (`encoder.brain_name == ""` falls through to `sim_config.brain.name`); existing M0 assertions (e.g. `artefact["brain_type"] == "mlpppo"`) MUST remain green.
- `test_smoke.py` extension — new `@pytest.mark.smoke test_run_evolution_smoke_hyperparam_mlpppo` exercising the CLI end-to-end with `--fitness learned_performance` against the pilot config (1 gen × pop 4 × K=2 / L=1); plus subprocess CLI tests for `--fitness` flag default, K=0 rejection (schema-then-K guard ordering), and missing-`hyperparam_schema` rejection with the Lamarckian-inheritance/M3 rationale.

## Capabilities

**Extended**: `evolution-framework` (no new capability). Three new requirements; existing six requirements unchanged.

## Impact

**Code:**

- `packages/quantum-nematode/quantumnematode/evolution/encoders.py` — add `HyperparameterEncoder` (NOT registered in `ENCODER_REGISTRY` — see Decision 0); add public `select_encoder(sim_config)` dispatch helper peer of M0's `get_encoder`
- `packages/quantum-nematode/quantumnematode/evolution/fitness.py` — add `LearnedPerformanceFitness`
- `packages/quantum-nematode/quantumnematode/evolution/loop.py` — two distinct edits, both covered in Phase 4.5 of `tasks.md`:
  - **`birth_metadata` wiring** (tasks 4.5.1-4.5.2): populate `birth_metadata["param_schema"]` from `sim_config.hyperparam_schema` at both `Genome` construction sites (worker handoff + lineage record) via the shared `build_birth_metadata(sim_config)` helper. Without this, hyperparameter genomes reach workers with empty `birth_metadata` and `HyperparameterEncoder.decode` cannot recover the schema.
  - **`brain_type` fallback** (task 4.5.7): the two `self.encoder.brain_name` reads at [loop.py:262](packages/quantum-nematode/quantumnematode/evolution/loop.py#L262) (`lineage.csv` rows) and [loop.py:378](packages/quantum-nematode/quantumnematode/evolution/loop.py#L378) (`best_params.json`) gain a fallback to `self.sim_config.brain.name` so that hyperparameter runs (whose encoder has `brain_name == ""`) still record the YAML's brain name, not the empty string.
- `packages/quantum-nematode/quantumnematode/utils/config_loader.py` — add `ParamSchemaEntry` model; add `hyperparam_schema` field on `SimulationConfig`; add `learn_episodes_per_eval` + `eval_episodes_per_eval` on `EvolutionConfig`; add cross-field validator for schema-name correctness
- `scripts/run_evolution.py` — three edits in `main()`, all in Phase 5:
  - Add `--fitness {success_rate, learned_performance}` argparse flag (default `success_rate` preserves M0 behaviour).
  - Replace the M0 brain-registry gate at [run_evolution.py:237-246](scripts/run_evolution.py#L237) (`if brain_name not in ENCODER_REGISTRY: ...; encoder = get_encoder(brain_name)`) with a single `encoder = select_encoder(sim_config)` call wrapped in try/except for `ValueError` surfacing. Without this replacement, the M0 gate would reject hyperparameter runs against any brain in `BRAIN_CONFIG_MAP` but not `ENCODER_REGISTRY` (e.g., `qvarcircuit`), defeating Decision 0's brain-agnostic dispatch design.
  - When `--fitness learned_performance`, validate `hyperparam_schema is not None` (first guard, the more fundamental check) and `evolution_config.learn_episodes_per_eval > 0` (second guard); reject with exit code 1 otherwise.

**Configs:**

- `configs/evolution/hyperparam_mlpppo_pilot.yml` (new)

**Tests:**

- `packages/quantum-nematode/tests/quantumnematode_tests/evolution/test_hyperparam_encoder.py` (new)
- `packages/quantum-nematode/tests/quantumnematode_tests/evolution/test_learned_fitness.py` (new)
- `packages/quantum-nematode/tests/quantumnematode_tests/evolution/test_config.py` (extended)
- `packages/quantum-nematode/tests/quantumnematode_tests/test_smoke.py` (extended)

**Scripts:**

- `scripts/campaigns/phase5_m2_hyperparam_mlpppo.sh` (new)

**Docs:**

- `artifacts/logbooks/012/hyperparam_pilot_mlpppo.md` (new — published as part of PR)
- `openspec/changes/2026-04-26-phase5-tracking/tasks.md` — M2.1–M2.5, M2.7 ticked (MLPPPO arm); plus a new "Phase 5 Research Questions" section recording RQ1 (Optimiser portfolio re-evaluation) — see `design.md` § Considered Alternatives — Optimiser choice for the rationale
- `docs/roadmap.md` Phase 5 Milestone Tracker — M2 row → `🟡 in progress`; plus a "Phase 5 research questions" paragraph after the tracker pointing readers at the new section
- `.claude/skills/nematode-run-evolution/skill.md` — short note on the `--fitness learned_performance` mode

## Breaking Changes

None. All additions are opt-in:

- `--fitness` defaults to `success_rate` (M0 behaviour).
- `hyperparam_schema` is `None` by default; absent in existing scenario configs.
- `learn_episodes_per_eval=0` by default; M0's `EpisodicSuccessRate` flow unaffected.
- `HyperparameterEncoder` only dispatches when `hyperparam_schema` is set; otherwise the existing brain.name → encoder dispatch is unchanged.

## Backward Compatibility

- All existing scenario and evolution configs (`configs/scenarios/**/*.yml`, `configs/evolution/{mlpppo,lstmppo}_foraging_small*.yml`) — load and run unchanged.
- M0 framework smoke (`test_run_evolution_smoke_mlpppo`) — same behaviour.
- LSTMPPO weight-evolution pilot — unaffected (PR 3 is a separate config, not a re-aim of this one).
