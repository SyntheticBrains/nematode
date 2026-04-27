## Overview

M2 adds two protocol-conforming components — `HyperparameterEncoder` and `LearnedPerformanceFitness` — and threads a new `hyperparam_schema:` YAML key through to the encoder. Everything else (the `EvolutionLoop`, optimisers, lineage tracker, checkpoint/resume, worker init, `Theme.HEADLESS` plumbing) is reused from M0 unchanged. This design records the non-obvious choices.

## Goals / Non-Goals

**Goals:**

- Evolve brain hyperparameters as a flat float vector with a per-slot schema, decode-time type interpretation, and Pydantic load-time field-name validation
- Train-then-eval fitness that reuses both the standard runner (train phase) and `FrozenEvalRunner` (eval phase) without forking either
- Opt-in everywhere: zero behavioural change for existing M0 configs/runs
- MLPPPO pilot lands with the framework so PR review can include actual GO/PIVOT/STOP results, not just code

**Non-Goals:**

- LSTMPPO pilot — separate PR (PR 3) gated on this one's decision
- Lamarckian / Baldwin inheritance — explicit M3/M4 deliverables
- Bayesian optimisation, random search baselines — out of M2 scope; CMA-ES + GA are the only optimisers
- Multi-objective fitness — single scalar success rate is fine for the pilot
- Hyperparameter search space inferred from brain config types automatically — explicit schema authoring is the M2 contract

## Design Decisions

### Decision 0: Encoder dispatch by `hyperparam_schema` presence, not by a new `--encoder` flag

The simplest API would be a CLI flag `--encoder {weights,hyperparam}` that selects between weight-evolution and hyperparameter-evolution. Rejected for two reasons:

1. **Pilot configs would have to repeat the choice on the CLI every time.** A `hyperparam_mlpppo_pilot.yml` *is* a hyperparameter pilot — that's its identity. Routing through a CLI flag means anyone running the pilot has to remember to pass the flag, and forgetting it would silently produce a weight-evolution run instead.
2. **The presence of `hyperparam_schema:` in YAML uniquely determines the encoder.** Without a schema, hyperparameter encoding is undefined; with one, weight encoding doesn't make sense (the genome dim is set by the schema, not the brain's weight count).

**Convention:** when `sim_config.hyperparam_schema is not None`, dispatch to `HyperparameterEncoder`. Otherwise, dispatch by `sim_config.brain.name` as M0 does. The CLI gains no new encoder flag.

**Implementation note:** the dispatch logic lives in a public helper `select_encoder(sim_config: SimulationConfig) -> GenomeEncoder` in `quantumnematode.evolution.encoders` (peer of `get_encoder`, the M0 brain-name dispatch). Reasons: (a) keeps `scripts/run_evolution.py:main()` thin; (b) makes the dispatch testable in isolation (see task 6.5 — without the helper, the only test surface is subprocess-invoking the CLI); (c) makes the helper available to any programmatic caller of `EvolutionLoop`, not just the CLI.

`select_encoder` does NOT route through `ENCODER_REGISTRY` for `HyperparameterEncoder` — it constructs `HyperparameterEncoder()` directly when `sim_config.hyperparam_schema is not None`. The registry is reserved for brain-name → encoder mappings (M0's two encoders); `HyperparameterEncoder` is not a brain-specific encoder and shouldn't pollute the brain-keyed registry. See the spec scenario "Hyperparameter encoder is NOT in the brain-keyed registry".

The `--fitness` flag IS added (for `success_rate` vs `learned_performance`) because there's still one real choice: a hyperparameter pilot could use `EpisodicSuccessRate` for a sanity baseline (frozen-weight eval of the random-init brain — useful to confirm the schema decode path before committing to a 30-train-episodes-per-genome budget). The reverse combination — weight encoder + `LearnedPerformanceFitness` — would amount to Lamarckian inheritance (the genome carries weights, the train phase further mutates them), which is M3's milestone responsibility, not this PR's. To avoid silently shipping a half-formed M3 prototype, M2's CLI rejects `--fitness learned_performance` when `hyperparam_schema is None`. M3 will revisit and either lift this restriction or replace `LearnedPerformanceFitness` with a Lamarckian-aware variant.

**Why:** matches the YAML-as-source-of-truth pattern from M0 (the `cma_diagonal: true` setting in the LSTMPPO pilot YAML doesn't need a CLI flag either — it just is what the campaign needs).

### Decision 1: `param_schema` is authored explicitly in YAML, not inferred from brain config types

Pydantic introspection could in principle auto-build a schema by walking `MLPPPOBrainConfig.model_fields` and using each field's type/bounds. Rejected:

1. **Most config fields shouldn't be evolved.** `seed`, `weights_path`, `_episode_count` etc. would be in the auto-schema and need a denylist. The denylist is itself an explicit author choice; once you have a denylist you're 80% of the way to an explicit allowlist.
2. **Bounds aren't on the config types.** `learning_rate: float = 0.0003` has no library-encoded bounds. The reasonable evolution range (`[1e-5, 1e-2]` log-scale) is an experimenter's choice.
3. **Future configs (M3 Lamarckian, M5 co-evolution) will evolve different subsets** — auto-inference would silently expand the search space when adding a new field to `MLPPPOBrainConfig`.

So the user authors a list under `hyperparam_schema:` in the pilot YAML. The encoder reads it, validates each `name` against the resolved brain config Pydantic model, and uses it as the source of truth for both genome dim and decode mapping.

### Decision 2: Schema-name validation at YAML load time, not at decode time

A typo in `hyperparam_schema[].name` (e.g. `actor_hidden_dimm`) would silently no-op via `model_copy(update={...})` in Pydantic v2 — `model_copy` does not validate keys against the model schema. The genome would evolve a slot that does nothing. This is exactly the kind of bug that a long-running pilot won't surface for hours.

**Decision:** add a Pydantic `@model_validator(mode="after")` on `SimulationConfig` that, when `hyperparam_schema is not None`, walks the schema and asserts every `name` is a real field on the resolved brain config Pydantic model. Reject the YAML at load time with a clear message naming the bogus field and the available options.

This requires `SimulationConfig` to know how to resolve "the brain config" — which is `sim_config.brain.config` typed as a union of all brain config classes. The validator dispatches via `BRAIN_CONFIG_MAP` (already imported in `config_loader.py`) to locate the concrete config class for the brain in question, then queries its `model_fields`.

**Why:** the alternative ("fail at first decode") burns hours of pilot wall time before surfacing a 1-character typo. Load-time validation catches it before the run starts.

### Decision 3: Genome layout — flat float vector with per-slot decode

Each schema entry occupies exactly one float slot in `genome.params`. CMA-ES samples a `num_params`-dim vector; the encoder slices each slot and applies the type-appropriate transform.

**Per-type transforms** (decode side):

- `float`: clip to `bounds`, then if `log_scale: true`, `value = exp(value)` (with the genome value being in log-space throughout); otherwise return as-is
- `int`: clip to `bounds`, then `int(round(value))`
- `bool`: `value > 0.0` (clip-at-zero — the same convention CMA-ES samples adapt to)
- `categorical`: `int(round(value)) mod len(values)`, then index into `values`

`initial_genome` does the inverse: samples uniform-in-bounds for floats/ints (log-uniform when `log_scale: true`), uniform-in-`{False, True}` for bools (sampled as ±1 then thresholded by the same `> 0` rule for round-trip consistency), uniform integer for categoricals (sampled as `[0, len(values))` index).

**Genome dim** = `len(param_schema)`. Tractable for full-cov CMA-ES (n=7 for the MLPPPO pilot). `cma_diagonal: false` is correct here.

**Why a flat float vector even for mixed types:** keeps the genome shape compatible with `CMAESOptimizer.ask()` and `GeneticAlgorithmOptimizer.ask()` unchanged. The optimisers don't know about types — they sample R^n and the encoder interprets.

**Acknowledged limitation: CMA-ES sees flat fitness plateaus across categorical bins.** A slot with `values: [lstm, gru]` decodes the same brain across an interval of float values, so the optimiser perceives a flat region in fitness space. For the M2 MLPPPO pilot this doesn't bite (no categoricals). The LSTMPPO pilot in PR 3 has one (`rnn_type`) — out of scope for this PR but documented for future readers.

### Decision 4: `ParamSchemaEntry` is a Pydantic model with type-conditional fields

```python
class ParamSchemaEntry(BaseModel):
    name: str
    type: Literal["float", "int", "bool", "categorical"]
    bounds: tuple[float, float] | None = None  # required for float/int; None for bool/categorical
    values: list[str] | None = None             # required for categorical; None for others
    log_scale: bool = False                     # only meaningful for float; ignored elsewhere
```

A `model_validator(mode="after")` enforces:

- `type=float`: requires `bounds`, `values is None`
- `type=int`: requires `bounds`, `values is None`, `log_scale=False`
- `type=bool`: `bounds is None`, `values is None`, `log_scale=False`
- `type=categorical`: requires `values` with ≥2 items, `bounds is None`, `log_scale=False`

These constraints fire at YAML load time. A schema entry with `{type: float, values: [a,b]}` is a contradiction and gets rejected immediately.

`birth_metadata["param_schema"]` stores the same entries as a list of **plain dicts** (NOT Pydantic model instances), produced via `entry.model_dump()`. The dicts pickle cheaply, decouple worker decode from a Pydantic-import dependency, and trivially survive Pydantic version upgrades. The loop is responsible for populating this metadata when constructing genomes — see Phase 4.5 in `tasks.md` and the spec scenario "Schema travels with the genome to workers".

**Coexistence with M0's `birth_metadata`:** M0 weight encoders populate `birth_metadata={"shape_map": ..., "brain_name": ...}` ([encoders.py:237](packages/quantum-nematode/quantumnematode/evolution/encoders.py#L237)). M2's `HyperparameterEncoder` populates `birth_metadata={"param_schema": [...]}`. The two key sets are independent and never coexist on the same genome — encoders are mutually exclusive per Decision 0 (`select_encoder` returns exactly one of them based on `hyperparam_schema` presence). The loop's wiring in Phase 4.5 reflects this: it populates `param_schema` when `sim_config.hyperparam_schema is not None`, and otherwise leaves `birth_metadata` empty (the M0 fallback in `_ClassicalPPOEncoder.decode` then derives `shape_map` from the fresh template at decode time — see [encoders.py:252-257](packages/quantum-nematode/quantumnematode/evolution/encoders.py#L252)).

### Decision 5: `LearnedPerformanceFitness` reuses both runners, no fork; rebuilds env between train and eval

Train phase = `StandardEpisodeRunner` (M0 standard contract — calls `learn` per-step + on success). Eval phase = `FrozenEvalRunner` (M0 dual-override). No new runner subclass is introduced.

**Critical: env state is reset between the train and eval phases.**

The runners reset the *brain*'s per-episode state (hidden states, STAM buffer, food handler, episode tracker — see `runners.py:640-649`) but they do NOT reset the *environment*'s state (food positions, agent position, agent HP, body length). After a 30-episode train phase, the env is in arbitrary post-training state — food possibly all consumed, agent in some corner, HP depleted. If we ran the eval phase against that env, we'd be measuring "how does the trained brain recover from where training left it" rather than "how does the trained brain perform from a fresh start."

The fitness function therefore **builds a fresh env for the eval phase** via a second `create_env_from_config(...)` call with the same seed. The brain's learned weights persist across the train→eval transition (that's the entire point of training); the env starts cleanly.

Per-evaluation flow:

```python
def evaluate(self, genome, sim_config, encoder, *, episodes, seed):
    # Defensive guards — mirror M0's EpisodicSuccessRate.evaluate which
    # guards `environment` and `reward` similarly (fitness.py:207-221).
    if sim_config.evolution is None:
        msg = "LearnedPerformanceFitness requires an `evolution:` block in the YAML ..."
        raise ValueError(msg)
    if sim_config.environment is None:
        msg = "LearnedPerformanceFitness.evaluate requires sim_config.environment to be set."
        raise ValueError(msg)
    if sim_config.reward is None:
        msg = "LearnedPerformanceFitness.evaluate requires sim_config.reward to be set."
        raise ValueError(msg)
    evolution_config = sim_config.evolution  # alias for ergonomic reads below

    if evolution_config.learn_episodes_per_eval == 0:
        msg = "LearnedPerformanceFitness requires learn_episodes_per_eval > 0; ..."
        raise ValueError(msg)

    # Resolve max_steps with the same fallback as M0's EpisodicSuccessRate
    # (fitness.py:222) — sim_config.max_steps is Optional, default 500.
    max_steps = sim_config.max_steps if sim_config.max_steps is not None else 500

    brain = encoder.decode(genome, sim_config, seed=seed)  # fresh, hyperparam-set brain

    # Train phase — fresh env; brain's weights mutate as it learns
    train_env = create_env_from_config(sim_config.environment, seed=seed, theme=Theme.HEADLESS)
    train_agent = _build_agent(brain, train_env, sim_config)
    train_runner = StandardEpisodeRunner()
    K = evolution_config.learn_episodes_per_eval
    for _ in range(K):
        train_runner.run(train_agent, sim_config.reward, max_steps)

    # Eval phase — fresh env (CRITICAL: post-train env state is arbitrary;
    # eval needs a clean start to measure trained-brain performance, not
    # post-training-residual performance).  Brain carries over with its
    # learned weights.
    eval_env = create_env_from_config(sim_config.environment, seed=seed, theme=Theme.HEADLESS)
    eval_agent = _build_agent(brain, eval_env, sim_config)
    eval_runner = FrozenEvalRunner()
    # L resolution:
    #  - if YAML set evolution.eval_episodes_per_eval, use that (no CLI override exists)
    #  - else fall back to the `episodes` kwarg, which is the loop's already-resolved
    #    `evolution_config.episodes_per_eval` (CLI overrides like `--episodes` are
    #    applied by `_resolve_evolution_config` in run_evolution.py BEFORE the loop
    #    is constructed, so `episodes` here is the user's effective value).  Reading
    #    `sim_config.evolution.episodes_per_eval` directly would miss CLI overrides.
    eval_eps = sim_config.evolution.eval_episodes_per_eval
    L = eval_eps if eval_eps is not None else episodes
    successes = 0
    for _ in range(L):
        result = eval_runner.run(eval_agent, sim_config.reward, max_steps)
        if result.termination_reason == TerminationReason.COMPLETED_ALL_FOOD:
            successes += 1
    return successes / L
```

The `episodes` arg from the `FitnessFunction` protocol is **the eval-phase fallback** for `LearnedPerformanceFitness` — used when `evolution.eval_episodes_per_eval` is None. The kwarg is already CLI-override-aware (the loop passes the loop's own resolved `evolution_config.episodes_per_eval`), so this fallback respects `--episodes`.

**Asymmetry:** `learn_episodes_per_eval` (K) is read directly from `sim_config.evolution.learn_episodes_per_eval` because there's no CLI override for it. Only `episodes_per_eval` is CLI-overridable today (via `--episodes`); if a future PR adds `--learn-episodes-per-eval` and `--eval-episodes-per-eval` flags, the resolution paths for K and L should converge.

**Why use the protocol kwarg rather than mutating `sim_config`:** the loop holds the resolved `evolution_config` on itself but does NOT patch `sim_config.evolution` to match. Fitness functions that need the resolved value have two options: (a) read the protocol's `episodes` kwarg (which the loop already passes from its resolved config), or (b) require the loop to also pass `evolution_config` as a new kwarg (broader protocol change). Option (a) keeps the protocol unchanged; option (b) would be cleaner long-term but is broader-scope and not needed for M2. We choose (a) and document the asymmetry.

**Why same seed for train and eval env:** the two envs sample the same initial food layout, agent start position, etc. The brain has already "seen" that layout during training — but it has only *learned weights* from that experience, not memoised the layout (the brain's runtime state is reset by `prepare_episode` between every run). Same-seed eval is the standard approach for evaluating learned policy quality on a known landscape; differentiating train and eval seeds would test generalisation, which is M5's co-evolution / M6's transgenerational concern, not M2's. If a future variant wants train/eval seed split (e.g. for held-out generalisation measurement), it should be a peer fitness class rather than a knob on this one.

**Train→eval brain-state carryover (acknowledged but benign):**

- `MLPPPOBrain.buffer` (the `RolloutBuffer` constructed once at `__init__`) may carry residual transitions into the eval phase if the train phase finished without firing a final PPO update (i.e. `is_full()` was False AND `len(buffer) < num_minibatches` at the last train episode's end). Eval doesn't read or write the buffer (`FrozenEvalRunner` neuters `agent.brain.learn`), so the residual transitions are never observed. No fitness impact.
- `MLPPPOBrain._episode_count` increments via `post_process_episode` on every episode, including eval. By the end of `evaluate(K=30, L=5)` it equals 35 and the LR scheduler has been called 35 times. Eval doesn't use LR for action selection (only for weight updates, which don't happen), so action distributions are unaffected. No fitness impact.
- Both observations are recorded here so a future review pass that spots them can confirm they're already-considered rather than re-discovered.

### Decision 6: Train phase runs in the same process as eval, not on a separate Pool

A train phase that mutates a 9k-weight MLPPPO brain over 30 episodes is real work — but it's still small enough to run inline within the existing per-genome worker. We do NOT spawn a separate optimiser-style training worker.

Reasons:

- The fitness function's contract is "given a genome, return a fitness score." Spawning sub-workers from inside a worker is a portability headache (multiprocessing semantics on macOS spawn vs Linux fork) and adds checkpoint/resume corner cases.
- Per-genome wall time is dominated by the eval episodes anyway (30 train + 5 eval at ~50 ms/episode = 1.75 s/genome — the parallel fitness Pool already amortises this across genomes).
- Resume semantics: `EvolutionLoop` checkpoints between generations, not within a generation. A crash mid-train-phase loses the genome's training but not the run; the next generation samples a fresh population.

### Decision 7: `feature_gating` evolution requires `feature_expansion` to be set non-`"none"`

The `MLPPPOBrainConfig.feature_gating: bool` field has a runtime cross-field constraint with `feature_expansion`: setting `feature_gating=True` while `feature_expansion="none"` raises `ValueError("feature_gating requires feature_expansion != 'none' (no features to gate)")` at brain construction.

For the M2 MLPPPO pilot, **`feature_gating` is NOT in the evolved schema** (the pilot evolves only continuous + integer hyperparameters). The cross-field constraint is therefore not exercised in this PR, but it's documented here for future pilot authors.

If a future pilot DOES evolve `feature_gating` as a bool, it must also either:

- Hold `feature_expansion` constant at a non-`"none"` value (set in the YAML's brain config, NOT evolved), OR
- Co-evolve `feature_expansion` as a categorical, paired with a schema-level constraint that the (`feature_gating`, `feature_expansion`) tuple never produces the `(True, "none")` combination.

The schema validator does NOT enforce cross-field brain-config constraints in v1 — that's encoder-time work and out of scope. v1 trusts the schema author to avoid invalid combinations.

### Decision 8: Pilot strategy — 4 seeds, MLPPPO, fitness-rising gate

Per the approved plan and the M-1 decision-gate framing:

- **4 seeds** (42, 43, 44, 45) — same as `nematode-run-experiments` skill convention
- **GO** if mean success rate across seeds ≥3pp over the hand-tuned MLPPPO baseline AND fitness still rising at gen 20 (i.e. the 20-gen budget didn't saturate)
- **PIVOT** if marginal: fitness rising but separation < 3pp, or separation > 3pp but flat by gen 20
- **STOP** if no separation: < 1pp or worse than baseline

The hand-tuned MLPPPO baseline is the existing `configs/scenarios/foraging/mlpppo_small_oracle.yml` running for the same eval-episode count under matched seeds.

The pilot is run during the spec-review window (after spec is approved, before/during implementation review) so the actual GO/PIVOT/STOP decision is in the PR body when the merge decision happens.

## Risks

1. **Schema authoring footgun** — typo in `name` field silently no-ops via `model_copy(update=...)`. Mitigated by Decision 2's load-time validator.

2. **Categorical convergence plateau** — CMA-ES sees flat fitness across categorical bins. Bounded blast radius for this PR (no categoricals in MLPPPO pilot); documented for PR 3 (LSTMPPO has `rnn_type`).

3. **Train-phase wall time at large K** — `learn_episodes_per_eval: 30` × 12 genomes × 4 seeds × 20 generations = 28,800 train episodes per pilot. At ~50 ms/episode (post-perf-fix MLPPPO), that's ~24 minutes wall time at parallel 1, or ~6 min at parallel 4. Acceptable. If the pilot decides to bump K, recompute.

4. **Pilot fitness landscape too flat over 20 generations** — pop 12 × n=7 evolved hyperparams may not separate from baseline. Mitigation: bounds tightened around hand-tuned values (so the pilot is searching neighbourhoods, not the universe); decision gate explicitly accepts "fitness still rising" as a PIVOT signal rather than requiring a converged win.

5. **Worker pickling of schemas** — `param_schema` travels in `Genome.birth_metadata` for every genome. Schema is small (≤8 entries) and uses Pydantic models that pickle cleanly. Mitigation: extend the existing pickle round-trip test to cover this.

6. **Train-phase stops mid-rollout** — `StandardEpisodeRunner.run()` calls `brain.learn()` per-step. If the train phase terminates early (max_steps, all food collected), the rollout buffer has partial data. This is the standard contract — same as how regular `run_simulation.py` handles short episodes — so no special handling. The eval phase starts with the brain in whatever state the train phase left it.

7. **Schema mutation across resume** — `EvolutionLoop._save_checkpoint` ([loop.py:195-209](packages/quantum-nematode/quantumnematode/evolution/loop.py#L195)) pickles the optimiser instance, which has `num_params` baked into its internal state (CMA-ES covariance matrix dim, GA population shape). On resume, the YAML is reloaded and `select_encoder(sim_config).genome_dim(sim_config)` returns `len(hyperparam_schema)`. If `hyperparam_schema` is modified between the original run and the resume (e.g., a new param added), the pickled optimiser samples vectors of the OLD length but `decode` walks the NEW schema — silently desynced. M2 inherits M0's checkpoint contract: `--resume` requires the YAML to be schema-compatible with the original run. The `nematode-run-evolution` skill's "operational pitfalls" note (added in this PR per Phase 10's docs scope) calls this out for users. Future PR could add a schema-hash check to the checkpoint payload (bumping `CHECKPOINT_VERSION`); out of M2 scope.

## Considered Alternatives — Optimiser choice for hyperparameter evolution

This PR uses CMA-ES (the M0-default optimiser) for hyperparameter evolution. That is a **pragmatic, not optimal** choice and worth flagging.

**Why CMA-ES is suboptimal for this use case:** the broader ML community standard for hyperparameter optimisation at the n=7–20 / few-hundred-evaluation budget is Bayesian Optimisation (BO) or Tree-structured Parzen Estimator (TPE) — Optuna's default and the one most widely used. Compared to CMA-ES at this scale:

- BO/TPE handles **categoricals natively** without the bin-plateau problem flagged in Decision 3.
- BO/TPE handles **log-scale floats natively** without manual encoding by the user.
- BO/TPE is generally **more sample-efficient at small budgets** (fewer evaluations to find good configs), which matters when each evaluation is K=30 train + L=5 eval episodes.
- BO/TPE handles **conditional parameters** more cleanly (e.g. "evolve `lstm_hidden_dim` only when `rnn_type=lstm`").

**Why we ship CMA-ES anyway in M2:**

- M0 already shipped CMA-ES as the default optimiser; reusing it means zero framework change for M2 beyond the new encoder + fitness.
- The M2 pilot is a **pilot** — its goal is to exercise the framework and get a GO/PIVOT/STOP signal, not to find the globally-optimal hyperparameters. CMA-ES is good enough for that.
- Adding an optimiser dependency (Optuna, Ax, etc.) is a real change with its own design questions — capability spec deltas, lockfile growth, dispatch surface. Bundling it into M2 widens the PR.

**The follow-up:** see Phase 5 Research Questions in [`openspec/changes/2026-04-26-phase5-tracking/tasks.md`](../2026-04-26-phase5-tracking/tasks.md). After M2 ships, we run a TPE/Optuna comparison on the same pilot config; if TPE substantially outperforms (heuristic: ≥5pp better fitness at the same evaluation budget), open a follow-up change to add an `OptunaOptimizer` adapter alongside CMA-ES + GA. The bar is concrete; the work is bounded; no commitment is made up front.

**Related future need:** M6.5 (NEAT topology evolution, currently flagged as optional) inherently requires a NEAT-specific optimiser, not CMA-ES or BO. So an "optimiser portfolio" exists in the Phase 5 future regardless. Recording the M2-driven question now feeds into that broader portfolio decision.

## Maintenance

- Adding a new param type (e.g. `set` for one-of-many) is a `Literal` extension to `ParamSchemaEntry.type` plus a new `case "<type>":` (or `elif entry.type == "<type>":`) branch INSIDE the single dispatch method `HyperparameterEncoder._decode_one` (per task 3.3a — single-method dispatch, not a method per type). The protocol is unchanged.
- Adding `LearnedPerformanceFitness` variants (e.g. with curriculum, with replay buffer reset between train and eval) follows the same fitness-class pattern. The shared `_build_agent` helper from M0 is reused.
- The `2026-04-27-add-hyperparameter-evolution` change archives in this PR, alongside implementation, per the M0 in-branch-archive pattern.

## Module dependency direction

`evolution.encoders.HyperparameterEncoder` reuses:

- `evolution.brain_factory.instantiate_brain_from_sim_config` (M0)
- `evolution.genome.Genome` (M0)

`evolution.fitness.LearnedPerformanceFitness` reuses:

- `evolution.fitness.FrozenEvalRunner` (M0)
- `evolution.fitness._build_agent` (M0)
- `agent.runners.StandardEpisodeRunner` (pre-M0)

No new module imports; no new dependencies to register.
