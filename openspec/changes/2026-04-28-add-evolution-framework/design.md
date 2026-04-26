## Overview

M0 is the brain-agnostic evolution framework that unblocks every subsequent Phase 5 milestone. The goal is a small, sharply scoped module that swaps in cleanly via a registry and reuses the project's existing optimisers and weight-persistence protocol. This design records why specific shape and scope choices were made.

## Goals / Non-Goals

**Goals:**

- Brain-agnostic evolution loop: any classical brain implementing `WeightPersistence` can be plugged in via a one-class encoder registration
- Genome encode → decode round-trip is deterministic — the same genome always produces the same first-step action on the same input
- Lineage CSV is the single source of truth for parent→child relationships across all of Phase 5
- Pickle resume works for runs that crash mid-campaign (50+ gen runs are real for M2/M3/M4)
- M2 and M3 can start without further framework PRs

**Non-Goals:**

- `LearnedPerformanceFitness` (learn-then-evaluate) — explicit M2 deliverable, not M0
- QVarCircuit (quantum) encoder — recorded Phase 5 decision; defer to Phase 6 if needed
- NEAT-style topology evolution — explicit M6.5 deliverable
- Co-evolution loop (two populations) — explicit M5 deliverable
- Inheritance strategies (Lamarckian, Baldwin) — explicit M3/M4 deliverables, not M0
- Replicating every CLI flag of the legacy script (e.g. `--init-params` JSON file injection); the new CLI provides the same core flag surface but skips niche legacy options unless they prove necessary

## Design Decisions

### Decision 0: Encoder API takes the full SimulationConfig

Constructing a fresh brain requires more than a `BrainConfig`. [`utils/brain_factory.setup_brain_model()`](packages/quantum-nematode/quantumnematode/utils/brain_factory.py#L51) needs `brain_type`, `brain_config`, `shots`, `qubits`, `device`, `learning_rate`, `gradient_method`, `gradient_max_norm`, and `parameter_initializer_config` — these live across multiple top-level fields of `SimulationConfig`.

The encoder protocol's methods therefore take **the full `SimulationConfig`** (not just a brain config):

```python
class GenomeEncoder(Protocol):
    brain_name: str
    def initial_genome(self, sim_config: SimulationConfig, *, rng: np.random.Generator) -> Genome: ...
    def decode(self, genome: Genome, sim_config: SimulationConfig) -> Brain: ...
    def genome_dim(self, sim_config: SimulationConfig) -> int: ...
```

Encoders delegate the actual brain construction to a thin helper at `evolution/brain_factory.py`:

```python
def instantiate_brain_from_sim_config(sim_config: SimulationConfig) -> Brain:
    """Single source of truth for fresh-brain construction in the evolution loop.
    Pulls all setup_brain_model arguments out of sim_config and dispatches."""
```

This wrapper exists so encoders don't each duplicate the 8-argument call to `setup_brain_model`. If `setup_brain_model`'s signature changes, the wrapper absorbs it.

**Note on `genome_dim()` cost:** `genome_dim(sim_config)` constructs a fresh brain to introspect its weight components — `MLPPPOBrain.input_dim` is computed at `__init__` time from `sensory_modules` ([mlpppo.py:177](packages/quantum-nematode/quantumnematode/brain/arch/mlpppo.py#L177)), so it can't be derived statically from `MLPPPOBrainConfig` alone. This makes `genome_dim()` a full PyTorch model construction, not a cheap lookup. The optimiser calls it once at startup to set `num_params`, so the cost is amortised. Implementations should NOT call `genome_dim()` per-generation or inside hot loops.

**Why pass `sim_config` rather than a narrower `EvolutionContext`:** `sim_config` is what the YAML config parses into and what gets pickled across worker processes anyway. Introducing a narrower context type adds a translation layer that isn't load-bearing.

**Why a dedicated `evolution/brain_factory.py` rather than calling `setup_brain_model` directly:** evolution-specific concerns — fixed `device=CPU` for fitness eval, ignoring quantum-only fields like `shots` for classical brains, defaulting `parameter_initializer_config` when not specified — live in one place. Future Phase 6 quantum encoder support will extend this wrapper, not every encoder.

### Decision 1: WeightPersistence with dynamic component discovery via denylist

`MLPPPOBrain` and `LSTMPPOBrain` already implement `WeightPersistence` (mlpppo.py:660, lstmppo.py:895). Their `get_weight_components()` returns a `dict[str, WeightComponent]` where each component bundles a name, a state dict, and metadata. Encoders use this protocol — they do **not** reach inside the brain to grab `actor.state_dict()` directly.

**The actual component sets** (verified during spec review):

- MLPPPO: `{"policy", "value", "optimizer", "training_state"}` plus a conditional `{"gate_weights"}` when `_feature_gating: true` is configured
- LSTMPPO: `{"lstm", "layer_norm", "policy", "value", "actor_optimizer", "critic_optimizer", "training_state"}` (note: there is **no** "actor" or "critic" component — the names are "policy" and "value" matching MLPPPO; "layer_norm" wraps a real LayerNorm tensor that MUST be in the genome)

**Discovery strategy — denylist, not allowlist:** encoders call `get_weight_components()` to retrieve all available components, then filter out a fixed denylist of non-genome state:

```python
NON_GENOME_COMPONENTS = {"optimizer", "actor_optimizer", "critic_optimizer", "training_state"}
genome_components = {k: v for k, v in all_components.items() if k not in NON_GENOME_COMPONENTS}
```

**Why dynamic discovery:**

- Picks up MLPPPO's conditional `gate_weights` automatically (when feature gating is enabled, those weights are real and must be in the genome)
- Picks up LSTMPPO's `layer_norm` without the encoder needing to know about it explicitly
- Survives future component additions to either brain without an encoder change
- The denylist is brain-architecture-agnostic — any future classical brain that follows the existing optimizer-naming conventions (`*_optimizer`, `training_state`) gets the right behaviour automatically

**Alternative considered:** explicit allowlist per encoder (e.g. `{"policy", "value"}` for MLPPPO). Rejected because it silently drops conditional components and creates encoder/brain coupling. A reviewer caught this gap in spec review — the original allowlist was wrong for both brains.

**Alternative considered:** flatten brain by walking `brain.actor.state_dict() | brain.critic.state_dict()` directly. Rejected because it duplicates knowledge that already lives in `get_weight_components()` and would silently drift if a brain adds a head.

### Decision 2: Reset `_episode_count = 0` and call `_update_learning_rate()` on decode

The exploration phase flagged that `MLPPPOBrain._episode_count` is part of `training_state` but is consulted by the LR scheduler at runtime. If a genome captured at episode 800 (with `_episode_count = 800`) is decoded into a fresh evolution run, the new brain inherits a stale episode count and the LR scheduler is in the wrong regime.

The encoder excludes `training_state` from the genome, but `load_weight_components` won't reset attributes the genome doesn't supply. So after `load_weight_components()`, the encoder explicitly:

1. Sets `brain._episode_count = 0`
2. Calls `brain._update_learning_rate()` to bring the LR into sync with the reset count

Without step 2, `_episode_count` is 0 but the LR is whatever the scheduler had computed at the previous `_update_learning_rate()` call (typically the last call from the source brain's lifetime). For M0's frozen-weight fitness this doesn't break anything — LR is unused without `.learn()` — but the contract is "decode produces a brain in the same state as a freshly constructed one", and a freshly constructed brain has both `_episode_count = 0` AND its initial LR. M2's `LearnedPerformanceFitness` depends on this contract.

LSTMPPOBrain has the same `_episode_count` + `_update_learning_rate()` pair, handled identically. The recurrent hidden state (`_pending_h_state`, `_pending_c_state`) resets at `prepare_episode()` per existing brain code, so no extra encoder handling.

**Why:** evolution evaluations need to start each genome at a known initial state. Without this reset, fitness would silently depend on which generation the genome was born in. This is the kind of bug that doesn't fail tests but invalidates a campaign.

### Decision 3: EpisodicSuccessRate is frozen — uses a `FrozenEvalRunner` that bypasses `agent.run_episode()`

For M0, fitness evaluates a brain initialised from the genome and run for K episodes **without** learning. The brain's weights stay fixed; we measure how good the genome's initial weights are.

**The implementation cannot just call `agent.run_episode()` directly** — spec review uncovered that [`StandardEpisodeRunner._terminate_episode`](packages/quantum-nematode/quantumnematode/agent/runners.py#L155) defaults `learn=True` and the success path ([runners.py:817-823](packages/quantum-nematode/quantumnematode/agent/runners.py#L817)) does not override it. Every successful episode calls `brain.learn()`, which for `MLPPPOBrain`/`LSTMPPOBrain` runs a real PPO update and mutates weights. That breaks the M0 frozen-weight contract.

**Solution: a `FrozenEvalRunner` in `evolution/fitness.py`** that mirrors `StandardEpisodeRunner.run()` but explicitly passes `learn=False, update_memory=False` on every termination path. Composition over copy-paste: it reuses the runner's helper methods (`_terminate_episode`, reward calc, sensory prep) by extending or wrapping `StandardEpisodeRunner`, not by re-implementing the per-step loop. ~60 LOC, well below the legacy script's ~80 LOC re-implementation, and the loop logic stays in one place (the standard runner).

**Sketch:**

```python
from quantumnematode.agent.agent import QuantumNematodeAgent
from quantumnematode.agent.runners import StandardEpisodeRunner
from quantumnematode.report.dtypes import TerminationReason
from quantumnematode.utils.config_loader import create_env_from_config


class FrozenEvalRunner(StandardEpisodeRunner):
    """Runs an episode without ever calling brain.learn() or update_memory().

    Overrides only `_terminate_episode` and forwards all other kwargs to the
    parent. This preserves the parent's `food_history=...` Ellipsis sentinel
    (which falls back to `agent.food_history` when omitted), avoiding silent
    data loss seen in earlier sketches that called `kwargs.get("food_history")`
    and converted the sentinel to None.
    """

    def _terminate_episode(self, agent, params, reward, **kwargs):
        # Force frozen behaviour regardless of caller's intent. All other
        # kwargs (success, termination_reason, food_history sentinel, etc.)
        # pass through unchanged.
        kwargs["learn"] = False
        kwargs["update_memory"] = False
        return super()._terminate_episode(agent, params, reward, **kwargs)


def _build_agent(brain, env, sim_config) -> QuantumNematodeAgent:
    """Centralise QuantumNematodeAgent construction for fitness eval."""
    sensing = sim_config.environment.sensing if sim_config.environment else None
    return QuantumNematodeAgent(
        brain=brain,
        env=env,
        satiety_config=sim_config.satiety,
        sensing_config=sensing,
    )


class EpisodicSuccessRate:
    def evaluate(self, genome, sim_config, encoder, *, episodes, seed) -> float:
        # Override sim_config.seed with the fitness function's seed BEFORE decode.
        # The brain's __init__ calls set_global_seed(self.seed) which seeds both
        # numpy and torch globals, AND get_rng(self.seed) which seeds brain.rng.
        # By overriding sim_config.seed we use the brain's existing seeding
        # infrastructure rather than fighting it (see Decision 3a).
        sim_config = sim_config.model_copy(update={"seed": seed})
        brain = encoder.decode(genome, sim_config)
        env = create_env_from_config(sim_config.environment, seed=seed)

        agent = _build_agent(brain, env, sim_config)
        runner = FrozenEvalRunner()
        successes = 0
        for _ in range(episodes):
            result = runner.run(agent, sim_config.reward, sim_config.max_steps)
            if result.termination_reason == TerminationReason.COMPLETED_ALL_FOOD:
                successes += 1
        return successes / episodes
```

**Note on success detection:** `EpisodeResult` ([runners.py:69](packages/quantum-nematode/quantumnematode/agent/runners.py#L69)) has fields `agent_path`, `termination_reason`, `food_history` — no `episode_success` attribute. The codebase convention is `result.termination_reason == TerminationReason.COMPLETED_ALL_FOOD` (used in [experiment/tracker.py:304](packages/quantum-nematode/quantumnematode/experiment/tracker.py#L304) and [multi_agent.py:949](packages/quantum-nematode/quantumnematode/agent/multi_agent.py#L949)). The fitness function uses the same convention.

**Note on agent constructor args:** `QuantumNematodeAgent.__init__` ([agent.py:245](packages/quantum-nematode/quantumnematode/agent/agent.py#L245)) takes 9 keyword args. The fitness function passes `brain`, `env`, `satiety_config=sim_config.satiety`, and `sensing_config=sim_config.environment.sensing` (or `None` if `sim_config.environment` is itself None). Other args use defaults — `theme`, `rich_style_config` are presentation-only; `agent_id` defaults to `"default"`; `maze_grid_size` and `max_body_length` are unused when an explicit `env` is provided. The private helper `_build_agent(brain, env, sim_config)` in `evolution/fitness.py` centralises this.

### Decision 3a: Seed application via `sim_config.model_copy(update={"seed": seed})`

The `evaluate()` signature includes `seed: int`. The fitness function applies it by **overriding `sim_config.seed` before calling `encoder.decode()`**, then forwarding the same `seed` to the env factory. Concretely:

```python
sim_config = sim_config.model_copy(update={"seed": seed})  # Pydantic v2 model_copy
brain = encoder.decode(genome, sim_config)                  # brain.__init__ now seeds globals to OUR seed
env = create_env_from_config(sim_config.environment, seed=seed)  # env RNG seeded directly
```

**Why this works:** `MLPPPOBrain.__init__` ([mlpppo.py:168-170](packages/quantum-nematode/quantumnematode/brain/arch/mlpppo.py#L168)) does:

```python
self.seed = ensure_seed(config.seed)
self.rng = get_rng(self.seed)         # seeds the brain's local numpy Generator
set_global_seed(self.seed)            # seeds np.random AND torch.manual_seed globals
```

`set_global_seed` ([seeding.py:65-90](packages/quantum-nematode/quantumnematode/utils/seeding.py#L65)) calls `np.random.seed(seed)` and `torch.manual_seed(seed)`. So by the time `decode()` returns, all three RNG sources (brain's local Generator, global numpy, global torch) are seeded with the fitness function's `seed`. We then seed the env RNG directly via `create_env_from_config(..., seed=seed)`.

**The bug an earlier sketch had:** doing `torch.manual_seed(seed)` *before* `decode()` was a no-op — the brain's constructor immediately re-seeded torch globally to `config.seed` (the YAML-configured seed, not the fitness function's `seed`). The fix is to override `config.seed` itself, so the brain's constructor uses the right value when it does its own seeding.

**Why use the brain's existing seeding rather than seeding ourselves:** the brain already seeds correctly for fresh construction. By overriding `sim_config.seed`, we re-use that infrastructure instead of fighting it — fewer manual `torch.manual_seed` calls, no order-of-operations traps, and any future addition to the brain's `__init__` seeding (e.g. CUDA RNG) propagates automatically.

**Why `model_copy` rather than mutating sim_config:** Pydantic v2 models are immutable by default. `model_copy(update={...})` is the v2 idiomatic way to produce a modified copy, used elsewhere in the codebase ([config_loader.py:818](packages/quantum-nematode/quantumnematode/utils/config_loader.py#L818)). Mutating in place would also mutate the caller's sim_config, which is undesirable across worker processes.

**Why the fitness function does this and not the encoder:** the encoder produces a brain from a genome — it shouldn't know about the seed of the *current evaluation*. Each call to `evaluate()` is its own seeded trial; the encoder is genome→brain regardless of which seed will be used for the trial.

Determinism is the contract the framework promises so that genome quality is meaningfully comparable across the population. M2's `LearnedPerformanceFitness` will use the same `model_copy` pattern.

**Why a runner subclass rather than a fork of the per-step loop:** the per-step loop in `StandardEpisodeRunner` is non-trivial (rewards, sensory prep, STAM updates, predator handling, termination conditions). Forking risks behavioural drift from the standard runner — exactly the maintenance debt that justified deleting the legacy script. Subclassing and overriding only `_terminate_episode` keeps the step semantics identical and makes the frozen guarantee visible at the type level.

**Why:** clean separation of concerns. `FrozenEvalRunner` is M0's primitive; M2's `LearnedPerformanceFitness` will use the standard runner with `learn=True` for K training episodes then a `FrozenEvalRunner` for L eval episodes — the same primitive composes both.

**Note on intra-evaluation state drift:** `post_process_episode()` is called by `_terminate_episode` ([runners.py:187](packages/quantum-nematode/quantumnematode/agent/runners.py#L187)) regardless of `learn`. It increments `_episode_count` and calls `_update_learning_rate()` between episodes within a single fitness evaluation. With `learn=False` enforced this does NOT affect reproducibility — the LR is unused without `.learn()`, action distribution depends only on weights and seeded RNG, and identical inputs produce identical fitness. The brain is weight-stateless across episodes but not counter-stateless.

**Alternative considered:** call `agent.run_episode()` directly and accept that successful episodes update weights. Rejected: it makes M0 fitness "minimal-learning fitness, learning happens only on successes" which collapses the M0 vs M2 distinction and surprises users (the same genome scores higher on episode 2 than episode 1 if it succeeded).

**Alternative considered:** add a `learn: bool` parameter to the public `agent.run_episode()` method. Rejected: bigger blast radius (changes a method other code depends on); subclassing achieves the same without touching the agent API.

**Alternative considered:** ship both fitness modes (frozen + learn-then-eval) in M0. Rejected: M0 grows by ~150 LOC and we make M2 design decisions prematurely.

### Decision 4: Pickle resume preserved in M0

The legacy script has mature pickle-based optimiser checkpoint/resume (run_evolution.py:642-660). CMA-ES covariance matrices are non-trivial to reconstruct; pickle handles them correctly. Long-running M3/M4 campaigns (50+ generations) genuinely need resume — a crash at gen 30 would otherwise burn the entire run.

**Why preserve in M0:** the pattern is brain-agnostic (it pickles the optimiser, not the brain). Porting it now is ~50 LOC and avoids re-implementing it later under deadline pressure. Resume tests are part of the M0 test plan to catch regressions early.

**What we add:** the checkpoint also stores the RNG state and lineage CSV path so resume reconstructs the full evaluation context, not just the optimiser.

### Decision 5: Lineage CSV columns and generation indexing

Columns: `generation, child_id, parent_ids, fitness, brain_type`. `parent_ids` is a `;`-joined string (CSV-friendly). `brain_type` is included so that future co-evolution runs (M5) — where predator and prey populations are interleaved in the same evolution_results directory — can be sliced by species.

**Generation indexing is 0-based.** A run with `generations: G` populates rows for generations `0, 1, ..., G-1` (inclusive lower, exclusive upper — Python's range convention). Total row count is `P × G` where `P` is `population_size`. Generation 0 rows have empty `parent_ids` (no prior generation to attribute to).

**Why CSV not JSON Lines:** existing experiment tracking uses CSV (artifacts/logbooks/011/). Tooling and human inspection are CSV-native. Append mode plays nicely with resume: the file just keeps growing.

### Decision 5a: parent_ids convention — every member of generation N-1 is a parent of every member of generation N

Neither `CMAESOptimizer.ask()` nor `GeneticAlgorithmOptimizer.ask()` exposes parent→child provenance to the loop. CMA-ES samples a population from a Gaussian whose mean and covariance are updated by all of generation N-1's fitnesses — there are no discrete parents. GA does internal tournament selection but the optimiser interface returns only the candidate solutions, not the parents that produced them.

**Convention:** for both algorithms, the lineage tracker records **every member of generation N-1 as a parent of every member of generation N**. So a row for a gen-N candidate has `parent_ids` listing all P genome IDs from gen N-1, `;`-joined.

**Why this convention:**

- For CMA-ES, this is semantically accurate: every previous candidate contributed (weighted by fitness rank) to the distribution that sampled the new generation.
- For GA, it's a slight over-approximation (a child only has 2 actual parents via crossover), but the optimiser doesn't expose those, and the over-approximation is harmless — downstream lineage analysis can still reconstruct fitness gradients across generations. If GA's true parent provenance becomes load-bearing for an analysis, we'd modify `GeneticAlgorithmOptimizer.ask()` to return parent indices alongside candidates — out of M0 scope.
- For generation 0, `parent_ids` is empty (no prior generation).

**Cost:** `parent_ids` strings get long. With population P=20, each gen-N row has a 20-UUID-long `parent_ids` field — about 720 chars (UUID is 36 chars + separator). For typical campaigns (50 gens × 20 pop = 1000 rows), the lineage CSV is ~750 KB. Acceptable.

**Why not omit `parent_ids` for CMA-ES:** keeping the convention uniform across algorithms means downstream tooling (lineage visualisations, fitness gradient analysis) doesn't need an algorithm-specific code path. The tracker writes the same CSV shape regardless of optimiser type.

### Decision 6: Encoder registry is a static dict, not a plugin discovery system

`ENCODER_REGISTRY: dict[str, type[GenomeEncoder]] = {"mlpppo": MLPPPOEncoder, "lstmppo": LSTMPPOEncoder}`. New encoders are added by editing this dict.

**Why:** matches the existing `BRAIN_CONFIG_MAP` pattern (config_loader.py:128-148). Discovery via entry points or decorators is over-engineering at this stage. A future Phase 6 quantum re-evaluation that adds `QVarCircuitEncoder` is a one-line registry edit.

### Decision 7: Two pilot configs in M0, not just one

Both `mlpppo_foraging_small.yml` and `lstmppo_foraging_small_klinotaxis.yml` ship in M0. The MLPPPO config is the cheap framework smoke; the LSTMPPO+klinotaxis config is the first-class biological target that M3/M4/M5/M6 will all build on.

**Why both now:** if only the MLPPPO config ships, M3 has to add the LSTMPPO+klinotaxis config plus prove the LSTMPPO encoder works at the same time, mixing framework concerns with science concerns. Shipping both in M0 means M3's PR is purely about Lamarckian inheritance, not "does the framework even work for LSTMPPO?"

### Decision 8: Delete the legacy script entirely

**The user's stated rationale (review feedback):** retaining legacy code under `scripts/legacy/` would tie M0 to suboptimal implementation choices — there'd be a temptation to mirror the old script's structure, copy its CLI flags verbatim, or maintain bug-compatibility "just in case." Deleting it cleanly signals that the new framework is free to make better choices.

**What we lose:** the existing CI smoke test (`test_run_evolution_smoke`) that exercised the QVarCircuit path. Replaced by a new MLPPPO smoke against the new framework — different brain, but the assertion ("CLI doesn't crash with minimal parameters") is the same.

**What we keep:** git history. `git log -- scripts/run_evolution.py` and `git show <commit>:scripts/run_evolution.py` retrieve the old code if anyone needs it for comparison. The `configs/evolution/qvarcircuit_foraging_small.yml` config is also deleted — no consumer remains.

**Future quantum brain support:** if a Phase 6 quantum re-evaluation needs evolution, a `QVarCircuitEncoder` is added cleanly to the new framework's registry (a one-class change). It will not resurrect the legacy script.

## Risks

1. **Encoder round-trip determinism.** Round-trip tests fix `torch.manual_seed(0)` before encode and again before decode, then assert that both brains produce identical actions on the same seeded input. CI runs CPU-only so cuDNN/CUDA non-determinism doesn't apply here. If a brain ever introduces eval-time stochasticity (e.g. dropout left on outside training mode), the test catches it.

2. **Pickle resume couples optimiser internals to a Python/library version.** Two distinct concerns:

   - **Schema changes we make** (e.g. add a new key to the checkpoint dict): mitigated by `checkpoint_version: int`, validated on load, with a clear error on mismatch.
   - **Python minor version drift or `cma` library updates**: pickle is best-effort across these. We don't try to handle this — recommendation is "single Python version per evolution campaign". Pin via `pyproject.toml` already (`python = ">=3.12,<3.13"`).

3. **`evolution:` block schema diverges from CLI flag defaults.** Easy to introduce drift where YAML default disagrees with CLI default. Mitigation: CLI flags default to `None` and only override the YAML value when explicitly passed. Single source of truth lives in `EvolutionConfig` Pydantic defaults.

4. **Smoke runs are slow if `episodes_per_eval` is too high.** A 10-gen × 8-pop × 15-eps smoke is 1200 episode-runs and could be 5+ minutes. Mitigation: pilot configs use `episodes_per_eval: 3` (the minimum that gives a meaningful fitness signal) so the smoke is sub-2-minute.

5. **Anyone with muscle memory of the legacy script will be surprised.** The legacy script is deleted (not preserved under `scripts/legacy/`). Mitigation: the new script logs `Brain type: <name>` prominently on startup; if someone runs it with the old QVarCircuit config, the error message names the registered brains and the breakage is immediate and obvious. Git history retrieves the old script if absolutely needed.

## Maintenance

- Every new encoder is one entry in `ENCODER_REGISTRY` plus one round-trip test
- Adding `LearnedPerformanceFitness` (M2) is a new class in `fitness.py`; the encoder protocol does not change
- Inheritance strategies (M3 Lamarckian, M4 Baldwin) live in a future `evolution/inheritance.py`; they consume the encoder + fitness protocols without modifying them
- The `2026-04-28-add-evolution-framework` change archives on merge (unlike the M-1 tracking change which stays open until M7)

## Module dependency direction

`evolution/` depends on `optimizers/` (uses `CMAESOptimizer`, `GeneticAlgorithmOptimizer`, `EvolutionResult`), `brain/` (uses `Brain`, `WeightPersistence`, concrete brain classes), `agent/` (uses `QuantumNematodeAgent`), `env/` (uses `create_env_from_config`), and `utils/` (uses `SimulationConfig`, `setup_brain_model`).

**`optimizers/` MUST NOT import from `evolution/`** — one-way dependency. The optimisers are general-purpose (CMA-ES on a sphere function works without any evolution framework); they pre-date this module and remain reusable independently. Any temptation to add evolution-specific helpers to `optimizers/evolutionary.py` is a smell — put them in `evolution/`.

## Forward-looking notes (out of M0 scope)

Recorded here so future Phase 5 milestones don't re-discover these.

### RolloutBuffer captures `brain.rng` by reference (M2 concern)

`MLPPPOBrain` and `LSTMPPOBrain` both construct a `RolloutBuffer` that captures `self.rng` by reference at construction time ([mlpppo.py:246](packages/quantum-nematode/quantumnematode/brain/arch/mlpppo.py#L246), [lstmppo.py:430](packages/quantum-nematode/quantumnematode/brain/arch/lstmppo.py#L430)).

For M0's frozen-weight fitness this is irrelevant — the buffer is consumed by `.learn()` which we never call. For M2's `LearnedPerformanceFitness`, fitness evaluations that include training will need to either:

- reach into `brain.buffer.rng` and reseed it too, or
- add a `brain.reseed(seed)` method that handles both `self.rng` and `self.buffer.rng`

Note: M0's seeding pattern (`sim_config.model_copy(update={"seed": seed})` before `encoder.decode()`) sidesteps this entirely because the brain's constructor creates a fresh `RolloutBuffer` with the new RNG. M2 only hits this if it tries to reseed a *post-construction* brain.
