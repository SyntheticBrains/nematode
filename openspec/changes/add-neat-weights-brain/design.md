# Design: Add NEAT-Weights Brain (`FeedforwardGABrain`)

## Context

The repository ships the `Brain` Protocol at [`brain/arch/_brain.py:411-434`](../../../packages/quantum-nematode/quantumnematode/brain/arch/_brain.py#L411-L434) (the contract every brain conforms to: `run_brain` / `update_memory` / `prepare_episode` / `post_process_episode` / `copy` + `history_data` / `latest_data`) and the decorator-registration plugin pattern at [`brain/arch/_registry.py`](../../../packages/quantum-nematode/quantumnematode/brain/arch/_registry.py) (the post-T2 refactor from `add-architecture-plugin-interface`). Adding a new brain is now a small mechanical task per [docs/architecture/plugin-developer-guide.md](../../../docs/architecture/plugin-developer-guide.md): new module + `@register_brain` decorator + `BrainType` enum entry + import from `__init__.py` + smoke config + tests.

The repository also ships a complete evolutionary-optimisation stack independently of the brain layer:

- [`optimizers/evolutionary.py:270-429`](../../../packages/quantum-nematode/quantumnematode/optimizers/evolutionary.py#L270-L429) provides `GeneticAlgorithmOptimizer` (mu+lambda + elite selection at [evolutionary.py:378](../../../packages/quantum-nematode/quantumnematode/optimizers/evolutionary.py#L378), tournament selection at [evolutionary.py:412](../../../packages/quantum-nematode/quantumnematode/optimizers/evolutionary.py#L412) with tournament size hard-coded to 3, Gaussian mutation, **uniform** crossover at [evolutionary.py:431](../../../packages/quantum-nematode/quantumnematode/optimizers/evolutionary.py#L431)). The `__init__` accepts `num_params`, `x0`, `population_size`, `sigma0` (initial step size + mutation std), `elite_fraction`, `mutation_rate`, `crossover_rate`, `seed`.
- [`evolution/loop.py`](../../../packages/quantum-nematode/quantumnematode/evolution/loop.py) drives the per-generation loop.
- [`evolution/genome.py:22`](../../../packages/quantum-nematode/quantumnematode/evolution/genome.py#L22) defines `Genome` (params, genome_id, parent_ids, generation, birth_metadata).
- [`evolution/encoders.py:370`](../../../packages/quantum-nematode/quantumnematode/evolution/encoders.py#L370) registers per-brain `GenomeEncoder` instances in `ENCODER_REGISTRY`. Currently only `MLPPPOEncoder` + `LSTMPPOEncoder` are registered. Both subclass `_ClassicalPPOEncoder` ([encoders.py:246-342](../../../packages/quantum-nematode/quantumnematode/evolution/encoders.py#L246-L342)), which (1) reads the brain's weights via the `WeightPersistence` protocol, (2) flattens them into a `Genome.params` numpy array, (3) on decode, reconstructs the brain via `instantiate_brain_from_sim_config()` + loads weights, AND (4) resets `brain._episode_count = 0` then calls `brain._update_learning_rate()` to bring the LR scheduler into sync. The `_episode_count` + `_update_learning_rate()` reach is PPO-specific; a GA brain has no LR scheduler.
- [`evolution/fitness.py:80-128`](../../../packages/quantum-nematode/quantumnematode/evolution/fitness.py#L80-L128) provides `FrozenEvalRunner` which **monkey-patches `agent.brain.learn` and `agent.brain.update_memory` to no-ops** for the duration of each evaluation episode. Fitness is computed externally by `EpisodicSuccessRate` ([fitness.py:211](../../../packages/quantum-nematode/quantumnematode/evolution/fitness.py#L211)) from episode-termination outcomes (not pulled from the brain). This is load-bearing for Decision 3 below.
- [`brain/weights.py:55`](../../../packages/quantum-nematode/quantumnematode/brain/weights.py#L55) defines the `WeightPersistence` protocol: `get_weight_components()` + `load_weight_components()`. Every brain that participates in evolution implements it.
- [`scripts/run_evolution.py`](../../../scripts/run_evolution.py) is the launcher (covered by the `nematode-run-evolution` skill). Always runs headless — there is no `--theme` flag.
- [`MLPPPOBrain`](../../../packages/quantum-nematode/quantumnematode/brain/arch/mlpppo.py) is the matched-capacity reference: 2-layer feed-forward, `actor_hidden_dim=64`, `critic_hidden_dim=64`, separate actor + critic heads.
- [`configs/evolution/mlpppo_foraging_small.yml`](../../../configs/evolution/mlpppo_foraging_small.yml) is the canonical evolution-config shape: `brain:` block (with brain-class hyperparameters only) + `reward:` + `environment:` + a required `evolution:` block (`algorithm`, `generations`, `population_size`, `episodes_per_eval`, `parallel_workers`, `checkpoint_every`).

The upcoming `weight-search-architecture-ranking` change needs four MUST architecture families per `phase6-tracking/design.md` Decision 4: connectome, MLP-PPO, LSTM/GRU-PPO, NEAT-evolved-weights. The first three exist; NEAT-weights does not. NEAT topology evolution lands at Tranche 8 via TensorNEAT — this change introduces only the weight-search variant on a fixed feed-forward topology.

Stakeholder: the upcoming `weight-search-architecture-ranking` change consumes `FeedforwardGABrain` as its GA-based weight-search comparator.

## Goals / Non-Goals

**Goals:**

- Ship `FeedforwardGABrain` as a Brain Protocol + WeightPersistence Protocol implementation registered through the existing decorator-registration pattern.
- Ship `FeedforwardGAEncoder` registered in `ENCODER_REGISTRY` so the evolution loop can encode + decode the brain's weights as a `Genome`.
- Match `MLPPPOBrain` small capacity (hidden width 64, 2 hidden layers, actor head only — critic is PPO-specific and irrelevant for GA fitness selection) so the cross-architecture comparison isolates the optimiser-family change (PPO gradient ascent vs GA population search) from a capacity confound.
- Land one smoke evolution-config that proves end-to-end weight evolution: short generation budget × small population on klinotaxis foraging, headless-runnable, completes without exception, fitness moves across generations.
- Establish protocol-conformance + encoder round-trip tests so the brain can be consumed by `weight-search-architecture-ranking` with confidence.

**Non-Goals:**

- NEAT topology evolution. Tranche 8 ships genuine topology search via TensorNEAT; this change is fixed-topology only.
- GA variants of other brain families (LSTM-GA, connectome-GA). Out of scope; consider only if `weight-search-architecture-ranking` surfaces evidence GA is useful for them.
- A new GA optimiser implementation. Reuse the existing `GeneticAlgorithmOptimizer` without modification.
- Extending `GeneticAlgorithmOptimizer` to expose tournament size as a hyperparameter. Tournament size is hard-coded to 3 inside `_tournament_select`; if the smoke or downstream comparison needs tunable tournament size, that's a separate optimiser-framework change.
- Modifying the L1 registry, evolution-loop API, or any other downstream interface. This change is purely additive (one new encoder entry in `ENCODER_REGISTRY` is additive — no existing entries change).
- Continuous-action heads. Discrete `DEFAULT_ACTIONS` (4 actions) is the action API for both Tranche 4 and this change.

## Decisions

### Decision 1 — Fixed feed-forward topology matching MLPPPO small

The brain ships with a fixed 2-hidden-layer feed-forward network: `input_dim → 64 → 64 → 4` (4 actions). No critic head, no recurrence. Matched capacity to `MLPPPOBrain` small ensures the cross-architecture comparison in `weight-search-architecture-ranking` isolates the optimiser-family change from network-capacity differences.

**Why not arbitrary topology?** Real NEAT evolves topology + weights; this brain evolves weights only. Making the topology configurable per-instance (e.g. `hidden_dims: [128, 128]`) is a generalisation we don't yet need — the comparison only requires the matched-capacity variant. Configurable topology becomes a follow-up if a real ablation needs it.

**Alternative considered: reuse `MLPPPOBrain`'s actor network class and bolt GA onto it.** Rejected because `MLPPPOBrain` is fused (topology + PPO learning rule) per the documented limitation in `add-architecture-plugin-interface`'s design.md. Cleanly factoring topology out from learning rule across the 19 legacy brains is deferred to a separate change; for one new brain it's cheaper to write the topology inline and reuse the GA optimiser.

### Decision 2 — Use existing `GeneticAlgorithmOptimizer` without modification

The brain ships with `WeightPersistence` Protocol conformance + a `FeedforwardGAEncoder` registered in `ENCODER_REGISTRY`. The `EvolutionLoop` orchestrates per-generation `ask` → evaluate fitness → `tell` against the optimiser; no per-brain coupling exists.

**Why not introduce a brain-specific GA?** Code duplication risk + maintenance cost. The existing optimiser is feature-complete for fixed-topology weight search: mu+lambda selection at [evolutionary.py:378](../../../packages/quantum-nematode/quantumnematode/optimizers/evolutionary.py#L378), tournament selection with hard-coded `tournament_size=3` at [evolutionary.py:412-417](../../../packages/quantum-nematode/quantumnematode/optimizers/evolutionary.py#L412-L417), Gaussian mutation, **uniform** crossover at [evolutionary.py:431-434](../../../packages/quantum-nematode/quantumnematode/optimizers/evolutionary.py#L431-L434) (per-gene Bernoulli mask), seeded numpy RNG.

**Why not introduce `neat-python` as a dependency?** The library would add ~5 MB + introduce a parallel evolution loop, parallel genome encoding, parallel fitness machinery — all duplicating what the repo already ships. The "NEAT" naming in `phase6-tracking/design.md` Decision 4 refers to the architecture-comparison row, not to a specific library; this change satisfies the row with the existing infrastructure.

### Decision 3 — GA fitness flows through `FrozenEvalRunner`, not through brain hooks

Per [`evolution/fitness.py:80-128`](../../../packages/quantum-nematode/quantumnematode/evolution/fitness.py#L80-L128), `FrozenEvalRunner` temporarily replaces `agent.brain.learn` and `agent.brain.update_memory` with no-ops for each evaluation episode. Fitness is computed externally by `EpisodicSuccessRate` ([fitness.py:211](../../../packages/quantum-nematode/quantumnematode/evolution/fitness.py#L211)) from the episode's termination outcome. The brain does NOT maintain a fitness accumulator and does NOT report a fitness score from `post_process_episode()`.

**Implication for the brain implementation:** `FeedforwardGABrain.update_memory()` and `FeedforwardGABrain.post_process_episode()` MUST be no-op-safe (no gradient computation, no parameter mutation, no error if called). They do NOT accumulate reward. `run_brain()` is the only method that needs real implementation: env features → 2 hidden layers (ReLU) → action logits → softmax sampling.

**Why?** Trying to fight `FrozenEvalRunner`'s monkey-patching would require either modifying the evolution framework (out of scope) or implementing a parallel fitness mechanism (duplication). The framework's "brain doesn't know it's being evaluated for GA" pattern is the documented contract — `MLPPPOBrain` + `LSTMPPOBrain` both work this way and are the existing examples of GA-evolved-via-encoder brains.

### Decision 4 — Encoder subclasses `_ClassicalPPOEncoder` with PPO-attribute shims

`FeedforwardGAEncoder` subclasses [`_ClassicalPPOEncoder`](../../../packages/quantum-nematode/quantumnematode/evolution/encoders.py#L246) and pins `brain_name = "feedforwardga"`. `_ClassicalPPOEncoder.decode()` at [encoders.py:305-306](../../../packages/quantum-nematode/quantumnematode/evolution/encoders.py#L305-L306) accesses `brain._episode_count` and `brain._update_learning_rate()` — both PPO-specific. The `FeedforwardGABrain` MUST provide these as shims:

```python
self._episode_count: int = 0  # written by encoder after decode; not used by GA brain
def _update_learning_rate(self) -> None:  # noqa: SLF001 - encoder contract
    """No-op: GA brain has no LR scheduler. Encoder calls this after decode."""
```

**Why subclass rather than write a fresh encoder base?** The encoder's weight-flatten + weight-restore mechanics work as-is for any `WeightPersistence`-implementing brain; the only friction is the PPO-attribute reach in `decode()`. Two shim attributes are cheaper than duplicating `_ClassicalPPOEncoder`'s ~80 lines. If a future change refactors `_ClassicalPPOEncoder` to remove the PPO-specific reach, the shims become inert and can be deleted.

**Alternative considered: write `_GAEncoder` as a fresh base.** Rejected because it duplicates the flatten/unflatten + `WeightPersistence` plumbing; the shim approach is documented and the friction is small.

### Decision 5 — Smoke config lives in `configs/evolution/`, not `configs/scenarios/`

The smoke config is `configs/evolution/feedforwardga_foraging_small.yml` mirroring [`configs/evolution/mlpppo_foraging_small.yml`](../../../configs/evolution/mlpppo_foraging_small.yml). Required structure: `brain:` (just brain-class hyperparameters), `reward:`, `satiety:`, `environment:`, `evolution:` (with `algorithm: ga`, `generations`, `population_size`, `episodes_per_eval`, `parallel_workers`, `checkpoint_every`).

**Why not `configs/scenarios/foraging/feedforwardga_small_klinotaxis.yml`?** Configs under `configs/scenarios/` are consumed by `scripts/run_simulation.py` and have no `evolution:` block. Evolution configs are a distinct family. Mixing the two would break both launchers.

**Why klinotaxis sensing in the smoke?** Phase 6 Decision 5 fixes the behaviours to klinotaxis/thermotaxis/predator-evasion; klinotaxis is the simplest. Forging-only config (no predators, no thermal) is sufficient for the brain smoke; downstream `weight-search-architecture-ranking` configs add the other behaviours.

## Risks / Trade-offs

| Risk | Mitigation |
|---|---|
| GA on a ~6,400-weight network has more parameters than evolution-framework configs typically handle well; convergence may be slow vs PPO at matched episode budget | Document the comparison framing explicitly in `weight-search-architecture-ranking`: the GA cell measures the optimiser-family change at matched capacity, not "GA is competitive in absolute terms." Slow GA convergence vs PPO is itself a finding worth recording. |
| Smoke config uses small population × few generations; may not catch GA hyperparameter sensitivity that surfaces at the full `weight-search-architecture-ranking` budget | Smoke is for end-to-end pipeline verification (no crashes, fitness moves across generations), not for hyperparameter validation. Real hyperparameter tuning lands in `weight-search-architecture-ranking`'s Phase 2 compute pre-flight. |
| Reusing the existing `GeneticAlgorithmOptimizer` means inheriting its hard-coded tournament size (3) and any defaults; if downstream tuning needs different tournament size, the optimiser must be extended | Document the constraint in the smoke config + design.md. If a downstream comparison genuinely needs tunable tournament size, that's a follow-up framework change, not part of this scope. |
| The brain's "no critic head" choice diverges from MLPPPO's structure (which has both actor + critic); strict capacity-matching might require including a critic-equivalent | The cross-architecture comparison measures policy quality, not value-estimate quality. The critic is a PPO-specific auxiliary; GA fitness is the episode return directly. Matching actor-side capacity is the load-bearing parity claim; matching critic-side capacity would actively confound the comparison by giving GA a free auxiliary network it doesn't use. |
| `_ClassicalPPOEncoder.decode()` reaches for `_episode_count` and `_update_learning_rate()`; the shim approach (Decision 4) is correct today but fragile if `_ClassicalPPOEncoder` changes its decode contract | Document the shim's reason inline. If the encoder's PPO-attribute reach is removed in a future refactor, the shims become inert (no behaviour change) and can be deleted. |

## Migration Plan

Fully additive. No existing brain, config, test, or downstream consumer is modified.

Rollback (if the brain causes issues during `weight-search-architecture-ranking` Phase 1a verification): revert the change. The L1 registry's decorator-registration pattern ensures absence of `FeedforwardGABrain` doesn't break any other code path — the `feedforwardga` config name simply becomes an unknown brain name at config-load time, which raises a clear error.

## Open Questions

None at design-finalisation time. The brain is small, self-contained, and reuses existing infrastructure. Hyperparameter choices for the smoke config are scoped to "does it train end-to-end?" not "does it train well" — the latter belongs in the consumer's Phase 2 compute pre-flight.
