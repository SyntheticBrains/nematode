# Design: Add NEAT-Weights Brain (`FeedforwardGABrain`)

## Context

The repository ships the `Brain` Protocol at [brain/arch/\_brain.py:346-368](../../../packages/quantum-nematode/quantumnematode/brain/arch/_brain.py) (the contract every brain conforms to: `run_brain` / `update_memory` / `prepare_episode` / `post_process_episode` / `copy` + `history_data` / `latest_data`) and the decorator-registration plugin pattern at [brain/arch/\_registry.py](../../../packages/quantum-nematode/quantumnematode/brain/arch/_registry.py) (the post-T2 refactor from `add-architecture-plugin-interface`). Adding a new brain is now a 5-file mechanical task per [docs/architecture/plugin-developer-guide.md](../../../docs/architecture/plugin-developer-guide.md): new module + `@register_brain` decorator + config class + brain_factory loader branch + smoke config.

The repository also ships a complete evolutionary-optimisation stack independently of the brain layer:

- [`optimizers/evolutionary.py`](../../../packages/quantum-nematode/quantumnematode/optimizers/evolutionary.py) provides `GeneticAlgorithmOptimizer` (mu+lambda selection, tournament selection, Gaussian mutation, two-point crossover) alongside CMA-ES and Optuna TPE.
- [`evolution/loop.py`](../../../packages/quantum-nematode/quantumnematode/evolution/loop.py), [`evolution/genome.py`](../../../packages/quantum-nematode/quantumnematode/evolution/genome.py), [`evolution/encoders.py`](../../../packages/quantum-nematode/quantumnematode/evolution/encoders.py), [`evolution/fitness.py`](../../../packages/quantum-nematode/quantumnematode/evolution/fitness.py) provide the surrounding training loop, parameter encoding, and fitness evaluation.
- [`scripts/run_evolution.py`](../../../scripts/run_evolution.py) is the launcher (covered by the `nematode-run-evolution` skill).
- [`MLPPPOBrain`](../../../packages/quantum-nematode/quantumnematode/brain/arch/mlpppo.py) is the matched-capacity reference: 2-layer feed-forward, `actor_hidden_dim=64`, `critic_hidden_dim=64`, separate actor + critic heads.

The upcoming `weight-search-architecture-ranking` change needs four MUST architecture families per `phase6-tracking/design.md` Decision 4: connectome, MLP-PPO, LSTM/GRU-PPO, NEAT-evolved-weights. The first three exist; NEAT-weights does not. NEAT topology evolution lands at Tranche 8 via TensorNEAT — this change introduces only the weight-search variant on a fixed feed-forward topology.

Stakeholder: the upcoming `weight-search-architecture-ranking` change consumes `FeedforwardGABrain` as its GA-based weight-search comparator (one C3 cell + curriculum smokes per the parent plan).

## Goals / Non-Goals

**Goals:**

- Ship `FeedforwardGABrain` as a Brain Protocol implementation registered through the existing decorator-registration pattern, with weights evolved by the existing `GeneticAlgorithmOptimizer`.
- Match `MLPPPOBrain` small capacity (hidden width 64, 2 hidden layers, actor head only — critic is PPO-specific and irrelevant for GA fitness selection) so the cross-architecture comparison isolates the optimiser-family change (PPO gradient ascent vs GA population search) from a capacity confound.
- Land one smoke config that proves end-to-end evolution: 10 generations × small population on klinotaxis foraging, headless-runnable, shows non-degenerate weight diversity in the population and a non-degraded best-fitness trajectory.
- Establish protocol-conformance + GA-determinism tests so the brain can be consumed by `weight-search-architecture-ranking` with confidence.

**Non-Goals:**

- NEAT topology evolution. Tranche 8 ships genuine topology search via TensorNEAT; this change is fixed-topology only.
- GA variants of other brain families (LSTM-GA, connectome-GA). Out of scope; consider only if `weight-search-architecture-ranking` surfaces evidence GA is useful for them.
- A new GA optimiser implementation. Reuse the existing `GeneticAlgorithmOptimizer` without modification.
- Modifying the L1 registry, evolution-loop API, or any other downstream interface. This change is purely additive.
- Continuous-action heads. Discrete `DEFAULT_ACTIONS` (4 actions) is the action API for both Tranche 4 and this change.

## Decisions

### Decision 1 — Fixed feed-forward topology matching MLPPPO small

The brain ships with a fixed 2-hidden-layer feed-forward network: `input_dim → 64 → 64 → 4` (4 actions). No critic head, no recurrence. Matched capacity to `MLPPPOBrain` small ensures the cross-architecture comparison in `weight-search-architecture-ranking` isolates the optimiser-family change from network-capacity differences.

**Why not arbitrary topology?** Real NEAT evolves topology + weights; this brain evolves weights only. Making the topology configurable per-instance (e.g. `hidden_dims: [128, 128]`) is a generalisation we don't yet need — the comparison only requires the matched-capacity variant. Configurable topology becomes a follow-up if a real ablation needs it.

**Alternative considered: reuse `MLPPPOBrain`'s actor network class and bolt GA onto it.** Rejected because `MLPPPOBrain` is fused (topology + PPO learning rule) per the documented limitation in `add-architecture-plugin-interface`'s design.md. Cleanly factoring topology out from learning rule across the 19 legacy brains is deferred to a separate change; for one new brain it's cheaper to write the topology inline and reuse the GA optimiser.

### Decision 2 — Use existing `GeneticAlgorithmOptimizer` without modification

The brain's `update_memory` / `post_process_episode` hooks accumulate per-episode fitness signal (return + survival + foods collected), and the brain integrates with the existing `EvolutionLoop` via the same `BrainGenome` encoder pattern used by other GA-trained configurations (see [`evolution/encoders.py`](../../../packages/quantum-nematode/quantumnematode/evolution/encoders.py)).

**Why not introduce a brain-specific GA?** Code duplication risk + maintenance cost. The existing optimiser is feature-complete (mu+lambda, tournament selection, Gaussian mutation, two-point crossover, configurable population size, seedable RNG).

**Why not introduce `neat-python` as a dependency?** The library would add ~5 MB + introduce a parallel evolution loop, parallel genome encoding, parallel fitness machinery — all duplicating what the repo already ships. The "NEAT" naming in `phase6-tracking/design.md` Decision 4 refers to the architecture-comparison row, not to a specific library; this change satisfies the row with the existing infrastructure.

### Decision 3 — Brain runs as a single forward pass per step; no episode-internal learning

`FeedforwardGABrain.run_brain()` is a single forward pass: env features → 2 hidden layers (ReLU) → action logits → softmax sampling. `update_memory()` accumulates step-level reward into the brain's per-episode fitness accumulator. `post_process_episode()` finalises the episode's fitness score for the GA's selection step. No within-episode parameter updates (which is the point — GA evolves between episodes, PPO updates within).

**Why?** Aligns with how the existing evolution-framework consumes brains: episodes are fitness samples; the optimiser decides which weights survive to the next generation. PPO updates within an episode would conflate the two learning signals.

## Risks / Trade-offs

| Risk | Mitigation |
|---|---|
| GA on a 2-layer 64-wide network has more parameters (~6,400 weights) than evolution-framework configs typically handle well; convergence may be slow vs PPO at matched episode budget | Document the comparison framing explicitly in `weight-search-architecture-ranking`: the GA cell measures the optimiser-family change at matched capacity, not "GA is competitive in absolute terms." Slow GA convergence vs PPO is itself a finding worth recording — it's the load-bearing question the comparison is designed to answer. |
| Smoke config uses small population × few generations; may not catch GA hyperparameter sensitivity that surfaces at the full `weight-search-architecture-ranking` budget | Smoke is for end-to-end pipeline verification (no crashes, fitness moves), not for hyperparameter validation. Real hyperparameter tuning lands in `weight-search-architecture-ranking`'s Phase 2 compute pre-flight. |
| Reusing the existing `GeneticAlgorithmOptimizer` means inheriting its hyperparameter defaults (tournament size, mutation σ, etc.) which were tuned for evolution-framework brain configs (much smaller param counts than 6,400) | The smoke config explicitly sets the GA hyperparameters at construction; downstream consumers (`weight-search-architecture-ranking` Phase 2) tune for the larger param count. The brain itself doesn't bake in defaults — it exposes them via the config. |
| The brain's "no critic head" choice diverges from MLPPPO's structure (which has both actor + critic); strict capacity-matching might require including a critic-equivalent | The cross-architecture comparison measures policy quality, not value-estimate quality. The critic is a PPO-specific auxiliary; GA fitness is the episode return directly. Matching actor-side capacity is the load-bearing parity claim; matching critic-side capacity would actively confound the comparison by giving GA a free auxiliary network it doesn't use. |

## Migration Plan

Fully additive. No existing brain, config, test, or downstream consumer is modified.

Rollback (if the brain causes issues during `weight-search-architecture-ranking` Phase 1a verification): revert the change. The L1 registry's decorator-registration pattern ensures absence of `FeedforwardGABrain` doesn't break any other code path — the `feedforwardga` config name simply becomes an unknown brain name at config-load time, which raises a clear error.

## Open Questions

None at design-finalisation time. The brain is small, self-contained, and reuses existing infrastructure. Hyperparameter choices for the smoke config are scoped to "does it train end-to-end?" not "does it train well" — the latter belongs in the consumer's Phase 2 compute pre-flight.
