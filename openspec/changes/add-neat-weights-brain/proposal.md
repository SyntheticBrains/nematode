# Proposal: Add NEAT-Weights Brain (`FeedforwardGABrain`)

## Why

The Phase 6 cross-architecture comparison sweep (Tranche 4 of `phase6-tracking`) needs four MUST architecture families to evaluate side-by-side, one of which is "NEAT-evolved weights" per `phase6-tracking/design.md` Decision 4. NEAT topology evolution lands at Tranche 8; this tranche only needs the weight-search variant. The repository already ships `GeneticAlgorithmOptimizer` (`packages/quantum-nematode/quantumnematode/optimizers/evolutionary.py`), a full evolution loop (`quantumnematode/evolution/`), genome / encoder / fitness infrastructure, and the `scripts/run_evolution.py` launcher — adding `neat-python` as a new dependency would duplicate all of it. The honest minimal increment is a feed-forward brain with matched-capacity topology (mirroring MLPPPO small) whose weights are evolved by the existing GA optimiser. This brain becomes the GA-based weight-search comparator in the upcoming `weight-search-architecture-ranking` change.

## What Changes

- New brain class `FeedforwardGABrain` in `packages/quantum-nematode/quantumnematode/brain/arch/feedforward_ga.py` implementing the `Brain` Protocol from `brain/arch/_brain.py:346-368`.
- Brain self-registers via `@register_brain` per the decorator-registration pattern established by `add-architecture-plugin-interface`.
- Matched-capacity feed-forward topology: hidden width + layer count matching `MLPPPOBrain` small (`actor_hidden_dim=64`, `num_hidden_layers=2`) so the cross-architecture comparison isolates the weight-search-optimiser change (PPO vs GA) from a capacity confound.
- Weights consumed from the existing `GeneticAlgorithmOptimizer`; no new optimiser code. Brain config exposes the GA hyperparameters (population size, mutation rate, crossover, selection) the optimiser already accepts.
- New brain-loader branch in `packages/quantum-nematode/quantumnematode/utils/brain_factory.py` — a single registry-consume entry per the post-refactor convention, not a 19-elif-style branch.
- One smoke config at `configs/scenarios/foraging/feedforwardga_small_klinotaxis.yml` mirroring `connectomeppo_small_klinotaxis.yml`'s structure: 500 episodes, single seed, klinotaxis sensing, headless-ready.
- Unit + smoke tests at `packages/quantum-nematode/tests/quantumnematode_tests/brain/test_feedforward_ga.py`: Protocol conformance, forward-pass shape, GA-update determinism with seeded RNG, end-to-end short-budget evolution run.

## Capabilities

### New Capabilities

- `feedforward-ga-brain`: Feed-forward neural-network brain whose weights are evolved by the existing GA optimiser. Matched capacity to MLPPPO small. Consumed by the upcoming `weight-search-architecture-ranking` change as the GA-based weight-search comparator. Distinct from the future NEAT topology search (Tranche 8), which evolves both topology and weights via TensorNEAT.

### Modified Capabilities

None. `brain-architecture` (the L1 registry) is consumed via its existing decorator surface; `evolution-framework` (the GA optimiser + evolution loop) is consumed via its existing public API. Neither requires requirements-level changes.

## Impact

- **New code**: ~1 brain module + ~1 config file + ~1 test file. Estimated 300-500 LOC including tests.
- **New dependencies**: none. The GA optimiser and evolution-loop infrastructure already exist.
- **Affected systems**: `brain_factory.py` gains a one-line registry-consume entry; `brain/arch/__init__.py` imports the new module so its `@register_brain` decorator fires at startup.
- **Downstream**: the upcoming `weight-search-architecture-ranking` change depends on this brain being available in the registry. The C1/C2/C3 curriculum cells for the GA-NEAT row in that comparison consume configs that instantiate `FeedforwardGABrain`.
- **Out of scope for this change**: NEAT topology evolution (Tranche 8, separate change), GPU-accelerated TensorNEAT integration (also Tranche 8), GA-evolved variants of other brain families (those would be follow-up changes if the cross-architecture comparison surfaces evidence GA is useful for them).
- **Backward compatibility**: fully additive — no existing brain, config, or test is modified.
