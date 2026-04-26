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

### Decision 1: WeightPersistence as the encoder primitive, not torch state_dict directly

`MLPPPOBrain` and `LSTMPPOBrain` already implement `WeightPersistence` (mlpppo.py:660, lstmppo.py:895). Their `get_weight_components()` returns a `dict[str, WeightComponent]` where each component bundles a name, a state dict, and metadata. Encoders use this protocol — they do **not** reach inside the brain to grab `actor.state_dict()` directly.

**Why:** the protocol is the contract. If we go around it, the encoder breaks the moment a brain refactors its internal module structure. The protocol also lets us select components — encoders explicitly request `{"policy", "value"}` for MLPPPO and `{"lstm", "actor", "critic"}` for LSTMPPO, **deliberately excluding** `optimizer` / `actor_optimizer` / `critic_optimizer` (Adam state, not part of the genome) and `training_state` (episode counters, runtime state). This separation is the right primitive for M3 Lamarckian inheritance later: M3 may choose to inherit optimiser state too, and the protocol lets it do so by widening the component set.

**Alternative considered:** flatten brain by walking `brain.actor.state_dict() | brain.critic.state_dict()` directly. Rejected because it duplicates knowledge that already lives in `get_weight_components()` and would silently drift if a brain adds a head.

### Decision 2: Reset `_episode_count = 0` explicitly on decode

The exploration phase flagged that `MLPPPOBrain._episode_count` is part of `training_state` but is consulted by the LR scheduler at runtime. If a genome captured at episode 800 (with `_episode_count = 800`) is decoded into a fresh evolution run, the new brain inherits a stale episode count and the LR scheduler is in the wrong regime.

The encoder excludes `training_state` from the genome, but `load_weight_components` won't reset attributes the genome doesn't supply. So the encoder explicitly sets `brain._episode_count = 0` after `load_weight_components`. Same for any analogous attributes on LSTMPPOBrain (verified during exploration: hidden state already resets at `prepare_episode()`, no extra handling needed).

**Why:** evolution evaluations need to start each genome at a known initial state. Without this reset, fitness would silently depend on which generation the genome was born in. This is the kind of bug that doesn't fail tests but invalidates a campaign.

### Decision 3: EpisodicSuccessRate is frozen — no `.learn()` call

For M0, fitness evaluates a brain initialised from the genome and run for K episodes **without** learning. The brain's weights stay fixed; we measure how good the genome's initial weights are.

**Why:** this is the cleanest possible fitness signal for proving the framework works. It also separates concerns: M2 introduces `LearnedPerformanceFitness` (learn for N episodes, then evaluate frozen) which has its own design questions (when does the LR scheduler reset, how is optimiser state seeded). Putting both in M0 forces M2's design decisions before M0 is even merged. Per the user-confirmed scope choice, M0 ships the simpler primitive.

**Alternative considered:** ship both fitness modes. Rejected: M0 grows by ~150 LOC and we make M2 design decisions prematurely.

### Decision 4: Pickle resume preserved in M0

The legacy script has mature pickle-based optimiser checkpoint/resume (run_evolution.py:642-660). CMA-ES covariance matrices are non-trivial to reconstruct; pickle handles them correctly. Long-running M3/M4 campaigns (50+ generations) genuinely need resume — a crash at gen 30 would otherwise burn the entire run.

**Why preserve in M0:** the pattern is brain-agnostic (it pickles the optimiser, not the brain). Porting it now is ~50 LOC and avoids re-implementing it later under deadline pressure. Resume tests are part of the M0 test plan to catch regressions early.

**What we add:** the checkpoint also stores the RNG state and lineage CSV path so resume reconstructs the full evaluation context, not just the optimiser.

### Decision 5: Lineage CSV columns

Columns: `generation, child_id, parent_ids, fitness, brain_type`. `parent_ids` is a `;`-joined string (CSV-friendly). `brain_type` is included so that future co-evolution runs (M5) — where predator and prey populations are interleaved in the same evolution_results directory — can be sliced by species.

**Why CSV not JSON Lines:** existing experiment tracking uses CSV (artifacts/logbooks/011/). Tooling and human inspection are CSV-native. Append mode plays nicely with resume: the file just keeps growing.

### Decision 6: Encoder registry is a static dict, not a plugin discovery system

`ENCODER_REGISTRY: dict[str, type[GenomeEncoder]] = {"mlpppo": MLPPPOEncoder, "lstmppo": LSTMPPOEncoder}`. New encoders are added by editing this dict.

**Why:** matches the existing `BRAIN_CONFIG_MAP` pattern (config_loader.py:128-148). Discovery via entry points or decorators is over-engineering at this stage. A future Phase 6 quantum re-evaluation that adds `QVarCircuitEncoder` is a one-line registry edit.

### Decision 7: Two pilot configs in M0, not just one

Both `mlpppo_foraging_small.yml` and `lstmppo_foraging_small_klinotaxis.yml` ship in M0. The MLPPPO config is the cheap framework smoke; the LSTMPPO+klinotaxis config is the first-class biological target that M3/M4/M5/M6 will all build on.

**Why both now:** if only the MLPPPO config ships, M3 has to add the LSTMPPO+klinotaxis config plus prove the LSTMPPO encoder works at the same time, mixing framework concerns with science concerns. Shipping both in M0 means M3's PR is purely about Lamarckian inheritance, not "does the framework even work for LSTMPPO?"

### Decision 8: Move the legacy script under `scripts/legacy/`, do not delete

**Why preserve:** the legacy script has running history, configs, and pickle artefacts that someone may want to re-run for comparison. Deleting it forces a git archaeology exercise to recover anything. Moving it costs nothing and signals "no maintenance" via the directory name.

**Why under `scripts/legacy/`:** any directory name signals deprecation. `legacy/` matches industry convention. We don't add to test coverage there; we don't refactor it; if it breaks, that's fine.

## Risks

1. **Encoder round-trip determinism is fragile.** Brains can have non-deterministic decode (e.g. `torch.nn.Module` parameter init uses CUDA RNG state on GPU). Mitigation: round-trip tests use `torch.manual_seed(0)` before encode and after decode, and assert action equality on a fixed seeded input. If a brain has hidden non-determinism (cuDNN modes, dropout in eval), tests catch it before merge.

2. **Pickle resume couples optimiser internals to a Python version.** CMA-ES library updates can break pickle compatibility. Mitigation: version-tag the checkpoint file (`checkpoint_version: int`) and validate on load. Reject incompatible checkpoints with a clear error rather than silent corruption.

3. **`evolution:` block schema diverges from CLI flag defaults.** Easy to introduce drift where YAML default disagrees with CLI default. Mitigation: CLI flags default to `None` and only override the YAML value when explicitly passed. Single source of truth lives in `EvolutionConfig` Pydantic defaults.

4. **Smoke runs are slow if `episodes_per_eval` is too high.** A 10-gen × 8-pop × 15-eps smoke is 1200 episode-runs and could be 5+ minutes. Mitigation: pilot configs use `episodes_per_eval: 3` (the minimum that gives a meaningful fitness signal) so the smoke is sub-2-minute.

5. **The new `scripts/run_evolution.py` will surprise anyone with muscle memory** (running a Phase 5 mlpppo evolution where they expected QVarCircuit). Mitigation: the script logs `Brain type: mlpppo` prominently on startup; the legacy script is one `legacy/` path away with identical flags.

## Maintenance

- Every new encoder is one entry in `ENCODER_REGISTRY` plus one round-trip test
- Adding `LearnedPerformanceFitness` (M2) is a new class in `fitness.py`; the encoder protocol does not change
- Inheritance strategies (M3 Lamarckian, M4 Baldwin) live in a future `evolution/inheritance.py`; they consume the encoder + fitness protocols without modifying them
- The `2026-04-28-add-evolution-framework` change archives on merge (unlike the M-1 tracking change which stays open until M7)
