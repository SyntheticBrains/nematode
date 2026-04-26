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

### Decision 8: Delete the legacy script entirely

**The user's stated rationale (review feedback):** retaining legacy code under `scripts/legacy/` would tie M0 to suboptimal implementation choices — there'd be a temptation to mirror the old script's structure, copy its CLI flags verbatim, or maintain bug-compatibility "just in case." Deleting it cleanly signals that the new framework is free to make better choices.

**What we lose:** the existing CI smoke test (`test_run_evolution_smoke`) that exercised the QVarCircuit path. Replaced by a new MLPPPO smoke against the new framework — different brain, but the assertion ("CLI doesn't crash with minimal parameters") is the same.

**What we keep:** git history. `git log -- scripts/run_evolution.py` and `git show <commit>:scripts/run_evolution.py` retrieve the old code if anyone needs it for comparison. The `configs/evolution/qvarcircuit_foraging_small.yml` config is also deleted — no consumer remains.

**Future quantum brain support:** if a Phase 6 quantum re-evaluation needs evolution, a `QVarCircuitEncoder` is added cleanly to the new framework's registry (a one-class change). It will not resurrect the legacy script.

## Risks

1. **Encoder round-trip determinism.** Round-trip tests fix `torch.manual_seed(0)` before encode and again before decode, then assert that both brains produce identical actions on the same seeded input. CI runs CPU-only so cuDNN/CUDA non-determinism doesn't apply here. If a brain ever introduces eval-time stochasticity (e.g. dropout left on outside training mode), the test catches it.

2. **Pickle resume couples optimiser internals to a Python version.** CMA-ES library updates can break pickle compatibility. Mitigation: version-tag the checkpoint file (`checkpoint_version: int`) and validate on load. Reject incompatible checkpoints with a clear error rather than silent corruption.

3. **`evolution:` block schema diverges from CLI flag defaults.** Easy to introduce drift where YAML default disagrees with CLI default. Mitigation: CLI flags default to `None` and only override the YAML value when explicitly passed. Single source of truth lives in `EvolutionConfig` Pydantic defaults.

4. **Smoke runs are slow if `episodes_per_eval` is too high.** A 10-gen × 8-pop × 15-eps smoke is 1200 episode-runs and could be 5+ minutes. Mitigation: pilot configs use `episodes_per_eval: 3` (the minimum that gives a meaningful fitness signal) so the smoke is sub-2-minute.

5. **Anyone with muscle memory of the legacy script will be surprised.** The legacy script is deleted (not preserved under `scripts/legacy/`). Mitigation: the new script logs `Brain type: <name>` prominently on startup; if someone runs it with the old QVarCircuit config, the error message names the registered brains and the breakage is immediate and obvious. Git history retrieves the old script if absolutely needed.

## Maintenance

- Every new encoder is one entry in `ENCODER_REGISTRY` plus one round-trip test
- Adding `LearnedPerformanceFitness` (M2) is a new class in `fitness.py`; the encoder protocol does not change
- Inheritance strategies (M3 Lamarckian, M4 Baldwin) live in a future `evolution/inheritance.py`; they consume the encoder + fitness protocols without modifying them
- The `2026-04-28-add-evolution-framework` change archives on merge (unlike the M-1 tracking change which stays open until M7)
