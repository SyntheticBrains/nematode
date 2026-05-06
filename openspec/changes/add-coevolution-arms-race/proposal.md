## Why

M1 shipped the pluggable predator-brain seam (`PredatorBrain` Protocol + `HeuristicPredatorBrain` adapter + per-predator metrics) so predators could co-evolve under M5. M5 is the headline scientific milestone for Phase 5: a Red Queen arms race between learnable prey (LSTMPPO+klinotaxis, the M3 Lamarckian winner) and learnable predators (MLPPPO with predator-specific I/O). The substrate is in place; this change builds the co-evolution loop, predator-stack genome encoders/fitness, hall-of-fame opposition, generality probe, and Red Queen metrics that turn the substrate into a runnable, decisive experiment.

A second motivation: M4 closed STOP after diagnosing single-task K=50 PPO as the substrate constraint for Baldwin (no task variation → no Baldwin axis). Co-evolution intrinsically introduces task variation — the prey's "task" is to survive the *current* predator population, which shifts every K predator generations under alternating training. M5.7 ships secondary Baldwin instrumentation that observes this naturally varying task distribution; the M5 verdict gate stays purely on Red Queen dynamics, but if the secondary readout fires it triggers M4.7 as a follow-up rather than re-opening M4.

## What Changes

- **New module `quantumnematode/env/mlpppo_predator_brain.py`**: `MLPPPOPredatorBrain` implementing the existing `PredatorBrain` Protocol from M1. Mirrors the agent-side MLPPPO architecture (reuses `DEFAULT_ACTOR_HIDDEN_DIM` / `DEFAULT_CRITIC_HIDDEN_DIM` / `DEFAULT_NUM_HIDDEN_LAYERS` from `quantumnematode.brain.arch.mlpppo`, with value head) via composition; predator-specific I/O encodes `(predator_position, agent_positions[:k_nearest=2], detection_radius/grid_size, damage_radius/grid_size, step_index/max_steps)` to a flat float vector and emits a 5-way categorical over `PredatorAction`. Implements `WeightPersistence` for genome encoder round-trip.
- **New module `quantumnematode/env/_predator_brain_pretrain.py`**: 50-episode behavioural-cloning pretrain helper that imitates `HeuristicPredatorBrain` decisions to bootstrap the gen-0 predator weights (avoids zero-fitness-gradient cold-start risk).
- **`PredatorBrainConfig.kind` Literal extension**: `"heuristic"` → `"heuristic" | "mlpppo_predator"`. The runtime dispatcher (`_build_predator_brain` in env.py) gains the `mlpppo_predator` branch. Pydantic schema in `config_loader.py` mirrors the extension.
- **New module `quantumnematode/evolution/predator_encoders.py`**: `MLPPPOPredatorEncoder` subclassing `_ClassicalPPOEncoder` (the parent class shared with `MLPPPOEncoder`); registered in a separate `PREDATOR_ENCODER_REGISTRY` to keep predator dispatch single-population from the agent-side `ENCODER_REGISTRY`.
- **New module `quantumnematode/evolution/predator_fitness.py`**: `PredatorEpisodicKillRate` and `PredatorLearnedPerformanceFitness` implementing the existing `FitnessFunction` Protocol (`evaluate(genome, sim_config, encoder, *, episodes, seed) -> float`). The predator fitness internally drives the multi-agent runner against frozen prey opponents (configured via `sim_config` patching at the call site) and reads `MultiAgentEpisodeResult.per_predator_kills` (primary signal) and `per_predator_prey_proximity_steps` (secondary signal when kills=0).
- **New module `quantumnematode/evolution/hall_of_fame.py`**: `HallOfFame` class — bounded `deque[Genome]` per side with quality-based eviction (lowest-fitness HoF entry replaced, NOT FIFO); `mix_with_pop(rng, pop, frac_hof=0.3)` produces 70/30 current-pop / HoF mixtures for opposition sampling.
- **New module `quantumnematode/evolution/redqueen_metrics.py`**: pure functions for the four Red Queen analysis primitives — `phenotypic_cycling` (Lomb-Scargle / autocorrelation), `trait_escalation` (linear regression of trait mean over generations), `fitness_lag` (cross-correlation between prey/predator fitness series), `coupled_rate` (rate of trait change covariance), `generality` (held-out opponent fitness curve).
- **New module `quantumnematode/evolution/coevolution.py`**: `CoevolutionLoop` orchestrator. Two `EvolutionLoop`-shaped sides (one per population), alternating-schedule controller (K=10 gens per side; opposing pop frozen during off-block; per-K-block fresh `OptunaTPEOptimizer` instance to clear stale opposition-conditioned posterior — re-construct rather than reset, since the existing `OptunaTPEOptimizer` at `quantumnematode.optimizers.evolutionary.OptunaTPEOptimizer` has no public reset method), HoF buffer per side, generality probe hook (every N gens against a held-out frozen opponent set). Reuses the `EvolutionLoop._evaluate_in_worker` 11-tuple worker pattern; opponent brain weights are injected via `sim_config` patching following the M2 idiom (the worker tuple is unchanged — only one `encoder` and one `fitness` per call).
- **New configs**: `configs/evolution/coevolution_pilot.yml` (30 gens × 2 seeds × pop 24 prey / 16 predator × K=10 alternating × K=20 ep/eval × HoF size 8 × pretrain on/off pilot ablation arms) and `configs/evolution/coevolution_full.yml` (50–70 gens × 4 seeds × same pop × full HoF + generality probe every 10 gens).
- **New campaign scripts**: `scripts/campaigns/run_coevolution.py` (per-seed driver), `phase5_m5_coevolution_pilot.sh`, `phase5_m5_coevolution_full.sh`, `aggregate_m5_pilot.py` (Red Queen aggregator + plots + verdict.csv + summary.md, mirroring `aggregate_baldwin_retry_pilot.py`'s 782-LoC reference), `baldwin_m5_secondary_eval.py` (M5.7 readout reusing `baldwin_f1_postpilot_eval.py`).
- **Pilot bootstrap ablation**: pilot runs both predator-bootstrap arms — one seed with heuristic-imitation pretrain, one cold-start — and uses the result to choose pretrain on/off for the full run.
- **Decision gate (softened-disjunctive)**: GO if (a) phenotypic cycling visible (autocorrelation peak at lag ∈ [3,15] with p\<0.05, OR dominant FFT bin > 2× median bin power) OR (b) trait escalation visible (linear-regression slope significantly non-zero with directional sign) over the 30-gen window — in ≥2 of 4 full-run seeds. Generality probe reports alongside but is not a verdict input. Concrete thresholds locked from pilot before full run.
- **M5.7 secondary Baldwin instrumentation hook**: per-gen prey hyperparam spread + F1-style elite-vs-prior signal-delta against the current predator pop. Observation only; readout (signal-delta > +0.05 at K′=10 across ≥2 of 4 seeds AND prey hyperparam spread on actor_lr/entropy_coef tightens by ≥30% from gen 5 → 30) triggers M4.7 follow-up but does NOT alter M5's GO/STOP.

Not a behaviour change for any existing scenario YAML — `PredatorBrainConfig.kind` defaults to `"heuristic"`, and no existing config sets `mlpppo_predator`. The M5 work touches only opt-in code paths.

## Capabilities

### New Capabilities

- `co-evolution`: covers the `CoevolutionLoop` orchestrator (alternating schedule, per-block TPE reset, dual-population worker dispatch), hall-of-fame buffer semantics (quality-based eviction, mix-with-pop sampling), generality probe (held-out opponent set semantics + cadence), and the softened-disjunctive decision gate. Distinct from `evolution-framework` which covers single-population evolution; the co-evolution capability layers on top.
- `red-queen-analysis`: covers the Red Queen metrics module (phenotypic cycling, trait escalation, fitness lag, coupled rate, generality) and the aggregator's verdict logic. Distinct concern from the loop itself: any future predator-prey dynamics analysis (e.g. for transfer learning experiments) reuses these primitives.

### Modified Capabilities

- `environment-simulation`: extends the M1 "Predator Entities in Dynamic Environments" requirement to register `MLPPPOPredatorBrain` as the second-shipped `PredatorBrain` implementation (alongside `HeuristicPredatorBrain`). Adds the predator-brain dispatcher's `mlpppo_predator` kind, the predator I/O encoding contract (k_nearest=2, normalised position/radii/step), and the `PredatorAction` 5-way categorical output mapping for learnable brains.
- `evolution-framework`: extends the existing genome-encoder + fitness-function abstractions to cover the predator side. Adds `MLPPPOPredatorEncoder` (separate `PREDATOR_ENCODER_REGISTRY` to keep predator dispatch isolated from agent-side `ENCODER_REGISTRY`), `PredatorEpisodicKillRate` / `PredatorLearnedPerformanceFitness` fitness functions, and the `HallOfFame` buffer abstraction (the latter is a primitive used by `co-evolution` but lives in the evolution-framework capability since the eviction policy + sampling semantics are general-purpose).

## Impact

**Code (new):**

- `packages/quantum-nematode/quantumnematode/env/mlpppo_predator_brain.py`
- `packages/quantum-nematode/quantumnematode/env/_predator_brain_pretrain.py`
- `packages/quantum-nematode/quantumnematode/evolution/predator_encoders.py`
- `packages/quantum-nematode/quantumnematode/evolution/predator_fitness.py`
- `packages/quantum-nematode/quantumnematode/evolution/hall_of_fame.py`
- `packages/quantum-nematode/quantumnematode/evolution/coevolution.py`
- `packages/quantum-nematode/quantumnematode/evolution/redqueen_metrics.py`

**Code (modified):**

- `packages/quantum-nematode/quantumnematode/env/predator_brain.py` — `PredatorBrainConfig.kind` Literal extension.
- `packages/quantum-nematode/quantumnematode/utils/config_loader.py` — `PredatorBrainConfigSchema.kind` Literal extension matching above.
- `packages/quantum-nematode/quantumnematode/env/env.py` — `_build_predator_brain` dispatcher gains the `mlpppo_predator` branch.

**Configs (new):**

- `configs/evolution/coevolution_pilot.yml`
- `configs/evolution/coevolution_full.yml`

**Scripts (new):**

- `scripts/campaigns/run_coevolution.py`
- `scripts/campaigns/phase5_m5_coevolution_pilot.sh`
- `scripts/campaigns/phase5_m5_coevolution_full.sh`
- `scripts/campaigns/aggregate_m5_pilot.py`
- `scripts/campaigns/baldwin_m5_secondary_eval.py`

**Tests (new):** ~70-80 cases across:

- `tests/env/test_mlpppo_predator_brain.py` (~15 cases) — Protocol conformance, weight round-trip, deterministic-action sanity, byte-equivalence at fixed seeds.
- `tests/env/test_predator_brain_pretrain.py` (~6 cases) — imitation loss decreases over 50 episodes; final action distribution matches heuristic on >70% of test states.
- `tests/evolution/test_predator_encoders.py` (~6 cases) — round-trip, dim correctness, registry integration.
- `tests/evolution/test_predator_fitness.py` (~8 cases) — kill-rate evaluation against synthetic episodes; secondary proximity signal when kills=0.
- `tests/evolution/test_hall_of_fame.py` (~10 cases) — quality-based eviction, sampling reproducibility, mix_with_pop fraction correctness.
- `tests/evolution/test_coevolution.py` (~15 cases) — alternating-schedule K-block boundaries, opposing pop frozen during off-block, HoF push timing, probe cadence.
- `tests/evolution/test_redqueen_metrics.py` (~12 cases) — synthetic series with known answers (sine → cycling, ramp → escalation, anti-correlated → fitness lag).

**Compute:** Pilot ~8 wall-hours total (2 seeds × ~4h); full campaign ~30-40 wall-hours total (4 seeds × ~7-10h). Pilot-first sequencing gates the full run.

**Out of scope (deferred or future work):** GA optimiser ablation (deferred — TPE for both sides per M3+ default; ablate if cycling stalls), predator-side pheromones / signalling, multi-predator cooperation, transfer to single-agent runners (M5.6 covers transfer eval as post-pilot ablation), quantum predator brains, architecture-changing schema knobs for predator (only weights co-evolve), Baldwin VERDICT under M5 (M5.7 is observation only — M4 STOP is unchanged unless readout fires, in which case M4.7 follow-up).

**Dependencies:** No new external dependencies. Uses existing torch (MLPPPO), numpy, optuna (TPE) stack.

**OpenSpec capability targets**: 2 new (`co-evolution`, `red-queen-analysis`) + 2 modified (`environment-simulation`, `evolution-framework`).

**Tracker / roadmap:** ticks tracker M5.0–M5.9 (`openspec/changes/2026-04-26-phase5-tracking/tasks.md`); flips Phase 5 milestone tracker M5 row in `docs/roadmap.md` on full-run verdict.
