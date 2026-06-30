# Proposal: Bit-Memory Working-Memory Positive Control

## Why

Across the T4 grid ranking ([Logbook 025](../../../docs/experiments/logbooks/025-weight-search-architecture-ranking.md)) and the T7 continuous ranking ([Logbook 029](../../../docs/experiments/logbooks/029-continuous-architecture-ranking.md)), the architecture comparison does **not** discriminate on the reactive cells — the field ties (or MLP wins outright) because the task is locally readable with a one-step temporal derivative and exercises **no working memory**. Before investing in memory-capable new architectures (minGRU / modified-S5) or the biologically-plausible area-restricted-search cell, we need to confirm the comparison **can separate working memory at all**. This change ships a deliberately-artificial **bit-memory positive control** as that instrument: a clean delayed-match-to-cue task that a memoryless policy provably cannot solve above chance. A positive result is the gate that unlocks the memory-axis work ([`T7.separation.new_arch_candidates`](../phase6-tracking/tasks.md) and [`T7.separation.ars_depletion`](../phase6-tracking/tasks.md)); a **null result is itself a strong finding** — it would mean the comparison cannot resolve working memory, and the memory-axis programme should not be pursued.

## What Changes

- **New deliberately-artificial delayed-match-to-cue task** (`cue → delay → response`), config-gated and off by default: a discrete binary cue is shown during a cue phase, **withheld** across a delay phase, and on a go-signalled response phase the agent must act on the *remembered* cue. The optimal action depends on information available earlier but absent at decision time, so a memoryless policy is pinned at chance.
- **Minimal observation with no external memory aids**: a dedicated **cue** channel (cue value during the cue phase, zeroed otherwise) and a **go-signal** channel (1 during the response phase). Deliberately **no STAM, no gradient sensing** — STAM is an external recency buffer fed into the observation, which would let a memoryless policy read the cue back; withholding it forces the policy to retain the cue in **internal recurrent state**. This is the design property that makes the control unambiguous.
- **Episode phase state machine** (cue / delay / response) and a **cue-conditioned reward** (correct response rewarded; incorrect not / penalised), with the cue sampled uniformly per episode so chance = 50%.
- **Per-arm configs for the existing 5 MUST arms** (`mlpppo`, `lstmppo`, `cfcppo`, `transformerppo`, `connectomeppo`) — **no new architecture code**; this runs the comparison we already have on a task that exercises the missing axis.
- **Evaluation + separation analysis**: a cue-match success-rate metric and a separation criterion (the recurrent / attention arms rise well above the memoryless MLP, which stays near chance), reusing the committed paired-seed Wilcoxon + bootstrap + BH-FDR statistics layer ([`architecture-comparison-protocol`](../../specs/architecture-comparison-protocol/spec.md)).
- **Explicitly non-gating** for Phase 6 Gate 3: the MUST integrated-C3 cells and the 029 ranking are unchanged. This task is the **decision gate** for elevating `T7.separation.ars_depletion` (its biological twin) and `T7.separation.new_arch_candidates` (the new memory arms).

## Capabilities

### New Capabilities

- **`bit-memory-positive-control`** — the delayed-match-to-cue task: its phase structure, the cue/go observation channels, the cue-conditioned reward, the no-external-memory-aid contract, and the separation-evaluation protocol. This is a self-contained, config-gated task capability; it composes with the existing environment, configuration, and sensing systems rather than changing their requirements.

### Modified Capabilities

None. The task is **additive and config-gated** — when the task is disabled (the default), the environment, observation pipeline, reward calculation, and configuration schema behave exactly as before. The new cue/go sensory modules and the `BitMemoryTaskConfig` block extend the existing systems without altering any existing spec's stated requirements.

## Impact

- **New sensory modules** (cue, go-signal) — `packages/quantum-nematode/quantumnematode/brain/modules.py` (`ModuleName` enum + `SENSORY_MODULES` registry) and the corresponding `BrainParams` fields in `brain/arch/_brain.py`.
- **Task phase machine + cue/go exposure** — the environment (`env/env.py` / `env/continuous_2d.py`) gains a config-gated bit-memory mode that tracks the per-episode cue + phase and **exposes** the cue/go state; the agent's param builder (`agent._create_brain_params`) reads it into the new `BrainParams.cue_signal` / `go_signal` fields (the env does not push into `BrainParams`).
- **Cue-conditioned reward + termination** — scored in the episode runner's **post-action** branch (`agent/runners.py`), since the response is the brain's just-emitted action; reward parameters live in `BitMemoryTaskConfig` (not `RewardConfig`), and episode-done is driven by the trial counter. The runner also bypasses the foraging/predator/thermotaxis/satiety/health handlers when the task is enabled.
- **Config schema** — a `BitMemoryTaskConfig` block in `utils/config_loader.py`, off by default.
- **Configs** — per-arm scenario configs under a new `configs/scenarios/bit_memory/` directory (the 5 MUST arms).
- **Analysis** — a `scripts/analysis/` separation harness (success rate + paired stats), reusing `weight_search_architecture_ranking` helpers.
- **Tests** — phase-machine transitions, cue withholding during the delay, cue-match reward scoring, observation-channel wiring.
- **No breaking changes; no new dependencies; no new architecture.** Existing arms and the existing training/episode plumbing are reused.
- **Tracker**: ticks `T7.separation.bit_memory_control` and records the separation verdict, which conditionally unblocks `T7.separation.ars_depletion` and `T7.separation.new_arch_candidates`.
