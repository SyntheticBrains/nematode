# Proposal: Chemosensory Associative-Memory Probe

## Why

The T7 memory-separation programme has two data points so far. The artificial **bit-memory** control
([Logbook 030](../../../docs/experiments/logbooks/030-bit-memory-positive-control.md)) confirmed the
architecture comparison **can** resolve working memory — but it is a deliberately non-biological cue.
Its intended biological twin, depletion-driven **area-restricted search**
([Logbook 032](../../../docs/experiments/logbooks/032-ars-source-depletion.md)), came back a
**short-horizon null**: environmental depletion did not induce an episode-scale memory demand, which
is biologically correct (real ARS is sensory + slow-neuromodulatory-state driven, not a remembered
map). That leaves the "capability exists, natural demand limited" conclusion resting on a **single**
naturalistic task (the ARS null) plus an artificial positive control.

This change adds a **second, more faithful** naturalistic memory demand: **chemosensory associative
memory**. *C. elegans* genuinely does gustatory/olfactory associative learning — it pairs a salt or
odour cue with **food** (or **starvation**) and shifts approach↔avoidance accordingly (gustatory
plasticity; Kunitomo/Iino, Torayama; the butanone-conditioning literature). Unlike ARS, the memory
here is genuinely *used to guide current action across a delay* — the same computational shape as
bit-memory, but grounded in a documented worm behaviour rather than an arbitrary channel. It is the
biological "remember-and-use" probe the ARS null could not provide.

## What Changes

- **New config-gated associative-memory task** (`conditioning → [optional reversal] → delay → response`), off by default, extending the bit-memory phase-machine scaffolding. In a **conditioning
  phase** the agent experiences two chemosensory cues in sequence, one paired with a positive outcome
  and one neutral/aversive (the pairing sampled fresh per trial); then **with probability
  `reversal_prob` a reversal block re-presents the two cues with flipped outcomes**. After a **delay
  phase** (cues withheld, no external memory aids) a go-signalled **response phase** requires a binary
  readout of the **current** (post-reversal) rewarded cue.
- **The reversal is the load-bearing demand — working-memory *update*, not just retention.** A static
  single association would be an isomorph of bit-memory (hold one bit across a delay). Probabilistic
  reversal requires *overwriting a held association on new evidence* — which bit-memory cannot test,
  is the **biologically faithful** operation (real *C. elegans* gustatory/olfactory conditioning is
  centrally about re-learning when the contingency flips), and plausibly **discriminates architectures
  in a new way** (an attention arm re-reads the recent window; a recurrent arm must actively overwrite
  its gated state — the opposite of the default-to-hold init load-bearing for minGRU/minLSTM in 031).
  Because reversal is probabilistic and the initial cue randomised, a **hold-only** policy is at chance
  on the reversal fraction (only genuinely-updating arms clear the bar) and a **memoryless** policy is
  at chance throughout. The per-trial span (incl. the reversal block) is kept within the attention
  arm's `window_size` so the Transformer detector is not confounded by window reach.
- **Minimal observation, no external memory aids**: dedicated cue-identity + outcome (valence)
  channels during conditioning, a go-signal channel during response, and — as with bit-memory —
  **no STAM and no gradient sensing** across the delay, so only internal recurrent state can carry
  the association. Reuses the bit-memory cue/go channel machinery; adds an outcome/valence channel.
- **Per-episode-random association + association-conditioned reward**: correct response (act on the
  food-paired cue) rewarded, incorrect not/penalised; the rewarded cue sampled uniformly so
  chance = 50%.
- **Per-arm configs for the existing MUST arms** (no new architecture code) — the memory arms
  (LSTM/CfC/Transformer/minGRU/minLSTM) vs the memoryless MLP; **matched entropy** across arms
  (architecture the only variable), with the **Transformer treated as the reliable detector** given
  the recurrent-PPO instability observed in 032.
- **Evaluation + separation analysis**: a response-accuracy metric + the separation criterion
  (memory arms rise above the memoryless MLP at chance), reusing the committed paired-seed
  Wilcoxon + bootstrap + BH-FDR statistics layer.
- **Explicitly non-gating** for Gate 3: the MUST integrated-C3 cells and the 029/032 rankings are
  unchanged. This is a naturalistic complement to bit-memory + ARS, giving the memory-separation
  conclusion a second biological data point.

## Capabilities

### New Capabilities

- **`associative-memory-probe`** — the delayed-associative-match task: its conditioning/delay/
  response phase structure, the cue-identity + outcome (valence) + go observation channels, the
  per-episode-random cue→valence pairing, the association-conditioned reward, the
  no-external-memory-aid contract, and the separation-evaluation protocol. A self-contained,
  config-gated task capability that composes with the existing environment, configuration, sensing,
  and bit-memory task systems rather than changing their requirements.

### Modified Capabilities

None. The task is **additive and config-gated** — when disabled (the default), the environment,
observation pipeline, reward calculation, and configuration schema behave exactly as before.

## Impact

- **Sensory channels** — reuse the bit-memory cue/go modules; add an **outcome/valence** channel
  (`brain/modules.py` `ModuleName` + registry, `BrainParams` field) exposed only during conditioning.
- **Task phase machine** — a config-gated associative-memory mode extending the bit-memory
  phase/trial machine (`env` + `agent._create_brain_params` exposure; the env does not push into
  `BrainParams`). Tracks the per-trial cue→valence pairing + phase.
- **Association-conditioned reward + termination** — scored in the runner's post-action branch
  (`agent/runners.py`), reward params in an `AssociativeMemoryTaskConfig` block (not `RewardConfig`);
  bypasses the foraging/predator/thermal/satiety handlers when enabled (as bit-memory does).
- **Config schema** — an `AssociativeMemoryTaskConfig` block in `utils/config_loader.py`, off by
  default.
- **Configs** — per-arm scenario configs under a new `configs/scenarios/associative_memory/`.
- **Analysis** — a `scripts/analysis/` separation harness (response accuracy + paired stats),
  reusing `weight_search_architecture_ranking` helpers.
- **Tests** — phase transitions, cue withholding across the delay, per-episode pairing randomisation,
  association-conditioned reward scoring, observation-channel wiring.
- **No breaking changes; no new dependencies; no new architecture.**
- **Tracker** — ticks `T7.separation.associative_memory` with the separation verdict.

## Non-Goals

- **Not** faithful slow-forming associative learning (association that consolidates over
  minutes-to-hours / repeated trials via neuromodulator-gated plasticity) — that is the single-shot
  within-episode *compression* of the real behaviour, and the faithful version is **phase-7** (an L4
  modulated-STDP application; see `docs/roadmap.md` § Known Gaps + Phase 7).
- **Not** a new architecture, statistics layer, or Gate-3 change.
- **Not** a spatial/foraging task — like bit-memory, spatial/foraging/predator/thermal dynamics are
  disabled so the demand is purely the remembered association.
- **Not** a per-episode / long-horizon association hold (condition once, probe across many trials).
  That exceeds the attention arm's `window_size` and risks the long-horizon learnability wall ARS hit
  (Logbook 032); it is a documented deferred harder variant, aligned with the phase-7 faithful
  slow-forming-memory work.
- **Not** a capacity variant (hold K concurrent cue→valence associations, probe one). A 302-neuron
  animal is not a multi-item working-memory system (less biological), and it is floor-prone at larger
  K. Deferred as a possible follow-up if the reversal probe separates.
