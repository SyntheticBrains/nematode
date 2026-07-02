# Design: Chemosensory Associative-Memory Probe

## Context

The bit-memory positive control ([Logbook 030](../../../docs/experiments/logbooks/030-bit-memory-positive-control.md))
proved the architecture comparison can resolve working memory; the ARS biological twin
([Logbook 032](../../../docs/experiments/logbooks/032-ars-source-depletion.md)) returned a
short-horizon null. This change adds a second, more biological memory demand — chemosensory
associative memory — that reuses the bit-memory task infrastructure: the config-gated phase/trial
machine, the cue/go observation channels, the no-external-memory-aid contract, the binary
action-sign readout, the association-conditioned response reward, and the separation-evaluation
harness. The one genuinely new piece is an **outcome (valence) channel** and a **conditioning phase**
that presents two cues so the agent must *bind* cue-identity to outcome, not echo a single bit.

## Goals / Non-Goals

**Goals:** a config-gated, off-by-default delayed-**associative**-match task where each trial presents
two cues with opposite outcomes (which cue is rewarded sampled per trial), withholds them across a
delay, and requires a binary readout of the remembered rewarded cue; a memoryless policy is pinned at
chance; the memory arms are expected to separate. Reuse bit-memory infrastructure wherever possible.

**Non-Goals:** not a new architecture / stats layer / Gate-3 change; not a spatial task; **not** a
per-episode / long-horizon association hold (deferred — see D1); not faithful slow-forming plasticity
(phase-7).

## Decisions

### D1 — Per-trial delayed-associative-match with **probabilistic within-trial reversal**, within the attention window

Each **trial** is `conditioning → [optional reversal] → delay → response`, sampled fresh per trial and
kept `< window_size` of the attention arm. Conditioning presents the two cues with opposite outcomes
(one rewarded); then **with probability `reversal_prob` the trial includes a reversal block** that
re-presents the two cues with **flipped** outcomes. The response phase requires the **current** rewarded
cue — the post-reversal association if the trial reversed, else the original.

**Why reversal (not a static association, and not capacity).** A single static association is
information-theoretically ~1 bit and mechanically an isomorph of bit-memory (select the +cue, hold one
scalar across the delay) — it would reproduce the 030 separation with little new information. Reversal
tests **working-memory *update*/flexibility** — overwriting a held association on new evidence — which
bit-memory's static hold cannot, and which is the **biologically faithful** operation (real *C. elegans*
gustatory/olfactory conditioning is centrally about re-learning when the cue↔outcome contingency
flips). It is also the operation most likely to **discriminate architectures in a new way**: an
attention arm re-reads the recent (post-reversal) window, whereas a recurrent arm must actively
*overwrite* its gated state — the opposite pressure to the default-to-hold retention-gate init that was
load-bearing for minGRU/minLSTM (031). Capacity (hold K bindings) is deferred: a 302-neuron animal is
not a multi-item-working-memory system (less biological) and it is floor-prone at larger K.

**The load-bearing property**: because reversal is *probabilistic* and the initial rewarded cue is
randomised, the current rewarded cue = initial ⊕ reversed. A **hold-only** policy (tracks the initial
association, ignores the reversal) is therefore at ~chance on the reversal fraction of trials, so
**only genuinely-updating arms clear the bar** — and a memoryless policy is at chance throughout. The
per-episode / long-horizon variant and the capacity variant are documented deferred harder variants.

### D2 — Observation: cue-identity + outcome (valence) channels during conditioning; two cues in sequence

Reuse the bit-memory **cue** channel for cue-identity (signed scalar: cue A = `+1`, cue B = `-1`) and
add one new **outcome** channel (valence: `+1` rewarded-paired, `-1` not). The conditioning phase
presents the two cues **sequentially** — `(cue=A, outcome=vA)` then `(cue=B, outcome=vB)`, with
exactly one outcome positive — so the agent must *bind* each cue-identity to its outcome and retain
which identity carried the positive outcome. Presentation order and the rewarded-cue assignment are
randomised per trial. The **reversal block reuses the same cue + outcome channels** — it re-presents
the two cues with flipped outcomes; there is deliberately **no separate reversal-signal channel**, so
the agent must infer the update from the new sensory evidence (the faithful analogue of the worm
re-learning from a changed contingency, not reading a flag). During the delay + response phases all of
cue/outcome are `0`.

### D3 — Minimal observation, no external memory aids (inherited crux)

As bit-memory: the observation is exactly the task channels (cue, outcome, go) — **no STAM, no
gradient sensing** — so only internal recurrent state can carry the association across the delay. The
config-resolve invariant (cue/outcome/go only; no STAM; oracle/none sensing) is asserted loudly; a
leaked cue invalidates the control.

### D4 — Binary action-sign readout of the remembered rewarded cue

Reuse bit-memory's readout: on the response step the binary response is `sign(turn)` (continuous arms)
/ `LEFT`/`RIGHT` (discrete), interpreted as the remembered rewarded **cue-identity** (`+1 → A`,
`-1 → B`). Correct iff it matches the trial's rewarded cue. No probe-cue channel is needed — the agent
recalls and emits the rewarded identity directly.

### D5 — Association-conditioned, response-phase-only reward; chance = 50%

Correct response rewarded (`reward_correct`), incorrect not/penalised (`penalty_wrong`), scored only
on response steps in the runner's post-action branch, against the trial's **current** rewarded cue
(post-reversal if the trial reversed, else the original). The initial rewarded cue and the reversal are
sampled uniformly/independently, so a chance policy scores 50%. Multi-trial episodes (as bit-memory)
give a robust per-episode accuracy; the analysis can additionally split accuracy by reversal vs
non-reversal trials to read the update demand directly.

### D6 — A dedicated task class + runner step, mirroring bit-memory's structure (not inline in env.py)

Bit-memory is a **dedicated `BitMemoryTask` class** in `env/bit_memory.py` (phase enum, `signals()`,
`advance()`, `record_response()`, `take_reward()`, `reset()`, `done()`, `num_responses()`), with the
env holding `self.bit_memory: BitMemoryTask | None` + a thin `get_bit_memory_signals()` getter
(`env/env.py:1085`), and the runner dispatching a **dedicated step** `_run_bit_memory_step`
(+ `_terminate_bit_memory`, `_bit_memory_turn`, `runners.py:696/651/624`) that is the *entire* step
when enabled — it never invokes the foraging/predator/thermal/satiety handlers or applies locomotion.
This change **mirrors that structure**: a parallel `AssociativeMemoryTask` in `env/associative_memory.py`
(with a two-cue conditioning phase + an outcome/valence signal), `self.associative_memory` +
`get_associative_signals()`, and a runner `_run_associative_memory_step`. An
`AssociativeMemoryTaskConfig` block gates it; the agent reads the getter into `BrainParams`
(`cue_signal`, new `outcome_signal`, `go_signal`) — the env never pushes into `BrainParams`. When
disabled (default) everything is byte-identical.

### D7 — The memory-relevant arm subset, matched entropy, cue/outcome/go sensing only

Run MLP (memoryless baseline) + LSTM / CfC / Transformer / minGRU / minLSTM. **Skip connectome** — 030
showed it is at chance on bit-memory (within-step settling, no cross-step memory), so it adds no signal
here. **Matched entropy** across arms (architecture the only variable); the Transformer is the
reliable detector (032). No new architecture code.

### D8 — Separation criterion and verdict

Separation = the memory arms clear both chance and the memoryless MLP on plateau-tail response
accuracy (paired-seed one-sided Wilcoxon + 80% bootstrap CI + BH-FDR, the committed layer). Reports
the verdict for `T7.separation.associative_memory`; a null is reported as such (a second naturalistic
data point on the "capability yes / natural demand limited" question).

## Risks / Trade-offs

- **Not a bit-memory re-skin (resolved by D1's reversal).** A static single association would be an
  isomorph of bit-memory; **probabilistic within-trial reversal** makes the demand
  *working-memory update* (overwrite a held association on new evidence), which bit-memory cannot test —
  so the probe earns its keep both biologically (worm re-learning) and as a new architecture
  discriminator (attention re-read vs recurrent overwrite). The residual binding computation (cue↔outcome)
  is the additional richness over a raw bit.
- **The task could still be "too easy" if all memory arms update perfectly** → the separation would look
  like 030. That is itself an informative outcome (update is not a discriminator here); the
  reversal-vs-non-reversal accuracy split (D5) surfaces *which* arms actually update, so a per-arm
  update deficit is visible even without an overall separation.
- **Recurrent-PPO instability** (032): treat the Transformer as the reliable detector; matched entropy;
  a learnability pre-check (a memory arm must clear the threshold on the easiest setting before the
  full panel — else debug the task, not the verdict).
- **Calibration** (conditioning/reversal/delay/response steps, `reversal_prob`, trials/episode, budget)
  kept within the Transformer window; calibrated before the panel (as bit-memory §6).

## Migration Plan

Additive + config-gated + off by default; no breaking changes, no new dependency, no new architecture.
Extends (does not modify) the bit-memory task capability.

## Open Questions

- Cue encoding: signed scalar (chosen, reuses the cue channel) vs two one-hot cue channels — revisit
  only if the signed scalar proves too easy/hard in calibration.
- Conditioning step count per cue (1 vs 2) and inter-cue gap — pinned in calibration.
- `reversal_prob` (default 0.5, which puts a hold-only policy at chance) — sweep in calibration; too
  low weakens the update signal, too high lets an "always-use-the-last-block" heuristic win.
- Whether a small `penalty_wrong` (vs 0) sharpens learning — settle in calibration.
- Explicit reversal-signal channel: deliberately **omitted** (the flipped outcomes are the update
  evidence); revisit only if inferring the reversal from outcomes proves unlearnable even for the
  Transformer at the easiest setting.
