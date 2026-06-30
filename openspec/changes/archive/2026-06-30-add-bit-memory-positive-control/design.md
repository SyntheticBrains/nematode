# Design: Bit-Memory Working-Memory Positive Control

## Context

The T4 ([Logbook 025](../../../docs/experiments/logbooks/025-weight-search-architecture-ranking.md)) and T7 ([Logbook 029](../../../docs/experiments/logbooks/029-continuous-architecture-ranking.md)) rankings both found the architecture comparison does not discriminate on the reactive cells — the gradient is locally readable with a one-step temporal derivative, so working memory is never exercised and the field ties (or MLP wins). Before committing to memory-capable new architectures (minGRU / modified-S5, [`T7.separation.new_arch_candidates`](../phase6-tracking/tasks.md)) or the biological area-restricted-search cell ([`T7.separation.ars_depletion`](../phase6-tracking/tasks.md)), this change builds the cheapest possible instrument to answer one question: **can the comparison separate working memory at all?**

The existing memory infrastructure does not answer this. Phase 3 temporal sensing ([Logbook 009](../../../docs/experiments/logbooks/009-temporal-sensing-evaluation.md)) and the [`short-term-associative-memory`](../../specs/short-term-associative-memory/spec.md) (STAM) module provide an exponential-decay recency buffer that is **fed into the observation vector** — which means a memoryless MLP can read recent history off its input and is never forced to use internal state. A clean positive control must do the opposite: withhold every external memory aid so the only way to solve the task is to retain information in internal recurrent state.

## Goals / Non-Goals

**Goals**

- A deliberately-artificial task that a memoryless policy provably cannot solve above chance, while a recurrent/attention policy can.
- Unambiguous attribution: a failure is a memory failure, never a navigation/perception failure.
- Reuse the existing 5 MUST arms and the existing training/episode plumbing — **no new architecture code**.
- A clear, statistically-tested **separation criterion** that produces a yes/no gate verdict.

**Non-Goals**

- Biological realism (this is explicitly artificial; the biological twin is `ars_depletion`).
- Changing the MUST integrated-C3 ranking or any Gate 3 criterion (non-gating).
- Bringing up new architectures (that is the conditional follow-on, gated on this control's verdict).
- A spatial / navigation task (rejected — see Decision 1).

## Decisions

### Decision 1 — Non-spatial delayed-match-to-cue (not spatial cued navigation)

The task is a pure memory instrument: per trial, a discrete cue is shown during a **cue phase**, withheld during a **delay phase**, and on a go-signalled **response phase** the agent must emit the action matching the cue. The cue is sampled uniformly per trial, so chance = 50%.

**Why over spatial cued navigation:** a positive control must isolate the variable it tests. In a spatial 2-goal task, a memoryless arm could fail because it cannot navigate, not because it cannot remember — confounding the very read-out the control exists to produce. The non-spatial form makes failure unambiguously a memory failure. It is also the standard working-memory probe in the RL literature (bsuite `memory_len`, POPGym repeat tasks, the Bakker 2002 T-maze) and is the cheapest to build and run. The tracker's "deliberately-artificial, non-biological" framing endorses this.

### Decision 2 — Minimal observation, no external memory aids (the crux)

The bit-memory observation is exactly two channels: a **cue** channel (the cue value during the cue phase, `0` otherwise) and a **go-signal** channel (`1` during the response phase, `0` otherwise). **No STAM, no gradient/klinotaxis sensing, no proprioceptive history.** This is the design property that makes the control valid: if STAM or any recency buffer were in the observation, a memoryless MLP could read the (decayed) cue back and the separation would collapse. Withholding them forces retention into internal recurrent state. The go-signal isolates *cue memory* from *timing/counting* — every arm is told when to respond, so the only thing that must be remembered is the cue value.

### Decision 3 — Multi-trial episodes; delay sized to the arms' reach

Each episode runs **N independent trials** (cue → delay → response), each with a freshly-sampled cue, rather than one trial per episode. Rationale: one binary decision per episode is too sparse a learning signal; N trials per episode densifies reward and is standard for memory benchmarks. Trials are independent (the cue does not carry across trials), so the task tests within-trial retention, not cross-trial interference.

The **delay length is a calibrated knob**, defaulted so the cue sits within every memory arm's reach so they *can* solve it (the point of a positive control). The binding constraint is the Transformer: it attends over a fixed `window_size = 16` window, so if the cue-to-response span exceeds 16 the Transformer cannot see the cue and would fail like the MLP — a confounded result. So the default per-trial span (`cue_steps + delay_steps + response_steps`) is kept comfortably under 16 (e.g. cue 1–2, delay 8–10, response 1). A **longer-delay variant (span > window)** that would separate unbounded-memory (LSTM/CfC) from windowed-attention (Transformer) is recorded as a deliberate follow-on, not part of the gate.

### Decision 4 — Binary cue encoding + action-sign readout

The cue is binary, encoded as `−1 / +1` on the cue channel during the cue phase. The response is read from the agent's action: for the continuous tanh-Gaussian arms (the 4 T7 arms), `sign(turn)` (turn ∈ `[−1, 1]`) gives the binary choice; for any discrete-action arm, `LEFT` vs `RIGHT`. This keeps the readout action-mode-agnostic and adds no new output head — the existing policy output IS the response. Movement is irrelevant in this task (Decision 6).

### Decision 5 — Cue-conditioned, response-phase-only reward

Reward is sparse and applied only on response steps: `+reward_correct` if the response action matches the cue, `−penalty_wrong` (or `0`) otherwise. Because the response is the brain's just-emitted action, this is scored in the runner's **post-action** branch (Decision 6), not as a pre-action reward term. The reward parameters live in `BitMemoryTaskConfig` (not `RewardConfig`), so the gating is self-contained. No shaping, no per-step movement reward (movement is inert). The cue-match success rate over response steps is the primary metric.

### Decision 6 — A config-gated mode on the existing environment (not a new env)

The task is realised as a **config-gated bit-memory mode** that reuses the existing environment + agent + runner stack, with the mechanics placed where the codebase actually performs them (verified against the agent/runner architecture — there is **no scoring `env.step()`**):

- **Phase machine + cue state — the environment.** `DynamicForagingEnvironment` / `Continuous2DEnvironment` (gated by `bit_memory_task.enabled`) owns the per-episode trial counter, samples the per-trial cue from the env RNG, advances `cue → delay → response`, and **exposes** the current cue/go state via a getter (e.g. `get_bit_memory_signals(agent_id)`).
- **Cue/go injection — the agent.** `BrainParams` is assembled in `agent._create_brain_params` (which already pulls per-agent env state, e.g. `predator_contact_zone`). That method reads the env's exposed cue/go state and populates the new `cue_signal` / `go_signal` fields. The env does **not** push into `BrainParams` — there is no such setter.
- **Response scoring + termination — the runner.** The response is the brain's just-emitted action (`top_action`), so it cannot be a pre-action reward term. The `StandardEpisodeRunner.run` loop, when the task is enabled, takes a **post-action branch**: on response steps it reads `sign(top_action.continuous[1])` (the turn), compares it to the trial cue, and applies `reward_correct` / wrong-response outcome; it drives episode-done off the **trial counter**, not `env.reached_goal()`.
- **Foraging deactivation — the runner + config.** The runner independently calls the predator / temperature / oxygen / starvation / food-collection / satiety handlers regardless of env config, so disabling those behaviours requires a runner-level early-branch (or env-config flags that make each handler a no-op) when the task is enabled — not merely an env setting. These exact touch-points are enumerated in tasks §3.
- **Per-run env recreation — carry the task.** The single-agent run loop recreates the env each episode via `agent.reset_environment()`, which rebuilds the env *without* the post-construction `bit_memory`; the recreation therefore re-attaches the task (rebinding its cue RNG to the new env's per-run-seeded RNG). Without this, only run 0 ran the task and later runs silently reverted to foraging — caught + fixed during bring-up.

**Why over a new standalone `BitMemoryEnvironment`:** the episode runner, agent loop, brain interface, PPO training, weight persistence, and experiment tracking are all coupled to the existing environment; a standalone env would need a parallel runner and re-wiring of that whole stack for a throwaway positive control. The mode reuses all of it and is removed cleanly by the off-by-default flag. The cost — gated branches in the env, the agent param builder, and the runner (cf. issue #206) — is accepted as bounded and config-isolated.

### Decision 7 — The existing 5 MUST arms, cue/go sensing only

The control runs the 5 MUST arms (`mlpppo`, `lstmppo`, `cfcppo`, `transformerppo`, `connectomeppo`) with `sensory_modules: [cue, go_signal]` (input_dim = 2 — each module registers `classical_dim = 1`, so the assembled observation is exactly 2-dim) and each arm's existing T7 action head. No new architecture code. The connectome is included even though its recurrence is **within-step settling** ("K recurrent updates" before motor pooling) rather than cross-step state — whether it retains a cue across the delay is itself an empirical question this control answers (prediction: connectome ≈ MLP, near chance, if it carries no cross-step state).

### Decision 8 — Separation criterion and the gate verdict

Primary metric: **cue-match success rate** per arm, n ≥ 8 paired seeds, reusing the committed paired-seed Wilcoxon + 80% bootstrap CI + BH-FDR layer ([`architecture-comparison-protocol`](../../specs/architecture-comparison-protocol/spec.md)). The control **fires (separation confirmed)** when the recurrent/attention arms (LSTM, CfC, Transformer) are **both** (a) well above chance (pre-registered threshold, e.g. ≥ 80%) **and** (b) significantly above the MLP (one-sided paired test, BH-FDR q < 0.05), with the **MLP at or near chance (~50%)**. A **null** (no arm beats chance, or the memory arms do not beat the MLP) is a strong finding in its own right — it means the comparison cannot resolve working memory — and **defers** `ars_depletion` and `new_arch_candidates`. The verdict is written into the tracker and the supporting analysis.

## Risks / Trade-offs

- **All arms fail (even LSTM at chance) → cannot tell "comparison can't resolve memory" from "task mis-built".** → Mitigation: a learnability pre-check — confirm at least one memory arm reaches the threshold on the easiest setting (short delay, generous budget) before running the full panel; if not, debug delay/budget/reward before declaring a null.
- **MLP scores above chance (info leak → control invalid).** → Would mean the cue leaks into the observation. The known leak path is STAM: `config_loader.apply_sensing_mode` appends `stam` to `sensory_modules` when `stam_enabled`, and `validate_sensing_config` **auto-enables STAM** under derivative/klinotaxis sensing modes — which would feed a recency buffer holding the (decayed) cue back to a memoryless arm. Mitigation (a **hard assertion, not prose**): when `bit_memory_task.enabled`, assert the resolved sensing mode is oracle/none, assert `stam` is absent from the resolved module list, and assert the assembled observation is exactly 2-dim (cue + go only); plus a test that the cue channel is exactly `0` on every delay/response step. The MLP-at-chance result is the validity canary.
- **Transformer fails because the span exceeds its window.** → Confounded (looks like a memory failure but is an architecture-config artifact). Mitigation: default the span < `window_size` (Decision 3); record the window/span relationship; the >window case is an explicit separate variant, not the gate.
- **Sparse reward → slow/instable training.** → Mitigation: multi-trial episodes (Decision 3) densify the signal; per-arm time-boxed tuning is fair for a positive control (compare arms at their best, as the C1/C2 convergence protocol does); adequate budget.
- **Mode bolted onto the large env class.** → Mitigation: fully config-gated and off by default; isolated tests for the phase machine + cue withholding; no behaviour change when disabled.

## Migration Plan

Config-gated and off by default — no migration. Every existing config and code path is unchanged when `bit_memory_task.enabled` is false (the default). Rollback is disabling the flag or reverting the additive modules. The new cue/go sensory modules and `BitMemoryTaskConfig` are additive; no existing config requires changes.

## Open Questions

- Exact step counts (`cue_steps`, `delay_steps`, `response_steps`, trials-per-episode) and the success threshold — calibrated empirically in tasks.md against the learnability pre-check, kept within the Transformer window.
- `n` for the panel: n ≥ 8 to mirror the ranking, or fewer (this is a positive control, not the ranking) — defaulting to n ≥ 8 for statistical consistency.
- Whether the connectome retains any cross-step state (empirical; informs how to read its result).
- Whether to ship the longer-delay (> window) unbounded-memory-vs-attention variant now or defer it to the `new_arch_candidates` round.

**Resolved during implementation (2026-06-30, [Logbook 030](../../../docs/experiments/logbooks/030-bit-memory-positive-control.md)):** calibration landed at `cue 2 / delay 8 / response 1` (span 11, within the Transformer window 16), 20 trials/episode, 1500 episodes, success threshold 0.80; the panel ran at **n = 8** (an n = 4 read showed the same separation in the means but a false NULL — the n = 4 one-sided Wilcoxon floor is 0.0625 — so n ≥ 8 is required). The connectome **does not** retain cross-step state (it sits at chance, indistinguishable from the MLP). The longer-delay variant is **deferred** to the `new_arch_candidates` round.
