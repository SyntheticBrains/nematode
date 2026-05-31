# Design: Add CfC / Liquid Brain (`CfCPPOBrain`)

## Context

- **`LSTMPPOBrain`** ([`brain/arch/lstmppo.py`](../../../packages/quantum-nematode/quantumnematode/brain/arch/lstmppo.py)) is the closest existing analog: a recurrent PPO brain with a rollout buffer (GAE + truncated-BPTT over sequential chunks), LayerNorm→recurrent→actor/critic, `WeightPersistence` conformance, and `@register_brain` integration. This change mirrors its PPO machinery and adapts the recurrent core.
- **Plugin registry** ([`brain/arch/_registry.py`](../../../packages/quantum-nematode/quantumnematode/brain/arch/_registry.py)): brains self-register via `@register_brain`; the launcher consumes them without per-architecture branches. `feedforward_ga.py` is the minimal reference for a new registered brain.
- **`ncps`** (added dependency, v1.0.1) provides `ncps.torch.CfC` and `ncps.wirings.AutoNCP`. **De-risked on torch 2.7** before this design:
  - `wiring = AutoNCP(units, num_actions)` builds a sparse sensory→inter→command→motor graph (e.g. `AutoNCP(32, 4)` → 32 units, 116 synapses).
  - `cfc = CfC(input_dim, wiring)`; sequence `cfc(x_(B,T,input_dim), h0_(B,units)) → out_(B,T,num_actions), h_(B,units)`; single-step `cfc(x_(1,1,input_dim), h_(1,units)) → out_(1,1,num_actions), h_new_(1,units)`. The output dimensionality equals the AutoNCP motor-neuron count.

## Decision 1 — CfC core with `AutoNCP` wiring; configurable actor head (motor-direct by default)

The recurrent core is always `CfC(input_dim, AutoNCP(units, num_actions))` — the structured sensory → interneuron → command → motor wiring is the scientific point of this arm and is used in every variant. The **actor head** is selected by an `actor_head` config field:

- **`"motor"` (default, NCP-authentic)**: the CfC output — the `num_actions` motor neurons of the NCP wiring — is used **directly as the actor logits**, with no separate actor MLP. In the *C. elegans* circuit the motor neurons drive action; using them as the policy head is what makes this a genuine "C. elegans-structured policy" rather than "a CfC feature extractor + generic head."
- **`"mlp"` (capacity fallback)**: a small actor MLP maps the full recurrent hidden state (`units`-dim) to the action logits, ignoring the motor-neuron output. The structured CfC recurrence is unchanged; only the readout gains capacity.

**Motor-head logit temperature (load-bearing).** The CfC's NCP motor neurons saturate to a bounded range — empirically ~[−0.8, +0.8] even under extreme inputs. Used raw as logits, that caps the softmax at ~60% probability on any action, structurally preventing a decisive policy (a real handicap when, e.g., fleeing a predator needs a sharp choice). In `"motor"` mode the brain therefore applies a **learnable scalar temperature** before the softmax (`logits = motor_output * logit_scale`, a trained `nn.Parameter` in the actor optimizer, initialised to `motor_logit_scale_init` ≈ 1.0). This preserves the NCP-authentic design (the motor neurons still decide *which* action) while letting the policy learn *how decisively* — without it, a poor `"motor"` ranking could be a logit-saturation artifact rather than an architectural result, defeating the purpose of the arm. The `"mlp"` head reads the unbounded hidden state and needs no temperature.

**Why configurable rather than committing to motor-direct?** Motor-as-actor is the literature-validated, authentic NCP usage (the motor neurons were the steering output in the 19-neuron lane-keeping result), but it gives the policy a *constrained, sparse* readout. On the hard integrated cell that could under-power the policy and risk mis-ranking the architecture for an implementation reason rather than an architectural one. The config flag lets the comparison start authentic and, if the foraging smoke or the integrated cell underperforms, flip to the MLP head with a one-line change (and report both — "does the NCP-authentic head cost performance?" is itself a result worth having) rather than a re-implementation. The `units` count is the other policy-capacity lever inside the structure (more command neurons → richer motor drive). The critic always reads the full hidden state (Decision 2), in both head modes.

**Why CfC, not LTC?** Both are in `ncps`. LTC numerically solves an ODE per step (3–10× slower, needs solver unfolding); CfC is the closed-form approximation that skips the solver (~1.5–3× MLP cost). For a CPU-bound 500–1000-step comparison, CfC is the right speed/fidelity trade. (`cfc_mode` is exposed as a config knob — `"default"` by default; `"pure"` selects the closed-form ODE with no gating MLP if a more compact/interpretable variant is wanted later.)

**Why `AutoNCP`, not a plain `units` integer (fully-connected CfC)?** The connectome-structured sparse wiring is the differentiator from the existing LSTM/GRU arm; a fully-connected CfC would just be "another dense RNN." (A hand-authored wiring mirroring a specific connectome subgraph is a compelling follow-up but out of scope — the auto-generated NCP wiring is the standard first cut.)

## Decision 2 — Critic MLP on the detached recurrent hidden state

The critic is a small MLP (`Linear(units → critic_hidden_dim) → ReLU → … → Linear(→ 1)`) applied to the CfC hidden state. The hidden state is **detached** before the critic (mirroring `LSTMPPOBrain`'s rationale: prevent value-loss gradients from distorting the recurrent representation). Actor and critic use separate Adam optimizers (actor optimizer covers the CfC core + input LayerNorm; critic optimizer covers the critic MLP).

## Decision 3 — Mirror the `LSTMPPOBrain` PPO machinery (do not refactor a shared base)

The rollout buffer (states, actions, log-probs, values, rewards, dones, per-step hidden state), GAE return/advantage computation, truncated-BPTT replay over sequential chunks (`bptt_chunk_length`), and the PPO clip objective (+ entropy bonus + value loss, `max_grad_norm` clip, `num_epochs`) are mirrored from `LSTMPPOBrain`. **Why mirror rather than extract a shared recurrent-PPO base?** Cleanly factoring the PPO machinery out of `LSTMPPOBrain` (which carries LSTM/GRU duality, a LayerNorm cell, and a transgenerational-bias path) into a reusable base is a larger refactor of a tested brain — out of scope for adding one new arm. Mirroring keeps this change self-contained; a shared base is a separate follow-up if a third recurrent brain arrives.

## Decision 4 — Single hidden state; drop the LSTM/TEI/LayerNorm-cell/LR-schedule baggage

CfC is single-state (one hidden tensor, no LSTM cell state) — so the `(h, c)` duality threaded through `LSTMPPOBrain` collapses to a single `h` in the buffer, `run_brain`, and the PPO replay. The new brain deliberately **omits** three `LSTMPPOBrain` features that are not needed: the transgenerational-prior bias path (`tei_prior`), the custom LayerNorm recurrent cell (CfC's continuous-time dynamics are inherently well-behaved; no recurrent-saturation fix is needed), and the warmup/decay LR schedule (flat `actor_lr` / `critic_lr`, matching the simplest stable configuration the LSTM arm converged to). Input features still pass through an `nn.LayerNorm(input_dim)` before the CfC.

## Decision 5 — Depend on `ncps`, do not reimplement CfC

`ncps` is the official Hasani/Lechner reference implementation (the same library behind the published lane-keeping / drone results). Reimplementing CfC + NCP wirings in-repo would duplicate a maintained library and risk divergence from the reference dynamics. The dependency is small and pure-Python-over-torch.

## Decision 6 — `WeightPersistence` yes; evolution `ENCODER_REGISTRY` entry no

`CfCPPOBrain` implements `WeightPersistence` (`get_weight_components` / `load_weight_components`) covering `cfc`, `critic`, `feature_norm`, the two optimizers, and a training-state counter — so checkpoints round-trip like the other PPO brains. It does **not** register a `CfCEncoder` in the evolution `ENCODER_REGISTRY`: this brain is PPO-trained (gradient) in the comparison, not evolved. Adding an encoder is a clean follow-up if a GA/CMA-ES-evolved CfC arm is ever wanted.

## Risks

| Risk | Mitigation |
|---|---|
| CfC fails to learn the hard integrated cell (like the LSTM arm needed stabilization) | This change ships only the brain + a **foraging** smoke (the easy cell). The combined-cell de-risk + any tuning happens in the downstream comparison, with a documented halt option if it can't learn. |
| `ncps` API / version drift | Pinned (`ncps==1.0.1`); the de-risked API (Context above) is exercised by the brain's unit tests. |
| AutoNCP wiring constraints (motor count, minimum units) | `AutoNCP(units, num_actions)` ties motor neurons to the action set by construction (a test asserts the output shape equals `num_actions`) and requires `units > num_actions + 2` (else it raises `ValueError`); the brain validates `units` at construction with a clear error message (default `units=32` satisfies it). |
| Motor-head logit saturation | Mitigated by the learnable logit temperature (Decision 1); a test asserts the motor head can produce a peaked policy (one action probability > 0.9) after the scale is raised. |
