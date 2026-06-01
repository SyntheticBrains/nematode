# Design: Add Spiking Brain (`SpikingPPOBrain`)

## Context

- **`CfCPPOBrain`** ([`brain/arch/cfc_ppo.py`](../../../packages/quantum-nematode/quantumnematode/brain/arch/cfc_ppo.py))
  is the closest existing analog and the template: a recurrent PPO brain with a **single** carried
  hidden state (no LSTM `(h, c)` duality), a per-step-state rollout buffer, GAE + truncated-BPTT over
  sequential chunks, a critic MLP on the detached hidden state, separate actor/critic optimizers,
  `WeightPersistence` conformance, an `entropy_coef` anneal hook, and `@register_brain` integration. A
  LIF membrane state is single-state just like the CfC hidden state, so this change mirrors CfC's
  machinery almost verbatim and swaps only the recurrent core.
- **Spiking substrate** ([`brain/arch/_spiking_layers.py`](../../../packages/quantum-nematode/quantumnematode/brain/arch/_spiking_layers.py)):
  `SurrogateGradientSpike` (custom sigmoid-derivative surrogate autograd fn, slope α), `LIFLayer` (Euler-integrated
  LIF over an `nn.Linear`), and `PopulationEncoder` are fully tested and reused. This change adds the
  recurrent + adaptive dynamics on top rather than taking an SNN-library dependency.
- **`spikingreinforce`** ([`brain/arch/spikingreinforce.py`](../../../packages/quantum-nematode/quantumnematode/brain/arch/spikingreinforce.py))
  is **not** reused: REINFORCE (no critic/GAE/clip) over a fixed 4-D input blind to thermotaxis /
  klinotaxis / short-term memory. It stays as the legacy `spikingreinforce` arm; the new brain is a
  separate registration.
- The recipe is settled by a pre-build investigation (literature survey + codebase forensics; the key
  papers are cited in the decisions below).

## Decision 1 — Recurrent adaptive-LIF core, one tick per env-step (carried membrane state)

The core is a **recurrent adaptive leaky-integrate-and-fire** layer that carries its state across
env-steps (one LIF update per env-step, `timesteps_per_step: int = 1`), not a feedforward LIF unrolled
over many inner ticks. Per step, for the hidden layer (`hidden_size` units):

- input current `I = W_in · LayerNorm(features) + W_rec · s_prev` — a learnable **recurrent
  spike-feedback** current (`W_rec`, the spiking-RNN connection) added to the encoded input current;
- membrane `v ← β ⊙ v + (1 − β) ⊙ I`, with a **learnable per-neuron decay** `β = σ(raw_β)`;
- adaptive threshold `θ = v_threshold + adapt_scale ⊙ a` (spike-frequency adaptation), spike
  `s = SurrogateGradientSpike(v − θ)`, **soft reset** `v ← v − s ⊙ θ`, adaptation `a ← ρ ⊙ a + s`.

**Why recurrent, not feedforward?** The C3 cell is partially observable and the comparison's leaders
(LSTM, CfC) win *because* they carry memory; a memoryless feedforward LIF would be structurally
handicapped (≈ MLP-tier). **Why adaptive (learnable decay + adaptation)?** The 2024–2026 POMDP-SNN
literature (GRSN, AAAI 2025) converged on gated/adaptive recurrent spiking neurons that match GRU on
partially-observable tasks with the *lowest* cross-seed variance and no training collapse — directly
relevant to this cell's known init-variance fragility. **Why one tick per step?** GRSN and SpikeGym
both run one spiking update per RL transition (state carried, like an RNN) and lose no learnability;
it keeps cost at ~1.5–2.5× MLP (LSTM/CfC territory) so the n-seed sweep fits in hours, versus 4–8×
for inner-tick unrolling. `timesteps_per_step` is exposed (default 1; 2 is the only cheap bump); for
`timesteps_per_step > 1` the encoded input current is held constant across the inner ticks while the
membrane/adaptation integrate, and the recurrent spike-feedback advances once per env-step.

**Scope of "adaptive" here.** This first cut implements learnable decay + adaptive threshold + a
recurrent spike-feedback current — *not* full GRSN-style GRU gating of the recurrent current. The
gating is a refinement whose marginal value is unproven under on-policy PPO (GRSN's parity result is
off-policy); it is an out-of-scope follow-up. The carried-state recurrence and learnable decay are the
load-bearing parts.

## Decision 2 — Spiking actor with a non-spiking leaky-integrator readout

The hidden layer is the **spiking actor**. A **non-spiking leaky-integrator** output layer integrates
the hidden spikes into a carried output membrane `m ← β_out ⊙ m + W_out · s`, and the action logits are
that output membrane (smooth membrane gives better policy gradients than binary spikes). This is the
standard SNN RL/classification readout (leaky-integrator output neurons). The deployed policy is the
spiking actor; the readout is a linear integrator — this is the field-standard definition of a spiking
RL agent. The first cut is **membrane-only** (no `output_mode` knob): a spike-rate readout is degenerate
at the default one-tick-per-step and would conflict with the non-spiking readout, so it is a follow-up.

## Decision 3 — Plain-ANN critic on the detached membrane state

The critic is a small MLP (`Linear(hidden_size → critic_hidden_dim) → ReLU → … → Linear(→ 1)`) over the
hidden layer's **detached** membrane potential `v` (mirrors CfC/LSTM: keep value-loss gradients out of
the recurrent representation). The critic is deliberately **non-spiking** — value regression wants
smooth, accurate gradients, and a spiking critic adds variance for no benefit on a CPU comparison
(PopSAN / the shallow-slope drone work both pair a spiking actor with a non-spiking critic). Actor and
critic use separate Adam optimizers.

## Decision 4 — Direct-current input encoding (population coding held in reserve)

The ~8-dim continuous sensory vector is encoded as **learnable direct current** (`W_in`, the linear
encoder above), not Poisson rate coding. Direct current preserves the continuous gradient signal every
food/predator/thermotaxis channel carries; Poisson coding injects sampling noise and needs large `T` to
average out — the opposite of the wall-time budget. `PopulationEncoder` (Gaussian tuning curves) is the
documented fallback if the actor cannot represent the gradients in de-risk — a cheap swap with the best
control pedigree, but it multiplies input dimensionality, so it is not the first cut.

## Decision 5 — Sigmoid-family surrogate, shallow slope, optional episode-based slope schedule

Spikes are non-differentiable; training uses the reused `SurrogateGradientSpike` (the in-repo
sigmoid-derivative surrogate, slope α) with a **shallow** slope `surrogate_slope: float = 2.0` (passed as
α at each forward so it is runtime-schedulable; a Zenke fast-sigmoid is an equivalent drop-in if ever
wanted). A shallow slope trains better in RL and is a direct
mitigation for the old spiking brain's early gradient explosion. An **optional** episode-based schedule
sharpens the slope as training progresses (`surrogate_slope_end` + `surrogate_slope_anneal_episodes`,
both `None` → flat) — shallow while exploring, sharper while fine-tuning. The two schedule fields are
validated as a **pair** (set both or neither) so a half-set config fails fast rather than silently
running flat — the same paired-field validation the CfC entropy schedule uses. (The reward-driven slope
variant from the literature is a follow-up; episode-based matches our entropy-anneal convention.)

## Decision 6 — Entropy-coefficient anneal wired in from day one

Stateful arms hit init-variance bimodality on this cell (~37% of seeds collapsed to 0% on LSTM/CfC at
flat low entropy; the proven fix is a high→low entropy anneal). A recurrent SNN is a stateful arm, so
the brain reuses the CfC/LSTM hook verbatim — `entropy_coef`, optional `entropy_coef_end`,
`entropy_decay_episodes`, with the same paired-field validation and the same `_get_entropy_coef()`
linear schedule — so the downstream C3 config can stabilize without a code change.

## Decision 7 — Extend the in-repo substrate; no SNN-library dependency

We extend the tested `_spiking_layers.py` primitives rather than add `snnTorch`. Rationale: we already
own a tested LIF + surrogate; the recurrent-adaptive dynamics are a contained amount of new PyTorch; we
keep **total control of the carried-state truncated-BPTT loop** (the integration hotspot, where a
library's recurrent-neuron abstraction can fight a custom BPTT replay); and we avoid a dependency whose
release cadence has slowed. The build investment is matched to the realistic ceiling (Decision 9): a
heavier library/architecture would not raise a PPO-bound ceiling. If de-risk shows the simplified
neuron is the bottleneck, gating or `snnTorch` can be swapped into the core **without** touching the
wrapper, buffer, or registration.

## Decision 8 — Mirror CfC PPO machinery; WeightPersistence yes, evolution encoder no

The rollout buffer (per-step neuron state), GAE, truncated-BPTT replay over `bptt_chunk_length` chunks,
and the PPO clip objective (+ entropy bonus + value loss, `max_grad_norm`, `num_epochs`) are mirrored
from `CfCPPOBrain`. The per-step **neuron state** (hidden `v`, adaptation `a`, last spikes `s`, output
membrane `m`) is the single carried state stored per step and detached at chunk boundaries, exactly as
CfC stores its single `h`. The brain implements `WeightPersistence` (input encoder, `W_rec`, learnable
decay/adaptation params, readout, critic, both optimizers, training-state counter) so checkpoints
round-trip; the per-step neuron state is **not** persisted (reset at `prepare_episode()`). It does
**not** register an evolution `ENCODER_REGISTRY` entry (PPO-trained, not evolved).

## Decision 9 — Honest ceiling: mid-pack under PPO (pre-registered)

The investigation records the realistic expectation: an SNN trained **on-policy with PPO** is most
likely **mid-pack** (low-to-high 70s%), the recurrent core pushing toward the top of that band; tying
LSTM/CfC (~84%) is the optimistic tail. The binding constraint is on-policy PPO itself, not the spiking
paradigm — held for a fair comparison. The arm is built for comparison completeness and the
biological-plausibility narrative, pre-registering "competitive mid-pack = success". The de-risk gate
(below) is the explicit off-ramp.

## Risks

| Risk | Mitigation |
|---|---|
| Spiking arm cannot learn the hard integrated cell | This change ships only the brain + a **foraging** smoke (easy cell). The combined-cell de-risk (C1 foraging → single-seed C3 → **HALT** if it can't learn) happens downstream, before any n-seed spend. |
| Init-variance bimodality on C3 (the LSTM/CfC failure mode) | Entropy anneal wired in from day one (Decision 6); adaptive-LIF recurrence is reported as *lower*-variance than vanilla recurrence (GRSN). |
| Hand-rolled recurrent neuron correctness (no vetted library) | Simple, well-understood LIF math; unit tests on the dynamics (decay, soft reset, adaptation, recurrence, surrogate gradient flow); CfC already proved the carried-state truncated-BPTT plumbing. Swap-in path to `snnTorch`/gating preserved if needed. |
| Early gradient explosion (hit by the old spiking brain) | Shallow surrogate slope + optional sharpening schedule (Decision 5); `max_grad_norm` clip in the PPO update; one tick per step shortens the BPTT chain. |
| Dead / vanishing spikes (silent neurons → zero gradient) | Soft reset; LIF decay + threshold init so neurons sit near firing; a test asserts non-degenerate logit variance over a forward-pass sample. |
| Mid-pack ceiling read as failure | Pre-registered as success criterion (Decision 9); de-risk gate is the off-ramp if it under-learns. |
