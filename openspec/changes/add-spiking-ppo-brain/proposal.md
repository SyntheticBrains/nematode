# Proposal: Add Spiking Brain (`SpikingPPOBrain`)

## Why

The Phase 6 cross-architecture comparison ranks brain architectures on an integrated foraging +
predator-evasion + thermotaxis cell. The architecture-promotion gate selected a **spiking /
neuromorphic** arm to broaden the comparison: alongside the wild-type connectome and the
*C. elegans*-inspired CfC liquid brain, a spiking neural network rounds out the biological-plausibility
axis — the connectome answers "does real wiring help?", CfC "does continuous-time *C. elegans*-style
structure help?", and spiking "does event-driven LIF dynamics + surrogate-gradient learning help?".

A prior spiking brain (`spikingreinforce`) exists but is **paradigm-incompatible** with this comparison
and is not reused: it is REINFORCE (no critic / GAE / clipped surrogate, vs the comparison's PPO arms)
and consumes a fixed 4-D input (`food_strength`, `food_angle`, `predator_strength`, `predator_angle`)
that is structurally blind to thermotaxis, klinotaxis lateral gradients, and short-term memory — so it
cannot represent the C3 sensory stack. Reusing it would measure old code on an incompatible substrate.

A pre-build investigation (literature survey + codebase forensics) settled the recipe. The decisive
correction over a naive feedforward-LIF port: the core is a **recurrent
adaptive-LIF** that carries membrane state across env-steps at one tick per step (GRSN, AAAI 2025) —
the cell is partially observable and the comparison's leaders (LSTM / CfC) win on memory, so a
memoryless feedforward LIF would be structurally handicapped. The investigation also records an honest
expectation: a spiking arm trained **on-policy with PPO** is most likely **mid-pack** (the on-policy
constraint, not the spiking paradigm, caps the ceiling — every 2024–2026 result where an SNN matches an
ANN in control uses off-policy/offline RL). It is built for comparison completeness and the
biological-plausibility narrative, pre-registering "competitive mid-pack = success".

This change ships **only** the `SpikingPPOBrain` and its registry integration. The combined-cell C3
config and the ranked evaluation live downstream in the comparison, not here.

## What Changes

- **New brain** `SpikingPPOBrain` (`name: spikingppo`, `brain_type: SPIKING_PPO`, `families: ("spiking",)`):
  a **recurrent adaptive leaky-integrate-and-fire (LIF)** core — learnable membrane decay, soft reset, a
  spike-feedback recurrent current, and the in-repo sigmoid-family surrogate gradient — that carries its
  membrane state across env-steps (one LIF tick per step). A learnable direct-current input encoder feeds the
  core; a non-spiking leaky-integrator readout produces the action logits (the **spiking actor**); a
  plain-ANN critic estimates value from the detached membrane state. Trained with PPO over
  truncated-BPTT sequence chunks, mirroring the `CfCPPOBrain` machinery (single recurrent state). The
  entropy-coefficient anneal proven on the LSTM / CfC arms is wired in from the start (stateful arms hit
  init-variance bimodality on this cell).
- **No new dependency.** The brain **extends the existing, fully-tested spiking substrate**
  (`brain/arch/_spiking_layers.py`: `SurrogateGradientSpike`, `LIFLayer`, `PopulationEncoder`) with
  learnable decay + recurrent dynamics, rather than adding `snnTorch`. `torch` is already present.
- **Registry integration**: `SPIKING_PPO` `BrainType`, `@register_brain` decorator, module import in
  `brain/arch/__init__.py`; loads through the existing plugin registry without launcher branches.
- **Smoke config** `configs/scenarios/foraging/spikingppo_small_klinotaxis.yml` proving the brain trains
  end-to-end on klinotaxis foraging.
- **Tests** under `tests/quantumnematode_tests/brain/arch/test_spikingppo.py`.

## Impact

- **No new dependency**: reuses the in-repo `_spiking_layers.py` substrate (no `snnTorch` / `norse` /
  `spikingjelly`). Total control of the surrogate + the carried-state truncated-BPTT loop.
- **Reuses**: the `CfCPPOBrain` PPO + truncated-BPTT pattern (single hidden state), the `@register_brain`
  plugin registry, the `WeightPersistence` protocol, the classical sensory-feature pipeline
  (`extract_classical_features`), and the `entropy_coef_end` / `entropy_decay_episodes` anneal hook.
- **Out of scope** (possible follow-ups): full GRSN-style gating of the recurrent current (this cut uses
  a learnable-decay + adaptive-threshold + recurrent-current LIF without GRU-style gates); population-coded
  input (held in reserve — direct-current is the first cut); a reward-driven surrogate-slope schedule
  (an optional episode-based slope schedule is included; the reward-driven variant is a follow-up); an
  `ENCODER_REGISTRY` entry for evolving spiking weights (this brain is PPO-trained, not evolved); the
  combined-behaviour C3 cell config and the ranked cross-architecture evaluation (downstream comparison).
