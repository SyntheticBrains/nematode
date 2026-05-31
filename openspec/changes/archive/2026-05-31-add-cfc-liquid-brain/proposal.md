# Proposal: Add CfC / Liquid Brain (`CfCPPOBrain`)

## Why

The Phase 6 cross-architecture comparison ranks brain architectures on an integrated foraging + predator-evasion + thermotaxis cell. After the four MUST architectures landed (connectome, MLP-PPO, LSTM/GRU-PPO, GA-weights), the architecture-promotion gate selected additional families to broaden the comparison. A **liquid / continuous-time** arm is the highest-value addition:

- **Closed-form Continuous-time networks (CfC) with Neural Circuit Policy (NCP) wiring** (Hasani, Lechner et al.) are continuous-time recurrent networks whose design is directly inspired by the *C. elegans* tap-withdrawal circuit. On a nematode task they are a uniquely apt scientific foil to the **wild-type connectome** arm: a *C. elegans-inspired* sparse, biologically-structured policy (sensory → interneuron → command → motor) versus the real connectome and versus generic recurrence (LSTM/GRU).
- A skeptical literature survey found CfC/NCP is the one promoted family with **strong, direct closed-loop control evidence** (19-neuron lane-keeping, *Nature Machine Intelligence* 2020; drone fly-to-target with state-of-the-art out-of-distribution generalization, *Science Robotics* 2023), a mature PyTorch library (`ncps`), and the best compute-to-evidence ratio (~1.5–3× a 2-layer MLP on CPU).

This change ships **only** the `CfCPPOBrain` and its registry integration. The cross-architecture comparison consumes it downstream as the liquid/continuous-time row (combined-cell config + ranked evaluation live in the comparison work, not here).

## What Changes

- **New brain** `CfCPPOBrain` (`name: cfcppo`, `brain_type: CFC_PPO`, `families: ("classical",)`): a CfC recurrent core with an `AutoNCP` wiring; by default the **motor neurons are the action logits** (NCP-authentic, no separate actor head; a learnable logit temperature keeps the bounded motor outputs usable as a decisive policy), with a configurable `actor_head: "mlp"` fallback that reads the hidden state through a small actor MLP for added policy capacity. A critic MLP on the recurrent hidden state; trained with PPO over truncated-BPTT sequence chunks. Mirrors the existing `LSTMPPOBrain` PPO machinery (rollout buffer, GAE, clip objective, weight persistence), adapted to CfC's single hidden state.
- **New dependency** `ncps` (the official Hasani/Lechner library: PyTorch `CfC` / `LTC` cells + `wirings`).
- **Registry integration**: `CFC_PPO` `BrainType`, `@register_brain` decorator, module import in `brain/arch/__init__.py`; verified to load through the existing plugin registry without per-architecture launcher branches.
- **Smoke config** `configs/scenarios/foraging/cfcppo_small_klinotaxis.yml` proving the brain trains end-to-end on klinotaxis foraging.
- **Tests** under `tests/quantumnematode_tests/brain/arch/test_cfcppo.py`.

## Impact

- **New dependency**: `ncps` (added to `packages/quantum-nematode/pyproject.toml`). No other deps; `torch` already present.
- **Reuses**: the `LSTMPPOBrain` PPO + truncated-BPTT pattern, the `@register_brain` plugin registry, the `WeightPersistence` protocol, the classical sensory-feature pipeline (`extract_classical_features`).
- **Out of scope** (possible follow-ups): an `ENCODER_REGISTRY` entry for evolving CfC weights (this brain is PPO-trained, not evolved, in the comparison); a hand-authored connectome-derived `ncps` wiring (the auto-generated `AutoNCP` wiring is used here); the combined-behaviour C3 cell config and the ranked cross-architecture evaluation (downstream comparison work).
