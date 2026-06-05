## Why

Phase 6 Tranche 5 is the platform refactor that moves the architecture-comparison sweep off the discrete grid onto a continuous-2D substrate with continuous-action control — the substrate the real-worm behavioural validation (T7) needs and the field's plate-arena geometry expects. It is also the tranche that closes **Gate 2** (L1 plugin parity in practice): adding a new architecture family must touch ≤ 6 files with no per-architecture branches in the simulation/training loops. T5 is deliberately scoped to *platform refactor + parity verification only*; the Rung-2 env-fidelity work (adaptive chemosensory sensor + static Fick gradient geometry) is T6, so Gate 2 closes against a single verifiable platform outcome rather than a bundled env-upgrade tranche (per `phase6-tracking` Decision 1).

The T4 discrete grid substrate stays intact and untouched, so the T4-vs-T7 env-upgrade delta (RQ5) has a stable baseline to compare against.

## What Changes

- **New continuous-2D environment**, selectable alongside the existing grid env via a config discriminator (the env is currently hard-coded). Float positions, kinematic movement (heading + step length), realistic scales (~1 mm worm body on a cm-scale plate). Native sinusoidal undulation, omega turns, and pirouettes remain out of scope (c302/Sibernetic export path covers those if ever needed).
- **Capture-radius food semantics** replacing exact grid-cell-equality consumption; continuous (uniform-over-bounds) source placement replacing integer sampling; Euclidean distances replacing the grid-idiomatic Manhattan ones (pheromone, predator-mechano, nearest-food); physically-scaled klinotaxis lateral head-sweep amplitude replacing the fixed ±1-cell offset.
- **Shared policy module** providing both discrete (Categorical) and continuous (**tanh-squashed Gaussian**, with log-det-Jacobian correction) action sampling, log-prob, and entropy. The four MUST PPO brains (MLP, LSTM, CfC, connectome) are migrated onto it. The existing discrete path is unified into the same module under a documented migration-regression bar (byte-equivalence or declared seeded-RNG tolerance on a smoke config per brain, per the T2/Phase-5 precedent).
- **Continuous action contract**: a Protocol-level extension carrying a `(speed, turn)` vector that the agent/simulation loop consumes generically — no per-architecture branching. This is the principal Gate-2 parity risk and the load-bearing design surface.
- **Continuous-output adapter for the connectome brain**: the motor-pool readout gains a `4 → (mean, log_std)` head; the chemical-synapse strict-mask and fixed gap junctions are preserved unchanged (they are upstream of the readout and orthogonal to output type — the roadmap's feared "strict-mask incompatible with continuous output" failure mode does not hold).
- **Transformer / attention architecture** added through the plugin interface as the Gate-2 plugin-parity verification vehicle (files-touched count + no-per-architecture-branches code review). Doubles as the roadmap's optional MAY transformer deliverable (a discrete-head comparison row on the grid substrate; a continuous head for the continuous-2D substrate is opportunistic, not required for Gate 2).
- **Gate 2 decision** recorded in the T5 logbook against the pre-registered G2.a–G2.d criteria.
- SHOULD/MAY architectures (quantum, spiking, reservoir, hybrid) get continuous heads opportunistically; not gating Gate 2.

## Capabilities

### New Capabilities

- `continuous-2d-environment`: continuous-2D coordinate substrate — float agent position, kinematic (heading + step-length) movement, capture-radius food consumption, continuous source placement, Euclidean field distances, physically-scaled klinotaxis sweep — selectable alongside the grid env, with the grid env unchanged.
- `continuous-action-policy`: shared action-policy module exposing discrete (Categorical) and continuous (tanh-squashed Gaussian) sampling/log-prob/entropy behind one interface, plus the continuous action contract the agent consumes generically.
- `transformer-brain`: a Transformer/attention PPO architecture registered through the plugin interface, serving as the Gate-2 plugin-parity verification and an optional comparison row.

### Modified Capabilities

- `brain-architecture`: registered PPO-family brains gain continuous-action heads via the shared policy module; the discrete path migrates to it under a regression bar; the plugin-parity verification (≤ 6 files, no per-architecture branches) becomes an enforced requirement.
- `connectome-ppo-brain`: motor-pool readout gains a continuous-output (Gaussian) adapter; strict-mask + fixed gap junctions preserved.
- `environment-simulation`: an environment-type switch is introduced; the continuous-2D environment registers under it; capture-radius and continuous-placement semantics are specified for the continuous env.
- `configuration-system`: env-type discriminator + continuous-environment fields (world bounds, body length scale, step length, capture radius, sweep amplitude) + continuous-action config fields.

## Impact

- **Code**: `brain/actions.py` (action contract); new shared policy module under `brain/arch/`; `brain/arch/{mlpppo,lstmppo,cfc_ppo,connectome_ppo}.py` (head + sampling + PPO-update migration); `env/env.py` (new `Continuous2DEnvironment` subclass + `EnvironmentType` generalisation); `agent/agent.py` + `agent/runners.py` (generic continuous-action consumption); `utils/config_loader.py` (env-type discriminator + continuous-env + continuous-action config blocks; continuous action dim is config-carried, not added to the per-arch infra switch); `utils/brain_factory.py` (`_build_infra_kwargs` branch for the transformer only if non-default); new `brain/arch/transformer*.py`; the `scripts/run_simulation.py` CLI-default `match brain_type` block is refactored toward a registry default so the parity claim holds at the entrypoint too.
- **Configs**: new continuous-2D scenario YAMLs under `configs/scenarios/`; existing grid scenarios untouched.
- **Tests**: continuous-policy unit tests (tanh-Gaussian log-prob/entropy/Jacobian); migration-regression tests for the four discrete brains; continuous-2D env tests (capture radius, kinematics, Euclidean fields); a plugin-parity test asserting the ≤ 6-files / no-branches budget; transformer-brain tests.
- **Gates**: closes Gate 2. No change to Gate 1 (closed) or Gate 3 (future).
- **Out of scope**: Rung-2 chemical-gradient fidelity + adaptive chemosensory sensor (T6); real-worm validation + the L2 re-run (T7); native body mechanics and 3D (Future Directions).
