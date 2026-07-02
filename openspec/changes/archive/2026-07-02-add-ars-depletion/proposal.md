# Proposal: Area-Restricted Search via Source-Depletion Dynamics

## Why

Two evaluations established the memory-axis result on an **artificial** instrument: the bit-memory
positive control confirmed the comparison resolves working memory ([Logbook 030](../../../docs/experiments/logbooks/030-bit-memory-positive-control.md)),
and the minGRU/minLSTM bring-up confirmed new memory arms work ([Logbook 031](../../../docs/experiments/logbooks/031-minimal-rnn-candidates.md)).
The open scientific question is whether that separation **reproduces on a biologically-valid task**,
or is an artifact of the deliberately-artificial delayed-match-to-cue probe.

This change builds the **biological twin**: **area-restricted search (ARS)** driven by
**source-depletion dynamics**. Food/chemical **depletes as the agent feeds**, so the static Fick
concentration field changes *within* an episode — the local gradient **flattens in place** as a
patch is exhausted. The T6 foraging calibration found the concrete behavioural need (tracker
`T6.gradients.2`, 2026-06-12): the **static Fick flat-top removes the near-source gradient** — the
worm reaches a high-concentration region with **no directional signal for the final approach**,
exactly the regime where *area-restricted search / within-episode memory* is the biological
strategy ([docs/nematode_biology.md](../../../docs/nematode_biology.md); Hills 2004, Dorfman 2022).
A reactive gradient-follower cannot optimally forage a depleting patch — the instantaneous gradient
lies once the patch flattens — whereas a memory policy remembers which patches it has exhausted.

This is the **source-dynamics half** decomposed out of the deferred `T6.gradients.2`, **not** the
full `∂C/∂t = D∇²C` PDE (which stays phase-7 out-of-scope — a point sensor cannot perceive global
field dynamics). The bit-memory gate fired, so `T7.separation.ars_depletion` is **unblocked**. As
with bit-memory, a **null is itself a finding**: if the memory arms do not separate from the
reactive MLP on the depleting cell, that tells us environmental depletion alone does not induce a
separable within-episode-memory demand.

## What Changes

- **Per-source depleting amplitude** (config-gated, **off by default**): each food source carries a
  **remaining amount**; its contribution to the food concentration field (Fick or exponential
  kernel) scales with that amount, so a depleting source's local bump **flattens in place** rather
  than vanishing.
- **Depletion on feeding**: each food-consumption event **decrements** the matched source's amount
  by a configured quantum. Depletion is applied **once per step at the consume event** — never as a
  side effect of a field read (klinotaxis samples `get_food_concentration` twice per step, plus the
  scalar and reward reads; a read-coupled decrement would deplete non-deterministically).
- **In-place flattening, removal at exhaustion**: a depleting source persists at reduced amplitude
  (its near-source gradient flattens) until its amount crosses a removal threshold (~0), at which
  point it is removed and the existing respawn contract fires (subject to `no_respawn`).
- **Substrate-agnostic, both consume paths**: applied in the grid and continuous-2D consume paths
  via a shared decrement helper (the continuous substrate is the ARS evaluation target). The
  `distance == 0` concentration special case SHALL read the **source's** amount, not the global
  `gradient_strength`.
- **Disabled = byte-identical**: when depletion is off (default), the field uses the global
  `gradient_strength` exactly as today; per-source amounts only activate under the flag, so existing
  and discrete-grid configs are unchanged.
- **A new continuous-2D ARS foraging cell** (depletion enabled + `no_respawn` / depletion-aware
  respawn) calibrated so the depletion induces the within-episode-memory demand.
- **Evaluation**: run the existing arm panel (incl. `mingruppo`/`minlstmppo`) on the ARS cell and
  test whether the 030/031 separation reproduces on this biologically-valid task, reusing the
  committed paired-seed Wilcoxon + bootstrap + BH-FDR statistics layer.
- **Explicitly the source-dynamics half, not the PDE; non-gating** for Phase 6 Gate 3 (the MUST
  integrated-C3 ranking is unchanged).

## Capabilities

### New Capabilities

- **`source-depletion-dynamics`** — the within-episode source-depletion mechanism: the per-source
  remaining-amount data model, the amount-scaled field contribution, the feeding-driven
  once-per-step decrement, in-place flattening + removal-at-exhaustion, the respawn interaction, the
  disabled-is-byte-identical contract, and the ARS within-episode-memory hypothesis + evaluation.
  This is a config-gated environment dynamic that composes with the existing field, foraging, and
  consumption systems.

### Modified Capabilities

- **`chemical-gradient-fidelity`** — the "field SHALL remain static (frozen at assay time)"
  requirement gains a **depletion carve-out**: when source-depletion is enabled the source
  **amplitude** is time-varying *within* an episode (the geometry / per-signal `D` stay frozen), and
  this is explicitly distinguished from the still-out-of-scope time-evolving `∂C/∂t = D∇²C` diffusion
  PDE. Default (depletion off) behaviour is unchanged.
- **`continuous-2d-environment`** — the capture-radius food-consumption requirement (consume =
  remove + respawn) gains the **graded-depletion variant**: when depletion is enabled, a consume
  event decrements the source and removes/respawns it only at exhaustion.

## Impact

- **Food data model** — a parallel index-aligned per-source amount store in `env/env.py`
  (`self.foods` stays a position list to avoid breaking the many `(x, y)` unpacking + `tuple in foods` call sites), kept in sync across removal/spawn via a small helper, initialised at the
  `__init__` declaration; the **base** `DynamicForagingEnvironment.copy()` carries it (the continuous
  override delegates to it).
- **Field reads** — `_food_field_magnitude` / `_compute_food_gradient_vector` /
  `get_food_concentration` (`env/env.py`) thread the per-source amount (only when depletion is
  enabled); the `distance == 0` special case reads the source amount (the one place global strength
  leaks into a per-source sum).
- **Consume paths** — the grid + continuous `consume_food_for` refactored to match the source by
  **index** (not value) and route through a shared decrement-vs-remove helper; `reached_goal_for`
  excludes exhausted sources (one gate covering consumption, the goal bonus, and multi-agent
  competition).
- **Reward coherence** — exclude below-threshold sources from the nearest-food distance metric
  (`agent/reward_calculator.py`) under depletion, so the distance-shaping reward does not pull toward
  flattened patches.
- **Renderer** — add an amount signature to the continuous fidelity renderer's food-heatmap cache key
  (`env/pygame_renderer.py`) so the position-fixed depleting field visualises live (the calibration
  pre-check needs it).
- **Config** — a depletion block on `ForagingParams` (`env/env.py`) + `ForagingConfig`
  (`utils/config_loader.py`), off by default, wired through `to_params()`.
- **Configs** — a new ARS scenario cell plus a `no_respawn`-only **control** cell (to attribute the
  separation to depletion vs `no_respawn`'s non-stationarity).
- **Evaluation** — reuses the committed paired-seed statistics layer; a foraging-success separation
  read on the ARS cell **and** the control, reporting depletion's marginal effect.
- **Tests** — amount-scaled field, once-per-step decrement, in-place flattening, removal at
  exhaustion, disabled-is-byte-identical, `copy()` transfer, both substrates.
- **No new dependencies; no new architecture.** Tracker: `T7.separation.ars_depletion`.
