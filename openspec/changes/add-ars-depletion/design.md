# Design: Area-Restricted Search via Source-Depletion Dynamics

## Context

The food concentration field on the continuous-2D substrate is computed **live from `self.foods`
on every sensing query** — no cache, no precomputed grid (`field_magnitude` at `env/env.py:76`,
summed at three read sites: `get_food_concentration`, `_compute_food_gradient_vector`,
`get_separated_gradients`). A "food" is a bare position tuple with **no amount**. Consuming a food
today **binary-removes** it from `self.foods` (`consume_food_for` — grid `env/env.py:2467`,
continuous `env/continuous_2d.py:318`) and respawns one elsewhere, so the field changes
*discontinuously and non-locally* — too coarse for ARS.

ARS needs the opposite: a source that **flattens in place** as it is fed upon, removing the
near-source gradient while the source persists, so a reactive gradient-follower loses its signal and
an internal-memory policy gains the advantage. This is the deferred **source-dynamics half of
`T6.gradients.2`** — explicitly **not** the full `∂C/∂t = D∇²C` PDE.

## Goals / Non-Goals

**Goals:**

- A config-gated **per-source depleting amplitude**: feeding decrements a source's remaining amount;
  its field contribution scales with that amount; it flattens in place and is removed only at
  exhaustion.
- **Byte-identical when disabled** — the default field path is unchanged (uses the global
  `gradient_strength`); per-source amounts activate only under the flag.
- Apply in **both** consume paths via a shared helper; the continuous substrate is the ARS target.
- A new continuous-2D **ARS cell** + an evaluation that tests whether the 030/031 memory separation
  reproduces on this biologically-valid task.

**Non-Goals:**

- **Not** the time-evolving `∂C/∂t = D∇²C` diffusion PDE (geometry / per-signal `D` stay frozen;
  only the source *amplitude* varies within an episode). Phase-7, out of scope.
- **Not** a hand-coded ARS turning rule (real *C. elegans* ARS is a behavioural state machine;
  here ARS must **emerge** from the policy because the environment demands it — the point is an
  architecture-memory probe, not a scripted local-search heuristic).
- **Not** a new architecture, statistics layer, or Gate-3 change.
- **Not** a field cache (the field is recomputed per query; a cache would need invalidation on every
  feed — avoid).

## Decisions

### D1 — Depletion model: per-source amount, feeding-driven decrement, amount-scaled field

Each food source carries a **remaining amount** `a ∈ [0, a₀]` (initial `a₀` configurable). Its
contribution to the field becomes `a · field_magnitude(distance, …)` (the amount **replaces** the
global `strength` for that source). A consume event decrements the matched source's amount by a
configured quantum `depletion_per_feed`; the source persists at the reduced amount until
`a ≤ removal_eps`, then it is removed (and the existing respawn fires, subject to `no_respawn`).

- **Alternative — continuous proximity drain** (the source drains every step the agent is within a
  feeding radius, independent of the satiety-gated consume): rejected as the default — it adds a
  second feeding mechanism and a per-step env tick. The per-consume decrement reuses the existing
  consume event and matches the tracker's "depletes when eaten" framing. Proximity drain is recorded
  as a fallback if discrete consume under-depletes the patch (Open Questions).
- **Alternative — the full `∂C/∂t` PDE:** out of scope (Non-Goals).

### D2 — Data model: a parallel index-aligned amount list (not a tuple/dataclass promotion)

Keep `self.foods: list[tuple]` as positions and add a parallel **`self.food_amounts: list[float]`**
index-aligned with it. Rationale: promoting foods to `(x, y, a)` tuples or a `FoodSource` dataclass
would break the many 2-tuple-unpacking and `tuple in self.foods` call sites (membership tests,
distance loops, rendering, metrics — `env/env.py:2131,2228,2482,2518`; `continuous_2d.py:315,324,
339,358`). The parallel list touches far less surface.

The two lists are kept in sync through a **single `_remove_food(index)` / `_add_food(pos, amount)`
helper pair** that mutates both; **every** `self.foods` mutation routes through it: the `__init__`
declaration (`env/env.py:1559` — initialise `food_amounts` alongside `self.foods = []`, since
`_initialize_foods` runs only conditionally and the store must never be read before it),
`_initialize_foods`, `spawn_food`, and **both** `consume_food_for` paths.

Two correctness details the consume + copy sites force:

- **Copy lives in the BASE class.** The foods transfer is `new_env.foods = self.foods.copy()` in
  `DynamicForagingEnvironment.copy()` (`env/env.py:4327`); the continuous
  `Continuous2DEnvironment.copy()` delegates to `super().copy()` and never touches `foods`, so
  `food_amounts` MUST be copied at the base `:4327` site (adding it to the subclass would be a no-op
  → silent desync on every snapshot).
- **Match by INDEX, not value.** Grid consume does `self.foods.remove(agent_tuple)` (`env/env.py:2483`)
  and continuous does `self.foods.remove(nearest)` (`continuous_2d.py:328`) — both locate a source by
  *value*. They MUST be refactored to compute the matched source **index** so the shared helper can
  decrement `food_amounts[index]`; otherwise two coincident foods would deplete the wrong source.

### D3 — Depletion is a once-per-step consume event, NOT a field-read side effect (the landmine)

Klinotaxis calls `get_food_concentration` **twice per step** (left/right head-sweep,
`agent/agent.py:860`), plus the scalar chemotaxis read and reward-term reads — so a decrement
triggered inside a field read would deplete a different amount depending on the sensing mode and the
number of reads. **Field reads stay pure**; depletion is applied exactly once per step, in
`consume_food_for` (the single consume event already gated by `reached_goal` + satiety). The field
reflects the new amount on the *next* step's reads — the within-episode change the memory demand
needs.

### D4 — Disabled is byte-identical

`_food_field_magnitude` gains an optional `source_amount` parameter; when `None` (depletion off, the
default) it uses the global `gradient_strength` exactly as today. The read sites pass a source amount
**only when depletion is enabled**. The `distance == 0` special case in `get_food_concentration`
(`env/env.py:2234`, today adds the global `gradient_strength`) reads the **source's** amount when
depletion is on, so a depleted source the worm sits on reads its true (reduced) strength. With
depletion off, `food_amounts` is unused and the field path is unchanged → existing + discrete-grid
configs are byte-stable.

### D5 — In-place flattening, removal at exhaustion, and the respawn interaction

A depleting source stays at its position with shrinking amplitude (the near-source gradient
flattens). It is removed only when `a ≤ removal_eps`; removal then triggers the existing
`spawn_food()` respawn — UNLESS `no_respawn` is set. **For the ARS cell `no_respawn` (or a
depletion-aware respawn that does not place a fresh full source next to a depleted one) is
expected**, so the patch genuinely flattens and is not rescued by a fresh source the reactive policy
can immediately gradient-follow to. `reached_goal_for` must treat a below-`removal_eps` source as
**not food** (no goal/reward once depleted) — one gate that atomically covers consumption, the goal
bonus, and multi-agent food competition (all key off `reached_goal_for`).

**Reward coherence.** The foraging distance-shaping reward must be made coherent with depletion:
`reward_calculator`'s nearest-food distance term (`get_nearest_food_distance_*`) measures distance to
source *positions* and would keep pulling the agent toward a flattened-but-not-yet-removed patch that
`reached_goal_for` now rejects — rewarding re-approach to exhausted patches and working *against* the
memory demand. Below-`removal_eps` sources MUST be excluded from the nearest-food distance metric
when depletion is enabled.

### D6 — Both substrates via a shared decrement helper + the distance-0 fix

Grid `consume_food_for` (`env/env.py:2467`) and continuous `consume_food_for`
(`continuous_2d.py:318`) both call a shared `_deplete_or_remove(index)` helper (decrement; remove +
respawn at exhaustion) so the continuous substrate — the ARS target — cannot silently miss it. The
`distance == 0` special case fix (D4) lives in the shared field path.

### D7 — Reward/satiety coupling is a lever, default off

The consume reward + satiety restoration stay **fixed per bite** by default (the memory demand comes
from the **field** flattening, not the reward), keeping the change minimal. An optional config lever
**scales the consume reward/satiety by the source's remaining amount** (a depleted patch yields less
food) — biologically natural and a stronger patch-abandonment pressure — left off by default and
available for the eval to dial in if the field-only demand is too weak.

### D8 — Evaluation: does the separation reproduce, honestly framed

A new continuous-2D ARS cell (depletion on, `no_respawn`, patch-structured food) run on the arm panel
(`mlpppo`, `lstmppo`, `cfcppo`, `transformerppo`, `connectomeppo`, `mingruppo`, `minlstmppo`),
n paired seeds, reusing the committed paired-seed Wilcoxon + bootstrap + BH-FDR layer. Primary
metric: plateau-tail **foraging success** (foods collected within the satiety budget) — a memory
policy that abandons depleted patches and routes to fresh ones should out-forage a reactive
gradient-follower that re-approaches flattened patches / stalls on depleted-flat regions.

**Attribution control — the `no_respawn` confound.** `no_respawn` *alone*, with depletion **off**,
already makes the food field non-stationary (eaten sources vanish and never return), which a memory
arm can exploit by remembering depleted *locations*. So a memory-vs-MLP separation on the depleting
cell is not by itself attributable to in-place depletion — it could be driven entirely by
`no_respawn`'s discrete-removal non-stationarity. The eval therefore includes a **`no_respawn`-only
control cell** (consume removes outright, depletion off). The depletion claim rests on depletion's
**marginal** effect over that control: a larger separation with depletion on (or a separation that
the control does not show).

**Renderer (calibration tooling).** The calibration pre-check visualises the depleting field with the
continuous fidelity renderer, whose food-heatmap surface cache keys on **positions only**
(`pygame_renderer.py:1497`). A position-fixed depleting field would render frozen, so an **amount
signature must be added to the cache key** when depletion is enabled (the live scalar/quiver getters
already read the current field and are fine — only the cached heatmap is stale).

**Honest framing (the null is a finding).** The mechanism *creates the condition* (in-place field
flattening); whether that flattening forces enough **non-gradient** search to *separate* the arms is
the empirical question. It depends on cell calibration (patch density, `depletion_per_feed`, `a₀`,
`no_respawn`, world size) — a tuning task, as with bit-memory's difficulty calibration. If the memory
arms do **not** beat the reactive MLP on a well-calibrated depleting cell, that is the finding:
environmental depletion alone does not induce a separable within-episode-memory demand at this
fidelity.

## Risks / Trade-offs

- **The memory demand may not emerge** (field flattening insufficient to force non-gradient search).
  → Mitigation: calibrate the cell (a learnability pre-check + a fidelity-renderer visualisation of
  the depleting field, like the 028/030 pre-checks); the reward-coupling lever (D7); report the null
  as a finding.
- **`food_amounts` desync with `foods`** (every mutation must mirror). → Mitigation: a single
  `_remove_food`/`_add_food` helper pair both substrates route through; tests for sync + `copy()`
  transfer + the off-is-byte-identical invariant.
- **Foraging economics shift** (a patch becomes multiple meals; total food may change). → Mitigation:
  this is a **new** cell, not the 029 C3 cell; non-gating; the eval calibrates it.
- **Two substrates** — the continuous path is the target; the shared helper (D6) prevents the grid
  path from diverging or the continuous path from missing the decrement.
- **Attribution confound** — `no_respawn` alone (no depletion) is itself a within-episode memory
  demand, so a separation on the depleting cell may not be attributable to depletion. → Mitigation:
  the `no_respawn`-only control cell (D8); the depletion claim rests on its marginal effect.
- **Reward/field incoherence** — the distance-shaping reward would pull toward depleted-but-present
  patches, fighting the memory demand. → Mitigation: exclude below-`removal_eps` sources from the
  nearest-food distance metric under depletion (D5).

## Migration Plan

Additive and behaviour-preserving. Depletion is opt-in via the `ForagingParams` /`ForagingConfig`
block (off by default); with it off the field, consumption, and `copy()` are byte-identical and the
existing `lstmppo`/etc. tests stay green. Rollback = remove the config block + the amount store + the
shared helper; no data or schema migration.

## Open Questions

- **Depletion granularity** — per-consume-event (D1, default) vs continuous proximity drain. Settle
  during the eval calibration: if the satiety-gated discrete consume under-depletes a patch (the worm
  eats once and leaves with the bump barely reduced), switch on proximity drain.
- **Reward coupling** — keep the consume reward fixed (D7 default) or scale it by amount. Settle in
  calibration: enable scaling if the field-only memory demand is too weak to separate the arms.
- **Eval metric** — plateau-tail foraging success (D8) vs a paired memory-vs-reactive contrast on the
  same seeds; pick the one that most cleanly reads the separation (likely success, matching 029).
