# Tasks: Area-Restricted Search via Source-Depletion Dynamics

## 1. Config block (off by default)

- [ ] 1.1 Add a source-depletion block to `ForagingParams` (`env/env.py:~231`) with sensible defaults: `source_depletion_enabled: bool = False`, `source_initial_amount: float = 1.0`, `depletion_per_feed: float = 0.25` (gradual — ~4 feeds to exhaust; NOT one-bite, which would just reproduce binary removal), `removal_eps: float = 1e-3` (and an optional `deplete_scales_reward: bool = False` lever, D7).
- [ ] 1.2 Mirror the block on `ForagingConfig` (`utils/config_loader.py:284-306`) with pydantic `Field(...)` validators (positive amounts; quantum ≤ initial); wire through `to_params()` (`config_loader.py:328-345`).

## 2. Data model (parallel amount store)

- [ ] 2.1 Add `self.food_amounts: list[float]` index-aligned with `self.foods`, initialised at the `__init__` declaration (`env/env.py:~1559`, alongside `self.foods = []`) AND repopulated in `_initialize_foods` (which runs only conditionally, so the store must exist before it) — each placed source starts at `source_initial_amount`. Keep the lists aligned **always** (append a default amount on add, pop on remove); the field consults amounts only when depletion is enabled.
- [ ] 2.2 Add a `_remove_food(index)` / `_add_food(pos, amount)` helper pair that mutates **both** lists; route every `self.foods` mutation through it (`_initialize_foods`, `spawn_food`, both `consume_food_for`) so the lists never desync.
- [ ] 2.3 Carry `food_amounts` in the **base** `DynamicForagingEnvironment.copy()` (`env/env.py:~4327`, where `new_env.foods = self.foods.copy()`) — NOT the continuous override, which delegates to `super().copy()` and never touches `foods` (adding it there would be a silent no-op).

## 3. Field reads (amount-scaled; off = byte-identical)

- [ ] 3.1 Give `_food_field_magnitude` (`env/env.py:2078`) an optional `source_amount: float | None = None`; when `None` use the global `gradient_strength` (today's path, byte-identical), else scale by `source_amount`.
- [ ] 3.2 Thread the per-source amount through the two superposition sites — `_compute_food_gradient_vector` (`env/env.py:2111`) and `get_food_concentration` (`env/env.py:2202`) — passing the source's amount **only when depletion is enabled**.
- [ ] 3.3 Fix the `distance == 0` special case in `get_food_concentration` (`env/env.py:2234`) to read the **source's** amount (not the global `gradient_strength`) when depletion is enabled.
- [ ] 3.4 Assert no field read mutates `food_amounts` (purity — the once-per-step landmine, D3).
- [ ] 3.5 Add an amount signature to the continuous fidelity renderer's food-heatmap cache key (`env/pygame_renderer.py:~1497`, keyed on positions only today) when depletion is enabled, so the position-fixed depleting field visualises live (needed for the 7.1 calibration; the live scalar/quiver getters already read the current field).

## 4. Consume paths (decrement vs remove; both substrates)

- [ ] 4.1 Add a shared `_deplete_or_remove(index)` helper: when depletion enabled, decrement `food_amounts[index]` by `depletion_per_feed`; if it crosses `removal_eps`, `_remove_food` + `spawn_food` (subject to `no_respawn`); when disabled, remove outright (today's behaviour).
- [ ] 4.2 Route grid `consume_food_for` (`env/env.py:~2467`) and continuous `consume_food_for` (`continuous_2d.py:~318`) through the shared helper — once per consume event (the only per-step depletion site, D3). Refactor BOTH to locate the matched source by **index** (they `foods.remove(agent_tuple)` / `foods.remove(nearest)` by value today) so the helper decrements the correct `food_amounts[index]` even when two foods coincide.
- [ ] 4.3 `reached_goal_for` / capture predicate (`env/env.py:~2461`, `continuous_2d.py:~309`): a source at/below `removal_eps` SHALL NOT count as reachable food (no goal/reward) — one gate that atomically covers consumption, the goal bonus, and multi-agent competition.
- [ ] 4.4 Reward coherence (D5): exclude below-`removal_eps` sources from the nearest-food distance metric (`agent/reward_calculator.py` → `get_nearest_food_distance_*`) when depletion is enabled, so the distance-shaping reward does not pull the agent toward exhausted patches (which would fight the memory demand).
- [ ] 4.5 (Lever, D7) if `deplete_scales_reward`, scale the consume reward/satiety by the source's remaining amount.

## 5. Tests

- [ ] 5.1 Disabled = byte-identical: with depletion off, `get_food_concentration` / gradient / consume / `copy()` match the pre-change behaviour (no amount effect).
- [ ] 5.2 Amount-scaled field: a half-depleted source contributes ~half the bump; the `distance==0` case reads the source amount.
- [ ] 5.3 Once-per-step purity: sampling the field N times (incl. klinotaxis 2×) leaves `food_amounts` unchanged; a single consume decrements exactly once.
- [ ] 5.4 In-place flattening + removal: a partially-depleted source persists in place (position fixed, amplitude reduced) and stays consumable; at `removal_eps` it is removed (+ respawn unless `no_respawn`); a below-threshold source is not consumable.
- [ ] 5.5 Integrity: `food_amounts` stays index-aligned across add/remove; continuous `copy()` preserves it; both substrates deplete.
- [ ] 5.6 Index-matching: two coincident (or near-coincident) sources — a consume depletes the **matched** source's amount, not another's (guards the value-vs-index refactor, 4.2).
- [ ] 5.7 Reward coherence: a below-`removal_eps` source is excluded from the nearest-food distance metric under depletion (the distance-shaping reward does not approach an exhausted patch; spec scenario).

## 6. Scenario config

- [ ] 6.1 A new continuous-2D ARS foraging cell (`configs/scenarios/foraging/…_ars_depletion…yml`): depletion enabled, `no_respawn` (or depletion-aware respawn), patch-structured food, klinotaxis sensing; mirror the 029 continuous-2D foraging recipe otherwise.
- [ ] 6.2 A `no_respawn`-only **control** cell (identical to 6.1 but depletion OFF — consume removes outright) to isolate depletion's marginal effect from `no_respawn`'s non-stationarity (the confound, D8).

## 7. Evaluation (does the separation reproduce?)

- [ ] 7.1 Learnability + calibration pre-check: visualise the depleting field (the continuous fidelity renderer) and run a quick read to confirm the cell induces non-gradient search (tune patch density / `depletion_per_feed` / `source_initial_amount` / `no_respawn`); record the calibration.
- [ ] 7.2 Run the arm panel (`mlpppo`, `lstmppo`, `cfcppo`, `transformerppo`, `connectomeppo`, `mingruppo`, `minlstmppo`) on the ARS cell, n paired seeds.
- [ ] 7.3 Run the same panel + seeds on the `no_respawn`-only control cell (6.2). Match the control's total available food to the ARS cell (depletion gives multiple feeds per source) so the marginal-separation comparison is on comparable foraging economics.
- [ ] 7.4 Separation read on **both** cells: plateau-tail foraging success per arm + paired-seed deltas vs the memoryless MLP (reuse the committed Wilcoxon + bootstrap + BH-FDR layer). The depletion claim = the **marginal** separation of the ARS cell over the control (D8); report the verdict — depletion-attributable separation, no_respawn-only separation, or null.
- [ ] 7.5 Write the logbook (objective / method / results / analysis / limitations) + committed supporting artefacts.

## 8. Docs + tracker

- [ ] 8.1 Tick `T7.separation.ars_depletion` with the verdict (separation = the biological twin confirmed; null = environmental depletion alone insufficient — itself a finding).
- [ ] 8.2 Add the logbook row to `docs/experiments/README.md`.
- [ ] 8.3 Document the `_ars_depletion` config variant suffix in the `AGENTS.md` scenario conventions (foraging family) for discoverability.

## 9. Gates

- [ ] 9.1 Targeted `pre-commit` during iteration; full `pre-commit run -a` before push.
- [ ] 9.2 `openspec validate add-ars-depletion --strict`.
- [ ] 9.3 Full `uv run pytest -m "not nightly"` green (no regression; the disabled-is-byte-identical assertion holds).
- [ ] 9.4 Archive the change in-PR (`openspec archive add-ars-depletion -y`).
