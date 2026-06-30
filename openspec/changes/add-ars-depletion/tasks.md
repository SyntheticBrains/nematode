# Tasks: Area-Restricted Search via Source-Depletion Dynamics

## 1. Config block (off by default)

- [x] 1.1 Add a source-depletion block to `ForagingParams` (`env/env.py:~231`) with sensible defaults: `source_depletion_enabled: bool = False`, `source_initial_amount: float = 1.0`, `depletion_per_feed: float = 0.25` (gradual — ~4 feeds to exhaust; NOT one-bite, which would just reproduce binary removal), `source_removal_eps: float = 1e-3` (and an optional `deplete_scales_reward: bool = False` lever, D7).
- [x] 1.2 Mirror the block on `ForagingConfig` (`utils/config_loader.py`) with pydantic `Field(gt=0)` validators + a `model_validator` rejecting quantum > initial; wire through `to_params()`.

## 2. Data model (parallel amount store)

- [x] 2.1 Add `self.food_amounts: list[float]` index-aligned with `self.foods`, initialised at the `__init__` declaration AND repopulated in `_initialize_foods`; each placed source starts at `source_initial_amount`. Lists kept aligned always; the field consults amounts only when depletion is enabled.
- [x] 2.2 Add a `_remove_food(index)` / `_add_food(pos, amount)` helper pair that mutates **both** lists; routed `_initialize_foods` + `spawn_food` appends through `_add_food` (consume routes through `_remove_food` via 4.1).
- [x] 2.3 Carry `food_amounts` in the **base** `DynamicForagingEnvironment.copy()` (next to `new_env.foods = self.foods.copy()`).

## 3. Field reads (amount-scaled; off = byte-identical)

- [x] 3.1 Give `_food_field_magnitude` an optional `source_amount: float | None = None`; when `None` use the global `gradient_strength` (byte-identical), else scale by `source_amount`.
- [x] 3.2 Thread the per-source amount through `_compute_food_gradient_vector` and `get_food_concentration` (enumerate; pass the amount only when depletion is enabled).
- [x] 3.3 Fix the `distance == 0` special case in `get_food_concentration` to read the **source's** amount when depletion is enabled.
- [x] 3.4 Field reads are pure — depletion is applied only at the consume event (4.1), never in a field read (covered by test 5.3).
- [x] 3.5 Add an amount signature to the continuous fidelity renderer's food-heatmap cache key (`_heatmap_cache_key`) when depletion is enabled, so the position-fixed depleting field visualises live.

## 4. Consume paths (decrement vs remove; both substrates)

- [x] 4.1 Add a shared `_deplete_or_remove(index)` helper: when depletion enabled, decrement `food_amounts[index]` by `depletion_per_feed`; while above `source_removal_eps` leave in place; else (or when disabled) `_remove_food` + `spawn_food` (subject to `no_respawn`).
- [x] 4.2 Route grid + continuous `consume_food_for` through the shared helper, located by **index** (not value) so the matched source drains even when two foods coincide.
- [x] 4.3 `reached_goal_for` (grid + continuous): a source at/below `source_removal_eps` does not count as reachable food — one gate covering consumption, the goal bonus, and multi-agent competition.
- [x] 4.4 Reward coherence (D5): confirmed no `reward_calculator` change is needed — exhausted sources are removed (4.1) so they are absent from all food signals; a *partially*-depleted source above the threshold is still valid food (correct to attract). The field-independent distance-shaping confound is handled at the cell level (6.1).
- [ ] 4.5 (Lever, D7) if `deplete_scales_reward`, scale the consume reward/satiety by the source's remaining amount.

## 5. Tests

- [x] 5.1 Disabled = byte-identical: with depletion off, `get_food_concentration` / gradient / consume / `copy()` match the pre-change behaviour (no amount effect). *(`test_disabled_field_ignores_amounts` — concentration + gradient off-paths ignore amounts; `test_disabled_consume_removes_outright`; the unchanged 216-test env suite covers the broader default path.)*
- [x] 5.2 Amount-scaled field: a half-depleted source contributes ~half the bump; the `distance==0` case reads the source amount. *(`test_amount_scales_field_and_distance_zero`.)*
- [x] 5.3 Once-per-step purity: sampling the field N times (incl. klinotaxis 2×) leaves `food_amounts` unchanged; a single consume decrements exactly once. *(`test_field_reads_are_pure` + `test_one_consume_decrements_once`.)*
- [x] 5.4 In-place flattening + removal: a partially-depleted source persists in place (position fixed, amplitude reduced) and stays consumable; at `removal_eps` it is removed (+ respawn unless `no_respawn`); a below-threshold source is not consumable. *(`test_persist_in_place_then_remove_at_exhaustion` + `test_below_threshold_not_consumable`.)*
- [x] 5.5 Integrity: `food_amounts` stays index-aligned across add/remove; continuous `copy()` preserves it; both substrates deplete. *(`test_copy_preserves_amounts` + `test_grid_substrate_depletes`.)*
- [x] 5.6 Index-matching: two coincident (or near-coincident) sources — a consume depletes the **matched** source's amount, not another's (guards the value-vs-index refactor, 4.2). *(`test_consume_matches_source_by_index`.)*
- [x] 5.7 Signal absence via removal: a source consumed to exhaustion is removed and therefore absent from the nearest-food distance metric (and the concentration/gradient) — no spent patch lingers in any food signal. *(`test_signal_absent_after_exhaustion`.)*

## 6. Scenario config

- [x] 6.1 A new continuous-2D ARS foraging cell (`configs/scenarios/foraging/…_ars_depletion…yml`): depletion enabled, `no_respawn` (or depletion-aware respawn), patch-structured food, klinotaxis sensing, and a low/zero `reward_distance_scale` so the within-episode-memory demand rests on the depleting field (not field-independent distance shaping); mirror the 029 continuous-2D foraging recipe otherwise. *(`mlpppo_small_continuous2d_fick_adaptive_klinotaxis_ars_depletion.yml` — mlpppo reference cell: depletion 1.0/0.25 ≈ 4 feeds, `no_respawn`, `reward_distance_scale: 0`; 5 patches × ~4 feeds ≈ 20 collectable, target 10. Loads + wires through the loader. Per-arm panel minted at 7.2 after calibration.)*
- [x] 6.2 A `no_respawn`-only **control** cell (identical to 6.1 but depletion OFF — consume removes outright) to isolate depletion's marginal effect from `no_respawn`'s non-stationarity (the confound, D8). *(`mlpppo_small_continuous2d_fick_adaptive_klinotaxis_no_respawn_control.yml` — depletion OFF, `foods_on_grid: 20` so single-feed sources match the ARS cell's ~20-feed total; residual patch-density caveat documented in-header.)*

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

- [ ] 9.1 Targeted `pre-commit` during iteration *(done per-file on every commit)*; full `pre-commit run -a` before push.
- [x] 9.2 `openspec validate add-ars-depletion --strict`. *(valid, 0 errors.)*
- [x] 9.3 Full `uv run pytest -m "not nightly"` green (no regression; the disabled-is-byte-identical assertion holds). *(4018 passed, 1 skipped, 2 xfailed, 0 failed.)*
- [ ] 9.4 Archive the change in-PR (`openspec archive add-ars-depletion -y`).
