# Tasks: fix-predator-sensing-biology

Phase 6 Tranche 3 (T3). Implements the corrected biology-driven two-channel predator-sensing model. Sub-tasks expand T3.1-T3.6 from [phase6-tracking/tasks.md:145-150](../phase6-tracking/tasks.md) into the biology-driven 10 sub-tasks per [design.md § Decision T3.10](design.md). Implementation order follows the layered dependency chain: env → BrainParams → agent population → STAM → modules → sensing config → sample configs → tests → tracker/docs → logbook.

## 1. Env-side contact-zone discrimination (T3.1)

- [x] 1.1 Add `ContactZone` enum to [packages/quantum-nematode/quantumnematode/env/env.py](../../../packages/quantum-nematode/quantumnematode/env/env.py) (or a sibling `env/predator.py` if env.py is already large) with members `NONE`, `ANTERIOR`, `POSTERIOR`, `LATERAL`. Use `StrEnum` for YAML-friendly serialisation.
- [x] 1.2 Implement `get_agent_predator_contact_zone_for(agent_id: str) -> ContactZone` on the env. Logic: if not in damage radius → `NONE`; else compute predator's relative bearing in agent-frame (using `agent_state.direction` per the existing `_compute_lateral_offsets` precedent at [agent.py:614-636](../../../packages/quantum-nematode/quantumnematode/agent/agent.py)); bucket per the ±45° / ±45° / lateral mapping documented in [design.md § Decision T3.3](design.md).
- [x] 1.3 Verify `is_agent_in_predator_contact_for(agent_id) -> bool` keeps its existing signature (returns True for any non-NONE zone). Legacy callers and SpikingReinforceBrain must continue working unchanged.
- [x] 1.4 Write `tests/quantumnematode_tests/env/test_predator_contact_zone.py`: 32 parameterised cases (predator at each of 8 surrounding cells × agent at each of 4 cardinal headings) asserting the expected zone. Plus 1 case asserting `NONE` outside damage radius. *Shipped: 38 tests (16 cardinal + 16 diagonal + 4 edge cases + 2 back-compat).*
- [x] 1.5 Run the new test + verify the existing `is_agent_in_predator_contact_for` tests still pass.

## 2. Env-side distal-chemo alias (T3.2)

- [ ] 2.1 Add `get_predator_sulfolipid_concentration(position: Position) -> float` method to env, delegating to `get_predator_concentration`. Docstring rebadges to "Liu et al. 2018 sulfolipid-analogue signal; exp-decay-sum placeholder; calibration deferred to T6/T7 per [design.md § Decision T3.5](design.md)".
- [ ] 2.2 No new tests for this alias (it's a pure delegation); existing `get_predator_concentration` tests cover the behaviour.

## 3. BrainParams field additions (T3.3)

- [ ] 3.1 In [brain/arch/\_brain.py](../../../packages/quantum-nematode/quantumnematode/brain/arch/_brain.py) `BrainParams`, add four new optional fields, all defaulting to `None`:
  - `predator_contact_intensity: float | None` — graded ASH response per [design.md § Decision T3.4](design.md), value ∈ [0, 1]
  - `predator_contact_zone: ContactZone | None` — anterior/posterior/lateral discrimination
  - `predator_distal_concentration: float | None` — sulfolipid-analogue concentration
  - `predator_distal_dconcentration_dt: float | None` — temporal derivative (populated by STAM)
- [ ] 3.2 Keep legacy fields `predator_contact: bool | None`, `predator_concentration: float | None`, `predator_dconcentration_dt: float | None` untouched. Document in the field docstrings that these are the legacy paths consumed by `nociception_*` modules and the reward calculator.
- [ ] 3.3 Import `ContactZone` from env into \_brain.py for the type annotation. Confirm no circular import (env imports BrainParams via `TYPE_CHECKING`).

## 4. Agent populates new BrainParams fields (T3.4)

- [ ] 4.1 In [agent/agent.py](../../../packages/quantum-nematode/quantumnematode/agent/agent.py) (locate the `_create_brain_params` or equivalent function — exploration found field population around lines 730-815, 881-894), populate the four new BrainParams fields from env + STAM:
  - `predator_contact_intensity` from `max(0, 1 - manhattan_dist / damage_radius)` clipped to [0, 1] using existing distance computation
  - `predator_contact_zone` from `env.get_agent_predator_contact_zone_for(agent_id)`
  - `predator_distal_concentration` from `env.get_predator_sulfolipid_concentration(agent_pos)` (alias of `get_predator_concentration`)
  - `predator_distal_dconcentration_dt` from STAM's `predator_distal` channel derivative output
- [ ] 4.2 Verify population happens unconditionally when env predator is enabled (so T4.0c's ConnectomePPO consumer can read them even before the new sensor modules are wired into the brain's sensory_modules list).
- [ ] 4.3 Verify legacy fields (`predator_contact`, `predator_concentration`, `predator_dconcentration_dt`) continue to be populated identically to today.

## 5. STAM channel split (T3.5)

- [ ] 5.1 In [agent/stam.py](../../../packages/quantum-nematode/quantumnematode/agent/stam.py) `CHANNEL_REGISTRY` (around line 57-93), add two new `STAMChannelDef` entries:
  - `predator_mechano`: `derivative_key="predator_mechano_dintensity_dt"`, `sensing_mode_attr="predator_mechano_mode"`
  - `predator_distal`: `derivative_key="predator_distal_dconcentration_dt"`, `sensing_mode_attr="predator_distal_mode"`
- [ ] 5.2 Keep the legacy `predator` channel entry (`derivative_key="predator_dconcentration_dt"`, `sensing_mode_attr="nociception_mode"`) as a frozen alias. Add a comment marking it deprecated-but-load-bearing-for-archived-configs.
- [ ] 5.3 Extend `resolve_active_channels(env, sensory_modules)` (or equivalent — exploration found `resolve_active_channels` at stam.py:96-130) to activate `predator_mechano` iff the active sensory_modules list contains a `predator_mechanosensation*` variant, and `predator_distal` iff it contains a `predator_chemosensation*` variant. `predator` (legacy) activates iff the list contains a `nociception*` variant.
- [ ] 5.4 Pre-flight: `grep -rn "num_channels" packages/quantum-nematode` and verify no code path hardcodes the predator-channel count (exploration found none; defensive grep confirms).
- [ ] 5.5 No new unit tests for STAM channel resolution beyond the integration-smoke pass at task 8 — existing STAM tests parameterise per-channel behaviour via the CHANNEL_REGISTRY, so adding two entries automatically extends the test matrix.

## 6. New sensor modules (T3.6)

- [ ] 6.1 In [brain/modules.py](../../../packages/quantum-nematode/quantumnematode/brain/modules.py) `ModuleName` StrEnum, add six new entries (per [design.md § Decision T3.1](design.md) — new modules use explicit `_oracle` suffix on the oracle variant, unlike the legacy bare-named convention):
  - `PREDATOR_MECHANOSENSATION_ORACLE = "predator_mechanosensation_oracle"`
  - `PREDATOR_MECHANOSENSATION_TEMPORAL = "predator_mechanosensation_temporal"`
  - `PREDATOR_MECHANOSENSATION_KLINOTAXIS = "predator_mechanosensation_klinotaxis"`
  - `PREDATOR_CHEMOSENSATION_ORACLE = "predator_chemosensation_oracle"`
  - `PREDATOR_CHEMOSENSATION_TEMPORAL = "predator_chemosensation_temporal"`
  - `PREDATOR_CHEMOSENSATION_KLINOTAXIS = "predator_chemosensation_klinotaxis"`
- [ ] 6.2 Implement six `_extract` core functions (one per ModuleName entry above). Patterns to mirror:
  - `_predator_mechanosensation_oracle_core` ← mirror `_nociception_core` (extract `predator_contact_intensity` + zone-as-angle, `classical_dim=2`)
  - `_predator_mechanosensation_temporal_core` ← mirror `_nociception_temporal_core` (extract intensity + dintensity_dt, classical_dim=2)
  - `_predator_mechanosensation_klinotaxis_core` ← mirror `_nociception_klinotaxis_core` (extract intensity + zone + dintensity_dt, classical_dim=3)
  - `_predator_chemosensation_oracle_core` ← mirror `_food_chemotaxis_core` (extract `predator_distal_concentration` + bearing-as-angle, classical_dim=2)
  - `_predator_chemosensation_temporal_core` ← mirror `_food_chemotaxis_temporal_core` (extract concentration + dconcentration_dt, classical_dim=2)
  - `_predator_chemosensation_klinotaxis_core` ← mirror `_food_chemotaxis_klinotaxis_core` (extract concentration + lateral + dconcentration_dt, classical_dim=3)
- [ ] 6.3 Register all six modules in `SENSORY_MODULES`. Mirror the existing `food_chemotaxis*` triples for the chemosensation family and the existing `nociception*` triples for the mechanosensation family.
- [ ] 6.4 Extend the `apply_sensing_mode` map in [utils/config_loader.py:797-805](../../../packages/quantum-nematode/quantumnematode/utils/config_loader.py) so configs that list `predator_mechanosensation_oracle` get auto-substituted to `_temporal` / `_klinotaxis` based on `sensing.predator_mechano_mode`, and similarly for `predator_chemosensation_oracle` based on `sensing.predator_distal_mode`. The translation logic mirrors how `food_chemotaxis` → `food_chemotaxis_temporal` translation works today, just with explicit `_oracle` as the base name rather than the bare `food_chemotaxis`.
- [ ] 6.5 Write `tests/quantumnematode_tests/brain/test_predator_modules.py`: six test classes (one per ModuleName), each asserting `extract` output shape, value range, and agent-direction-relative orientation. Mirror the patterns at [test_modules.py](../../../packages/quantum-nematode/tests/quantumnematode_tests/brain/test_modules.py) and [test_klinotaxis_modules.py](../../../packages/quantum-nematode/tests/quantumnematode_tests/brain/test_klinotaxis_modules.py).
- [ ] 6.6 Run the new module tests + verify existing module tests are unaffected.

## 7. SensingConfig mode fields (T3.7)

- [ ] 7.1 In [utils/config_loader.py](../../../packages/quantum-nematode/quantumnematode/utils/config_loader.py) `SensingConfig` (line 724-761), add two new fields:
  - `predator_mechano_mode: SensingMode = SensingMode.ORACLE`
  - `predator_distal_mode: SensingMode = SensingMode.ORACLE`
- [ ] 7.2 Keep `nociception_mode: SensingMode = SensingMode.ORACLE` field untouched. Existing `SensingMode` StrEnum unchanged.
- [ ] 7.3 Write `tests/quantumnematode_tests/utils/test_legacy_nociception_configs_load.py`: regression test that parametrises over every config under `configs/evolution/*klinotaxis*.yml` (and any other file naming legacy `nociception*` modules), asserting `configure_brain(load_simulation_config(...))` succeeds without error. Discover files at test-collection time (no hard-coded list).
- [ ] 7.4 Run the new regression test on the 22 archived configs; verify all pass.

## 8. Sample configs (T3.8)

- [ ] 8.1 Author 2-3 small predator-evasion smoke configs under `configs/scenarios/pursuit/` (or wherever existing predator-evasion configs live):
  - `mlpppo_small_predator_evasion.yml` — mlpppo, small, predator enabled, sensory_modules includes `food_chemotaxis_klinotaxis` + `predator_mechanosensation_klinotaxis` + `predator_chemosensation_klinotaxis`. STAM enabled. Headless-runnable.
  - `lstmppo_small_predator_evasion.yml` — same modules under LSTM-PPO.
  - `connectomeppo_small_predator_evasion.yml` — **OPTIONAL** at T3, only if cheap. ConnectomePPO won't actually consume the new BrainParams fields until T4.0c wires the projection — but the config can still load and run with the brain emitting random-policy-shaped output. Defer if it adds noise.
- [ ] 8.2 Verify each new config loads via `configure_brain(load_simulation_config(...))` and runs one episode headless via `uv run python scripts/run_simulation.py --config <new> --runs 1 --theme headless --episodes 5`.
- [ ] 8.3 No learning required at this stage — these are integration-smoke configs for T4 to consume.

## 9. Tracker + roadmap updates (T3.9)

- [ ] 9.1 In [openspec/changes/phase6-tracking/tasks.md](../phase6-tracking/tasks.md), tick T3.1-T3.6 sub-tasks as work lands. Final pass to confirm all ticked.
- [ ] 9.2 Add the T4 reward-ablation carry-forward sub-task. Specifically: extend the existing `T4 — Connectome-constrained`, `T4 — MLP-PPO`, `T4 — LSTM / GRU-PPO`, and `T4 — NEAT-evolved` sections with a sibling row each that records the reward-shape ablation per [design.md § Decision T3.7](design.md). Suggested wording: "T4.{arch}.predator_evasion_reward_ablation — compare existing `gradient_proximity` reward against `distal_chemo_penalty + binary_contact_damage_trigger` variant on the same architecture × seed grid".
- [ ] 9.3 Flip [docs/roadmap.md](../../../docs/roadmap.md) Phase 6 Tranche Tracker T3 row from `🔲 not started` → `🟡 in progress` on first commit of this branch, and → `✅ complete` after the implementation PR merges. Add the logbook anchor link once T3.10 lands.

## 10. Logbook section (T3.10)

- [ ] 10.1 Per [phase6-tracking/tasks.md T3.6](../phase6-tracking/tasks.md), T3's verification + biology-change documentation lives as a section inside the T4 L2 first-pass logbook (`docs/experiments/logbooks/0XX-l2-first-pass.md`) rather than its own logbook. If that logbook doesn't exist yet at T3-merge time, author a placeholder stub with the T3 section pre-filled. If T4 lands later, T3's section moves into the canonical T4 logbook.
- [ ] 10.2 Quantify the behavioural difference under matched conditions per [phase6-tracking/tasks.md T3.2](../phase6-tracking/tasks.md): run one of the new sample configs (e.g. `mlpppo_small_predator_evasion.yml`) with new sensors vs the equivalent existing klinotaxis-nociception config; record per-50-ep success curves + per-episode-mean reward over 100-200 ep; document the delta. This is "did the sensor correction change behaviour visibly?" not a full L2 evaluation (that's T4's job).
- [ ] 10.3 Cross-link the T3 section from this OpenSpec change's archive entry once the change is archived post-merge.

## 11. Pre-merge verification

- [ ] 11.1 Run full pre-commit on all changed files: `uv run pre-commit run --files <changed-files>`.
- [ ] 11.2 Run the full quantum-nematode test suite: `uv run pytest -m "not nightly" packages/quantum-nematode/`. Triage any new failures.
- [ ] 11.3 Run `openspec validate fix-predator-sensing-biology --strict` and confirm clean.
- [ ] 11.4 Audit staged-file sizes vs `.gitattributes` LFS rules — flag any > 100 KB file not covered by an existing rule.
- [ ] 11.5 Scan staged content for absolute home paths (`/Users/...`, `/home/...`, `C:\\Users\\...`) and `file:///` URI prefixes; sanitise to repo-relative references before committing. Apply to every commit on the branch, not only the pre-merge gate.
- [ ] 11.6 **Pause for user authorisation** before `git push` or `gh pr create`. Per project convention, remote-state mutations require explicit user approval each time — a prior approval doesn't authorise subsequent pushes.
