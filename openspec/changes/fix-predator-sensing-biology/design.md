# Design: fix-predator-sensing-biology

This document records the design decisions T3 must commit to. Each decision is bounded — only the connectome-side wiring, the reward-ablation, and the literature-tuning are explicitly deferred to later tranches with pointers to where they land.

## Existing infrastructure consumed

Listed up-front so the design references are concrete:

- **`SensoryModule` registry** at [brain/modules.py](../../../packages/quantum-nematode/quantumnematode/brain/modules.py) (line ~668 — `SENSORY_MODULES` dict). New modules slot in as new `ModuleName` enum + extractor + registry entry. Same pattern as existing `food_chemotaxis` / `food_chemotaxis_temporal` / `food_chemotaxis_klinotaxis` triples.
- **`apply_sensing_mode` translation hook** at [utils/config_loader.py](../../../packages/quantum-nematode/quantumnematode/utils/config_loader.py) (line ~772). Maps oracle module names → temporal/klinotaxis variants by suffix when `nociception_mode != ORACLE`. Will be extended with new mode-attr lookups for the new modules.
- **`STAMChannelDef` registry** at [agent/stam.py](../../../packages/quantum-nematode/quantumnematode/agent/stam.py) (line ~57-93 — `CHANNEL_REGISTRY` dict). New channels need `name`, `derivative_key`, `sensing_mode_attr` triples.
- **`BrainParams`** at [brain/arch/\_brain.py](../../../packages/quantum-nematode/quantumnematode/brain/arch/_brain.py) (line ~110-270). Pydantic model; new optional fields default to `None`.
- **Env predator detection** at [env/env.py](../../../packages/quantum-nematode/quantumnematode/env/env.py): `is_agent_in_predator_contact_for(agent_id) -> bool` (line ~2471 — delegates to `is_agent_in_damage_radius_for`), `get_predator_concentration(position) -> float` (line ~2016 — exp-decay sum), `get_separated_gradients(agent_pos) -> dict` (oracle mode only).

## Decision T3.1 — New sensor-module naming

**Decision**: scientific-function naming aligned with the existing registry pattern.

- **Contact-mechanosensory triple** (new — mirrors existing `mechanosensation` module for boundary contact, narrower scope to predator-derived contact):
  - `predator_mechanosensation_oracle` (2-dim)
  - `predator_mechanosensation_temporal` (2-dim)
  - `predator_mechanosensation_klinotaxis` (3-dim)
- **Distal-chemosensory triple** (new — mirrors `food_chemotaxis` but with `_chemosensation` suffix; the module names the *sensor*, not a behaviour. Liu et al. 2018 calls the underlying biology "predator-sulfolipid-induced chemosensory escape", so chemosensation is the literature-accurate framing.):
  - `predator_chemosensation_oracle` (2-dim)
  - `predator_chemosensation_temporal` (2-dim)
  - `predator_chemosensation_klinotaxis` (3-dim)
- `pheromone_alarm_ascr` (deferred to T6/T7 — mirrors `pheromone_alarm` / `pheromone_aggregation` / `pheromone_food_marking` already in the registry; `_ascr` suffix marks ascaroside-mediated per Jang et al. 2012)

**Naming convention — explicit `_oracle` suffix for new modules.** The legacy families (`food_chemotaxis`, `nociception`, `mechanosensation`, etc.) carry no suffix on their oracle variants — a historical accident from before the project supported multiple sensing modes. T3 adopts an **explicit `_oracle` suffix** on the new modules so the mode is self-describing without requiring readers to know that "bare-named = oracle". The legacy bare-named modules stay as-is (frozen historical record per Decision T3.2); only new modules adopt the corrected convention. Future per-modality additions (e.g. the deferred `pheromone_alarm_ascr`) should likewise carry explicit mode suffixes.

**Biologically-preferred default**: the klinotaxis variant of each triple is the biologically-preferred default for new T4 sample configs and downstream cells. Phase 5's empirical pattern (every `configs/evolution/*klinotaxis*` config selected `nociception_klinotaxis`) ran ahead of the formal biology framing; T3 codifies it. Oracle + temporal variants ship for ablation but should not be the default selection in new configs.

**Alternatives considered**:

- *Composite single `predator_biology` module emitting a multi-feature vector*: rejected because every other module in the registry is single-modality and downstream brains assume `classical_dim ∈ {1, 2, 3}` and slice accordingly. A composite would force a dim ≥ 4 special case nothing else uses.
- *Function-named (`predator_contact_detection`, `predator_distance_detection`)*: rejected because it diverges from the existing scientific-name convention (`proprioception`, `mechanosensation`, `nociception`) which is what `food_chemotaxis` already mirrors.
- *Bare-name the oracle variant (e.g. `predator_mechanosensation` without `_oracle` suffix) to match the legacy convention*: rejected. The legacy bare-name convention is the historical accident the new convention corrects; staying bare-named just propagates the ambiguity. The asymmetry between legacy and new modules is intentional and marks the inflection point in the project's naming convention.

## Decision T3.2 — Legacy compatibility

**Decision**: frozen-legacy + new. Keep `nociception` / `nociception_temporal` / `nociception_klinotaxis` modules in the registry **forever, exactly as-is**. Never auto-substitute. New configs select new modules; archived configs select legacy modules. Both code paths coexist.

**Why**: the 22 archived Phase 5 evolution configs under `configs/evolution/*klinotaxis*` are historical record. They were the substrate for the Lamarckian / Baldwin / coevolution / TEI logbooks. Silently changing their behaviour would invalidate the reproducibility of those logbooks. Hard-renaming would break load.

**Implementation contract**:

- `ModuleName.NOCICEPTION` / `NOCICEPTION_TEMPORAL` / `NOCICEPTION_KLINOTAXIS` enum entries stay.
- `SENSORY_MODULES[ModuleName.NOCICEPTION]` etc. stay. Description strings gain a "(deprecated — see predator_mechanosensation_oracle / predator_chemosensation_oracle and their temporal/klinotaxis variants)" suffix but the `extract` callable is unchanged.
- `SensingConfig.nociception_mode: SensingMode` field stays. New configs simply set `predator_mechano_mode` and `predator_distal_mode` and omit `nociception_mode` (which defaults to `ORACLE` but is irrelevant if no `nociception_*` module is in `sensory_modules`).
- A regression test (`tests/quantumnematode_tests/utils/test_legacy_nociception_configs_load.py`) parametrises over the 22 archived configs and asserts `configure_brain(load_simulation_config(...))` succeeds without error.

## Decision T3.3 — Contact zone discrimination

**Decision**: env method `get_agent_predator_contact_zone_for(agent_id) -> ContactZone` returns enum `{NONE, ANTERIOR, POSTERIOR, LATERAL}` based on the predator's relative bearing vs the agent's heading on the grid.

**Bearing-to-zone mapping**:

- ANTERIOR: predator within ±45° of agent's forward heading
- POSTERIOR: predator within ±45° of opposite-of-heading
- LATERAL: ±45° to ±135° on either side (i.e. anything that doesn't cleanly map to ASH-anterior or PLM-posterior receptive fields)
- NONE: no contact (predator outside damage radius)

**Modelling caveat (documented in design.md so future readers know it's intentional)**: real *C. elegans* anterior-vs-posterior discrimination is *anatomical* — ASH dendrites physically terminate at the head, PLM dendrites at the tail. A predator touching the head fires ASH directly; a predator touching the tail fires PLM directly. There's no "compute relative bearing" step in the biology.

On a 1-cell grid agent (or even body-length-2), the agent has no spatial extent and so there is no biological-anatomy equivalent of head-vs-tail location. The bearing-vs-heading approach is a *behavioural proxy* for what the body anatomy would do if the agent had spatial extent: "the predator approached from in front" ≈ ASH fired; "the predator approached from behind" ≈ PLM fired. This is a faithful model of the *behavioural* signal even on a 1-cell agent, but it's not anatomically literal.

**T5 / T7 follow-up**: continuous-2D physics gives the agent real body extent, at which point the ContactZone discrimination can shift from bearing-based to actual body-overlap-based. Recorded as a T5/T7 carry-forward.

**Alternatives considered**:

- *Two env methods (anterior + posterior bool)*: rejected — doubles the API surface; lateral case forces an awkward "neither" return. Single typed return is cleaner.
- *Bool + relative-bearing float, brain derives zone*: rejected — leaks the discrimination logic into every brain consumer; we'd reimplement the bearing-vs-heading comparison N times.

The existing `is_agent_in_predator_contact_for(agent_id) -> bool` keeps its signature and now returns `True` for any non-NONE zone. Legacy callers and SpikingReinforceBrain continue to work unchanged.

## Decision T3.4 — Graded ASH + STAM-driven habituation

**Decision**: replace the boolean `predator_contact` input to the new mechanosensation module with a graded `predator_contact_intensity: float ∈ [0, 1]`. Habituation kinetics ride on the existing STAM exponential-decay buffer via a new `predator_mechano` STAM channel; do NOT add a bespoke habituation-state field.

**Intensity formula**:

```text
predator_contact_intensity = max(0, 1 − manhattan_dist_to_nearest_predator / damage_radius)
```

- At Manhattan distance 0 (literally overlapping the predator) → 1.0
- At the damage-radius edge → 0.0
- Outside damage radius → 0.0 (caller may also short-circuit via `is_agent_in_predator_contact_for`)

**Habituation kinetics**:

- STAM default `decay_rate=0.1` → half-life ≈ 7 steps ≈ 10-15s at typical 1-2s/step. Matches Hilliard et al. 2005's "seconds-to-minutes" ASH adaptation timescale.
- The temporal-derivative output `predator_mechano_dintensity_dt` produced by the STAM channel falls out automatically. "Habituated" = small derivative + steady-state mean.
- No new decay parameter. No new state field. STAM owns this.

**The boolean `predator_contact: bool | None` stays in BrainParams** for the legacy `nociception` module path and for the reward calculator's existing contact penalty. The graded intensity is a separate field consumed only by the new `predator_mechanosensation` module.

**Alternatives considered**:

- *Bespoke `predator_contact_habituation_level` field tracked agent-side*: rejected — duplicates STAM's job; introduces a parallel habituation-state subsystem with its own decay constant to tune.
- *Defer habituation entirely*: rejected — ships a graded scalar that no temporal averaging touches; biologically less defensible than the chosen approach.

## Decision T3.5 — Distal chemo signal as placeholder

**Decision**: reuse the existing `env.get_predator_concentration(position)` exp-decay sum for the distal-chemo signal. Add a sibling alias `get_predator_sulfolipid_concentration(position)` that delegates to the same code path. Document the framing analogy to Liu et al. 2018 in the docstring; note that the decay constant is **not** literature-calibrated yet.

**Why a placeholder**: literature-calibrating the decay constant from Liu et al. 2018's plate-assay distances to grid-cell units is a research deliverable (involves picking a physical-cm-to-grid-cell mapping that also depends on agent step size, episode duration, etc.). Doing it in T3 risks committing to a calibration that the T6 env-fidelity tranche or T7 real-worm validation later has to revisit. Better to ship the alias + framing now and let T6/T7 own the parameter sweep.

The existing per-scenario `predator.gradient_decay_constant` config knob (default 10.0 cells per `configs/scenarios/foraging/lstmppo_small_klinotaxis.yml`) stays as the tuning lever.

**Alternatives considered**:

- *Add a separate `get_predator_sulfolipid_concentration` with its own decay constant tuned to Liu et al. 2018 distances*: rejected for T3 (deferred to T6/T7).
- *Keep both: existing `predator_concentration` as "raw distance proxy" + new `predator_sulfolipid_concentration` as the biologically motivated channel*: rejected — doubles the env surface and the BrainParams surface for a distinction the existing tests cannot tell apart yet.

## Decision T3.6 — STAM channel split

**Decision**: replace the single `predator` STAM channel with two channels — `predator_mechano` and `predator_distal`. Keep `predator` as a deprecated alias resolved by `resolve_active_channels` when a config selects a legacy `nociception_*` module.

**Channel definitions**:

- `predator_mechano` — `derivative_key="predator_mechano_dintensity_dt"`, `sensing_mode_attr="predator_mechano_mode"`. Activates iff env predator enabled AND active sensory_modules list contains a `predator_mechanosensation*` module.
- `predator_distal` — `derivative_key="predator_distal_dconcentration_dt"`, `sensing_mode_attr="predator_distal_mode"`. Activates iff env predator enabled AND active sensory_modules list contains a `predator_chemosensation*` module.
- `predator` (frozen-legacy alias) — `derivative_key="predator_dconcentration_dt"`, `sensing_mode_attr="nociception_mode"`. Activates iff env predator enabled AND active sensory_modules list contains a `nociception*` module.

The three channels are mutually exclusive in practice — a single config picks one family or the other, not both. If a config somehow lists both legacy and new predator modules, all three channels activate (no special-case rejection, just additional STAM dims).

**STAM dimension implications**:

- `compute_memory_dim(num_channels) = 2 * num_channels + 3` formula stays unchanged.
- Configs using only new channels see same dim as configs using only legacy (1 predator channel either way).
- Configs using both see +2 dims per extra channel (one weighted-mean + one derivative).
- `_StamSensoryModule.to_classical` already returns dynamic-shape array based on `resolve_active_channels`, so brain consumers absorb the dim change without code edits.

**Implementation pre-flight**: `grep -rn "num_channels" packages/quantum-nematode` to confirm no code path hardcodes the channel count. The exploration agent found none, but a defensive grep at implementation time confirms.

## Decision T3.7 — Reward stays env-coupled in T3

**Decision**: ship zero changes to `packages/quantum-nematode/quantumnematode/agent/reward_calculator.py`. The existing three reward modes (`default`, `gradient_only`, `gradient_proximity`) continue to call `env.get_predator_concentration(agent_pos)` and `env.is_agent_in_danger_for(agent_id)` directly.

**Why now**: routing reward through `BrainParams.predator_distal_concentration` (the new sensor channel's output) would make reward depend on the agent's *sensing mode* — agent-in-oracle vs agent-in-klinotaxis would learn against different rewards on identical env state. That breaks the cross-sensing-mode comparability the archived Phase 5 configs implicitly rely on, and breaks the "reward depends only on env state, not on sensor configuration" invariant every Phase 5 architecture comparison was built on.

**Carry forward to T4**: a new ablation row inside each `T4.*.predator_evasion` cell evaluates "existing `gradient_proximity` reward" vs "distal-chemo penalty + binary contact-damage trigger" reward variant. Biological-faithfulness vs training-signal tradeoffs belong in T4's matched empirical evaluation, not in a unilateral T3 reward-shape change. Concretely: a new `T4.0d` (sibling to the existing `T4.0a`-`T4.0f`) sub-task added to [phase6-tracking/tasks.md](../phase6-tracking/tasks.md) tracks this carry-forward.

## Decision T3.8 — Brain consumer migration

**Decision**: the 19 brains that use the `sensory_modules` pipeline migrate via config-level changes only (their `preprocess` methods stay untouched). `SpikingReinforceBrain` (which bypasses the pipeline) and `ConnectomePPOBrain` (whose sensor projection is hardcoded per T2's implementation) are intentionally NOT migrated in T3.

**Migration matrix**:

| Brain category | In T3 scope? | Migration path |
|---|---|---|
| 18 sensory_modules brains (mlpppo, lstmppo, crh, etc.) | Sample configs only | Sample T4-prep configs under `configs/scenarios/pursuit/` list `predator_mechanosensation_klinotaxis` + `predator_chemosensation_klinotaxis` in `sensory_modules` (klinotaxis is the biologically-preferred default; oracle / temporal variants available for ablation) |
| `SpikingReinforceBrain` (hardcoded 4-feature preprocess) | No | Keeps legacy `predator_gradient_*` consumption; documented as known asymmetry in T4 design.md |
| `ConnectomePPOBrain` (hardcoded single-channel food projection) | Populate-but-don't-consume | T3 ensures new BrainParams fields are populated; T4.0c adds `predator_gains` projection matrix |
| 18 sensory_modules brains running existing predator configs (Phase 5 archived) | Not touched | Frozen-legacy path; `nociception_*` modules continue to work byte-identically |

**Quantum brains via `to_quantum()`**: the new modules implement `to_quantum` returning `[rx, ry, rz]` matching the existing transform-type discipline. Adding a quantum brain to a config that lists the new modules costs only as much as computing `to_quantum(params)` — no new wiring.

## Decision T3.9 — Test bar

**Decision**:

- **MUST**: unit tests for each new module's `extract` function (feature shape + value range + agent-heading-relative orientation). Existing convention at [test_modules.py](../../../packages/quantum-nematode/tests/quantumnematode_tests/brain/test_modules.py).
- **MUST**: env-level tests for anterior/posterior/lateral zone discrimination. 32 parameterised cases (predator at each of 8 surrounding cells × agent heading at each of 4 cardinal directions). New file `tests/quantumnematode_tests/env/test_predator_contact_zone.py`.
- **MUST**: legacy-config-load regression. Every archived `configs/evolution/*klinotaxis*.yml` (and any other config that names `nociception*` modules) must LOAD without error. New file `tests/quantumnematode_tests/utils/test_legacy_nociception_configs_load.py`.
- **SHOULD**: integration smoke through `scripts/run_simulation.py` end-to-end with new sensors on a small headless config. ~50 LOC. `@pytest.mark.smoke`.
- **MAY**: heuristic-predator-brain interaction smoke (agent approaches predator → graded distal rises monotonically → contact triggers zone flag). Could live as a logbook screenshot rather than a CI test.

## Decision T3.10 — Scope boundary

**Decision**: 10 T3 sub-tasks (env zone → distal alias → BrainParams → agent population → STAM split → new modules → SensingConfig fields → sample configs → tracker/docs → logbook section). Estimate 1.5-2 weeks for implementation + verification.

The original phase6-tracking T3.1-T3.6 (6 sub-tasks) expanded to 10 because the biology-driven scope split into "contact mechanosensation" and "distal chemosensation" channels created additional layers (STAM split, two sensing modes, two module families) rather than the original single-sensor swap. Still within the 1-2 week T3 budget because:

- Most layers are mechanical edits to existing registries (modules, STAM, SensingConfig).
- ConnectomePPO + SpikingReinforce are explicitly out of scope (deferred to T4.0c and a future small refactor respectively).
- Reward calculator stays untouched.
- Existing tests for the legacy `nociception_*` path don't need modification (legacy path is byte-frozen).

## Modelling caveats — recorded for future readers

These are documented intentionally so future tranches know they're known limitations, not oversights:

1. **ContactZone is bearing-vs-heading on a 1-cell agent, not anatomical head-vs-tail.** Biologically accurate behaviourally; not anatomically literal. T5/T7 may revisit when continuous-2D body extent exists.
2. **Distal-chemo decay constant is the placeholder exp-decay sum, not Liu et al. 2018 calibrated.** T6/T7 owns the parameter sweep.
3. **SpikingReinforceBrain stays on legacy oracle predator gradient.** Known asymmetry in the T4 predator-evasion comparison — flagged in T4 design.md.
4. **Reward stays env-coupled.** The biological-faithfulness reward shape lands as a T4 ablation, not as a T3 unilateral change.
5. **ADL ascaroside pheromone channel is deferred to T6/T7.** The `pheromone_alarm_ascr` module name is reserved for that work.

## Open questions

None at design-finalisation time. All decisions above are concrete and committed.
