# Proposal: fix-predator-sensing-biology

## Why

The simulation's predator-detection model is biologically wrong in four ways the *C. elegans* nociception literature is explicit about. The result is that every predator-evasion experiment we've run (including the 22 archived Phase 5 evolution runs) trains an agent against a sensor surface that doesn't match real-worm biology. Fixing this is a pre-requisite for the four upcoming Tranche 4 L2 predator-evasion cells (connectome / MLP-PPO / LSTM-PPO / NEAT-weights × predator_evasion); doing it before T4 means the L2 baselines consume the corrected signal from the start instead of needing to be re-run.

Specific corrections required:

1. **ADL is not a touch sensor.** ADL is the pheromone/ascaroside chemosensor (Jang et al. 2012, *Neuron*). The original tracker framing "ASH/ADL contact-based nociception" is biologically incoherent — ADL belongs on a pheromone branch, not a contact-mechanosensory branch.
2. **Pure-contact is biologically incomplete.** Liu et al. 2018 (*Nat. Commun.*) documents real distal-chemosensory predator detection in *C. elegans*: *Pristionchus pacificus* secretes sulfolipids that ASH + ASI detect at a distance, driving escape behaviour. Removing all distance sensing makes the model *less* biologically faithful, not more.
3. **A single nociception scalar collapses anterior-vs-posterior structure.** The canonical escape circuit (Pirri & Alkema 2012; Kawano et al. 2011) splits by body region: anterior contact (ASH / ALM / AVM) drives AVA/AVD/AVE → backwards reversal; posterior contact (PLM) drives AVB/PVC → forward acceleration. A single bool/scalar can't represent this.
4. **ASH is graded + adapts.** ASH calcium responses scale with stimulus magnitude and habituate over seconds-to-minutes (Hilliard et al. 2005, *EMBO J.*). The current binary `predator_contact: bool` loses both properties.

Pre-existing flag: the bug was originally identified in [Logbook 011](../../../docs/experiments/logbooks/011-multi-agent-evaluation.md) as "chemosensory at distance is biologically wrong for nociception", but that framing was incomplete — the full correction is the two-channel split documented here.

## What Changes

T3 lands a corrected, biology-driven two-channel predator-sensing model while preserving the 22 archived Phase 5 evolution configs as byte-loadable historical record. New surface:

- **Contact-mechanosensory channel** via a triple of new sensor modules — `predator_mechanosensation_oracle` (2-dim), `predator_mechanosensation_temporal` (2-dim), `predator_mechanosensation_klinotaxis` (3-dim) — mirroring the role-set of the existing `nociception` family (which is bare-named for its oracle variant by historical convention; T3 adopts an explicit `_oracle` suffix for new modules so the mode is self-describing). The channel emits `predator_contact_intensity: float ∈ [0, 1]` (graded, not boolean) and `predator_contact_zone: ContactZone ∈ {NONE, ANTERIOR, POSTERIOR, LATERAL}`. Zone is inferred from the predator's relative bearing vs the agent's heading on the grid (documented modelling simplification — see § Decision T3.3 in design.md). **Klinotaxis is the biologically-preferred variant**, matching the head-sweep sensing pattern Phase 5 already adopted; oracle + temporal are available for ablation but should not be the default selection in new configs.
- **Distal-chemosensory channel** via a triple of new sensor modules — `predator_chemosensation_oracle` (2-dim), `predator_chemosensation_temporal` (2-dim), `predator_chemosensation_klinotaxis` (3-dim) — mirroring the role-set of the existing `food_chemotaxis` family (same `_oracle` suffix convention as above). The channel emits `predator_distal_concentration: float` and `predator_distal_dconcentration_dt: float`. The underlying env signal reuses the existing exp-decay concentration sum (documented as a Liu et al. 2018 sulfolipid analogue — calibration deferred to T6/T7). **Klinotaxis is again the biologically-preferred variant.**
- **STAM split**: the single `predator` STAM channel splits into `predator_mechano` + `predator_distal` so habituation kinetics ride on existing exponential-decay temporal averaging (Hilliard et al. 2005 timescale matches STAM's default half-life ~7 steps).
- **Env extension**: new `env.get_agent_predator_contact_zone_for(agent_id) -> ContactZone` method. `is_agent_in_predator_contact_for` keeps its existing bool return for legacy callers.
- **SensingConfig extension**: new `predator_mechano_mode` + `predator_distal_mode` fields. Existing `nociception_mode` field is untouched (legacy configs continue to use it).

What stays **frozen-legacy** (kept exactly as-is, never auto-substituted):

- The `nociception` / `nociception_temporal` / `nociception_klinotaxis` sensor modules. New configs use the new module names; archived configs continue to load and behave byte-identically.
- The `predator_contact: bool` BrainParams field. Continues to power the legacy `nociception` module path and the reward calculator's contact penalty.
- The `predator` STAM channel. Resolved as a deprecated alias when configs select legacy `nociception` modules.

What stays **out of T3 scope** (deferred):

- `ConnectomePPOBrain` sensor-projection wiring (the `predator_gains` matrix onto ASH/ASI/ALM/AVM/PLM). Owned by T4.0c. T3 just populates the new `BrainParams` fields so T4.0c can consume them.
- `SpikingReinforceBrain` migration off its hardcoded 4-feature `preprocess` to the `sensory_modules` pipeline. Spiking-reinforce stays on the legacy oracle predator gradient; documented as known asymmetry in the T4 design.md.
- The `reward_calculator.gradient_proximity` reward path stays env-coupled and unchanged. A new T4 ablation row will compare existing reward against a proposed "distal-chemo penalty + binary contact-damage trigger" variant under matched empirical conditions.
- Literature-calibrated sulfolipid decay constant (Liu et al. 2018 plate-assay distance mapping) — deferred to T6/T7.
- ADL ascaroside channel (`pheromone_alarm_ascr`) — deferred to T6/T7 env-fidelity work.
- Reconsidering the `ContactZone` bearing-vs-heading proxy when T5's continuous-2D physics gives the agent real body extent.

## Capabilities

### New Capabilities

- `predator-sensing-biology`: biologically-grounded two-channel predator detection. Owns the six new sensor modules (`predator_mechanosensation_oracle` / `_temporal` / `_klinotaxis` and `predator_chemosensation_oracle` / `_temporal` / `_klinotaxis`, all wired through the `apply_sensing_mode` translation map), the new `ContactZone` enum, the new env methods, the new STAM channels, and the new SensingConfig mode fields. Distinct from the legacy `nociception` capability surface in `brain-architecture` which stays frozen.

### Modified Capabilities

- `brain-architecture`: gains five new `BrainParams` fields (`predator_contact_intensity`, `predator_contact_zone`, `predator_distal_concentration`, `predator_distal_dconcentration_dt`, `predator_mechano_dintensity_dt`). Legacy fields untouched. No change to the Brain Protocol surface; consumer brains pick up the new fields via the existing `sensory_modules` pipeline. The `predator_mechano_dintensity_dt` field carries the STAM-computed temporal derivative of `predator_contact_intensity` for the mechanosensation temporal + klinotaxis modules, independent of the legacy `predator_dconcentration_dt` field which legacy `nociception_*` modules continue to read.
- `environment-simulation`: gains the `ContactZone` enum, `get_agent_predator_contact_zone_for(agent_id)` method, and the `get_predator_sulfolipid_concentration` alias (delegates to existing `get_predator_concentration`).
- `configuration-system`: `SensingConfig` gains `predator_mechano_mode: SensingMode` and `predator_distal_mode: SensingMode`. Existing `nociception_mode: SensingMode` untouched.
- `short-term-associative-memory`: STAM `CHANNEL_REGISTRY` gains `predator_mechano` + `predator_distal` channels. Legacy `predator` channel kept as deprecated alias.

## Impact

**Affected code** (T3 owns):

- `packages/quantum-nematode/quantumnematode/env/env.py` — `ContactZone` enum, zone-discrimination method, sulfolipid alias
- `packages/quantum-nematode/quantumnematode/brain/arch/_brain.py` — new BrainParams fields
- `packages/quantum-nematode/quantumnematode/agent/agent.py` — populate new fields in `_create_brain_params` (or equivalent)
- `packages/quantum-nematode/quantumnematode/agent/stam.py` — channel registry split
- `packages/quantum-nematode/quantumnematode/brain/modules.py` — register six new modules `PREDATOR_MECHANOSENSATION_ORACLE` / `_TEMPORAL` / `_KLINOTAXIS` + `PREDATOR_CHEMOSENSATION_ORACLE` / `_TEMPORAL` / `_KLINOTAXIS`; extend `apply_sensing_mode` map
- `packages/quantum-nematode/quantumnematode/utils/config_loader.py` — `SensingConfig` gains two new mode fields
- 2-3 new small predator-evasion smoke configs under `configs/scenarios/pursuit/` (mlpppo + lstmppo small variants; connectomeppo only if cheap pre-T4.0c)
- New tests:
  - `env/test_predator_contact_zone.py` — 32 parameterised cases for zone discrimination
  - `brain/test_predator_modules.py` — module-extract feature shape + value range
  - `utils/test_legacy_nociception_configs_load.py` — regression that 22 archived configs LOAD without error

**Affected docs**:

- `openspec/changes/phase6-tracking/tasks.md` — tick T3.1-T3.6 sub-tasks; add the T4 reward-ablation carry-forward row
- `docs/roadmap.md` — Phase 6 Tranche Tracker T3 row flipped to ✅ complete after merge
- A new section inside the T4 logbook stub at [`docs/experiments/logbooks/024-l2-first-pass.md`](../../../docs/experiments/logbooks/024-l2-first-pass.md) — per phase6-tracking T3.6, T3 verification rides alongside the L2 cells that consume it, not its own logbook. The T4 body lands below the T3 prerequisite section when T4 work begins.

**Behavioural impact**:

- Existing brains (19 sensory_modules-based + 1 connectomeppo) running existing configs see **zero change** — the legacy `nociception_*` modules are unchanged and the BrainParams legacy fields are unchanged. Brains running new configs (T4-prep) see two new sensor channels in their input.
- The 22 archived Phase 5 evolution configs continue to LOAD identically. Regression-tested.
- Predator-evasion behaviour under new configs will differ from under old configs — this is the intended biological correction, and the magnitude of the difference is what phase6-tracking T3.2 calls for ("quantify the behavioural difference under matched conditions") and what the T4 evasion cells will measure.

**Out of scope (deferred, with explicit pointers)**:

- ConnectomePPO predator-gains projection → **T4.0c**
- `gradient_proximity` reward vs distal-chemo+contact-damage reward ablation → **T4 evasion cells**
- Literature-calibrated sulfolipid decay → **T6 / T7**
- ADL ascaroside channel → **T6 / T7**
- SpikingReinforceBrain modules-pipeline migration → **future small refactor; not Gate-relevant**
- Continuous-2D revisit of the ContactZone bearing-vs-heading proxy → **T5 / T7**
