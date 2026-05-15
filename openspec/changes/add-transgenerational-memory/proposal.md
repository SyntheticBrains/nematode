# Proposal: Add Transgenerational Memory

## Why

C. elegans inherits learned pathogen avoidance across generations via small-RNA signalling — F0 worms exposed to PA14 produce offspring that avoid PA14 lawns despite never seeing them directly, with retention decaying over F1/F2/F3 (Posner 2023; Kaletsky 2025 *eLife*; Akinosho/Vidal-Gadea 2025 *eLife*). The mechanism is biologically contested but defended: the Hunter lab failed to replicate (Gainey 2024 *eLife*), Murphy lab rebutted (bioRxiv 2024), and Vidal-Gadea independently validated F1+F2 retention in November 2025 with documented protocol-sensitivity. To our knowledge, no published computational model replicates this phenotype with a calibrated multi-generation decay envelope. Prior work on epigenetic evolutionary computation (Turner et al. 2023 *RSOS*; Stolfi & Alba 2018) and Lamarckian neuro-evolution (Sasaki & Tokoro 1999; Luo et al. 2024 arXiv:2403.19545) addresses adjacent mechanisms — methylation-style silencing for optimisation, or parent→child weight-copy — but none ties an abstract heritable bias signal to a specific biological decay curve (Kaletsky's F2 ≈ 0.5–0.6 of F0 anchor).

Phase 5's transgenerational-memory milestone is the next ready-to-start deliverable after M3 GO (Lamarckian inheritance ✅), M4 STOP (Baldwin), and M5 STOP (co-evolution). The substrate is ready: M3 shipped a proven heritable-substrate pattern (`WeightPersistence` + `LamarckianInheritance`, bit-exact across 18 LSTMPPO tensors); the existing `PredatorType.STATIONARY` toxic-zone entity *is* a pathogen lawn by construction; nociception sensing is wired and tested. The remaining gap is the abstract sRNA-style substrate that biases offspring action distributions independently of trained weights — exactly what TEI describes biologically.

## What Changes

- Add a new heritable substrate: `TransgenerationalMemory` dataclass carrying a per-action additive logit bias (`torch.Tensor` of shape `(num_actions,)`), extracted from F0 elite policies via a deterministic telemetry pass and multiplicatively decayed at generation boundaries.
- Add a new `InheritanceStrategy` Protocol implementation `TransgenerationalInheritance` with `kind() == "transgenerational"`, mirroring `LamarckianInheritance` / `BaldwinInheritance` patterns.
- Apply the TEI bias inside the LSTMPPO actor head: brain instances expose a `tei_prior` attribute (default `None`); the runner sets it before each episode; `run_brain()` adds the bias to actor logits before softmax at every step. Brain Protocol signature unchanged.
- Extend `EvolutionLoop` with a per-generation `lawn_schedule` that toggles `pathogen_lawns_enabled` and `ppo_train_episodes` per generation, enabling F0 (pathogen on, training on) → F1/F2/F3 (pathogen off, training off) experiment design.
- Add a config-loader `transgenerational` block (Pydantic `TransgenerationalConfig`, `LawnScheduleEntry`) with an `enabled` boolean ablation switch and `decay_factor` parameter. Validator enforces TEI-on ⇒ `inheritance: transgenerational`, TEI-off ⇒ `inheritance: none`.
- Repurpose existing `Predator(predator_type=STATIONARY)` as the pathogen-lawn entity by configuring the existing `predators:` YAML block with `predator_type: stationary` + `speed: 0` + larger `damage_radius`. "Pathogen lawn" is documentation vocabulary only; no new config-loader keyword, no new env entity, no new sensory channel, no new reward path.
- Add a per-generation choice-index evaluator and paired-arm aggregator (TEI-on vs TEI-off) producing the F0→F3 retention table against the decision gate F1 ≥40% × F0, F2 ≥25%, F3 ≥15%, monotone non-increasing decay.
- Add a hard pre-flight F0-calibration smoke gate (`0.45 ≤ F0 ≤ 0.85`) before the full M6.5 campaign is unblocked.

## Capabilities

### New Capabilities

- `transgenerational-memory`: per-action additive logit-bias substrate inherited across generations; serialised round-trip; multiplicative decay at generation boundary; telemetry-pass extraction from F0 elite policy; `apply_to_logits()` integration point.

### Modified Capabilities

- `evolution-framework`: extend `InheritanceStrategy` `kind()` literal with `"transgenerational"`; add per-generation `lawn_schedule` consumer in the loop; extend per-child inheritance resolution to return TEI substrate paths alongside weight checkpoint paths.
- `lstm-ppo-brain`: add `tei_prior` attribute (default `None`) read inside `run_brain()` and added to actor logits before softmax at every step; apply at every step (not just episode start) so LSTM dynamics cannot drown it out.
- `configuration-system`: add Pydantic `TransgenerationalConfig` and `LawnScheduleEntry` models; extend `EvolutionConfig.inheritance` Literal with `"transgenerational"`; add validator enforcing the TEI-on/off ↔ inheritance-strategy pairing.

## Impact

- **New code**: ~2400 LoC across 10 new files (dataclass, strategy, configs, campaign shell, two scripts, four test files).
- **Modified code**: ~217 LoC delta across 6 existing files (`brain/arch/lstmppo.py`, `agent/runners.py`, `evolution/loop.py`, `evolution/inheritance.py`, `utils/config_loader.py`, plus tracker tick in `tasks.md`).
- **Brain Protocol**: unchanged (additive attribute, not signature change). The `tei_prior` attribute is defined ONLY on `LSTMPPOBrain` (defaulted to `None`); non-LSTMPPO brains do not define the attribute. The worker dispatches via `hasattr(brain, "tei_prior")`, so non-LSTMPPO brains are observationally inert under any inheritance setting.
- **Existing inheritance modes**: no behaviour change. `inheritance: none/lamarckian/baldwin` paths remain byte-equivalent.
- **Env/reward path**: no change. Pathogen lawns repurpose existing `PredatorType.STATIONARY` plumbing.
- **Compute envelope**: pilot ~4 wall-hours paired (1 seed × pop 6 × 4 gens); full ~16 wall-hours paired (4 seeds × pop 16 × 4 gens). Aligns with prior Phase 5 milestone envelopes.
- **Decision-gate evaluation**: per-seed retention table aggregated cross-seed for GO/PIVOT/STOP verdict. User-reviewed before logbook 018 publication.
