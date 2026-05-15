# Design: Transgenerational Memory

## Context

Phase 5's transgenerational-memory milestone replicates C. elegans pathogen avoidance inheritance computationally. The biological mechanism: F0 worms exposed to pathogenic lawns (e.g. PA14) produce sRNA signals that survive into F1/F2/F3 offspring and bias avoidance behaviour without re-exposure. The literature consensus is "real but fragile" (Akinosho/Vidal-Gadea 2025); decay envelope from Murphy lab quantitative work (Kaletsky 2025) places F2 ≈ 0.5–0.6 of F0 on PA14 lawns.

The adjacent computational literature does not, to our knowledge, replicate this specific phenotype with a calibrated multi-generation decay envelope. Epigenetic evolutionary computation (Turner et al. 2023 *RSOS*; Stolfi & Alba 2018 EpiGenetic Algorithms; arXiv:1903.03854) implements methylation-style silencing layers for optimisation, not biological-phenotype replication. Lamarckian neuro-evolution (Sasaki & Tokoro 1999; Luo et al. 2024 arXiv:2403.19545; Sci Rep 2023 robot-evolution paper) passes weights parent→child without modelling an abstract decaying bias signal. High-fidelity C. elegans neuromechanical simulators (BAAIWorm Nat Comp Sci 2024; modWorm; CeDNe bioRxiv 2025) are single-generation. M6's honest novelty framing is therefore "first abstract-substrate computational replication of a specific transgenerational behavioural-inheritance phenotype with a biologically-calibrated multi-generation decay envelope" — three load-bearing qualifiers that carve uncontested territory without overclaiming relative to the epigenetic-EC line.

**Current state.** M3 shipped `LamarckianInheritance` + `WeightPersistence` (bit-exact tensor round-trip across 18 LSTMPPO weight tensors). M4 closed STOP (substrate constraint on Baldwin). M5 closed STOP (architecture asymmetry). Both single-population evolution loop (`EvolutionLoop`) and multi-agent runner (`MultiAgentSimulation`) are stable. The existing `PredatorType.STATIONARY` toxic-zone entity (env/env.py:91–95) is functionally a pathogen lawn — `speed=0`, larger damage radius, perceived via nociception sensor.

**Constraints.**

1. *Decision-gate calibration sensitivity.* The literature gate (F1 ≥40%, F2 ≥25%, F3 ≥15% of F0, monotone non-increasing) requires F0 baseline to land in a calibratable envelope; floor or ceiling F0 makes the gate uninterpretable.
2. *Confounder elimination.* Any signal carried by implicit channels (shared seeds, common training history, hyperparameter persistence) would mask the substrate's true contribution. M6.6's TEI-on/off ablation must be a one-bit difference.
3. *Brain Protocol stability.* M6 must not cascade signature changes through 19 brain subtypes.
4. *Compute envelope.* Must fit within ~20 wall-hours total to stay within prior Phase 5 milestone norms.

**Stakeholders.** Phase 5 tracker (openspec/changes/phase5-tracking/tasks.md M6.1–M6.8); Phase 5 synthesis logbook (future M8); roadmap (docs/roadmap.md Phase 5 milestone table).

## Goals / Non-Goals

**Goals**

- A substrate (`TransgenerationalMemory`) that carries an inheritable behavioural-bias signal independent of trained weights.
- A `TransgenerationalInheritance` strategy slotting into `EvolutionLoop` via the existing `InheritanceStrategy` Protocol with zero behaviour change to `none`/`lamarckian`/`baldwin` paths.
- Per-generation `lawn_schedule` toggling F0 (pathogen on, training on) → F1/F2/F3 (pathogen off, training off) inside a single `EvolutionLoop` invocation, so paired-arm bookkeeping stays clean.
- A measurable per-agent `pathogen_choice_index` (aggregator alias) — derived from existing per-step damage-radius detection (`env/env.py:2286 is_agent_in_damage_radius_for`). The underlying storage uses existing predator-avoidance trace columns; "pathogen_choice_index" is the aggregator output name, not a new metric.
- A hard pre-flight F0-calibration smoke gate that prevents wasted compute on an uncalibratable decision gate.
- A paired-arm ablation (`transgenerational.enabled: true|false`) where the substrate is the *only* cross-arm difference.

**Non-Goals**

- Concrete biological-protein tying (maco-1, Pfs1, vab-1, P11, Argonaute, SET-24, HCF-1). The abstract substrate is the entire mechanism.
- Multi-pathogen channels. Single sRNA-token only.
- Brain-architecture evolution (NEAT — separate milestone M7).
- Lamarckian weight flow within M6. Pure-TEI vs `NoInheritance` is the milestone ablation. A `transgenerational+weights` `kind()` is a staged follow-up triggered only if F0→F1 collapses to chance under pure-TEI.
- Training in F1+. `ppo_train_episodes=0` for inheriting generations is the experimental design.
- New env entity, sensory channel, reward path, or optimiser.

## Decisions

### D1. Substrate shape — per-action additive logit bias of shape `(num_actions,)`

The substrate is a single `torch.Tensor` of shape `(num_actions,)` (`num_actions = 4` by default: `FORWARD / LEFT / RIGHT / STAY` per `brain/actions.py:20 DEFAULT_ACTIONS`; LSTMPPO `num_actions=4` per `brain/arch/lstmppo.py:320`), added to actor logits at every step before softmax. Clamped to `|x| ≤ 2.0` in `__post_init__` (Boltzmann ratio cap ≈ 7.4×). The substrate shape is read from the runtime `brain.num_actions` so the design remains correct if a future config selects the 6-action set.

*Why.* Three options considered:

1. *Learned 32-dim vector injected into LSTM input* — high capacity, but couples to LSTM dynamics and loses the linguistic clarity of "bias offspring toward avoidance."
2. *Scalar coefficient on a fixed avoidance direction* — too rigid; cannot express compound behavioural shifts.
3. *Per-action additive logit bias* — minimal, interpretable, exactly matches the biological framing. ~12 bytes per genome; structurally overflow-impossible across thousands of generations. Mathematically equivalent to a per-action probability multiplier after softmax (log-domain Boltzmann tilt). Composable with arbitrary actor outputs.

We pick (3). Kaletsky's F2 ≈ 0.55 choice index corresponds to ~3× action-probability tilt; the 7.4× cap leaves headroom.

### D2. Generation-boundary decay; F0 telemetry extraction

`TransgenerationalInheritance.select_parents` returns the top-1 elite per generation (lex-tie-broken on `genome_id`), mirroring `BaldwinInheritance` (`evolution/inheritance.py:62-145`). The substrate is *extracted* from the F0 elite's policy via a post-fitness telemetry pass: run the F0 elite over fixed probe positions near a pathogen lawn (deterministic RNG seed) and record action-probability bias relative to a no-lawn baseline. Decay applied multiplicatively at `inherit_from()`: `child.bias = parent.bias * decay_factor`. With `decay_factor = 0.6`: F0 = 1.0, F1 = 0.6, F2 = 0.36, F3 = 0.216 — matches biological "diluted across generations."

*Why decay at generation boundary, not at `prepare_episode()`.* Per-episode decay would re-decay the same parent's substrate every episode — wrong. Generation-boundary decay matches biological semantics and keeps `prepare_episode()` stateless w.r.t. lineage depth.

*Why telemetry-pass extraction separate from fitness eval.* The choice-index metric is computed from raw episode traces; telemetry runs on deterministic probe positions *after* fitness eval. The two metrics are computed from disjoint data, asserted in tests.

### D3. Pathogen entity — repurpose `PredatorType.STATIONARY`

`PredatorType.STATIONARY` already exists at `env/env.py:91-95` ("Does not move, acts as a toxic zone with larger damage radius") with a dedicated `speed=0` early exit at `env/env.py:590-591`. It *is* a pathogen lawn by construction.

*Why repurpose, not add new entity.* The nociception sensor + aversive-gradient perception is wired and unit-tested for stationary toxic zones. A new entity would duplicate physics + reward path. We use the existing `predators:` YAML block with `predator_type: stationary` — no new config-loader keyword, no new env code, no new sensory channel; "pathogen lawn" is documentation vocabulary only.

*Trade-off.* Vocabulary drift in lineage CSVs ("predator" not "pathogen"). Mitigated by aggregator alias columns (`pathogen_choice_index` parallel to underlying `predator_avoidance`); storage uses existing plumbing. Logbook 018 § Methodology documents the repurposing.

### D4. Per-generation `lawn_schedule` inside a single config

Single YAML config drives F0→F3 via a `lawn_schedule: [{generation, pathogen_lawns_enabled, ppo_train_episodes}, ...]` block. `EvolutionLoop.run()` consumes the schedule per-gen just before `optimizer.ask()` (`evolution/loop.py:541-546`) and rebuilds env config from the per-gen entry.

*Why single config over chained configs.* Four chained configs would require campaign-shell complexity to chain F0 → F1 → F2 → F3 and pass the substrate across processes. A single config keeps it inside a single `EvolutionLoop.run()` invocation, simplifies resume semantics, and keeps paired-arm bookkeeping clean.

*Branch gating.* The schedule consumer is wrapped in `if cfg.transgenerational is not None:` so the no-op path is byte-equivalent to current behaviour. Precedent: `NoInheritance` vs `LamarckianInheritance` use the same gating pattern.

*Genome birth_metadata is gen-specific under TEI.* `Genome.birth_metadata = build_birth_metadata(sim_config)` is computed per-child inside `EvolutionLoop` ([loop.py:142-146](packages/quantum-nematode/quantumnematode/evolution/loop.py#L142-L146)). Because the per-gen `sim_config_for_gen` differs across generations under TEI (`predators.enabled` toggles, `learn_episodes_per_eval` toggles), each generation's children record gen-specific config in their birth metadata. This is desired forensic provenance — F0 birth_metadata reflects pathogen-on training conditions; F1+ birth_metadata reflects pathogen-off frozen-eval conditions. Logbook 018 readers can reconstruct each generation's experimental conditions from the lineage CSV without consulting the YAML.

### D5. Choice index — fraction of episode steps outside damage radius

`choice_index = 1 - (steps_inside_damage_radius / total_steps)`, per agent × per episode, mean-aggregated across all agents × all episodes per generation. Reuses existing per-agent damage detection at `env/env.py:2286 is_agent_in_damage_radius_for(agent_id)` (single-agent helper at `:2282 is_agent_in_damage_radius`). The aggregator records the per-step boolean alongside existing per-step trace data; no new env method or reward path is required.

*Why time-out over leaving-events.* Time-outside-lawn integrates over the whole episode and is robust to short transient incursions, matching how lawn-avoidance is measured in wet-lab literature. Kaletsky et al. eLife 2025 use this convention; their F2 ≈ 0.5–0.6 anchors the decision gate.

*Honest comparability caveat.* Kaletsky's wet-lab choice index is computed via a two-lawn assay (pathogenic vs non-pathogenic) — worms choose between options. Our single-lawn time-outside-damage metric is approximate, not exact, to Kaletsky's convention. The decision-gate ratios (F1 ≥40%, F2 ≥25%, F3 ≥15% of F0) compare *within* our metric across generations, so the relative-retention contract is well-defined; absolute numerical comparability to wet-lab is approximate. Logbook 018 § Methodology SHALL document this distinction.

### D6. Pure-TEI vs `NoInheritance` ablation (M6.6)

TEI-on arm: `inheritance: transgenerational`. TEI-off arm: `inheritance: none`. Same seeds, same env. The substrate is the *only* cross-arm difference; substrate signal cannot be confounded by weight flow.

*Why pure-TEI, not paired with Lamarckian.* Mixing TEI with Lamarckian weight flow would make the substrate vs. weight-flow contribution un-attributable. Pure-TEI is the cleanest single-bit ablation. If F0→F1 collapses to chance in the TEI-off arm (because F1 cannot rescue an untrained policy with bias alone), schedule a staged follow-up with a new `kind() == "transgenerational+weights"` — explicitly out of scope for this milestone.

**Escalation criterion for the staged follow-up.** The staged `transgenerational+weights` follow-up SHALL be triggered iff, in the pilot or full campaign, the TEI-off arm's F1 mean choice index is below `0.10` (within RNG noise of the chance floor on a 4-action policy) AND the TEI-on arm's F1 mean choice index is also below `0.40 × F0_mean` (i.e. the substrate alone cannot rescue F1 to gate-passing levels). If only the TEI-off arm collapses but TEI-on is gate-passing, no escalation — the substrate carries the signal as designed. If both arms gate-fail by other means (F2/F3 thresholds), the verdict is STOP under M6, not escalation.

*Validator enforcement.* `transgenerational.enabled=true ⇒ inheritance=transgenerational`; `enabled=false ⇒ inheritance=none`. Implemented in `config_loader.py` Pydantic validator with a unit test.

### D7. Brain hook — attribute on brain instance, set via fitness.evaluate

The TEI prior is set by `LearnedPerformanceFitness.evaluate` in `quantumnematode/evolution/fitness.py`, NOT by the runner or the worker. The integration layer: `EvolutionLoop` builds per-child worker tuples with `tei_prior_source: tuple[Path, float, int] | None` (the `(f0_substrate_path, decay_factor, lineage_depth)` triple, or None). `_evaluate_in_worker` forwards `tei_prior_source` as a kwarg to `fitness.evaluate`, mirroring the existing `warm_start_path_override` / `weight_capture_path` forwarding pattern. Inside `fitness.evaluate`, immediately after `brain = encoder.decode(...)` (fitness.py:429) and BEFORE `_build_agent(...)` (fitness.py:460), the substrate is loaded, decayed `lineage_depth` times, and assigned to `brain.tei_prior` via `hasattr`-dispatch. The runner code (`StandardEpisodeRunner`, `FrozenEvalRunner`, `MultiAgentSimulation`) is unchanged — runners are TEI-agnostic. Single-agent and multi-agent paths apply the prior identically because both read `brain.tei_prior` from the same brain instance.

LSTMPPO reads `self.tei_prior` inside `run_brain()` (sampling forward pass at lstmppo.py:601, before softmax at line 602) AND inside `learn()` (training forward pass at lstmppo.py:747, before softmax at line 748). Both call sites add the same prior tensor to logits before softmax.

*Why both call sites, not just sampling.* PPO computes a probability ratio `exp(new_log_probs - old_log_probs)` over the policy-update batch. `old_log_probs` is the log-probability recorded at action-sampling time (under the biased distribution). `new_log_probs` is recomputed at update time from the current policy's forward pass. If the training forward pass omits the bias while the sampling forward pass includes it, the two log-probs reflect different distributions and the PPO ratio is systematically wrong — silently corrupting F0 training. Applying the bias in BOTH paths keeps the distribution consistent across sampling and update; the bias is a constant additive offset on the actor head, so its effect on the PPO objective is well-defined (gradients flow through the actor parameters; the bias itself is non-trainable additive state that shifts the policy's exploration distribution).

The `tei_prior` SHALL be constant across an episode (it is set once by `fitness.evaluate` post-decode and is not mutated during the step loop), which the call ordering at fitness.py:429 → fitness.py:460 enforces by construction. The implementer SHALL add an assertion at the top of `learn()` that confirms `self.tei_prior` is unchanged from the value seen during the most recent rollout episode whose data is being updated.

*Why attribute, not Protocol signature extension.* Extending `Brain.prepare_episode(tei_prior=None)` would cascade through 19 brain subtypes. The attribute approach keeps the Protocol stable: non-LSTMPPO brains default `tei_prior = None` and ignore it. Minimal blast radius.

*Why apply at every step, not just `prepare_episode()`.* The bias is a persistent additive offset on the actor head. Applying only at episode start lets LSTM dynamics drown it out within ~1 step. Persistence test in `test_lstmppo_transgenerational_prior.py` asserts elevated action-0 probability across 100 LSTM-rollout steps with `bias = +2.0` on action-0.

*Implementation note on `frozen=True` dataclass + `__post_init__` clamping.* `TransgenerationalMemory` is `frozen=True` to prevent cross-generation aliasing mutation. Frozen dataclasses cannot reassign fields directly inside `__post_init__`; the clamping pass SHALL use `object.__setattr__(self, "logit_bias", clamped_tensor)` (the canonical Python pattern for frozen-dataclass post-init mutation). Alternatively, a module-level factory function MAY pre-clamp before construction; pick whichever the implementer finds clearer at the time.

### D8. Hard pre-flight F0-calibration gate

Before M6.5 full campaign is unblocked, run an F0-only single-gen smoke (1 seed × pop 6 × ~50 episodes ≈ 30 minutes). Pass iff `0.45 ≤ mean F0 choice_index ≤ 0.85`.

*Why hard gate.* The decision-gate ratios (40%/25%/15% × F0) are uninterpretable if F0 is at the floor (~chance) or ceiling (~ceiling-saturated). A hard pre-flight gate prevents wasting ~16 wall-hours on an uncalibratable run. If F0 < 0.45, retune `damage_radius` and `ppo_train_episodes`. If F0 > 0.95, reduce damage radius or training episodes. The tracker's permissive caveat ("refining to tighter bounds during M6 design is sensible") explicitly authorises gate refinement post-calibration.

### D9. Compute envelope

- F0 calibration smoke: ~30 minutes.
- Pilot: 1 seed × pop 6 × 4 gens × ~100 ep/eval, paired ≈ **4 wall-hours**.
- Full: 4 seeds × pop 16 × 4 gens × ~100 ep/eval, paired ≈ **16 wall-hours**.

Aligns with prior Phase 5 milestone envelopes (Baldwin retry was ~30+ wall-hours). Pilot verdict is preliminary (single seed); full verdict gates logbook 018.

## Risks / Trade-offs

- **Action-prior numerical instability — strong bias collapses exploration.** → Clamp `|logit_bias| ≤ 2.0` in `__post_init__`. Unit test asserts clamp. Kaletsky F2 ≈ 0.55 corresponds to ~3× action-probability tilt; 7.4× cap leaves headroom.
- **sRNA-token capacity overflow across generations.** → Fixed shape `(num_actions,)`. Overflow structurally impossible; ~12 bytes per genome × thousands of generations fits in kilobytes.
- **F0 baseline at floor or ceiling — gate uncalibratable.** → Hard pre-flight calibration gate (D8). M6.5 cannot start until gate passes.
- **TEI-on vs TEI-off confound — implicit pathway carries signal.** → Config-loader validator forces the pairing (D6). Same seeds + same env + same schedule. The substrate is the only difference.
- **LSTMPPO + prior interaction — LSTM drowns prior after step 1.** → Apply at every step inside `run_brain()` before softmax (D7). Persistence test across 100 steps.
- **Pathogen-lawn vocabulary drift in artefacts.** → Aggregator alias columns; logbook 018 § Methodology documents the repurposing (D3).
- **Telemetry-pass extraction biases F0 measurement.** → Telemetry runs *after* fitness eval, on deterministic probe positions. Choice-index from raw episode traces, telemetry on disjoint probe trace. Test asserts disjoint data sources (D2).

## Migration Plan

The change is additive — no breaking changes to existing inheritance modes (`none`/`lamarckian`/`baldwin`) or to non-LSTMPPO brains.

**Rollout order (matches commit grouping):**

1. Scaffold `TransgenerationalInheritance` strategy stubs + Literal extensions (config_loader, inheritance.py).
2. Add `TransgenerationalMemory` dataclass.
3. Wire LSTMPPO `tei_prior` attribute + runner pass-through.
4. Fill in `TransgenerationalInheritance.inherit_from()` + checkpoint round-trip.
5. Add per-generation `lawn_schedule` consumer in `EvolutionLoop`.
6. Ship YAML config + campaign shell.
7. Ship per-gen evaluator + paired-arm aggregator.
8. (Post-experiment) publish logbook 018, tick tracker + roadmap.

**Rollback.** Each commit is self-contained. If a downstream commit fails, the preceding state is byte-equivalent to pre-change `main` for all non-TEI inheritance modes. The Pydantic validator + default `None` attribute path means partial-rollout states remain valid.

## Open Questions

None remaining. All six open questions raised during planning (substrate shape, ablation pairing, F0 calibration gate, brain hook style, compute budget, config-schedule granularity) were resolved with explicit user decisions before the OpenSpec scaffolding step. The one *deferred* question — whether to add a `transgenerational+weights` kind — is conditional on pilot results and explicitly out of scope for this milestone.
