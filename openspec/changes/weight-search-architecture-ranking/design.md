# Design: Weight-Search Architecture Ranking on the Existing Grid Substrate

## Context

This change executes Phase 6 Tranche 4 per [`openspec/changes/phase6-tracking/tasks.md`](../phase6-tracking/tasks.md) — the first cross-architecture comparison sweep of the project. Inputs in hand from the just-merged Phase 6 work:

- **L1 plugin registry** (post-T2): all four MUST architecture families instantiate through the same `instantiate_brain(...)` code path, with no per-architecture branches in `setup_brain_model()`, simulation loop, or training loop. See [`docs/architecture/plugin-developer-guide.md`](../../../docs/architecture/plugin-developer-guide.md) and [`openspec/specs/brain-architecture/spec.md`](../../specs/brain-architecture/spec.md).
- **`ConnectomePPOBrain`** (post-T2): wild-type Cook 2019 hermaphrodite connectome with chemical-synapse strict-mask + fixed gap-junction weights, instantiable via `name: connectomeppo`. Klinotaxis sensing + canonical food projection shipped at T2; predator projection deferred to this change ([`fix-predator-sensing-biology` design.md Decision T3.8](../archive/2026-05-24-fix-predator-sensing-biology/design.md)).
- **Corrected predator biology** (post-T3): two-channel contact-mechanosensation + distal-chemosensation, ContactZone discrimination, graded contact intensity. Sample configs at `configs/scenarios/pursuit/{mlpppo,lstmppo}_small_predator_biology_klinotaxis.yml`. Logbook 024 documents the convergence-rate gap vs legacy (MLPPPO 3% vs 51% at 100 ep) flagged as Modelling caveat 6 of the predator-sensing change's archived design.md.
- **`FeedforwardGABrain`** (dependency: `add-neat-weights-brain`): the GA-based weight-search comparator on a matched-capacity feed-forward topology. Consumed in Phase 1a verification + Phase 4's GA C-cells.
- **Gate 1 R2b reference run** (logbook 023): connectome PPO on klinotaxis with `entropy_coef=0.005` eliminates the late-training drift the canonical `entropy_coef=0.02` produces. T4.0d locks `0.005` for all `T4.connectome.*` cells.
- **Reusable analysis utilities**: `scripts/campaigns/aggregate_m613_pilot.py:compute_cross_arm_delta_stats` (lines 329-418) provides paired-seed deltas, one-sided Wilcoxon, and 80% bootstrap CIs (1000 resamples, seeded RNG).
- **Existing combined-behaviour env**: `DynamicForagingEnvironment` already supports food + predators + thermotaxis + aerotaxis simultaneously (see `configs/scenarios/oxygen_thermal_pursuit/mlpppo_large_oracle.yml`). The Phase 4 cells use a subset (food + predator + thermotaxis, no aerotaxis per [`phase6-tracking/design.md § Decision 5`](../phase6-tracking/design.md)).

Outputs that gate downstream work:

- T7 (L2 re-run on the upgraded substrate) measures the env-upgrade delta against this change's grid baseline (Gate 3 G3.c).
- T9 (Phase 6 synthesis logbook) consumes the architecture ranking + connectome verdict for the headline framing decision (RQ3).
- Phase 6 Mid-phase decision Gate 3 consumes G3.a (cells at the Phase 5 statistical bar) and G3.b (connectome ranking).

Normative requirements live in [`specs/architecture-comparison-protocol/spec.md`](specs/architecture-comparison-protocol/spec.md), [`specs/connectome-ppo-brain/spec.md`](specs/connectome-ppo-brain/spec.md), and [`specs/predator-sensing-biology/spec.md`](specs/predator-sensing-biology/spec.md); this document records rationale and decisions.

## Goals / Non-Goals

**Goals:**

- Ship a paired-seed, integrated-behaviour, four-architecture comparison on the existing discrete-grid substrate with BH-FDR-corrected statistics, producing a publishable architecture ranking + connectome verdict.
- Settle the predator-sensing convergence-rate question (Phase 0 = T4.0g) before the comparison cells run, so the comparison evaluates a settled canonical sensor + reward surface rather than re-running cells if the investigation invalidates them.
- Lock the canonical `predator_gains` projection in `ConnectomePPOBrain` (deferred from T3) so the connectome cells have a documented sensor pathway for predator inputs.
- Establish the architecture-promotion gate (Phase 4.5) as an explicit written decision moment — informed by Phase 4 results + remaining budget + roadmap relevance — for whether to extend the comparison to SHOULD/MAY architectures before publishing.
- Produce the cross-cell analysis (paired-seed Wilcoxon + bootstrap CIs + BH-FDR) and publish the logbook with supporting/\* persistence.

**Non-Goals:**

- **Env-upgrade work**. Continuous-2D coordinates, continuous-action heads, Rung 2 chemical gradients, and log-concentration adaptation kinetics all ship in T5/T6 — strictly out of this change. This change is "L2 first pass on the existing grid substrate" verbatim per Decision 1.
- **Real-worm validation**. Deferred to T7. The grid substrate is below the fidelity threshold needed for defensible behavioural-number comparisons against published Bargmann chemotaxis indices or escape latencies.
- **NEAT topology search**. The GA cell in this change evolves weights on a fixed feed-forward topology (per `add-neat-weights-brain`'s design.md Decision 1). Genuine NEAT topology evolution ships at T8 with TensorNEAT.
- **Sensor-projection ablation as a planned deliverable**. T4.0c says "document the chosen mapping in design.md" — singular. Decision 7 reserves ablation as a contingent follow-up gated by Phase 4 evidence; this change ships one canonical predator projection without ablation.
- **SHOULD/MAY architecture cells as a Phase 4 deliverable**. Phase 4.5's gate decides whether to promote them; default is SKIP unless the gate's promotion criteria fire.
- **Compute-infrastructure changes** (parallel-execution framework, GPU scheduling, etc.). The existing per-config invocation of `scripts/run_simulation.py` is the launcher; the `nematode-run-experiments` skill orchestrates parallel groups where they fit.
- **MLP-PPO migration to GA, or LSTM-PPO migration to GA, or any other family-crossing variants**. Out of scope.

## Decisions

### Decision 1 — Curriculum-then-integrated cell structure per architecture (C1 → C2 → C3)

Each architecture's comparison row is a sequence of three cells: C1 foraging-only smoke (n=1), C2 foraging+predator smoke (n=1), C3 foraging+predator+thermotaxis primary cell (n ≥ 4). Only C3 carries the comparison; C1 and C2 are deliberately throwaway de-risking smokes.

**Why curriculum.** Per-architecture sensor-wiring failures surface in C1/C2 at small cost (single-seed, short budget) before the expensive n=4 C3 cells launch. Net cost over "combined only" is ~2 cheap runs per architecture = 8 extra runs total; far less than re-running an n=4 C3 (4 × the runtime) because of an undiagnosed sensor-pipeline bug. Phase 5 M4/M4.5 precedent established the pattern of spending compute on smokes to de-risk load-bearing cells.

**Why integrated, not per-behaviour.** Phase 6 Decision 5 fixes the three behaviours; the comparison's framing is "how well does each brain integrate competing pressures." Running three separate per-behaviour cells per architecture (12 total) would measure isolated competence, not integration — and would inflate the test count for the MCC correction. The existing `oxygen_thermal_pursuit` config family already demonstrates that the env handles three-behaviour integration in one config; this change adopts the same pattern minus aerotaxis (excluded per Decision 5).

**Alternative considered: "combined only" without C1/C2 smokes.** Rejected because the dependency chain (Phase 0 → Phase 1c/1d → Phase 4) means a sensor-wiring bug discovered in C3 would unwind days of compute. The curriculum is cheap insurance.

**Alternative considered: longer curriculum (C0 = no-environment-stimulus baseline, C1 foraging, C2 +predator, C3 +thermo, C4 + ablations).** Rejected as scope creep; the existing C1/C2/C3 catches the high-probability failure modes, and C0/C4 would not add proportional value.

### Decision 2 — Multiple-comparisons correction is BH-FDR within-pass, committed in this design.md before Phase 4 launches

Phase 6 Decision 6 defaults Gate 3's MCC to Holm-Bonferroni within-pass (12 tests per pass) but explicitly defers the choice to this change. This change commits to **Benjamini-Hochberg FDR (BH-FDR)** at α=0.05, applied to the active test set within Phase 4 (and inherited by T7 per Decision 6's "consistent strategy across T4 and T7" rule).

**Why BH-FDR over Holm-Bonferroni:**

- The cross-architecture tests are dependent (architectures evaluated on the same env, same seeds where possible). BH-FDR has higher power than Holm-Bonferroni under positive dependence (Benjamini & Yekutieli 2001); Holm-Bonferroni assumes independence and over-corrects.
- Phase 5 did not pre-specify any family-wise correction (M2/M3/M4 ran with raw paired-seed Wilcoxon + bootstrap CIs); Phase 6 introduces correction because the cell count is higher (and the cell count grows further if Phase 4.5 promotes SHOULD/MAY architectures). BH-FDR is the standard tool for the larger cell counts.
- Phase 5 M6.13 used `compute_cross_arm_delta_stats` without an MCC layer; this change wraps that function's outputs in a BH-FDR adjustment step.

**Why commit before Phase 4 launches:** Phase 6 Decision 6's recalibration mechanism explicitly forbids changing MCC mid-pass to forestall goalpost-moving. The Plan-agent's risk flag during this change's planning: cell variance is likely to differ visibly across architectures (GA population variance vs PPO bootstrapped variance) and the temptation to switch MCC after seeing the variance is real. Pre-commitment eliminates the option.

**Alternative considered: experiment-wide correction (Phase 4 + T7 = 24 tests).** Rejected because T4 and T7 are deliberate separate publishable passes per [`phase6-tracking/design.md § Decision 1`](../phase6-tracking/design.md). Within-pass correction respects that boundary.

**Alternative considered: no correction.** Rejected because at 12+ tests the per-test α=0.05 floor produces too many false discoveries to support Gate 3's "publishable architecture ranking" claim.

### Decision 3 — Architecture-promotion gate (Phase 4.5) is a written GO/SKIP decision per candidate SHOULD/MAY architecture

After Phase 4 C3 results land, decide per candidate (quantum, spiking, reservoir, hybrid per `phase6-tracking/design.md § Decision 4`) whether to add it to the comparison before publishing. Decision criteria:

- (a) Per-architecture compute fits the remaining budget after Phase 4 spend.
- (b) Promoted architecture is roadmap-relevant (e.g. quantum row addresses RQ4 — c302/NeuroML interop investigations may motivate a quantum baseline).
- (c) Phase 4 results don't already settle the headline (if connectome decisively wins on all three behaviours, adding spiking is unlikely to flip the framing).

Output: a written GO/SKIP verdict per candidate landed in this change's design.md. SKIP rationales should reference the specific Decision-4 SHOULD/MAY classification and the Phase 6 Decision 4 deferral mechanism (e.g. "spiking deferred to Phase 7 L4 STDP per Decision 4 SHOULD verdict; no Phase 6 substrate change makes spiking newly viable").

**Why an explicit gate, not a continuous decision:** Two failure modes to avoid — (i) silently extending the comparison without stating why (which leaves later readers unable to reconstruct the scope) and (ii) silently dropping a promotion mid-stream (same problem). A written gate forces the rationale to land in writing.

**Why per-candidate, not all-or-nothing:** The candidates have very different costs (spiking is expensive; reservoir is cheap), and the promotion criteria fire independently. All-or-nothing would force false bundling.

### Decision 4 — Canonical `predator_gains` projection in `ConnectomePPOBrain` (no ablation by default)

The `ConnectomePPOBrain` predator-sensor projection ships as one canonical mapping in Phase 1b: ASH + ASI sensory neurons consume the distal-chemosensation channel; ALM + AVM sensory neurons consume the anterior-zone contact-mechanosensation; PLM sensory neurons consume the posterior-zone contact-mechanosensation. Lateral-zone contact routes to ALM+PLM at half-weight (no canonical lateral-only mechanosensor neuron, but ALM+PLM together carry the lateral signal degenerately).

**Why these neurons:** Direct from the Bargmann literature already cited in [`fix-predator-sensing-biology` design.md](../archive/2026-05-24-fix-predator-sensing-biology/design.md): ASH is the canonical polymodal nociceptor (Hilliard et al. 2005); ALM/AVM are the anterior touch receptors (Pirri & Alkema 2012); PLM is the posterior touch receptor; ASI carries the Liu et al. 2018 distal sulfolipid signal.

**Why no ablation by default:** [`phase6-tracking/design.md § Decision 7`](../phase6-tracking/design.md) clarifies that sensor-projection ablations are T4-scope but reserves them as contingent follow-ups gated by Phase 4 evidence — not pre-planned. Pre-planning the ablation would inflate the cell count without an evidence-based motivation. If the connectome C3 cell underperforms and the projection is a candidate cause, the ablation rolls into a follow-up change.

**Alternative considered: ablate the projection upfront.** Rejected because each ablation variant doubles the connectome compute footprint and Phase 4's compute budget is the tightest of the change.

### Decision 5 — Phase 0 runs concurrently with Phase 1a + 1b; gates only Phase 1c + 1d

Phase 0 (predator-sensing convergence investigation) is config + eval work on existing MLPPPO. It runs concurrently with Phase 1a (verify `feedforwardga` integration — no code change) and Phase 1b (implement `predator_gains` projection — code work on `ConnectomePPOBrain` that touches no Phase 0 files). Phase 1c (connectome predator + thermotaxis configs) and Phase 1d (combined-behaviour configs per architecture) are gated by Phase 0's canonical sensor + reward output because those configs consume Phase 0's choices.

**Why concurrent.** Phase 0 has compute latency (matched-compute eval at 500 ep × n=4 seeds takes wall-clock days); blocking Phase 1a/1b on it serialises wall-time work that doesn't need serialising. Files touched are disjoint.

**Why gate 1c/1d on 0:** The whole point of running Phase 0 first is that its output is the input to those configs. Drafting Phase 1c configs before Phase 0 lands would require either guessing the canonical sensor/reward (and re-doing the configs after) or shipping configs that don't reflect Phase 0's findings.

### Decision 6 — Logbook supporting data persists at `docs/experiments/logbooks/supporting/`, never `tmp/`

Logbook artefacts need to remain reachable across machine states, deployments, and across the project's review history; `tmp/` is ephemeral and would silently break logbook citations the moment a working tree is cleared or moved. The scratchpads under `tmp/evaluations/weight-search-architecture-ranking/` are useful as in-flight working forensics during active evaluation, but any artefact a published logbook needs to cite belongs in version control under `docs/experiments/logbooks/supporting/`. This separation also keeps the logbook self-contained: a future reader can reproduce or audit a result by walking from the logbook → `supporting/` files → repository state at that commit, without needing the original author's local `tmp/`.

The formal normative requirement (which files must live where, and the prohibition on logbook → `tmp/` references) lives in [`specs/architecture-comparison-protocol/spec.md`](specs/architecture-comparison-protocol/spec.md) § "Logbook Supporting Data Persistence Discipline".

### Decision 7 — Per-architecture reward-weight tuning is allowed in C3 only with pre-C3 commitment + documentation (rationale)

The integrated C3 cells run three reward components simultaneously (food, predator, thermal). Per-architecture reward-weight tuning may be needed to prevent one component from dominating an architecture's learning (e.g. an LSTM may need lower predator weighting than MLP to prevent predator-aversion overwhelming foraging).

**Why allow tuning at all:** Forcing identical reward weights across architectures with very different learning dynamics conflates "this architecture is bad at the task" with "the reward shape was wrong for this architecture." Allowing tuning while documenting it preserves the comparison's honesty (the reader can see exactly what each architecture optimised against) without forcing artificially identical conditions.

**Why default to no tuning, and why escalate to n=2 C2 before tuning:** A single-seed C2 smoke is one noisy observation per architecture; routinely picking weights on it would fit noise and undermine the comparison's apples-to-apples-ness. Requiring both a large-magnitude C2 imbalance AND a second seed before tuning is allowed is the minimum evidence bar that justifies per-arch divergence.

**Why pre-commit the weights before C3 launches:** Per-architecture tuning is the symmetric bias source to mid-Phase-4 MCC switching (Decision 2). Both bias sources are forestalled by pre-commitment. Tuning + pre-commitment + documentation is fine; tuning + retuning + silence is not.

The formal normative rules (default = no tuning, tuning trigger requiring C2 \<50% vs another architecture AND ≥ n=2 C2 seeds, pre-C3 documentation, no mid-C3 retuning) live in [`specs/architecture-comparison-protocol/spec.md`](specs/architecture-comparison-protocol/spec.md) § "Per-architecture C3 reward-weight tuning discipline".

### Decision 8 — Canonical neuron-name targets, bilateral broadcast convention, and combined-config directory pre-decided

**Canonical predator-circuit neurons.** The Cook 2019 hermaphrodite connectome (consumed at [`packages/quantum-nematode/quantumnematode/connectome/neurons.py`](../../../packages/quantum-nematode/quantumnematode/connectome/neurons.py)) registers bilateral pairs with `L`/`R` suffixes as separate neurons. The canonical predator-circuit targets for the `predator_gains` projection are: `ASHL`, `ASHR`, `ASIL`, `ASIR` (distal-chemosensation, 4 targets); `ALML`, `ALMR`, `AVM` (anterior-contact, 3 targets — `AVM` is unilateral); `PLML`, `PLMR` (posterior-contact, 2 targets); `ALML` + `ALMR` + `PLML` + `PLMR` at half-weight (lateral-contact, 4 targets reusing the anterior + posterior gains). Gain-matrix shapes in `specs/connectome-ppo-brain/spec.md` reflect these counts.

**Bilateral broadcast convention.** Each L/R pair is counted as separate targets in the gain-matrix shape. The implementation MAY use one column per L/R member (identical initial values that diverge under PPO updates) OR a single shared column broadcast across L/R members — the choice is documented inline in `ConnectomePPOBrain` per the spec scenario.

**Combined-config directory split (PPO + GA cells live in different directories).** Phase 1d configs split across two directories deliberately, mirroring the existing `configs/scenarios/` vs `configs/evolution/` distinction:

- **PPO architectures** (MLPPPO, LSTMPPO, ConnectomePPO) — three combined configs live at `configs/scenarios/foraging_predator_thermal/{mlpppo,lstmppo,connectomeppo}_small_combined_klinotaxis.yml`. This directory parallels the existing `oxygen_thermal_pursuit/` noun-modifier compound.
- **GA architecture** (FeedforwardGA) — one combined config lives at `configs/evolution/feedforwardga_small_combined_klinotaxis.yml`. Evolution configs require an `evolution:` block per the `mlpppo_foraging_small.yml` precedent and are launched via `scripts/run_evolution.py`, NOT `scripts/run_simulation.py`. Putting the GA config under `configs/scenarios/` would break the launcher convention.

Rejected alternative 1: `combined_klinotaxis/` (`klinotaxis` is a sensing mode, not an env class, and breaks the existing taxonomy). Rejected alternative 2: `configs/evolution/foraging_predator_thermal/` for the GA variant only — too much directory nesting for one file. The deliberate split is documented here AND in Task 1d.2 so downstream readers know to look in two places for the four combined configs.

**Logbook number is 025.** Logbooks/ tops out at `024-predator-sensing-biology.md` (post-T3 merge); the next number is `025`. The Phase 5 logbook publication SHALL use `docs/experiments/logbooks/025-weight-search-architecture-ranking.md` and `docs/experiments/logbooks/supporting/025-weight-search-architecture-ranking/`. Tasks 0.8, 6.5, 6.6 use 025 (not the `0XX-` placeholder).

**Scope change vs phase6-tracking T4 (12 cells → 4 integrated cells).** The original `phase6-tracking/tasks.md` § T4 lists 12 per-behaviour cells (4 architectures × 3 behaviours). This change collapses to 4 integrated C3 cells (one per architecture, all three behaviours active) per Decision 1 + user instruction during planning. The 12-cell row structure in `phase6-tracking/tasks.md` is therefore stale once this change starts. Phase 3 will amend the T4 row structure to reflect the integrated-C3 pattern, applied in the same commit/PR as Section 3's MCC commitment — synchronous amendment matters because the tracker is read by future Phase 6 sessions to answer "what's the next tranche?" and a stale row structure mid-implementation would mislead them. Phase 5 will update Decision 6's G3.a wording ("All 12 MUST cells in T7…") to widen for the integrated-C3 pattern that T7 inherits from this change. (Tasks 1f.1, 1f.2, 6.10 carry the operational checklist.)

**Compute budget pre-estimate (sanity check before Phase 2).** Counting realistic floor:

- Phase 0: **~40 canonical-budget runs actual** (2 Step A pre-flight at n=1 + 20 Step B at n=4 × 5 variants + 12 Step B re-run after Bug 1 fix at n=4 × 3 variants + 8 extended validation at n=4 × 2 variants). Pre-estimate was 6+ variants × n=4 seeds = ~24 runs; the overshoot reflects the unplanned re-run after the Bug 1 fix landed mid-investigation + the orthogonal-stack extended validation. Wall-clock impact ~8-10h across parallel batches, within the same order of magnitude as the pre-estimate.
- Phase 2 pre-flight: 4 runs
- Phase 4 curriculum: 4 architectures × (1 C1 + 1 C2 + 4 C3 seeds) = 24 runs
- Phase 4 ablations: 1 strict-vs-soft × n=4 + 4 reward-shape × n=4 = 20 runs
- Phase 4.5 promotion (if any GO): +6 runs per promoted architecture

Floor without promotions: **~88 canonical-budget runs** (40 Phase 0 actual + 48 Phase 2/4/4.5 estimate). With one promotion: ~94. Decision 1's "4-6 weeks" estimate in [`phase6-tracking/design.md`](../phase6-tracking/design.md) predates this change's curriculum + ablation structure, so a sanity-check before any compute spend is warranted. Phase 2 will measure per-run wall-time and confirm the total fits the budget against the assumed parallelism factor; if projections exceed the "4-6 weeks" estimate by more than ~2× (whether driven by per-run wall-time growth, parallelism shortfall, or run-count growth), the team should amend Decision 1 rather than silently absorb the overrun. Task 2.4 owns the operational trigger arithmetic.

## Phase 0 canonical-variant selection

Phase 0 (Tasks 0.1-0.8, the predator-sensing convergence investigation deferred from `fix-predator-sensing-biology` Modelling caveat 6) ran 40 canonical-budget simulation runs across 6 sensor + reward variants at n=4 seeds × 500 episodes on MLPPPO small + klinotaxis sensing on the existing grid env with pursuit predators. The investigation surfaced two implementation bugs and ranked 6 candidate variants for the canonical predator-evasion sensor + reward locked for Phase 4.

### Bugs uncovered + fixed

- **STAM composite channel recognition** (commit `c25588a1`): the composite `predator_biology_klinotaxis` sensor module shipped without recognition in `agent/stam.py:resolve_active_channels`, causing the brain-side STAM dim (9, counting both new channels) to disagree with the env-side STAM dim (7, falling back to legacy `predator` channel) and crashing the first forward pass with `mat1 and mat2 shapes cannot be multiplied (1x16 and 18x64)`. Fix: explicit recognition of `predator_biology_klinotaxis` activates both `predator_mechano` and `predator_distal` STAM channels.
- **Bug 1: predator_lateral_gradient silent silencing under new-biology configs** (commit `65a5b517`): `agent.py:_compute_temporal_data` gated `predator_lateral_gradient` population on the legacy `nociception_mode == KLINOTAXIS`. New-biology configs set `predator_distal_mode: klinotaxis` (the new field name) but leave `nociception_mode` at its ORACLE default; the lateral gradient field was therefore silently None, and the chemo channel's directional `angle` feature collapsed to 0.0. **The brain literally had no head-sweep directional information about predators** despite the chemo channel claiming klinotaxis mode. This single bug explained essentially the entire 44pp convergence gap observed in Phase 0 Step A (A2 new biology 23% vs A1 legacy 67%). Fix: dual gate (legacy OR new) — both paths feed the same env-side `get_predator_concentration` field; the dual gate just removes the silent silencing of the new path.

Bug 2 (the documented `predator_sulfolipid_concentration` placeholder alias of `predator_concentration`) is explicitly deferred to T6/T7 per archived `fix-predator-sensing-biology/design.md` Decision T3.5. Post-Bug-1-fix the corrected biology is structurally bio-faithful (right neurons, right channels, right signal contracts) and outperforms legacy by 14pp; further bio-fidelity work is not blocking Phase 4.

### Final ranking (n=4 seeds × 500 episodes, last-25 mean success, all post-Bug-1-fix)

| Rank | Variant | last-25 success | death rate |
|---|---|---|---|
| 🏆 **1** | **B0.5 canonical sensors + `distal_chemo_contact_trigger` reward** | **81.0% ± 5.0** | **18.0%** |
| 2 | B0.6 sparse_fix sensor + `distal_chemo_contact_trigger` reward | 78.0% ± 6.9 | 22.0% |
| 3 | B0.3 sparse_fix sensor + default reward | 73.0% ± 6.0 | 27.0% |
| 4 | A1 legacy `nociception_klinotaxis` (reference) | 67.0% ± 7.6 | 32.0% |
| 5 | A2 canonical new biology + default reward | 65.0% ± 16.1 | 32.0% |
| 6 | B0.4 composite single-channel | 16.0% ± 7.3 | 83.0% |

### Canonical lock

The Phase 4 C-curriculum predator-evasion cells carry forward the following selected default (the normative `SHALL` lives in the spec's "Canonical variant selected (Phase 0 outcome)" scenario):

- **Sensors**: `predator_mechanosensation_klinotaxis` + `predator_chemosensation_klinotaxis` (the canonical two-channel new biology, biology-default per `fix-predator-sensing-biology` Decision T3.1)
- **Reward**: `reward_mode: distal_chemo_contact_trigger` (new dual-mechanism reward shipped in this change — continuous distal-chemo penalty via `env.get_predator_concentration` + binary contact damage trigger at `dist <= 1`)

**Rationale**:

- **Beats legacy by +14pp** (81% vs 67% last-25 success) with the **tightest variance** of all variants (5.0 std vs A1's 7.6). The corrected biology is fully viable when its directional signal is properly wired post-Bug-1.
- **Lowest death rate** of all variants (18% vs A1's 32%) — predator-evasion is meaningfully better, not just food-collection.
- **Biological fidelity**: uses the corrected two-channel sensors as designed (no sensor ablations), and the reward shape mirrors the dual-channel sensor split (continuous distal aversion via the chemo pathway + sharp contact pain via the mechano pathway).
- **Design simplicity**: only the reward mode differs from the A2 baseline; the sensor configuration is unmodified canonical biology.
- **Orthogonal stacking does not compound** (Finding 1 above: B0.6 sparse_fix sensor + new reward scores 78% vs B0.5's 81%, Δ=-3pp within noise). The sparse-fix sensor's "always-on distal fallback off-contact" becomes redundant once the new reward provides a continuous distal-chemo penalty — the brain sees the same information from two channels with no compounding gain. B0.5 is the simpler choice.
- **Composite single-channel is structurally inferior** (Finding 2 above: B0.4 score is 16% both pre- and post-Bug-1 — Δ=0pp). It dropped `lateral_gradient` from its 4-dim output by design, so Bug 1 never affected it, and the brain has no head-sweep directional info via the composite's encoding. Confirms `lateral_gradient` is load-bearing for klinotaxis predator-evasion. Composite is not viable.

### Carry-forward implications for downstream tranches

- **T7 (L2 re-run on upgraded substrate)** is expected to consume the same selected default (canonical two-channel sensors + `distal_chemo_contact_trigger` reward) unless the env-upgrade work plausibly invalidates the Phase 0 evidence. The normative carry-forward `SHALL` lives in `predator-sensing-biology` spec § "Canonical variant becomes the carry-forward for Phase 6 downstream work".
- **T6/T7 owns the deferred Bug 2 work** (literature-calibrated sulfolipid decay constant per Liu et al. 2018 plate-assay distances) per `fix-predator-sensing-biology/design.md` Decision T3.5. Post-Bug-1 the corrected biology is bio-faithful at the structural level; the sulfolipid calibration is an env-fidelity refinement, not a Phase 4 blocker.

Phase 0 forensics (per-variant per-seed CSV summaries + scratchpad) persist at `docs/experiments/logbooks/supporting/025-weight-search-architecture-ranking/phase-0/`.

## Risks / Trade-offs

| Risk | Trigger | Mitigation / Pivot |
|---|---|---|
| ~~Phase 0 fails to close the convergence-rate gap (no sensor variant + reward shape gets within 0.5σ of legacy at canonical T4 budget)~~ **RESOLVED** | After 4-6 Phase 0 sub-investigations, no variant clears the threshold | **Did not fire.** Phase 0 overshot legacy by +14pp (B0.5 canonical sensors + new reward 81.0% ± 5.0 vs legacy 67.0% ± 7.6). The mitigations (raise predator-cell budget; drop predator from this change) were not needed. See § "Phase 0 canonical-variant selection" — the convergence gap was a `predator_lateral_gradient` silent-silencing bug (commit `65a5b517`), not a substrate finding. |
| Connectome wall-time makes the C3 sweep infeasible | Phase 2 projects > 14 days wall-time at n=4 across the 4 C3 cells | Apply Decision 1 amendment mechanism: either (a) n=3 on the most expensive C3 cell with documented per-cell rationale, or (b) reduce `forward_pass_depth` from 4 to 3 on the connectome cell (degradation noted; stronger pivot). Do not silently extend schedule. |
| C1/C2 smokes pass on an architecture but C3 diverges (three-behaviour integration breaks one behaviour for that architecture) | After C1/C2 green on architecture X, X's C3 fails to learn or one behaviour collapses | Diagnose-and-fix on that architecture only. Per Decision 7, per-architecture reward-weight tuning is allowed with documentation. If unfixable, drop architecture from C3 and document the failure mode — better than misleading apples-to-oranges comparison. |
| MCC strategy revisited mid-Phase 4 because variance differs > 3× across architectures | Forbidden by [`phase6-tracking/design.md § Decision 6`](../phase6-tracking/design.md) once Phase 4 starts | Forestalled by Decision 2 above (pre-commitment to BH-FDR). If reviewers request Holm-Bonferroni instead, default to BH-FDR with written justification — do not allow mid-Phase 4 changes. |
| Phase 4.5 promotion criteria fire for an architecture mid-Phase 4 (compute headroom obvious before Phase 4 completes) | Phase 4 partial results suggest a SHOULD/MAY promotion is warranted | Defer the promotion to Phase 4.5's explicit decision moment regardless — running promotion mid-stream would silently inflate the comparison and break the MCC commitment's test count. Phase 4.5 happens after all Phase 4 C3 cells complete. |
| Connectome predator projection wiring (Phase 1b) introduces a regression on the Gate 1 klinotaxis baseline (logbook 023's R2b result) | Phase 1b smoke run shows last-25 mean klinotaxis success > 3 percentage points below R2b's reported baseline at the equivalent seed | Pin the projection to the predator pathway only; the food projection from T2 is untouched. The smoke bar is "within ±3pp of R2b" not "byte-identical activations" — adding `nn.Parameter` tensors perturbs PyTorch RNG-stream consumption order during `__init__` which can shift downstream `nn.init.normal_` draws on existing parameters by float-ulp amounts. If a >3pp regression appears, the projection has accidentally affected the food path — diagnose before proceeding to Phase 1c. |

## Migration Plan

This change is research/evaluation work; deployment is publication of the logbook + tracker updates. Rollback applies only to the code changes inside Phase 1b (`predator_gains` projection in `ConnectomePPOBrain`):

- The `predator_gains` projection is config-gated (active only when the brain config lists predator-sensing modules). Existing Gate 1 klinotaxis configs (no predator modules) are unaffected.
- If the projection causes test failures, revert the Phase 1b commit; the change continues with Phase 1c/1d configs that simply don't run on the connectome architecture until the projection lands cleanly.

No migration needed for the analysis scripts (additive under `scripts/analysis/`) or the new configs (additive under `configs/scenarios/`).

## Open Questions

1. **Phase 0 sensor + reward winners.** **Settled** — see § "Phase 0 canonical-variant selection" above. Canonical = `predator_mechanosensation_klinotaxis` + `predator_chemosensation_klinotaxis` sensors + `reward_mode: distal_chemo_contact_trigger`, beating legacy by +14pp last-25 mean success at n=4 × 500 ep. Phase 0 also uncovered + fixed two implementation bugs (composite STAM channel recognition; `predator_lateral_gradient` silent silencing) — the second was load-bearing and closed the entire convergence gap.
2. **Per-architecture reward weights for C3.** Settled per-architecture during Phase 4 with documented rationale per Decision 7. Some architectures may need none; others may need explicit tuning. Captured in design.md amendment + per-cell config commits.
3. **Compute-budget projection.** Settled at Phase 2; informs T4.0a in design.md.
4. **Phase 4.5 GO/SKIP per candidate architecture.** Settled at Phase 4.5; documented in this design.md.
5. **Whether to amend [`phase6-tracking/design.md § Decision 6`](../phase6-tracking/design.md) to reflect this change's BH-FDR choice as the canonical T4+T7 strategy.** Most likely yes; T7 inherits the strategy and Decision 6's "default Holm-Bonferroni" wording is now stale. Coordinated with `phase6-tracking` tasks.md update in Phase 5.
