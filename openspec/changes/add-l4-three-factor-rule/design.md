# Design — minimal rate-based three-factor rule

## Context

This is the third of the three pieces the Phase 7 2×2 needs: the rule seam exists, the eligibility trace exists, and neither yet does anything. Two decisions were ratified with the user before this change was authored (recorded below as Decisions 2 and 3); the rest follow from constraints already in the specs.

The panel is **not** in scope. Nothing here licenses a claim about wiring; it ships the mechanism the arms will hold constant, plus the evidence that the mechanism is what it says it is.

## Decision 1 — a second rule behind the existing seam, not a new brain

Selection by config field on `ConnectomePPOBrain`, exactly as the wiring control and the std mode did.

The alternative — a `ConnectomePlasticBrain` — was already rejected once, when the PPO update was extracted, and the reasoning has since strengthened. The D10 panel compares arms that must differ in **one** dimension. Two brain classes cannot guarantee that: every divergence in sensing, buffering, episode lifecycle, or telemetry becomes a rival explanation for whatever the panel measures. One class with one switched component makes "everything else is held constant" a structural fact rather than a maintenance promise.

The cost is a brain class that now hosts two update regimes, and the honest consequence is that `ConnectomePPOBrain` is misnamed once this lands. Renaming touches the registry, configs, weight files, and four logbooks' worth of provenance; it is recorded as follow-up debt rather than smuggled into a change whose diff should stay readable.

## Decision 2 — the trace becomes temporally causal (ratified with the user)

`E ← λE + M∘(h_prev ⊗ h)`, superseding v1's `M∘(h hᵀ)`.

The v1 requirement pre-authorised this amendment and named the alternatives; this change takes the previous-step pre-synaptic option and dates it.

**Why it matters, stated precisely.** The adjacency mask already carried most of the directional information — an edge `i→j` with no reciprocal `j→i` accrued eligibility in one direction only, because the mask zeroed the other. What the symmetric form could not do was distinguish the two directions of a **reciprocal** pair, and it encoded no temporal order at all. A three-factor rule rests on the claim "this synapse participated in causing that activity"; an eligibility term computed from a single instant cannot express it. Using the previous step's settled state as the pre-synaptic factor makes the trace causal and, for reciprocal pairs, directional.

This is a sharpening, not a repair, and it is worth saying so: v1 would have produced a runnable panel. But the flagship asks whether *wiring* is load-bearing, and a rule whose credit assignment is blind to the direction of half the reciprocal edges would weaken exactly the claim the panel exists to support.

**Cost**: one extra `(n_neurons,)` buffer, reset with `E`; the first step of each episode contributes no eligibility; and the "inert until consumed" property must be re-verified, because the formula change touches the code path that guarantees it.

## Decision 3 — only the chemical synapses are plastic (ratified with the user)

`w_chem` changes; sensory gains, motor readout, and action-noise parameters do not.

The hypothesis is that the *connectome* learns. If the readout adapted too, a wild-type advantage could arise from a better-fitted decoder rather than from the wiring — the confound the 2×2 exists to exclude. Freezing everything outside the trace's support makes the contrast structural.

**The risk is real and is accepted with a detector.** The motor readout is a 4→2 orthogonally-initialised map from pooled motor activity; frozen, it is an arbitrary fixed decoder the plastic connectome must learn to drive. That may prove too hard, in which case both plastic arms sit at chance and the primary contrast is null *by incapacity* rather than by degeneracy — two very different findings that must not be confused.

The D2 sanity floors are exactly this detector: a rule that learns anything clears the frozen-weights floor. A plastic arm that fails to clear it reports "the rule did not learn under this configuration", not "wiring is inert". Recorded as the interpretation rule now, before any run, so it cannot be chosen after seeing results.

If the floor does fail, the recorded fallback is a documented variant extending the update to the readout — a different, weaker claim, taken deliberately and named as such rather than slipped in as a fix.

## Decision 4 — the modulator is a prediction error, not a reward

`δ = r − b`, with `b` an exponential moving average of reward.

Not a refinement. This project's reward streams are predominantly one-signed over long stretches (step penalties, proximity penalties), and `Δw ∝ r · E` under a persistently negative reward would drive every eligible synapse in one direction regardless of which behaviour earned it — the rule would measure reward *magnitude*, not reward *surprise*, and would be indistinguishable from a decay term with extra steps.

An EMA baseline is the minimal correct choice: no value head, no critic, no bootstrapping, one scalar of state. A learned critic would be strictly better at credit assignment and is strictly out of scope — it reintroduces the gradient machinery this rule exists to avoid, and its own learning dynamics would confound the wiring contrast.

## Decision 5 — per-step cadence

The rule steps once per environment step, when that step's reward arrives.

This is what makes it a *three-factor* rule rather than a batched Hebbian approximation: the neuromodulator's arrival time relative to the standing eligibility is the mechanism. Batching to a rollout would average modulators across steps whose traces have already decayed by different amounts, discarding the temporal alignment the trace was built to provide.

Consequence: the rollout buffer, GAE, and the value head are dormant under this rule. They are not removed — the same brain still serves the PPO arms — but the spec requires they go unexercised, so a future refactor cannot quietly couple them.

## Decision 6 — bounded plasticity (and why Dale's law is not imposed)

Hebbian rules diverge; this one ships with a decay term, a magnitude clamp, and saturation telemetry.

Decay and clamp are configurable and load-time validated. The telemetry matters as much as the bounds: a rule pinned at its clamp is still "learning" by any weight-change metric while having become a constant function, and a panel arm that saturated in its first hundred steps must be visible as such, not inferred afterwards from flat returns.

**Sign preservation was specified and then withdrawn at spec review**, and the reversal is instructive enough to record. The argument for it was biological: a chemical synapse's sign follows from its neurotransmitter, so an update flipping excitatory to inhibitory models a transition the animal cannot make. That argument is sound about *C. elegans* and false about *this substrate*. Initial chemical weights are drawn from a zero-mean distribution — each synapse's sign is an arbitrary draw, not a neurotransmitter fact. Enforcing Dale's law over random signs would freeze noise, and would prevent the rule from correcting a synapse whose initial sign was simply wrong. `Neuron.neurotransmitter` exists in the data model but no initialisation path consumes it.

The constraint is therefore dropped, and the prerequisite recorded: Dale's law becomes enforceable once synapse signs are neurotransmitter-derived, which is natural work for the receptor-grounded shipment that already plans atlas curation. Imposing it earlier would have been a plausibility claim the substrate could not support.

## Decision 7 — the frozen readout is anatomical, not random (ratified with the user)

Decision 3 freezes the readout. That only makes sense if what is frozen is defensible, and reconnaissance showed it was not: the readout is a **random orthogonal** 2×4 matrix over pools whose meaning is fixed anatomy.

The pools are `VB, DB, VA, DA` — the canonical *C. elegans* motor classes. `D` and `V` denote dorsal and ventral muscle innervation, so their difference is body bending; `B` and `A` denote forward and backward locomotion drivers, so their difference is translation. The action space is exactly `(speed, turn)`. The correct map is therefore not merely learnable — it is *derivable*, and it is two linear contrasts.

Under PPO this was invisible: the readout is a learnable parameter, so an arbitrary starting point costs some training and nothing else. Freezing it changes the situation completely — the plastic arms would be permanently asked to drive a decoder that scrambles the anatomical semantics of their own motor neurons. That is both less faithful than the animal (whose motor neurons drive muscle through fixed anatomy) and the largest single contributor to the incapacity risk named in Decision 3.

So under the three-factor rule the readout is initialised to the contrast: `turn ← dorsal − ventral`, `speed ← B-type − A-type`.

Three properties make this cheap:

- The contrast matrix is **orthonormal** — unit-norm rows, exactly orthogonal — so it is a *specific* orthogonal matrix, not a departure from the existing `orthogonal_(gain=1.0)` scale or conditioning. The initial policy is no more degenerate than before.
- The values are written **after** the existing orthogonal draw, so the RNG stream is unperturbed and every other parameter remains byte-identical across rule selections.
- It is **arm-neutral**: wild-type and rewired-null share an identical readout, so the primary contrast still isolates wiring.

PPO arms keep the random draw. Changing them would alter the initialisation of the Logbook 029/034 substrate — the panel's own descriptive reference frame, frozen four days ago under Amendment A — for no benefit to this change.

## Where this design sits on biological plausibility

Recorded because it is the standing question behind the whole shipment, and because the honest answer has parts that are strong and parts that are not.

**Load-bearing and genuine.** The rule is *local*: it uses pre-synaptic activity, post-synaptic activity, and one global scalar, with no backward pass, no weight transport, and no per-synapse error signal. That is the specific property PPO cannot claim and the reason this rule family is the right instrument for the wiring question. Around it: rate coding (defensible as higher fidelity than STDP for a graded, largely non-spiking animal), decaying eligibility, neuromodulatory gating, EM-derived connectivity, non-plastic gap junctions, anatomically-pooled motor classes, and now an anatomically-derived motor decoder.

**One credential is weaker than it is often stated, and spec review corrected it here.** The connectome supplies *which edges exist*; it does not supply their strengths. Initial chemical weights are drawn from a zero-mean distribution scaled by each neuron's chemical in-degree — the Cook-2019 synapse **counts** determine edge existence and enter the scale only as an edge tally, never as a magnitude. So the substrate is anatomically-constrained in its **topology** and randomly initialised in its **weights**. That is a defensible and common modelling choice, but "synapse-count-derived initial weights" overstates it, and the frozen-weights baseline the panel pins (roadmap D2) uses that wording — a discrepancy flagged for the change that builds that arm.

**Abstractions that are staged rather than ignored.** The modulator is one global scalar with no receptor specificity or spatial extent; the receptor-grounded stack is a named later shipment, and the panel is explicitly re-run under the grounded rule there.

**Gaps that remain, stated rather than glossed.** A plausible learning *rule* is not a plausible learning *problem*, and the problem carries most of the remaining distance: the reward is engineered shaping the animal has no access to, the baseline is effectively a reward oracle computed outside the network, and exploration is Gaussian action noise with a frozen scale — an RL artefact whose more plausible state-dependent form was attempted and failed its gate. None of these are introduced here, and none are fixable within this change, but they bound what the 2×2 can claim: it can say the wiring is or is not legible to a local, neuromodulated, rate-based rule. It cannot say the animal learns this task this way.

## Decision 8 — the value head is skipped, not stubbed (added at spec review)

Spec review found that "the value head SHALL NOT be exercised" was a declared outcome with no mechanism: the action path computes a state value on **every** step, through a property that delegates to the PPO rule. A three-factor rule owning no critic would therefore fail on the first action.

Two repairs were available. The rule could expose a null value head, keeping the call site alive and returning zeros — smaller diff, and wrong: it would give a rule that owns no critic the appearance of one, leave a dead tensor flowing through the action path, and make the spec's "no value head" claim true only by wording. Instead the value computation is **skipped** under the plastic rule, with the per-step and bootstrap value state left unset and the experience buffer left unappended.

The back-compatibility accessors for the value head and optimiser are kept, but under the plastic rule they raise an error naming the active rule rather than surfacing an attribute error from inside the delegation — the difference between "this rule has no optimiser" and a traceback that reads like a bug.

## Risks

- **The rule may not learn under a frozen readout** (Decision 3). Detected by the D2 frozen-weights floor; interpretation rule fixed in advance.
- **`trace_decay = 0.9` remains an uncalibrated placeholder.** A.2 declined to claim its biology and this change does not either: it is now load-bearing, so it becomes a named hyperparameter of the panel, tuned once against the sanity floors and then frozen across all arms — never per-arm, which would break D2's frozen-recipe rule.
- **Two update regimes in one class** (Decision 1) make the class harder to read; mitigated by the dispatch living in one place and by the byte-identity test on the default path.
- **Amending a shipped formula invalidates nothing measured** — traces have never been consumed, and no logbook result depends on `E`. Stated explicitly so the amendment is not mistaken for a substrate change of the kind Amendment A froze.
