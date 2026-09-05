# Add the L4 sanity-floor arms

## Why

The Phase 7 panel's primary contrast is plastic wild-type against plastic rewired-null. That contrast is only interpretable if a null result can be distinguished from a rule that never learned at all, which is what the two sanity floors exist to establish. They are not competitors to the plastic arm; they are the instruments that decide what its number means.

This matters more here than the phrase "sanity floor" suggests. The plasticity rule updates chemical synapses only, driving a **frozen** motor readout — an accepted risk recorded when that decision was ratified, with the frozen-weights floor named in advance as its detector. If the plastic arm fails to clear the floor, the finding is "the rule did not learn under this configuration", never "the wiring is inert". Fixing that reading before any run is the point; discovering afterwards that we cannot tell the two apart would leave the flagship uninterpretable.

## What Changes

### 1. The frozen-weights floor

Wild-type topology, initial weights, no learning. It answers "does the task reward anything at all above an untrained network?" and bounds the plastic arm from below.

The load-bearing detail is that it must sit on the **same substrate** as the arm it bounds. Under the plasticity rule the motor readout is set to the anatomical contrast rather than a random draw, so a frozen arm configured under the gradient rule would carry a *different decoder* — and any gap between the two arms would then confound the decoder with the learning. The floor is therefore configured as the plasticity rule with updates frozen, which the rule already honours totally: no learning term, no decay, no clamp.

No new mechanism is required. This is a configuration and a requirement pinning what that configuration must hold constant.

### 2. The unmodulated-Hebbian floor

The rule's distinguishing feature is its third factor. Removing it leaves `Δw = η·E` — pure co-activity learning, with the reward stream observed but never applied.

This is the sharper of the two floors, because it isolates the claim the shipment actually makes. A plastic arm that beats frozen weights has learned *something*; a plastic arm that beats unmodulated Hebbian has learned something **from reward**. Without this arm, a wild-type advantage could come from correlation structure in the wiring alone, which is a different and much weaker result than the one the 2×2 is designed to license.

Selected by extending the rule selection rather than adding a rule: the modulator is switched off inside the same update, so every other code path is shared with the arm it is compared against.

**The prediction error is still computed and reported when unmodulated**, and deliberately so. Both arms then record what the reward stream was doing, and only one records having used it — making "the reward was available and ignored" visible in telemetry rather than inferred from a config flag.

### 3. Roadmap D2's frozen-weights wording, corrected

D2 pins the frozen baseline as "wild-type topology, **Cook-2019 synapse-count-derived initial weights**, no learning". The implementation does not do that, and has never done it: the connectome supplies **which edges exist**; the weights along them are drawn `N(0, 1/√(chemical in-degree))`, and `syn.weight` — the actual EM synapse count — never reaches `w_chem`.

The wording is corrected rather than the initialisation. Changing initialisation would alter the substrate that Amendment A froze four days ago and that Logbooks 029 and 034 are recorded against, for no benefit to the question being asked — and it would do so inside a change whose purpose is to establish *reference points*, which is the worst possible moment to move one. A dated note records what the substrate actually is: anatomically constrained in topology, randomly initialised in weight.

### 4. Configuration

Two configs derived from the plastic wild-type arm by the minimal key delta, so the four arms of the floor comparison differ only where they claim to.

## Impact

- **Affected specs**: `learning-rules` gains the unmodulated mode and its telemetry semantics; `connectome-ppo-brain` gains the substrate-matching requirement the floors depend on.
- **Affected code**: the modulation switch in the three-factor rule; the rule-selection field and its validator; two configs.
- **Affected docs**: roadmap D2's dated correction; architectures and config vocabulary.
- **Default behaviour unchanged**: both floors are opt-in configurations. No existing config, weight file, or recorded result is touched.
- **Corrected, not built**: an earlier tracker note claimed the frozen arm needs weight persistence. It does not — the arm runs from topology, seed, and the freeze flag. The connectome brain's genuine lack of persistence is filed separately; it blocks the imitation-warm-start arm, not this one.
- **Not in scope**: the rewired-null plastic arm, the matched-rule MLP, the panel itself, and any run. This change ships the reference points and the requirements that keep them comparable.
