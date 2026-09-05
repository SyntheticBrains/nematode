# Design — L4 sanity-floor arms

## Context

Two reference points, no new mechanism beyond a switch. The interesting content is what the floors are *for* and what has to be held constant for them to mean anything.

The panel is not in scope. Neither is the rewired-null plastic arm or the matched-rule MLP.

## Decision 1 — the floors sit on the plastic substrate, not the gradient one

A frozen-weights arm can be configured two ways: the gradient rule with updates frozen, or the plasticity rule with updates frozen. They are not equivalent, and the difference is easy to miss.

Under the plasticity rule the motor readout is set to the anatomical contrast; under the gradient rule it keeps its random orthogonal draw. A floor built on the gradient rule would therefore run a **different decoder** from the arm it bounds, and any gap between them would confound decoding with learning — the exact confound the anatomical readout was introduced to remove. Whichever way that gap fell, it would be misread: a plastic arm beating such a floor might be beating a worse decoder, and one failing to beat it might be losing to a better one.

Both floors are therefore configured under the plasticity rule. The freeze is already total — no learning term, no decay, no clamp — so "frozen" means the weights are bit-identical to initialisation, not merely un-optimised.

The spec states this as a requirement rather than leaving it to the configs, because it is exactly the kind of constraint a later arm would violate by accident.

## Decision 2 — "vanilla rule" resolved as unmodulated Hebbian

D2 and D10 both require a vanilla-rule floor and neither defines it; D1's only nearby use of the word describes a rule family that was ruled out as unimplementable on this substrate. The definition is therefore made here.

The rule has two features that make it more than correlation learning: the neuromodulatory third factor and the decaying eligibility trace. Ablating the modulator gives `Δw = η·E` — pure co-activity learning — and that is the floor with the most diagnostic power, because it isolates the claim the shipment is actually making.

The reasoning is worth stating plainly, since the weaker floor would have looked adequate. Beating frozen weights shows only that *something* was learned. The 2×2 asks whether the wild-type wiring is legible **to a reward-driven rule**; if a wild-type advantage survived with the reward removed, the advantage would be a property of the wiring's correlation structure under any Hebbian process, which is a different and considerably weaker claim. Only the unmodulated floor can distinguish those.

Ablating the trace instead was considered and rejected as the primary floor: it still uses reward, so it bounds a narrower question ("does temporal credit assignment help?") and leaves the central one untested. It remains available as a later ablation if the panel's result makes it interesting.

**Implemented as a mode on the existing rule, not a second rule class.** The panel's arms must differ in one dimension; two classes cannot guarantee that eligibility accumulation, masking, decay, clamping and reporting stay identical, and those are precisely the paths a floor must share with what it bounds.

## Decision 3 — telemetry still reports the reward the unmodulated arm ignores

The prediction error and baseline are computed and reported in unmodulated mode even though the update discards them.

This costs one subtraction and buys a real property: both arms record what the reward stream was doing, and only one records having used it. The ablation becomes visible in the run's own telemetry rather than inferable only from a configuration flag — so a mislabelled arm, or a config that failed to apply, is detectable from its output. Given that this floor's entire job is to be a trustworthy reference point, that is worth more than the arithmetic it saves.

## Decision 4 — D2's wording is corrected; the initialisation is not

D2 pins the frozen baseline as "Cook-2019 **synapse-count-derived** initial weights". The implementation has never done that: the connectome determines which edges exist, weights along them are drawn `N(0, 1/√(chemical in-degree))`, and `syn.weight` never reaches `w_chem`. The in-degree that sets the scale is an edge tally, not a synapse count.

Two ways to close the gap, and the choice is not close:

- **Change the initialisation** to derive magnitudes from synapse counts. This would alter the substrate frozen four days ago under Amendment A and recorded against by Logbooks 029 and 034, invalidating the panel's own descriptive reference frame — inside a change whose purpose is to establish reference points. The worst possible moment to move one.
- **Correct the wording**, with a dated note recording what the substrate actually is. Chosen.

The substrate is honestly described as **anatomically constrained in topology, randomly initialised in weight**. That is a common and defensible modelling choice; the error was only in claiming more than it does. Whether weights *should* be synapse-count-derived is a real question, and a legitimate future change — one that would have to re-baseline everything measured against the current substrate, which is precisely why it cannot be smuggled in as a wording fix.

## Decision 5 — weight persistence is out of scope, and the note claiming otherwise was wrong

An earlier tracker note asserted the frozen-weights arm needs weight-component persistence. Checking it rather than inheriting it: the arm runs from the topology, the seed, and the freeze flag, none of which involve saved weights. The claim was an over-claim and is corrected.

The underlying gap is real and separately filed: the connectome brain implements no persistence, so `save_weights` is a **silent** no-op for it — the explicit `--save-weights` path fails loudly, but the automatic one writes nothing and says nothing. Plastic runs therefore discard the very synapses they modified. That blocks the imitation-warm-start arm, which needs to load weights, and it costs the panel the ability to inspect *how* the surviving weights differ. It does not block anything here.

## Risks

- **Both floors could be beaten trivially**, making them uninformative rather than wrong. That would itself be worth knowing before the panel runs, which is why they ship first.
- **The unmodulated floor might beat the plastic arm.** Not a failure of this change: it would mean reward modulation is hurting under the current hyperparameters, and it would be far better to learn that from a floor than from the flagship contrast.
- **`trace_decay` and the plasticity rate are still uncalibrated.** Both floors share them with the plastic arm by construction, so a bad value degrades all arms together rather than biasing the comparison — but the floors are the natural place to notice it.
