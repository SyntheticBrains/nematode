# Design — matched-rule MLP arm

## Context

The arm that makes the ranking test quantitative. The interesting decisions are about what "matched" and "substrate-generic" have to mean in code for the comparison to be honest; the MLP-specific mechanics follow from those.

The panel is not in scope.

## Decision 1 — one rule over a seam, not a second rule

The roadmap's D10 states the rule is substrate-generic and withdrew the contrary claim. The implementation contradicts that: the rule names five connectome attributes. Two ways to add an MLP arm:

- **A second rule class for the MLP**, duplicating the update with MLP names. Small diff, and it would quietly make "matched rule" false — two implementations of the same equation drift, and the panel would be comparing arms trained by code that merely started out identical.
- **One rule reading a Protocol seam.** Chosen. The connectome exposes what it already has under the seam's names; the MLP exposes its layers the same way; the rule cannot tell them apart.

The seam is exactly what the rule touches and nothing more: the plastic weight tensors, their aligned eligibility traces, their masks, the projector, and the traces-enabled flag. It supports a *list* of plastic tensors from the start, because the MLP has one per layer and the connectome one in total; special-casing "one tensor" would have been a second refactor a change later.

The connectome-named class survives as an alias. It is imported by tests and the brain; a rename that breaks them buys nothing.

## Decision 2 — wrap the actor, do not rebuild it

`MLPPPOBrain` builds its actor as an `nn.Sequential` at a fixed point in construction. Two things depend on that object staying exactly what it is: the torch-RNG stream (orthogonal init consumes it, in order, and every PPO result recorded for this brain depends on that order), and weight persistence (`policy` is `self.actor.state_dict()`, keyed by the Sequential's module names).

The topology therefore holds *references* to the actor's `Linear` modules and registers its trace buffers on itself, not on the actor. `self.actor` is the same object before and after; its state dict is unchanged; RNG is untouched because traces are zero-initialised. PPO byte-identity is then structural rather than something to verify after the fact — though it is verified anyway.

## Decision 3 — same-step eligibility for feedforward layers

`E_l ← λ·E_l + post_l ⊗ pre_l`, using the same step's input and output.

This is not an inconsistency with the connectome's previous-step form; it is the same principle applied to a different structure. The connectome needed `h_prev ⊗ h` because its same-step product is `h ⊗ h` — symmetric, giving both directions of a reciprocal pair identical eligibility and encoding no order. A feedforward layer's pre and post are *different populations*, and the layer orders them causally within the step: `pre_l` demonstrably precedes and causes `post_l`. Using the previous step's input would instead credit a layer's synapses for an output they had no path to.

Orientation is `(out, in)` to match `nn.Linear.weight`, so the trace and the weight it updates are the same shape by construction. `post_l` is the layer's output after its nonlinearity where one exists — the rate-code analogue of the connectome's settled tanh state — and the raw output for the final layer, which has none.

## Decision 4 — every weight matrix is plastic; biases are not (ratified with the user)

All three `Linear` weight matrices learn. Biases are frozen: the connectome has no bias terms, and a Hebbian rule has no pre-synaptic partner for a bias — treating it as a synapse from a constant unit would be an invention with no counterpart on the other arm. `log_std` is frozen and the critic dormant, as on the connectome.

**The output layer learns, and that is the deliberate asymmetry.** The tempting symmetry — freeze the MLP's output layer because the connectome's readout is frozen — compares two frozen objects that are nothing alike. The connectome's readout is the anatomical contrast, chosen because a frozen decoder must respect what its inputs mean. The MLP's output layer is an arbitrary orthogonal draw with no meaning to respect. Freezing it would put back on the yardstick exactly the incapacity the anatomical readout removed from the connectome, and any gap between the arms would partly measure that handicap.

Letting it learn gives the MLP strictly more plastic freedom than the connectome. That makes the yardstick **conservative for the claim being tested**: if the plastic connectome reaches the band of an arm that had more room to adapt, the ranking result is stronger than it would be against a hobbled MLP. If the connectome falls short, the asymmetry is named in the reading, not discovered afterwards.

Rejected: gradient descent on the output layer with plasticity on the hidden ones. Probably the best-performing MLP, and no longer a matched rule — D2's within-regime requirement is the whole reason this arm exists.

## Decision 5 — one definition of the plasticity hyperparameters

The plasticity fields move to a mixin both brain configs inherit. "Matched" is a claim about the numbers being equal; two hand-copied blocks of defaults are equal only until someone edits one, and nothing would announce the divergence. One definition, one validator, and a test that the two configs report identical plasticity defaults.

The magnitude bound is checked against the MLP's initialisation for the same reason it was checked against the connectome's: a stabiliser must not modify the thing it stabilises before the process it bounds begins. Orthogonal init at gain √2 puts the MLP's largest initial weight near 0.65 across seeds; the shared bound of 3.0 clears it by a wide margin.

## Decision 6 — the value head is skipped, not stubbed, here too

Both of the MLP brain's action paths compute the critic's value on every step. Under a plastic rule the call is skipped and the per-step value left unset — the same choice made for the connectome, for the same reason: a rule that owns no critic must not be handed one to keep an unused call site alive. The critic and optimiser are still *constructed*, because construction order is what RNG identity depends on; they are simply never used.

## Risks

- **The traced forward could diverge from `self.actor(features)`.** They iterate the same modules in the same order, and a test pins them bitwise-equal; any future reordering fails visibly.
- **The MLP arm may learn very well or not at all.** Either is informative for a yardstick, and the floors already bound it. What must not happen is the arm learning for a reason the connectome cannot share — which is why biases, `log_std`, and the critic are held to the connectome's treatment exactly.
- **Feature gating and expansion are frozen if present.** They are off on the C3 cell; the plastic config is derived from it. A gated MLP under plasticity is unspecified rather than wrong, and stated as such.
