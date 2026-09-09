# Design: a decorrelating term for the local rule

## The mechanism this is aimed at

The trace is `E = outer(prev_h, h)` accumulated with decay, and `h = tanh(preact)`, so `E` carries
a sign: the Hebbian term potentiates a synapse where pre- and post-synaptic activity agree in sign
and depresses it where they disagree. Three consequences follow, and they explain Logbook 044's
result without any new assumption.

1. **Under random signs the network brakes itself.** An inhibitory synapse driven by positive
   pre-synaptic activity pushes its target negative, the trace goes negative, and the update makes
   the synapse *more* negative — a self-limiting loop, and with half the synapses inhibitory by
   construction it ran everywhere.
2. **Grounded, that loop mostly disappears.** At 80% excitatory the dominant loop is positive
   feedback: co-activity potentiates, potentiation increases co-activity, and only homeostasis and
   the magnitude bound stand in the way. The measured collapse — 31.5 to 14.0 on the wild-type,
   17.4 to 9.1 on the rewired null — is what that looks like.
3. **The rule fights the sign structure rather than using it.** Left free it ends with 15% of
   grounded synapses carrying the opposite sign to their transmitter; enforced, it drives 13% of
   them to exactly zero. Either way, the update's idea of what a synapse should do is set by
   activity correlation alone, and the atlas's identity is at best a constraint applied afterwards.

The two variants below attack (2) from opposite directions: one restores an inhibitory brake by
using the identity the atlas supplies, the other removes runaway growth without any identity at
all. Running both is what separates "inhibition specifically" from "any decorrelating term".

## Variant 1: anti-Hebbian inhibitory plasticity

For every synapse the atlas grounds as inhibitory, the Hebbian term's sign is flipped:

```text
u = η · m · E / ρ_E · (−1 if the synapse's grounded sign is negative else +1)
```

Ungrounded synapses and grounded excitatory synapses are untouched. The decay, the mask, Dale's
law, homeostasis and the clamp keep their existing order and meaning.

What this does mechanistically: an inhibitory synapse that is doing its job — firing when its
target is suppressed, so pre and post anticorrelate and `E < 0` — is currently potentiated toward
zero, unwinding the inhibition. With the flip it is driven further negative, so **co-activity
strengthens what the synapse does rather than what its weight is**. That is the sense in which
this is Hebbian in *effect* and anti-Hebbian in *weight*, and it is why it needs the atlas: before
B.1 there was no identity to key on, and flipping the update on an arbitrarily drawn sign would
have been flipping a coin.

The biological anchor is the electrosensory-lobe connectome (Perks et al., *Nature*, 2026-09-02):
anti-Hebbian depression sits at identified sites within a circuit that is otherwise Hebbian, and
the pairing of excitatory wiring with anti-Hebbian plasticity at specific synapses is what lets
that circuit build a negative image of predictable input. The claim borrowed here is the
arrangement — *which* synapses learn with which sign — not the physiology of the mormyrid
synapse, and the record says so.

**It has no hyperparameter.** The term is the existing update with one factor of −1 on an
identified subset, so this arm needs no pilot and no pin, which also means it cannot be tuned into
a result.

## Variant 2: Oja decorrelation

The substrate-general alternative, needing no sign information:

```text
u = η · m · E / ρ_E − η · λ_w · w − η · γ · y² · w
```

`y` is the post-synaptic activity of the step — the same vector the trace's post-synaptic factor
was built from — broadcast along each weight's post-synaptic axis, and `γ` is a configured
coefficient. **The seam does not carry that activity today**: it exposes weights, traces, masks
and fan-in axes and nothing about what the units did. This change adds `plastic_post_activities`
to the `PlasticTopology` seam, one vector per plastic tensor, indexed along the axis complementary
to the fan-in axis (axis `1` for the connectome's `[pre, post]` matrix, axis `0` for a `Linear`
`[out, in]` weight). The connectome exposes it as a view over the activity buffer its trace update
already keeps; the MLP topology retains each layer's post-activation at trace time. The rule
reads it only under `oja`, so a topology that never selects the term pays nothing. The subtractive term
is the classic Oja normalisation: growth is opposed in proportion to how active the post-synaptic
unit is and how large the weight already is, which both removes runaway growth and decorrelates a
unit's inputs.

It overlaps in effect with homeostasis, which rescales a unit's incoming norm after each update,
and the design keeps both because they act at different granularities: homeostasis is one scalar
per unit applied afterwards, the Oja term is per synapse and inside the update, so it changes the
*direction* the weights move and not only their length. The screen reports both arms with
homeostasis on, as every panel arm has been, so the comparison is against the same stabiliser the
committed values were produced under.

`γ` has no value to inherit and is pinned by a small declared pilot: seeds 1–2 of the wild-type
grounded arm at the panel budget over `γ ∈ {0.01, 0.1, 1.0}`, pinning the highest mean plateau
tail, ties toward the smaller `γ`.

## Configuration

Two fields on the shared plasticity mixin, defaults off:

| field | default | meaning |
|---|---|---|
| `plasticity_decorrelation` | `none` | one of `none`, `anti_hebbian_inhibitory`, `oja` |
| `plasticity_oja_coefficient` | `0.0` | `γ`, `≥ 0`; zero with `oja` selected is rejected |

`anti_hebbian_inhibitory` has no field of its own. Its load-time check cannot sit on the mixin,
which the MLP config shares and which knows nothing about signs: it lives on the connectome config
beside the existing synapse-sign validator, requiring `synapse_signs: atlas`, and is **duplicated
as a brain-construction guard**, since `model_copy` skips validators and the campaign runner
derives configs that way — the defect the sign-grounding work found and fixed once already. The
MLP config rejects the variant outright: it has no transmitter identities to key on.

## Where the terms sit

Unchanged order, with the new term inside the masked update beside the decay:

1. Hebbian term, trace-normalised, **sign-flipped on grounded inhibitory synapses** (variant 1).
2. Weight decay toward zero.
3. **Oja term** (variant 2).
4. Mask, then write.
5. Dale's-law sign projection, if enforced.
6. Homeostatic rescale.
7. Clamp, last, so the bound always holds.

Variant 1 changes a factor inside step 1 rather than adding a term, so it cannot alter the
update's magnitude — only where it points. That matters for reading the result: an arm that
recovers under variant 1 did so by reallocating the same amount of change, not by changing less.

## Signs reach the rule whenever they are grounded

Today the brain hands the rule its sign vector only when Dale's law is enforced, because
enforcement was the only consumer. Variant 1 reads the signs without constraining them, so the
hand-off becomes unconditional whenever `synapse_signs: atlas`. Enforcement stays a separate
switch, and the two compose: a run may flip the update on inhibitory synapses and also refuse to
let any synapse cross its sign.

## Telemetry

One key beside the existing plasticity series: the share of the update's total magnitude carried
by the decorrelating term — the flipped subset's share under variant 1, the Oja term's share
under variant 2, zero under `none`. Read per variant: under variant 1 the update's magnitude is
unchanged by construction, so the share says how much of it was *redirected*; under variant 2 it
says how much of the change the Oja term contributed, and an arm that recovers with a near-zero
share recovered by something other than the term — as the rate multiplier separated consolidating
from not moving.

## The registered test

Logbook 044's grounded Hebbian protocol, unchanged except for the rule keys: `wt_hebbian_atlas`
and `rn_hebbian_atlas` under each variant, seeds 1–16 paired, 1000 episodes, plateau-tail
full-clear success, the single registered extension of a fresh run at 1.5× for a non-converged
run. Four arms, 64 runs. The comparators are **committed and not re-run**: 044's grounded per-seed values on the same seeds
1–16 (wild-type mean 14.0, rewired 9.1), paired seed for seed, and, descriptively, panel 2's
random-sign values (31.5, 17.4).

Four one-sided paired tests corrected together under BH-FDR at α = 0.05:

- **D1** wild-type anti-Hebbian over wild-type grounded Hebbian — *the prediction*.
- **D2** wild-type Oja over wild-type grounded Hebbian.
- **D3** wild-type over rewired null under the anti-Hebbian variant.
- **D4** wild-type over rewired null under the Oja variant.

**Verdict**, assigned in order: `insufficient_seeds`; then `no_recovery` when neither D1 nor D2
confirms — the registered outcome in which 044's prediction fails, and the record states that the
missing inhibitory brake was not what limited the rule; then `recovery_specific` when D1 confirms
and D2 does not, `recovery_general` when D2 confirms and D1 does not, and `recovery_both` when
both do. D3 and D4 annotate the verdict and never change it, since the wiring contrast is a
different question and has been unconfirmable at this sample size in four panels.

Two further annotations, computed and reported but never verdict-changing:
`full_recovery`, when the 80% bootstrap interval of a recovered arm's mean plateau tail over seeds
1–16 includes or exceeds panel 2's committed random-sign mean for the same wiring (31.5 wild-type,
17.4 rewired) — the difference between "the term helps" and "the term restores what grounding
cost"; and `decorrelation_share`, the telemetry above, so that a recovery with a
near-zero share is flagged as attributable to something other than the term.

## What a result means

A recovery licenses the decorrelating term as part of the rule and makes it the substrate for the
structured-instruction work that follows; it does **not** license the 2×2 panel, which remains
gated on the clone assay, and a variant that recovers reward-free learning may still fail to hold
a policy. `no_recovery` closes the inhibitory-brake explanation of Logbook 044's collapse and
leaves that result as a fact about the substrate with no known rule-level remedy, which is a
material input to the 7a shipment decision.

## Alternatives considered

- **BCM sliding threshold** — a running post-synaptic threshold that turns potentiation into
  depression below it. Attractive and untested here; it is a third arm on the same harness if
  either variant recovers, and adding it now would put three arms and two pilots into one change.
- **Flipping the update on grounded excitatory synapses instead** — the mirror control. Not run:
  it is a prediction nothing motivates, and the family is already four tests.
- **Removing homeostasis for these arms** — would confound the comparison with the committed
  values, which were produced with it on.
- **Screening these variants through the clone assay** — the wrong instrument: the assay asks
  whether a mechanism holds a cloned policy, and this prediction is about learning from random
  weights. Either variant can be put through the assay later if it is proposed for the panel.
