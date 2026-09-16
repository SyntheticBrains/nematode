# Was the four-class pooling the bottleneck? (L.1)

## Why

[L.0](../../../docs/experiments/logbooks/064-l4-frozen-features.md) returned
`wiring_is_inert_as_features`: under `readout_only` — the one biologically plausible learner on this
substrate that reaches competence — the wild-type connectome shows **no significant advantage** over
its degree-preserving rewired null at the registered ≥ 20% bar. So
[034](../../../docs/experiments/logbooks/034-connectome-structure-controls.md)'s degree-statistics
verdict now holds in a third learning regime.

A null at one readout width leaves the obvious question unanswered, and L.1 was **registered as a
conditional before L.0 ran** precisely so the promotion would not be a reaction to its result: the
learner reads the connectome through a **mean-pool over four motor classes into an 8-parameter
map**. Thirty-nine motor neurons enter, four numbers come out, and each neuron carries `1/|class|`
of its class's influence. If the wild-type wiring's advantage lives in *which neuron* fires rather
than *which class*, that pool destroys it before the learner sees it, and L.0's null would be a fact
about the pooling rather than about the wiring.

> *Is the four-class pooling the bottleneck through which the wiring's features are invisible?*

It is also the right test for L.0's reverse-direction lean — the null reaching competence on a median
405 episodes against the wild type's 568, at q = 0.058 reversed. A bottleneck can produce both a null
and a lean.

## What changes

**The confound this change exists to avoid.** A per-neuron readout is `(2, 39)` — **78 parameters
against the pooled map's 8**. Running width 39 against L.0's width 4 and reading a gain would
confound *"the wiring's features were there and the pool hid them"* with *"ten times the parameters
learn faster on any features at all"*. **So the question is not a main effect, it is an interaction**:

> *Does widening the readout help the wild type **more than** it helps the degree-matched shuffle?*

Only a positive interaction says the pool was hiding wiring-specific structure. A width main effect
with no interaction says more parameters help, wiring-blind — which closes L.1 and leaves L.0's
verdict standing with the width objection retired rather than outstanding.

- **A full 2×2 in one campaign**: width {pooled, per-neuron} × wiring {wild type, rewired null},
  plus a frozen floor for **each of the four cells**, on the `hard350` cell at seeds 1–32 — L.0's
  cell, learner and seeds.

  | arm | wiring | readout | learner | runs |
  |---|---|---|---|---|
  | `wt_pooled` | wild type | `(2, 4)`, 8 params | readout learns, `w_chem` frozen | 32 |
  | `rn_pooled` | rewired null | `(2, 4)` | the same | 32 |
  | `wt_wide` | wild type | `(2, 39)`, 78 params | the same | 32 |
  | `rn_wide` | rewired null | `(2, 39)` | the same | 32 |
  | `wt_pooled_frozen` | wild type | `(2, 4)` | nothing learns | 32 |
  | `rn_pooled_frozen` | rewired null | `(2, 4)` | nothing learns | 32 |
  | `wt_wide_frozen` | wild type | `(2, 39)` | nothing learns | 32 |
  | `rn_wide_frozen` | rewired null | `(2, 39)` | nothing learns | 32 |

  **256 runs.** Each learning arm is gated against a floor **at its own width**. An earlier draft of
  this proposal ran two floors instead of four, reasoning that the two widths compute the same
  function at initialisation so a frozen arm cannot depend on width. That is true in exact arithmetic
  and **false in floating point**: measured on the real class sizes the two paths differ by 7.45e-9
  on the action mean, because a slice `mean()` and a dot product with pre-divided weights round
  differently, and 3000 episodes of action sampling amplify that into different trajectories. The
  invariance still matters and is still asserted — but as a statement about the **policy**, within
  tolerance, not about the run.

- **The per-neuron readout is initialised by expanding the pooled one**, not drawn afresh:
  `w_i = pooled_{class(i)} / |class(i)|`. This is the design decision the whole comparison rests on,
  and it buys three things no fresh draw could:

  1. The `(2, 4)` orthogonal draw still happens at the same point and consumes the same randomness,
     so **the width-4 arms stay byte-identical to L.0's runs** and the RNG streams of the two widths
     match at a seed.
  2. At initialisation the two widths compute **the same policy to within floating-point rounding**,
     so the arms do not start from different behaviour.
  3. The **anatomical contrast** `set_anatomical_readout` writes expands the same way, so a frozen
     arm's policy does not depend on width.

  What is left as the only difference between the widths is **the space the learner can move in**,
  which is the manipulation.

- **The primary metric is `auc_success`, not `episodes_to_30pct_success`**, and this is a departure
  from block V's instrument that is registered with its reason. L.0 met **asymmetric censoring** on
  this exact cell: five wild-type seeds of 32 never reached competence within 3000 episodes against
  one for the null. A right-censored metric cannot support a difference of differences when the
  censoring rate itself differs across the cells of the 2×2 — widening the readout is expected to
  reduce censoring, which would inflate the interaction by construction.
  `auc_success` is defined for every seed at the same horizon. `episodes_to_30pct_success` is
  reported beside it with its censoring counted per cell.

- **A new harness** `scripts/analysis/l4_readout_width.py`, the R.1c/R.1d/R.2/L.0 sibling pattern,
  reusing the committed metric and statistics layers verbatim. `auc_success` comes from
  **`connectome_structure_efficiency`, called once per width** with the wild and rewired arms mapped
  onto its own two — which yields per-seed values for all four cells of the 2×2 with that module
  unmodified. Re-implementing AUC here would be the duplication V.4's review caught. It reports the
  interaction, both main effects, four learning gates and the untrained prior under one BH-FDR
  family.

## The registered readings

| reading | when |
|---|---|
| `pooling_was_not_the_limit` | no interaction, whatever the width main effect does. L.0's verdict stands and the width objection is **retired rather than left open**; L.4 and L.5 stay `closed-unopened` |
| `pooling_hid_structure` | interaction positive — widening helps the wild type more. The pool was hiding wiring-specific structure; **L.4 and L.5 reopen** as their gates registered |
| `width_favours_the_shuffle` | interaction negative — widening helps the null more. L.0's reverse-direction lean strengthens at width; **reported as the reverse direction and not explained** |
| `no_learning` | a gate fails: an arm that carries the claim did not learn, so the interaction is uninterpretable |
| `insufficient_seeds` | too few paired seeds survived |

## Impact

- Affected specs: `plasticity-evaluation`
- Affected code: `packages/quantum-nematode/quantumnematode/brain/arch/connectome_ppo.py` (a
  `readout_width` field on `ConnectomePPOBrainConfig` and its topology plumbing; the readout's shape,
  its anatomical contrast, its eligibility buffer and the symmetric learning-signal projection),
  `configs/scenarios/foraging/` (four new arms — two learning, two floors),
  `scripts/analysis/l4_readout_width.py` (new)
- **L.0's harness and block V's two harnesses are READ-ONLY here.**
