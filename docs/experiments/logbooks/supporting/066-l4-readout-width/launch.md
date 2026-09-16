# L.1 — readout width: the registered protocol

Registered in `openspec/changes/add-l4-readout-width`, reviewed and committed **before** any
registered seed runs. The pilot and the two instrument checks below ran first and spent none.

## The question

[L.0](../../064-l4-frozen-features.md) returned `wiring_is_inert_as_features`: under `readout_only`,
the wild-type connectome shows no significant advantage over its degree-preserving rewired null. But
that learner reads the connectome through a **mean-pool over four motor classes into an 8-parameter
map** — 39 motor neurons in, four numbers out, each carrying `1/|class|` of its class's influence. If
the wiring's advantage lives in *which neuron* fires rather than *which class*, the pool destroys it
before the learner sees it.

L.1 was **registered as a conditional before L.0 ran**, so the promotion is not a reaction to its
result.

> *Is the four-class pooling the bottleneck through which the wiring's features are invisible?*

## The question is an interaction, and that is the whole design

A per-neuron readout has **78 parameters against 8**. Comparing it to L.0's pooled arm and reading a
gain would confound *"the wiring's features were there and the pool hid them"* with *"ten times the
parameters learn faster on anything"*. So width is **crossed** with wiring:

| | pooled (8 params) | per-neuron (78) |
|---|---|---|
| wild type | A | B |
| rewired null | C | D |

- **width main effect** `(B+D) − (A+C)` — do more parameters help? Almost certainly, and
  uninformative alone.
- **wiring main effect** `(A+B) − (C+D)` — L.0 measured the `A − C` half.
- **interaction** `(B−A) − (D−C)` — **the primary.** Does widening help the wild type *more*?

Only the interaction separates the two stories. Both main effects are reported beside it and **never
in its place**.

## The arms — 768 runs

Eight arms × seeds **1–96** at 3000 episodes. The four wide configs differ from their committed L.0
partners in the **`readout_width` key alone**, verified by loading both and diffing the resolved
config.

| arm | wiring | readout | learner |
|---|---|---|---|
| `wt_pooled` / `rn_pooled` | wild type / null | `(2, 4)` | readout learns, `w_chem` frozen |
| `wt_wide` / `rn_wide` | wild type / null | `(2, 39)` | the same |
| `wt_pooled_frozen` / `rn_pooled_frozen` | wild type / null | `(2, 4)` | nothing learns |
| `wt_wide_frozen` / `rn_wide_frozen` | wild type / null | `(2, 39)` | nothing learns |

**The per-neuron readout is initialised by expanding the pooled one** — `W[k,i] = pooled[k,c(i)] / |class(i)|` — rather than drawn afresh. The `(2,4)` orthogonal draw still happens at the same point,
which buys three properties, all asserted by test:

| property | verified |
|---|---|
| the RNG stream is untouched | every non-readout parameter and float buffer **bitwise identical** across widths |
| the widths start from the same policy | action means agree to **1.5e-8** |
| the anatomical contrast expands the same way | agrees to **3.0e-8**, per-neuron signs preserved |

So the only difference between the widths is **the space the learner can move in**.

### Why four floors and not two

An earlier draft shared one floor per wiring, reasoning that the widths compute the same function at
initialisation so a frozen arm cannot depend on width. **True in exact arithmetic, false in float32**:
a slice `mean()` and a dot product with pre-divided weights round differently, and the classes are
unequal (VB 11, DB 7, VA 12, DA 9) so the divisors are not powers of two. The action is sampled around
that mean, so the arms diverge within an episode.

The pilot confirms both halves of this:

| | pooled floor | wide floor |
|---|---|---|
| wild type | 1.829 foods | 1.829 |
| rewired null | 4.119 | **4.116** |

Same policy; **not** the same run. Each learning arm is gated against a floor at its own width.

## The primary metric, and why it departs from block V's

`auc_success`, not `episodes_to_30pct_success`. L.0 met **asymmetric censoring** on this exact cell —
five wild-type seeds of 32 never competent against one for the null. A difference of differences
cannot be read on a metric whose censoring rate differs across the cells being differenced, and
widening the readout is *expected* to reduce censoring, which would move the interaction for a reason
unrelated to the wiring. `auc_success` is defined for every seed at the same horizon. The censored
metric is reported beside it **with its censoring counted per cell**, and a disagreement in direction
is reported rather than resolved.

## Sensitivity — sized by the pilot, not assumed

**The minimum interaction worth detecting is not arbitrary.** L.0 found the rewired null **ahead by
0.1076** on `auc_success`. For widening to mean the pool hid wiring structure, the interaction must
**flip that sign** — so it must exceed ~0.108. Anything smaller changes no verdict and reopens nothing.

The interaction's spread is `sd = sd(D)·√(2(1−rho))`, where `D` is the wiring difference at one width.
The first registration bounded `rho` at zero and **expected that to be pessimistic**: the two widths at
a seed share the task draws, the RNG stream and the initial policy, so their wiring differences ought
to correlate.

**The pilot says otherwise.** At seeds 101–104, `rho = +0.02` and the realised interaction sd was
**0.3770** against the independence bound of 0.4487 — learning amplifies the divergence enough to wash
out the shared start. Four points make that noisy, but it is the only evidence and it points at
independence rather than away from it.

| seeds | detectable at 80% | against the 0.108 threshold |
|---|---|---|
| 32 (first registered) | 0.187 | **misses it** — finds only a large interaction |
| **96 (registered)** | **0.1077** | **matches** it |

96 seeds resolves 0.1077 against a 0.1076 threshold: the panel **sits at** 80% power for exactly the
sign-flipping effect rather than clearing it. These are normal-approximation planning figures, not the
registered rank test's power. The realised figure and the measured `rho` are recomputed from the
panel's own deltas and reported with the result.

## The registered readings

Gates are read **first** — an interaction between arms that did not learn is uninterpretable.

| reading | when | consequence |
|---|---|---|
| `pooling_hid_structure` | interaction positive | the pool was hiding wiring-specific structure; **L.4 and L.5 reopen** |
| `width_favours_the_shuffle` | interaction negative | L.0's reverse-direction lean strengthens at width; **reported and not explained** |
| `pooling_was_not_the_limit` | no interaction | L.0's verdict stands, the width objection **retired at the stated sensitivity**; L.4/L.5 stay closed |
| `no_learning` | a gate fails | uninterpretable |
| `insufficient_seeds` | too few paired seeds | — |

The interaction has two registered directions, so its p is a real two-sided `2×min` capped at 1 —
`min(p_up, p_down)` is not a p-value and doubles the type-I rate, the defect PR #375 caught in L.0's
prior check.

## The instrument checks

**L.0's harness reproduces its record** (task 4.3), re-run today on its committed campaign: gates
+14.801 and +14.701 at q = 0.000, prior −0.975 at q = 0.217, `w_chem` drift 0.00 both wirings, verdict
`wiring_is_inert_as_features`, and `auc_success` wild 0.43 against rewired 0.54 for a **−0.108** gap —
the figure this panel is sized against.

**The pooled arms are byte-identical to L.0's runs** (task 4.2) — two seeds re-run under the new
`readout_width` key and compared against L.0's committed logs. This is what licenses treating the
pooled cells as the same thing L.0 measured.

`l4_frozen_features.py`, `wiring_premise.py` and `connectome_structure_efficiency.py` are **read-only
here and a test asserts it**. The last is actively reused: `auc_success` and the censoring counts come
from calling it **once per width**, which yields all four cells with that module unmodified.

## What a positive interaction would not license

1. **Not a claim the four-class pool is biologically wrong.** It is a stand-in for the neuromuscular
   system at either width; 39 weights is not more biological than 8, it is less.
2. **Not an endpoint claim.** Everything under `readout_only` is about learning speed and area under
   the curve.
3. **Not a read-across to block V's PPO result.** Different learner, different axis.
4. **Not a mechanism.** V.2 scored 64 rewirings on four graph properties fixed in advance and none
   predicts learning time. A positive interaction would say the pool hid *something*, not what.

## The honest prior

I expect **`pooling_was_not_the_limit`**. L.0 was null and leaned the other way; V.2 found no graph
property predicting learning time; and 034, R.1c and R.2 are negative in three other regimes. The
pilot's four seeds showed a large apparent interaction (+0.379, 3/4 seeds) — **that is not evidence,
was not read, and did not inform the sizing**, which came from the spread alone. It is recorded here
because the runs happened and because a prior written after seeing it would not be a prior.
