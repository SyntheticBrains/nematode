# Design — L.1, readout width

## The question is an interaction, and that is the whole design

The naive version of L.1 runs a per-neuron readout, compares it to L.0's pooled one, and reads a
gain. That comparison cannot answer anything: the per-neuron readout has **78 parameters against 8**,
so a gain is equally well explained by *"more parameters learn faster"* — which is true of almost any
learner and says nothing about the connectome.

The manipulation has to be crossed with the wiring:

```text
                pooled (8 params)     per-neuron (78 params)
wild type            A                        B
rewired null         C                        D
```

- **width main effect** — `(B + D) − (A + C)`: do more parameters help? Almost certainly yes, and
  uninformative on its own.
- **wiring main effect** — `(A + B) − (C + D)`: L.0 measured the `A − C` half and found nothing.
- **interaction** — `(B − A) − (D − C)`: **does widening help the wild type more than the shuffle?**
  This is the only contrast that distinguishes "the pool hid wiring-specific structure" from "more
  parameters help".

The interaction is the primary. Everything else in this design exists to make it readable.

## The per-neuron readout expands the pooled one

The pooled readout is `(2, 4)` over four class means. The pool is a **mean**, so

```text
mu_k = Σ_c readout[k, c] · (1 / |c|) · Σ_{i ∈ c} h_i
```

A per-neuron readout `(2, 39)` reproduces that function exactly when
`W[k, i] = readout[k, class(i)] / |class(i)|`. So the per-neuron arm is initialised **by expanding
the pooled draw**, not by drawing afresh. Three properties follow, and each removes a confound that
a fresh `(2, 39)` orthogonal draw would have introduced:

| property | why it matters |
|---|---|
| The `(2, 4)` orthogonal draw still happens at the same point, with the same shape | The RNG stream is untouched, so **the width-4 arms are byte-identical to L.0's committed runs** and the two widths consume the same randomness at a seed. Verified by diffing two seeds against L.0's logs, not assumed. |
| At initialisation the two widths compute the **same policy, to within floating-point rounding** | The arms do not diverge because they started from different behaviour. Without this, "width" would be confounded with "a different initial policy", which is how V.4's coupling caveat got written. |
| The anatomical contrast expands the same way | `set_anatomical_readout` writes a derivable dorsal/ventral and forward/backward contrast; expanded per neuron it is the same map, so **a frozen arm's policy does not depend on width**. |

The classes are **unequal** — VB 11, DB 7, VA 12, DA 9, totalling 39 — so the `1/|class|` factor is
per class and not a single constant. Getting that wrong would silently reweight the motor pools.

### The floors are four arms, and the reason is floating point

An earlier draft ran two floors. The reasoning was that a frozen arm never updates and the two widths
are the same function at initialisation, so `wt_frozen` at width 39 **is** `wt_frozen` at width 4 —
the same trajectory, step for step, and running both would spend 64 runs on duplicate logs.

**That is true in exact arithmetic and false in floating point.** A slice `mean()` and a dot product
with pre-divided weights do not round identically, and the motor classes are unequal (11, 7, 12, 9),
so the divisors are not powers of two. Measured on the real class sizes, the two paths differ on the
action mean by **7.45e-9**:

```text
pooled : [0.30981287360191345, -0.01437794417142868 ]
wide   : [0.30981287360191345, -0.014377951622009277]
```

The action is then sampled from a Gaussian around that mean, so the two arms take slightly different
actions from the first step, and 3000 episodes amplify the difference into unrelated trajectories. A
width-39 floor is **not** the same run as a width-4 floor, and gating a wide learning arm against a
pooled floor would compare it to a control it never shared a trajectory with.

So each learning arm is gated against a floor at **its own width** — four floors, 256 runs. The
invariance test stays, because it is what licenses calling the two widths the same *policy*, but it
asserts equality **within tolerance** and makes no claim about the runs.

## Why `auc_success` is the primary, against block V's instrument

Block V and L.0 read `episodes_to_30pct_success`, right-censored at the 3000-episode horizon. On this
exact cell L.0 met **asymmetric censoring**: five wild-type seeds of 32 never became competent against
one for the null — which is also why L.0's means (1039.8 against 644.7) diverge much further than its
medians (568 against 405).

A difference of differences cannot be read on a metric whose censoring rate differs across the cells
being differenced. Widening the readout is *expected* to reduce censoring; if it reduces it more on
the side that had more of it, the interaction moves for a reason that has nothing to do with the
wiring. **`auc_success` is defined for every seed at the same horizon and cannot be censored**, so it
is the primary here. `episodes_to_30pct_success` is reported beside it **with its censoring counted
per cell of the 2×2**, so a reader can see whether the two metrics agree and why if they do not.

This is a deliberate departure from the instrument block V and L.0 used, registered in advance with
its reason, because the alternative is an instrument that is wrong for this contrast.

## Power, and the honest statement of it

The interaction is a difference of two differences, so its per-seed variance is roughly the **sum** of
the two differences' variances — about **2× that of a single contrast**, or **√2 on the standard
error**. At 32 paired seeds this panel is therefore **materially less sensitive to the interaction
than L.0 was to its main contrast**, on the same seeds and the same cell.

That is registered here rather than discovered afterwards, and it bounds what a null can claim. The
exact detectable effect is computed from L.0's own observed per-seed spread and carried as a field in
the record before the campaign runs (task 4.4), not asserted here. **A null interaction will be
reported as "no interaction detected at this panel's sensitivity", with that sensitivity stated** —
not as "the pooling was demonstrably not the limit".

## What a positive interaction would not license

Fixed now, because these are the readings the result will be pulled toward:

1. **Not a claim that the four-class pool is biologically wrong.** It is a stand-in for the
   neuromuscular system at either width; 39 weights is not more biological than 8, it is less.
2. **Not an endpoint claim.** Everything under `readout_only` is about time to competence and area
   under the learning curve, as R.2, L.0 and block V have been.
3. **Not a read-across to block V's PPO result.** Different learner, different axis.
4. **Not a mechanism.** V.2 scored 64 rewirings on four graph properties fixed in advance and none
   predicts learning time. A positive interaction would say the pool hid *something*; it would not
   say what.

## Read-only, and why

`scripts/analysis/l4_frozen_features.py` (L.0's harness), `scripts/analysis/wiring_premise.py` and
`scripts/analysis/connectome_structure_efficiency.py` are **unmodified here and a test asserts it**.
The last of those is not merely left alone but actively **reused**: `auc_success` and the censoring
counts come from calling `connectome_structure_efficiency.analyse` **once per width**, with this
change's wild and rewired arms mapped onto its own two, which yields per-seed values for all four
cells of the 2×2 without touching it. Re-deriving AUC in the new harness would duplicate a committed
metric — the defect V.4's review caught in its first draft.
L.0's width-4 numbers are one cell of this 2×2, so editing the instrument that produced them would
let "the instrument changed" compete with the interaction — the lesson V.4 registered as a spec
requirement and which this change inherits.
