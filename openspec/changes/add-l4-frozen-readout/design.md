# R.1d — the frozen readout: design

## What is frozen, and what is left to test

Under the rule the connectome's `plastic_weights` is `w_chem` alone. Three tensors PPO trains are
therefore frozen, and R.1c eliminated two of them:

| frozen tensor | status after R.1c |
|---|---|
| `food_gains` (sensory projection) | **eliminated** — PPO moves it 0.177 relative, cosine 0.985 to its init, so freezing it costs little |
| `log_std` (action-noise scale) | **eliminated** — swept by hand across std 0.368–1.0, null (4.361 / 4.056 / 4.154 foods) |
| **`readout`** (2×4, motor-class means → action) | **untested**, and the largest relative norm change of the three at 0.783 |

The readout is not a random matrix: it is initialised anatomically, speed as the B-vs-A motor-class
contrast and turn as D-vs-V, each unit-normed. So the question is not "does a random readout hurt" but
**"is this anatomical prior good enough for the rule to learn behind it"** — and PPO moving it 78% in a
tenth of a run is the reason to doubt it.

## The arms

One cell and one mask, held at R.1c's best measured operating point: the calibrated hard-food connectome
cell, `motor` (39 units, 323 adaptable synapses, 156 draws per decision), σ 0.1,
`initial_log_std: -1.0`. Seeds 1–8, 3000 episodes.

| readout | learning | frozen | source |
|---|---|---|---|
| `anatomical` | **3.751** | **3.150** | R.1c's committed `motor` pair — **not re-run** |
| `ppo` | new | new | that seed's PPO run on this cell, **harvested at `initial_log_std: -1.0`** |
| `rotated` | new | new | a random direction at the **same Frobenius norm** as that seed's PPO readout |

**32 new runs**, plus **8 PPO harvest runs** whose checkpoints supply the readouts.

### Reusing R.1c's anatomical pair is conditional on an equivalence test

The `ppo` and `rotated` arms load a prepared checkpoint; R.1c's `anatomical` pair never did. Loading also
triggers `reset_state()` — fresh running scales, a restarted perturbation schedule, re-anchored
homeostatic norm targets — and `buffer.reset()`. Tracing each says they **should** be equivalent to a
fresh construction, because the preparation leaves every non-readout tensor at that seed's own
initialisation, so the re-anchored targets are the same targets. But the anatomical pair anchors all three
comparisons, and "should be" is not the standard for that.

**So a test decides it, not an argument.** Construct the baseline brain at a seed; load a prepared
checkpoint whose readout is the **anatomical default** into a second identically-constructed brain; assert
every tensor **bit-identical**.

- **If it passes**, R.1c's committed `motor` pair is the anatomical arm and 16 runs are saved.
- **If it fails**, the anatomical pair is **re-run through the load path** — 16 runs, about 30 minutes —
  and R.1c's values become a cross-check on what loading changed rather than the comparator.

Either way the record states which branch was taken and why. Comparing a loaded arm against an unloaded
one and attributing the difference to the payload is the error this removes.

### Only the readout moves

The preparation writes, per seed, a checkpoint with the substituted `readout` and **every other tensor at
that seed's own fresh initialisation**. A wholesale load of a PPO checkpoint would also bring:

- **`log_std`** — changing the action noise the rule runs at, an axis R.1c already swept and pinned;
  and for the same reason the **harvest itself is pinned to `initial_log_std: -1.0`**. The committed
  hard-food PPO config sets no `initial_log_std`, so it trains at the default — an action std of **1.0**
  against these arms' **0.368** — and a readout adapted to a distribution the rule does not use would be
  a second co-adaptation on top of the one declared below. The harvest's only job is to produce a readout
  for *this* arm, so it is configured to match it;
- **`w_chem`** — making the arm a **clone assay**, which is B.0's experiment and not this one;
- **`food_gains`** — an axis R.1c eliminated, and a second simultaneous change.

The preparation is asserted by test to differ from the baseline arm's initialisation in the readout and
nothing else.

### Why `rotated` is the arm that makes this interpretable

A bare `ppo`-versus-`anatomical` contrast cannot separate two readings with different consequences:

| pattern | reading | what follows |
|---|---|---|
| `ppo` > `rotated` > `anatomical` | the readout's **direction** matters and PPO found a good one | the follow-up is a **better-grounded** readout, from the atlas or from real-worm kinematics |
| `rotated` ≈ `ppo` > `anatomical` | the anatomical prior is **actively bad**; almost any change helps | the follow-up is to fix the prior, and the "PPO found something" reading is withdrawn |
| all three alike | the readout is **not** the handicap | eliminated; R.2 becomes the live path |

Same norm rather than same distribution, because a rotation isolates direction while leaving the
readout's scale — which interacts with the action distribution — untouched.

### The co-adaptation caveat, stated rather than solved

PPO's readout is co-adapted with PPO's `w_chem`, and the rule always starts from a random `w_chem`. So a
readout tuned to PPO's solution may be *unsuited* to whatever the rule would otherwise find, and this
design cannot rule that out — it can only show whether the readout helps, hurts or does nothing from a
random start. Where `ppo` underperforms `rotated`, that is the reading the record will offer, and it is
a reason the negative branch is weaker evidence than the positive one.

## The reading

**Plateau-tail mean foods** through I.2's graded family, each readout's learning arm against **its own**
frozen control, paired by seed, one-sided, BH-FDR across the three readouts. Both registered minima, as
in R.1c: **1.0 foods** of 20, and **10% of the reachable gap** taken against that arm's own frozen mean.

**The reachable gap uses the matched reference.** The harvest runs PPO at these seeds and at this action
scale, so it supplies a PPO level on the same 8 seeds and the same `initial_log_std` as the arms — a
better denominator than 058's 19.31 foods, which was measured over 32 seeds at std 1.0. Both are
reported; the **matched** figure binds, which is where R.1c's own correction ended up.

**Reported separately, as R.1c established:** whether an arm **beats its floor**, and whether it
**reaches competence** (the committed 20% full-clear threshold). Only the second would make block V's
time-to-competence contrast defined, and only the second bears on R.1b.

**Credited-synapse drift** per arm. R.1c found it invariant at 1.37–1.38× across every dimension; if a
better readout changes that, the drift is the first place it would show.

## Outcomes, fixed before the run

| verdict | test | what follows |
|---|---|---|
| `readout_is_the_handicap` | a substituted readout beats its own floor by both minima **and reaches competence** | the handicap is located. **The follow-up is substrate fidelity — a better-grounded readout — and not PPO**, since a rule needing a gradient-trained tensor is not a plausible local learner |
| `readout_helps_but_not_enough` | beats its floor by both minima, does not reach competence | a **result, not a rescue**: the readout is part of the limit and not all of it. R.1b stays blocked, R.2 stays live, and the record says which part is accounted for |
| `readout_not_the_handicap` | no substituted readout beats `anatomical` | the third of three frozen tensors is eliminated. The failure is in credit assignment over the horizon, which the invariant 1.4× drift already points at, and **R.2 becomes the live path** |

The `rotated` arm's position is reported in every branch, since it decides which follow-up a positive
result licenses.

### Stop clauses — void until found

- **The `anatomical` pair does not reproduce R.1c's values** (learning 3.751, frozen 3.150 at these
  seeds and settings), checked by re-scoring R.1c's logs through this change's harness. A mismatch means
  the two harnesses disagree and nothing here is interpretable. This is separate from the equivalence
  test above: that one decides whether R.1c's arms may serve as the anatomical comparator at all, this one
  checks the harness reads them the same way.
- **A prepared checkpoint differs from the baseline in more than the readout.** Asserted by test before
  any arm runs.
- **The PPO harvest does not learn the cell.** If PPO fails at these seeds there is no good readout to
  harvest and the arm is void, not negative.

## Honest prior

**`readout_helps_but_not_enough`, with `readout_not_the_handicap` a close second.**

For it: the readout is the last frozen tensor standing, PPO moves it more than anything else it trains,
and a 2×4 matrix mapping four motor-class means to speed and turn is a very narrow bottleneck for a
302-neuron network to express a policy through.

Against it: R.1c's **invariance is the hard fact to explain**. Credited-synapse drift is 1.37–1.38× at
every dimension, and the learning arm sits in a 2-food band across three axes. A rule that writes 1.4×
its own weight norm in a direction that does not help is not obviously a rule waiting for a better output
map — it looks like one whose credit assignment is wrong, and a readout cannot fix credit assignment over
350 steps. That is why the negative branch hands off to e-prop rather than to another readout variant.

## What this cannot be

**Not a deliverable, whatever it returns.** D1 asks for a biologically plausible local rule. An arm that
needs a readout trained by backpropagation is not one, and a positive result here may **not** be cited as
"a local rule learns the connectome". It would be a **located handicap**, whose repair has to come from a
better-grounded anatomical readout — a substrate-fidelity question of the shape B.1 answered for synapse
signs — or from a rule that can shape its own output map without collapsing, which Logbook 040 measured
as the hard part.

## Risks

- **One mask.** Held at `motor` because R.1c measured it best; a readout that helps only at a different
  dimension would be missed, and the record says so.
- **Eight seeds**, the power limit every R result carries; hence both effect minima.
- **The co-adaptation caveat above**, which makes the negative branch weaker evidence than the positive.
- **A harvested readout is a moving target**: PPO's readout after 3000 episodes differs from its readout
  after 300. The harvest is taken at the end of a full run, stated, and a partially-trained variant is
  named as the follow-up rather than run here.
