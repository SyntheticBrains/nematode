# R.1d — the frozen motor readout: the registered protocol

Registered in `openspec/changes/add-l4-frozen-readout`, reviewed and committed **before** the arms run.

## The question

[R.1c](../../061-l4-reduced-perturbation.md) closed the perturbation dimension — from **1208 draws per
scored decision down to 39**, no declared set beat its own frozen control and none reached competence,
with credited-synapse drift invariant at **1.37–1.38×**. Under the rule the connectome may write
**`w_chem` alone**, so three tensors PPO trains are frozen, and R.1c eliminated two:

| frozen tensor | status |
|---|---|
| `food_gains` (sensory projection) | **eliminated** — PPO moves it 0.177 relative, cosine 0.985 |
| `log_std` (action-noise scale) | **eliminated** — swept by hand, null (4.361 / 4.056 / 4.154 foods) |
| **`readout`** (2×4, motor-class means → action) | **untested**, the largest relative norm change at 0.783 |

## What the harvest showed, and why there are four arms

The harvest passed its stop clause: PPO reaches **18.945 foods of 20 and 66.48% full clear** at these
eight seeds and at the arms' own action scale, tight across seeds (18.76–19.06). **18.945 is therefore the
matched reference** the minima are computed against; 058's 19.31 came from 32 seeds at an action std of
1.0 and is reported beside it.

**PPO does not refine the anatomical prior — it replaces it.** Norm **7.820** against the anatomical
default's **1.414** (5.5×), at a cosine of **−0.178**. A representative harvested readout:
`[[2.22, 2.81, −0.84, −3.20], [3.01, −0.57, 4.65, 0.05]]`.

That required a fourth arm, added before any arm ran. The readout multiplies the motor-class means into
the action mean while the action noise is **fixed** at std 0.368, so a 5.5× readout is a large shift in
commitment versus exploration — nothing to do with reading the motor classes better. Without
`anatomical_scaled`, "the direction matters" and "the scale matters" are not separable.

| arm | norm | cosine to anatomical | isolates | runs |
|---|---|---|---|---|
| `anatomical` | 1.414 | +1.000 | the committed baseline | **R.1c's `motor` pair — reused** |
| `anatomical_scaled` | 7.820 | +1.000 | **scale only** | new |
| `rotated` | 7.820 | −0.028 | direction at PPO's scale | new |
| `ppo` | 7.820 | −0.178 | direction at PPO's scale | new |

**48 new runs** — three substituted readouts × (learning, frozen) × 8 seeds — at the `motor` mask, σ 0.1
and `initial_log_std: -1.0`, 3000 episodes.

### Only the readout moves, and the reuse is licensed by a test

Each prepared checkpoint is that seed's **own initialisation** with one tensor substituted; `w_chem`,
`food_gains`, `log_std` and the wiring buffers are asserted untouched before the file is written. A
wholesale PPO load would change an action-noise axis R.1c pinned, make the arm a **clone assay**, and move
the **homeostatic norm targets** the rule re-anchors from the weights it loads.

R.1c's anatomical pair never went through the load path, so reusing it was made conditional on an
equivalence test rather than on argument. **It passed**: loading a prepared anatomical-readout checkpoint
leaves all 16 topology tensors, the rule's baseline and the homeostatic norm targets **bit-identical**. So
the pair is reused and 16 runs are saved. Had it failed, the pair would have been re-run through the load
path and R.1c's values kept as a cross-check.

## The reading

**Plateau-tail mean foods** through I.2's graded family, each readout against **its own** frozen control —
a better readout raises the floor too — paired, one-sided, BH-FDR across the four. Both minima: **1.0
foods** of 20, and **10% of the reachable gap** against the matched 18.945.

**Reported separately**: whether an arm **beats its floor**, and whether it **reaches competence** (20%
full clear). Only the second makes block V's time-to-competence contrast defined and only the second bears
on R.1b. R.1c's verdict condition proved weaker than its own consequence; this keeps them apart from the
start.

**Credited-synapse drift** per arm, to see whether a better readout moves the 1.37–1.38× that was
invariant across every dimension in R.1c.

## What each ordering means, fixed before the run

| ordering | reading | follow-up |
|---|---|---|
| `anatomical_scaled` ≈ `ppo` > `anatomical` | the **scale** mattered — commitment against a fixed action noise | the readout-scale / action-noise interaction, **not** a better-grounded readout |
| `ppo` > `anatomical_scaled` ≈ `rotated` | PPO's **direction** matters and it found a good one | a better-grounded readout |
| `rotated` ≈ `ppo` > `anatomical_scaled` | any direction change helps; the anatomical direction is actively bad | fix the prior; withdraw "PPO found something" |
| all four alike | the readout is **not** the handicap | R.2 becomes the live path |

Anything else is reported as **mixed**, as it is, rather than resolved toward the nearest pattern.

## Outcomes

| verdict | test | what follows |
|---|---|---|
| `readout_is_the_handicap` | a substituted readout beats its floor by both minima **and reaches competence** | the handicap is located; the repair is a better-grounded readout, **not** PPO |
| `readout_helps_but_not_enough` | beats its floor by both minima, no competence | a **result, not a rescue**. R.1b stays blocked, R.2 stays live |
| `readout_not_the_handicap` | no substituted readout beats its floor | the third of three frozen tensors is eliminated; **R.2 becomes the live path** |

### What this cannot be, whatever it returns

**Not a candidate for D1.** A rule taking its readout from a gradient-trained run is not a biologically
plausible local learner. A positive result **locates a handicap**; it may not be cited as "a local rule
learns the connectome". The harness carries this as a field, not only as prose.

Making the readout plastic is out of scope: [Logbook 040](../../040-l4-panel.md) recorded a 96% forager
collapsing to zero within three episodes once its readout learned.

## Honest prior

**`readout_helps_but_not_enough`, with `readout_not_the_handicap` a close second.** The readout is the last
frozen tensor standing, PPO moves it more than anything else it trains, and a 2×4 matrix is a narrow
bottleneck for a 302-neuron network.

Against it: **R.1c's invariance is the hard fact**. Credited-synapse drift is 1.37–1.38× at every dimension
and the learning arm sat in a 2-food band across three axes. A rule writing 1.4× its own weight norm in a
direction that does not help looks like one whose credit assignment is wrong, and a readout cannot fix
credit assignment over 350 steps. That is why the negative branch hands off to e-prop rather than to
another readout variant.

## Reproduce

```bash
P=configs/scenarios/foraging/connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_hard350_nodepert_motor

# 1. harvest (done): PPO on this cell at the arm's own action scale
uv run python scripts/run_campaign.py \
  --config configs/scenarios/foraging/connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_hard350_readoutharvest.yml \
  --seeds 1-8 --runs 3000 --output-dir campaigns/readout-harvest -- --theme headless --track-experiment

# 2. prepare: one tensor substituted, everything else at that seed's own init
for SRC in ppo rotated anatomical_scaled; do
  uv run python scripts/prepare_readout_checkpoints.py --config ${P}.yml --source $SRC \
    --seeds 1-8 --harvest-dir campaigns/readout-harvest --out-dir campaigns/readout-checkpoints
done

# 3. the arms — 48 runs. `--track-experiment` is REQUIRED for the drift column.
uv run python scripts/run_campaign.py \
  $(for SRC in ppo rotated anatomical_scaled; do
      printf -- "--config %s_%s.yml --config %s_%s_frozen.yml " "$P" "$SRC" "$P" "$SRC"
    done) \
  --seeds 1-8 --runs 3000 --output-dir campaigns/readout-substitution \
  -- --theme headless --track-experiment

# 4. score against R.1c's committed anatomical pair
uv run python scripts/analysis/l4_frozen_readout.py \
  --anatomical campaigns/reduced-perturbation --substituted campaigns/readout-substitution \
  --seeds 1-8 --out docs/experiments/logbooks/supporting/062-l4-frozen-readout/frozen_readout.json \
  --csv docs/experiments/logbooks/supporting/062-l4-frozen-readout/per-seed.csv
```
