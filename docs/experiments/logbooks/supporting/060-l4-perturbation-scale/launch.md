# R.1 — the perturbation dimension: the registered protocol

Registered in `openspec/changes/add-l4-perturbation-scale`, reviewed and committed **before** the run.

## The question

Node perturbation forms its eligibility from what each unit's own noise did to a scalar outcome. With
N perturbed units the per-trial gradient estimate has signal-to-noise ~1/√N, so the trials needed for
a given amount of progress grow roughly as **N** ([Werfel, Xie & Seung 2005][wxs]). That dimension
differs across every platform this rule has been measured on, and **no experiment has varied it**:

| platform | plastic layers | perturbed units | draws per decision | result |
|---|---|---|---|---|
| the one-step control | 1 (`Linear(K, 8)`) | **8** | 8 | **passes** — 89.0% of the floor-to-optimum gap (I.1) |
| the MLP yardstick | 2 (`64 → 64`) | **128** | 128 | fails, and below its own frozen control (I.3b) |
| the connectome | 1 (302 neurons) | **302** | **1208** — every unit at each of 4 settling steps | fails on every multi-step task (040–047) |

The pattern this phase reads as "one-step works, multi-step fails" is equally consistent with "8 units
works, 128 and 302 do not".

## S1 — the arithmetic, where the rule works

The committed one-step contextual-association control at `HIDDEN ∈ {8, 16, 32, 64, 128}`, seeds 1–8,
20 000 trials, at I.1's passing configuration — σ 0.2, rate 1e-3, `trace_decay` 0.9, homeostasis on,
action noise `exp(-1)` — with the width as the only axis.

**Why this platform carries the arithmetic.** The task is solvable by 8 units, so no width in the grid
lacks capacity and every further unit adds only noise. N is isolated in a way it cannot be on a
behavioural cell, where width is capacity too.

**What is measured**

1. **The control's own registered pass rule** per width: mean at least halfway from the closed-form
   cue-blind floor to the closed-form optimum, on ≥ 7 of 8 seeds.
2. **Trials-to-criterion** per seed — the first trial whose trailing 100-trial mean crosses that
   threshold. This is the quantity 1/N makes a claim about; a pass/fail reading discards it. A
   two-consecutive-block variant is reported alongside as a robustness column.
3. **The analytic reference at every width.** The floor and the optimum are closed-form properties of
   the *task*, but what a network with a **frozen random readout** can reach is a property of the
   *width*. A width where the reference itself misses the pass bar is **void at that width**.
4. **Reachability-normalised score** — the rule's gap fraction over the reference's at the same width,
   beside the raw fraction. The raw fraction is what the earlier results quote; the normalised one is
   what is comparable across widths.

**The 1/N test.** OLS of `log2(trials)` on `log2(N)` over per-seed values, bootstrap CI over seeds
(resampling seeds, not points, so the pairing survives). The prediction is a slope of **+1**; the
registered bar for "depends on N in the predicted direction at all" is **≥ 0.5 with a CI excluding
0**. A Spearman of the per-width medians is reported as description only.

**Censoring is reported, not absorbed.** A seed that never crosses has no criterion time: those seeds
are excluded from the fit and **counted**, per width. V.3 came within one review comment of a false
null from exactly this shape — gates reading positive while 0% of seeds crossed the threshold.

**The derived budget** for 128 units and for the connectome's 302 units / 1208 draws is an
**extrapolation from a five-point fit read outside its range**, is flagged as one in the JSON and in
print, and is never reported as a measurement. Both readings of the connectome's dimension are given,
because they differ by 4× and nothing in the record says which the arithmetic tracks.

## S2 — the rescue, where the rule fails

The MLP yardstick on the calibrated hard-food cell — block V's cell, so a rule that learns it is
directly comparable to the **+23.5%** PPO achieved there — at
`actor_hidden_dim ∈ {4, 8, 16, 32, 64}` with two hidden layers and hidden-only plasticity: **8, 16,
32, 64 and 128 perturbed units, the same grid as S1**. Seeds 1–8, 3000 episodes.

| arm | rule | `freeze_updates` | σ | what it is for |
|---|---|---|---|---|
| learning | `three_factor` / `node_perturbation` | false | 0.2 | the measurement |
| frozen | `three_factor` / `node_perturbation` | **true** | **0.2** | the do-nothing floor **at that width** |
| capability | `ppo` | false | — | whether the width can hold a competent policy at all |

**The frozen arm perturbs.** It carries the same σ and freezes only the *update*. The cost the
perturbation imposes on the policy is therefore present in both arms and **cancels**, so the contrast
measures the benefit of the update alone — which is what a claim about estimator quality needs — and
not the net effect of switching perturbation on.

**The capability arm exists because the prediction runs toward small N**, which is exactly where
capacity runs out: without it a width-4 null could not be told from a width-4 refutation. It shares
`activation: tanh`, the width, the layer count, `initial_log_std` and `entropy_coef` with the plastic
arms, so what it certifies is this network's capacity and not another's. A width failing it is
**uninterpretable**, never a null, and is excluded from the trend with the exclusion printed. Two
parts, both reported: it beats the do-nothing floor, and it reaches the committed competence
threshold. Its comparator perturbs and it does not, so it is a **floor check on the width, not a
matched pair**, and it is **not** the committed calibrated MLP arm for this cell (that one is relu at
width 64).

**The reading.** Plateau-tail mean foods, I.2's graded family, learning against its own frozen
control, paired, one-sided, BH-FDR across the five widths. Full-clear success is recorded and is
expected at the floor; where it is, the record says so rather than reporting a null. **A shift counts
only if it is significant *and* both** at least **1.0 foods** of the cell's 20 — I.3b's 0.5-of-10 bar
in proportion — **and** at least **10% of that width's own PPO-minus-frozen gap**.

**Drift** per width — relative weight distance from the frozen control, within a width only — so
*starved of signal and sitting still*, which is what 1/N predicts at large N, is distinguishable from
*writing a great deal in a worsening direction*, which is what I.3b measured at 1.28–1.31× the
weight's own norm and which no budget fixes.

**A declared alternative at the small-N end.** Width 4 sits **below the input dimension** and is the
only grid point giving 8 perturbed units, the count at which the rule passes its control. If its
capability arm fails, `num_hidden_layers: 1` at width 8 supplies the same 8 units without a
bottleneck — used **only** on that failure, its one-layer architecture stated wherever its number
appears, and **added** rather than substituted, so width 4's failure stays on the record.

## Outcomes, fixed before the run

| verdict | S1 | S2 | what follows |
|---|---|---|---|
| `scale_limited` | slope ≥ 0.5, CI excluding 0, **and 128 units fails the control's own pass rule** | ≥ 1 width beats its frozen control by both minima | the failures are **located**: a scale property. A reduced-perturbation connectome variant becomes the obvious registration, and every panel negative is re-read as under-budgeted — as a **new registration**, not a re-labelling of a committed verdict |
| `arithmetic_only` | slope ≥ 0.5, CI excluding 0 | no width beats its control | the arithmetic is real **and does not rescue the task**: the multi-step failure is a **second, independent** defect, with I.3b's worsening-direction drift the standing candidate. Node perturbation closes as a family member; R.2 (e-prop) proceeds. **This is the expected outcome** |
| `not_scale_limited` | CI contains 0 **and** 128 units passes | reported, and cannot be read as being about scale | the arithmetic is not the binding constraint. The 8-vs-128 confound closes the uninteresting way and **every existing negative keeps its reading** |

A mixed reading — a slope below the bar with 128 failing, or a positive S2 with a flat S1 — is
recorded as **mixed with both halves stated**, not resolved toward the nearer verdict.

### Stop clauses — void until found, not results

- **`HIDDEN = 8` does not reproduce I.1's pass.** The platform has drifted; nothing is interpretable
  until that is found.
- **The analytic reference fails at a width.** That width is void; at 8 units, the whole sweep is.
- **The pilot finds no room after the declared remedy** — the committed C1 food-only cell
  (`max_steps: 800`, target 10), applied as a dated amendment carrying the pilot table that forced it,
  with block-V comparability given up. S2 is then reported as platform-limited and S1 carries the
  change alone: **S1 does not depend on S2 and is not weakened by its absence.**

## Honest prior

**`arithmetic_only`, with S1's 128-unit cell the genuinely open question.**

The reason to doubt a rescue is I.3b's drift number. A rule starved of signal per trial moves little
and slowly; this one writes **1.28–1.31× its own weight norm** in a direction that makes the policy
worse, at every eligibility horizon. That is not the signature of too little signal — it is the
signature of a signal pointing the wrong way, and no budget repairs a sign.

The reason to run it anyway is that the record cannot currently say whether the rule was ever given a
workable scale, S1 costs minutes, and a confirmed slope changes what the phase's negatives mean even
with no rescue: "the rule fails on multi-step tasks" becomes "the rule fails on multi-step tasks **and
was run at 16–151× the dimension its one success was measured at**", which is a materially different
sentence to publish.

## Pre-run verification

The width is a parameter with the pinned `HIDDEN = 8` as its default, so every value recorded by
I.0–I.3b must reproduce unchanged. **Checked before the sweep**: the committed control was re-run at
the default and compared against
[`048-l4-rule-positive-control/control.json`](../048-l4-rule-positive-control/control.json) —
**77 common leaves, 0 differing, none missing**. The 49 keys present only in the re-run are the
node-perturbation arms I.1 added after 048 was written.

## Amendment, 2026-09-13 — the depth control

S1's registered grid ran first, as the protocol requires, and **the rule passes at every width, 8
through 128**, with no growth in time-to-criterion (slope **−0.149**, CI [−0.239, −0.061]). That closes
the registered question and leaves **one** shape difference between the platform the rule passes and
the platform it fails: 128 units as **one** layer of 128 here, against **two** of 64 in the yardstick.

**Added**: the yardstick's exact arrangement — `hidden 64, layers 2`, 128 perturbed units — on the same
control, same seeds, same budget, same pass rule, reported beside the matched one-layer cell. It is the
cheapest remaining way to separate shape from task, and the registered grid cannot make that
comparison.

It was chosen **after** seeing S1's slope and is reported with that provenance. A pass means the
arrangement is not the problem and the failure localises to multi-step credit assignment — what I.3's
delay result predicted in advance. It says nothing about long episodes, which is S2's question.

## Disclosure

The S2 pilot runs on seeds **101–104**, disjoint from the registered 1–8. S1 has no pilot: it runs on
the registered seeds directly, because it is the committed control at a new width rather than a new
platform.

## Reproduce

```bash
# S1 — the arithmetic (minutes; its stop clauses gate the rest)
uv run python scripts/analysis/l4_perturbation_scale.py --s1 \
  --out docs/experiments/logbooks/supporting/060-l4-perturbation-scale/scale.json \
  --csv docs/experiments/logbooks/supporting/060-l4-perturbation-scale/s1-per-seed.csv

# S2 pilot — the two extreme widths, all three arms, disjoint seeds
P=configs/scenarios/foraging/mlpppo_small_continuous2d_fick_adaptive_klinotaxis_hard350
uv run python scripts/run_campaign.py \
  --config ${P}_nodepert_w04.yml --config ${P}_nodepert_w04_frozen.yml --config ${P}_ppo_w04.yml \
  --config ${P}_nodepert_w64.yml --config ${P}_nodepert_w64_frozen.yml --config ${P}_ppo_w64.yml \
  --seeds 101-104 --runs 3000 --output-dir campaigns/perturbation-scale-pilot \
  -- --theme headless --track-experiment

# S2 campaign — 120 runs. `--track-experiment` is REQUIRED: the drift column reads final.pt
# through the experiment record's exports_path, and without it every drift figure comes back empty.
uv run python scripts/run_campaign.py \
  $(for W in 04 08 16 32 64; do
      printf -- "--config %s_nodepert_w%s.yml --config %s_nodepert_w%s_frozen.yml --config %s_ppo_w%s.yml " \
        "$P" "$W" "$P" "$W" "$P" "$W"
    done) \
  --seeds 1-8 --runs 3000 --output-dir campaigns/perturbation-scale \
  -- --theme headless --track-experiment
```

[wxs]: https://papers.nips.cc/paper_files/paper/2003/hash/f7e9050c92a851b0016442ab604b0488-Abstract.html

______________________________________________________________________

## Pilot outcome, 2026-09-13 — the platform has room, and the pilot inverted the registered prior

24 runs on disjoint seeds 101–104, all succeeded, 1048 s wall clock at 16 workers.

| width | perturbed units | PPO foods | PPO clear | frozen foods | learning foods | learning clear | effect | drift |
|---|---|---|---|---|---|---|---|---|
| 4 | **8** | 19.76 | 90.4% | 0.68 | **19.63** | **92.8%** | **+18.94** | 0.958 |
| 64 | **128** | 19.79 | 94.5% | 3.27 | 4.78 | 4.1% | +1.51 | 1.400 |

**The registered purpose is satisfied**: PPO sits at ~19.8 of 20 foods and 90–94% full clear while the
frozen controls sit at 0.68 and 3.27 foods and 0% clear, so the platform has room at both extremes and
the **declared remedy is not applied**. The cell and its 350-step budget stand as calibrated.

**The pilot also inverted the honest prior.** At eight perturbed units the rule reaches 19.63 foods
and 92.8% full clear — level with PPO — from a frozen floor of 0.68. At the yardstick's own 128 units
it reaches 4.78 against a frozen 3.27 while PPO on that width reaches 19.79. Drift separates the two
regimes: 0.958 of the weight's own norm where the rule solves the cell, 1.400 where it barely leaves
its floor, against the 1.28–1.31 I.3b measured at 128 units.

**No significance claim is taken from the pilot and none is available.** At four pairs the smallest p
an exact one-sided paired test can return is **0.0625**, above the 0.05 level, so the floor half of
each capability gate cannot fire whatever the data does. Both widths therefore read
`capability_undecided` rather than failed, and the effect sizes above are **descriptive**. The
registered campaign runs eight seeds, where that floor is 0.0039.

This does not contradict S1. A one-step task has a single credited decision, so the estimator's
dimension barely matters there; over 350 steps the per-unit credit is diluted across units **and**
time. The two sweeps together say the dimension bites only when the horizon does — and the campaign's
job is now to **locate the breakpoint between 8 and 128 units**, which the registered grid already
spans.

**Two instrument defects were found by the pilot and fixed before the campaign**, both of the shape
that turns a sample size into a finding:

- drift took its seed set from I.3b's module constant, so on 101–104 it read no pair at all and
  reported nothing available — the right answer for the wrong reason, and a silent mismatch for any
  campaign not on seeds 1–8;
- the capability gate reported **fails** when its test could not reach the level at that many pairs.
  It now reports `underpowered` with the smallest reachable p, and a width whose gate is undecided is
  neither a null nor uninterpretable.
