# 066: The Four-Class Pool Was Hiding the Wiring (7a-ii L.1 / Phase 7)

**Status**: completed — **`pooling_hid_structure`**, and it is **the first positive for the wild-type
wiring under a biologically plausible learner in this phase**. Crossing readout width with wiring on
the `hard350` cell over **96 paired seeds**, 768 of 768 runs succeeded, the **interaction is +0.2818**
on `auc_success` (CI [+0.2281, +0.3340], q = 0.000, wild-type-gains-more on **74 of 96** seeds):
widening the readout from the four-class mean-pool (8 parameters) to one weight per motor neuron (78)
helps the wild type **and hurts the degree-preserving null**. At the pooled width the **null is ahead**
(0.5106 against 0.4140) — [L.0](064-l4-frozen-features.md)'s result, reproduced on 96 seeds. At the
per-neuron width the **wild type is ahead** (0.5437 against 0.3585). The sign flips.

**No width main effect was detected**: **−0.0112, q = 0.591, CI [−0.0374, +0.0126]**. That is a
failure to detect, not a demonstration of absence — but the interval bounds any capacity effect well
below the **+0.2818** interaction, so a capacity-only explanation is not supported by this data. That
is the whole reason L.1 was registered as a crossed 2×2 rather than a comparison against L.0's
committed numbers.

All four learning gates fire at q = 0.000 and **neither prior detects a pre-update difference between
the wirings** (−0.509, q = 0.462 at both widths). **L.4 and L.5 reopen**, as their gates registered.

**One thing this does not establish**: *why the wide null got worse*. Nothing in the registered design
predicts it, and no mechanism is offered here.

**Branch**: `feat/l4-readout-width`.

**Date**: 2026-09-17.

**OpenSpec change**: `add-l4-readout-width` (extends `plasticity-evaluation`: a capacity manipulation
crossed with a structure contrast is read as an interaction; a metric is chosen for the contrast it
must support, and a departure is registered with its reason).

## Objective

L.0 returned `wiring_is_inert_as_features`: under `readout_only` — the one biologically plausible
learner on this substrate that reaches competence — the wild-type connectome showed no significant
advantage over its degree-preserving rewired null. But that learner reads the connectome through a
**mean-pool over four motor classes into an 8-parameter map**: 39 motor neurons enter, four numbers
leave, and each neuron carries `1/|class|` of its class's influence. If the wiring's advantage lives
in *which neuron* fires rather than *which class*, the pool destroys it before the learner sees it.

L.1 was **registered as a conditional before L.0 ran**, so the promotion is not a reaction to its
result.

> *Is the four-class pooling the bottleneck through which the wiring's features are invisible?*

## Method — why this had to be an interaction

A per-neuron readout has **78 parameters against 8**. Comparing it to L.0's pooled arm and reading a
gain would confound *"the wiring's features were there and the pool hid them"* with *"ten times the
parameters learn faster on anything"*. Crossing width with wiring separates them, and **only the
interaction** `(wt_wide − wt_pooled) − (rn_wide − rn_pooled)` distinguishes the two stories. Both main
effects are reported beside it and never in its place.

**Eight arms × seeds 1–96** at 3000 episodes: four learning arms and **a frozen floor at each width**,
because the two widths are the same policy but not the same run (below). The four wide configs differ
from their committed L.0 partners in the **`readout_width` key alone**, verified by loading both and
diffing the resolved config.

**The per-neuron readout is initialised by expanding the pooled one** — `W[k,i] = pooled[k,c(i)] / |class(i)|` — rather than drawn afresh. The `(2,4)` orthogonal draw still happens at the same point,
which buys three properties, all asserted by test: every non-readout parameter and float buffer is
**bitwise identical** across widths (the RNG stream is untouched, so a pooled arm is byte-identical to
L.0's runs); the two widths compute **the same policy at initialisation** (action means agree to
1.5e-8); and the anatomical contrast expands the same way. The only difference between the widths is
**the space the learner can move in**.

**`auc_success` is the primary, not block V's `episodes_to_30pct_success`** — a departure registered
with its reason before the campaign ran, because a difference of differences cannot be read on a metric
whose censoring rate varies across the cells being differenced. The result vindicates the choice: see
the censoring table below.

The protocol is [`supporting/066-l4-readout-width/launch.md`](supporting/066-l4-readout-width/launch.md).

## Results

### Gates and priors, read before the interaction

| test | contrast | Δ (foods over its own-width floor) | q | seeds |
|---|---|---|---|---|
| gate | `wt_pooled` | **+14.291** | 0.000 | 96/96 |
| gate | `rn_pooled` | **+14.684** | 0.000 | 96/96 |
| gate | `wt_wide` | **+10.814** | 0.000 | 88/96 |
| gate | `rn_wide` | **+6.828** | 0.000 | 74/96 |
| prior | `wt_pooled_frozen − rn_pooled_frozen` | −0.509 | 0.462 | — |
| prior | `wt_wide_frozen − rn_wide_frozen` | −0.509 | 0.462 | — |

All four arms plainly learned, so the interaction is interpretable. **Neither prior detects a
pre-update difference between the wirings** — identical at both widths, as expected, since the floors
are the same policy. At 96 pairs that is a failure to detect rather than a demonstration of absence,
but it is the check that the wirings are not separated before a single update.

### The 2×2 — `auc_success`, the primary (n = 96 paired)

| | pooled (8 params) | per-neuron (78) | widening buys |
|---|---|---|---|
| **wild type** | 0.4140 | **0.5437** | **+0.130** |
| **rewired null** | 0.5106 | 0.3585 | **−0.152** |
| wiring effect at this width | **−0.0966** (null ahead) | **+0.1852** (wild ahead) | |

All nine registered tests — the interaction, both main effects, the four gates and the two priors —
are corrected under **one BH-FDR family**, as the change registered. Raw p is kept beside each q in
the JSON.

| contrast | Δ | CI | q | seeds |
|---|---|---|---|---|
| **interaction (primary)** | **+0.2818** | [+0.2281, +0.3340] | **0.000** | **74/96** |
| width main effect | −0.0112 | [−0.0374, +0.0126] | 0.591 | 44/96 |
| wiring main effect | +0.0443 | [+0.0122, +0.0743] | 0.088 | 55/96 |

**Read the columns.** At the pooled width the null is ahead — L.0's finding, now on 96 seeds instead of
32\. At the per-neuron width the wild type is ahead. **No width main effect was detected**, with the
interval bounding it far below the interaction, so the extra parameters are not supported as doing the
work on their own; what the data shows is a gain on wild-type wiring and a cost on the shuffle.

### `episodes_to_30pct_success`, reported beside it — and why it is not the primary

| cell | mean episodes | censored |
|---|---|---|
| `wt_pooled` | 1051.4 | **17/96 (18%)** |
| `rn_pooled` | 729.4 | 8/96 (8%) |
| `wt_wide` | **239.9** | **2/96 (2%)** |
| `rn_wide` | 370.4 | 6/96 (6%) |

Interaction **−452.45 episodes, q = 0.005**. The metric is **lower-is-better**, so a negative
interaction means the wild type gained more — **the same direction as the primary**. The two metrics
agree.

**The censoring is exactly the asymmetry the registered metric choice anticipated**: 18% of `wt_pooled`
seeds never reached competence against 2% of `wt_wide`. Those seeds sit at the 3000-episode cap, which
inflates `wt_pooled`'s mean and would have inflated the interaction by construction had the censored
metric been primary. It is reported here with its censoring counted **per cell** so the reader can see
that.

**One cell-level divergence, stated rather than smoothed**: on episodes *both* wirings improve with
width (the null 729 → 370), while on `auc_success` the null gets *worse* (0.5106 → 0.3585). The
interaction agrees in direction on both metrics; the cell-level disagreement is what differential
censoring does to a right-censored mean, and it is another reason the uncensored metric carries the
reading.

### Sensitivity — the panel was sized for this, and the effect clears it

| | registered before the campaign | realised |
|---|---|---|
| correlation between widths, ρ | 0.02 (pilot, 4 seeds) | **0.083** |
| interaction sd | 0.3770 | 0.4160 |
| detectable at 80% | 0.1077 | **0.1189** |
| minimum interesting (sign-flip threshold) | **0.1076** | — |
| **observed interaction** | — | **0.2818** |

The pilot's near-zero ρ was right, which is why the panel was resized from 32 seeds to 96 before it
ran. The realised spread is slightly worse than the pilot predicted, so the detectable effect is 0.1189
rather than 0.1077 — and **the observed interaction is 2.4× that**, comfortably clear of the 0.108
sign-flip threshold the panel was built to resolve.

## Integrity — a disk-full crash, and what was checked because of it

The campaign filled the volume and died at run 337. Each run writes **~0.65–0.7 GB outside** the
campaign directory — ~440 MB of exports, of which ~380–415 MB is `session/data/detailed` (larger on
learning arms than frozen ones), plus ~256 MB of verbose log — so 768 runs needed ~500 GB. *(Corrected
2026-09-18: this was first recorded as "~966 MB per run, ~530 GB", a figure obtained by dividing the
whole `exports/` and `logs/` on the volume at the crash by L.1's 337 runs; that charged other
campaigns' output then on disk to L.1.)* A
cost registered nowhere and not measured before launch. The campaign directory itself is 161 MB, which
is why it was invisible until it was not.

What was done, and what it licenses:

- **Survivors were verified by parsing, not by file size.** One log was 4097 bytes and unparseable — a
  size check would have counted it as a completed run and fed a corrupt seed into the panel. It was
  deleted and re-run with the other 430.
- **The cleanup was scoped to L.1's own output.** Every session ID was mapped from the campaign logs to
  its export directory (337 matched, 0 missing); only `session/data/detailed` was removed, keeping
  `weights/final.pt` for the drift checks other harnesses read, and leaving **52 other campaigns'
  exports untouched**. All 337 still parsed afterwards, confirming the analysis never depended on it.
- **No code changed across the two halves.** No commits after the launch commit, and a clean tree for
  `packages/`, `scripts/` and `configs/`.
- **The one arm split across the crash was tested for a discontinuity.** `rn_wide` ran seeds 1–49
  before and 50–96 after: Mann-Whitney **p = 0.53**, against **p = 0.38** for a same-launch arm split
  at the same index. No detectable difference, on an arm that is one of the four interaction cells.

**Four further defects found in review, after the reading and before the merge.** None changes the
verdict; two were latent bugs that could have bitten a later arm, and all are recorded because the
panel's credibility rests on the instrument.

- **A cross-width checkpoint load mutated the brain before refusing it.** `load_state_dict` skips a
  mismatched tensor and raises at the *end*, having already copied the ones that matched — so a
  rejected load left the brain holding the file's `w_chem` beside its own readout, a mixed state no
  config describes. Shapes are now checked before anything is written; a test snapshots every
  parameter, attempts a rejected load, and asserts nothing moved.
- **The state-dependent log-std head was sized to the action count, not the readout width.** Under
  `per_neuron` it would have received 39 features into a 4-input layer. Latent here — no L.1 arm uses
  a state-dependent std — but it is the same `_N_ACTIONS` double-duty defect fixed elsewhere in this
  change and missed at this site.
- **The nine registered tests were not all in one BH-FDR family.** The gates and priors were
  corrected among themselves while the interaction and both main effects sat on raw p — not the
  procedure the change registered. Corrected across all nine: the interaction stays at q = 0.000, the
  priors move to 0.462, and the **wiring main effect moves from raw p = 0.058 to q = 0.088**, which
  changes no claim since it was already reported as secondary.
- **Stale comments in the four new configs**, inherited from their L.0 parents: the wide learning arms
  described an "8-parameter map over four pooled motor-class means" and a gradient taken with respect
  to a class mean, and the wide floors described a node-perturbation eligibility this campaign does
  not use. Documentation only; the resolved configs still differ from their parents in the
  `readout_width` key alone, verified by loading both.

**A harness defect found while reading, and fixed.** The agreement check compared the two metrics' raw
signs, but `auc_success` is higher-is-better and `episodes_to_30pct_success` is lower-is-better, so
agreement means **opposite** raw signs. It reported a direction disagreement that does not exist, and
would have written a false caveat into this record. The check now orients the censored metric before
comparing, with a test pinning both orientations.

## What this establishes, and what it does not

1. **The four-class pool was hiding wiring-specific structure.** The wild-type connectome's features
   are legible to a learner that can read individual motor neurons, and are not legible through a
   four-class mean. L.0's null was a fact about the readout as much as about the wiring.
2. **A capacity-only explanation is not supported.** No width main effect was detected (−0.0112,
   q = 0.591, CI [−0.0374, +0.0126]) — a failure to detect rather than a demonstration of absence,
   with the interval bounding any such effect far below the +0.2818 interaction. This is the claim
   the crossed design was built to license and the reason a comparison against L.0's committed
   numbers could not have licensed it.
3. **It does not say the pool is biologically wrong.** The four-class map is a stand-in for the
   neuromuscular system at either width; 39 weights is not more biological than 8, it is less.
4. **It is not an endpoint claim and not a read-across to block V.** Everything under `readout_only` is
   learning speed and area under the curve, and block V's effect is PPO on a different axis.
5. **No mechanism, and one open puzzle.** V.2 scored 64 rewirings on four graph properties fixed in
   advance and none predicts learning time. And **nothing here explains why the wide null got worse** —
   the registered design predicts an interaction, not that direction of it. Reported as an observation.
6. **D2's primary remains unmet.** This learner leaves `w_chem` frozen, so nothing here converts Phase
   7's SPLIT.

## Consequences

**L.4 (synapse signs as features) and L.5 (gap junctions as features) reopen**, on the gate registered
before L.0 ran: they were `closed-unopened` because against a null there is nothing for a feature
ablation to have changed. There now is.

*(Added 2026-09-19.)* **The wide wiring effect is rate-dependent.** L.4's rate-matched baseline
([Logbook 067](067-l4-feature-ablations.md)) re-ran `wt_wide` and `rn_wide` at `plasticity_rate`
0.0001 on the same 96 seeds: the wiring effect there is **−0.0977, null ahead**, against +0.1852
here. Both wirings learn better at the lower rate — the null by +0.44, the wild type by +0.16 — so
the verdict above stands as read **at 0.001** and may not be cited without that rate. The open
puzzle above gains a candidate, that 0.001 is too high for a 78-parameter readout on the rewired
graph, which nothing has tested.

*(Extended 2026-09-19.)* **The interaction itself does not survive the rate.** L.1b
([Logbook 068](068-l1b-rate-calibration.md)) completed this 2×2 at 0.0001 — the two pooled learning
cells were all that was missing — over the same 96 seeds: **the interaction there is −0.0657**
(q = 0.000) against **+0.2818** here, a **three-way of +0.3475** (q = 0.000, 81/96) that is larger
than this effect, and **the sign reverses**. At 0.0001 the dominant effect is capacity — width main
effect **+0.6178 on 96/96 seeds**, where this panel detected none — so how much readout width matters
is itself set by the rate, and the wild type leads at **neither** width there. L.1b's reverse
direction is **not credited** (23% of its registered minimum, primary axis only, secondary voided by
censoring). **The verdict above is unchanged and is a verdict at `plasticity_rate` 0.001**; the "sign
flips" reading belongs to width × wiring × rate.

## Artefacts

- [`supporting/066-l4-readout-width/launch.md`](supporting/066-l4-readout-width/launch.md) — the protocol, the pilot, the sensitivity arithmetic
- [`supporting/066-l4-readout-width/readout_width.json`](supporting/066-l4-readout-width/readout_width.json) — the full reading
- [`supporting/066-l4-readout-width/per-seed.csv`](supporting/066-l4-readout-width/per-seed.csv) — one row per cell, arm and seed, with the censoring column
