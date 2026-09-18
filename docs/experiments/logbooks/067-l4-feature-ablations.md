# 067: Both Features Carry the Per-Neuron Effect, and the Effect Depends on the Learning Rate (7a-ii L.4 + L.5 / Phase 7)

**Status**: completed — **L.5 `carries_the_effect`** and **L.4 `carries_the_effect`, qualified *carries
or unlearnable***; and a third result the registration did not name, which conditions the first two
and [L.1](066-l4-readout-width.md) itself. Ten arms over **96 paired seeds**, 960 of 960 runs
succeeded.

**L.5 — gap junctions off.** The interaction is **−0.2035** on `auc_success` (CI [−0.2449, −0.1564],
q = 0.000, feature-carried-it on **73 of 96** seeds): with the electrical synapses zeroed, the wild
type's +0.1852 lead over its degree-preserving null at the per-neuron width **is gone** — the ablated
wiring effect is **−0.0183** (q = 0.033, null marginally ahead). But it is gone because **both wirings
learned far better without gap junctions** — the wild type from 0.5437 to 0.7322, the null from 0.3585
to 0.7506 — and the null gained twice as much. Gap junctions are a handicap for this learner on both
wirings, and a larger one on the rewired graph. That is what `carries_the_effect` means here, and it
is not what the word suggests.

**L.4 — atlas signs, at the rate-matched 0.0001.** The interaction is **−0.1319** (CI \[−0.1678,
−0.0986\], q = 0.000, **66 of 96**), clearing the registered 0.123 minimum, and the **gains diagnostic
fires**: both wirings learn less with grounded signs than with the random draw (−2.9 and −0.5 foods
over their floors, q = 0.000 and 0.046), so the reading is **carries or unlearnable**. And the premise
the reading was built on failed at this rate: **in the rate-matched wide baseline the null is ahead**
(0.7990 against 0.7013, wiring effect **−0.0977**). There was no wild-type advantage at 0.0001 for the
signs to carry. What the interaction measures is that grounding the signs hurts the wild type
(−0.270) more than the null (−0.138).

**The result nobody registered: L.1's positive is rate-dependent.** On the same 96 seeds and wirings,
the per-neuron wiring effect is **+0.1852** at `plasticity_rate` 0.001 and **−0.0977** at 0.0001.
Lowering the rate helped the wild type by +0.158 and the null by **+0.440**. L.1's "first positive for
the wild-type wiring under a plausible learner" holds at the rate it ran at and reverses one decade
below it. This is a descriptive, unregistered contrast between two campaigns — reported with its size
because it conditions everything above, and not as a result of its own.

**The structural probe on L.1's puzzle is null** (ρ = +0.032, p = 0.378): among the 96 rewirings,
within-class input correlation does not predict how much the per-neuron readout hurt. The rewirings
span 0.014–0.027 on that statistic against 0.154 for the wild type, so the null is a null over a
spread an order of magnitude narrower than the contrast the hypothesis was drawn from.

**Branch**: `feat/l4-feature-ablations`.

**Date**: 2026-09-19.

**OpenSpec change**: `add-l4-feature-ablations` (extends `plasticity-evaluation`: a feature ablation on
a positive registers a minimum effect as a decision rule; a committed baseline is reused only under a
byte-identity check; a mechanism probe is registered before its correlation is computed).

## Objective

L.1 read `pooling_hid_structure`: at the per-neuron readout width the wild type leads its
degree-preserving null by **+0.1852** on `auc_success`, where at the pooled width the null led. A
positive on "the wiring" cannot say which part of the wiring carries it, so L.4 and L.5 — gated before
L.0 ran on L.1 reading positive — each remove one feature at the per-neuron width and ask whether the
effect survives:

- **L.4** grounds the synapse signs in the atlas (`synapse_signs: atlas`), replacing the random draw on
  3,176 of 3,709 chemical synapses; magnitudes, norms and the RNG stream untouched. **Not a clean
  removal**, and registered as such: the readout pool's 311 grounded inputs go from ~50/50 to **275 E /
  36 I**, which can move the tanh operating point — the reason two diagnostics exist.
- **L.5** zeroes the electrical matrix in the forward pass (`enable_gap_junctions: false`); **every
  parameter bitwise identical**. 199 of 1,093 gap junctions touch the pool, 47 lie within it, all 39
  pool neurons carry one. Both wirings lose them symmetrically: the null rewires them too.

Each is read as an interaction against L.1's wide baseline, `I = (wt_abl − rn_abl) − (wt_wide − rn_wide)`, per seed, with `carries_the_effect` requiring significance **and** `abs(Δ) ≥ 0.123` —
two-thirds of the +0.1852 that an ablation at this width can actually remove. The full protocol, both
dated amendments and the rate-check outcome are in
[`supporting/067-l4-feature-ablations/launch.md`](supporting/067-l4-feature-ablations/launch.md).

## Method — what ran, and the two departures registered before it ran

**Eight ablated arms**, each one key from its committed L.1 wide parent (asserted by exact-key test):
two learning arms and two frozen floors per ablation. **The baseline is L.1's committed wide arms**,
reused on evidence: `wt_wide` and `rn_wide` at seed 1 were re-run under the new output controls
(`--no-detailed-export --no-file-log`) and reproduced L.1's logs on all nine parsed fields
(`campaigns/export-flags-identity`). Both halves of every interaction pass through the same
`connectome_structure_efficiency` call from campaign logs.

**Departure 1 — the gains diagnostic** (amendment, 2026-09-18, after the pilot and before any
registered seed). The pilot's atlas arms gained +2.2 and +2.8 foods over their floors against the wide
arms' +13.7 and +7.8, bimodally, with the floors diagnostic quiet. That is a case the rules as written
read wrong — a `carries` that is really "the substrate became unlearnable for both wirings and the wild
type had more to lose". The diagnostic compares each ablated arm's gain over its floor to the wide arm's,
per wiring; if both are significantly smaller, a `carries` on L.4 is reported *carries or
unlearnable*. Registered for atlas only, since the pilot's nogap gains exceeded the baseline's.

**Departure 2 — the registered rate check** (second amendment, same day). The collapse had a
mechanistic candidate — 275 E / 36 I pushing the readout's presynaptic activities toward saturation,
the signature of a rate too high for the substrate — and one precedent (L.0's own rate check). Both
atlas learning arms ran at 0.0001 and 0.01 on disjoint seeds 101–104 under a three-outcome rule fixed
before the runs. **Outcome B fired**: 0.0001 learned cleanly on both wirings (+15.0 and +16.1 foods
over floor, no seed below floor) against +2.2/+2.8 at 0.001 and a full collapse at 0.01. Per the rule,
L.4's learning arms run at **0.0001**, and two wide learning arms at 0.0001 — one key from L.1's wide
parents, piloted on 101–104 (+17.3 and +15.0 over floor) — join as the **rate-matched baseline**, so
L.4's interaction compares learners at one rate. Floors are reused, the rate being inert under
`freeze_updates`. **L.5 is unchanged at 0.001 against L.1's baseline.** The honest prior for the check
was "A or C"; B fired, and the prior stands as written.

**The campaign**: ten arms × seeds 1–96 at 3000 episodes, **960 runs**, with `--track-experiment --no-detailed-export --no-file-log`; disk measured before launch at 17.1 MB of exports plus a 0.46 MB
log per run.

## Results

### Gates and priors, read before either interaction

| ablation | test | contrast | Δ (foods over its own floor) | q | seeds |
|---|---|---|---|---|---|
| L.4 | gate | `wt_atlas` @ 0.0001 | **+13.275** | 0.000 | 93/96 |
| L.4 | gate | `rn_atlas` @ 0.0001 | **+15.443** | 0.000 | 96/96 |
| L.4 | prior | `wt_atlas_frozen − rn_atlas_frozen` | −0.295 | 0.715 | — |
| L.5 | gate | `wt_nogap` | **+17.003** | 0.000 | 96/96 |
| L.5 | gate | `rn_nogap` | **+16.803** | 0.000 | 95/96 |
| L.5 | prior | `wt_nogap_frozen − rn_nogap_frozen` | −0.089 | 0.520 | — |

All four ablated learning arms plainly learn, and neither prior detects a pre-update difference
between the wirings, so both interactions are interpretable. All ten registered tests — per ablation
the interaction, the ablated wiring effect, two gates and the prior — sit in **one BH-FDR family**; raw
p is beside each q in the JSON.

### L.5 — gap junctions off, at 0.001 against L.1's baseline (n = 96 paired)

| | wide (L.1) | gap junctions off | removal buys |
|---|---|---|---|
| **wild type** | 0.5437 | **0.7322** | **+0.189** (75/96) |
| **rewired null** | 0.3585 | **0.7506** | **+0.392** (94/96) |
| wiring effect | **+0.1852** (wild ahead, 71/96) | **−0.0183** (q = 0.033, 38/96) | |

| contrast | Δ | CI | q | seeds |
|---|---|---|---|---|
| **interaction (primary)** | **−0.2035** | [−0.2449, −0.1564] | **0.000** | **73/96** |
| ablated wiring effect | −0.0183 | [−0.0331, −0.0016] | 0.033 | 38/96 |
| minimum for `carries` | abs(Δ) ≥ 0.123 | — | **cleared** (removes 110% of +0.1852) | |

**Reading: `carries_the_effect`.** The interaction is significant, negative, and above the minimum:
without gap junctions the wild type's advantage is not merely reduced but gone, with the null a hair
ahead. **Read the cells, not the label.** Both wirings improved, and by a lot — the null by more than
twice what the wild type gained. The gains diagnostic, computed for both ablations though registered
for atlas only, puts numbers on it: the nogap arms gain **+6.189** (wild type, q = 0.000, 85/96) and
**+9.975** foods (null, q = 0.000, 93/96) *more* over their floors than the wide arms do. Nothing
about "gap junctions carry the wild-type advantage" is false, and nothing about it is what a reader
would take it to mean: the advantage consists in the wild type's gap junctions **costing it less** than
the rewired gap junctions cost the null.

**The floors diagnostic fires on L.5, and cannot qualify it.** The nogap frozen floors sit **below** the
wide floors: −1.089 foods (wild type, q = 0.001) and −1.509 (null, q = 0.000). Removing gap junctions
moves the frozen operating point on both wirings. The diagnostic was registered for L.4 alone, so it
qualifies nothing here — a post-hoc qualification is exactly what the spec forbids — and it is reported
because the record would be misleading without it. Were it registered for L.5, the reading would be
*carries or saturates*.

### L.4 — atlas signs, at 0.0001 against the rate-matched wide baseline (n = 96 paired)

| | wide @ 0.0001 | atlas signs @ 0.0001 | grounding buys |
|---|---|---|---|
| **wild type** | 0.7013 | 0.4315 | **−0.270** (20/96 positive) |
| **rewired null** | **0.7990** | 0.6610 | **−0.138** (29/96 positive) |
| wiring effect | **−0.0977** (**null** ahead, 20/96) | **−0.2296** (null ahead, 16/96) | |

| contrast | Δ | CI | q | seeds |
|---|---|---|---|---|
| **interaction (primary)** | **−0.1319** | [−0.1678, −0.0986] | **0.000** | **66/96** |
| ablated wiring effect | −0.2296 | [−0.2623, −0.1978] | 0.000 | 16/96 |
| baseline wiring effect (reference, outside the family) | −0.0977 | [−0.1150, −0.0792] | raw p = 2e-9 | 20/96 |
| minimum for `carries` | abs(Δ) ≥ 0.123 | — | **cleared** (71% of +0.1852) | |

**Reading: `carries_the_effect`, qualified *carries or unlearnable*.** The interaction is significant,
negative and above the minimum, and the **gains diagnostic fires**: the atlas arms gain **−2.884**
foods less over their floors than the wide arms (wild type, q = 0.000, 30/96 positive) and **−0.516**
less (null, q = 0.046, 40/96). Grounding the signs reduces learnability on both wirings at the same
rate, the wild type's more. The **floors diagnostic is quiet** (−0.227 and −0.441 foods, q = 0.933), so
the frozen operating point did not detectably move; what moved was what could be learned on it.

**And the reading's premise is not met at this rate.** The registered rule treats the ablation as
subtracting from a positive wild-type advantage, and reports the interaction as a fraction of
+0.1852 — L.1's figure, measured at 0.001. In the rate-matched baseline the wiring effect is
**−0.0977: the null is ahead** before any sign is grounded. The 71% is arithmetic against a referent
that is not present in the cells the interaction was computed from. What the interaction says, in the
only form it can be cited in: **at the per-neuron width and 0.0001, grounded signs hurt the wild type
by more than they hurt the null**, on 66 of 96 seeds. It does not say the signs carried a wild-type
advantage, because at this rate there was none to carry.

### The wide wiring effect depends on the learning rate — unregistered, and the record's third result

The rate-matched baseline was added so that L.4's interaction would compare learners at one rate. It
also measured, on the same 96 seeds and the same 96 rewirings, what the wide arms do one decade below
L.1's rate. The `launch.md` said this number "will be measured at 96" and refused to read it at four.

| wide arm | `auc_success` @ 0.001 (L.1) | @ 0.0001 | lowering the rate buys |
|---|---|---|---|
| **wild type** | 0.5437 | 0.7013 | **+0.158** (71/96) |
| **rewired null** | 0.3585 | **0.7990** | **+0.440** (95/96) |
| wiring effect | **+0.1852** (wild ahead) | **−0.0977** (null ahead) | rate × wiring **−0.283**, CI [−0.343, −0.222], 16/96 |

Both wirings learn better at the lower rate; the null learns *much* better. **The sign of the per-neuron
wiring effect flips with the rate.** No test on this contrast was registered, so no q is attached and
it is not a reading — but it is not noise-shaped either (95 of 96 rewirings improved), and it is the
single fact that most changes how the rest of this record, and L.1, may be cited. L.1's positive is a
positive **at `plasticity_rate` 0.001**, the value R.2 pinned at the pooled width and L.1 inherited
without a sweep at the new width. **L.1's open puzzle — why the wide null got worse — has a candidate
in it**: 0.001 is too high a rate for a 78-parameter readout on the rewired graph, and the wild type
tolerated it. That is a candidate, not a mechanism. It predicts that L.1's interaction shrinks or
reverses at 0.0001, which a rate × width × wiring design would test and this record does not.

### `episodes_to_30pct_success`, beside the primary, with censoring per cell

| cell | mean episodes | censored |
|---|---|---|
| `wt_wide` @ 0.0001 | 306.1 | 0/96 |
| `rn_wide` @ 0.0001 | 184.4 | 0/96 |
| `wt_atlas` @ 0.0001 | **1147.6** | **22/96 (23%)** |
| `rn_atlas` @ 0.0001 | 406.3 | 3/96 |
| `wt_wide` @ 0.001 (L.1) | 239.9 | 2/96 |
| `rn_wide` @ 0.001 (L.1) | 370.4 | 6/96 |
| `wt_nogap` | 189.8 | 0/96 |
| `rn_nogap` | 173.6 | 1/96 |

The metric is lower-is-better. **L.4** agrees with the primary in direction and significance: the
interaction is **+619.6 episodes** (p = 6.5e-7, 64/96), grounding costing the wild type far more time
than the null — and 22 wild-type atlas seeds never reached competence, against none in either wide
arm, which is the asymmetric censoring the registered metric choice anticipated and why `auc_success`
carries the reading. **L.5** agrees in direction only: +146.7 episodes, **p = 0.229**, not
significant — both nogap arms reach competence in under 200 episodes, so there is little time for the
metric to separate. A significant primary beside a non-significant secondary is reported as that; it
does not weaken the primary and it is not smoothed into agreement.

### Sensitivity

| | registered | L.4 realised | L.5 realised |
|---|---|---|---|
| interaction sd | 0.416 | 0.2693 | 0.3300 |
| standard error | 0.0425 | 0.0275 | 0.0337 |
| detectable at 80% | 0.119 | 0.0770 | 0.0943 |
| power at the 0.123 minimum | ~80% | 0.99 | 0.95 |
| observed interaction | — | **−0.1319** | **−0.2035** |

The panel was sized for the minimum and resolved well past it on both ablations. The realised spreads
are narrower than L.1's, so the detectable effect is smaller than registered; the registered minimum,
not the realised one, is what the readings were held to.

### The structural probe — null, and the reason it had to be

Registered before its correlation was computed: across the 96 rewirings, Spearman ρ between a seed's
within-class presynaptic Jaccard and its `rn_wide − rn_pooled` from L.1, one-sided positive, minimum
ρ ≥ 0.3.

| | |
|---|---|
| ρ | **+0.032** |
| p (one-sided, positive) | 0.378 |
| reading | **`probe_null`** — the puzzle is not this, and stays a puzzle |
| wild-type Jaccard (VB, DB, VA, DA) | 0.068, 0.140, 0.175, 0.231; mean **0.154** |
| rewired Jaccard, 96 seeds | mean **0.020**, range 0.014–0.027 |

The descriptive companion holds: the wild type's within-class input correlation exceeds every one of
the 96 rewirings, by 6×. The registered test ran **across rewirings**, where the statistic takes 96
distinct values with a standard deviation of 0.003 (coefficient of variation 0.17) — a real spread,
and a rank correlation is well defined on it. What the null says is that, **within that spread**,
more within-class correlation did not go with a smaller loss from widening. What it does not reach is
the contrast the hypothesis was drawn from: the wild type sits 0.13 above the highest rewiring, an
order of magnitude beyond the range the test could see. The feasibility look compared the wild type
against three rewirings and never asked how the predictor spreads across the population the test
would run over; had it, the test would have been registered as the narrower question it is. **Not
evidence against the hypothesis; a null on the part of it this test reaches.** A test that reaches
the rest would manipulate within-class correlation directly rather than wait for a null model to
vary it.

## Integrity

- **960 of 960 runs succeeded**, in 78,661 s of wall-clock across ten arms, all parseable by `read_log`.
  The campaign directory is 469 MB; the per-run export cost under the controls was measured before
  launch and held.
- **No code changed during the campaign.** No commits touching `packages/`, `scripts/run_simulation.py`,
  `scripts/run_campaign.py` or `configs/` since the launch commit; a clean tree at scoring.
- **The baseline reuse is licensed by evidence**: the seed-1 identity check reproduced L.1's logs on
  every parsed field. L.5's baseline is L.1's committed 0.001 wide arms; L.4's baseline is the
  rate-matched 0.0001 re-run of the same two configs (one key from their parents) with L.1's floors.
- **Everything computed for both ablations is reported for both**, including the two diagnostics
  registered for atlas alone; what was registered for one is applied to one.
- **Both honest priors were wrong.** L.5 was registered as "survives" — gap junctions being the part of
  the wiring most like the degree statistics the null preserves — and read `carries`. L.4 was
  "uncertain, leaning carries for the operating-point reason", and the label came back as predicted
  while the operating point did not move (floors quiet) and the sense of it — subtracting from a
  wild-type advantage — did not hold at the matched rate. The rate check's prior ("A or C") was wrong
  too. All three stand as written.

## What this establishes, and what it does not

1. **At the per-neuron width and `plasticity_rate` 0.001, removing gap junctions removes the wild-type
   advantage** — by lifting both wirings, the rewired null more. The advantage consists in the wild
   type being hurt less by its gap junctions than the null is by its rewired ones. Reading:
   `carries_the_effect`. **May not be cited as** "gap junctions help the wild type", as evidence that
   the electrical wiring is functional structure, or as anything about 0.0001, where L.5 did not run.
2. **At the per-neuron width and 0.0001, grounded signs reduce learnability on both wirings, the wild
   type's more**, with the frozen operating point not detectably moved. Reading: `carries_the_effect`,
   qualified *carries or unlearnable*. **May not be cited as** "the signs carry the wild-type
   advantage" — at this rate the null is ahead before grounding — nor, per the launch record, as
   evidence that the *signs* are the feature while the gains diagnostic is fired.
3. **The per-neuron wiring effect is rate-dependent: +0.1852 at 0.001, −0.0977 at 0.0001**, on the
   same seeds. Unregistered, descriptive, and the fact that conditions L.1: **L.1's positive may not be
   cited without its rate**, and "the first positive for the wild-type wiring under a plausible
   learner" is a claim at one pinned rate that a decade lower does not reproduce. Which rate is the
   "right" one is not a question this data answers; both arms learn better at 0.0001, so the pinned
   rate is the worse operating point for either wiring.
4. **Neither ablation reads `survives_without_it`**, so the effect does **not** live in the directed
   chemical graph's connectivity alone — the reading 7.5 would have needed for the placed-plasticity
   rung to follow directly.
5. **The probe is null over the range a rewired population offers**, which does not reach the
   wild-type contrast that motivated it. Nothing here explains why the wide null got worse at 0.001;
   the rate result supplies a candidate and no test.
6. **Not a mechanism, not an endpoint claim, not a read-across to block V, and no committed verdict
   changes.** L.1's `pooling_hid_structure` stands as read; what changes is the set of things it may be
   cited as. D2's primary remains unmet: `w_chem` is frozen throughout.

## Consequences

**The phase's positive, restated (task 7.4).** L.1's result lives, as far as these two ablations can
place it, in **the electrical synapses' interaction with the learner's operating point** — present at
0.001, removed with the gap junctions, and absent at 0.0001 — and not in the chemical graph's
connectivity on its own. The phase-after-7 rung that manipulates the feature directly is the ladder's
**rung (3), dynamics — gap-junction coupling as a dynamical term rather than a fixed symmetric
matrix**, which also absorbs B.6's deferred question of gap junctions under plasticity. It takes a
precondition that did not exist before this record: **a registered rate × wiring calibration at the
per-neuron width**, because nothing about the per-neuron effect may be cited at a rate it has not been
measured at, and the one rate it was pinned at is the worse operating point for both wirings.

**Two lessons for the [phase protocol](../../research/phase-protocol.md)**, recorded there with dates:
a pinned setting carried to a new width is a hypothesis again (principle 7 — the rate came from R.2 at
8 parameters and was never swept at 78); and the feasibility arithmetic for a correlation must compare
the predictor's spread across the units the test runs over with the contrast that motivated the
hypothesis, so the test is registered as the question it can answer (principle 2).

**The synthesis (Z.1) carries** L.4 and L.5 as done, and carries the rate dependence as a condition on
L.1 rather than a new rung within Phase 7's closing scope.

## Artefacts

- [`supporting/067-l4-feature-ablations/launch.md`](supporting/067-l4-feature-ablations/launch.md) — the protocol, both dated amendments, the rate check and its outcome, the pilots
- [`supporting/067-l4-feature-ablations/feature_ablations.json`](supporting/067-l4-feature-ablations/feature_ablations.json) — the full reading, both ablations, the ten-test family, both diagnostics for both
- [`supporting/067-l4-feature-ablations/per-seed.csv`](supporting/067-l4-feature-ablations/per-seed.csv) — one row per cell and seed across all eight scored cells, with the censoring column
- [`supporting/067-l4-feature-ablations/structural_probe.json`](supporting/067-l4-feature-ablations/structural_probe.json) — the probe, its descriptive companion and the per-seed Jaccard
- `scripts/analysis/l4_feature_ablations.py`, `scripts/analysis/l4_structural_probe.py` — the harnesses
