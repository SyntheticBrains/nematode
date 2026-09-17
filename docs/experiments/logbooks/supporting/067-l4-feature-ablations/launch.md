# L.4 + L.5 — feature ablations at the per-neuron width: the registered protocol

Registered in `openspec/changes/add-l4-feature-ablations`, reviewed and committed **before** any arm
runs. The byte-identity check and the disk measurement below ran first and spent no registered seed.

## The question

[L.1](../../066-l4-readout-width.md) read `pooling_hid_structure`: at the per-neuron readout width
the wild type leads its degree-preserving null by **+0.1852** on `auc_success`, where at the pooled
width the null led. That is the phase's first positive for the wild-type wiring under a plausible
learner, and it begs a question a positive on "the wiring" cannot answer on its own:

> *Which part of the wiring carries it — the directed chemical graph, the symmetric electrical one,
> or the sign structure the random draw imposes?*

Each ablation removes one and asks whether the effect survives.

## Each ablation is an interaction against L.1's wide baseline

```text
I_ablation = (wt_ablated − rn_ablated) − (wt_wide − rn_wide)      per seed, then the paired test
```

Negative: the feature carried some of the effect. Near zero: the effect survived. Positive: removing
the feature helped the wild type more — not predicted, reported as itself.

| ablation | key | what it removes | clean? |
|---|---|---|---|
| **L.4 atlas** | `synapse_signs: atlas` | the random sign draw on 3,176 of 3,709 chemical synapses; magnitudes, norms, RNG untouched | **No** — takes the pool's 311 grounded inputs from ~50/50 to **275 E / 36 I**, which can move the tanh operating point |
| **L.5 nogap** | `enable_gap_junctions: false` | the electrical matrix, zeroed in the forward pass; **every parameter bitwise identical** | **Yes** — forward pass and nothing else. 199 of 1,093 gap junctions touch the pool, 47 within it, all 39 pool neurons carry one |

Both wirings lose gap junctions symmetrically: the degree-preserving rewiring swaps them too.

## The arms — 768 runs, plus a reused baseline

Eight new arms × seeds **1–96**, each differing from its committed L.1 wide parent in **one key**
(asserted by exact-key test): two learning arms and two floors per ablation.

**The baseline is L.1's committed wide arms — reused on evidence, not argument.** The ablated arms run
under the new output controls (`--no-detailed-export --no-file-log`), which L.1's runs did not have.
So `wt_wide` and `rn_wide` at seed 1 were re-run under the controls and compared to L.1's logs on
every field `read_log` parses: **identical on all 9 fields, both arms** (`campaigns/export-flags-identity`).
Had any field differed, the baseline would have been re-run in full with nothing reused. Both halves
of every interaction pass through the same `connectome_structure_efficiency` call, from campaign logs.

**Disk, measured before launch.** A run under the controls writes **17.1 MB** of exports plus a
**0.46 MB** campaign log, so 768 runs need **~13.1 GB** — against ~500 GB without the controls,
which is what filled the volume during L.1.

## The minimum effect is a decision rule

The gap L.1's review recorded, closed here. **The quantity an ablation can remove is the wide wiring
effect, +0.1852** — not L.1's +0.2818 interaction, which includes the pooled cells no per-neuron
ablation touches; a feature carrying all of the effect gives an interaction of −0.185.

`carries_the_effect` requires the interaction to be significant **and** `abs(Δ) ≥ 0.123` —
**two-thirds** of 0.1852: the feature is the *majority* carrier. Sized from L.1's realised spread
(sd 0.416 at n = 96, ρ ≈ 0.08 between conditions, treated as independent): se 0.0425, detectable
0.119 at 80%, **~80% power at the minimum** (z = 2.89). Half (0.093) would sit at ~59% and need ~160
seeds plus a re-run baseline, so it is not registered.

## The registered readings, per ablation — never pooled

| reading | when |
|---|---|
| `carries_the_effect` | interaction significantly **negative** and `abs(Δ) ≥ 0.123`. On L.4, qualified *carries or saturates* if the floors diagnostic fires |
| `survives_without_it` | no significant interaction — **a failure to detect**, size and CI carried — **and** the ablated wiring effect itself significant and positive |
| `amplifies` | interaction significantly **positive**. Reported, not explained |
| `inconclusive_at_this_sensitivity` | significant but below the minimum, or neither the interaction nor the ablated wiring effect significant |
| `no_learning`, `insufficient_seeds` | as L.1; the seed floor is 5, where 2⁻ⁿ first clears 0.05 |

**Ten tests in one BH-FDR family**: per ablation the interaction, the ablated wiring effect, two
gates against the ablated floors, and the prior between them. The two interactions share the baseline
half and are positively dependent, under which BH-FDR holds; the sharing is also why they are read
separately.

**The L.4 floors diagnostic**, outside the family: each atlas frozen floor against the wide frozen
floor of the same wiring, two-sided, BH over the pair. Arms in which nothing learns, so a difference
is the operating point. If it fires at q ≤ 0.05, a `carries` on L.4 is reported as **carries or
saturates**. It qualifies a reading; it never rescues one.

## The structural probe, registered before its correlation is computed

L.1's open puzzle: the per-neuron readout made the **null worse** (0.5106 → 0.3585). Hypothesis:
degree-preserving rewiring **decorrelates the inputs within each motor class**, so 39 per-neuron
weights fit seed-specific noise the four class means averaged out.

- **Statistic**: mean pairwise Jaccard of presynaptic sets within a motor class, over the four classes.
- **Looked at so far**: feasibility only — wild type 0.07–0.23 per class, three rewirings 0.01–0.04.
- **Registered test**: across the 96 rewirings, Spearman ρ between a seed's Jaccard and its
  `rn_wide − rn_pooled` from L.1's committed `per-seed.csv`, **one-sided positive**, q = 0.05,
  minimum **ρ ≥ 0.3**. Companion: the wild type against the 96 rewirings' distribution.
- **A positive licenses a follow-up that manipulates within-class correlation directly. It is not a
  mechanism claim.**

## Amendment 2026-09-18, after the pilot and before the campaign — the gains diagnostic

The pilot (32 runs, seeds 101–104, all succeeded) passed its three stop-clause checks and exposed a
hole in the registration. The atlas learning arms gained **+2.2 and +2.8 foods** over their own
floors against the wide arms' **+13.7 and +7.8** at the same seeds — and **bimodally**: at seeds
101–102 both atlas arms finished *below* their own floors (−2.5, −1.5, −6.2, −0.7 foods) while at
103–104 they learned (+6.3 to +11.2), with both atlas arms censored on 2 of 4 seeds. The registered
floors diagnostic was **quiet** (q > 0.5): the frozen operating point did not move. What moved was
learnability on top of it, for **both** wirings.

That is a reading the rules as written get wrong. If it holds at 96 seeds the gates will likely pass,
the ablated wiring effect will sit near zero, the interaction near −0.2, and the harness will read
`carries_the_effect` — when what happened is that grounding the signs made the substrate nearly
unlearnable for wild type and shuffle alike, and the wild type simply had more to lose.

**The gains diagnostic**, registered now: each atlas arm's gain over its floor against the wide arm's
gain over its floor, paired per seed, per wiring, two-sided, BH over the pair, outside the family. If
**both** are significantly smaller, a `carries_the_effect` on L.4 is reported as **carries or
unlearnable**. It is the floors diagnostic's logic applied to gains rather than floors; it qualifies a
reading and never rescues one; and it applies to atlas only, since the nogap arms' gains (+17.1 and
+16.3) exceeded the baseline's. The nogap pattern — both wirings gaining, the null more — is **not
read** at four seeds.

## The honest prior

**L.5: `survives_without_it`.** Gap junctions are symmetric and degree-scaled — the part of the wiring
most like the degree statistics the null preserves, and 034's verdict was that degree statistics are
what the wiring's endpoint contribution amounts to. The per-neuron effect more likely lives in the
directed chemical graph.

**L.4: uncertain, leaning `carries_the_effect` — for the operating-point reason as much as any
feature reason**, which is why the diagnostic exists. B.1's finding that grounded signs made a rule
learn *worse* is not evidence either way about features.

## What no reading would license

Not a mechanism (V.2 found no graph property predicting learning time); not an endpoint claim; not a
read-across to block V's PPO result; and a `carries` on L.4 with the diagnostic fired is not
evidence that the *signs* are the feature.

## Reproduce

```bash
P=configs/scenarios/foraging/connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_hard350_eprop
FLAGS="--theme headless --track-experiment --no-detailed-export --no-file-log"

# 0. the byte-identity check that licenses reusing L.1's baseline (ran first)
uv run python scripts/run_campaign.py --config ${P}_readout_only_wide.yml \
  --config ${P}_readout_only_wide_rewired_null.yml --seeds 1 --runs 3000 \
  --output-dir campaigns/export-flags-identity -- $FLAGS

# 1. pilot on DISJOINT seeds 101-104 -- 32 runs, no reading at four pairs
uv run python scripts/run_campaign.py \
  --config ${P}_readout_only_wide_atlas.yml --config ${P}_readout_only_wide_atlas_rewired_null.yml \
  --config ${P}_frozen_wide_atlas.yml --config ${P}_frozen_wide_atlas_rewired_null.yml \
  --config ${P}_readout_only_wide_nogap.yml --config ${P}_readout_only_wide_nogap_rewired_null.yml \
  --config ${P}_frozen_wide_nogap.yml --config ${P}_frozen_wide_nogap_rewired_null.yml \
  --seeds 101-104 --runs 3000 --output-dir campaigns/feature-ablations-pilot -- $FLAGS

# 2. the registered panel -- 768 runs on seeds 1-96
uv run python scripts/run_campaign.py <the same eight configs> \
  --seeds 1-96 --runs 3000 --output-dir campaigns/feature-ablations -- $FLAGS

# 3. score both ablations against L.1's committed baseline
uv run python scripts/analysis/l4_feature_ablations.py --campaign campaigns/feature-ablations \
  --baseline campaigns/readout-width \
  --out docs/experiments/logbooks/supporting/067-l4-feature-ablations/feature_ablations.json \
  --csv docs/experiments/logbooks/supporting/067-l4-feature-ablations/per-seed.csv

# 4. the structural probe -- no runs; L.1's committed per-seed file and 97 topology builds
uv run python scripts/analysis/l4_structural_probe.py \
  --per-seed docs/experiments/logbooks/supporting/066-l4-readout-width/per-seed.csv \
  --out docs/experiments/logbooks/supporting/067-l4-feature-ablations/structural_probe.json
```
