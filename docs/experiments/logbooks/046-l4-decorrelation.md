# 046: Decorrelation — Building the Brake the Substrate Cannot Ground (7a-ii / Phase 7)

**Status**: completed — **`no_recovery`** under the pre-registered verdict map. Logbook 044 found
that grounding the substrate's synapse signs made reward-free Hebbian learning substantially worse
and issued a falsifiable prediction: a rule with an anti-Hebbian or decorrelating term should
recover the loss. This built that rule and ran the test. Neither term recovers — D1 **−0.90**,
D2 **−2.60**, both q = .487, and `full_recovery` false on every arm. The registered
decorrelation-share annotation is what makes the negative worth having, because the two arms
failed for different reasons. The anti-Hebbian arm's share is **0.057 — exactly the grounded
inhibitory fraction of the substrate**, since the atlas grounds 214 inhibitory synapses out of
3,709: it redirected every synapse it could reach and moved the outcome by −0.9 points. **A
transmitter-only atlas does not ground enough inhibition to build a brake from**, which promotes
the receptor layer from fidelity work to a prerequisite for testing this hypothesis at all. The
Oja arm supplied **2.2%** of the effective update at its pinned coefficient and reproduced the
comparator exactly on eight of sixteen seeds — a weak test of that term, at a setting the pilot
chose over a stronger one the grid also offered.

**Branch**: `feat/l4-decorrelation` (PR #333).

**Date**: 2026-09-09.

**OpenSpec change**: `add-l4-decorrelation` (archived; the two terms, the seam's post-synaptic
activity member, the registered test; extends capabilities `learning-rules`,
`connectome-ppo-brain`, `l4-plasticity-panel`).

## Objective

Test Logbook 044's prediction directly: build the decorrelating rule it said should work, and
re-run its own Hebbian protocol under that rule against its own committed values.

## Background

[Logbook 044](044-l4-atlas-signs.md) grounded 3,176 of 3,709 chemical synapses in the Wang 2024
atlas and found reward-free Hebbian learning fell from 31.5 to 14.0 on the wild-type and 17.4 to
9.1 on the rewired null. Its reading was that a purely potentiating rule on an 80%-excitatory
network has no inhibitory brake. [Logbook 045](045-l4-consolidation.md) then screened three
consolidation mechanisms and none held a cloned competent policy, with the finding that slowing
the update is not the same as consolidating a policy. That left the other open question: not
whether the rule stops, but whether it has the right update at all on a substrate whose signs are
real.

Recon sharpened 044's mechanism before anything was built. The eligibility trace is
`outer(prev_h, h)` with `h = tanh(preact)`, so it is **sign-carrying**: the rule potentiates a
synapse where pre- and post-synaptic activity agree in sign and depresses it where they disagree.
Under random signs, an inhibitory synapse driven by positive input pushed its target negative, the
trace went negative, and the update drove the synapse further negative — a self-limiting loop
running on half the substrate by construction. Grounded, the network is 80% excitatory and that
loop largely disappears; what remains is positive feedback checked only by homeostasis and the
bound. That is a mechanism, not a metaphor, and it predicts exactly what 044 measured.

## Hypothesis

Pre-registered before either campaign (`supporting/046-l4-decorrelation/launch.md` committed
first). Four one-sided paired tests corrected together under BH-FDR at α = 0.05: **D1** the
wild-type anti-Hebbian arm over the committed wild-type grounded Hebbian values (the prediction);
**D2** the wild-type Oja arm over the same; **D3** and **D4** the wild-type over the rewired null
under each variant.

Verdict in order: `insufficient_seeds`; **`no_recovery`** when neither D1 nor D2 confirms — the
outcome in which 044's prediction fails; then `recovery_specific`, `recovery_general`,
`recovery_both`. D3 and D4 annotate and never decide. Two further annotations, computed and never
verdict-changing: **`full_recovery`**, whether an arm's 80% bootstrap interval reaches the
committed random-sign mean for its wiring — the difference between a term that helps and one that
restores what grounding cost — and the **decorrelation share**, so a result whose share is near
zero is recorded as attributable to something other than the term.

## Method

**Anti-Hebbian inhibitory** negates the Hebbian term at every synapse the atlas grounds as
inhibitory, leaving grounded excitatory and ungrounded synapses alone. An inhibitory synapse doing
its job — firing while its target is suppressed, so pre and post anticorrelate — is otherwise
potentiated toward zero, unwinding the inhibition; negated, it is driven further negative, so
co-activity strengthens what the synapse *does* rather than what its weight *is*. The term's
magnitude is unchanged, so the variant redirects the update rather than resizing it. **It has no
hyperparameter**, so the variant carrying the primary test was neither piloted nor tuned. The
biological anchor is the arrangement reported in the electrosensory lobe — anti-Hebbian depression
at identified sites inside an otherwise Hebbian circuit (Perks et al., *Nature*, 2026-09-02) — not
the physiology of the mormyrid synapse.

**Oja** subtracts `η · γ · y² · w`, the classic normalisation, needing no transmitter identity. It
reads post-synaptic activity through `plastic_post_activities`, a seam member added here that both
substrates expose as a view over state their trace updates already keep. `γ` was pinned by a
declared pilot on seeds 1–2 over `{0.01, 0.1, 1.0}`: **0.1**, scoring 22.2 against 19.4 and 18.8.

**The test** is 044's Hebbian protocol with the rule keys changed and nothing else: four arms,
seeds 1–16 paired, 1000 episodes, plateau-tail full-clear success, comparators read from 044's
committed per-seed table on the same seeds and never re-run. 64 runs; **no run needed the
registered extension**.

## Results

### The registered family

| test | contrast | mean Δ | 80% CI | q | +seeds | result |
|---|---|---|---|---|---|---|
| D1 | wt anti-Hebbian − wt grounded Hebbian | −0.90 | −5.48 … +3.00 | 0.487 | 9/16 | fail |
| D2 | wt Oja − wt grounded Hebbian | −2.60 | −5.80 … +0.45 | 0.487 | 3/16 | fail |
| D3 | wt − rn under anti-Hebbian | +2.58 | −5.05 … +9.78 | 0.487 | 8/16 | fail |
| D4 | wt − rn under Oja | +2.02 | −4.53 … +8.26 | 0.487 | 7/16 | fail |

**Verdict: `no_recovery`.** `full_recovery` is false on every arm; no interval approaches its
random-sign target.

| arm | mean | 80% CI | random-sign target | reaches it | decorrelation share |
|---|---|---|---|---|---|
| wt_antihebb | 13.1 | 7.4 … 19.0 | 31.5 | no | **0.057** |
| rn_antihebb | 10.5 | 6.6 … 14.8 | 17.4 | no | 0.058 |
| wt_oja | 11.4 | 6.7 … 16.6 | 31.5 | no | 0.022 |
| rn_oja | 9.3 | 6.0 … 13.0 | 17.4 | no | 0.024 |

### Per-seed, wild-type

| seed | grounded Hebbian | anti-Hebbian | Oja |
|---|---|---|---|
| 1 | 10.0 | 15.6 | 12.4 |
| 2 | 32.8 | 42.0 | 32.0 |
| 3 | 7.6 | 5.2 | 7.6 |
| 4 | 7.6 | 7.2 | 7.6 |
| 5 | 13.2 | 11.2 | 13.2 |
| 6 | 28.4 | **41.6** | 29.6 |
| 7 | 1.6 | 2.8 | 1.6 |
| 8 | 3.2 | 7.2 | 3.2 |
| 9 | **50.4** | **0.0** | **3.6** |
| 10 | 5.6 | 4.0 | 5.6 |
| 11 | 1.2 | 2.0 | 1.2 |
| 12 | 1.6 | 3.2 | 1.6 |
| 13 | 1.2 | 0.8 | 1.2 |
| 14 | 2.0 | 1.2 | 2.0 |
| 15 | 0.4 | 2.4 | 0.4 |
| 16 | 56.8 | 62.8 | 59.2 |

## Analysis

- **The prediction fails, and the reason is a fidelity limit rather than a refutation.** The
  anti-Hebbian arm's share of 0.057 is exactly the grounded inhibitory fraction: the atlas
  identifies 214 inhibitory synapses of 3,709, so the term acted on every synapse it was able to
  and changed the outcome by −0.9 points. The hypothesis was that the missing inhibitory brake
  explains 044's collapse; what this shows is that **the brake cannot be built at
  transmitter-only fidelity**. The animal's inhibition is not only presynaptic identity —
  cholinergic and glutamatergic synapses can be inhibitory through their post-synaptic receptor —
  and that is precisely what the receptor layer would supply. The result therefore promotes that
  layer from fidelity work to a prerequisite for asking this question properly.
- **The Oja arm is a weak test, at a setting chosen over a stronger one.** Its term supplied 2.2%
  of the effective update at `γ = 0.1`, and the arm reproduces the committed comparator *exactly*
  on eight of sixteen seeds. Share scales with the coefficient, so the grid's top value of `1.0`
  would have contributed about 18% and parity with the Hebbian term comes near `γ ≈ 4.5`. The grid
  did contain a substantial setting; the pilot chose against it on two seeds. D2 is a real result
  about the term as pinned and says less about the term as it could have been run.
- **One seed decides D1's sign, and it is the comparator's best.** Seed 9 falls from 50.4 to 0.0
  under the variant. Excluding it, the anti-Hebbian arm is **+2.4** over the remaining fifteen
  seeds and up on nine, three substantially (seed 6 +13.2, seed 2 +9.2, seed 1 +5.6). That
  exclusion is post-hoc and licenses nothing; it is recorded because the same bimodal shape has now
  decided four panels and because Logbook 042's standing lesson is to register a statistic matched
  to it in advance.
- **The wiring contrast remains unconfirmed**, at +2.6 and +2.0 with both intervals spanning zero,
  as in every panel since the first.

## Conclusions

- **A negative with a measured cause beats a negative without one.** The share annotation
  converted "two terms failed" into "one term did everything the substrate allows, and the other
  supplied 2.2% of the update", and only the first is evidence about the hypothesis.
- **Sign identity alone is not enough inhibition.** 5.8% of synapses is what a transmitter atlas
  can mark inhibitory; the network needs the receptor layer before an inhibitory-brake rule can be
  said to have been tested.
- **The variant with no hyperparameter carried the primary test**, which is why its failure is
  informative and the tuned arm's is not. That property is worth preserving in future rule work.
- Three rule-level interventions have now been tried on this substrate — consolidation (three
  mechanisms), sign grounding, and decorrelation (two terms) — and none has moved the wild-type
  connectome off its floors.

## Limitations

- The anti-Hebbian variant is testable only as far as the atlas grounds inhibition; its negative
  is conditional on that fidelity and is not a statement about anti-Hebbian plasticity in general.
- The Oja coefficient was pinned on two seeds, and the stronger setting the grid offered was never
  run at n = 16.
- n = 16, with a bimodal outcome one seed can dominate — the fourth panel in a row where that is
  true, and still without a registered statistic matched to it.
- The share annotation was corrected mid-review: its first computation compared raw traces against
  a rate-scaled term, quantities in different units. All 64 runs were repeated under the corrected
  telemetry and every per-seed value was identical, confirming the annotation never entered the
  update, but the first published Oja figure (0.0026) was wrong and the claims resting on it were
  retracted.
- Both variants were tested from random initialisation, which is what 044's prediction was about.
  Neither has been through the clone assay, so neither is licensed for the panel.

## Next Steps

**B.4b structured, pathway-specific instruction** — the last main rule item in 7a-ii — with the
**receptor layer (B.3) now a prerequisite** for any further test of the inhibitory-brake
hypothesis rather than optional fidelity work. Carried forward: a fast quality-gated brake remains
untested (Logbook 045), the rigidity family remains the nearest miss on the clone assay, and a
registered statistic matched to a bimodal outcome remains unchosen.

## Data References

- Registration and design: `openspec/changes/archive/2026-09-09-add-l4-decorrelation/`;
  capabilities `openspec/specs/learning-rules/spec.md`,
  `openspec/specs/connectome-ppo-brain/spec.md`, `openspec/specs/l4-plasticity-panel/spec.md`.
- Everything the test produced:
  [supporting/046-l4-decorrelation/](supporting/046-l4-decorrelation/details.md) — `launch.md`
  (protocol, family, verdict map, pilot grid and pins, all before the runs), `panel.json`,
  `per-seed.csv`, `curves.csv`, `_manifest.txt`, `details.md`.
- Tooling: `learning_rules/three_factor.py`, `brain/arch/_topology.py` (the seam),
  `brain/arch/_mlp_topology.py`, `brain/arch/connectome_ppo.py`,
  `scripts/analysis/l4_decorrelation.py`; arm configs under
  `configs/scenarios/foraging_predator_thermal/`.
